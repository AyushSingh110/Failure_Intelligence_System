"""
Drives the real /monitor handler and the real ground-truth pipeline with every
I/O leaf replaced: shadow fan-out, the 8B judge, Serper search, Wikidata search,
Wikipedia retrieval, the answer cache, notifications and persistence.

Nothing above the leaves is faked, so a fixture exercises the same parsers,
pipeline branches, fix engine and route logic as production. Used by the pytest
suite and by research/reliability/e_r1_part_a/run_part_a.py, which runs the
same fixtures under evals/hermetic.py; hence a small Patcher instead of pytest's
monkeypatch.
"""
from __future__ import annotations

import importlib
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional


class Patcher:
    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any]] = []

    def setattr(self, target: Any, name: str, value: Any) -> None:
        obj = importlib.import_module(target) if isinstance(target, str) else target
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self) -> None:
        while self._undo:
            obj, name, old = self._undo.pop()
            setattr(obj, name, old)


# ── leaf fakes ───────────────────────────────────────────────────────────────

@dataclass
class Shadow:
    text:       str
    success:    bool = True
    confidence: str  = "MEDIUM"   # HIGH=3, MEDIUM=2, LOW=1 (engine/groq_service.py)


_WEIGHT = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}


class FakeGroq:
    """
    Shadow fan-out returns `shadows`. complete() is the 8B judge: `judge(prompt)`
    returns the reply text, or None for a failed call (success=False).
    """

    def __init__(self, shadows: list[Shadow], judge: Optional[Callable[[str], Optional[str]]] = None):
        self.shadows = shadows
        self.judge = judge
        self.judge_prompts: list[str] = []

    def _fan_out(self):
        from engine.groq_service import GroqModelResponse
        return [
            GroqModelResponse(
                model_name=f"shadow-{i}", output_text=s.text if s.success else "",
                success=s.success, latency_ms=1.0,
                error="" if s.success else "simulated shadow outage",
                model_confidence=s.confidence, confidence_weight=_WEIGHT[s.confidence],
            )
            for i, s in enumerate(self.shadows)
        ]

    def fan_out(self, prompt, system_message=None, cache_scope=None):
        return self._fan_out()

    def fan_out_with_confidence(self, prompt, system_message=None, cache_scope=None):
        return self._fan_out()

    def complete(self, prompt, model_name=None, max_tokens=None, temperature=None, **_):
        from engine.groq_service import GroqModelResponse
        self.judge_prompts.append(prompt)
        text = self.judge(prompt) if self.judge else None
        return GroqModelResponse(
            model_name="judge", output_text=text or "", success=text is not None,
            latency_ms=1.0, error="" if text is not None else "simulated judge outage",
        )


class FakeCacheCollection:
    """One stored ground_truth_cache document, returned for any query."""

    def __init__(self, doc: dict):
        self.doc = doc

    def find_one(self, *_a, **_k):
        return dict(self.doc)

    def find(self, *_a, **_k):
        return [dict(self.doc)]

    def update_one(self, *_a, **_k):
        return None

    def create_index(self, *_a, **_k):
        return None


def cache_doc(answer: str, source: str, source_class: str, age_days: int) -> dict:
    at = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    return {
        "_id": "fixture", "tenant_id": "reliability-fixture", "schema": 2,
        "source_class": source_class, "question_text": "fixture", "question_vector": None,
        "verified_answer": answer, "source": source, "confidence": 0.95,
        "verified_by": "system", "verified_at": at, "use_count": 3,
    }


# ── scenario ─────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    prompt:          str
    primary:         str
    shadows:         list[Shadow] = field(default_factory=list)
    root_cause:      str = "FACTUAL_HALLUCINATION"
    jury_confidence: float = 0.60
    high_risk:       Optional[bool] = True     # None = real build_failure_signal
    judge:           Optional[Callable[[str], Optional[str]]] = None
    claim:           Optional[tuple[str, str, str]] = None      # (subject, property, value)
    wikidata_entity: Optional[dict] = None     # {"id", "label", "description"}
    serper_payload:  Optional[dict] = None     # raw Serper JSON; None = Serper off
    wikipedia:       Optional[tuple[str, str]] = None           # (summary, judge answer); None = no article
    cache:           Optional[dict] = None     # a cache_doc(); None = miss
    jury:            bool = True               # False = the jury returns no verdict


@dataclass
class Outcome:
    response:          Any
    classifier_inputs: dict
    signal_log:        dict
    notifications:     list[str]
    judge_prompts:     list[str]


def _jury(root_cause: str, confidence: float):
    from app.schemas import AgentVerdict, JuryVerdict
    verdict = AgentVerdict(
        agent_name="DomainCritic", root_cause=root_cause, confidence_score=confidence,
        mitigation_strategy="fixture", evidence={"layers_fired": ["fixture"]},
    )
    return JuryVerdict(verdicts=[verdict], primary_verdict=verdict,
                       jury_confidence=confidence, failure_summary=f"Diagnosis: {root_cause}")


def load_route_from_source(source: str, name: str) -> types.ModuleType:
    """Execute a historical app/routes/monitor.py next to the current engine (B0)."""
    import sys
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    sys.modules[name] = module   # dataclasses resolve string annotations through sys.modules
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


def run_monitor(sc: Scenario, route_module: Optional[types.ModuleType] = None) -> Outcome:
    import app.routes.monitor as current_route
    from app.auth_guard import Principal
    from app.schemas import FailureSignalVector, MonitorRequest
    from starlette.requests import Request as StarletteRequest

    route = route_module or current_route
    p = Patcher()
    groq = FakeGroq(sc.shadows, sc.judge)
    captured: dict = {"clf": {}, "log": {}, "notify": []}
    try:
        import app.limiter as limiter_mod
        if limiter_mod.limiter is not None:
            p.setattr(limiter_mod.limiter, "enabled", False)
        p.setattr(route.settings, "groq_enabled", True)
        p.setattr(route.settings, "groq_api_key", "fixture-key")
        p.setattr("engine.groq_service", "get_groq_service", lambda: groq)

        if sc.high_risk is not None:
            def _signal(outputs, _risk=sc.high_risk):
                return FailureSignalVector(
                    agreement_score=0.5, fsd_score=0.5, answer_counts={o: 1 for o in outputs},
                    entropy_score=0.5, ensemble_disagreement=_risk, ensemble_similarity=0.5,
                    high_failure_risk=_risk,
                )
            p.setattr(route, "build_failure_signal", _signal)

        p.setattr("engine.detector.embedding", "compute_embedding_distance",
                  lambda a, b: {"embedding_distance": 0.0})
        p.setattr(route.failure_agent, "run_diagnostic",
                  lambda request, **kw: types.SimpleNamespace(
                      jury=_jury(sc.root_cause, sc.jury_confidence) if sc.jury else None))

        # persistence and outbound side effects
        p.setattr("storage.database", "save_inference", lambda inference: True)
        p.setattr("app.auth", "increment_usage", lambda tenant_id: True)
        p.setattr("storage.signal_logger", "log_signal",
                  lambda **f: captured["log"].update(f) or "fixture-log")
        p.setattr("storage.signal_logger", "attach_request_id", lambda *a, **k: True)
        for name in ("notify_human_review", "notify_attack_detected", "notify_degradation_spike"):
            p.setattr("app.notifications", name,
                      lambda *a, _n=name, **k: captured["notify"].append(_n))

        # classifier: real model, inputs recorded
        import engine.failure_classifier as clf
        real_predict = clf.predict
        p.setattr(clf, "predict", lambda **kw: captured["clf"].update(kw) or real_predict(**kw))

        # verifier leaves
        p.setattr("engine.claim_extractor", "extract_claim",
                  lambda text, question="": _claim(sc.claim, text))
        p.setattr("engine.verifier.wikidata_verifier", "_fetch_search_results",
                  lambda query, limit, timeout, headers: [dict(sc.wikidata_entity)] if sc.wikidata_entity else [])
        settings = route.settings
        p.setattr(settings, "serper_enabled", sc.serper_payload is not None)
        p.setattr(settings, "serper_api_key", "fixture-key" if sc.serper_payload is not None else "")
        p.setattr("engine.verifier.serper_verifier", "_search_google",
                  lambda query, api_key, timeout: sc.serper_payload)
        p.setattr("engine.ground_truth_cache", "_get_collection",
                  lambda: FakeCacheCollection(sc.cache) if sc.cache else None)
        p.setattr("engine.rag.retriever", "fetch_wikipedia_summary",
                  lambda prompt, *a, **k: sc.wikipedia[0] if sc.wikipedia else "")
        p.setattr("engine.rag.rag_pipeline", "get_grounded_answer",
                  lambda prompt, summary, *a, **k: sc.wikipedia[1] if sc.wikipedia else "")

        principal = Principal(tenant_id="reliability-fixture", subject="fixture@example.test",
                              role="tenant", credential_kind="api_key")
        request = StarletteRequest({"type": "http", "method": "POST", "path": "/api/v1/monitor",
                                    "headers": [], "client": ("127.0.0.1", 1234)})
        response = route.monitor(
            request,
            MonitorRequest(prompt=sc.prompt, primary_output=sc.primary,
                           primary_model_name="fixture-model", run_full_jury=True),
            principal=principal,
        )
    finally:
        p.undo()
    return Outcome(response, captured["clf"], captured["log"], captured["notify"], groq.judge_prompts)


def _claim(spec, text):
    if spec is None:
        return None
    from engine.claim_extractor import ExtractedClaim
    subject, prop, value = spec
    return ExtractedClaim(subject=subject, property=prop, value=value, raw_text=text)


def delivered_by_sdk(response_dict: dict, primary_output: str, on_proposal=None,
                     sdk_module: Optional[types.ModuleType] = None) -> str:
    """
    What fie/monitor.py's `correct` mode returns for this server response: the
    real correct_wrapper is run with the FIE client replaced by one that returns
    `response_dict`.
    """
    # `fie.monitor` the attribute is the decorator; the module is in sys.modules.
    sdk = sdk_module or importlib.import_module("fie.monitor")

    class _Client:
        def __init__(self, config):
            pass

        def monitor(self, *a, **k):
            return response_dict

        def _send_telemetry(self, *a, **k):
            return None

    p = Patcher()
    try:
        p.setattr(sdk, "FIEClient", _Client)
        p.setattr("fie.output_scanner", "scan_output_async", lambda **k: None)
        p.setattr(sdk, "preflight_check",
                  lambda prompt, domain=None: types.SimpleNamespace(blocked=False, attack_type=None))

        extra = {"on_proposal": on_proposal} if on_proposal is not None else {}   # legacy SDK lacks it

        @sdk.monitor(fie_url="http://127.0.0.1:9", api_key="fixture", mode="correct", log_results=False,
                     **extra)
        def model(prompt: str) -> str:
            return primary_output

        return model("fixture prompt")
    finally:
        p.undo()
