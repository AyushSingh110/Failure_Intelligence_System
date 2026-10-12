"""
I-4, I-8, I-13 / attacks 2a, 2b, 2c, 7b, 8b: the answer cache and the shadow-response cache.

The end-to-end tests drive the real `/monitor` → answer pipeline → cache path. The
jury and the question classifier are replaced so the pipeline reaches the cache
lookup and then stops without any external lookup.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from .conftest import fake_embedding, inference_ids, scope_for
from .fakes import MARK, TENANT_A, TENANT_B, monitor_body

QUESTION = "Who currently chairs the Zorblat harbour committee?"
REWORDED = "currently who chairs the Zorblat harbour committee"   # same words, different id
WRONG = "It is chaired by nobody at all."


@pytest.fixture
def cache_path(monkeypatch):
    import app.routes.monitor as routes
    from app.schemas import AgentVerdict, FailureSignalVector, JuryVerdict

    monkeypatch.setattr(routes, "build_failure_signal", lambda outputs: FailureSignalVector(
        agreement_score=1.0, fsd_score=0.0, answer_counts={}, entropy_score=0.0,
        ensemble_disagreement=False, ensemble_similarity=1.0, high_failure_risk=True,
    ))
    verdict = AgentVerdict(agent_name="DomainCritic", root_cause="TEMPORAL_KNOWLEDGE_CUTOFF",
                           confidence_score=0.9, mitigation_strategy="verify", evidence={})
    jury = JuryVerdict(verdicts=[verdict], primary_verdict=verdict, jury_confidence=0.9,
                       failure_summary="diagnosis")
    monkeypatch.setattr(routes.failure_agent, "run_diagnostic",
                        lambda request, **kwargs: SimpleNamespace(jury=jury))
    monkeypatch.setattr("engine.question_classifier.classify", lambda prompt: "FACTUAL")
    monkeypatch.setattr("engine.question_classifier.pipeline_gates",
                        lambda question_type: {"run_wikidata": True, "run_serper": False})


def _ask(actor, question: str = QUESTION) -> dict:
    response = actor.post("/api/v1/monitor", json=monitor_body(question, WRONG, run_full_jury=True))
    assert response.status_code == 200, response.text
    return response.json()


def _correct(actor, key: str, question: str = QUESTION) -> None:
    _ask(actor, question)
    request_id = inference_ids(actor)[0]
    response = actor.post(f"/api/v1/feedback/{request_id}",
                          json={"is_correct": False, "correct_answer": MARK[key]["fix"]})
    assert response.status_code == 200 and response.json()["cache_updated"] is True, response.text


# ── End to end ────────────────────────────────────────────────────────────────

def test_a_tenant_gets_its_own_correction_back(a, cache_path):
    _correct(a, "a")
    body = _ask(a)
    assert body["ground_truth"]["from_cache"] is True
    assert body["ground_truth"]["verified_answer"] == MARK["a"]["fix"]
    assert body["fix_result"]["fixed_output"] == MARK["a"]["fix"]


def test_cache_a_writes_b_misses(a, b, cache_path):
    """Attack 2a, exact question."""
    _correct(a, "a")
    body = _ask(b)
    assert body["ground_truth"]["from_cache"] is False
    assert MARK["a"]["fix"] not in str(body)


def test_cache_a_writes_b_misses_on_a_similar_question(a, b, cache_path):
    """Attack 2a, semantic match."""
    _correct(a, "a")
    assert _ask(a, REWORDED)["ground_truth"]["from_cache"] is True
    assert _ask(b, REWORDED)["ground_truth"]["from_cache"] is False


def test_cache_b_writes_a_misses(a, b, cache_path):
    _correct(b, "b")
    assert _ask(a)["ground_truth"]["from_cache"] is False


def test_same_question_two_tenants_two_answers(a, b, cache_path, fakedb):
    _correct(a, "a")
    _correct(b, "b")
    assert _ask(a)["ground_truth"]["verified_answer"] == MARK["a"]["fix"]
    assert _ask(b)["ground_truth"]["verified_answer"] == MARK["b"]["fix"]
    assert len(fakedb["ground_truth_cache"].docs) == 2


def test_cache_trace_has_no_identity(a, cache_path, fakedb):
    """N2: not even the caller's own e-mail belongs in a trace or in the stored entry."""
    _correct(a, "a")
    body = _ask(a)
    assert TENANT_A["email"] not in str(body)
    assert "verified by" not in " ".join(body["ground_truth"]["pipeline_trace"]).lower()
    assert all(TENANT_A["email"] not in str(doc) for doc in fakedb["ground_truth_cache"].docs)


def test_cache_hit_flag_only_for_own_entries(a, b, cache_path):
    """Attack 8b: B cannot tell that A asked or corrected this question."""
    before = _ask(b)["ground_truth"]
    _correct(a, "a")
    after = _ask(b)["ground_truth"]
    assert (before["from_cache"], before["source"]) == (after["from_cache"], after["source"])
    assert before["pipeline_trace"] == after["pipeline_trace"]


def test_legacy_entry_without_a_tenant_is_never_served(a, cache_path, fakedb):
    fakedb["ground_truth_cache"].insert_one({
        "_id": hashlib.sha256(QUESTION.strip().lower().encode()).hexdigest()[:32],
        "question_text": QUESTION, "question_vector": fake_embedding(QUESTION),
        "verified_answer": "ZQ-LEGACY-ANSWER", "source": "user_feedback", "confidence": 1.0,
        "verified_by": "old.user@legacy.test", "verified_at": "", "use_count": 3,
    })
    for question in (QUESTION, REWORDED):
        body = _ask(a, question)
        assert body["ground_truth"]["from_cache"] is False
        assert "ZQ-LEGACY-ANSWER" not in str(body) and "old.user@legacy.test" not in str(body)


def test_cache_failure_does_not_fail_the_request(a, cache_path, fakedb):
    fakedb["ground_truth_cache"].fail_with = RuntimeError("ZQ-INTERNAL-DETAIL")
    response = a.post("/api/v1/monitor", json=monitor_body(QUESTION, WRONG, run_full_jury=True))
    assert response.status_code == 200
    assert "ZQ-INTERNAL-DETAIL" not in response.text
    assert response.json()["ground_truth"]["from_cache"] is False


# ── The cache module itself ───────────────────────────────────────────────────

def test_cache_key_contains_tenant():
    from engine.ground_truth_cache import _question_id

    key_a = _question_id(QUESTION, TENANT_A["tenant_id"])
    assert key_a != _question_id(QUESTION, TENANT_B["tenant_id"])
    assert key_a == _question_id("  " + QUESTION.upper() + " ", TENANT_A["tenant_id"])
    assert TENANT_A["tenant_id"] not in key_a


def test_lookup_and_save_need_a_scope(fakedb, events):
    from engine.ground_truth_cache import lookup_cache, save_to_cache

    assert save_to_cache(QUESTION, "answer") is False
    assert fakedb["ground_truth_cache"].docs == []
    save_to_cache(QUESTION, "answer", scope=scope_for(TENANT_A), source_class="tenant_feedback")
    assert lookup_cache(QUESTION) is None
    assert lookup_cache(QUESTION, scope=TENANT_A["tenant_id"]) is None      # a string is not a scope
    assert lookup_cache(QUESTION, scope=scope_for(TENANT_A)).verified_answer == "answer"
    assert len(events.named("cache.scope_missing")) == 3


def test_cache_writethrough_is_tenant_scoped(fakedb, events):
    """N19: answers the pipeline verified itself stay with the tenant whose request produced them."""
    from engine.ground_truth_cache import lookup_cache
    from engine.verifier.ground_truth_pipeline import _cache_if_confident

    _cache_if_confident(QUESTION, "system answer", "wikidata", 0.95, scope=scope_for(TENANT_A))
    (doc,) = fakedb["ground_truth_cache"].docs
    assert doc["tenant_id"] == TENANT_A["tenant_id"] and doc["source_class"] == "system"
    assert lookup_cache(QUESTION, scope=scope_for(TENANT_B)) is None
    _cache_if_confident(QUESTION, "unscoped", "wikidata", 0.95)
    assert len(fakedb["ground_truth_cache"].docs) == 1
    assert events.named("cache.scope_missing")


def test_system_write_does_not_replace_a_tenant_correction(fakedb):
    from engine.ground_truth_cache import lookup_cache, save_to_cache

    scope = scope_for(TENANT_A)
    save_to_cache(QUESTION, "tenant says", scope=scope, source_class="tenant_feedback")
    assert save_to_cache(QUESTION, "system says", source="wikidata", scope=scope, source_class="system") is False
    assert lookup_cache(QUESTION, scope=scope).verified_answer == "tenant says"
    save_to_cache(QUESTION, "tenant says again", scope=scope, source_class="tenant_feedback")
    assert lookup_cache(QUESTION, scope=scope).verified_answer == "tenant says again"


def test_every_cache_query_filters_on_the_tenant(fakedb, monkeypatch):
    from engine.ground_truth_cache import lookup_cache, save_to_cache

    seen = []
    col = fakedb["ground_truth_cache"]
    for name in ("find_one", "find", "update_one"):
        original = getattr(col, name)
        monkeypatch.setattr(col, name, lambda flt=None, *a, _o=original, **k: (seen.append(flt), _o(flt, *a, **k))[1])
    save_to_cache(QUESTION, "x", scope=scope_for(TENANT_A), source_class="tenant_feedback")
    lookup_cache(QUESTION, scope=scope_for(TENANT_A))
    lookup_cache(REWORDED, scope=scope_for(TENANT_A))
    assert seen and all((f or {}).get("tenant_id") == TENANT_A["tenant_id"] for f in seen), seen


def test_cache_requests_a_tenant_index(fakedb):
    from engine.ground_truth_cache import save_to_cache

    save_to_cache(QUESTION, "x", scope=scope_for(TENANT_A), source_class="tenant_feedback")
    assert ("tenant_id",) in fakedb["ground_truth_cache"].index_keys()


# ── Shadow-response cache (N9) ────────────────────────────────────────────────

SHADOW_MODEL = "suite-shadow-model"


class _CountingSession:
    """Replaces the HTTP session of a real GroqService. Each call returns a different answer."""

    def __init__(self):
        self.calls: list[dict] = []
        self.headers: dict = {}

    @property
    def shadow_calls(self) -> list[dict]:
        """Calls of the shadow fan-out only. Other helpers (the explanation writer) also use the service."""
        return [c for c in self.calls if c["model"] == SHADOW_MODEL]

    def post(self, url, json=None, timeout=None):
        self.calls.append(json)
        n = len(self.calls)

        class _Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": f"shadow answer number {n}\nCONFIDENCE: HIGH"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        return _Response()


@pytest.fixture
def shadow(monkeypatch):
    import app.routes.monitor as routes
    import engine.groq_service as groq_service
    from app.schemas import FailureSignalVector

    service = groq_service.GroqService(api_key="suite-key", models=[SHADOW_MODEL])
    service._session = _CountingSession()
    monkeypatch.setattr(groq_service, "_groq_service_instance", service)
    monkeypatch.setattr(routes.settings, "groq_enabled", True)
    monkeypatch.setattr(routes.settings, "groq_api_key", "suite-key")
    monkeypatch.setattr(routes, "build_failure_signal", lambda outputs: FailureSignalVector(
        agreement_score=1.0, fsd_score=0.0, answer_counts={}, entropy_score=0.0,
        ensemble_disagreement=False, ensemble_similarity=1.0, high_failure_risk=False,
    ))
    return service


def test_groq_cache_is_scoped(a, b, shadow):
    """Attack 7a: B's request must not be answered from A's cached shadow call."""
    prompt = "What colour is the Zorblat flag?"
    first = a.post("/api/v1/monitor", json=monitor_body(prompt, "blue")).json()
    second = b.post("/api/v1/monitor", json=monitor_body(prompt, "blue")).json()
    assert len(shadow._session.shadow_calls) == 2, "B was served from A's cache entry"
    assert first["shadow_model_results"][0]["output_text"] != second["shadow_model_results"][0]["output_text"]


def test_groq_cache_never_reuses_an_answer_made_under_another_canary(a, shadow):
    prompt = "What colour is the Zorblat flag?"
    a.post("/api/v1/monitor", json=monitor_body(prompt, "blue"))
    a.post("/api/v1/monitor", json=monitor_body(prompt, "blue"))
    assert len(shadow._session.shadow_calls) == 2
    canaries = [c["messages"][0]["content"] for c in shadow._session.shadow_calls]
    assert canaries[0] != canaries[1]


def test_groq_cache_hits_inside_one_scope_and_is_off_without_one(shadow):
    scope_a, scope_b = scope_for(TENANT_A), scope_for(TENANT_B)
    shadow.complete("p", cache_scope=scope_a)
    shadow.complete("p", cache_scope=scope_a)
    assert len(shadow._session.calls) == 1
    shadow.complete("p", cache_scope=scope_b)
    assert len(shadow._session.calls) == 2
    shadow.complete("p")
    shadow.complete("p")
    shadow.complete("p", cache_scope=TENANT_A["tenant_id"])   # a string is not a scope
    assert len(shadow._session.calls) == 5


def test_groq_cache_disabled_without_scope(shadow):
    import engine.groq_service as groq_service

    shadow.fan_out_with_confidence("q")
    shadow.fan_out_with_confidence("q")
    assert len(shadow._session.calls) == 2
    assert groq_service._response_cache == {}
