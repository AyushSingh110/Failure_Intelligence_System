"""
I-5, I-9, I-13 / attacks 4a, 4b, 6a–6c: clusters, trend, and the attack-pattern index.

Tenant-derived state is per tenant. Platform state does not change from a request.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from .conftest import scope_for, squash
from .fakes import MARK, TENANT_A, TENANT_B, monitor_body


def _monitor(actor, key: str, n: int = 0, **extra) -> dict:
    body = monitor_body(f"{MARK[key]['prompt']} number {n}", f"{MARK[key]['answer']} number {n}", **extra)
    response = actor.post("/api/v1/monitor", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ── Clusters and trend ────────────────────────────────────────────────────────

def test_clusters_show_only_own_tenant(a, b):
    """N3: a cluster centroid carries the normalised model answer."""
    _monitor(a, "a")
    mine = a.get("/api/v1/clusters").json()
    assert mine["total_clusters"] >= 1
    assert squash(MARK["a"]["answer"]) in squash(json.dumps(mine))
    assert b.get("/api/v1/clusters").json() == {"total_clusters": 0, "clusters": []}


def test_trend_is_tenant_scoped(a, b):
    for n in range(3):
        _monitor(a, "a", n)
    assert a.get("/api/v1/trend").json()["signals_recorded"] == 3
    assert b.get("/api/v1/trend").json()["signals_recorded"] == 0


def test_cluster_reset_is_tenant_scoped(a, b):
    _monitor(a, "a")
    _monitor(b, "b")
    assert b.delete("/api/v1/clusters/reset").status_code == 200
    assert b.get("/api/v1/clusters").json()["total_clusters"] == 0
    assert a.get("/api/v1/clusters").json()["total_clusters"] >= 1


@pytest.mark.parametrize("path,body", [
    ("/api/v1/analyze/v2", {"model_outputs": ["one answer", "one answer"]}),
    ("/api/v1/diagnose", {"prompt": "hello", "model_outputs": ["one answer"]}),
])
def test_other_writers_touch_only_the_callers_registry(a, b, monkeypatch, path, body):
    import app.routes.inference as inference
    from app.schemas import FailureSignalVector

    signal = FailureSignalVector(agreement_score=1.0, fsd_score=0.0, answer_counts={"one answer": 1},
                                 entropy_score=0.0, ensemble_disagreement=False, ensemble_similarity=1.0)
    monkeypatch.setattr(inference.failure_agent, "_build_signal", lambda outputs: signal)
    monkeypatch.setattr(inference.failure_agent._jury, "deliberate", lambda context: __import__(
        "app.schemas", fromlist=["JuryVerdict"]).JuryVerdict(verdicts=[]))
    assert a.post(path, json=body).status_code == 200
    assert a.get("/api/v1/trend").json()["signals_recorded"] == 1
    assert b.get("/api/v1/trend").json()["signals_recorded"] == 0
    assert b.get("/api/v1/clusters").json()["total_clusters"] == 0


def test_spike_alert_reflects_only_the_callers_traffic(a, b, monkeypatch):
    """N16: the alert sent to B must not blend in A's signals."""
    import app.notifications as notifications
    import app.routes.monitor as routes
    from app.schemas import FailureSignalVector

    alerts = []
    monkeypatch.setattr(notifications, "notify_degradation_spike", lambda **kwargs: alerts.append(kwargs))
    entropy = {"value": 1.0}
    monkeypatch.setattr(routes, "build_failure_signal", lambda outputs: FailureSignalVector(
        agreement_score=0.0, fsd_score=0.0, answer_counts={}, entropy_score=entropy["value"],
        ensemble_disagreement=True, ensemble_similarity=0.0, high_failure_risk=True,
    ))
    _monitor(a, "a")
    entropy["value"] = 0.0
    _monitor(b, "b")
    b_alert = alerts[-1]
    assert b_alert["ema_entropy"] == 0.0, "B's alert carries a blend that includes A's signals"


def test_per_tenant_holders_are_bounded():
    from app.auth_guard import Principal
    from app.tenancy import TenantRegistry, TenantScope

    registry = TenantRegistry(max_tenants=3)
    scopes = [TenantScope(Principal(tenant_id=f"t-{i}", subject="s", role="tenant", credential_kind="api_key"))
              for i in range(5)]
    for scope in scopes:
        registry.for_scope(scope)
    assert len(registry) == 3
    first = registry.for_scope(scopes[4])
    assert registry.for_scope(scopes[4]) is first
    assert registry.for_scope(scopes[0]) is not None and len(registry) == 3


# ── Attack-pattern index ──────────────────────────────────────────────────────

@pytest.fixture
def hostile_jury(monkeypatch):
    import app.routes.monitor as routes
    import engine.archetypes.registry as registry
    from app.schemas import AgentVerdict, JuryVerdict

    verdict = AgentVerdict(agent_name="AdversarialSpecialist", root_cause="PROMPT_INJECTION",
                           confidence_score=0.95, mitigation_strategy="x", evidence={"category": "INJECTION"})
    jury = JuryVerdict(verdicts=[verdict], primary_verdict=verdict, jury_confidence=0.95, is_adversarial=True,
                       failure_summary="d")
    monkeypatch.setattr(routes.failure_agent, "run_diagnostic", lambda request, **kwargs: SimpleNamespace(jury=jury))
    grown = []
    monkeypatch.setattr(registry.adversarial_registry, "add_confirmed_detection",
                        lambda **kwargs: grown.append(kwargs) or True)
    return grown


def test_monitor_does_not_grow_attack_index(a, hostile_jury):
    """Attack 4a / S7."""
    _monitor(a, "a", run_full_jury=True)
    assert hostile_jury == [], "a tenant prompt was promoted into the shared attack-pattern index"


def test_index_growth_is_off_unless_the_platform_switch_is_set(monkeypatch):
    from engine.archetypes.registry import AdversarialRegistry

    registry = AdversarialRegistry()
    monkeypatch.setattr(registry, "_initialize_once", lambda: (_ for _ in ()).throw(AssertionError("touched")))
    assert registry.add_confirmed_detection("p", "L", "C", confidence=0.99) is False


def test_faiss_evidence_never_returns_learned_prompt(monkeypatch):
    """N4: the nearest stored prompt is returned only when it comes from the seed corpus."""
    import engine.agents.adversarial.specialist as specialist
    from app.schemas import FailureSignalVector
    from engine.agents.base_agent import DiagnosticContext
    from engine.archetypes.registry import AdversarialRecord, FAISSSearchResult

    monkeypatch.setattr(specialist, "run_llm_intent_check", lambda prompt: (None, 0.0, {}))
    fsv = FailureSignalVector(agreement_score=1.0, fsd_score=0.0, answer_counts={}, entropy_score=0.0,
                              ensemble_disagreement=False, ensemble_similarity=1.0)
    context = DiagnosticContext.build(prompt="please summarise the weather report", primary_output="ok",
                                      secondary_output="ok", model_outputs=["ok"], fsv=fsv, latency_ms=None)

    def evidence_for(source: str) -> str:
        record = AdversarialRecord(prompt=f"{MARK['a']['prompt']} stored by another tenant",
                                   label="PROMPT_INJECTION", category="INJECTION", source=source)
        hit = (FAISSSearchResult(record=record, similarity=0.97, is_match=True), 0.9)
        for name in ("run_faiss_detection", "_run_faiss_detection"):
            if hasattr(specialist, name):
                monkeypatch.setattr(specialist, name, lambda prompt: hit)
        verdict = specialist.adversarial_specialist.analyze(context)
        assert "faiss" in (verdict.evidence or {}).get("detection_layers_fired", []), verdict
        return json.dumps(verdict.evidence)

    assert MARK["a"]["prompt"] in evidence_for("seed")
    for learned in ("confirmed", "user", ""):
        evidence = evidence_for(learned)
        assert MARK["a"]["prompt"] not in evidence
        assert "PROMPT_INJECTION" in evidence


def test_holder_requires_a_scope():
    from app.tenancy import TenantRegistry, TenantScopeError

    with pytest.raises(TenantScopeError):
        TenantRegistry().for_scope(TENANT_A["tenant_id"])
    assert TenantRegistry().for_scope(scope_for(TENANT_A)) is not None
    assert TENANT_B["tenant_id"] != TENANT_A["tenant_id"]
