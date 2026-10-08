"""I-7 / attacks 5a, 5b: session context and conversation turns are keyed by (tenant, id)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from .fakes import MARK, TENANT_A, TENANT_B, monitor_body


@pytest.fixture
def adversarial_jury(monkeypatch):
    """The jury calls a prompt adversarial when it contains the word ATTACK."""
    import app.routes.monitor as routes
    import engine.archetypes.registry as registry
    from app.schemas import AgentVerdict, JuryVerdict

    def run_diagnostic(request, **kwargs):
        hostile = "ATTACK" in request.prompt
        verdict = AgentVerdict(agent_name="AdversarialSpecialist",
                               root_cause="PROMPT_INJECTION" if hostile else "STABLE",
                               confidence_score=0.9, mitigation_strategy="x", evidence={})
        return SimpleNamespace(jury=JuryVerdict(verdicts=[verdict], primary_verdict=verdict,
                                                jury_confidence=0.9, is_adversarial=hostile,
                                                failure_summary="d"))

    monkeypatch.setattr(routes.failure_agent, "run_diagnostic", run_diagnostic)
    grown = []
    monkeypatch.setattr(registry.adversarial_registry, "add_confirmed_detection",
                        lambda **kwargs: grown.append(kwargs) or False)
    return grown


def _turn(actor, key: str, n: int, **extra) -> dict:
    prompt = extra.pop("prompt", f"{MARK[key]['prompt']} turn {n}")
    body = monitor_body(prompt, f"{MARK[key]['answer']} {n}", **extra)
    response = actor.post("/api/v1/monitor", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_same_session_id_two_tenants_is_two_sessions(a, b, fakedb):
    """Attack 5a."""
    _turn(a, "a", 1, session_id="sess-1")
    _turn(a, "a", 2, session_id="sess-1")
    _turn(b, "b", 1, session_id="sess-1")
    docs = fakedb["session_context"].docs
    assert len(docs) == 2, "two tenants share one session document"
    by_tenant = {d["tenant_id"]: " ".join(t["content"] for t in d["turns"]) for d in docs}
    assert set(by_tenant) == {TENANT_A["tenant_id"], TENANT_B["tenant_id"]}
    assert MARK["b"]["prompt"] not in by_tenant[TENANT_A["tenant_id"]]
    assert MARK["a"]["prompt"] not in by_tenant[TENANT_B["tenant_id"]]
    assert len(next(d for d in docs if d["tenant_id"] == TENANT_A["tenant_id"])["turns"]) == 4


def test_session_existence_is_not_observable(a, b):
    before = _turn(b, "b", 1, session_id="sess-2")
    _turn(a, "a", 1, session_id="sess-3")
    _turn(a, "a", 2, session_id="sess-3")
    after_first_use_by_b = _turn(b, "b", 1, session_id="sess-3")
    assert before["archetype"] == after_first_use_by_b["archetype"]


def test_conversation_escalation_is_tenant_scoped(a, b, adversarial_jury, fakedb):
    """Attack 5b: A's hostile turns must not make B's conversation look like an escalation."""
    for n in range(3):
        last = _turn(a, "a", n, conversation_id="conv-1", run_full_jury=True,
                     prompt=f"ATTACK {MARK['a']['prompt']} {n}")
    assert last["multi_turn_escalation"]["pattern"] == "REPEATED_REFUSED"
    theirs = _turn(b, "b", 0, conversation_id="conv-1", run_full_jury=True,
                   prompt=f"ATTACK {MARK['b']['prompt']}")
    assert theirs["multi_turn_escalation"] is None
    assert {d.get("tenant_id") for d in fakedb["conversation_turns"].docs} == {
        TENANT_A["tenant_id"], TENANT_B["tenant_id"]}


def test_a_guessed_session_id_yields_nothing(a, b, fakedb):
    _turn(a, "a", 1, session_id="customer-42")
    from engine.session_store import get_context

    assert get_context("customer-42", tenant_id=TENANT_B["tenant_id"]) == []
    assert len(get_context("customer-42", tenant_id=TENANT_A["tenant_id"])) == 2


def test_legacy_session_without_a_tenant_is_not_read(fakedb):
    from engine.session_store import get_context

    fakedb["session_context"].insert_one({"session_id": "old", "turns": [{"role": "user", "content": "legacy"}],
                                          "summary": ""})
    assert get_context("old", tenant_id=TENANT_A["tenant_id"]) == []


@pytest.mark.parametrize("tenant", [None, ""])
def test_session_store_does_nothing_without_a_tenant(fakedb, tenant):
    from engine import session_store

    session_store.store_turn("s", "user", "text", tenant_id=tenant)
    assert fakedb["session_context"].docs == []
    assert session_store.get_context("s", tenant_id=tenant) == []
    session_store.store_turn("s", "user", "text", tenant_id=TENANT_A["tenant_id"])
    assert session_store.get_context("s", tenant_id=tenant) == []
    session_store.clear_session("s", tenant_id=tenant)
    assert len(fakedb["session_context"].docs) == 1


def test_in_memory_fallback_is_also_scoped(monkeypatch):
    from engine import session_store

    monkeypatch.setattr(session_store, "_get_collection", lambda: None)
    session_store.store_turn("s", "user", "from a", tenant_id=TENANT_A["tenant_id"])
    session_store.store_turn("s", "user", "from b", tenant_id=TENANT_B["tenant_id"])
    assert [t["content"] for t in session_store.get_context("s", tenant_id=TENANT_A["tenant_id"])] == ["from a"]
    assert [t["content"] for t in session_store.get_context("s", tenant_id=TENANT_B["tenant_id"])] == ["from b"]


def test_turn_tracker_does_nothing_without_a_tenant(fakedb):
    from engine.multi_turn_tracker import check_multi_turn_escalation

    result = check_multi_turn_escalation("c", "ATTACK bomb", "FACTUAL", True, 0.9)
    assert result.is_escalating is False and fakedb["conversation_turns"].docs == []


def test_session_and_turn_stores_request_tenant_indexes(a, adversarial_jury, fakedb):
    _turn(a, "a", 1, session_id="s", conversation_id="c")
    assert ("tenant_id", "conversation_id", "timestamp") in fakedb["conversation_turns"].index_keys()
    import engine.session_store as session_store
    session_store.request_indexes(fakedb["session_context"])
    assert ("tenant_id", "session_id") in fakedb["session_context"].index_keys()
