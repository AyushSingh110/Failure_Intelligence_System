"""
SDK `correct` mode never substitutes (D-065, OD-S0-3).

Covers fie.monitor's correct_wrapper and the OpenAI / Anthropic wrappers against:
an updated server, an older server without `reliability`, a legacy
fix_applied=True response, an incompletely verified proposal, a tenant
correction, and the documented response format (MonitorResponse).
"""
from __future__ import annotations

import types

import pytest

from fie.correction import (
    AUTO_REPLACE, CONTRACT_CURRENT, CONTRACT_LEGACY, UNVERIFIED, proposal_from_response,
)

from .harness import delivered_by_sdk

PRIMARY = "Alexander Graham Bell invented the telephone."
OTHER = "Antonio Meucci invented the telephone."


def _legacy(**fix) -> dict:
    """A response from a server before D-060: no `reliability` key at all."""
    base = {"fixed_output": OTHER, "fix_applied": True, "fix_strategy": "SHADOW_CONSENSUS"}
    base.update(fix)
    return {"high_failure_risk": True, "fix_result": base}


def _current(evidence="INSUFFICIENT_EVIDENCE", action="FLAG", source="SHADOW_CONSENSUS") -> dict:
    return {
        "high_failure_risk": True,
        "fix_result": {"fixed_output": PRIMARY, "fix_applied": False, "fix_strategy": "SHADOW_CONSENSUS",
                       "proposed_output": OTHER},
        "reliability": {"evidence_state": evidence, "answer_action": action, "correction_applied": False,
                        "proposal_source": source, "proposal_status": UNVERIFIED, "warn_only": True},
    }


def _run(response: dict):
    got = []
    delivered = delivered_by_sdk(response, PRIMARY, on_proposal=got.append)
    return delivered, got


def test_auto_replacement_is_off_and_not_configurable():
    import inspect

    import fie.correction as fc
    assert AUTO_REPLACE is False
    assert "auto_replace" not in inspect.signature(importlib_monitor().monitor).parameters
    assert "AUTO_REPLACE" not in inspect.getsource(importlib_monitor())   # no switch read anywhere
    assert fc.CorrectionProposal.__dataclass_fields__["applied"].default is False


def importlib_monitor():
    import importlib
    return importlib.import_module("fie.monitor")


def test_updated_server_original_answer_and_proposal():
    delivered, got = _run(_current())
    assert delivered == PRIMARY
    [p] = got
    assert (p.proposed_output, p.server_contract, p.status, p.applied) == (OTHER, CONTRACT_CURRENT, UNVERIFIED, False)
    assert p.evidence_state == "INSUFFICIENT_EVIDENCE"


def test_older_server_without_reliability_and_no_fix():
    delivered, got = _run({"high_failure_risk": False, "fix_result": None})
    assert delivered == PRIMARY and got == []


def test_legacy_fix_applied_true_is_only_a_proposal():
    delivered, got = _run(_legacy())
    assert delivered == PRIMARY
    [p] = got
    assert (p.proposed_output, p.server_contract, p.evidence_state, p.status) == (
        OTHER, CONTRACT_LEGACY, "UNKNOWN", UNVERIFIED)


def test_missing_reliability_is_not_read_as_verification():
    """A legacy response even claiming a ground-truth source is still unverified."""
    delivered, got = _run(_legacy(fix_strategy="GT_VERIFIED (Wikidata (Q11035))", fix_confidence=1.0))
    assert delivered == PRIMARY
    assert got[0].status == UNVERIFIED and got[0].answer_action == "UNKNOWN"


@pytest.mark.parametrize("evidence, action", [
    ("INSUFFICIENT_EVIDENCE", "FLAG"), ("VERIFIER_UNAVAILABLE", "PASS_QUALIFIED"),
    ("CONTRADICTED", "FLAG"), ("SUPPORTED", "FLAG"),
    ("SUPPORTED", "PASS_VERIFIED"),   # not reachable on the server today; still no substitution
])
def test_incompletely_verified_proposal_is_never_delivered(evidence, action):
    delivered, got = _run(_current(evidence, action))
    assert delivered == PRIMARY
    assert got[0].status == UNVERIFIED and got[0].applied is False


def test_tenant_correction_is_a_labelled_proposal():
    delivered, got = _run(_current(source="cache:tenant_feedback"))
    assert delivered == PRIMARY
    assert got[0].source == "cache:tenant_feedback" and got[0].status == UNVERIFIED


def test_callback_failure_does_not_break_the_call():
    def boom(_p):
        raise RuntimeError("user bug")
    assert delivered_by_sdk(_current(), PRIMARY, on_proposal=boom) == PRIMARY


@pytest.mark.parametrize("response", [
    {}, {"fix_result": None, "reliability": None}, {"fix_result": {}},
    {"fix_result": {"fix_applied": True, "fixed_output": ""}},
    {"fix_result": {"fix_applied": True, "fixed_output": PRIMARY}},   # "fix" identical to the answer
])
def test_degenerate_responses_give_original_and_no_proposal(response):
    delivered, got = _run(response)
    assert delivered == PRIMARY and got == []


def test_documented_response_format_round_trip():
    """A full MonitorResponse, as the server serialises it, is read without error."""
    from app.schemas import FailureSignalVector, FixResult, MonitorResponse, ReliabilityAssessment
    resp = MonitorResponse(
        shadow_model_results=[], all_model_outputs=[PRIMARY], ollama_available=False,
        failure_signal_vector=FailureSignalVector(
            agreement_score=1.0, fsd_score=0.0, answer_counts={}, entropy_score=0.0,
            ensemble_disagreement=False, ensemble_similarity=1.0),
        archetype="STABLE", embedding_distance=0.0, high_failure_risk=True, failure_summary="x",
        fix_result=FixResult(fixed_output=PRIMARY, fix_applied=False, fix_strategy="SHADOW_CONSENSUS",
                             fix_explanation="", original_output=PRIMARY, root_cause="FACTUAL_HALLUCINATION",
                             proposed_output=OTHER),
        reliability=ReliabilityAssessment(evidence_state="INSUFFICIENT_EVIDENCE", answer_action="FLAG",
                                          proposal_source="SHADOW_CONSENSUS", proposal_status=UNVERIFIED),
    )
    delivered, got = _run(resp.model_dump(mode="json"))
    assert delivered == PRIMARY and got[0].server_contract == CONTRACT_CURRENT


def test_end_to_end_with_the_real_route():
    """The updated server's own response (A13: escalation then Wikipedia) through the SDK."""
    from .e_r1_part_a import CASES
    from .harness import run_monitor
    case = next(c for c in CASES if c.id == "A13")
    body = run_monitor(case.scenario).response.model_dump(mode="json")
    got = []
    assert delivered_by_sdk(body, case.scenario.primary, on_proposal=got.append) == case.scenario.primary
    assert got and got[0].server_contract == CONTRACT_CURRENT and got[0].source == "RAG_GROQ_GROUNDING"


# ── provider wrappers ────────────────────────────────────────────────────────

class _FakeFIE:
    def __init__(self, result):
        self.result = result

    def monitor(self, **kwargs):
        return self.result


def _openai_response(text):
    msg = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _anthropic_response(text):
    return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])


@pytest.mark.parametrize("module, make, read", [
    ("fie.integrations.openai", _openai_response, lambda r: r.choices[0].message.content),
    ("fie.integrations.anthropic", _anthropic_response, lambda r: r.content[0].text),
])
@pytest.mark.parametrize("server", ["legacy", "current"])
def test_provider_wrapper_never_mutates_the_response(module, make, read, server):
    import importlib
    Client = importlib.import_module(module).Client
    client = object.__new__(Client)          # skip the provider SDK import in __init__
    got = []
    client._fie_client = _FakeFIE(_legacy() if server == "legacy" else _current())
    client._run_full_jury = True
    client._on_proposal = got.append
    response = make(PRIMARY)
    client._run_fie_correct("Who invented the telephone?", PRIMARY, "model", 1.0, response)
    assert read(response) == PRIMARY
    assert got and got[0].proposed_output == OTHER and got[0].applied is False


def test_proposal_from_response_reads_only_dicts():
    assert proposal_from_response(None, PRIMARY) is None
    assert proposal_from_response("not a dict", PRIMARY) is None
