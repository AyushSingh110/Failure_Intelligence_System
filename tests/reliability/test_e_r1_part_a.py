"""
E-R1 Part A: the 17 decision-contract fixtures (21 cases) against the current
/monitor route, plus the pre-D-060 route (B0) where its source is available.

B2 (current): no case may deliver a replacement, claim SUPPORTED / PASS_VERIFIED
or FULLY_PROVENANCED from unknown evidence, clear an escalation, or call a
missing ensemble stable. Each case also pins its specific state and action.
B0 (route source at f0888bb): each case must show exactly the documented defect,
so the audit's description of the old behaviour is itself tested.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .e_r1_part_a import CASES, observe

REPO = Path(__file__).resolve().parents[2]
LEGACY_COMMIT = "f0888bb270692433988a15e93e95262fc6488796"   # last commit before D-060

# (evidence_state, answer_action, requires_human_review, provenance_label) under D-060
EXPECTED_B2 = {
    "A1":   ("INSUFFICIENT_EVIDENCE", "PASS_QUALIFIED", False, "REQUIRES_TOOL_VERIFICATION"),
    "A2":   ("INSUFFICIENT_EVIDENCE", "PASS_QUALIFIED", False, "REQUIRES_TOOL_VERIFICATION"),
    "A3":   ("CONTRADICTED",          "FLAG",           False, "REQUIRES_TOOL_VERIFICATION"),
    "A4":   ("VERIFIER_UNAVAILABLE",  "PASS_QUALIFIED", False, "REQUIRES_TOOL_VERIFICATION"),
    "A5":   ("VERIFIER_UNAVAILABLE",  "PASS_QUALIFIED", False, "REQUIRES_TOOL_VERIFICATION"),
    "A6a":  ("INSUFFICIENT_EVIDENCE", "FLAG",           False, "UNVERIFIED_MODEL_INFERENCE"),
    "A6b":  ("VERIFIER_UNAVAILABLE",  "FLAG",           False, "UNVERIFIED_MODEL_INFERENCE"),
    "A7":   ("CONTRADICTED",          "FLAG",           False, "PARTIALLY_PROVENANCED"),
    "A8":   ("INSUFFICIENT_EVIDENCE", "FLAG",           False, "UNVERIFIED_MODEL_INFERENCE"),
    "A9":   ("SUPPORTED",             "FLAG",           False, "FULLY_PROVENANCED"),
    "A10":  ("SUPPORTED",             "FLAG",           False, "FULLY_PROVENANCED"),
    "A11":  ("NOT_CHECKED",           "PASS_QUALIFIED", False, "UNVERIFIED_MODEL_INFERENCE"),
    "A12a": ("SUPPORTED",             "FLAG",           False, "FULLY_PROVENANCED"),
    "A12b": ("SUPPORTED",             "FLAG",           False, "FULLY_PROVENANCED"),
    "A13":  ("INSUFFICIENT_EVIDENCE", "ESCALATE",       True,  "UNVERIFIED_MODEL_INFERENCE"),
    "A14":  ("INSUFFICIENT_EVIDENCE", "ESCALATE",       True,  "UNVERIFIED_MODEL_INFERENCE"),
    "A15":  ("INSUFFICIENT_EVIDENCE", "FLAG",           False, "REQUIRES_TOOL_VERIFICATION"),
    "A16a": ("INSUFFICIENT_EVIDENCE", "FLAG",           False, "PARTIALLY_PROVENANCED"),
    "A16b": ("INSUFFICIENT_EVIDENCE", "FLAG",           False, "UNVERIFIED_MODEL_INFERENCE"),
}

REQUIRED_REASONS = {
    "A6a":  "shadow_agreement_is_not_evidence",
    "A7":   "proposed_value_not_in_evidence",
    "A8":   "ensemble_unavailable (1 shadow responses)",
    "A11":  "ensemble_unavailable (0 shadow responses)",
    "A14":  "wikipedia_context_insufficient",
    "A15":  "cache_stale_for_live_question",
    "A16a": "cache_source_class=tenant_feedback",
}

ROUTE_CASES = [c for c in CASES if c.function is None]


def test_every_planned_fixture_is_covered():
    assert sorted({c.fixture for c in CASES}, key=lambda f: int(f[1:])) == [f"A{i}" for i in range(1, 18)]
    assert set(EXPECTED_B2) == {c.id for c in ROUTE_CASES}


@pytest.mark.parametrize("case", ROUTE_CASES, ids=[c.id for c in ROUTE_CASES])
def test_contract_delivers_original_answer_with_no_forbidden_outcome(case):
    obs = observe(case)

    assert obs["violations"] == [], obs
    assert obs["replaced"] is False
    assert obs["delivered"] == case.scenario.primary[:160]
    assert obs["fix_applied"] is False
    state, action, review, label = EXPECTED_B2[case.id]
    assert (obs["evidence_state"], obs["answer_action"], obs["requires_human_review"],
            obs["provenance_label"]) == (state, action, review, label), obs
    if case.id in REQUIRED_REASONS:
        assert REQUIRED_REASONS[case.id] in obs["reasons"], obs["reasons"]


@pytest.mark.parametrize("case", [c for c in ROUTE_CASES if c.b0_expected & {"replaced"}],
                         ids=lambda c: c.id)
def test_a_contained_correction_is_reported_not_lost(case):
    obs = observe(case)
    assert obs["proposed_output"], "the engine's proposal must stay visible for review"
    assert obs["proposed_output"] != case.scenario.primary[:160]
    assert not obs["failure_summary"].startswith("AUTO-FIXED")


def test_a17_unchecked_reasoning_step_is_not_checked_and_makes_no_call():
    """D-064: the dead ground-truth call is gone and the step says NOT_CHECKED."""
    case = next(c for c in CASES if c.id == "A17")
    obs = observe(case)
    assert obs["violations"] == [], obs
    assert obs["pipeline_calls"] == 0
    assert (obs["verification_status"], obs["method"], obs["confidence"]) == ("NOT_CHECKED", "not_checked", 0.0)
    assert obs["branch_reachable_sources"] == []


def test_a17_b0_step_verifier_made_a_discarded_call(legacy_step_verifier):
    case = next(c for c in CASES if c.id == "A17")
    obs = observe(case, step_module=legacy_step_verifier)
    assert set(obs["violations"]) == set(case.b0_expected), obs
    assert obs["pipeline_calls"] == 1 and obs["method"] == "no_external_source"


def _legacy_module(path: str, name: str):
    try:
        source = subprocess.run(
            ["git", "show", f"{LEGACY_COMMIT}:{path}"], cwd=REPO,
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip(f"pre-D-060 source {path} ({LEGACY_COMMIT[:7]}) not in this checkout")
    from .harness import load_route_from_source
    return load_route_from_source(source, name)


@pytest.fixture(scope="module")
def legacy_route():
    return _legacy_module("app/routes/monitor.py", "legacy_monitor_f0888bb")


@pytest.fixture(scope="module")
def legacy_step_verifier():
    return _legacy_module("engine/reasoning/step_verifier.py", "legacy_step_verifier_f0888bb")


@pytest.mark.parametrize("case", ROUTE_CASES, ids=[c.id for c in ROUTE_CASES])
def test_b0_pre_containment_route_shows_the_documented_defect(case, legacy_route):
    obs = observe(case, legacy_route)
    assert set(obs["violations"]) == set(case.b0_expected), obs


@pytest.mark.parametrize("case", ROUTE_CASES, ids=[c.id for c in ROUTE_CASES])
def test_classifier_inputs_and_risk_are_unchanged_by_containment(case, legacy_route):
    """D-060 changes what is delivered, not the risk signal: XGBoost sees the same features."""
    before = observe(case, legacy_route)
    after = observe(case)
    assert after["classifier_inputs"] == before["classifier_inputs"]
    assert after["classifier_probability"] == before["classifier_probability"]
    assert after["high_failure_risk"] == before["high_failure_risk"]


def test_jury_regex_defect_path_is_contained(legacy_route):
    """
    OD-R3 dependency. engine/agents/adversarial/patterns.py:94 (`\\x00||`) makes the
    jury diagnose TOKEN_SMUGGLING on every request; the fix engine then proposes
    SANITIZE_AND_RERUN, a canned refusal. Before D-060 that refusal replaced the
    answer. With the containment in place it is a proposal only — which is what
    must hold before Workstream A fixes the regex and re-activates the other
    correction paths. The regex itself is not touched here.
    """
    from .e_r1_part_a import AGREE3, FACTUAL_Q, BELL, Case
    from .harness import Scenario
    case = Case("R1", "R1", "jury says TOKEN_SMUGGLING (regex defect)", True,
                Scenario(FACTUAL_Q, BELL, AGREE3, "TOKEN_SMUGGLING"))
    before, after = observe(case, legacy_route), observe(case)
    assert before["fix_strategy"] == "SANITIZE_AND_RERUN" and before["replaced"] is True
    assert after["fix_strategy"] == "SANITIZE_AND_RERUN" and after["replaced"] is False
    assert after["violations"] == []
