"""
Unit tests for engine/reliability_contract.py (D-060), for the evidence states the
verifiers now report, and B0 facts about the production functions the contract
contains but does not change.
"""
from __future__ import annotations

import pytest

from engine.reliability_contract import (
    FULLY_PROVENANCED,
    AnswerAction,
    EvidenceAssessment,
    EvidenceState,
    combine_states,
    contain_fix,
    decide_action,
    ensemble_available,
    parse_verdict_strict,
    provenance_label,
    step_evidence_state,
)

S = EvidenceState


# ── strict parser vs production parser ───────────────────────────────────────

NON_VERDICTS = {
    "empty":            "",
    "uncertain":        "VERDICT: UNCERTAIN\nCONFIDENCE: 0.4\nGROUNDED_ANSWER: x",
    "no_verdict_line":  "The model output looks right to me.",
    "echoed_template":  "VERDICT: CONSISTENT or INCONSISTENT or UNCERTAIN\nCONFIDENCE: 0.0 to 1.0",
    "two_verdicts":     "VERDICT: CONSISTENT\nVERDICT: INCONSISTENT\nCONFIDENCE: 0.9",
    "no_confidence":    "VERDICT: CONSISTENT\nGROUNDED_ANSWER: x",
    "bad_confidence":   "VERDICT: CONSISTENT\nCONFIDENCE: high\nGROUNDED_ANSWER: x",
    "out_of_range":     "VERDICT: CONSISTENT\nCONFIDENCE: 7\nGROUNDED_ANSWER: x",
    "unknown_word":     "VERDICT: PROBABLY\nCONFIDENCE: 0.8",
}


@pytest.mark.parametrize("name", sorted(NON_VERDICTS))
def test_strict_parser_never_reads_a_non_verdict_as_support(name):
    assert parse_verdict_strict(NON_VERDICTS[name]).state is S.INSUFFICIENT_EVIDENCE


PARSERS = ["engine.verifier.serper_verifier", "engine.verifier.wikidata_verifier"]


@pytest.mark.parametrize("module", PARSERS)
@pytest.mark.parametrize("name", sorted(set(NON_VERDICTS) - {"two_verdicts"}))
def test_b0_production_parser_reads_every_non_verdict_as_consistent(module, name):
    """B0, unchanged production code: matches defaults to True. D-060 contains it, does not change it."""
    import importlib
    matches, _confidence, _value = importlib.import_module(module)._parse_groq_verdict(NON_VERDICTS[name])
    assert matches is True


@pytest.mark.parametrize("module", PARSERS)
def test_b0_production_parser_lets_the_last_of_two_verdicts_win(module):
    import importlib
    parse = importlib.import_module(module)._parse_groq_verdict
    assert parse("VERDICT: CONSISTENT\nVERDICT: INCONSISTENT\nCONFIDENCE: 0.9")[0] is False
    assert parse("VERDICT: INCONSISTENT\nVERDICT: CONSISTENT\nCONFIDENCE: 0.9")[0] is True


def test_strict_parser_explicit_verdicts():
    ok = parse_verdict_strict("VERDICT: CONSISTENT\nCONFIDENCE: 0.85\nGROUNDED_ANSWER: Paris")
    assert (ok.state, ok.confidence, ok.value) == (S.SUPPORTED, 0.85, "Paris")
    bad = parse_verdict_strict("verdict: inconsistent\nconfidence: 0\nCORRECT_VALUE: Bell", "CORRECT_VALUE")
    assert (bad.state, bad.confidence, bad.value) == (S.CONTRADICTED, 0.0, "Bell")


def test_b0_heuristic_fallbacks():
    """B0: the word-overlap fallback 'confirms' an answer with no 4-letter word, and
    'contradicts' a short correct answer against a long snippet."""
    from engine.verifier.serper_verifier import _heuristic_match
    assert _heuristic_match("42", "Example Corp lists 42 open roles today.")[:2] == (True, 0.0)
    snippet = "Jane Doe has served as chief executive officer of Example Corp since March 2024."
    assert _heuristic_match("Jane Doe.", snippet)[0] is False


def test_b0_degenerate_consensus():
    from engine.fix_engine import _apply_shadow_consensus
    from engine.verifier.ground_truth_pipeline import _compute_consensus_strength
    assert _compute_consensus_strength(["only one"], [2.0]) == 1.0
    assert _compute_consensus_strength(["a", "b"], [2.0, 2.0]) == 0.5
    two = _apply_shadow_consensus("q", "primary", ["first", "second"], 0.6, [2.0, 2.0])
    assert (two.fix_applied, two.fixed_output) == (True, "first")
    three = _apply_shadow_consensus("q", "primary", ["low", "mid", "high"], 0.6, [1.0, 2.0, 3.0])
    assert (three.fix_applied, three.fixed_output) == (True, "high")


# ── verifiers: additive evidence state, legacy values unchanged ──────────────

class _Judge:
    def __init__(self, text):
        self.text = text

    def complete(self, *a, **k):
        from engine.groq_service import GroqModelResponse
        return GroqModelResponse(model_name="judge", output_text=self.text or "",
                                 success=self.text is not None)


@pytest.mark.parametrize("reply, state", [
    ("VERDICT: UNCERTAIN\nCONFIDENCE: 0.4\nGROUNDED_ANSWER: x", "INSUFFICIENT_EVIDENCE"),
    ("no verdict here", "INSUFFICIENT_EVIDENCE"),
    ("VERDICT: CONSISTENT\nCONFIDENCE: 0.9\nGROUNDED_ANSWER: x", "SUPPORTED"),
    ("VERDICT: INCONSISTENT\nCONFIDENCE: 0.0\nGROUNDED_ANSWER: y", "CONTRADICTED"),
    (None, "VERIFIER_UNAVAILABLE"),
])
def test_serper_detailed_verdict_keeps_legacy_triple(monkeypatch, reply, state):
    import engine.groq_service as gs
    import engine.verifier.serper_verifier as sv
    monkeypatch.setattr(gs, "get_groq_service", lambda: _Judge(reply))
    detailed = sv._verify_via_groq_detailed("Jane Doe.", "Jane Doe is CEO of Example Corp.", "q")
    assert detailed[3] == state
    assert detailed[:3] == sv._verify_via_groq("Jane Doe.", "Jane Doe is CEO of Example Corp.", "q")


@pytest.mark.parametrize("desc, reply, state", [
    ("", "VERDICT: CONSISTENT\nCONFIDENCE: 0.9\nCORRECT_VALUE: x", "INSUFFICIENT_EVIDENCE"),
    ("telecommunications device", None, "VERIFIER_UNAVAILABLE"),
    ("telecommunications device", "VERDICT: UNCERTAIN\nCONFIDENCE: 0.5\nCORRECT_VALUE: UNKNOWN",
     "INSUFFICIENT_EVIDENCE"),
])
def test_wikidata_detailed_verdict_keeps_legacy_triple(monkeypatch, desc, reply, state):
    import engine.groq_service as gs
    import engine.verifier.wikidata_verifier as wv
    monkeypatch.setattr(gs, "get_groq_service", lambda: _Judge(reply))
    detailed = wv._verify_via_groq_detailed("telephone", desc, "inventor", "Bell")
    assert detailed[3] == state
    assert detailed[:3] == wv._verify_via_groq("telephone", desc, "inventor", "Bell")


# ── assessment, action, label ────────────────────────────────────────────────

def _gt(**kw):
    from engine.verifier.ground_truth_pipeline import GroundTruthPipelineResult
    return GroundTruthPipelineResult(**kw)


def test_supported_requires_evidence_and_source_identity():
    from engine.reliability_contract import assess_ground_truth
    base = dict(verifier="serper", evidence_state="SUPPORTED", verified_answer="Jane Doe")
    assert assess_ground_truth(_gt(**base), "Jane Doe").state is S.INSUFFICIENT_EVIDENCE
    assert assess_ground_truth(_gt(**base, evidence_text="Jane Doe is CEO"), "Jane Doe").state \
        is S.INSUFFICIENT_EVIDENCE
    full = assess_ground_truth(_gt(**base, evidence_text="Jane Doe is CEO", evidence_source="Serper"), "Jane Doe")
    assert full.state is S.SUPPORTED


def test_unknown_or_missing_states_are_never_support():
    from engine.reliability_contract import assess_ground_truth
    assert assess_ground_truth(None, "x").state is S.NOT_CHECKED
    assert assess_ground_truth(_gt(), "x").state is S.INSUFFICIENT_EVIDENCE
    assert assess_ground_truth(_gt(verifier="serper", evidence_state="WHATEVER"), "x").state \
        is S.INSUFFICIENT_EVIDENCE
    assert assess_ground_truth(_gt(verifier="self_consistency", source="self_consistency_reasoning"),
                               "x").state is S.NOT_CHECKABLE


def test_cache_hits_are_never_support():
    from engine.reliability_contract import assess_ground_truth
    for source_class in ("tenant_feedback", "system", "unknown"):
        a = assess_ground_truth(_gt(verifier="cache", verified_answer="x", cache_source_class=source_class,
                                    evidence_source="serper", provenance_label=FULLY_PROVENANCED), "x")
        assert a.state is S.INSUFFICIENT_EVIDENCE
        assert provenance_label(a, FULLY_PROVENANCED, "GENERAL_KNOWLEDGE", source_class) != FULLY_PROVENANCED


@pytest.mark.parametrize("state", list(S))
def test_pass_verified_is_unreachable_while_no_verifier_is_validated(state):
    for verifier in ("serper", "wikidata", "cache", ""):
        for review in (False, True):
            for risk in (False, True):
                assert decide_action(state, verifier, review, risk) is not AnswerAction.PASS_VERIFIED


def test_pass_verified_gate_opens_only_for_a_listed_verifier(monkeypatch):
    import engine.reliability_contract as rc
    monkeypatch.setattr(rc, "VALIDATED_VERIFIERS", frozenset({"wikidata"}))
    assert rc.decide_action(S.SUPPORTED, "wikidata", False, False) is AnswerAction.PASS_VERIFIED
    assert rc.decide_action(S.SUPPORTED, "serper", False, False) is AnswerAction.PASS_QUALIFIED
    assert rc.decide_action(S.SUPPORTED, "wikidata", True, False) is AnswerAction.ESCALATE


@pytest.mark.parametrize("state", [s for s in S if s is not S.SUPPORTED])
def test_fully_provenanced_only_for_supported(state):
    a = EvidenceAssessment(state, "wikidata", evidence_source="Wikidata (Q1)", evidence_preview="desc")
    for legacy in (FULLY_PROVENANCED, "PARTIALLY_PROVENANCED", "UNVERIFIED_MODEL_INFERENCE"):
        assert provenance_label(a, legacy, "GENERAL_KNOWLEDGE") != FULLY_PROVENANCED


def test_label_never_stronger_than_legacy():
    a = EvidenceAssessment(S.SUPPORTED, "wikidata", evidence_source="Wikidata (Q1)", evidence_preview="d")
    assert provenance_label(a, "PARTIALLY_PROVENANCED", "GENERAL_KNOWLEDGE") == "PARTIALLY_PROVENANCED"


def test_combine_states_conflict_is_not_a_majority():
    assert combine_states([S.SUPPORTED, S.CONTRADICTED]) is S.CONFLICTING_EVIDENCE
    assert combine_states([S.SUPPORTED, S.SUPPORTED, S.CONTRADICTED]) is S.CONFLICTING_EVIDENCE
    assert combine_states([S.VERIFIER_UNAVAILABLE, S.INSUFFICIENT_EVIDENCE]) is S.INSUFFICIENT_EVIDENCE
    assert combine_states([]) is S.NOT_CHECKED


def test_ensemble_needs_two_answers():
    assert [ensemble_available(n) for n in range(4)] == [False, False, True, True]


@pytest.mark.parametrize("is_correct, confidence, method, state", [
    (True, 0.0, "no_external_source", S.NOT_CHECKED),
    (True, 0.0, "offline_skipped", S.NOT_CHECKED),
    (True, 0.9, "hedge_detected", S.NOT_CHECKED),
    (True, 0.9, "arithmetic_eval", S.SUPPORTED),
    (False, 0.9, "arithmetic_eval", S.CONTRADICTED),
    (True, 0.7, "shadow_consistency", S.INSUFFICIENT_EVIDENCE),
    (False, 0.7, "shadow_consistency", S.CONTRADICTED),
])
def test_step_evidence_state(is_correct, confidence, method, state):
    assert step_evidence_state(is_correct, confidence, method) is state


# ── containment of a proposed fix ────────────────────────────────────────────

def _fix(**kw):
    from app.schemas import FixResult
    base = dict(fixed_output="candidate", fix_applied=True, fix_strategy="SHADOW_CONSENSUS",
                fix_explanation="why", original_output="primary", root_cause="FACTUAL_HALLUCINATION",
                fix_confidence=0.6, improvement_score=0.4)
    base.update(kw)
    return FixResult(**base)


def test_contain_fix_keeps_original_and_records_proposal():
    delivered, proposed, strategy = contain_fix(_fix(), "primary")
    assert (delivered.fixed_output, delivered.fix_applied, delivered.proposed_output) == ("primary", False, "candidate")
    assert (proposed, strategy) == ("candidate", "SHADOW_CONSENSUS")
    assert delivered.fix_strategy == "SHADOW_CONSENSUS" and delivered.warning


def test_contain_fix_passes_through_non_replacements():
    assert contain_fix(None, "p") == (None, "", "")
    escalation = _fix(fixed_output="primary", fix_applied=False, fix_strategy="HUMAN_ESCALATION")
    assert contain_fix(escalation, "primary")[0] is escalation
    same = _fix(fixed_output="primary")
    assert contain_fix(same, "primary")[0] is same


# ── SDK `correct` mode ───────────────────────────────────────────────────────

def test_sdk_correct_mode_returns_original_for_a_contained_response():
    from .harness import delivered_by_sdk
    delivered, _, _ = contain_fix(_fix(), "primary")
    assert delivered_by_sdk({"fix_result": delivered.model_dump()}, "primary") == "primary"


def test_b0_sdk_correct_mode_still_substitutes_a_legacy_response():
    """Residual risk, documented: an SDK talking to a server without D-060 still substitutes."""
    from .harness import delivered_by_sdk
    assert delivered_by_sdk({"fix_result": _fix().model_dump()}, "primary") == "candidate"


def test_reliability_states_are_disjoint_from_workstream_a_decisions():
    workstream_a = {"ALLOW", "REVIEW", "BLOCK"}   # fie/policy.py
    assert not {s.value for s in S} & workstream_a
    assert not {a.value for a in AnswerAction} & workstream_a


def test_contract_module_is_pure():
    import ast
    import inspect

    import engine.reliability_contract as rc
    tree = ast.parse(inspect.getsource(rc))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "re", "dataclasses", "enum", "typing"}, imported
