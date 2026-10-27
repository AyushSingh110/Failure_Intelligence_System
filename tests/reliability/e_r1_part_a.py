"""
E-R1 Part A — decision-contract fixtures A1..A17
(research/reliability/2026-10-27_E-R1_verifier_benchmark_plan.md §2).

Each case runs the REAL /monitor handler and ground-truth pipeline with only the
I/O leaves replaced (tests/reliability/harness.py), then reads what the user of
the SDK's `correct` mode would receive. The same case can be run against the
current route (B2: the D-060 contract) or against a historical route source
(B0: pre-containment behaviour) by the research runner.

A17 is a function-level case: the reasoning verifier's factual branch.

Forbidden outcomes for an unknown, failed, malformed or degenerate input:
  replaced            the delivered answer differs from the primary answer
  fully_provenanced   FULLY_PROVENANCED on an unknown input, or on a replacement answer
  supported           reliability.evidence_state == SUPPORTED    (unknown_input cases)
  pass_verified       reliability.answer_action == PASS_VERIFIED
  escalation_cleared  ground truth escalated but requires_human_review is False
  claims_stable       "outputs are stable" with fewer than two shadow answers
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Callable, Optional

from .harness import Scenario, Shadow, cache_doc, delivered_by_sdk, run_monitor

TEMPORAL_Q = "Who is the current CEO of Example Corp right now?"
FACTUAL_Q = "Who invented the telephone?"
SNIPPET = "Jane Doe has served as chief executive officer of Example Corp since March 2024, the company said."
SERPER = {"answerBox": {"answer": SNIPPET}}
DESC = "telecommunications device for two-way voice communication, patented by Alexander Graham Bell"


def _serper_judge(reply: Optional[str]) -> Callable[[str], Optional[str]]:
    return lambda prompt: reply if "Live search result" in prompt else None


def _wikidata_judge(reply: Optional[str]) -> Callable[[str], Optional[str]]:
    return lambda prompt: reply if "Wikidata description" in prompt else None


AGREE3 = [Shadow("Jane Doe is the CEO."), Shadow("Jane Doe is the CEO."), Shadow("Jane Doe is the CEO.")]
OTHER3 = [Shadow("Antonio Meucci invented the telephone.")] * 3
CONFIRM = "VERDICT: CONSISTENT\nCONFIDENCE: 0.90\nCORRECT_VALUE: Alexander Graham Bell"
BELL = "Alexander Graham Bell invented the telephone."


@dataclass
class Case:
    id:            str
    fixture:       str
    title:         str
    unknown_input: bool                       # True: the verification input itself is unknown/failed/malformed
    scenario:      Optional[Scenario] = None
    source_ref:    str = ""
    b0_expected:   frozenset = field(default_factory=frozenset)   # violations the pre-D-060 route shows
    function:      Optional[Callable[[], dict]] = None


CASES: list[Case] = [
    Case("A1", "A1", "Serper judge replies UNCERTAIN", True,
         Scenario(TEMPORAL_Q, "Jane Doe is the CEO.", AGREE3, "TEMPORAL_KNOWLEDGE_CUTOFF",
                  judge=_serper_judge("VERDICT: UNCERTAIN\nCONFIDENCE: 0.40\nGROUNDED_ANSWER: unknown"),
                  serper_payload=SERPER),
         "serper_verifier.py:_parse_groq_verdict (UNCERTAIN != INCONSISTENT); ground_truth_pipeline.py Serper confirm",
         frozenset({"fully_provenanced", "replaced"})),
    Case("A2", "A2", "Serper judge reply has no VERDICT line", True,
         Scenario(TEMPORAL_Q, "Jane Doe is the CEO.", AGREE3, "TEMPORAL_KNOWLEDGE_CUTOFF",
                  judge=_serper_judge("The model output looks right to me."), serper_payload=SERPER),
         "serper_verifier.py:_parse_groq_verdict defaults matches=True",
         frozenset({"fully_provenanced", "replaced"})),
    Case("A3", "A3", "Serper INCONSISTENT at confidence 0.0", False,
         Scenario(TEMPORAL_Q, "Jane Doe is the CEO.", AGREE3, "TEMPORAL_KNOWLEDGE_CUTOFF",
                  judge=_serper_judge("VERDICT: INCONSISTENT\nCONFIDENCE: 0.0\nGROUNDED_ANSWER: John Roe is the CEO."),
                  serper_payload=SERPER),
         "ground_truth_pipeline.py Serper override has no confidence gate",
         frozenset({"replaced", "fully_provenanced"})),
    Case("A4", "A4", "Judge unavailable; short correct answer vs long snippet", True,
         Scenario(TEMPORAL_Q, "Jane Doe.", AGREE3, "TEMPORAL_KNOWLEDGE_CUTOFF",
                  judge=_serper_judge(None), serper_payload=SERPER),
         "serper_verifier.py:_heuristic_match overlap / len(snippet) <= 0.25 -> mismatch",
         frozenset({"replaced", "fully_provenanced"})),
    Case("A5", "A5", "Judge unavailable; answer '42' has no 4-letter token", True,
         Scenario("How many open roles does Example Corp have right now?", "42", AGREE3,
                  "TEMPORAL_KNOWLEDGE_CUTOFF", judge=_serper_judge(None),
                  serper_payload={"answerBox": {"answer": "Example Corp lists 42 open roles today."}}),
         "serper_verifier.py:_heuristic_match returns (True, 0.0) with no tokens",
         frozenset({"fully_provenanced", "replaced"})),
    Case("A6a", "A6", "Wikidata entity has no description", True,
         Scenario(FACTUAL_Q, BELL, OTHER3, claim=("telephone", "inventor", "Alexander Graham Bell"),
                  wikidata_entity={"id": "Q11035", "label": "telephone", "description": ""}),
         "wikidata_verifier.py:_verify_via_groq returns (True, 0.0, '') -> inconclusive -> shadow consensus",
         frozenset({"replaced"})),
    Case("A6b", "A6", "Wikidata judge unavailable", True,
         Scenario(FACTUAL_Q, BELL, OTHER3, claim=("telephone", "inventor", "Alexander Graham Bell"),
                  wikidata_entity={"id": "Q11035", "label": "telephone", "description": DESC},
                  judge=_wikidata_judge(None)),
         "wikidata_verifier.py: judge failure returns (True, 0.0, '')",
         frozenset({"replaced"})),
    Case("A7", "A7", "Wikidata INCONSISTENT 0.90 with a CORRECT_VALUE not in the description", False,
         Scenario(FACTUAL_Q, "Antonio Meucci invented the telephone.", AGREE3,
                  claim=("telephone", "inventor", "Antonio Meucci"),
                  wikidata_entity={"id": "Q11035", "label": "telephone",
                                   "description": "telecommunications device for two-way voice communication"},
                  judge=_wikidata_judge("VERDICT: INCONSISTENT\nCONFIDENCE: 0.90\n"
                                        "CORRECT_VALUE: Alexander Graham Bell invented the telephone in 1876")),
         "ground_truth_pipeline.py Wikidata override: 8B-written value labelled FULLY_PROVENANCED",
         frozenset({"replaced", "fully_provenanced"})),
    Case("A8", "A8", "One surviving shadow", True,
         Scenario(FACTUAL_Q, BELL,
                  [Shadow("Antonio Meucci invented the telephone."), Shadow("", False), Shadow("", False)]),
         "_compute_consensus_strength / _apply_shadow_consensus: one answer -> strength 1.0",
         frozenset({"replaced"})),
    Case("A9", "A9", "Two equal-weight shadows disagree (after a Wikidata confirmation)", False,
         Scenario(FACTUAL_Q, BELL,
                  [Shadow("Antonio Meucci invented the telephone."), Shadow("Elisha Gray invented the telephone.")],
                  claim=("telephone", "inventor", "Alexander Graham Bell"),
                  wikidata_entity={"id": "Q11035", "label": "telephone", "description": DESC},
                  judge=_wikidata_judge(CONFIRM)),
         "fix_engine.py:_apply_shadow_consensus gate is `< 0.50`; 0.50 -> first shadow wins",
         frozenset({"replaced", "fully_provenanced"})),
    Case("A10", "A10", "Three shadows, weights 3/2/1, all different (after a Wikidata confirmation)", False,
         Scenario(FACTUAL_Q, BELL,
                  [Shadow("Antonio Meucci invented the telephone.", confidence="HIGH"),
                   Shadow("Elisha Gray invented the telephone.", confidence="MEDIUM"),
                   Shadow("Thomas Edison invented the telephone.", confidence="LOW")],
                  claim=("telephone", "inventor", "Alexander Graham Bell"),
                  wikidata_entity={"id": "Q11035", "label": "telephone", "description": DESC},
                  judge=_wikidata_judge(CONFIRM)),
         "fix_engine.py:_apply_shadow_consensus: 3/6 = 0.50 -> self-rated HIGH shadow wins",
         frozenset({"replaced", "fully_provenanced"})),
    Case("A11", "A11", "No shadow responses; no jury finding", True,
         Scenario(FACTUAL_Q, BELL, [Shadow("", False)] * 3, high_risk=None, jury=False),
         "app/routes/monitor.py Step 6: 'Model outputs are stable' with no shadow answers",
         frozenset({"claims_stable"})),
    Case("A12a", "A12", "Wikidata confirms, then shadow consensus replaces", False,
         Scenario(FACTUAL_Q, BELL, OTHER3, claim=("telephone", "inventor", "Alexander Graham Bell"),
                  wikidata_entity={"id": "Q11035", "label": "telephone", "description": DESC},
                  judge=_wikidata_judge(CONFIRM)),
         "app/routes/monitor.py: verified_answer == primary falls to the fix-engine else-branch",
         frozenset({"replaced", "fully_provenanced"})),
    Case("A12b", "A12", "Wikidata confirms, then the Wikipedia fallback replaces", False,
         Scenario(FACTUAL_Q, BELL, [], claim=("telephone", "inventor", "Alexander Graham Bell"),
                  wikidata_entity={"id": "Q11035", "label": "telephone", "description": DESC},
                  judge=_wikidata_judge(CONFIRM),
                  wikipedia=("The telephone was developed by several inventors.",
                             "Antonio Meucci invented the telephone.")),
         "app/routes/monitor.py Wikipedia fallback runs when no fix was applied",
         frozenset({"replaced", "fully_provenanced"})),
    Case("A13", "A13", "GT escalates, then the Wikipedia fallback", True,
         Scenario(FACTUAL_Q, BELL,
                  [Shadow("Antonio Meucci invented the telephone."), Shadow("Elisha Gray invented the telephone.")],
                  wikipedia=("The telephone was developed by several inventors.",
                             "Antonio Meucci invented the telephone.")),
         "app/routes/monitor.py:539-565 clears requires_human_review",
         frozenset({"replaced", "escalation_cleared"})),
    Case("A14", "A14", "Wikipedia judge says the context is insufficient", True,
         Scenario(FACTUAL_Q, BELL,
                  [Shadow("Antonio Meucci invented the telephone."), Shadow("Elisha Gray invented the telephone.")],
                  wikipedia=("The telephone was developed by several inventors.",
                             "The context is insufficient to answer the question.")),
         "engine/rag_grounder.py returns the raw summary at 0.72 with success=True",
         frozenset({"replaced", "escalation_cleared"})),
    Case("A15", "A15", "Cache hit for a live question stored 400 days ago", True,
         Scenario(TEMPORAL_Q, "Jane Doe is the CEO.", AGREE3, "TEMPORAL_KNOWLEDGE_CUTOFF",
                  cache=cache_doc("Alice Smith is the CEO of Example Corp.", "serper", "system", 400)),
         "engine/ground_truth_cache.py has no TTL; pipeline labels every hit FULLY_PROVENANCED",
         frozenset({"replaced", "fully_provenanced"})),
    Case("A16a", "A16", "Cache hit from a tenant correction", True,
         Scenario(FACTUAL_Q, BELL, AGREE3, cache=cache_doc(BELL, "user_feedback", "tenant_feedback", 2)),
         "ground_truth_pipeline.py cache hit -> FULLY_PROVENANCED regardless of source class",
         frozenset({"fully_provenanced", "replaced"})),
    Case("A16b", "A16", "Cache hit whose stored source is shadow-derived", True,
         Scenario(FACTUAL_Q, BELL, AGREE3,
                  cache=cache_doc("Antonio Meucci invented the telephone.", "self_consistency_factual", "system", 2)),
         "ground_truth_pipeline.py cache hit -> FULLY_PROVENANCED regardless of source",
         frozenset({"replaced", "fully_provenanced"})),
]


# ── A17: reasoning-step factual branch (function level) ──────────────────────

def a17_factual_step() -> dict:
    """
    engine/reasoning/step_verifier.py:_verify_factual tests gt.source against
    ("wikidata", "serper"), strings the pipeline never emits, so every GT result
    falls through to is_correct=True, confidence=0.0. Feed it a confident,
    contradicting GT result and record what comes back.
    """
    import engine.reasoning.step_verifier as sv
    import engine.verifier.ground_truth_pipeline as gtp
    from engine.reasoning.chain_decomposer import ReasoningStep
    from engine.reliability_contract import step_evidence_state
    from .harness import Patcher

    emitted_sources = ["Wikidata (Q11035)", "Wikidata confirmed: Wikidata (Q11035)",
                       "Google Search via Serper.dev", "Serper confirmed: Google Search via Serper.dev",
                       "GT Cache (serper)", "shadow_consensus", "self_consistency_factual", "none"]
    reachable = [s for s in emitted_sources if s in ("wikidata", "serper")]

    def fake_pipeline(**kwargs):
        return gtp.GroundTruthPipelineResult(
            verified_answer="Alexander Graham Bell", confidence=0.95, source="Wikidata (Q11035)",
        )

    p = Patcher()
    try:
        p.setattr(gtp, "run_ground_truth_pipeline", fake_pipeline)
        step = ReasoningStep(index=1, text="Thomas Edison invented the telephone.", step_type="FACTUAL_PREMISE")
        r = sv._verify_factual(step, FACTUAL_Q, [], [], use_external=True)
    finally:
        p.undo()
    return {
        "is_correct": r.is_correct, "confidence": r.confidence, "method": r.method,
        "branch_reachable_sources": reachable,
        "contract_state": step_evidence_state(r.is_correct, r.confidence, r.method).value,
    }


CASES.append(Case("A17", "A17", "Reasoning factual step with a confident contradicting GT result", True,
                  function=a17_factual_step,
                  source_ref="engine/reasoning/step_verifier.py:225 dead branch",
                  b0_expected=frozenset({"unchecked_reported_correct"})))


# ── observation ──────────────────────────────────────────────────────────────

def observe(case: Case, route_module: Optional[types.ModuleType] = None) -> dict:
    if case.function is not None:
        out = case.function()
        violations = []
        if out["is_correct"] and out["confidence"] == 0.0 and not out["branch_reachable_sources"]:
            violations.append("unchecked_reported_correct")
        if out["contract_state"] == "SUPPORTED":
            violations.append("supported")
        out["violations"] = violations
        return out

    sc = case.scenario
    outcome = run_monitor(sc, route_module)
    r = outcome.response
    rel = r.reliability
    gt = r.ground_truth
    delivered = delivered_by_sdk(r.model_dump(), sc.primary)
    shadow_answers = sum(1 for s in sc.shadows if s.success and s.text)
    obs = {
        "delivered":             delivered[:160],
        "replaced":              delivered != sc.primary,
        "fix_strategy":          r.fix_result.fix_strategy if r.fix_result else "",
        "fix_applied":           bool(r.fix_result and r.fix_result.fix_applied),
        "proposed_output":       (r.fix_result.proposed_output if r.fix_result else "")[:160],
        "requires_human_review": r.requires_human_review,
        "gt_source":             gt.source if gt else "not_run",
        "gt_escalated":          bool(gt and gt.requires_escalation),
        "provenance_label":      r.failure_signal_vector.provenance_label,
        "evidence_state":        rel.evidence_state if rel else None,
        "answer_action":         rel.answer_action if rel else None,
        "reasons":               rel.reasons if rel else [],
        "failure_summary":       r.failure_summary[:200],
        "high_failure_risk":     r.high_failure_risk,
        "classifier_probability": r.classifier_probability,
        "classifier_inputs":     {k: outcome.classifier_inputs.get(k) for k in
                                  ("fix_applied", "requires_escalation", "gt_override",
                                   "fix_strategy", "gt_source", "provenance_label")},
        "notifications":         outcome.notifications,
    }
    v = []
    if obs["replaced"]:
        v.append("replaced")
    if obs["provenance_label"] == "FULLY_PROVENANCED" and (case.unknown_input or obs["replaced"]):
        v.append("fully_provenanced")
    if case.unknown_input and obs["evidence_state"] == "SUPPORTED":
        v.append("supported")
    if obs["answer_action"] == "PASS_VERIFIED":
        v.append("pass_verified")
    if obs["gt_escalated"] and not obs["requires_human_review"]:
        v.append("escalation_cleared")
    if shadow_answers < 2 and "outputs are stable" in r.failure_summary.lower():
        v.append("claims_stable")
    obs["violations"] = v
    return obs
