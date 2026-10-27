"""
Output-reliability decision contract — Workstream B, stage S0 (D-060).

WHAT THIS MODULE DECIDES
------------------------
Given what the monitor's verification and fix machinery produced, it decides
three things and nothing else:

  1. the evidence state of the primary answer      (EvidenceState)
  2. the action taken on the answer                (AnswerAction)
  3. the provenance label the response may carry

The primary answer is always the answer delivered. While WARN_ONLY is True no
path in this module produces a replacement: an engine correction is kept as a
*proposal* (FixResult.proposed_output) and the original answer is returned.

RULES (audit 2026-10-27 §4, E-R1 Part A)
-----------------------------------------
* Missing, malformed, uncertain or unavailable verification is never support.
* Shadow-model agreement and self-reported confidence are never evidence.
* SUPPORTED requires an explicit, well-formed verdict *and* a retained evidence
  passage *and* a source identity. FULLY_PROVENANCED requires the same.
* PASS_VERIFIED additionally requires a verifier named in VALIDATED_VERIFIERS.
  That set is empty until E-R1 Part B passes and the owner approves a threshold,
  so PASS_VERIFIED is unreachable in this stage.
* A fallback never clears a human-review escalation.

This is separate from Workstream A's malicious-input decisions
(`fie/policy.py`: ALLOW / REVIEW / BLOCK). Nothing here reads or changes them.

Pure: no network, no model, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EvidenceState(str, Enum):
    SUPPORTED             = "SUPPORTED"               # explicit verdict + evidence + source id (unvalidated judge)
    CONTRADICTED          = "CONTRADICTED"            # explicit contrary verdict; a candidate only
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"   # looked, found nothing usable, or verdict not usable
    CONFLICTING_EVIDENCE  = "CONFLICTING_EVIDENCE"    # usable sources disagree
    VERIFIER_UNAVAILABLE  = "VERIFIER_UNAVAILABLE"    # provider off, failed or timed out
    NOT_CHECKABLE         = "NOT_CHECKABLE"           # no external source applies (opinion, reasoning, code)
    NOT_CHECKED           = "NOT_CHECKED"             # verification was not attempted


class AnswerAction(str, Enum):
    PASS_VERIFIED  = "PASS_VERIFIED"    # requires a validated verifier; unreachable in S0
    PASS_QUALIFIED = "PASS_QUALIFIED"   # delivered without a verification claim
    FLAG           = "FLAG"             # delivered with a warning
    ESCALATE       = "ESCALATE"         # delivered, queued for human review
    # CORRECTED is deliberately absent: the monitor is warn-only (OD-R2).


WARN_ONLY = True

# Verifiers whose SUPPORTED precision has been measured and approved. Empty until
# E-R1 Part B passes and the owner sets the threshold (OD-R7).
VALIDATED_VERIFIERS: frozenset[str] = frozenset()

FULLY_PROVENANCED = "FULLY_PROVENANCED"
PARTIALLY_PROVENANCED = "PARTIALLY_PROVENANCED"
UNVERIFIED = "UNVERIFIED_MODEL_INFERENCE"
_TOOL_LABELS = ("REQUIRES_TOOL_VERIFICATION", "NULL_REQUIRED_BUT_MISSING")
_LIVE_CATEGORIES = ("LIVE_WORLD_STATE", "USER_SPECIFIC_STATE", "MIXED_SYNTHESIS")
_EXTERNAL_VERIFIERS = ("serper", "wikidata")
_MIN_ENSEMBLE = 2   # engine/detector/consistency.py needs two shadows to call an outlier


# ── Strict verdict parser (E-R1 arm B1) ──────────────────────────────────────

_VERDICTS = {
    "CONSISTENT":   EvidenceState.SUPPORTED,
    "INCONSISTENT": EvidenceState.CONTRADICTED,
    "UNCERTAIN":    EvidenceState.INSUFFICIENT_EVIDENCE,
}


@dataclass(frozen=True)
class VerdictParse:
    state:      EvidenceState
    confidence: Optional[float]
    value:      str
    reason:     str


def parse_verdict_strict(text: Optional[str], value_key: str = "GROUNDED_ANSWER") -> VerdictParse:
    """
    Tri-state reading of the 3-line VERDICT / CONFIDENCE / <value_key> reply that
    the Serper and Wikidata judges are asked for.

    Only one well-formed verdict line with one of the three allowed words and a
    numeric confidence inside [0, 1] counts. Everything else — empty text, no
    verdict line, two verdict lines, an echoed template, a missing, non-numeric
    or out-of-range confidence — is INSUFFICIENT_EVIDENCE. The production parser
    (`_parse_groq_verdict`) reads all of those as "consistent".
    """
    if not text or not text.strip():
        return VerdictParse(EvidenceState.INSUFFICIENT_EVIDENCE, None, "", "empty_verdict")

    verdicts: list[str] = []
    confidences: list[str] = []
    value = ""
    key = value_key.upper() + ":"
    for line in text.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            verdicts.append(line.split(":", 1)[1].strip().upper())
        elif upper.startswith("CONFIDENCE:"):
            confidences.append(line.split(":", 1)[1].strip())
        elif upper.startswith(key):
            value = line.split(":", 1)[1].strip()

    if len(verdicts) != 1 or verdicts[0] not in _VERDICTS:
        reason = "missing_verdict" if not verdicts else "malformed_verdict"
        return VerdictParse(EvidenceState.INSUFFICIENT_EVIDENCE, None, value, reason)

    state = _VERDICTS[verdicts[0]]
    if state is EvidenceState.INSUFFICIENT_EVIDENCE:
        return VerdictParse(state, None, value, "verdict_uncertain")

    if len(confidences) != 1:
        return VerdictParse(EvidenceState.INSUFFICIENT_EVIDENCE, None, value, "missing_confidence")
    try:
        confidence = float(confidences[0])
    except ValueError:
        return VerdictParse(EvidenceState.INSUFFICIENT_EVIDENCE, None, value, "malformed_confidence")
    if not 0.0 <= confidence <= 1.0:
        return VerdictParse(EvidenceState.INSUFFICIENT_EVIDENCE, None, value, "confidence_out_of_range")

    return VerdictParse(state, confidence, value, "explicit_verdict")


def combine_states(states: list[EvidenceState]) -> EvidenceState:
    """
    Combine verdicts from several sources on one claim. A supporting and a
    contradicting source together are CONFLICTING_EVIDENCE, never a majority.
    The current pipeline consults one source per request, so this is reached
    only by future multi-source checks; it is defined now so the rule is fixed.
    """
    present = set(states)
    if EvidenceState.SUPPORTED in present and EvidenceState.CONTRADICTED in present:
        return EvidenceState.CONFLICTING_EVIDENCE
    for s in (EvidenceState.CONFLICTING_EVIDENCE, EvidenceState.CONTRADICTED, EvidenceState.SUPPORTED):
        if s in present:
            return s
    for s in (EvidenceState.INSUFFICIENT_EVIDENCE, EvidenceState.VERIFIER_UNAVAILABLE,
              EvidenceState.NOT_CHECKABLE):
        if s in present:
            return s
    return EvidenceState.NOT_CHECKED


# ── Ground-truth result → evidence state ─────────────────────────────────────

@dataclass
class EvidenceAssessment:
    state:             EvidenceState
    verifier:          str = ""
    reasons:           list[str] = field(default_factory=list)
    evidence_source:   str = ""
    evidence_preview:  str = ""
    proposal_grounded: Optional[bool] = None   # is a proposed replacement found in the evidence?


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".,!?")


def assess_ground_truth(gt: Any, primary_output: str, time_sensitive: bool = False) -> EvidenceAssessment:
    """
    Read a GroundTruthPipelineResult (engine/verifier/ground_truth_pipeline.py).
    `gt` is None when the pipeline was not run. `time_sensitive` is the caller's
    judgement that the question asks about current state.
    """
    if gt is None:
        return EvidenceAssessment(EvidenceState.NOT_CHECKED, reasons=["verification_not_attempted"])

    verifier = getattr(gt, "verifier", "") or ""
    raw = getattr(gt, "evidence_state", "") or ""
    evidence = getattr(gt, "evidence_text", "") or ""
    source = getattr(gt, "evidence_source", "") or ""
    reasons: list[str] = []

    if verifier == "cache":
        reasons.append("cache_entry_has_no_retained_evidence")
        source_class = getattr(gt, "cache_source_class", "") or "unknown"
        reasons.append(f"cache_source_class={source_class}")
        if time_sensitive or getattr(gt, "provenance_category", "") in _LIVE_CATEGORIES:
            # No as-of policy is approved, so no cached answer is fresh enough.
            reasons.append("cache_stale_for_live_question")
        return EvidenceAssessment(EvidenceState.INSUFFICIENT_EVIDENCE, verifier, reasons, source)

    if verifier in ("self_consistency", "none_applicable"):
        reasons.append("no_external_source_for_question_type")
        if verifier == "self_consistency":
            reasons.append("shadow_agreement_is_not_evidence")
        return EvidenceAssessment(EvidenceState.NOT_CHECKABLE, verifier, reasons)

    try:
        state = EvidenceState(raw) if raw else EvidenceState.NOT_CHECKED
    except ValueError:
        state = EvidenceState.INSUFFICIENT_EVIDENCE
        reasons.append("unknown_evidence_state")

    if state is EvidenceState.NOT_CHECKED and verifier == "":
        reasons.append("no_verifier_reached")
        state = EvidenceState.INSUFFICIENT_EVIDENCE

    if state is EvidenceState.SUPPORTED and (not evidence.strip() or not source.strip()):
        reasons.append("evidence_or_source_identity_missing")
        state = EvidenceState.INSUFFICIENT_EVIDENCE

    proposal_grounded = None
    proposal = getattr(gt, "verified_answer", "") or ""
    if proposal and _norm(proposal) != _norm(primary_output):
        proposal_grounded = bool(evidence) and _norm(proposal) in _norm(evidence)
        if not proposal_grounded:
            reasons.append("proposed_value_not_in_evidence")

    if getattr(gt, "source", "") == "shadow_consensus":
        reasons.append("shadow_agreement_is_not_evidence")

    return EvidenceAssessment(state, verifier, reasons, source, evidence[:200], proposal_grounded)


# ── Action and label ─────────────────────────────────────────────────────────

def decide_action(
    state:             EvidenceState,
    verifier:          str,
    requires_review:   bool,
    high_failure_risk: bool,
) -> AnswerAction:
    if requires_review:
        return AnswerAction.ESCALATE
    if state is EvidenceState.SUPPORTED and verifier in VALIDATED_VERIFIERS:
        return AnswerAction.PASS_VERIFIED
    if state in (EvidenceState.CONTRADICTED, EvidenceState.CONFLICTING_EVIDENCE) or high_failure_risk:
        return AnswerAction.FLAG
    return AnswerAction.PASS_QUALIFIED


def provenance_label(
    assessment:     EvidenceAssessment,
    legacy_label:   str,
    category:       str,
    cache_source_class: str = "",
    time_sensitive: bool = False,
) -> str:
    """
    The label the response may carry. Never stronger than the legacy label, and
    FULLY_PROVENANCED only for SUPPORTED (which already implies evidence + source).
    """
    if assessment.state is EvidenceState.SUPPORTED:
        return FULLY_PROVENANCED if legacy_label == FULLY_PROVENANCED else legacy_label
    if legacy_label in _TOOL_LABELS:
        return legacy_label
    if (category in _LIVE_CATEGORIES or time_sensitive) and assessment.state is not EvidenceState.NOT_CHECKABLE:
        return "REQUIRES_TOOL_VERIFICATION"
    if assessment.verifier == "cache":
        if cache_source_class == "tenant_feedback":
            return PARTIALLY_PROVENANCED
        return PARTIALLY_PROVENANCED if assessment.evidence_source in _EXTERNAL_VERIFIERS else UNVERIFIED
    if (assessment.verifier in _EXTERNAL_VERIFIERS and assessment.evidence_preview
            and assessment.state is not EvidenceState.VERIFIER_UNAVAILABLE):
        # A passage was retrieved and judged, but support was not established.
        return PARTIALLY_PROVENANCED
    return UNVERIFIED


# ── Containment of a proposed fix ────────────────────────────────────────────

NOT_APPLIED_WARNING = (
    "FIE proposed a correction but did not apply it: corrections are unvalidated "
    "and the monitor is warn-only. The original answer is returned unchanged."
)


def contain_fix(proposed: Any, primary_output: str) -> tuple[Any, str, str]:
    """
    Return (fix_result_to_deliver, proposed_output, proposed_strategy).

    `proposed` is the app.schemas.FixResult the engine would have applied. When it
    would have replaced the answer, the delivered copy keeps every field but has
    fixed_output = the original, fix_applied = False, the candidate moved to
    proposed_output, and a warning.
    """
    if proposed is None:
        return None, "", ""
    candidate = proposed.fixed_output or ""
    if not proposed.fix_applied or not candidate or candidate == primary_output:
        return proposed, "", ""
    warning = NOT_APPLIED_WARNING + (f" {proposed.warning}" if proposed.warning else "")
    delivered = proposed.model_copy(update={
        "fixed_output":    primary_output,
        "fix_applied":     False,
        "proposed_output": candidate,
        "fix_explanation": f"[Not applied — warn-only] {proposed.fix_explanation}",
        "warning":         warning,
    })
    return delivered, candidate, proposed.fix_strategy


def ensemble_available(shadow_successes: int) -> bool:
    return shadow_successes >= _MIN_ENSEMBLE


# ── Reasoning steps (engine/reasoning/step_verifier.py) ──────────────────────

_STEP_NO_SIGNAL = frozenset({"no_external_source", "offline_skipped", "hedge_detected", "passthrough"})


def step_evidence_state(is_correct: bool, confidence: float, method: str) -> EvidenceState:
    """
    A StepVerificationResult reports is_correct=True with confidence 0.0 when
    nothing was checked. That is NOT_CHECKED, never support. Only a deterministic
    arithmetic evaluation can support a step; shadow agreement and text overlap
    can raise a flag but cannot vouch for one.
    """
    if confidence <= 0.0 or method in _STEP_NO_SIGNAL:
        return EvidenceState.NOT_CHECKED
    if not is_correct:
        return EvidenceState.CONTRADICTED
    if method == "arithmetic_eval":
        return EvidenceState.SUPPORTED
    return EvidenceState.INSUFFICIENT_EVIDENCE
