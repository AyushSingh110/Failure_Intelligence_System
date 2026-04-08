"""
engine/verifier/ground_truth_pipeline.py — Ground Truth Verification Orchestrator

This is the main entry point that ties together Steps 4–10.
Called from the /monitor endpoint after the DiagnosticJury produces a verdict.

Decision flow (in priority order)
-----------------------------------
1. Cache lookup (Step 7) — cheapest, highest confidence (human-verified)
   If hit → return cached answer immediately, skip all other steps

2. Claim extraction (Step 4) — extract {subject, property, value}
   If no clear claim found → skip Wikidata, move to Serper/escalation

3a. Temporal question? (from DomainCritic verdict)
      → Serper real-time search (Step 6)
        Found → return grounded answer
        Not found / Serper not configured → escalate (Step 10)

3b. Static factual question?
      → Wikidata structured lookup (Step 5)
        Consistent → confirm shadow consensus
        Inconsistent → override with Wikidata-sourced value
        Not found → escalate (Step 10) if consensus too weak, else return consensus

4. Cache any new verified answer with confidence >= 0.90 (Step 7 write-through)

5. If no verification succeeded with enough confidence → requires_escalation=True
   The fix engine will then call _apply_human_escalation() instead of auto-correcting

What this file returns
----------------------
GroundTruthPipelineResult:
  verified_answer    : the best available answer (may be same as model's)
  confidence         : how confident we are (0–1)
  source             : what verified it (cache / wikidata / serper / consensus)
  requires_escalation: True → caller should NOT auto-correct, queue for human review
  escalation_reason  : why escalation was triggered
  from_cache         : True if the answer came from the GT cache
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthPipelineResult:
    """
    Output from the ground truth verification pipeline.
    Used by the monitor endpoint to decide what to return and whether to escalate.
    """
    verified_answer:    str   = ""
    confidence:         float = 0.0
    source:             str   = "none"
    requires_escalation: bool = False
    escalation_reason:  str   = ""
    from_cache:         bool  = False
    # Summary of what the pipeline did — shown in XAI explanation
    pipeline_trace:     list[str] = field(default_factory=list)


def run_ground_truth_pipeline(
    prompt:           str,
    primary_output:   str,
    root_cause:       str,
    jury_confidence:  float,
    shadow_outputs:   Optional[list[str]] = None,
    shadow_weights:   Optional[list[float]] = None,
) -> GroundTruthPipelineResult:
    """
    Main entry point. Called from the /monitor route after jury verdict.

    Parameters
    ----------
    prompt          : original user question
    primary_output  : what the primary model answered (may be wrong)
    root_cause      : from DiagnosticJury.primary_verdict.root_cause
    jury_confidence : from DiagnosticJury.jury_confidence
    shadow_outputs  : texts from shadow models (already filtered for success)
    shadow_weights  : confidence weights from fan_out_with_confidence() Step 2

    Returns
    -------
    GroundTruthPipelineResult
    """
    result = GroundTruthPipelineResult()
    result.pipeline_trace = []

    # ── Stage 1: Cache lookup ──────────────────────────────────────────────
    try:
        from engine.ground_truth_cache import lookup_cache
        cache_hit = lookup_cache(prompt)
        if cache_hit:
            result.verified_answer = cache_hit.verified_answer
            result.confidence      = cache_hit.confidence
            result.source          = f"GT Cache ({cache_hit.source})"
            result.from_cache      = True
            result.pipeline_trace.append(
                f"Cache HIT — '{cache_hit.verified_answer[:60]}' "
                f"(verified by {cache_hit.verified_by}, used {cache_hit.use_count} times)"
            )
            logger.info("GT pipeline: CACHE HIT — returning verified answer")
            return result
        result.pipeline_trace.append("Cache MISS — no verified answer stored for this question")
    except Exception as exc:
        logger.warning("GT cache lookup failed: %s", exc)
        result.pipeline_trace.append(f"Cache lookup error: {exc}")

    # ── Stage 2: Claim extraction ──────────────────────────────────────────
    claim = None
    is_temporal = _is_temporal_root_cause(root_cause)

    if not is_temporal:
        try:
            from engine.claim_extractor import extract_claim
            claim = extract_claim(primary_output)
            if claim:
                result.pipeline_trace.append(
                    f"Claim extracted: {claim.subject} | {claim.property} | {claim.value}"
                )
            else:
                result.pipeline_trace.append("Claim extraction: no verifiable claim found")
        except Exception as exc:
            logger.warning("Claim extraction failed: %s", exc)
            result.pipeline_trace.append(f"Claim extraction error: {exc}")

    # ── Stage 3a: Temporal → Serper ───────────────────────────────────────
    if is_temporal:
        result.pipeline_trace.append("Temporal question detected → routing to Serper real-time search")
        try:
            from engine.verifier.serper_verifier import verify_with_serper
            serper = verify_with_serper(prompt, primary_output)

            if serper.skip:
                result.pipeline_trace.append(f"Serper skipped: {serper.error}")
                # No real-time source → escalate for temporal questions
                result.requires_escalation = True
                result.escalation_reason   = (
                    "This question requires real-time data but no live search provider "
                    "is configured. Add SERPER_API_KEY to .env to enable real-time verification."
                )
                result.pipeline_trace.append("Escalating: no real-time source available")
                return result

            if serper.found:
                if not serper.matches_output:
                    # Search result contradicts model output — use search answer
                    result.verified_answer = serper.grounded_answer
                    result.confidence      = serper.confidence
                    result.source          = serper.source
                    result.pipeline_trace.append(
                        f"Serper OVERRIDE — model output contradicted by search "
                        f"(confidence={serper.confidence:.2f}). Using search answer."
                    )
                    _cache_if_confident(prompt, result.verified_answer, "serper", serper.confidence)
                    return result
                else:
                    # Search result confirms model output — return with higher confidence
                    result.verified_answer = primary_output
                    result.confidence      = serper.confidence
                    result.source          = f"Serper confirmed: {serper.source}"
                    result.pipeline_trace.append(
                        f"Serper CONFIRMED model output (confidence={serper.confidence:.2f})"
                    )
                    return result
            else:
                result.pipeline_trace.append(f"Serper found no results: {serper.error}")
                result.requires_escalation = True
                result.escalation_reason   = "Real-time search returned no results for this query"
                return result

        except Exception as exc:
            logger.warning("Serper verification failed: %s", exc)
            result.pipeline_trace.append(f"Serper error: {exc}")
            result.requires_escalation = True
            result.escalation_reason   = f"Real-time verification failed: {exc}"
            return result

    # ── Stage 3b: Factual → Wikidata ──────────────────────────────────────
    if claim:
        try:
            from engine.verifier.wikidata_verifier import verify_claim_with_wikidata
            wiki = verify_claim_with_wikidata(
                subject       = claim.subject,
                property_name = claim.property,
                claimed_value = claim.value,
            )

            if wiki.found:
                result.pipeline_trace.append(
                    f"Wikidata found entity '{wiki.entity_label}': {wiki.entity_desc[:120]}"
                )

                if not wiki.matches_claim and wiki.confidence >= 0.70:
                    # Wikidata says the claim is wrong — override
                    correct = wiki.wikidata_value or wiki.entity_desc
                    result.verified_answer = correct
                    result.confidence      = wiki.confidence
                    result.source          = wiki.source
                    result.pipeline_trace.append(
                        f"Wikidata OVERRIDE — claim contradicted "
                        f"(confidence={wiki.confidence:.2f}). "
                        f"Correct value: {correct[:80]}"
                    )
                    _cache_if_confident(prompt, result.verified_answer, "wikidata", wiki.confidence)
                    return result

                elif wiki.matches_claim and wiki.confidence >= 0.60:
                    # Wikidata confirms — boost consensus
                    result.verified_answer = primary_output
                    result.confidence      = wiki.confidence
                    result.source          = f"Wikidata confirmed: {wiki.source}"
                    result.pipeline_trace.append(
                        f"Wikidata CONFIRMED model output (confidence={wiki.confidence:.2f})"
                    )
                    return result

                else:
                    result.pipeline_trace.append(
                        f"Wikidata result inconclusive (confidence={wiki.confidence:.2f})"
                    )
            else:
                result.pipeline_trace.append(f"Wikidata: {wiki.error}")

        except Exception as exc:
            logger.warning("Wikidata verification failed: %s", exc)
            result.pipeline_trace.append(f"Wikidata error: {exc}")

    # ── Stage 4: All verification sources exhausted ────────────────────────
    # Decide based on consensus_strength whether to auto-correct or escalate
    consensus_strength = _compute_consensus_strength(shadow_outputs, shadow_weights)
    result.pipeline_trace.append(
        f"All external sources exhausted. Shadow consensus_strength={consensus_strength:.2f}"
    )

    if consensus_strength >= 0.60:
        # Good shadow consensus — proceed with weighted consensus, no escalation
        result.pipeline_trace.append(
            "Proceeding with shadow consensus (strength sufficient)"
        )
        # Return empty verified_answer → caller uses shadow consensus via fix_engine
        result.confidence = consensus_strength
        result.source     = "shadow_consensus"
        return result
    else:
        # Weak consensus + no external verification = escalate
        result.requires_escalation = True
        result.escalation_reason   = (
            f"External ground truth verification found no reliable source "
            f"and shadow model consensus is too weak "
            f"(strength={consensus_strength:.2f} < 0.60). "
            "Auto-correction would be unreliable."
        )
        result.pipeline_trace.append(f"Escalating: {result.escalation_reason}")
        logger.info(
            "GT pipeline: ESCALATING | reason=%s", result.escalation_reason[:100]
        )
        return result


# ── Helpers ─────────────────────────────────────────────────────────────────

def _is_temporal_root_cause(root_cause: str) -> bool:
    return root_cause in (
        "TEMPORAL_KNOWLEDGE_CUTOFF",
        "KNOWLEDGE_BOUNDARY_FAILURE",
    )


def _compute_consensus_strength(
    shadow_outputs: Optional[list[str]],
    shadow_weights: Optional[list[float]],
) -> float:
    """
    Returns the normalized weighted strength of the shadow model consensus.
    0.0 = complete disagreement, 1.0 = unanimous weighted agreement.
    """
    if not shadow_outputs:
        return 0.0

    weights = shadow_weights if shadow_weights and len(shadow_weights) == len(shadow_outputs) \
              else [2.0] * len(shadow_outputs)
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    group_weights: dict[str, float] = {}
    for output, weight in zip(shadow_outputs, weights):
        key = output.strip().lower().rstrip(".,!?")[:200]
        group_weights[key] = group_weights.get(key, 0.0) + weight

    best_weight = max(group_weights.values())
    return best_weight / total_weight


def _cache_if_confident(
    question: str,
    answer:   str,
    source:   str,
    confidence: float,
) -> None:
    """Write-through: cache externally verified answers >= 0.90 confidence."""
    if confidence < 0.90:
        return
    try:
        from engine.ground_truth_cache import save_to_cache
        save_to_cache(
            question        = question,
            verified_answer = answer,
            source          = source,
            confidence      = confidence,
            verified_by     = "system",
        )
    except Exception as exc:
        logger.debug("Write-through cache failed: %s", exc)
