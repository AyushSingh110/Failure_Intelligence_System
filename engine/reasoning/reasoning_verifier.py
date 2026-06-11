from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from engine.reasoning.chain_decomposer import decompose_reasoning_chain, ReasoningStep
from engine.reasoning.step_verifier    import verify_steps, StepVerificationResult, STEP_FAIL_THRESHOLD
from engine.reasoning.socratic_probe   import run_socratic_probe, SocraticProbeResult

logger = logging.getLogger(__name__)

# Confidence threshold for the socratic probe to count as a contradiction
_SOCRATIC_THRESHOLD = 0.50


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ReasoningVerificationResult:
    """
    Unified verdict from the reasoning verification pipeline.
    Consumers should check `failure_detected` first, then `failure_type`.
    """
    failure_detected:    bool
    failure_type:        str    # one of the failure classifications above
    confidence:          float  # 0.0–1.0

    # Step-level detail
    steps:               list[ReasoningStep]        = field(default_factory=list)
    step_results:        list[StepVerificationResult] = field(default_factory=list)
    first_failed_step:   Optional[int]  = None    # 1-based index, None if no failure
    total_steps:         int            = 0

    # Socratic probe detail
    socratic:            Optional[SocraticProbeResult] = None

    # Human-readable trace for XAI
    pipeline_trace:      list[str] = field(default_factory=list)

    # Convenience
    @property
    def failure_summary(self) -> str:
        if not self.failure_detected:
            return f"Reasoning chain verified ({self.total_steps} steps, no failure detected)."
        step_info = f" (step {self.first_failed_step})" if self.first_failed_step else ""
        return (
            f"Reasoning failure detected{step_info}: {self.failure_type} "
            f"(confidence={self.confidence:.0%})."
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def verify_reasoning(
    question:       str,
    primary_answer: str,
    shadow_outputs: list[str],
    shadow_weights: Optional[list[float]] = None,
    use_groq:       bool = True,
) -> ReasoningVerificationResult:
    """
    Run the full reasoning verification pipeline.

    Parameters
    ----------
    question       : The original user prompt.
    primary_answer : The model's full reasoning output being evaluated.
    shadow_outputs : Shadow ensemble outputs (used for consistency + Socratic probing).
    shadow_weights : Optional confidence weights for shadow outputs.
    use_groq       : If False, forces offline mode (heuristics only, no API calls).

    Returns
    -------
    ReasoningVerificationResult — single unified verdict.
    """
    trace: list[str] = []
    result = ReasoningVerificationResult(
        failure_detected  = False,
        failure_type      = "NO_FAILURE_DETECTED",
        confidence        = 0.0,
    )

    if not primary_answer or len(primary_answer.strip()) < 10:
        trace.append("reasoning_verifier: answer too short — skipping")
        result.pipeline_trace = trace
        return result

    # ── Phase 1: Decompose into steps ────────────────────────────────────────
    steps = decompose_reasoning_chain(question, primary_answer, use_groq=use_groq)
    result.steps       = steps
    result.total_steps = len(steps)

    trace.append(
        f"reasoning_verifier: decomposed into {len(steps)} steps "
        f"({'groq' if use_groq else 'heuristic'} mode)"
    )

    if not steps:
        trace.append("reasoning_verifier: no steps extracted — skipping step verification")
        result.pipeline_trace = trace
        return result

    # ── Phase 2: Step-level verification ─────────────────────────────────────
    step_results = verify_steps(steps, question, shadow_outputs, shadow_weights, use_groq=use_groq)
    result.step_results = step_results

    failed_steps = [
        r for r in step_results
        if not r.is_correct and r.confidence > STEP_FAIL_THRESHOLD
    ]

    trace.append(
        f"reasoning_verifier: step verification complete — "
        f"{len(failed_steps)}/{len(step_results)} steps failed"
    )

    # Find first failure and classify it
    if failed_steps:
        first_fail = min(failed_steps, key=lambda r: r.step_index)
        result.first_failed_step = first_fail.step_index
        result.confidence        = first_fail.confidence

        # Map step failure type to reasoning failure classification
        _type_map = {
            "ARITHMETIC":         "ARITHMETIC_ERROR",
            "FACTUAL_PREMISE":    "FACTUAL_GROUNDING_FAIL",
            "LOGICAL_INFERENCE":  "LOGICAL_GAP",
            "CONCLUSION":         "UPSTREAM_PROPAGATION",
        }
        result.failure_type     = _type_map.get(first_fail.step_type, "LOGICAL_GAP")
        result.failure_detected = True

        trace.append(
            f"reasoning_verifier: first failure at step {first_fail.step_index} "
            f"({first_fail.step_type}) — {first_fail.note[:100]}"
        )

        # Early-exit: high-confidence arithmetic errors don't need Socratic probing
        if result.failure_type == "ARITHMETIC_ERROR" and result.confidence >= 0.85:
            result.pipeline_trace = trace
            logger.info(
                "reasoning_verifier: arithmetic failure | step=%d | conf=%.3f",
                first_fail.step_index, result.confidence,
            )
            return result

    # ── Phase 3: Socratic probe ───────────────────────────────────────────────
    # Run even when no step failed — catches hidden assumption errors
    socratic = run_socratic_probe(
        question        = question,
        primary_answer  = primary_answer,
        shadow_outputs  = shadow_outputs,
        use_groq        = use_groq,
    )
    result.socratic = socratic

    trace.append(
        f"reasoning_verifier: socratic probe — "
        f"probes={socratic.probes_run} "
        f"contradiction={socratic.contradiction_found} "
        f"score={socratic.contradiction_score:.3f} "
        f"method={socratic.method}"
    )

    if socratic.contradiction_found and socratic.contradiction_score >= _SOCRATIC_THRESHOLD:
        if not result.failure_detected or socratic.contradiction_score > result.confidence:
            result.failure_detected = True
            result.failure_type     = "SOCRATIC_CONTRADICTION"
            result.confidence       = max(result.confidence, socratic.contradiction_score)
            trace.append(
                f"reasoning_verifier: socratic contradiction raised to primary failure "
                f"(score={socratic.contradiction_score:.3f})"
            )

    result.pipeline_trace = trace

    logger.info(
        "reasoning_verifier: failure=%s type=%s conf=%.3f first_step=%s",
        result.failure_detected, result.failure_type,
        result.confidence, result.first_failed_step,
    )

    return result
