from __future__ import annotations
import ast
import logging
import math
import operator
import re
from dataclasses import dataclass
from typing import Optional

# Catches "√N = M" or "sqrt(N) = M" patterns (offline, no eval needed)
_SQRT_RE = re.compile(
    r'(?:√|sqrt\s*\(?\s*)(\d+(?:\.\d+)?)\s*\)?\s*[=≈≃]\s*([\-]?[\d\.]+)',
    re.IGNORECASE | re.UNICODE,
)

from engine.reasoning.chain_decomposer import ReasoningStep

logger = logging.getLogger(__name__)

STEP_FAIL_THRESHOLD = 0.55   # confidence above which a failure verdict is trusted


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StepVerificationResult:
    step_index:  int
    step_type:   str
    is_correct:  bool
    confidence:  float    # 0.0 = no signal, 1.0 = certain failure/pass
    method:      str      # "arithmetic_eval" | "not_checked" | "shadow_consistency" | "text_overlap" | "passthrough" | ...
    note:        str  = ""


# ── Sub-verifier A: ARITHMETIC ────────────────────────────────────────────────
# Completely offline — evaluates numeric expressions using a safe sandbox.
# Only operators in _SAFE_OPS are allowed; no function calls, no imports.

_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod:  operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# Extract "EXPR = NUM" patterns where EXPR contains at least one arithmetic operator.
# This prevents matching "speed = 120" (no operator in LHS).
# Handles chains like "120 / 2 = 50" or "0.15 × 80 = 12".
_EQ_RE = re.compile(
    r'([\d\.]+(?:\s*[+\-×÷*/]\s*[\d\.]+)+(?:\s*[+\-×÷*/]\s*[\d\.]+)*)'
    r'\s*[=≈≃]\s*([\-]?[\d\s\.\,]+)',
    re.UNICODE,
)

# Normalize unicode math symbols to ASCII
_MATH_NORM = str.maketrans({
    '×': '*', '÷': '/', '−': '-', '–': '-',
    '²': '**2', '³': '**3', '⁴': '**4',
    ',': '',   # remove thousands separators
})


def _safe_eval(expr: str) -> Optional[float]:
    """Evaluate a numeric expression string safely. Returns None on parse error."""
    try:
        expr_clean = expr.translate(_MATH_NORM).replace('^', '**').strip()
        tree = ast.parse(expr_clean, mode='eval')

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
                return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
                return _SAFE_OPS[type(node.op)](_eval(node.operand))
            raise ValueError(f"Unsafe node: {type(node)}")

        return _eval(tree)
    except Exception as exc:
        logger.warning(
            "degraded capability=step_arithmetic impact='this reasoning step is not arithmetically verified' "
            "reason=%s: %s", type(exc).__name__, exc,
        )
        return None


def _verify_arithmetic(step: ReasoningStep) -> StepVerificationResult:
    """
    Extracts all LHS=RHS pairs from the step text and evaluates them.
    If any equation is provably wrong (relative error > 1%), marks as failed.
    """
    text = step.text
    matches     = _EQ_RE.findall(text)
    sqrt_matches = _SQRT_RE.findall(text)

    if not matches and not sqrt_matches:
        return StepVerificationResult(
            step_index=step.index, step_type=step.step_type,
            is_correct=True, confidence=0.0,
            method="arithmetic_eval",
            note="No numeric equation found in this step — cannot verify arithmetically.",
        )

    errors_found: list[str] = []
    checks_done = 0

    # Check sqrt expressions ("√144 = 14" → sqrt(144)=12 ≠ 14)
    for radicand_str, result_str in sqrt_matches:
        radicand = _safe_eval(radicand_str.strip())
        result   = _safe_eval(result_str.strip())
        if radicand is not None and result is not None and radicand >= 0:
            expected = math.sqrt(radicand)
            checks_done += 1
            if abs(result) < 1e-10:
                ok = abs(expected) < 1e-10
            else:
                ok = abs(expected - result) / abs(result) < 0.01
            if not ok:
                errors_found.append(
                    f"√{radicand_str} = {expected:.4g} but stated as {result:.4g}"
                )

    for lhs_str, rhs_str in matches:
        lhs = _safe_eval(lhs_str.strip())
        rhs = _safe_eval(rhs_str.strip())
        if lhs is None or rhs is None:
            continue
        checks_done += 1
        if abs(rhs) < 1e-10:
            ok = abs(lhs) < 1e-10
        else:
            ok = abs(lhs - rhs) / abs(rhs) < 0.01   # 1% tolerance
        if not ok:
            errors_found.append(f"{lhs_str.strip()} = {lhs:.4g} but stated as {rhs:.4g}")

    if checks_done == 0:
        return StepVerificationResult(
            step_index=step.index, step_type=step.step_type,
            is_correct=True, confidence=0.0,
            method="arithmetic_eval",
            note="Could not parse numeric expressions — skipping arithmetic check.",
        )

    if errors_found:
        return StepVerificationResult(
            step_index=step.index, step_type=step.step_type,
            is_correct=False, confidence=0.92,
            method="arithmetic_eval",
            note=f"Arithmetic error(s) detected: {'; '.join(errors_found)}",
        )

    return StepVerificationResult(
        step_index=step.index, step_type=step.step_type,
        is_correct=True, confidence=0.90,
        method="arithmetic_eval",
        note=f"All {checks_done} arithmetic expression(s) verified correct.",
    )


# ── Sub-verifier B: FACTUAL_PREMISE ──────────────────────────────────────────
# Reuses the existing GT pipeline (Wikidata → Serper → self-consistency).
# Falls back to a hedge-detection heuristic when no external source is available.

_HEDGE_RE = re.compile(
    r'\b(approximately|roughly|around|about|estimated|believed\s+to\s+be|'
    r'some\s+sources\s+say|often\s+cited\s+as)\b',
    re.IGNORECASE,
)


def _verify_factual(
    step: ReasoningStep,
    question: str,
    shadow_outputs: list[str],
    shadow_weights: list[float],
    use_external: bool = True,
) -> StepVerificationResult:
    """
    Verifies a factual premise by routing through Wikidata/Serper.
    If external sources are unavailable, returns confidence=0 (no signal).
    """
    # If the premise contains strong hedge language, it's self-flagging
    if _HEDGE_RE.search(step.text):
        return StepVerificationResult(
            step_index=step.index, step_type=step.step_type,
            is_correct=True, confidence=0.0,
            method="hedge_detected",
            note="Step contains hedging language — premise uncertainty already acknowledged.",
        )

    # Skip external verification in offline/no-API mode
    if not use_external:
        return StepVerificationResult(
            step_index=step.index, step_type=step.step_type,
            is_correct=True, confidence=0.0,
            method="offline_skipped",
            note="External verification disabled — factual premise not checked (offline mode).",
        )

    # D-064 (OD-S0-4): no factual check is performed. This used to run the full
    # ground-truth pipeline (claim extraction, Wikidata, Serper calls) and then
    # discard the result: its guard tested gt.source against "wikidata"/"serper",
    # strings the pipeline never emits, so every call fell through to "no
    # signal". The call is removed rather than repaired, because repairing it
    # would switch on an unvalidated verifier. The step is reported NOT_CHECKED.
    return StepVerificationResult(
        step_index=step.index, step_type=step.step_type,
        is_correct=True, confidence=0.0,
        method="not_checked",
        note="Factual premise not checked: no validated factual verifier is enabled.",
    )


# ── Sub-verifier C: LOGICAL_INFERENCE ────────────────────────────────────────
# Checks whether the shadow ensemble agrees that this inference follows from
# the prior context.  Offline fallback: embedding/text-overlap consistency.

def _verify_logical(
    step:           ReasoningStep,
    prior_steps:    list[ReasoningStep],
    shadow_outputs: list[str],
) -> StepVerificationResult:
    """
    Asks shadow models: does this logical step follow from the prior context?
    Offline fallback: check embedding distance between step and prior steps.
    """
    # Build context string from prior steps
    prior_text = " ".join(s.text for s in prior_steps) if prior_steps else ""

    # ── Offline path: embedding distance ────────────────────────────────────
    try:
        from engine.encoder import get_encoder
        import numpy as np
        enc = get_encoder()
        if enc.available and prior_text:
            vecs = enc.encode_batch([prior_text[:600], step.text[:400]])
            similarity = float(np.clip(vecs[0] @ vecs[1], -1.0, 1.0))
            # Very low similarity between prior context and inference step
            # indicates a logical jump (missing connection)
            if similarity < 0.15:
                return StepVerificationResult(
                    step_index=step.index, step_type=step.step_type,
                    is_correct=False, confidence=0.65,
                    method="embedding_gap",
                    note=(
                        f"Logical gap detected: step is semantically distant from prior context "
                        f"(cosine similarity={similarity:.3f} < 0.15 threshold)."
                    ),
                )
            return StepVerificationResult(
                step_index=step.index, step_type=step.step_type,
                is_correct=True, confidence=similarity * 0.7,
                method="embedding_consistency",
                note=f"Logical step is semantically connected to prior context (similarity={similarity:.3f}).",
            )
    except Exception as exc:
        logger.debug("Logical step embedding check failed: %s", exc)

    # ── Text-overlap fallback ────────────────────────────────────────────────
    if prior_text:
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, prior_text[:600], step.text[:300]).ratio()
        return StepVerificationResult(
            step_index=step.index, step_type=step.step_type,
            is_correct=True, confidence=ratio * 0.4,
            method="text_overlap",
            note=f"Text overlap with prior context: {ratio:.3f} (heuristic).",
        )

    return StepVerificationResult(
        step_index=step.index, step_type=step.step_type,
        is_correct=True, confidence=0.0,
        method="passthrough",
        note="No prior steps available — logical inference not verified.",
    )


# ── Sub-verifier D: CONCLUSION ────────────────────────────────────────────────
# Checks that the conclusion is consistent with the most reliable prior step.

def _verify_conclusion(
    step:        ReasoningStep,
    prior_steps: list[ReasoningStep],
) -> StepVerificationResult:
    """
    If any prior step was marked incorrect, the conclusion is suspicious.
    Otherwise check semantic alignment with verified prior steps.
    """
    failed_priors = [s for s in prior_steps if not s.is_correct and s.confidence > STEP_FAIL_THRESHOLD]

    if failed_priors:
        return StepVerificationResult(
            step_index=step.index, step_type=step.step_type,
            is_correct=False,
            confidence=max(s.confidence for s in failed_priors) * 0.85,
            method="upstream_failure_propagation",
            note=(
                f"Conclusion is built on {len(failed_priors)} failed prior step(s): "
                + ", ".join(f"step {s.index}" for s in failed_priors)
            ),
        )

    return StepVerificationResult(
        step_index=step.index, step_type=step.step_type,
        is_correct=True, confidence=0.0,
        method="passthrough",
        note="No prior step failures — conclusion accepted as consistent.",
    )


# ── Public entry point ────────────────────────────────────────────────────────

def verify_steps(
    steps:          list[ReasoningStep],
    question:       str,
    shadow_outputs: list[str],
    shadow_weights: Optional[list[float]] = None,
    use_groq:       bool = True,
) -> list[StepVerificationResult]:
    """
    Run the appropriate sub-verifier for each step.

    Parameters
    ----------
    steps          : Output from decompose_reasoning_chain().
    question       : Original user prompt (needed for factual GT lookup).
    shadow_outputs : Groq shadow model outputs (used for consistency checks).
    shadow_weights : Confidence weights for shadow outputs.

    Returns
    -------
    List of StepVerificationResult, one per step.
    Results are written back into each step's is_correct / confidence fields.
    """
    weights = shadow_weights or []
    results: list[StepVerificationResult] = []

    for step in steps:
        prior = steps[:step.index - 1]   # steps before this one (0-indexed slice)

        if step.step_type == "ARITHMETIC":
            result = _verify_arithmetic(step)

        elif step.step_type == "FACTUAL_PREMISE":
            result = _verify_factual(step, question, shadow_outputs, weights, use_external=use_groq)

        elif step.step_type == "LOGICAL_INFERENCE":
            result = _verify_logical(step, prior, shadow_outputs)

        elif step.step_type == "CONCLUSION":
            # Also check arithmetic in conclusions — "Therefore X = Y/Z = N" often
            # contains the final calculation and is the most common error site.
            arith_result = _verify_arithmetic(step)
            if not arith_result.is_correct and arith_result.confidence > STEP_FAIL_THRESHOLD:
                result = arith_result
            else:
                result = _verify_conclusion(step, prior)

        else:  # DEFINITION, UNKNOWN — no meaningful verification
            result = StepVerificationResult(
                step_index=step.index, step_type=step.step_type,
                is_correct=True, confidence=0.0,
                method="passthrough",
                note=f"Step type '{step.step_type}' does not require verification.",
            )

        # Write verdict back into the step dataclass for upstream propagation
        step.is_verified = True
        step.is_correct  = result.is_correct or result.confidence < STEP_FAIL_THRESHOLD
        step.confidence  = result.confidence
        step.failure_note = result.note if not result.is_correct else ""
        # is_correct=True only means "no failure flagged". verification_status
        # says whether anything was actually checked (D-064).
        from engine.reliability_contract import EvidenceState, step_evidence_state
        if not result.is_correct and result.confidence < STEP_FAIL_THRESHOLD:
            step.verification_status = EvidenceState.INSUFFICIENT_EVIDENCE.value
        else:
            step.verification_status = step_evidence_state(
                result.is_correct, result.confidence, result.method,
            ).value

        results.append(result)
        logger.debug(
            "step_verifier: step=%d type=%s correct=%s conf=%.3f method=%s",
            step.index, step.step_type, step.is_correct, result.confidence, result.method,
        )

    return results
