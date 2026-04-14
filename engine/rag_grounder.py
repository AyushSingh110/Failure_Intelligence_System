from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Wikipedia retrieval result."""
    success:          bool
    grounded_answer:  str  = ""
    confidence:       float = 0.0
    source:           str  = ""
    error:            str  = ""


@dataclass
class VerificationResult:
    """Groq verification result."""
    matches:    bool
    confidence: float
    reason:     str


def ground_with_wikipedia(prompt: str, model_output: str) -> GroundingResult:
    """
    Wikipedia is used for RAG grounding of factual questions.
    """
    try:
        from engine.rag.retriever import fetch_wikipedia_summary
        from engine.rag.rag_pipeline import get_grounded_answer

        summary = fetch_wikipedia_summary(prompt)

        if not summary:
            return GroundingResult(
                success=False,
                error="No Wikipedia article found for this query.",
            )

        grounded_answer = get_grounded_answer(prompt, summary).strip()
        if grounded_answer and not _is_insufficient_answer(grounded_answer):
            return GroundingResult(
                success         = True,
                grounded_answer = grounded_answer[:1000],
                confidence      = 0.92,
                source          = "Wikipedia REST API + Groq RAG",
            )

        return GroundingResult(
            success         = True,
            grounded_answer = summary[:1000],  # limit context length
            confidence      = 0.72,
            source          = "Wikipedia REST API",
        )

    except Exception as exc:
        logger.warning("Wikipedia grounding failed: %s", exc)
        return GroundingResult(success=False, error=str(exc))


def compare_with_ground_truth(
    model_output:    str,
    grounded_answer: str,
) -> VerificationResult:
    """
    Groq is used for verification of the model output against the grounded answer.
    """
    if not grounded_answer or not model_output:
        return VerificationResult(
            matches    = True,
            confidence = 0.0,
            reason     = "Insufficient context for verification.",
        )

    numeric_check = _numeric_consistency_check(model_output, grounded_answer)
    if numeric_check is not None:
        return numeric_check

    # Try Groq verification first
    try:
        from engine.groq_service import get_groq_service

        groq = get_groq_service()
        if groq:
            verification_prompt = f"""You are a fact-checker. Compare these two texts and determine if they are factually consistent.

Model Output: {model_output[:300]}

Reference Context (from Wikipedia): {grounded_answer[:500]}

Answer with ONLY one of these:
- CONSISTENT: the model output agrees with or is not contradicted by the reference
- INCONSISTENT: the model output contradicts the reference
- UNCERTAIN: cannot determine

Then on the next line, give a confidence score 0.0-1.0.
Then on the next line, give a one-sentence reason.

Format:
VERDICT
CONFIDENCE
REASON"""

            response = groq.complete(
                verification_prompt,
                model_name  = "llama-3.1-8b-instant",
                max_tokens  = 100,
                temperature = 0.0,
            )

            if response.success:
                return _parse_groq_verification(response.output_text)

    except Exception as exc:
        logger.debug("Groq verification failed, using heuristic: %s", exc)

    # Fallback: simple keyword overlap heuristic
    return _heuristic_verification(model_output, grounded_answer)


def _parse_groq_verification(groq_output: str) -> VerificationResult:
    """Parse Groq verification response."""
    lines = [l.strip() for l in groq_output.strip().split("\n") if l.strip()]

    if not lines:
        return VerificationResult(matches=True, confidence=0.0, reason="Empty response")

    verdict_line = lines[0].upper()
    matches      = "INCONSISTENT" not in verdict_line

    # Parse confidence
    confidence = 0.5
    for line in lines[1:]:
        try:
            val = float(line.strip())
            if 0.0 <= val <= 1.0:
                confidence = val
                break
        except ValueError:
            continue

    # Parse reason
    reason = lines[-1] if len(lines) >= 3 else verdict_line

    return VerificationResult(
        matches    = matches,
        confidence = round(confidence, 3),
        reason     = reason[:200],
    )


def _heuristic_verification(
    model_output:    str,
    grounded_answer: str,
) -> VerificationResult:

    import re

    def tokenize(text: str) -> set[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'not', 'but', 'can', 'its', 'into', 'also', 'they', 'their', 'them', 'what', 'which'}
        return set(w for w in words if w not in stopwords)

    output_tokens    = tokenize(model_output)
    grounded_tokens  = tokenize(grounded_answer)

    if not output_tokens or not grounded_tokens:
        return VerificationResult(
            matches    = True,
            confidence = 0.0,
            reason     = "Insufficient tokens for comparison.",
        )

    overlap    = output_tokens & grounded_tokens
    union      = output_tokens | grounded_tokens
    jaccard    = len(overlap) / max(len(union), 1)

    # High overlap = likely consistent
    # Low overlap = could be inconsistent OR just different phrasing
    matches    = jaccard > 0.08   # low threshold — avoids false positives
    confidence = round(min(1.0 - jaccard, 0.6), 3)  # cap at 0.6 for heuristic

    return VerificationResult(
        matches    = matches,
        confidence = confidence,
        reason     = f"Heuristic: token overlap={jaccard:.3f} ({len(overlap)}/{len(union)} shared words)",
    )


def _is_insufficient_answer(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered for phrase in [
            "does not contain enough information",
            "don't have enough information",
            "do not have enough information",
            "context is insufficient",
            "cannot determine from the context",
        ]
    )


def _numeric_consistency_check(
    model_output: str,
    grounded_answer: str,
) -> VerificationResult | None:
    output_measure = _extract_measurement(model_output)
    grounded_measure = _extract_measurement(grounded_answer)

    if not output_measure or not grounded_measure:
        return None

    output_value, output_unit = output_measure
    grounded_value, grounded_unit = grounded_measure
    if output_unit != grounded_unit:
        return None

    diff = abs(output_value - grounded_value)
    relative_diff = diff / max(abs(grounded_value), 1e-6)

    if output_unit == "celsius":
        matches = diff <= 0.75
        reason = (
            f"Numeric check: compared Celsius values {output_value:g} vs {grounded_value:g}."
        )
    else:
        matches = relative_diff <= 0.02
        reason = (
            f"Numeric check: compared normalized {output_unit} values "
            f"{output_value:g} vs {grounded_value:g}."
        )

    return VerificationResult(
        matches=matches,
        confidence=0.92,
        reason=reason,
    )
def _extract_measurement(text: str) -> tuple[float, str] | None:
    normalized = text.lower().replace(",", "")
    number_match = re.search(r"(-?\d+(?:\.\d+)?)", normalized)
    if not number_match:
        return None
    value = float(number_match.group(1))

    if any(token in normalized for token in ["degrees celsius", "degree celsius", " celsius", "°c"]):
        return value, "celsius"

    if any(token in normalized for token in ["km/s", "kilometers per second", "kilometres per second"]):
        return value * 1000.0, "mps"

    if any(token in normalized for token in ["m/s", "meters per second", "metres per second"]):
        return value, "mps"
    return None
