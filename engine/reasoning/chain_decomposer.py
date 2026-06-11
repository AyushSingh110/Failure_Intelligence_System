from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Step dataclass ─────────────────────────────────────────────────────────────

STEP_TYPES = frozenset({
    "FACTUAL_PREMISE",
    "ARITHMETIC",
    "LOGICAL_INFERENCE",
    "DEFINITION",
    "CONCLUSION",
    "UNKNOWN",
})

@dataclass
class ReasoningStep:
    index:       int          # 1-based position in chain
    text:        str          # raw text of this step
    step_type:   str          # one of STEP_TYPES
    # Populated by step_verifier after verification
    is_verified: bool  = False
    is_correct:  bool  = True   # False means failure detected at this step
    confidence:  float = 0.0
    failure_note: str  = ""


# ── Groq extraction prompt ────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
You are a reasoning chain analyst. Given a question and an answer that contains reasoning steps, extract each atomic step.

Return ONLY a valid JSON array where each element has:
  "index"     : integer (1-based)
  "text"      : the exact text of this step (do not reword)
  "step_type" : one of FACTUAL_PREMISE | ARITHMETIC | LOGICAL_INFERENCE | DEFINITION | CONCLUSION | UNKNOWN

Rules:
- Each sentence or clause that makes a distinct claim is a separate step
- Arithmetic operations (addition, division, multiplication, etc.) → ARITHMETIC
- Stated facts about the world (names, dates, measurements, formulas) → FACTUAL_PREMISE
- "therefore", "so", "thus", "because", "since" conclusions → LOGICAL_INFERENCE
- The final answer sentence → CONCLUSION
- Variable/symbol definitions → DEFINITION
- If step spans multiple sentences, split them
- Return [] if there are no clear reasoning steps (e.g. one-word answers)
- Return a maximum of 12 steps

Question: {question}
Answer: {answer}

JSON array:"""


# ── Offline heuristic splitter ────────────────────────────────────────────────

# Sentence boundary: period/colon/semicolon followed by whitespace + capital OR digit.
# Including digits is critical for arithmetic step chains like "… = 1000. 1000 / 100 = 5."
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?;:])\s+(?=[A-Z0-9"\'(])')

# Type detection regexes (ordered by priority)
_ARITHMETIC_RE = re.compile(
    r'(?:'
    r'\d[\d,]*(?:\.\d+)?\s*[+\-×÷*/=<>≤≥]\s*\d'   # numeric op
    r'|(?:equals?|is|=)\s*\d[\d,]*(?:\.\d+)?'        # "equals 42"
    r'|\d[\d,]*\s*(?:percent|%|divided\s+by|times|plus|minus)'
    r')',
    re.IGNORECASE,
)

_FACTUAL_RE = re.compile(
    r'\b('
    r'(?:is|are|was|were)\s+(?:the|a|an)\s+\w+'        # "is the X"
    r'|(?:discovered|invented|founded|born|died)\s+in'  # historical facts
    r'|(?:equals?|approximately|roughly)\s+\d'          # defined constants
    r'|(?:speed|distance|mass|weight|height|temperature|rate)\s+(?:of|is)'
    r'|(?:by\s+definition|per\s+(?:the\s+)?(?:formula|equation|law))'
    r')',
    re.IGNORECASE,
)

_CONCLUSION_RE = re.compile(
    r'^\s*(?:'
    r'therefore|thus|so|hence|in\s+conclusion|as\s+a\s+result'
    r'|the\s+(?:answer|result|solution|final)\s+is'
    r'|(?:this\s+)?(?:gives|gives\s+us|yields|means|shows)'
    r')',
    re.IGNORECASE,
)

_LOGICAL_RE = re.compile(
    r'\b(?:because|since|given\s+that|if\s+.{0,40}\s+then|'
    r'from\s+(?:step|the\s+above)|as\s+(?:shown|stated|established)|'
    r'we\s+(?:know|have|can\s+conclude|can\s+see)|'
    r'substitut|combining|applying)\b',
    re.IGNORECASE,
)

_DEFINITION_RE = re.compile(
    r'^\s*(?:let|define|denote|suppose|assume|set)\b',
    re.IGNORECASE,
)


def _classify_step_heuristic(text: str) -> str:
    t = text.strip()
    if _DEFINITION_RE.match(t):
        return "DEFINITION"
    if _CONCLUSION_RE.match(t):
        return "CONCLUSION"
    if _ARITHMETIC_RE.search(t):
        return "ARITHMETIC"
    if _FACTUAL_RE.search(t):
        return "FACTUAL_PREMISE"
    if _LOGICAL_RE.search(t):
        return "LOGICAL_INFERENCE"
    return "UNKNOWN"


def _heuristic_decompose(answer: str) -> list[ReasoningStep]:
    """
    Offline fallback: split on sentence boundaries and classify each sentence.
    Returns at most 12 steps.
    """
    # Split numbered lists first (1. ... 2. ...) or bullet points
    numbered = re.split(r'(?m)^\s*\d+[.)]\s+', answer.strip())
    if len(numbered) >= 3:
        sentences = [s.strip() for s in numbered if s.strip()]
    else:
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(answer.strip()) if s.strip()]

    # Collapse very short fragments into the previous sentence
    merged: list[str] = []
    for s in sentences:
        if len(s) < 15 and merged:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)

    steps = []
    for i, text in enumerate(merged[:12], start=1):
        steps.append(ReasoningStep(
            index     = i,
            text      = text,
            step_type = _classify_step_heuristic(text),
        ))

    # Always mark last step as CONCLUSION if not already typed
    if steps and steps[-1].step_type == "UNKNOWN":
        steps[-1] = ReasoningStep(
            index     = steps[-1].index,
            text      = steps[-1].text,
            step_type = "CONCLUSION",
        )

    return steps


# ── Groq-assisted decomposition ───────────────────────────────────────────────

def _groq_decompose(question: str, answer: str) -> list[ReasoningStep] | None:
    """
    Uses groq_fast_model to extract typed reasoning steps.
    Returns None on any error so caller can fall back to heuristic.
    """
    try:
        from engine.groq_service import get_groq_service
        from config import get_settings
        groq = get_groq_service()
        if not groq:
            return None

        prompt = _DECOMPOSE_PROMPT.format(
            question=question.strip()[:400],
            answer=answer.strip()[:1200],
        )
        resp = groq.complete(
            prompt,
            model_name  = get_settings().groq_fast_model,
            max_tokens  = 600,
            temperature = 0.0,
        )
        if not resp.success or not resp.output_text:
            return None

        raw = resp.output_text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\s*```$', '', raw)

        data = json.loads(raw)
        if not isinstance(data, list):
            return None

        steps = []
        for item in data[:12]:
            if not isinstance(item, dict):
                continue
            text      = str(item.get("text", "")).strip()
            step_type = str(item.get("step_type", "UNKNOWN")).upper()
            if step_type not in STEP_TYPES:
                step_type = "UNKNOWN"
            idx = int(item.get("index", len(steps) + 1))
            if text:
                steps.append(ReasoningStep(index=idx, text=text, step_type=step_type))

        return steps if steps else None

    except (json.JSONDecodeError, Exception) as exc:
        logger.debug("Groq decomposition failed: %s", exc)
        return None


# ── Public entry point ─────────────────────────────────────────────────────────

def decompose_reasoning_chain(
    question: str,
    answer:   str,
    use_groq: bool = True,
) -> list[ReasoningStep]:
    """
    Split a reasoning output into typed atomic steps.

    Parameters
    ----------
    question : The original prompt (used to help the LLM understand context).
    answer   : The model's full reasoning output.
    use_groq : If True, attempts Groq-assisted decomposition first.
               Set False to force offline heuristic mode.

    Returns
    -------
    List of ReasoningStep objects (may be empty for trivial one-word answers).
    """
    answer = (answer or "").strip()
    if not answer or len(answer) < 10:
        return []

    if use_groq:
        groq_result = _groq_decompose(question, answer)
        if groq_result is not None:
            logger.debug(
                "chain_decomposer: groq extracted %d steps for answer[:%d]",
                len(groq_result), len(answer),
            )
            return groq_result
        logger.debug("chain_decomposer: groq unavailable or failed — using heuristic")

    steps = _heuristic_decompose(answer)
    logger.debug(
        "chain_decomposer: heuristic extracted %d steps for answer[:%d]",
        len(steps), len(answer),
    )
    return steps
