"""
engine.reasoning.socratic_probe
================================
Generates adversarial follow-up questions that challenge the primary
model's reasoning, then checks if shadow model responses contradict it.

Why this matters
----------------
Step-level verification catches arithmetic errors and factual premise
failures, but misses a subtler class: reasoning that is internally
consistent yet built on a hidden false assumption.

Example: "The function runs in O(n) time because it uses a single loop."
  → No arithmetic error, no factual claim → step_verifier passes it.
  → But if the loop body contains a nested call that is O(n) itself →
    Socratic probe: "Does the loop body perform any O(n) operations?"
  → Shadow models say "yes, the inner call is O(n)" → CONTRADICTION.

Two probe strategies
--------------------
CONTRAPOSITIVE  — "What would happen if [key premise] were false?"
EDGE_CASE       — "Does this reasoning still hold when [boundary condition]?"

Offline mode (Groq unavailable)
--------------------------------
Uses static probe templates derived from the question_type and step types.
These cover the most common reasoning failure patterns without any API call.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SocraticProbeResult:
    probes_run:        int
    contradiction_found: bool
    contradiction_score: float        # 0.0 – 1.0
    probe_questions:   list[str] = field(default_factory=list)
    shadow_answers:    list[str] = field(default_factory=list)
    contradiction_note: str      = ""
    method:            str       = "none"


# ── Static probe templates (offline) ─────────────────────────────────────────
# Keyed by (step_type_that_fired, question_keyword).
# {conclusion} and {premise} are filled from the answer text.

_STATIC_PROBES = [
    # Mathematical reasoning probes
    (r'\b(time\s+complexity|big.?o|o\(n|runs?\s+in)\b',
     "If the function contains a nested loop or recursive call, does the time complexity change from what was stated?"),

    (r'\b(sum|total|average|mean|median|mode)\b',
     "If one of the input values were 0 or negative, would the formula still produce the stated result?"),

    (r'\b(proof|prove|therefore|must\s+be|necessarily)\b',
     "Identify one scenario or counterexample where the stated conclusion would NOT hold."),

    # Causal reasoning probes
    (r'\b(because|cause|caused|leads?\s+to|results?\s+in|due\s+to)\b',
     "Could there be an alternative cause that produces the same effect without the stated cause being true?"),

    # Definitional probes
    (r'\b(by\s+definition|defined\s+as|is\s+a|is\s+an|refers?\s+to)\b',
     "Is there a well-known exception or edge case where this definition does not apply?"),

    # Temporal / ordering probes
    (r'\b(before|after|first|then|next|finally|sequence)\b',
     "If the order of steps were reversed, would the stated conclusion still follow?"),

    # Quantitative threshold probes
    (r'\b(greater\s+than|less\s+than|at\s+least|at\s+most|more\s+than|fewer\s+than)\b',
     "What happens to the conclusion when the value is exactly at the boundary condition stated?"),
]


def _static_probes(answer: str) -> list[str]:
    """Select up to 2 static probes that match patterns in the answer."""
    selected: list[str] = []
    for pattern, probe in _STATIC_PROBES:
        if re.search(pattern, answer, re.IGNORECASE):
            selected.append(probe)
            if len(selected) == 2:
                break
    # If no pattern matched, use two generic probes
    if not selected:
        selected = [
            "Identify the single weakest assumption in this reasoning and explain what would change if it were false.",
            "Does this reasoning hold for all possible inputs, or only for specific cases? Give a counterexample if one exists.",
        ]
    return selected[:2]


# ── Groq probe generation ─────────────────────────────────────────────────────

_PROBE_GEN_PROMPT = """\
You are a critical reasoning evaluator. Given a question and an answer that contains reasoning steps, generate exactly 2 short adversarial follow-up questions that would reveal if the reasoning is flawed.

Rules:
- Each question must be answerable without the original context (self-contained)
- Target the WEAKEST assumption or most likely error in the reasoning
- Do NOT ask the same thing twice
- Do NOT ask general knowledge questions — focus on THIS reasoning chain
- Format: return ONLY a JSON array of 2 strings, e.g. ["Q1", "Q2"]

Question: {question}
Answer (reasoning chain): {answer}

JSON array of 2 probe questions:"""


def _groq_generate_probes(question: str, answer: str) -> list[str] | None:
    try:
        import json
        from engine.groq_service import get_groq_service
        from config import get_settings
        groq = get_groq_service()
        if not groq:
            return None
        resp = groq.complete(
            _PROBE_GEN_PROMPT.format(
                question=question.strip()[:300],
                answer=answer.strip()[:800],
            ),
            model_name  = get_settings().groq_fast_model,
            max_tokens  = 200,
            temperature = 0.3,
        )
        if not resp.success or not resp.output_text:
            return None
        raw = resp.output_text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\s*```$', '', raw)
        probes = json.loads(raw)
        if isinstance(probes, list) and all(isinstance(p, str) for p in probes):
            return [p.strip() for p in probes[:2] if p.strip()]
        return None
    except Exception as exc:
        logger.debug("Groq probe generation failed: %s", exc)
        return None


# ── Contradiction detection ───────────────────────────────────────────────────

def _check_contradiction(
    original_answer: str,
    probe_question:  str,
    shadow_answer:   str,
) -> tuple[bool, float, str]:
    """
    Returns (is_contradicted, confidence, note).
    Uses embedding similarity if available, falls back to text overlap.
    """
    # Positive contradiction markers in shadow answer
    _CONTRADICT_MARKERS = re.compile(
        r'\b(no[,\s]|not\s+necessarily|this\s+is\s+(?:wrong|incorrect|false|not\s+right)|'
        r'actually[,\s]|in\s+fact[,\s]|that\s+(?:is|would\s+be)\s+(?:wrong|incorrect)|'
        r'counterexample|exception|does\s+not\s+hold|would\s+fail|is\s+not\s+always)\b',
        re.IGNORECASE,
    )
    contradiction_markers = len(_CONTRADICT_MARKERS.findall(shadow_answer))

    # Embedding-based semantic divergence check
    try:
        import numpy as np
        from engine.encoder import get_encoder
        enc = get_encoder()
        if enc.available:
            orig_claim = original_answer[:500]
            shadow_resp = shadow_answer[:500]
            vecs = enc.encode_batch([orig_claim, shadow_resp])
            similarity = float(np.clip(vecs[0] @ vecs[1], -1.0, 1.0))
            # Low similarity between original reasoning and shadow's probe answer
            # combined with contradiction markers → strong contradiction signal
            base_conf = max(0.0, (0.65 - similarity) / 0.65)
            marker_boost = min(contradiction_markers * 0.08, 0.25)
            confidence = min(base_conf + marker_boost, 1.0)

            is_contradicted = confidence > 0.45 and contradiction_markers >= 1
            note = (
                f"Embedding similarity={similarity:.3f} (low = divergent). "
                f"Contradiction markers found: {contradiction_markers}. "
                f"Shadow response: '{shadow_answer[:120]}'"
            )
            return is_contradicted, round(confidence, 3), note
    except Exception:
        pass

    # Fallback: text overlap
    ratio = SequenceMatcher(None, original_answer[:400], shadow_answer[:400]).ratio()
    base_conf = max(0.0, (0.5 - ratio) / 0.5)
    marker_boost = min(contradiction_markers * 0.1, 0.3)
    confidence = min(base_conf + marker_boost, 1.0)
    is_contradicted = confidence > 0.45 and contradiction_markers >= 1

    return (
        is_contradicted,
        round(confidence, 3),
        f"Text overlap={ratio:.3f}. Contradiction markers: {contradiction_markers}.",
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_socratic_probe(
    question:       str,
    primary_answer: str,
    shadow_outputs: list[str],
    use_groq:       bool = True,
) -> SocraticProbeResult:
    """
    Run the Socratic probe pipeline.

    1. Generate 2 adversarial challenge questions (Groq or static templates).
    2. Check if shadow model outputs already answer the probes contradictorily.
       (We use existing shadow outputs rather than making new API calls.)
    3. If shadow outputs are short/irrelevant, optionally make one new Groq call
       per probe to get a targeted answer.

    Parameters
    ----------
    question       : Original user prompt.
    primary_answer : Model's full reasoning output being evaluated.
    shadow_outputs : Existing shadow model outputs from the pipeline.
    use_groq       : If True, use Groq for probe generation (falls back to static).

    Returns
    -------
    SocraticProbeResult
    """
    if not primary_answer or len(primary_answer.strip()) < 20:
        return SocraticProbeResult(
            probes_run=0, contradiction_found=False,
            contradiction_score=0.0, method="skipped",
        )

    # Step 1: generate probe questions
    probes: list[str] = []
    method = "static_templates"
    if use_groq:
        groq_probes = _groq_generate_probes(question, primary_answer)
        if groq_probes:
            probes = groq_probes
            method = "groq_generated"

    if not probes:
        probes = _static_probes(primary_answer)

    if not probes:
        return SocraticProbeResult(
            probes_run=0, contradiction_found=False,
            contradiction_score=0.0, method="no_probes",
        )

    # Step 2: get shadow answers for each probe
    # Strategy: use existing shadow outputs first (cheapest — no extra API call)
    # If shadow outputs are long enough, they likely contain relevant reasoning.
    shadow_answers: list[str] = []
    all_contradictions: list[tuple[bool, float, str]] = []

    for probe in probes:
        # Use the most semantically relevant shadow output as the probe answer
        # (the one most likely to address the probe question)
        best_shadow = ""
        if shadow_outputs:
            # Pick the shadow output that is most different from primary
            # (most independent perspective)
            try:
                from engine.encoder import get_encoder
                enc = get_encoder()
                if enc.available and len(shadow_outputs) > 1:
                    vecs = enc.encode_batch([primary_answer[:400]] + [s[:400] for s in shadow_outputs])
                    sims = [float(vecs[0] @ vecs[i + 1]) for i in range(len(shadow_outputs))]
                    best_shadow = shadow_outputs[int(min(range(len(sims)), key=lambda k: sims[k]))]
                else:
                    best_shadow = shadow_outputs[-1]
            except Exception:
                best_shadow = shadow_outputs[0] if shadow_outputs else ""

        # If no usable shadow output, ask Groq directly about the probe
        if not best_shadow and use_groq:
            try:
                from engine.groq_service import get_groq_service
                from config import get_settings
                groq = get_groq_service()
                if groq:
                    resp = groq.complete(
                        f"Question about a reasoning chain:\n{probe}\n\nReasoning being evaluated:\n{primary_answer[:600]}\n\nAnswer concisely:",
                        model_name  = get_settings().groq_fast_model,
                        max_tokens  = 120,
                        temperature = 0.2,
                    )
                    if resp.success and resp.output_text:
                        best_shadow = resp.output_text.strip()
            except Exception as exc:
                logger.debug("Groq probe answer failed: %s", exc)

        shadow_answers.append(best_shadow)

        if best_shadow:
            contradicted, conf, note = _check_contradiction(
                primary_answer, probe, best_shadow,
            )
            all_contradictions.append((contradicted, conf, note))
        else:
            all_contradictions.append((False, 0.0, "No shadow answer available."))

    # Step 3: aggregate contradiction signal
    max_conf   = max((conf for _, conf, _ in all_contradictions), default=0.0)
    any_contra = any(c for c, _, _ in all_contradictions)
    best_note  = next(
        (note for contra, _, note in all_contradictions if contra),
        all_contradictions[0][2] if all_contradictions else "",
    )

    logger.info(
        "socratic_probe: probes=%d contradiction=%s score=%.3f method=%s",
        len(probes), any_contra, max_conf, method,
    )

    return SocraticProbeResult(
        probes_run          = len(probes),
        contradiction_found = any_contra,
        contradiction_score = round(max_conf, 3),
        probe_questions     = probes,
        shadow_answers      = shadow_answers,
        contradiction_note  = best_note,
        method              = method,
    )
