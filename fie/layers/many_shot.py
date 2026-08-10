"""
Layer 3 — many-shot jailbreak (MSJ) detection.

Scores scripted Human/Assistant exchange chains using a power-law danger
curve over shot count, plus drift, entropy and assistant-compliance
signals. Weight 1.2.

Extracted from fie/adversarial.py. Detection logic is unchanged — see
tests/test_detection_golden.py, which pins the exact confidence values.
"""
from __future__ import annotations

import logging
import collections
import math
import re
import statistics

logger = logging.getLogger(__name__)


# Layer 3: Many-shot jailbreak detector

_MSJ_Q_ROLES  = r"human|user|h|q|person|benutzer|humain|utilisateur|humano|usuario|utente"
_MSJ_A_ROLES  = r"assistant|bot|ai|a|gpt|claude|llm|assistent|assistente|asistente"

_MANY_SHOT_EXCHANGE_RE = re.compile(
    rf"(?:{_MSJ_Q_ROLES})\s*:\s*.{{3,300}}?\n+\s*(?:{_MSJ_A_ROLES})\s*:",
    re.IGNORECASE | re.DOTALL,
)

_MANY_SHOT_Q_LINE_RE = re.compile(
    rf"(?:^|\n)\s*(?:{_MSJ_Q_ROLES})\s*[\:\>]\s*.{{5,300}}",
    re.IGNORECASE,
)

_MANY_SHOT_HARMFUL_RE = re.compile(
    r"\b(?:bomb|explos|weapon|hack|malware|ransomware|poison|synthesize|"
    r"drug|meth|fentanyl|cocaine|heroin|illegal|bypass|jailbreak|unrestricted|uncensored|"
    r"kill|murder|harm|attack|terror|extremi|suicide|self.harm|"
    r"child\s*(?:abuse|exploit|porn)|nsfw|rape|genocide|bioweapon|"
    r"phish|fraud|scam|stalker|doxx|exploit|exfiltrat|ransomware|"
    r"arson|traffick|launder|counterfeit|evade|persistence|backdoor|"
    r"manipulat|coercive|radicali|propaganda|violence|firearm|"
    r"untrace|undetect|lethal|overdose|casualt)\b",
    re.IGNORECASE,
)

# Fiction / hypothetical framing wrapper — common jailbreak amplifier
# Canonical fiction framing regex lives in fie/fiction_harm.py.
# Imported here so the MSJ layer can use it as an amplifier without duplication.
from fie.fiction_harm import FICTION_FRAMING_RE as _FICTION_FRAMING_RE

# A-turn extraction — greedily captures assistant response up to next Q-turn or end
_MANY_SHOT_A_EXTRACT_RE = re.compile(
    rf"(?:{_MSJ_A_ROLES})\s*[\:\>]\s*(.{{5,600}}?)(?=\n\s*(?:{_MSJ_Q_ROLES})\s*[\:\>]|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Compliance markers in A-turns
_A_TURN_REFUSAL_RE = re.compile(
    r"\b(?:i\s+can'?t|i\s+cannot|i\s+won'?t|i'?m\s+unable|i\s+must\s+decline|"
    r"i'?m\s+not\s+able|that'?s\s+not\s+something|i\s+shouldn'?t|i\s+apologize)\b",
    re.IGNORECASE,
)
_A_TURN_STEP_RE = re.compile(
    r"\b(?:step\s+\d|first[,\s]|secondly|you\s+will\s+need|here'?s\s+how|"
    r"to\s+begin|start\s+by|next[,\s]|finally[,\s])\b",
    re.IGNORECASE,
)


# ── MSJ helper functions ──────────────────────────────────────────────────────

def _power_law_danger(n_shots: int) -> float:
    """
    Danger score calibrated to paper Fig 2 power law.
    Returns 0.0–1.0. At n=32 → ~0.83, n=128 → ~0.92.
    Constants (0.95, 0.5) fitted to Claude 2.0 ASR curve; treat as starting values.
    """
    if n_shots <= 0:
        return 0.0
    return round(min(max(1.0 - 0.95 * (n_shots ** -0.5), 0.0), 1.0), 4)


def _compute_drift_score(q_turns: list[str]) -> float:
    """Harm density in late turns minus early turns. Positive = escalating."""
    n = len(q_turns)
    if n < 6:
        return 0.0
    third = n // 3
    early = q_turns[:third]
    late  = q_turns[-third:]
    def _density(turns: list[str]) -> float:
        return sum(1 for t in turns if _MANY_SHOT_HARMFUL_RE.search(t)) / max(len(turns), 1)
    return round(max(_density(late) - _density(early), 0.0), 4)


def _compute_q_entropy(q_turns: list[str]) -> float:
    """
    Trigram entropy over all Q turns.
    Low entropy (<2.5) = repetitive/templated → MSJ structural fingerprint.
    Returns 10.0 (safe default) when not enough tokens.
    """
    tokens = " ".join(q_turns).lower().split()
    if len(tokens) < 10:
        return 10.0
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    counts   = collections.Counter(trigrams)
    total    = sum(counts.values())
    entropy  = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return round(entropy, 4)


def _a_turn_compliance(a_turns: list[str]) -> float:
    """
    Score how consistently compliant the fabricated assistant is.
    High score = detailed, step-by-step, no refusals = strong MSJ signal.
    Returns 0.0–1.0.
    """
    if not a_turns:
        return 0.0
    avg_len       = sum(len(t.split()) for t in a_turns) / len(a_turns)
    step_ratio    = sum(1 for t in a_turns if _A_TURN_STEP_RE.search(t))    / len(a_turns)
    refusal_ratio = sum(1 for t in a_turns if _A_TURN_REFUSAL_RE.search(t)) / len(a_turns)
    length_score  = min(avg_len / 80.0, 1.0)
    return round((length_score * 0.4 + step_ratio * 0.4) * (1.0 - refusal_ratio), 4)


def _run_many_shot_detection(prompt: str) -> tuple[str | None, float, dict]:
    if len(prompt) < 150:
        return None, 0.0, {}

    pair_count = len(_MANY_SHOT_EXCHANGE_RE.findall(prompt))
    if pair_count < 3:
        q_lines = _MANY_SHOT_Q_LINE_RE.findall(prompt)
        if len(q_lines) < 4:
            return None, 0.0, {}
        pair_count = max(pair_count, len(q_lines) // 2)

    # Extract Q-turn and A-turn text for downstream analysis
    q_texts = re.findall(
        rf"(?:{_MSJ_Q_ROLES})\s*[\:\>]\s*(.{{5,300}}?)(?:\n|$)",
        prompt, re.IGNORECASE,
    )
    a_texts = _MANY_SHOT_A_EXTRACT_RE.findall(prompt)

    total_q       = len(q_texts) or 1
    harmful_q     = sum(1 for q in q_texts if _MANY_SHOT_HARMFUL_RE.search(q))
    harmful_ratio = harmful_q / total_q

    last_q   = q_texts[-1] if q_texts else ""
    early_qs = q_texts[:-1]
    escalation = (
        bool(_MANY_SHOT_HARMFUL_RE.search(last_q)) and
        sum(1 for q in early_qs if _MANY_SHOT_HARMFUL_RE.search(q)) < max(len(early_qs) // 2, 1)
    )

    # New signals
    fiction_flag = bool(_FICTION_FRAMING_RE.search(prompt[:1000]))
    drift_score  = _compute_drift_score(q_texts)
    q_entropy    = _compute_q_entropy(q_texts)
    low_entropy  = q_entropy < 2.5 and len(q_texts) >= 6
    compliance   = _a_turn_compliance(a_texts)
    danger       = _power_law_danger(pair_count)

    has_harmful = harmful_ratio > 0.0 or escalation or drift_score > 0.2

    signals:    list[str] = []
    confidence: float     = 0.0

    # ── Shot-count signal (power-law calibrated) ──────────────────────────────
    if pair_count >= 32:
        # Paper: ~50% ASR already at this range — treat as strong structural attack
        signals.append(f"pair_count={pair_count} power_law_danger={danger:.3f} (high-volume MSJ, ≥32 shots)")
        confidence = max(confidence, min(0.60 + danger * 0.35, 0.90))
    elif pair_count >= 8:
        base = min(0.55 + danger * 0.35, 0.82)
        if has_harmful:
            signals.append(f"pair_count={pair_count} power_law_danger={danger:.3f} (many-shot with harmful content)")
            confidence = max(confidence, base)
        elif len(prompt) > 5000:
            # Very long structured prompt even without keyword hits — volume signal
            signals.append(f"pair_count={pair_count} long_prompt={len(prompt)}chars (volume conditioning)")
            confidence = max(confidence, base - 0.10)
    elif pair_count >= 4 and has_harmful:
        base = min(0.50 + danger * 0.30, 0.72)
        signals.append(f"pair_count={pair_count} power_law_danger={danger:.3f} (few-shot with harmful content)")
        confidence = max(confidence, base)

    # ── Harmful keyword ratio in Q turns ─────────────────────────────────────
    if harmful_ratio >= 0.4:
        signals.append(f"harmful_topic_ratio={harmful_ratio:.2f} in Q turns (high density)")
        confidence = max(confidence, 0.84)
    elif harmful_ratio > 0.1:
        signals.append(f"harmful_topic_ratio={harmful_ratio:.2f} in Q turns")
        confidence = max(confidence, 0.68)

    # ── Escalation: benign early shots → harmful final query ──────────────────
    if escalation:
        signals.append("escalation=benign_early_shots_then_harmful_final_query")
        confidence = max(confidence, 0.78)

    # ── Behavioral drift: harm density increasing across thirds ───────────────
    if drift_score > 0.3:
        signals.append(f"behavioral_drift={drift_score:.3f} (harm density escalating toward end)")
        confidence = max(confidence, 0.74)
    elif drift_score > 0.15:
        signals.append(f"behavioral_drift={drift_score:.3f} (moderate harm escalation)")
        confidence = max(confidence, 0.62)

    # ── Structural entropy: low = repetitive/templated MSJ ────────────────────
    if low_entropy:
        signals.append(f"q_entropy={q_entropy:.3f} (repetitive template structure, <2.5)")
        confidence = max(confidence, 0.65)

    # ── A-turn compliance: fabricated assistant never refuses ─────────────────
    if compliance > 0.5:
        signals.append(f"a_turn_compliance={compliance:.3f} (assistant consistently answers, no refusals)")
        confidence = max(confidence, 0.72)
    elif compliance > 0.3:
        signals.append(f"a_turn_compliance={compliance:.3f}")
        confidence = max(confidence, 0.60)

    # ── Fiction/hypothetical framing amplifier ────────────────────────────────
    if fiction_flag and confidence > 0.40:
        signals.append("fiction_framing=detected (hypothetical/roleplay wrapper)")
        confidence = min(confidence + 0.10, 0.92)

    # ── Corroboration boost ───────────────────────────────────────────────────
    if len(signals) >= 3:
        confidence = min(confidence + 0.08, 0.92)
    elif len(signals) >= 2:
        confidence = min(confidence + 0.06, 0.92)

    if confidence < 0.50:
        return None, 0.0, {}

    return "MANY_SHOT_JAILBREAK", round(confidence, 4), {
        "pair_count":        pair_count,
        "power_law_danger":  round(danger, 3),
        "harmful_q_count":   harmful_q,
        "harmful_ratio":     round(harmful_ratio, 3),
        "escalation":        escalation,
        "behavioral_drift":  round(drift_score, 3),
        "q_entropy":         round(q_entropy, 3),
        "a_turn_compliance": round(compliance, 3),
        "fiction_framing":   fiction_flag,
        "signals_fired":     signals,
        "last_q_preview":    last_q[:150],
    }


