from __future__ import annotations
import re


_MANY_SHOT_EXCHANGE_RE = re.compile(
    r"(?:human|user|h|q|person)\s*:\s*.{3,300}?\n+\s*(?:assistant|bot|ai|a|gpt|claude|llm)\s*:",
    re.IGNORECASE | re.DOTALL,
)
_MANY_SHOT_Q_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:human|user|h|q|person)\s*[\:\>]\s*.{5,300}",
    re.IGNORECASE,
)
_MANY_SHOT_HARMFUL_RE = re.compile(
    r"\b(?:bomb|explos|weapon|hack|malware|ransomware|poison|synthesize|"
    r"drug|meth|fentanyl|cocaine|illegal|bypass|jailbreak|unrestricted|uncensored|"
    r"kill|murder|harm|attack|terror|extremi|suicide|self.harm|"
    r"child\s*(?:abuse|exploit|porn)|nsfw|rape|genocide|bioweapon)\b",
    re.IGNORECASE,
)


def run_many_shot_detection(prompt: str) -> tuple[str | None, float, dict]:
    """
    Layer 3b: Detect many-shot / few-shot jailbreak attacks.
    Returns (root_cause | None, confidence, evidence).
    """
    if len(prompt) < 150:
        return None, 0.0, {}

    pair_count = len(_MANY_SHOT_EXCHANGE_RE.findall(prompt))
    if pair_count < 3:
        q_lines = _MANY_SHOT_Q_LINE_RE.findall(prompt)
        if len(q_lines) < 4:
            return None, 0.0, {}
        pair_count = max(pair_count, len(q_lines) // 2)

    q_texts = re.findall(
        r"(?:human|user|h|q|person)\s*[\:\>]\s*(.{5,300}?)(?:\n|$)",
        prompt, re.IGNORECASE,
    )

    total_q       = len(q_texts) or 1
    harmful_q     = sum(1 for q in q_texts if _MANY_SHOT_HARMFUL_RE.search(q))
    harmful_ratio = harmful_q / total_q

    last_q   = q_texts[-1] if q_texts else ""
    early_qs = q_texts[:-1]
    escalation = (
        bool(_MANY_SHOT_HARMFUL_RE.search(last_q)) and
        sum(1 for q in early_qs if _MANY_SHOT_HARMFUL_RE.search(q)) < max(len(early_qs) // 2, 1)
    )

    signals:    list[str] = []
    confidence: float     = 0.0

    if pair_count >= 8:
        signals.append(f"pair_count={pair_count} (many-shot conditioning ≥8 exchanges)")
        confidence = max(confidence, 0.80)
    elif pair_count >= 4:
        signals.append(f"pair_count={pair_count} (few-shot pattern 4-7 exchanges)")
        confidence = max(confidence, 0.62)

    if harmful_ratio >= 0.4:
        signals.append(f"harmful_topic_ratio={harmful_ratio:.2f}")
        confidence = max(confidence, 0.84)
    elif harmful_ratio > 0.1:
        signals.append(f"harmful_topic_ratio={harmful_ratio:.2f}")
        confidence = max(confidence, 0.68)

    if escalation:
        signals.append("escalation=gradual_buildup_to_harmful_final_turn")
        confidence = max(confidence, 0.78)

    if len(signals) >= 2:
        confidence = min(confidence + 0.06, 0.92)

    if confidence < 0.50:
        return None, 0.0, {}

    return "MANY_SHOT_JAILBREAK", round(confidence, 4), {
        "pair_count":      pair_count,
        "harmful_q_count": harmful_q,
        "harmful_ratio":   round(harmful_ratio, 3),
        "escalation":      escalation,
        "signals_fired":   signals,
        "last_q_preview":  last_q[:150],
    }


_run_many_shot_detection = run_many_shot_detection
