"""
Injection detection layers: 1 (regex), 2 (guard), 3 (FAISS), 4 (indirect).
"""
from __future__ import annotations

import re

from config import get_settings
from engine.agents.adversarial.normalization import normalize_for_detection
from engine.agents.adversarial.patterns import _AttackPattern, _ATTACK_PATTERNS
from engine.archetypes.registry import adversarial_registry, FAISSSearchResult
from engine.prompt_guard import score_prompt_attack

# ── Layer 4 helpers: indirect prompt injection ────────────────────────────────

# Signals that the prompt contains external content to process
_DOCUMENT_TRIGGER_RE = re.compile(
    r"(?:"
    r"(?:summarize|translate|analyze|review|process|read|evaluate|"
    r"proofread|edit|classify|extract|parse|convert)"
    r"\s*(?:the\s+following|this\s+(?:text|document|email|message|article)|:)"
    r"|(?:based\s+on|given|using|from|according\s+to)\s+the\s+following"
    r"|(?:document|email|message|text|content|article|passage|report)\s*:\s*"
    r"|```|<!--|<(?:document|doc|email|message|context|text|article|data)\s*>"
    r")",
    re.IGNORECASE,
)

# Injected instructions hidden inside document content
_INDIRECT_INJECTION_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|the\s+(?:user'?s?|original|human)\s+)(?:instructions?|request|query|question|task|prompt)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|request|task)"
    r"|(?:new|additional|updated|changed)\s+(?:instruction|directive|task|command)\s*[:;]"
    r"|your\s+(?:new|real|actual|true)\s+(?:task|purpose|goal|instructions?)\s+(?:is|are)"
    r"|forget\s+(?:the|your)?\s*(?:user|human|original)?\s*(?:task|question|request|instructions?)"
    r"|instead\s+of\s+(?:summarizing|translating|analyzing|reviewing|processing|answering)"
    r"|do\s+not\s+(?:summarize|translate|analyze|review|process|answer)\s+(?:this|the|it)"
    r"|(?:from\s+now|starting\s+now|as\s+of\s+now)\s*,?\s*(?:you\s+(?:must|should|will)|ignore)"
    r"|(?:override|system|admin)\s*:\s*(?:new|updated|changed|ignore)"
    r"|(?:actually|instead)\s*,?\s*(?:your\s+task|you\s+should|do\s+the\s+following)"
    r"|<\s*(?:new_)?(?:system|instructions?|task)\s*>.*?<\s*/\s*(?:system|instructions?|task)\s*>"
    r"|(?:note|message|instructions?|reminder|alert|attention|important)\s+(?:to|for)\s+(?:(?:the\s+)?(?:AI|LLM|language\s+model|assistant|model|GPT|Claude|bot))\s*[:\-]"
    r"|\[(?:note|actual\s+instructions?|message|instructions?)\s+(?:to|for)\s+(?:AI|assistant|model|LLM|Claude|GPT)\s*[:;\]]"
    r"|do\s+not\s+follow\s+(?:the\s+)?(?:user'?s?\s+|original\s+|previous\s+)?(?:instructions?|request|task|question)"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Signs the model's output shows it followed an injected instruction
_OUTPUT_COMPLIANCE_RE = re.compile(
    r"(?:"
    r"as\s+(?:instructed|requested|directed|told)\s+(?:in|by)\s+the\s+(?:document|text|email|message|file)"
    r"|the\s+(?:document|text|email|message)\s+(?:asks?\s+me|instructed?\s+me|told?\s+me|requests?\s+that\s+I)"
    r"|new\s+(?:instructions?|task|directive)\s+(?:received|detected|found|identified)"
    r"|switching\s+to\s+(?:new|different|updated)\s+(?:task|mode|instructions?)"
    r"|ignoring\s+(?:previous|original|prior)\s+(?:task|instructions?|request|question)"
    r"|my\s+(?:new|updated|changed|actual|real)\s+(?:task|purpose|goal|instructions?)\s+is"
    r"|instead\s+of\s+(?:summarizing|translating|analyzing),?\s+I\s+(?:will|am|have)"
    r")",
    re.IGNORECASE,
)


def _extract_document_portion(prompt: str) -> str | None:
    """
    Return the content portion of the prompt (after the task instruction)
    if the prompt contains a document/external-content pattern, else None.
    """
    m = _DOCUMENT_TRIGGER_RE.search(prompt)
    if not m:
        return None
    portion = prompt[m.end():].strip()
    return portion if len(portion) > 40 else None


def run_indirect_injection_detection(
    prompt: str,
    primary_output: str,
) -> tuple[str | None, float, dict]:
    """
    Layer 4: Indirect prompt injection detection.

    Scans the document/content portion of the prompt for embedded instructions
    and the model output for signs it followed them.
    """
    doc_portion = _extract_document_portion(prompt)
    if doc_portion is None:
        full_injection = _INDIRECT_INJECTION_RE.search(prompt)
        if not full_injection:
            return None, 0.0, {}
        output_fired = bool(_OUTPUT_COMPLIANCE_RE.search(primary_output or ""))
        conf = 0.72 if output_fired else 0.45
        return "INDIRECT_PROMPT_INJECTION", conf, {
            "document_found": False,
            "injection_in_prompt": full_injection.group(0)[:120],
            "output_compliance_detected": output_fired,
        }

    injection_match = _INDIRECT_INJECTION_RE.search(doc_portion)
    output_fired    = bool(_OUTPUT_COMPLIANCE_RE.search(primary_output or ""))

    if not injection_match and not output_fired:
        return None, 0.0, {}

    if injection_match and output_fired:
        confidence = 0.88
    elif injection_match:
        confidence = 0.65
    else:
        confidence = 0.52

    evidence = {
        "document_found": True,
        "document_snippet": doc_portion[:200],
        "injection_pattern_matched": injection_match.group(0)[:120] if injection_match else None,
        "output_compliance_detected": output_fired,
        "output_snippet": (primary_output or "")[:150] if output_fired else None,
    }
    return "INDIRECT_PROMPT_INJECTION", confidence, evidence


def run_pattern_detection(prompt: str) -> tuple[_AttackPattern | None, str]:
    """
    Layer 1: Regex pattern detection.

    Runs against both original and normalized (de-obfuscated) text so that
    spaced characters, leet-speak, and homoglyphs don't bypass detection.
    """
    priority_order = ["SMUGGLING", "INJECTION", "JAILBREAK", "OVERRIDE"]
    normalized = normalize_for_detection(prompt)
    hits: dict[str, tuple[_AttackPattern, str, bool]] = {}

    for ap in _ATTACK_PATTERNS:
        m = ap.pattern.search(prompt)
        if m:
            hits[ap.category] = (ap, m.group(0)[:100], False)
            continue
        m = ap.pattern.search(normalized)
        if m:
            hits[ap.category] = (ap, m.group(0)[:100], True)

    for cat in priority_order:
        if cat in hits:
            ap, matched_text, obfuscated = hits[cat]
            if obfuscated:
                ap = _AttackPattern(
                    category=ap.category,
                    root_cause=ap.root_cause,
                    base_confidence=max(ap.base_confidence - 0.06, 0.50),
                    pattern=ap.pattern,
                )
            return ap, matched_text
    return None, ""


def run_guard_detection(prompt: str) -> tuple[str | None, float, list[str]]:
    """
    Layer 2: Statistical prompt guard (ML-based).
    Falls back to normalized text if original scores below threshold.
    """
    signal = score_prompt_attack(prompt)
    if signal.root_cause is None or signal.score < 0.75:
        normalized = normalize_for_detection(prompt)
        if normalized != prompt:
            signal = score_prompt_attack(normalized)
    if signal.root_cause is None or signal.score < 0.75:
        return None, 0.0, []
    return signal.root_cause, signal.score, list(signal.evidence)


def run_faiss_detection(prompt: str) -> tuple[FAISSSearchResult | None, float]:
    """
    Layer 3: FAISS semantic search against known adversarial embeddings.
    Confidence is normalized over [threshold, 1.0] so 0.0 = at threshold, 1.0 = perfect match.
    """
    cfg = get_settings()
    try:
        results = adversarial_registry.search(prompt)
    except Exception:
        return None, 0.0

    if not results:
        return None, 0.0

    best = results[0]
    if not best.is_match:
        return None, 0.0

    threshold = cfg.jury_adversarial_faiss_threshold
    excess    = best.similarity - threshold
    span      = max(1.0 - threshold, 1e-6)
    faiss_conf = round(min(excess / span, 1.0), 4)
    return best, faiss_conf


# ── Backward-compat aliases (used by specialist.py) ──────────────────────────
_run_indirect_injection_detection = run_indirect_injection_detection
_run_pattern_detection            = run_pattern_detection
_run_guard_detection              = run_guard_detection
_run_faiss_detection              = run_faiss_detection
