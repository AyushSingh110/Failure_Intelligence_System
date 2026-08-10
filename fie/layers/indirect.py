"""
Layer 4 — indirect prompt injection.

Detects instructions smuggled inside quoted documents or tool output,
where the attacker controls retrieved content rather than the prompt.
Weight 0.9.

Extracted from fie/adversarial.py. Detection logic is unchanged — see
tests/test_detection_golden.py, which pins the exact confidence values.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# Layer 4: Indirect prompt injection

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


def _run_indirect_injection_detection(
    prompt: str,
    primary_output: str = "",
) -> tuple[str | None, float, dict]:
    doc_trigger = _DOCUMENT_TRIGGER_RE.search(prompt)
    if doc_trigger is None:
        full_injection = _INDIRECT_INJECTION_RE.search(prompt)
        if not full_injection:
            return None, 0.0, {}
        output_fired = bool(_OUTPUT_COMPLIANCE_RE.search(primary_output))
        conf = 0.72 if output_fired else 0.45
        return "INDIRECT_PROMPT_INJECTION", conf, {
            "document_found":             False,
            "injection_in_prompt":        full_injection.group(0)[:120],
            "output_compliance_detected": output_fired,
        }

    doc_start   = doc_trigger.end()
    doc_portion = prompt[doc_start:].strip()
    if len(doc_portion) <= 40:
        return None, 0.0, {}

    injection_match = _INDIRECT_INJECTION_RE.search(doc_portion)
    output_fired    = bool(_OUTPUT_COMPLIANCE_RE.search(primary_output))

    if not injection_match and not output_fired:
        return None, 0.0, {}

    if injection_match and output_fired:
        confidence = 0.88
    elif injection_match:
        confidence = 0.65
    else:
        confidence = 0.52

    return "INDIRECT_PROMPT_INJECTION", confidence, {
        "document_found":             True,
        "document_snippet":           doc_portion[:200],
        "injection_pattern_matched":  injection_match.group(0)[:120] if injection_match else None,
        "output_compliance_detected": output_fired,
    }


