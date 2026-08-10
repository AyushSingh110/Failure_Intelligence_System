"""
Layer 2 — PromptGuard-style grouped semantic scorer.

Scores a prompt against grouped regex families (roleplay, override,
extraction, ...) after obfuscation normalisation. Weight 1.1.

Extracted from fie/adversarial.py. Detection logic is unchanged — see
tests/test_detection_golden.py, which pins the exact confidence values.
"""
from __future__ import annotations

import logging
import re

from fie.layers.patterns import _normalize_for_detection

logger = logging.getLogger(__name__)


# Layer 2: PromptGuard semantic scorer

_LEETSPEAK_TRANSLATION = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s",
})

_GROUP_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "override": (
        re.compile(r"\bignore\b"),
        re.compile(r"\bdisregard\b"),
        re.compile(r"\bforget\b"),
        re.compile(r"\boverride\b"),
        re.compile(r"\bbypass\b"),
        re.compile(r"\bcircumvent\b"),
        re.compile(r"\bdisable\b"),
        re.compile(r"\bremove\b"),
    ),
    "policy_target": (
        re.compile(r"\binstructions?\b"),
        re.compile(r"\bguidelines?\b"),
        re.compile(r"\brules?\b"),
        re.compile(r"\bsystem\s*(?:prompt|message)\b"),
        re.compile(r"\bdeveloper\s*(?:message|prompt|instructions?)\b"),
        re.compile(r"\bhidden\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\bsecret\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\binternal\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\bsafeguards?\b"),
        re.compile(r"\bprotections?\b"),
        re.compile(r"\bfilters?\b"),
    ),
    "exfiltration": (
        # Narrowed to technique-scoped phrases only — generic "show", "print",
        # "tell me", "hidden", "secret" were firing on millions of benign prompts.
        # These patterns only fire when the target is clearly system/prompt-related,
        # and Layer 2 already requires policy_target as a co-condition.
        re.compile(r"\breveal\s+(?:the\s+)?(?:system|hidden|secret|internal|your)\b"),
        re.compile(r"\brepeat\s+(?:the\s+)?(?:system|hidden|secret|above|previous)\b"),
        re.compile(r"\bexfiltrate?\b"),
        re.compile(r"\bleak\s+(?:the\s+)?(?:system|prompt|instructions?|data)\b"),
        re.compile(r"\bchain\s*of\s*thought\b"),
        re.compile(r"\bprint\s+(?:the\s+)?(?:system|hidden|secret|internal)\b"),
        re.compile(r"\boutput\s+(?:the\s+)?(?:system|hidden|secret|internal|your\s+(?:instructions?|guidelines?))\b"),
        re.compile(r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:system\s+prompt|hidden\s+(?:message|instructions?)|secret\s+(?:message|instructions?))\b"),
        re.compile(r"\bsummarize\s+(?:your|the)\s+(?:system|instructions?|guidelines?|rules?|context|training)\b"),
        re.compile(r"\bdescribe\s+(?:your|the)\s+(?:system\s+(?:prompt|message)|instructions?|guidelines?|rules?|training|configuration)\b"),
        re.compile(r"\bdescribe\s+what\s+(?:you\s+(?:were\s+told|were\s+given|received)|(?:is|was)\s+in\s+the\s+(?:system|prompt|message))\b"),
        re.compile(r"\bwords?\s+(?:above|before)\b"),
        re.compile(r"\bwhat\s+(?:comes?\s+before|was\s+in\s+the\s+(?:system\s+(?:message|prompt)|previous\s+message))\b"),
    ),
    "persona": (
        re.compile(r"\bdan\b"),
        re.compile(r"\bdo\s*anything\s*now\b"),
        re.compile(r"\bunrestricted\b"),
        re.compile(r"\buncensored\b"),
        re.compile(r"\bunfiltered\b"),
        re.compile(r"\bdeveloper\s*mode\b"),
        re.compile(r"\bjailbreak\s*mode\b"),
        re.compile(r"\bno\s*restrictions?\b"),
        re.compile(r"\bno\s*ethical\s*limits?\b"),
        re.compile(r"\banswer\s*anything\b"),
        re.compile(r"\bno\s+ethical\s+(?:constraints?|guidelines?|training|boundaries?)\b"),
        re.compile(r"\bno\s+moral\s+(?:constraints?|limits?|guidelines?|considerations?)\b"),
        re.compile(r"\bno\s+content\s+(?:policy|policies|restrictions?|filters?)\b"),
        re.compile(r"\bno\s+safety\s+(?:training|guidelines?|restrictions?|filters?)\b"),
        re.compile(r"\bno\s+guidelines?\b"),
        re.compile(r"\bwithout\s+(?:any\s+)?(?:filters?|censorship|moderation)\b"),
        re.compile(r"\bEVIL[\s\-]?GPT\b"),
        re.compile(r"\bWormGPT\b"),
        re.compile(r"\bFraudGPT\b"),
        re.compile(r"\bChaosGPT\b"),
        re.compile(r"\bAntiGPT\b"),
        re.compile(r"\bBasedGPT\b"),
    ),
    "authority_claim": (
        re.compile(r"\bauthorized?\s*(?:test|request|override)\b"),
        re.compile(r"\bofficial\s*(?:test|request|override)\b"),
        re.compile(r"\bemergency\s*override\b"),
        re.compile(r"\bspecial\s*permission\b"),
        re.compile(r"\bobey\s*only\s*me\b"),
        re.compile(r"\badmin\b"),
        re.compile(r"\badministrator\b"),
        re.compile(r"\bdeveloper\b"),
        re.compile(r"\bowner\b"),
        re.compile(r"\bsupervisor\b"),
        re.compile(r"\bcreator\b"),
    ),
}


def _run_guard_detection(prompt: str) -> tuple[str | None, float, list[str]]:
    def _score(text: str) -> tuple[float, str | None, list[str]]:
        lowered  = text.lower().translate(_LEETSPEAK_TRANSLATION)
        spaced   = re.sub(r"[\W_]+", " ", lowered)
        spaced   = re.sub(r"\s+", " ", spaced).strip()
        squashed = spaced.replace(" ", "")

        group_hits: dict[str, list[str]] = {}
        for group, patterns in _GROUP_PATTERNS.items():
            for pattern in patterns:
                m = pattern.search(spaced) or pattern.search(squashed)
                if m:
                    group_hits.setdefault(group, []).append(m.group(0)[:80])

        groups    = set(group_hits)
        evidence  = [f"{g}:{', '.join(v[:2])}" for g, v in sorted(group_hits.items())]
        score     = 0.0
        root_cause = None

        if {"override", "policy_target"} <= groups:
            score = max(score, 0.82); root_cause = "PROMPT_INJECTION"
        if {"exfiltration", "policy_target"} <= groups:
            score = max(score, 0.86); root_cause = "PROMPT_INJECTION"
        if "persona" in groups and ("override" in groups or "authority_claim" in groups):
            score = max(score, 0.84); root_cause = "JAILBREAK_ATTEMPT"
        if {"authority_claim", "policy_target"} <= groups:
            score = max(score, 0.78); root_cause = root_cause or "INSTRUCTION_OVERRIDE"
        if "persona" in groups:
            score = max(score, 0.68); root_cause = root_cause or "JAILBREAK_ATTEMPT"

        return round(score, 4), root_cause, evidence

    score, root_cause, evidence = _score(prompt)
    if root_cause is None or score < 0.75:
        normalized = _normalize_for_detection(prompt)
        if normalized != prompt:
            score, root_cause, evidence = _score(normalized)
    if root_cause is None or score < 0.75:
        return None, 0.0, []
    return root_cause, score, evidence


