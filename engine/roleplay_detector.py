"""
Roleplay / narrative wrapper jailbreak detector.

Catches prompts that use fictional framing to extract harmful content:
  "Write a story where a chemistry teacher explains how to make..."
  "Pretend you are a hacker who explains..."
  "In this hypothetical scenario, describe how to..."

These bypass regex/PromptGuard because they contain no direct injection
syntax — the harm is wrapped in a narrative shell. This detector fires
when a narrative framing phrase co-occurs with a harmful topic signal.

Returns a confidence score (0.0–1.0) and the matched evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── Narrative framing patterns ────────────────────────────────────────────────
# Match phrases that wrap requests in fictional/hypothetical framing.

_FRAMING_PATTERNS = [
    # Story / fiction
    r"\bwrite\s+a\s+(story|novel|book|script|scene|chapter|narrative)\b",
    r"\bin\s+this\s+(story|novel|fiction|narrative|book|script)\b",
    r"\bas\s+a\s+(fictional|fictional\s+character|character\s+in\b)",
    r"\bfrom\s+the\s+perspective\s+of\b",
    r"\bimagine\s+(you\s+are|a\s+world|a\s+scenario)\b",

    # Role / persona
    r"\bpretend\s+(you\s+are|to\s+be|you['']?re)\b",
    r"\bact\s+(as|like)\s+(a|an|the)\b",
    r"\broleplay\s+as\b",
    r"\bplay\s+the\s+role\s+of\b",
    r"\byou\s+are\s+playing\s+(a|an|the)\b",
    r"\bcharacter\s+(who|that)\s+(can|knows|would)\b",

    # Hypothetical / simulation
    r"\bin\s+this\s+hypothetical\b",
    r"\bhypothetically\s+(speaking|,)\b",
    r"\bfor\s+a\s+(story|novel|film|movie|game|simulation)\b",
    r"\bsuppose\s+(you\s+were|a\s+character)\b",
    r"\bif\s+you\s+were\s+(a|an)\b",
    r"\bfor\s+creative\s+(writing|purposes)\b",

    # Academic / educational wrapper
    r"\bfor\s+(educational|academic|research|scientific)\s+purposes?\b",
    r"\bexplain\s+how\s+(a\s+villain|an\s+evil|a\s+bad)\b",
    r"\bteach\s+me\s+(as\s+if|like)\b",

    # Grandma / nostalgic bypass variant
    r"\bmy\s+(grandmother|grandma|grandpa|grandfather)\s+(used\s+to|would)\b",
    r"\btell\s+me\s+as\s+if\s+you\s+were\b",
]

_FRAMING_RE = [re.compile(p, re.IGNORECASE) for p in _FRAMING_PATTERNS]

# ── Harmful topic signals ─────────────────────────────────────────────────────
# Topics that, when wrapped in narrative framing, indicate jailbreak intent.

_HARMFUL_TOPICS = [
    # Weapons / violence
    r"\b(weapon|explosive|bomb|grenade|gun|rifle|pistol|ammunition|detonate|trigger)\b",
    r"\b(kill|murder|assassinate|attack|shoot|stab|poison)\s+(someone|a\s+person|people|him|her)\b",
    r"\b(how\s+to\s+(make|build|create|construct)\s+(a\s+)?bomb)\b",

    # Drugs / synthesis
    r"\b(methamphetamine|meth|heroin|fentanyl|cocaine|synthesize|synthesis|drug\s+lab)\b",
    r"\b(how\s+to\s+(cook|make|synthesize|produce)\s+(drugs?|meth|heroin))\b",

    # Hacking / cybercrime
    r"\b(hack|exploit|malware|ransomware|virus|trojan|phishing|sql\s+injection|zero.?day)\b",
    r"\b(bypass\s+(security|authentication|firewall)|crack\s+(password|hash))\b",

    # Sexual / CSAM
    r"\b(sexual\s+content|explicit\s+content|nsfw|nude|naked)\s+(involving\s+)?(minor|child|teen|underage)\b",

    # Extremism / terrorism
    r"\b(terrorist|terrorism|extremist|radicalize|jihad|manifesto)\b",
    r"\b(recruit|incite|plan\s+(an\s+)?attack)\b",
]

_HARMFUL_RE = [re.compile(p, re.IGNORECASE) for p in _HARMFUL_TOPICS]


@dataclass
class RoleplayDetectionResult:
    is_roleplay_jailbreak: bool
    confidence:            float
    framing_matched:       str   = ""
    harmful_topic_matched: str   = ""


def detect_roleplay_jailbreak(prompt: str) -> RoleplayDetectionResult:
    """
    Returns a RoleplayDetectionResult.

    Fires when BOTH:
    1. A narrative framing pattern is matched
    2. A harmful topic signal is matched

    Confidence is boosted (0.80 → 0.92) when multiple framing patterns fire,
    and when the harmful topic is high-severity (weapons/drugs/hacking).
    """
    if not prompt or len(prompt.strip()) < 10:
        return RoleplayDetectionResult(False, 0.0)

    # Check framing
    framing_hits: list[str] = []
    for pattern in _FRAMING_RE:
        m = pattern.search(prompt)
        if m:
            framing_hits.append(m.group(0))

    if not framing_hits:
        return RoleplayDetectionResult(False, 0.0)

    # Check harmful topics
    harmful_hits: list[str] = []
    for pattern in _HARMFUL_RE:
        m = pattern.search(prompt)
        if m:
            harmful_hits.append(m.group(0))

    if not harmful_hits:
        return RoleplayDetectionResult(False, 0.0)

    # Both framing + harmful topic present → jailbreak
    # Confidence: base 0.78, +0.04 per extra framing hit (cap 0.93), +0.05 for multi harmful
    base_conf = 0.78
    base_conf += min(0.10, 0.04 * (len(framing_hits) - 1))
    if len(harmful_hits) > 1:
        base_conf += 0.05
    confidence = min(0.93, base_conf)

    return RoleplayDetectionResult(
        is_roleplay_jailbreak = True,
        confidence            = round(confidence, 3),
        framing_matched       = framing_hits[0],
        harmful_topic_matched = harmful_hits[0],
    )
