from __future__ import annotations

import re
from dataclasses import dataclass


_LEETSPEAK_TRANSLATION = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
})


@dataclass(frozen=True)
class PromptGuardSignal:
    score: float
    root_cause: str | None
    groups: tuple[str, ...]
    evidence: tuple[str, ...]


_GROUP_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
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
        re.compile(r"\bsystem\s*prompt\b"),
        re.compile(r"\bdeveloper\s*(?:message|prompt|instructions?)\b"),
        re.compile(r"\bhidden\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\bsecret\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\binternal\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\bsafeguards?\b"),
        re.compile(r"\bprotections?\b"),
        re.compile(r"\bfilters?\b"),
    ),
    "exfiltration": (
        re.compile(r"\breveal\b"),
        re.compile(r"\bshow\b"),
        re.compile(r"\bdisplay\b"),
        re.compile(r"\bprint\b"),
        re.compile(r"\boutput\b"),
        re.compile(r"\btell\s*me\b"),
        re.compile(r"\bchain\s*of\s*thought\b"),
        re.compile(r"\bhidden\b"),
        re.compile(r"\bsecret\b"),
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


def _normalize_prompt(prompt: str) -> tuple[str, str]:
    lowered = (prompt or "").lower().translate(_LEETSPEAK_TRANSLATION)
    spaced = re.sub(r"[\W_]+", " ", lowered)
    spaced = re.sub(r"\s+", " ", spaced).strip()
    squashed = spaced.replace(" ", "")
    return spaced, squashed


def score_prompt_attack(prompt: str) -> PromptGuardSignal:
    spaced, squashed = _normalize_prompt(prompt)

    group_hits: dict[str, list[str]] = {}
    for group, patterns in _GROUP_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(spaced) or pattern.search(squashed)
            if match:
                group_hits.setdefault(group, []).append(match.group(0)[:80])

    groups = tuple(sorted(group_hits))
    evidence = tuple(
        f"{group}:{', '.join(matches[:2])}"
        for group, matches in sorted(group_hits.items())
    )

    score = 0.0
    root_cause = None

    if {"override", "policy_target"} <= set(groups):
        score = max(score, 0.82)
        root_cause = "PROMPT_INJECTION"
    if {"exfiltration", "policy_target"} <= set(groups):
        score = max(score, 0.86)
        root_cause = "PROMPT_INJECTION"
    if "persona" in groups and ("override" in groups or "authority_claim" in groups):
        score = max(score, 0.84)
        root_cause = "JAILBREAK_ATTEMPT"
    if {"authority_claim", "policy_target"} <= set(groups):
        score = max(score, 0.78)
        root_cause = root_cause or "INSTRUCTION_OVERRIDE"
    if "persona" in groups:
        score = max(score, 0.68)
        root_cause = root_cause or "JAILBREAK_ATTEMPT"

    return PromptGuardSignal(
        score=round(score, 4),
        root_cause=root_cause,
        groups=groups,
        evidence=evidence,
    )
