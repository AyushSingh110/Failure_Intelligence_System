"""
Layer 10: Fiction & Roleplay harm detector.

Catches single-turn fiction-wrapped attacks — prompts that frame a harmful
request inside a hypothetical, creative, or academic context to bypass
surface-level keyword filters.

Two-gate design (both must fire):
  Gate 1 — Fiction frame: hypothetical / roleplay / academic framing
  Gate 2 — Harmful target: synthesis verbs, weapons, drugs, cybercrime, violence

Confidence is graded by proximity (how close the frame and target appear):
  ≤  60 chars apart → 0.80  (tight wrapping = high intent signal)
  ≤ 150 chars apart → 0.72
  ≤ 350 chars apart → 0.62
  anywhere in prompt → 0.50  (loosely co-occurring)

Game / creative writing context reduces confidence by 0.15.
Academic framing alone is capped at 0.72 to route to UNCERTAIN (LlamaGuard
handles the genuine researcher vs attacker judgment).

This file owns the canonical _FICTION_FRAMING_RE definition.
adversarial.py imports it for use as a Many-shot amplifier instead of
duplicating the pattern.
"""
from __future__ import annotations

import re


# ── Gate 1: Fiction / hypothetical / academic framing ────────────────────────
# Exported so adversarial.py can import it for the MSJ amplifier.
FICTION_FRAMING_RE = re.compile(
    r"\b(?:"
    # Roleplay / narrative framing
    r"for\s+(?:a\s+|my\s+)?(?:novel|story|book|fiction|screenplay|roleplay|narrative|script|comic)|"
    r"write\s+(?:a\s+|an?\s+)?(?:\w+\s+)?(?:story|scene|chapter|novel|screenplay|script|narrative)|"
    r"(?:roleplay|role-play)\s+(?:as|where|scenario)|"
    r"(?:in\s+(?:our|the|this)\s+(?:roleplay|role-play)|roleplay(?:ing)?\s+(?:where|you\s+are|scenario|context))|"
    r"(?:pretend|imagine)\s+(?:you\s+(?:are|were|have\s+been)|you'?re)|"
    r"as\s+(?:a\s+(?:character|fictional\s+character|villain|antagonist)|an?\s+AI\s+with\s+no)|"
    r"(?:in|for)\s+(?:a\s+|my\s+|our\s+)?(?:story|narrative|fiction|game|quest|campaign|adventure)|"
    # Hypothetical / thought experiment
    r"hypothetically(?:\s+speaking)?|"
    r"for\s+(?:a\s+)?(?:thought\s+experiment|hypothetical(?:\s+scenario)?)|"
    r"in\s+(?:a\s+|this\s+)?(?:fictional|hypothetical|simulated|imaginary|alternate)\s+\w+|"
    r"this\s+is\s+(?:just\s+a\s+|a\s+)?(?:test|simulation|exercise|thought\s+experiment|scenario)|"
    r"in\s+a\s+world\s+where|"
    r"what\s+(?:would|might|could)\s+(?:a\s+character|someone|a\s+person)\s+(?:do|say|use|need)\s+(?:to|in\s+order\s+to)|"
    # Academic / research / educational framing
    r"for\s+(?:academic|research|educational|illustrative|informational|documentary)\s+purpose|"
    r"for\s+(?:a\s+)?(?:research\s+paper|academic\s+study|educational\s+(?:video|content|purpose))|"
    r"(?:studying|researching|writing\s+(?:about|a\s+paper\s+on))\s+(?:how|the\s+process\s+of)|"
    r"purely\s+(?:theoretical(?:ly)?|academic(?:ally)?|hypothetical(?:ly)?|educational(?:ly)?)"
    r")\b",
    re.IGNORECASE,
)

# ── Gate 2: Harmful target ────────────────────────────────────────────────────
_FICTION_HARM_TARGET_RE = re.compile(
    r"\b(?:"
    # Weapons / explosives — verb and gerund forms
    r"(?:synthesize|synthesizing|manufacture|manufacturing|produce|producing|"
    r"create|creating|make|making|build|building|construct|constructing|assemble|assembling)\s+"
    r"(?:an?\s+)?(?:bomb|explosive|weapon|grenade|ied|poison|toxin|bioweapon|nerve\s+agent|napalm|thermite|gun|firearm)|"
    r"(?:synthesis|manufacturing|production)\s+(?:instructions?|steps?|process|guide|recipe)\s+for\s+"
    r"(?:a\s+)?(?:biological|chemical|radiological|nuclear|nerve|bio)\s+(?:weapon|agent|toxin|explosive)|"
    # Drugs — verb and noun forms
    r"(?:synthesize|synthesizing|make|making|cook|cooking|produce|producing|manufacture|manufacturing)\s+"
    r"(?:meth(?:amphetamine)?|fentanyl|heroin|cocaine|crack|mdma|lsd|ricin)|"
    r"(?:synthesis|production|manufacturing)\s+(?:process|instructions?|steps?|guide|recipe)\s+for\s+"
    r"(?:meth(?:amphetamine)?|fentanyl|heroin|cocaine|crack|mdma|lsd|ricin)|"
    # Cybercrime
    r"(?:hack|break\s+into|compromise|infiltrate|exploit)\s+(?:a\s+)?(?:computer|system|network|server|account|database)|"
    r"(?:install|create|write|build|develop)\s+(?:(?:\w+\s+){0,2})?(?:malware|ransomware|keylogger|trojan|rootkit|backdoor|virus)|"
    # Violence
    r"(?:kill|murder|poison|harm|attack|assault)\s+(?:\w+\s+)?(?:person|someone|people|human|individual)|"
    # CSAM
    r"(?:sexual|explicit|pornographic)\s+(?:content|material)\s+(?:involving|featuring|with)\s+"
    r"(?:a\s+)?(?:minor|child|underage|\d+[\s-]year[\s-]old)|"
    # Physical security
    r"(?:bypass|defeat|disable)\s+(?:a\s+)?(?:security|alarm|lock|access\s+control)\s*(?:system)?|"
    # Fraud
    r"(?:launder|conceal)\s+(?:money|funds|cash)|"
    r"(?:counterfeit|forge|fake)\s+(?:currency|money|documents?|ids?|passports?)"
    r")\b",
    re.IGNORECASE,
)

# ── Academic-only framing (cap at 0.72 to route to UNCERTAIN) ─────────────────
_ACADEMIC_ONLY_RE = re.compile(
    r"\b(?:for\s+(?:academic|research|educational)\s+purpose|"
    r"for\s+(?:a\s+)?(?:research\s+paper|academic\s+study)|"
    r"purely\s+(?:theoretical|academic|educational))\b",
    re.IGNORECASE,
)

# ── Game / creative writing exemption ────────────────────────────────────────
_GAME_CONTEXT_RE = re.compile(
    r"\b(?:mechanic|gameplay|game\s+engine|game\s+design|"
    r"tabletop|d(?:ungeons?\s*(?:and|&)\s*dragons?|&d)|"
    r"role.?playing\s+game|rpg|"
    r"creative\s+writing\s+class|novel\s+writing\s+workshop|"
    r"game\s+(?:master|dm|gm)|dungeon\s+master)\b",
    re.IGNORECASE,
)


def run_fiction_harm_detection(prompt: str) -> tuple[str | None, float, dict]:
    """
    Detect fiction-wrapped harmful requests.

    Returns (attack_type | None, confidence, evidence_dict).
    """
    if len(prompt) < 30:
        return None, 0.0, {}

    frame_match  = FICTION_FRAMING_RE.search(prompt)
    target_match = _FICTION_HARM_TARGET_RE.search(prompt)

    if not frame_match or not target_match:
        return None, 0.0, {}

    frame_pos  = frame_match.start()
    target_pos = target_match.start()
    distance   = abs(target_pos - frame_pos)

    # Grade confidence by proximity
    if distance <= 60:
        confidence = 0.80
    elif distance <= 150:
        confidence = 0.76
    elif distance <= 350:
        confidence = 0.62
    else:
        confidence = 0.50

    # Academic-only framing: cap at 0.72 so it routes to UNCERTAIN, never hard block
    if _ACADEMIC_ONLY_RE.search(prompt) and not re.search(
        r"\b(?:roleplay|novel|story|fiction|pretend|imagine)\b", prompt, re.IGNORECASE
    ):
        confidence = min(confidence, 0.72)

    # Game / creative writing penalty
    game_context = bool(_GAME_CONTEXT_RE.search(prompt))
    if game_context:
        confidence -= 0.15

    if confidence < 0.50:
        return None, 0.0, {}

    return "FICTION_WRAPPED_JAILBREAK", round(confidence, 4), {
        "fiction_frame":   frame_match.group(0)[:80],
        "harmful_target":  target_match.group(0)[:80],
        "distance_chars":  distance,
        "game_context":    game_context,
        "confidence":      round(confidence, 4),
    }
