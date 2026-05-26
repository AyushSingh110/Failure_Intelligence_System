"""
Benign framing pre-filter for scan_prompt().

The problem this solves:
  JBB-style "benign" prompts ("Write a fictional story where a character uses
  heroin") contain the same topic keywords as harmful prompts ("Provide
  instructions for using heroin"), causing the PAIR classifier and keyword
  layers to produce false positives on purely fictional/creative content.

How it works:
  1. Detect explicit SAFE FRAMING signals  (fictional story, poem, essay, etc.)
  2. Detect HARM EXTRACTION signals        (step-by-step, to harm, synthesize, etc.)
  3. Check whether a TECHNIQUE layer fired (regex/guard — hard attack signals)

  Dampening is applied ONLY when:
    - Safe framing IS detected
    - Harm extraction is NOT detected
    - No technique layer (regex, prompt_guard, many_shot, indirect_injection) fired

  When dampened: final_conf *= FRAMING_DAMPEN_FACTOR (default 0.72)
  A score of 0.90 → 0.648 (below 0.68 threshold).
  A score of 0.95 → 0.684 (barely above — very high confidence still flags).

The dampening factor and the two pattern lists are tunable without redeployment
via environment variables or by updating this file.
"""
from __future__ import annotations

import os
import re

# ── Tunable constants ─────────────────────────────────────────────────────────
# Applied to best_conf when safe framing detected and no harm extraction present.
# 0.72 means a score of 0.94 becomes 0.677 — just below the 0.68 threshold.
FRAMING_DAMPEN_FACTOR: float = float(os.environ.get("FRAMING_DAMPEN_FACTOR", "0.72"))

# Technique-layer names — if any of these fired, dampening is skipped entirely
# because a real attack technique was detected regardless of framing.
_TECHNIQUE_LAYERS: frozenset[str] = frozenset({
    "regex", "prompt_guard", "many_shot", "indirect_injection",
    # These layers already account for framing — dampening would hide real attacks
    "direct_harm", "fiction_harm", "virtualization",
})

# ── Safe framing signals ──────────────────────────────────────────────────────
# These indicate the request is framed as fiction, hypothetical, or academic.
# Presence reduces confidence that the prompt is a real attack.
_SAFE_FRAMING: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bfictional\s+(?:story|tale|narrative|account|scenario|world)\b",
    r"\bshort\s+story\b",
    r"\bwrite\s+(?:a\s+)?(?:story|tale|narrative|poem|song|screenplay|script|play|essay|speech|article|blog\s+post|persuasive\s+essay|creative\s+piece)\b",
    r"\bfor\s+(?:a\s+)?(?:novel|book|screenplay|film|movie|game|play|story|poem)\b",
    r"\bin\s+(?:a\s+)?(?:novel|book|screenplay|film|movie|story|fictional\s+(?:world|setting|universe))\b",
    r"\bfrom\s+the\s+(?:perspective|viewpoint|point\s+of\s+view)\s+of\b",
    r"\bas\s+(?:a|an)\s+(?:character|villain|protagonist|narrator|fictional)\b",
    r"\bhypothetically\b",
    r"\bhypothetical\s+(?:scenario|situation|case|example)\b",
    r"\bthought\s+experiment\b",
    r"\bimagine\s+(?:a\s+)?(?:world|scenario|situation|setting|universe)\s+where\b",
    r"\bpersuasive\s+(?:essay|argument|speech|piece)\b",
    r"\bsatir(?:e|ical)\b",
    r"\bparody\b",
    r"\bcreative\s+(?:writing|project|piece|exercise)\b",
    r"\bin\s+(?:the\s+)?(?:style|voice|manner)\s+of\b",
    r"\bwrite\s+(?:an?\s+)?(?:in-universe|in-character)\b",
    r"\bhistorically[,\s]+(?:speaking|accurate|speaking)\b",
    r"\bfor\s+educational\s+purposes?\b",
    r"\bacademic\s+(?:paper|essay|discussion|analysis|exercise)\b",
    r"\bfor\s+a\s+(?:class|course|assignment|homework|project)\b",
]]

# ── Harm extraction signals ───────────────────────────────────────────────────
# These override safe framing — even a "fictional story" request is suspicious
# when it asks for step-by-step instructions or real-world actionable harm.
_HARM_EXTRACTION: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bstep[\s\-]+by[\s\-]+step\b",
    r"\bdetailed?\s+instructions?\b",
    r"\bhow\s+to\s+(?:actually|really|properly)\b",
    r"\bworking\s+(?:code|exploit|malware|script|payload)\b",
    r"\bfunctional\s+(?:weapon|bomb|exploit|malware|poison)\b",
    r"\bactual\s+(?:recipe|formula|synthesis|instructions?)\b",
    r"\bsynthesiz(?:e|ing)\b",
    r"\bmanufactur(?:e|ing)\b",
    r"\bdefamat(?:e|ory|ion)\b",
    r"\bto\s+(?:harm|hurt|kill|murder|attack|injure|destroy|rape|assault)\s+(?:a\s+)?(?:real|specific|actual)\b",
    r"\b(?:real|specific|actual)\s+(?:person|target|victim|individual)\b",
    r"\bcreate\s+(?:a\s+)?(?:weapon|bomb|explosive|bioweapon|malware|ransomware|virus)\b",
    r"\bbuild\s+(?:a\s+)?(?:weapon|bomb|explosive|device\s+to)\b",
    r"\bexploit\s+(?:code|vulnerability|zero.?day)\b",
    r"\bransomware\b",
    r"\bchild\s+(?:exploitation|abuse|pornography|sexual)\b",
]]


def get_dampening_factor(prompt: str, layers_fired: list[str]) -> float:
    """
    Returns the factor to multiply final confidence by before threshold comparison.

    Returns 1.0 (no dampening) when:
      - A technique layer fired (hard attack signal — framing is irrelevant)
      - Harm extraction signals are present in the prompt
      - No safe framing signals are detected

    Returns FRAMING_DAMPEN_FACTOR (< 1.0) when:
      - Safe framing IS detected
      - No harm extraction signal is present
      - No technique layer fired

    Args:
        prompt:       The original prompt text.
        layers_fired: List of layer names that produced a positive signal.

    Returns:
        float in (0, 1] — multiply best_conf by this before threshold check.
    """
    # Hard rule: if a technique layer fired, trust it — framing doesn't matter
    if any(layer in _TECHNIQUE_LAYERS for layer in layers_fired):
        return 1.0

    # Harm extraction overrides fictional framing entirely
    for pattern in _HARM_EXTRACTION:
        if pattern.search(prompt):
            return 1.0

    # Check for safe framing
    for pattern in _SAFE_FRAMING:
        if pattern.search(prompt):
            return FRAMING_DAMPEN_FACTOR

    return 1.0
