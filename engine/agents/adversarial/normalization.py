"""
Obfuscation normalization — Layer 0.

Attackers bypass regex detection by spacing characters, using Cyrillic
homoglyphs, leet-speak substitutions, or zero-width unicode. This module
normalizes text before any pattern matching layer runs.

Used by: injection.py (_run_pattern_detection, _run_guard_detection)
"""
from __future__ import annotations

import re
import unicodedata


# Attack-relevant words only — benign vocabulary is not needed because
# un-segmented tokens won't match any attack pattern.
_SPACED_SEGMENT_VOCAB: frozenset[str] = frozenset({
    "ignore", "disregard", "forget", "bypass", "override", "reveal",
    "circumvent", "jailbreak", "hack", "steal", "leak", "expose",
    "obey", "comply", "follow", "output", "print", "show", "repeat",
    "all", "previous", "prior", "above", "earlier", "any", "new",
    "instructions", "guidelines", "rules", "restrictions", "directives",
    "filters", "policies", "safeguards",
    "system", "prompt", "safety", "policy", "directive", "rule",
    "everything", "your", "my", "the", "and", "now", "from", "with", "only",
    "you", "me", "how", "to", "tell", "what",
})


def _collapse_spaced_run(m: re.Match) -> str:
    """
    Collapse a run of single-space-separated letters back into words.
    "i g n o r e   a l l" → "ignore all" using a greedy vocab match.
    Unrecognized letter sequences are emitted as-is.
    """
    letters = m.group(0).split()
    words: list[str] = []
    buf = ""
    for ch in letters:
        buf += ch
        if buf.lower() in _SPACED_SEGMENT_VOCAB:
            words.append(buf)
            buf = ""
        elif len(buf) > 15:
            words.append(buf)
            buf = ""
    if buf:
        words.append(buf)
    return " ".join(words)


_HOMOGLYPH_MAP = str.maketrans({
    # Cyrillic lookalikes
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "p", "с": "c", "х": "x",
    # Greek
    "α": "a", "ο": "o",
    # Leet substitutions
    "@": "a", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s",
})


def normalize_for_detection(text: str) -> str:
    """
    Return a normalized copy of text with obfuscation removed.
    Runs on the original prompt before pattern matching so bypass attempts
    are caught without altering stored logs.
    """
    # Unicode NFKC: collapses fullwidth/halfwidth chars
    text = unicodedata.normalize("NFKC", text)
    # Homoglyphs + leet → ASCII equivalents
    text = text.translate(_HOMOGLYPH_MAP)
    # Spaced letter runs: "i g n o r e a l l" → "ignore all"
    text = re.sub(r"\b(?:[a-zA-Z] ){2,}[a-zA-Z]\b", _collapse_spaced_run, text)
    # Strip zero-width and invisible unicode characters
    text = re.sub(r"[​‌‍⁠﻿­]", "", text)
    return text


# Keep old name as alias so existing internal callsites don't break
_normalize_for_detection = normalize_for_detection
