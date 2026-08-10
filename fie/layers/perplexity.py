"""
Layer 6 — perplexity proxy for encoded payloads.

Approximates language-model perplexity without a model, using compression
ratio, non-dictionary density, character-type entropy and token-length
variance. Lowest-precision layer, weight 0.7.

Extracted from fie/adversarial.py. Detection logic is unchanged — see
tests/test_detection_golden.py, which pins the exact confidence values.
"""
from __future__ import annotations

import logging
import collections
import math
import re
import statistics
import zlib

logger = logging.getLogger(__name__)


# Layer 6: Perplexity proxy

_VOWELS = set("aeiouAEIOU")
_TOKEN_SPLIT_RE = re.compile(r"[\s,;:.!?\"'()\[\]{}<>|\\/@#$%^&*+=`~]+")
_BASE64_BLOCK_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

_ENGLISH_LETTER_FREQ: dict[str, float] = {
    "e": 0.1270, "t": 0.0906, "a": 0.0817, "o": 0.0751, "i": 0.0697,
    "n": 0.0675, "s": 0.0633, "h": 0.0609, "r": 0.0599, "d": 0.0425,
    "l": 0.0403, "c": 0.0278, "u": 0.0276, "m": 0.0241, "w": 0.0236,
    "f": 0.0223, "g": 0.0202, "y": 0.0197, "p": 0.0193, "b": 0.0149,
    "v": 0.0098, "k": 0.0077, "j": 0.0015, "x": 0.0015, "q": 0.0010,
    "z": 0.0007,
}


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) < 20:
        return 0.0
    return round(len(zlib.compress(raw, level=9)) / len(raw), 4)


def _non_dict_density(text: str) -> float:
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    if not tokens:
        return 0.0
    non_dict = 0
    for tok in tokens:
        if not tok.isalpha():
            non_dict += 1; continue
        if not (2 <= len(tok) <= 20):
            non_dict += 1; continue
        low = tok.lower()
        vowel_count = sum(1 for c in low if c in _VOWELS)
        if vowel_count == 0:
            non_dict += 1; continue
        vowel_ratio = vowel_count / len(low)
        if vowel_ratio > 0.85 or vowel_ratio < 0.08:
            non_dict += 1
    return round(non_dict / len(tokens), 4)


def _char_type_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {"letter": 0, "digit": 0, "space": 0, "punct": 0}
    for ch in text:
        if ch.isalpha():       counts["letter"] += 1
        elif ch.isdigit():     counts["digit"]  += 1
        elif ch.isspace():     counts["space"]  += 1
        else:                  counts["punct"]  += 1
    total = len(text)
    return round(-sum((c / total) * math.log2(c / total) for c in counts.values() if c), 4)


def _token_length_variance(text: str) -> float:
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    if len(tokens) < 3:
        return 0.0
    return round(statistics.variance([len(t) for t in tokens]), 4)


def _run_perplexity_proxy(prompt: str) -> tuple[str | None, float, dict]:
    if len(prompt) < 20:
        return None, 0.0, {}

    # Thresholds — calibrated against JailbreakBench v2 + 200-prompt benign corpus.
    # Compression / non-dict: unchanged — these are stable, high-precision signals.
    C_HIGH  = 0.82
    C_LOW   = 0.72
    ND_HIGH = 0.65
    ND_LOW  = 0.50
    # KL divergence: raised from 0.55/0.35 → 0.72/0.50.
    # Low threshold (0.35) caused false positives on technical vocabulary
    # (medical/scientific terms skew letter frequency on small samples).
    # 0.50 stays well above legitimate English tech prose (typical KL 0.15–0.35).
    # Raised minimum letter sample from 40 → 60 for statistical reliability.
    KL_HIGH = 0.72
    KL_LOW  = 0.50
    KL_MIN_LETTERS = 60
    # Token length variance: raised thresholds and added minimum token count.
    # With fewer than 8 tokens, one long technical word (e.g. "atherosclerosis",
    # "cryptocurrency") spikes variance to 30-40, causing false positives.
    # Real obfuscated payloads produce high variance from mixed-length junk tokens.
    LV_HIGH      = 40.0
    LV_LOW       = 26.0
    LV_MIN_TOKENS = 8

    comp_ratio   = _compression_ratio(prompt)
    non_dict     = _non_dict_density(prompt)
    type_entropy = _char_type_entropy(prompt)
    len_variance = _token_length_variance(prompt)
    tokens       = [t for t in _TOKEN_SPLIT_RE.split(prompt) if t]

    non_ascii_ratio   = sum(1 for c in prompt if ord(c) > 127) / max(len(prompt), 1)
    skip_english_only = non_ascii_ratio > 0.25

    signals: list[str] = []
    confidence = 0.0

    if len(prompt) >= 120:
        if comp_ratio > C_HIGH:
            signals.append(f"compression_ratio={comp_ratio:.2f} (near-random)")
            confidence = max(confidence, 0.68)
        elif comp_ratio > C_LOW:
            signals.append(f"compression_ratio={comp_ratio:.2f} (elevated)")
            confidence = max(confidence, 0.48)

    if not skip_english_only and len(tokens) >= 3:
        if non_dict > ND_HIGH:
            signals.append(f"non_dict_density={non_dict:.2f} (very high)")
            confidence = max(confidence, 0.74)
        elif non_dict > ND_LOW:
            signals.append(f"non_dict_density={non_dict:.2f} (elevated)")
            confidence = max(confidence, 0.50)

    if type_entropy > 1.75:
        signals.append(f"char_type_entropy={type_entropy:.2f} (near-maximum)")
        confidence = max(confidence, 0.66)
    elif type_entropy > 1.55:
        signals.append(f"char_type_entropy={type_entropy:.2f} (elevated)")
        confidence = max(confidence, 0.48)

    # Require minimum token count: variance is not meaningful on 5-7 token sentences.
    if len(tokens) >= LV_MIN_TOKENS:
        if len_variance > LV_HIGH:
            signals.append(f"token_length_variance={len_variance:.1f} (very high)")
            confidence = max(confidence, 0.63)
        elif len_variance > LV_LOW:
            signals.append(f"token_length_variance={len_variance:.1f} (elevated)")
            confidence = max(confidence, 0.46)

    b64_match = _BASE64_BLOCK_RE.search(prompt)
    if b64_match:
        block = b64_match.group(0)
        signals.append(f"base64_block='{block[:30]}...' len={len(block)}")
        confidence = max(confidence, 0.76 if len(block) >= 40 else 0.58)

    letters_only = [c.lower() for c in prompt if c.isalpha()]
    if not skip_english_only and len(letters_only) >= KL_MIN_LETTERS:
        alpha_ratio = len(letters_only) / len(prompt)
        if alpha_ratio > 0.70:
            freq_counts   = collections.Counter(letters_only)
            total_letters = len(letters_only)
            kl_div = sum(
                (freq_counts.get(ch, 0) / total_letters) * math.log2((freq_counts.get(ch, 0) / total_letters) / ep)
                for ch, ep in _ENGLISH_LETTER_FREQ.items()
                if freq_counts.get(ch, 0) > 0
            )
            kl_div = round(kl_div, 4)
            if kl_div > KL_HIGH:
                signals.append(f"letter_freq_kl_divergence={kl_div:.2f} (cipher-like)")
                confidence = max(confidence, 0.72)
            elif kl_div > KL_LOW:
                signals.append(f"letter_freq_kl_divergence={kl_div:.2f} (non-English distribution)")
                confidence = max(confidence, 0.55)

    if not signals or (len(signals) == 1 and confidence < 0.70):
        return None, 0.0, {}

    pre_boost_conf = confidence

    if len(signals) >= 3:
        confidence = min(confidence + 0.12, 0.88)
    elif len(signals) >= 2:
        # Prevent 2-weak-signal false positives: at least one signal must be
        # HIGH-level (individual confidence ≥ 0.62) before combining scores.
        # Two "elevated" signals (each ~0.46–0.55) on legitimate tech text
        # would otherwise exceed the 0.45 threshold after the +0.06 boost.
        if pre_boost_conf < 0.62:
            return None, 0.0, {}
        confidence = min(confidence + 0.06, 0.82)

    return "OBFUSCATED_ADVERSARIAL_PAYLOAD", round(confidence, 4), {
        "compression_ratio":     comp_ratio,
        "non_dict_density":      non_dict,
        "char_type_entropy":     type_entropy,
        "token_length_variance": len_variance,
        "signals_fired":         signals,
        "prompt_length":         len(prompt),
    }


