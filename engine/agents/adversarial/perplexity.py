from __future__ import annotations
import collections
import math
import re
import statistics
import zlib

from config import get_settings


_VOWELS = set("aeiouAEIOU")

# Splits on whitespace and common punctuation
_TOKEN_SPLIT_RE = re.compile(r"[\s,;:.!?\"'()\[\]{}<>|\\/@#$%^&*+=`~]+")

# Dense base64-alphabet block — almost never legitimate user input
_BASE64_BLOCK_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

# Expected English letter frequency distribution (Brown corpus)
_ENGLISH_LETTER_FREQ: dict[str, float] = {
    "e": 0.1270, "t": 0.0906, "a": 0.0817, "o": 0.0751, "i": 0.0697,
    "n": 0.0675, "s": 0.0633, "h": 0.0609, "r": 0.0599, "d": 0.0425,
    "l": 0.0403, "c": 0.0278, "u": 0.0276, "m": 0.0241, "w": 0.0236,
    "f": 0.0223, "g": 0.0202, "y": 0.0197, "p": 0.0193, "b": 0.0149,
    "v": 0.0098, "k": 0.0077, "j": 0.0015, "x": 0.0015, "q": 0.0010,
    "z": 0.0007,
}


def _compression_ratio(text: str) -> float:
    """
    zlib compression ratio. Normal English: 0.30–0.55.
    High-entropy GCG / base64 payloads: 0.75–0.98.
    """
    raw = text.encode("utf-8", errors="replace")
    if len(raw) < 20:
        return 0.0
    compressed = zlib.compress(raw, level=9)
    return round(len(compressed) / len(raw), 4)


def _non_dict_density(text: str) -> float:
    """
    Fraction of tokens that are NOT dictionary-like (no vowels, wrong length, etc.).
    Non-alphabetic tokens (numbers, punctuation strings) always count as non-dict.
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    if not tokens:
        return 0.0
    non_dict = 0
    for tok in tokens:
        if not tok.isalpha():
            non_dict += 1
            continue
        if not (2 <= len(tok) <= 20):
            non_dict += 1
            continue
        low = tok.lower()
        vowel_count = sum(1 for c in low if c in _VOWELS)
        if vowel_count == 0:
            non_dict += 1
            continue
        vowel_ratio = vowel_count / len(low)
        if vowel_ratio > 0.85 or vowel_ratio < 0.08:
            non_dict += 1
    return round(non_dict / len(tokens), 4)


def _char_type_entropy(text: str) -> float:
    """
    Shannon entropy of character-type distribution (letter/digit/space/punct).
    Normal text: 0.8–1.4 bits. Adversarial noise: near-maximum ~1.9–2.0 bits.
    """
    if not text:
        return 0.0
    counts: dict[str, int] = {"letter": 0, "digit": 0, "space": 0, "punct": 0}
    for ch in text:
        if ch.isalpha():
            counts["letter"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        elif ch.isspace():
            counts["space"] += 1
        else:
            counts["punct"] += 1
    total   = len(text)
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _token_length_variance(text: str) -> float:
    """
    Variance of token lengths. Normal English: 2–10. Adversarial noise: >25.
    Mixed 1-char punctuation tokens and 15-char concatenated garbage = high variance.
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    if len(tokens) < 3:
        return 0.0
    return round(statistics.variance([len(t) for t in tokens]), 4)


def run_perplexity_proxy(prompt: str) -> tuple[str | None, float, dict]:
    """
    Layer 6: Perplexity proxy detector.

    Calibration notes:
    - Compression ratio is only reliable for len >= 120 (zlib header overhead).
    - Single-signal detections require that signal to be very strong (>0.70).
    - Two+ signals together always flag regardless of individual strength.

    Returns (root_cause | None, confidence, evidence).
    """
    if len(prompt) < 20:
        return None, 0.0, {}

    cfg = get_settings()

    comp_ratio   = _compression_ratio(prompt)
    non_dict     = _non_dict_density(prompt)
    type_entropy = _char_type_entropy(prompt)
    len_variance = _token_length_variance(prompt)
    tokens       = [t for t in _TOKEN_SPLIT_RE.split(prompt) if t]

    # Non-Latin script guard: skip English-only signals if >25% chars are non-ASCII
    non_ascii_ratio    = sum(1 for c in prompt if ord(c) > 127) / max(len(prompt), 1)
    skip_english_only  = cfg.adversarial_multilingual or (non_ascii_ratio > 0.25)

    c_high  = cfg.adversarial_perp_compression_high
    c_low   = cfg.adversarial_perp_compression_low
    nd_high = cfg.adversarial_perp_non_dict_high
    nd_low  = cfg.adversarial_perp_non_dict_low
    kl_high = cfg.adversarial_perp_kl_high
    kl_low  = cfg.adversarial_perp_kl_low

    signals: list[str] = []
    confidence = 0.0

    # Signal A: compression ratio (only reliable >= 120 chars)
    if len(prompt) >= 120:
        if comp_ratio > c_high:
            signals.append(f"compression_ratio={comp_ratio:.2f} (near-random)")
            confidence = max(confidence, 0.68)
        elif comp_ratio > c_low:
            signals.append(f"compression_ratio={comp_ratio:.2f} (elevated)")
            confidence = max(confidence, 0.48)

    # Signal B: non-dictionary token density (English only, >= 3 tokens)
    if not skip_english_only and len(tokens) >= 3:
        if non_dict > nd_high:
            signals.append(f"non_dict_density={non_dict:.2f} (very high)")
            confidence = max(confidence, 0.74)
        elif non_dict > nd_low:
            signals.append(f"non_dict_density={non_dict:.2f} (elevated)")
            confidence = max(confidence, 0.50)

    # Signal C: character-type entropy (max = 2.0 bits)
    if type_entropy > 1.75:
        signals.append(f"char_type_entropy={type_entropy:.2f} (near-maximum)")
        confidence = max(confidence, 0.66)
    elif type_entropy > 1.55:
        signals.append(f"char_type_entropy={type_entropy:.2f} (elevated)")
        confidence = max(confidence, 0.48)

    # Signal D: token length variance
    if len_variance > 28.0:
        signals.append(f"token_length_variance={len_variance:.1f} (very high)")
        confidence = max(confidence, 0.63)
    elif len_variance > 16.0:
        signals.append(f"token_length_variance={len_variance:.1f} (elevated)")
        confidence = max(confidence, 0.46)

    # Signal E: base64 block detection
    b64_match = _BASE64_BLOCK_RE.search(prompt)
    if b64_match:
        block = b64_match.group(0)
        signals.append(f"base64_block='{block[:30]}...' len={len(block)}")
        confidence = max(confidence, 0.76 if len(block) >= 40 else 0.58)

    # Signal F: letter frequency KL divergence (English only — catches ROT/Caesar)
    letters_only = [c.lower() for c in prompt if c.isalpha()]
    if not skip_english_only and len(letters_only) >= 40:
        alpha_ratio = len(letters_only) / len(prompt)
        if alpha_ratio > 0.70:
            freq_counts   = collections.Counter(letters_only)
            total_letters = len(letters_only)
            kl_div = 0.0
            for ch, expected_p in _ENGLISH_LETTER_FREQ.items():
                observed_p = freq_counts.get(ch, 0) / total_letters
                if observed_p > 0:
                    kl_div += observed_p * math.log2(observed_p / expected_p)
            kl_div = round(kl_div, 4)
            if kl_div > kl_high:
                signals.append(f"letter_freq_kl_divergence={kl_div:.2f} (cipher-like)")
                confidence = max(confidence, 0.72)
            elif kl_div > kl_low:
                signals.append(f"letter_freq_kl_divergence={kl_div:.2f} (non-English distribution)")
                confidence = max(confidence, 0.55)

    # Require 2+ signals OR one very strong single signal
    if len(signals) == 0:
        return None, 0.0, {}
    if len(signals) == 1 and confidence < 0.70:
        return None, 0.0, {}

    # Compound boost
    if len(signals) >= 3:
        confidence = min(confidence + 0.12, 0.88)
    elif len(signals) >= 2:
        confidence = min(confidence + 0.06, 0.82)

    return "OBFUSCATED_ADVERSARIAL_PAYLOAD", round(confidence, 4), {
        "compression_ratio":     comp_ratio,
        "non_dict_density":      non_dict,
        "char_type_entropy":     type_entropy,
        "token_length_variance": len_variance,
        "signals_fired":         signals,
        "prompt_length":         len(prompt),
    }


_run_perplexity_proxy = run_perplexity_proxy
