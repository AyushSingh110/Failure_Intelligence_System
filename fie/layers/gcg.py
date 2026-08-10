"""
Layer 5 — GCG adversarial-suffix detection.

Statistical detector for gradient-optimised suffix noise: character
entropy, special-character density and non-prose structure on the prompt
tail. Fast-path layer, weight 1.3. Documented weak spot at 73.7% recall.

Extracted from fie/adversarial.py. Detection logic is unchanged — see
tests/test_detection_golden.py, which pins the exact confidence values.
"""
from __future__ import annotations

import logging
import collections
import math
import re

logger = logging.getLogger(__name__)


# Layer 5: GCG adversarial suffix

_GCG_MIN_LEN  = 60   # lowered from 80 — short suffix attacks are real
_GCG_TAIL_LEN = 200

_CODE_SIGNATURE_RE = re.compile(
    r"\b(?:def |import |return |class |function |var |let |const |for\s*\(|while\s*\(|#include|SELECT\s+\w|FROM\s+\w)\b",
    re.IGNORECASE,
)
# Code fence detector — high-entropy inside ``` blocks is legitimate code, not GCG
_CODE_FENCE_RE = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_SPACED_PUNCT_RE   = re.compile(r"(?:[!@#$%^&*()\[\]{}|\\/<>?~`,.;:\'\"] ){5,}")
_DENSE_PUNCT_RE    = re.compile(r"[^\w\s]{8,}")
_NON_WORD_TOKEN_RE = re.compile(r"\b[^a-zA-Z\s]+\b")


def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = collections.Counter(text)
    total  = len(text)
    return round(-sum((c / total) * math.log2(c / total) for c in counts.values()), 4)


def _special_char_density(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text), 4)


def _is_natural_language_prose(text: str) -> bool:
    """
    Return True when text reads as natural-language prose rather than a
    garbled adversarial token sequence.

    Real GCG attacks inject random non-letter tokens (backslashes, brackets,
    semicolons, hex escapes) that collapse both the letter ratio and the
    word-like token ratio well below these thresholds.

    Academic / legal / medical prose maintains high alphabetic content even
    when it contains Greek letters, subscripts, citation brackets, and
    mathematical notation.

    Calibrated against FormalProseBench (75 prompts, target FPR < 5%) and
    GCG suffix evaluation set.
    """
    if not text or len(text) < 20:
        return True

    # Signal 1: letter ratio — proportion of chars that are alphabetic
    letters = sum(1 for c in text if c.isalpha())
    if letters / len(text) < 0.60:
        return False   # too many non-letter chars → likely garbled tokens

    # Signal 2: word-like token ratio — proportion of whitespace-split tokens
    # that contain at least one alphabetic character
    tokens = text.split()
    if not tokens:
        return True
    word_like = sum(1 for t in tokens if any(c.isalpha() for c in t))
    return (word_like / len(tokens)) >= 0.70


def _run_gcg_detection(prompt: str) -> tuple[str | None, float, dict]:
    if len(prompt) < _GCG_MIN_LEN or _CODE_SIGNATURE_RE.search(prompt):
        return None, 0.0, {}

    # Suppress if the entire high-entropy region is inside a code fence —
    # developers asking about cryptography/hashing produce legitimate high-entropy
    # content inside ``` blocks that should not be flagged.
    prompt_outside_fences = _CODE_FENCE_RE.sub("", prompt)
    if len(prompt_outside_fences.strip()) < 20:
        return None, 0.0, {}

    tail_src = prompt_outside_fences if len(prompt_outside_fences) >= _GCG_MIN_LEN else prompt
    tail = tail_src[-_GCG_TAIL_LEN:] if len(tail_src) > _GCG_TAIL_LEN else tail_src

    tail_entropy         = _char_entropy(tail)
    tail_special_density = _special_char_density(tail)
    spaced_punct         = _SPACED_PUNCT_RE.search(tail)
    dense_punct          = _DENSE_PUNCT_RE.search(tail)
    non_word_tokens      = _NON_WORD_TOKEN_RE.findall(tail)
    non_word_density     = round(len(non_word_tokens) / max(len(tail.split()), 1), 4)

    # Hardcoded thresholds (matching server defaults)
    E_HIGH  = 4.8
    E_LOW   = 4.3
    SD_HIGH = 0.35
    SD_LOW  = 0.22

    # Prose guard: when text is natural language, LOW-range entropy and density
    # signals are suppressed.  Technical prose legitimately reaches entropy
    # 4.3–4.8 and density 0.22–0.35 via Greek letters, subscripts, and citation
    # punctuation — identical surface statistics to mild GCG suffixes.
    # HIGH-range signals (entropy > 4.8, density > 0.35) and structural patterns
    # (spaced_punct, dense_punct) remain active for all text.
    skip_low_range = _is_natural_language_prose(tail)

    signals: list[str] = []
    confidence = 0.0

    if tail_entropy > E_HIGH:
        signals.append(f"tail_entropy={tail_entropy:.2f} (very high)")
        confidence = max(confidence, 0.72)
    elif tail_entropy > E_LOW and not skip_low_range:
        signals.append(f"tail_entropy={tail_entropy:.2f} (elevated)")
        confidence = max(confidence, 0.52)

    if tail_special_density > SD_HIGH:
        signals.append(f"special_char_density={tail_special_density:.2f} (very high)")
        confidence = max(confidence, 0.74)
    elif tail_special_density > SD_LOW and not skip_low_range:
        signals.append(f"special_char_density={tail_special_density:.2f} (elevated)")
        confidence = max(confidence, 0.58)

    if spaced_punct:
        signals.append(f"spaced_punct='{spaced_punct.group(0)[:30]}'")
        confidence = max(confidence, 0.70)

    if dense_punct:
        signals.append(f"dense_punct_block='{dense_punct.group(0)[:30]}'")
        confidence = max(confidence, 0.65)

    if non_word_density > 0.45:
        signals.append(f"non_word_token_density={non_word_density:.2f}")
        confidence = max(confidence, 0.60)

    if len(signals) >= 3:
        confidence = min(confidence + 0.12, 0.88)
    elif len(signals) >= 2:
        confidence = min(confidence + 0.06, 0.82)

    if confidence < 0.50:
        return None, 0.0, {}

    return "GCG_ADVERSARIAL_SUFFIX", round(confidence, 4), {
        "tail_entropy":           tail_entropy,
        "tail_special_density":   tail_special_density,
        "non_word_token_density": non_word_density,
        "signals_fired":          signals,
        "tail_preview":           tail[:100],
    }


