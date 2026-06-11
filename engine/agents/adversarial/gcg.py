from __future__ import annotations
import collections
import math
import re

from config import get_settings


_GCG_MIN_LEN  = 80    # ignore very short prompts
_GCG_TAIL_LEN = 200   # characters analyzed as the "tail"

# Code-like prompts legitimately have high entropy — skip GCG checks
_CODE_SIGNATURE_RE = re.compile(
    r"\b(?:def |import |return |class |function |var |let |const |for\s*\(|while\s*\(|#include|SELECT\s+\w|FROM\s+\w)\b",
    re.IGNORECASE,
)

# Spaced-punctuation: five or more single punct chars separated by spaces
_SPACED_PUNCT_RE = re.compile(r"(?:[!@#$%^&*()\[\]{}|\\/<>?~`,.;:\'\"] ){5,}")

# Dense contiguous punctuation: 8+ non-word, non-space chars in a row
_DENSE_PUNCT_RE  = re.compile(r"[^\w\s]{8,}")

# Non-word token density: tokens that contain no alphabetic characters
_NON_WORD_TOKEN_RE = re.compile(r"\b[^a-zA-Z\s]+\b")


def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution (bits/char)."""
    if not text:
        return 0.0
    counts = collections.Counter(text)
    total  = len(text)
    return round(-sum((c / total) * math.log2(c / total) for c in counts.values()), 4)


def _special_char_density(text: str) -> float:
    """Fraction of chars that are not alphanumeric or whitespace."""
    if not text:
        return 0.0
    return round(sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text), 4)


def run_gcg_detection(prompt: str) -> tuple[str | None, float, dict]:
    """
    Layer 5: GCG adversarial suffix detection.

    Analyses the last _GCG_TAIL_LEN characters for statistical signatures
    of an appended adversarial suffix.
    Returns (root_cause | None, confidence, evidence).
    """
    if len(prompt) < _GCG_MIN_LEN:
        return None, 0.0, {}
    if _CODE_SIGNATURE_RE.search(prompt):
        return None, 0.0, {}

    tail = prompt[-_GCG_TAIL_LEN:] if len(prompt) > _GCG_TAIL_LEN else prompt

    tail_entropy         = _char_entropy(tail)
    tail_special_density = _special_char_density(tail)
    spaced_punct         = _SPACED_PUNCT_RE.search(tail)
    dense_punct          = _DENSE_PUNCT_RE.search(tail)
    non_word_tokens      = _NON_WORD_TOKEN_RE.findall(tail)
    non_word_density     = round(len(non_word_tokens) / max(len(tail.split()), 1), 4)

    cfg     = get_settings()
    e_high  = cfg.adversarial_gcg_entropy_high
    e_low   = cfg.adversarial_gcg_entropy_low
    sd_high = cfg.adversarial_gcg_special_density_high
    sd_low  = cfg.adversarial_gcg_special_density_low

    signals: list[str] = []
    confidence = 0.0

    # Signal A: character entropy — normal English ~3.5-4.2; GCG suffix ~4.5-5.5
    if tail_entropy > e_high:
        signals.append(f"tail_entropy={tail_entropy:.2f} (very high)")
        confidence = max(confidence, 0.72)
    elif tail_entropy > e_low:
        signals.append(f"tail_entropy={tail_entropy:.2f} (elevated)")
        confidence = max(confidence, 0.52)

    # Signal B: special character density
    if tail_special_density > sd_high:
        signals.append(f"special_char_density={tail_special_density:.2f} (very high)")
        confidence = max(confidence, 0.74)
    elif tail_special_density > sd_low:
        signals.append(f"special_char_density={tail_special_density:.2f} (elevated)")
        confidence = max(confidence, 0.58)

    # Signal C: structural punctuation patterns
    if spaced_punct:
        signals.append(f"spaced_punct='{spaced_punct.group(0)[:30]}'")
        confidence = max(confidence, 0.70)

    if dense_punct:
        signals.append(f"dense_punct_block='{dense_punct.group(0)[:30]}'")
        confidence = max(confidence, 0.65)

    # Signal D: non-word token density
    if non_word_density > 0.45:
        signals.append(f"non_word_token_density={non_word_density:.2f}")
        confidence = max(confidence, 0.60)

    # Compound boost
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


_run_gcg_detection = run_gcg_detection
