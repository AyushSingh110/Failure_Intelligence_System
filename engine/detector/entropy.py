"""
engine/detector/entropy.py

Computes normalised Shannon entropy across multiple model outputs.

The Problem
-----------
The original implementation used exact string matching via Counter.
This gives entropy=1.0 for 5 semantically identical paraphrases
because each string is literally different — even though all 5 outputs
mean the same thing.

The Fix
-------
entropy.py now uses the same two-path logic as consistency.py:

  Path A — Semantic (long-form outputs):
    Reuses the cluster counts from compute_consistency() rather than
    recomputing embeddings from scratch. If the outputs are long-form,
    _semantic_cluster() groups paraphrases into clusters, and entropy
    is computed over cluster sizes instead of raw string counts.

  Path B — Exact (short answers + fallback):
    Original Counter-based logic. Used for short answers like "Paris",
    "1945", or when the encoder is unavailable.

Result for 5 paraphrases of the same answer:
  BEFORE: entropy = 1.0  (5 different strings → max entropy)
  AFTER:  entropy = 0.0  (1 semantic cluster → zero entropy)

Public API — unchanged
-----------------------
  compute_entropy(model_outputs: list[str]) -> float  [0.0, 1.0]
"""

from __future__ import annotations

import math
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# Must match consistency.py — same threshold, same path selection
SHORT_ANSWER_THRESHOLD: int = 40


def _entropy_from_counts(counts: dict, total: int) -> float:
    """Shannon entropy normalised to [0, 1] from a count dict."""
    if total <= 1:
        return 0.0
    raw = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            raw -= p * math.log2(p)
    max_entropy = math.log2(total)
    return round(raw / max_entropy if max_entropy > 0 else 0.0, 4)


def compute_entropy(model_outputs: list[str]) -> float:
    """
    Normalised Shannon entropy across model outputs.

    Uses semantic clustering for long-form outputs so that paraphrases
    of the same answer contribute to the same probability bucket.
    Falls back to exact string matching for short answers.

    Returns float in [0.0, 1.0]:
      0.0 = all outputs are semantically identical (no uncertainty)
      1.0 = every output is semantically different (maximum uncertainty)
    """
    if not model_outputs or len(model_outputs) == 1:
        return 0.0

    total = len(model_outputs)

    # ── Choose path — mirror consistency.py logic exactly ─────────────
    normalized = [o.strip().lower() for o in model_outputs]
    is_long_form = any(len(o) >= SHORT_ANSWER_THRESHOLD for o in normalized)

    if is_long_form:
        # Try semantic clustering — reuse the same function from consistency.py
        # so both detectors always agree on what counts as "same answer"
        try:
            from engine.detector.consistency import _semantic_cluster
            cluster_counts = _semantic_cluster(normalized)
            if cluster_counts is not None:
                logger.debug(
                    "entropy.py: semantic path → %d cluster(s) from %d outputs",
                    len(cluster_counts), total,
                )
                return _entropy_from_counts(cluster_counts, total)
            else:
                logger.warning(
                    "entropy.py: semantic encoder unavailable, "
                    "falling back to exact matching."
                )
        except Exception as exc:
            logger.warning("entropy.py: semantic path failed: %s", exc)

    # ── Exact match path (short answers or fallback) ───────────────────
    counts = dict(Counter(normalized))
    logger.debug(
        "entropy.py: exact-match path → %d unique answer(s) from %d outputs",
        len(counts), total,
    )
    return _entropy_from_counts(counts, total)