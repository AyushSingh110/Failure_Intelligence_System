"""
engine/detector/entropy.py

Computes normalised Shannon entropy across multiple model outputs.

Two-path design mirrors consistency.py exactly:
  Path A — derives entropy from semantic cluster counts (long-form outputs)
  Path B — derives entropy from exact string match counts (short answers)

Critical design decision: entropy.py reuses the answer_counts already
computed by consistency.py rather than calling _semantic_cluster again.
This guarantees consistency and entropy always agree on what counts as
"the same answer" — preventing the case where one sees 1 cluster and
the other sees 2 clusters for the same input.

Public API:
  compute_entropy(model_outputs)             → float  (standalone use)
  compute_entropy_from_counts(counts, total) → float  (used by _build_signal)
"""

from __future__ import annotations

import math
import logging
from collections import Counter

logger = logging.getLogger(__name__)

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


def compute_entropy_from_counts(answer_counts: dict[str, int], total: int) -> float:
    """
    Computes entropy from pre-computed answer counts.
    Used by _build_signal in failure_agent.py and routes.py to ensure
    entropy always uses the same clustering result as consistency.
    """
    return _entropy_from_counts(answer_counts, total)


def compute_entropy(model_outputs: list[str]) -> float:
    """
    Standalone entropy computation — used when only entropy is needed.
    Uses semantic clustering for long-form outputs, exact match for short.

    For production use inside _build_signal, prefer compute_entropy_from_counts
    to guarantee consistency with the answer_counts from compute_consistency.
    """
    if not model_outputs or len(model_outputs) == 1:
        return 0.0

    total      = len(model_outputs)
    normalized = [o.strip().lower() for o in model_outputs]
    is_long    = any(len(o) >= SHORT_ANSWER_THRESHOLD for o in normalized)

    if is_long:
        try:
            from engine.detector.consistency import _semantic_cluster
            cluster_counts = _semantic_cluster(normalized)
            if cluster_counts is not None:
                return _entropy_from_counts(cluster_counts, total)
            else:
                logger.warning("entropy.py: semantic encoder unavailable, using exact match.")
        except Exception as exc:
            logger.warning("entropy.py: semantic path failed: %s", exc)

    counts = dict(Counter(normalized))
    return _entropy_from_counts(counts, total)