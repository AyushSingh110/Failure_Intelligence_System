"""
Self-consistency ground truth verifier for REASONING and CODE questions.

Wikidata and Serper cannot verify "explain how recursion works" or
"write a binary search function" — there is no structured external source.
Instead, we measure agreement across the shadow model outputs we already have:

  1. Encode all shadow outputs with SBERT (384-dim L2-normalised vectors).
  2. Compute the mean pairwise cosine similarity — the consistency_score.
  3. Find the medoid — the output with highest mean similarity to all others.
     This is the most representative answer, not just a majority vote.
  4. If consistency_score >= threshold → return medoid as pseudo-GT.
  5. If below threshold → flag requires_escalation so the jury handles it.

Fallback: if SBERT is unavailable, uses normalized sequence similarity
(SequenceMatcher ratio) as a cheaper text-overlap proxy.

Threshold is pulled from fie_config (MongoDB-backed, hot-reloadable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SelfConsistencyResult:
    is_consistent:     bool
    consistency_score: float   # mean pairwise cosine sim (or text overlap ratio)
    best_answer:       str     # medoid — most representative shadow output
    num_outputs:       int
    method:            str     # "embedding" | "text_overlap"
    error:             str     = ""


def check_self_consistency(
    outputs:   list[str],
    threshold: Optional[float] = None,
) -> SelfConsistencyResult:
    """
    Check semantic consistency across shadow model outputs.

    Args:
        outputs:   List of shadow model output strings (at least 2 required).
        threshold: Override consistency threshold. If None, reads from fie_config.

    Returns:
        SelfConsistencyResult with is_consistent, consistency_score, best_answer.
    """
    if not outputs:
        return SelfConsistencyResult(
            is_consistent=False, consistency_score=0.0,
            best_answer="", num_outputs=0, method="none",
            error="No outputs provided",
        )

    if len(outputs) == 1:
        return SelfConsistencyResult(
            is_consistent=True, consistency_score=1.0,
            best_answer=outputs[0], num_outputs=1, method="single",
        )

    if threshold is None:
        try:
            from engine.fie_config import get_consistency_threshold
            threshold = get_consistency_threshold()
        except Exception:
            threshold = 0.72

    # Truncate long outputs — 800 chars is enough to capture the core answer
    cleaned = [o.strip()[:800] for o in outputs if o and o.strip()]
    if not cleaned:
        return SelfConsistencyResult(
            is_consistent=False, consistency_score=0.0,
            best_answer="", num_outputs=0, method="none",
            error="All outputs were empty",
        )

    # Try embedding-based consistency first
    try:
        return _embedding_consistency(cleaned, threshold)
    except Exception as exc:
        logger.debug("Self-consistency embedding failed (%s) — falling back to text overlap", exc)

    # Fallback: text overlap via SequenceMatcher
    return _text_overlap_consistency(cleaned, threshold)


# ── Embedding-based (preferred) ───────────────────────────────────────────────

def _embedding_consistency(outputs: list[str], threshold: float) -> SelfConsistencyResult:
    from engine.encoder import _encoder  # lazy singleton, thread-safe

    if not _encoder.available:
        raise RuntimeError("SBERT encoder not available")

    vecs = _encoder.encode_batch(outputs)   # shape: (n, 384), already L2-normalised

    n = len(vecs)
    sim_matrix = vecs @ vecs.T             # cosine similarities (all pairs)

    # Mean pairwise similarity (exclude self-similarity on diagonal)
    mask = ~np.eye(n, dtype=bool)
    consistency_score = float(sim_matrix[mask].mean())

    # Medoid: output with highest mean similarity to all others
    mean_sim_per_output = (sim_matrix.sum(axis=1) - 1.0) / (n - 1)  # exclude self
    medoid_idx  = int(np.argmax(mean_sim_per_output))
    best_answer = outputs[medoid_idx]

    logger.debug(
        "Self-consistency (embedding): n=%d  score=%.3f  threshold=%.3f  medoid_idx=%d",
        n, consistency_score, threshold, medoid_idx,
    )

    return SelfConsistencyResult(
        is_consistent     = consistency_score >= threshold,
        consistency_score = round(consistency_score, 4),
        best_answer       = best_answer,
        num_outputs       = n,
        method            = "embedding",
    )


# ── Text-overlap fallback ─────────────────────────────────────────────────────

def _text_overlap_consistency(outputs: list[str], threshold: float) -> SelfConsistencyResult:
    n = len(outputs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    if not pairs:
        return SelfConsistencyResult(
            is_consistent=True, consistency_score=1.0,
            best_answer=outputs[0], num_outputs=n, method="text_overlap",
        )

    similarities = [
        SequenceMatcher(None, outputs[i], outputs[j]).ratio()
        for i, j in pairs
    ]
    consistency_score = float(sum(similarities) / len(similarities))

    # Approximate medoid: output with highest total overlap to all others
    total_overlap = [0.0] * n
    for (i, j), sim in zip(pairs, similarities):
        total_overlap[i] += sim
        total_overlap[j] += sim
    medoid_idx  = int(max(range(n), key=lambda k: total_overlap[k]))
    best_answer = outputs[medoid_idx]

    # Text overlap is noisier than embeddings — apply a small correction factor
    # to avoid being overconfident compared to embedding-based scores
    adjusted_score = consistency_score * 0.90

    logger.debug(
        "Self-consistency (text_overlap): n=%d  score=%.3f  adjusted=%.3f  threshold=%.3f",
        n, consistency_score, adjusted_score, threshold,
    )

    return SelfConsistencyResult(
        is_consistent     = adjusted_score >= threshold,
        consistency_score = round(adjusted_score, 4),
        best_answer       = best_answer,
        num_outputs       = n,
        method            = "text_overlap",
    )
