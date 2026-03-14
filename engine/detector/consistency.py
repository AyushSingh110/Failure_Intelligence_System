"""
engine/detector/consistency.py

Computes sample agreement, FSD score, and answer distribution across
multiple sampled model outputs.

The Problem with Exact String Matching
----------------------------------------
The original implementation counted answers using exact string matching
after prefix stripping. This works perfectly for short factual answers:

    "Paris", "Paris", "Paris"  →  agreement = 1.0  ✓

But completely breaks for long-form paraphrases:

    "Quantum entanglement is a phenomenon where two particles become
     correlated so that measuring one instantly determines the state
     of the other, regardless of distance."

    "Quantum entanglement occurs when two particles share a quantum
     state, meaning the measurement of one instantaneously affects
     the other no matter how far apart they are."

Both sentences mean exactly the same thing. Exact match counts them
as two different answers → agreement=0.2, entropy=1.0 → FALSE POSITIVE
HALLUCINATION_RISK. The system was misdiagnosing correct behaviour.

The Fix: Two-Path Consistency
-------------------------------
Path A — Semantic clustering (long-form outputs):
  When at least one output exceeds SHORT_ANSWER_THRESHOLD (40 chars)
  AND sentence-transformers is installed:

    1. Encode all outputs to 384-dim L2-normalised vectors via the
       shared SentenceEncoder singleton (engine/encoder.py).
    2. Greedily cluster outputs: two outputs are "the same answer"
       if their cosine similarity ≥ SEMANTIC_SIMILARITY_THRESHOLD (0.82).
    3. agreement_score = largest cluster size / total outputs.
    4. answer_counts   = { representative_text : cluster_size }.

  Cluster centroids are updated as a running mean and re-normalised
  after each assignment so cosine similarity stays valid.

Path B — Exact string matching (short answers + fallback):
  Used when:
    - All outputs are short (≤ 40 chars) — "Paris", "1945", "Au"
    - sentence-transformers is not installed
    - The encoder raised an exception

  Exact match is actually more accurate than semantic similarity for
  single-word or single-number answers where paraphrasing is impossible.

Public API — completely unchanged
-----------------------------------
  compute_consistency(model_outputs: list[str]) -> ConsistencyResult
    ConsistencyResult["agreement_score"]  float  [0, 1]
    ConsistencyResult["fsd_score"]        float  [0, 1]
    ConsistencyResult["answer_counts"]    dict[str, int]

No changes needed anywhere else in the codebase.

Threshold tuning (no config change required — constants defined here):
  SEMANTIC_SIMILARITY_THRESHOLD = 0.82
    Two outputs with cosine similarity ≥ 0.82 are treated as the same
    answer. Lower this to merge more aggressively. Raise it to be stricter.

  SHORT_ANSWER_THRESHOLD = 40
    Outputs shorter than 40 chars always use exact matching.
    "Paris" and "paris" match after _normalize(). "1945" always exact.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────

# Cosine similarity at or above this → two outputs are the "same answer"
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.82

# Outputs shorter than this always use exact matching (faster + more accurate)
SHORT_ANSWER_THRESHOLD: int = 40


# ── Public return type (unchanged from original) ──────────────────────────

class ConsistencyResult(TypedDict):
    agreement_score: float
    fsd_score:       float
    answer_counts:   dict[str, int]


# ── LLM prefix / suffix normalization (unchanged from original) ───────────

_PREFIX_PASS_1 = re.compile(
    r"^("
    r"the (answer|result|correct answer|final answer) is[:\s]?|"
    r"answer\s*:|result\s*:|output\s*:|response\s*:|"
    r"therefore[,:]?\s*|thus[,:]?\s*|hence[,:]?\s*|"
    r"in (conclusion|summary)[,:]?\s*|to summarize[,:]?\s*|"
    r"in other words[,:]?\s*|that is[,:]?\s*|"
    r"so[,:]?\s*|finally[,:]?\s*|"
    r"based on (the|this|my|above|that)[^,]*[,:]?\s*|"
    r"given (that|the|this)[^,]*[,:]?\s*"
    r")",
    flags=re.IGNORECASE,
)

_PREFIX_PASS_2 = re.compile(
    r"^("
    r"yes[,:]?\s*|no[,:]?\s*|"
    r"\d+[.)]\s*|[a-d][.)]\s*|\([a-d]\)\s*"
    r")",
    flags=re.IGNORECASE,
)

_TRAILING_PATTERN = re.compile(r"[\s.,!?;:)]+$")


def _normalize(text: str) -> str:
    """
    Multi-pass normalization — strips LLM reasoning prefixes and
    trailing noise.  Each pass runs up to 3 times to handle chains.
    """
    result = text.strip()
    for _ in range(3):
        before = result
        result = _PREFIX_PASS_1.sub("", result).strip()
        result = _PREFIX_PASS_2.sub("", result).strip()
        if result == before:
            break
    result = _TRAILING_PATTERN.sub("", result)
    return result.lower().strip()


# ══════════════════════════════════════════════════════════════════════════════
# Path A — Semantic clustering
# ══════════════════════════════════════════════════════════════════════════════

def _semantic_cluster(
    normalized_outputs: list[str],
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> dict[str, int] | None:
    """
    Groups outputs into semantic clusters using cosine similarity.

    Returns dict { representative_text: cluster_size }
    Returns None if the encoder is unavailable → caller falls back to exact match.

    Algorithm:
      For each output vector, find the first existing cluster whose
      centroid has cosine similarity ≥ threshold. If none found, start
      a new cluster. Update centroids as running means (re-normalised).
    """
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
    except Exception as exc:
        logger.warning("Could not import encoder: %s", exc)
        return None

    if not encoder.available:
        return None

    try:
        vecs = encoder.encode_batch(normalized_outputs)  # (N, 384), L2-norm
    except Exception as exc:
        logger.warning("Semantic clustering encoding failed: %s", exc)
        return None

    n = len(normalized_outputs)

    cluster_centroids: list[np.ndarray] = []
    cluster_labels:    list[str]        = []   # representative text per cluster
    cluster_sizes:     list[int]        = []

    for i in range(n):
        vec      = vecs[i]
        best_c   = -1
        best_sim = threshold - 1e-6   # must strictly exceed threshold

        for c_idx, centroid in enumerate(cluster_centroids):
            # L2-normalised vectors → dot product = cosine similarity
            sim = float(np.dot(vec, centroid))
            if sim > best_sim:
                best_sim = sim
                best_c   = c_idx

        if best_c == -1:
            # No close cluster → start a new one
            cluster_centroids.append(vec.copy())
            cluster_labels.append(normalized_outputs[i])
            cluster_sizes.append(1)
        else:
            # Assign to best cluster, update centroid as running mean
            old_n    = cluster_sizes[best_c]
            new_n    = old_n + 1
            new_cent = (cluster_centroids[best_c] * old_n + vec) / new_n
            norm     = np.linalg.norm(new_cent)
            if norm > 1e-8:
                new_cent = new_cent / norm
            cluster_centroids[best_c] = new_cent
            cluster_sizes[best_c]     = new_n

    return {
        label: size
        for label, size in zip(cluster_labels, cluster_sizes)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Path B — Exact string matching (original logic, used as fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _exact_cluster(normalized_outputs: list[str]) -> dict[str, int]:
    """Original exact-match counting. Accurate for short answers."""
    return dict(Counter(normalized_outputs))


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def compute_consistency(model_outputs: list[str]) -> ConsistencyResult:
    """
    Computes agreement, FSD, and answer distribution across model outputs.

    Automatically selects semantic clustering for long-form outputs and
    exact matching for short answers or when the encoder is unavailable.

    agreement_score : fraction of outputs in the largest semantic cluster.
    fsd_score       : (top_count - second_count) / total.
    answer_counts   : { representative_text : cluster_size }
    """
    # ── Edge cases ─────────────────────────────────────────────────────
    if not model_outputs:
        return ConsistencyResult(
            agreement_score=0.0,
            fsd_score=0.0,
            answer_counts={},
        )

    if len(model_outputs) == 1:
        normalized = _normalize(model_outputs[0])
        return ConsistencyResult(
            agreement_score=1.0,
            fsd_score=0.0,
            answer_counts={normalized: 1},
        )

    normalized_outputs = [_normalize(o) for o in model_outputs]
    total_samples      = len(normalized_outputs)

    # ── Choose path ────────────────────────────────────────────────────
    # Use semantic clustering only when at least one output is long-form.
    # Short answers ("Paris", "1945") never need semantic grouping —
    # exact match is both faster and more accurate for single tokens.
    is_long_form = any(len(o) >= SHORT_ANSWER_THRESHOLD for o in normalized_outputs)

    answer_counts: dict[str, int] | None = None

    if is_long_form:
        answer_counts = _semantic_cluster(normalized_outputs)
        if answer_counts is not None:
            logger.debug(
                "consistency.py: semantic path → %d cluster(s) from %d outputs",
                len(answer_counts), total_samples,
            )
        else:
            logger.warning(
                "consistency.py: semantic encoder unavailable, "
                "falling back to exact matching for long-form outputs. "
                "Install sentence-transformers for accurate paraphrase detection."
            )

    if answer_counts is None:
        # Short answers OR semantic path unavailable → exact match
        answer_counts = _exact_cluster(normalized_outputs)
        logger.debug(
            "consistency.py: exact-match path → %d unique answer(s) from %d outputs",
            len(answer_counts), total_samples,
        )

    # ── Compute scores from answer_counts ──────────────────────────────
    sorted_counts  = sorted(answer_counts.values(), reverse=True)
    top_count      = sorted_counts[0]
    second_count   = sorted_counts[1] if len(sorted_counts) > 1 else 0

    agreement_score = top_count / total_samples
    fsd_score       = (top_count - second_count) / total_samples

    return ConsistencyResult(
        agreement_score=round(agreement_score, 4),
        fsd_score=round(fsd_score, 4),
        answer_counts=answer_counts,
    )