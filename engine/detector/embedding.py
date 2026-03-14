"""
engine/detector/embedding.py

Semantic similarity between two text outputs.

Implementation: Sentence-Transformer Embeddings (Phase 2 upgrade)
------------------------------------------------------------------
Uses the shared SentenceEncoder (engine/encoder.py) which loads
all-MiniLM-L6-v2 lazily on first call and caches it as a singleton.

The model is L2-normalised, so cosine similarity = dot product of the
two 384-dim vectors. This correctly handles:
  - Synonyms:    "happy" vs "joyful"           → high similarity
  - Paraphrases: "Paris is the capital" vs
                 "The capital city is Paris"    → high similarity
  - Contradictions: "Paris" vs "Berlin"        → low similarity

Fallback — n-gram cosine similarity
-------------------------------------
If sentence-transformers is not installed OR the encoder failed to load,
the function automatically falls back to character trigram cosine
similarity (the original Phase 1 implementation). This means the file
works correctly in every environment:

  sentence-transformers installed  →  transformer path  (better)
  sentence-transformers missing    →  n-gram path       (legacy, still works)

The fallback is logged at WARNING level so it is visible in production
but never crashes the system.

Public API — unchanged from Phase 1
-------------------------------------
  compute_embedding_distance(text_a, text_b) -> EmbeddingResult
    EmbeddingResult["embedding_distance"]   float  0.0 (identical) → 1.0 (opposite)
    EmbeddingResult["semantic_similarity"]  float  1.0 (identical) → 0.0 (opposite)

No changes needed in routes.py, agents, or tests.

Config flags (config.py)
--------------------------
  embedding_use_transformer: bool = True   # set False to force n-gram path
  embedding_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
  embedding_ngram_size: int = 3            # used by n-gram fallback only
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import TypedDict

import numpy as np

from config import get_settings
from engine.encoder import get_encoder

logger = logging.getLogger(__name__)


# ── Public return type (unchanged from Phase 1) ────────────────────────────

class EmbeddingResult(TypedDict):
    embedding_distance:  float
    semantic_similarity: float


# ── LLM prefix stripper (kept from Phase 1) ───────────────────────────────
# Still useful in the n-gram fallback path.

_ANSWER_PREFIX_PATTERN = re.compile(
    r"^(the answer is|result:|answer:|output:|response:|therefore[,:]?|so[,:]?|"
    r"in conclusion[,:]?|to summarize[,:]?|finally[,:]?)\s*",
    flags=re.IGNORECASE,
)


def _strip_llm_prefix(text: str) -> str:
    return _ANSWER_PREFIX_PATTERN.sub("", text.strip()).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Path A — Sentence-Transformer (primary)
# ══════════════════════════════════════════════════════════════════════════════

def _transformer_similarity(text_a: str, text_b: str) -> float | None:
    """
    Encodes both texts with the shared SentenceEncoder and returns
    cosine similarity in [-1, 1] (clipped to [0, 1] for distance math).

    Returns None if the encoder is unavailable — caller falls back to n-gram.
    """
    encoder = get_encoder()

    try:
        # encode_batch is more efficient than two encode() calls —
        # the model processes both texts in one forward pass
        vecs = encoder.encode_batch([text_a, text_b])   # shape (2, 384), L2-normalised

        # Force lazy-load attempt first; if unavailable, caller should fallback.
        if not encoder.available:
            return None

        # L2-normalised → dot product = cosine similarity
        similarity = float(np.dot(vecs[0], vecs[1]))

        # Numerical safety clip: fp32 dot product can land just outside [-1, 1]
        return max(0.0, min(1.0, similarity))

    except Exception as exc:
        logger.warning("Transformer similarity failed, falling back to n-gram: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Path B — Character n-gram cosine similarity (fallback, original Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

def _build_ngram_vector(text: str, n: int | None = None) -> dict[str, float]:
    cfg = get_settings()
    n   = n or cfg.embedding_ngram_size
    normalized = _strip_llm_prefix(text).lower()
    if len(normalized) < n:
        return {normalized: 1.0} if normalized else {}
    ngrams = [normalized[i: i + n] for i in range(len(normalized) - n + 1)]
    counts = Counter(ngrams)
    total  = sum(counts.values())
    return {gram: count / total for gram, count in counts.items()}


def _ngram_cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared = set(vec_a.keys()) & set(vec_b.keys())
    dot    = sum(vec_a[k] * vec_b[k] for k in shared)
    mag_a  = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b  = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _ngram_similarity(text_a: str, text_b: str) -> float:
    vec_a = _build_ngram_vector(text_a)
    vec_b = _build_ngram_vector(text_b)
    return _ngram_cosine(vec_a, vec_b)


# ══════════════════════════════════════════════════════════════════════════════
# Public API — identical signature and return type as Phase 1
# ══════════════════════════════════════════════════════════════════════════════

def compute_embedding_distance(text_a: str, text_b: str) -> EmbeddingResult:
    """
    Computes semantic similarity and distance between two text outputs.

    Strategy (selected automatically at runtime):
      1. Transformer path  — if sentence-transformers is installed and
                             embedding_use_transformer=True in config
      2. N-gram fallback   — if transformer unavailable or disabled

    Parameters
    ----------
    text_a : str   Primary model output (or any text A)
    text_b : str   Secondary model output (or any text B)

    Returns
    -------
    EmbeddingResult with:
      embedding_distance   : 1 - similarity  (0.0 = identical, 1.0 = opposite)
      semantic_similarity  : cosine similarity (0.0 to 1.0)
    """
    cfg = get_settings()

    # ── Edge cases (same as Phase 1) ──────────────────────────────────
    if not text_a.strip() and not text_b.strip():
        return EmbeddingResult(embedding_distance=0.0, semantic_similarity=1.0)

    if not text_a.strip() or not text_b.strip():
        return EmbeddingResult(embedding_distance=1.0, semantic_similarity=0.0)

    # ── Choose path ────────────────────────────────────────────────────
    similarity: float | None = None

    if cfg.embedding_use_transformer:
        similarity = _transformer_similarity(text_a, text_b)
        if similarity is None:
            logger.warning(
                "SentenceEncoder unavailable. "
                "Falling back to n-gram similarity for this call. "
                "Install sentence-transformers: pip install sentence-transformers"
            )

    if similarity is None:
        # embedding_use_transformer=False in config, or transformer path failed
        similarity = _ngram_similarity(text_a, text_b)
        logger.debug("embedding.py: using n-gram path (similarity=%.4f)", similarity)
    else:
        logger.debug("embedding.py: using transformer path (similarity=%.4f)", similarity)

    return EmbeddingResult(
        embedding_distance=round(1.0 - similarity, 4),
        semantic_similarity=round(similarity, 4),
    )