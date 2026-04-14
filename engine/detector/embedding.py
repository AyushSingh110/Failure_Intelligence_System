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


# Public return type 

class EmbeddingResult(TypedDict):
    embedding_distance:  float
    semantic_similarity: float


# LLM prefix stripper

_ANSWER_PREFIX_PATTERN = re.compile(
    r"^(the answer is|result:|answer:|output:|response:|therefore[,:]?|so[,:]?|"
    r"in conclusion[,:]?|to summarize[,:]?|finally[,:]?)\s*",
    flags=re.IGNORECASE,
)


def _strip_llm_prefix(text: str) -> str:
    return _ANSWER_PREFIX_PATTERN.sub("", text.strip()).strip()



# Sentence-Transformer 
def _transformer_similarity(text_a: str, text_b: str) -> float | None:
    """
    Encodes both texts with the shared SentenceEncoder and returns
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



# Character n-gram cosine similarity (fallback)


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



# Public API — identical signature and return type 

def compute_embedding_distance(text_a: str, text_b: str) -> EmbeddingResult:
    """
    Computes semantic similarity and distance between two text outputs.
    """
    cfg = get_settings()

    #  Edge cases  
    if not text_a.strip() and not text_b.strip():
        return EmbeddingResult(embedding_distance=0.0, semantic_similarity=1.0)

    if not text_a.strip() or not text_b.strip():
        return EmbeddingResult(embedding_distance=1.0, semantic_similarity=0.0)

    # ── Choose path 
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