"""
embedding.py

Semantic similarity between two text outputs.

Current Implementation: Character N-gram Cosine Similarity
-----------------------------------------------------------
Uses overlapping trigrams to build a sparse TF vector per string.
Works well for lexical similarity ("Paris" vs "paris") but cannot
bridge synonyms ("happy" vs "joyful") or paraphrases.

Phase 2 Upgrade Path: Sentence-Transformer Embeddings
-------------------------------------------------------
Swap _build_ngram_vector() for a real encoder like:
  from sentence_transformers import SentenceTransformer
  _model = SentenceTransformer("all-MiniLM-L6-v2")

  def _encode(text: str) -> list[float]:
      return _model.encode(text).tolist()

Then replace the dict-based cosine with a vector version.
The rest of the public API (compute_embedding_distance) stays identical —
no changes needed in routes, agents, or tests.

Known Limitation (current phase):
  "The answer is Paris" and "Paris" will score ~0.55, not 1.0,
  because trigrams differ. This is addressed in consistency.py
  via the LLM prefix stripper.
"""

import math
import re
from collections import Counter
from typing import TypedDict

from config import get_settings

settings = get_settings()

# Patterns LLMs prepend before the actual answer.
# Stripping these before building vectors reduces false disagreement.
_ANSWER_PREFIX_PATTERN = re.compile(
    r"^(the answer is|result:|answer:|output:|response:|therefore[,:]?|so[,:]?|"
    r"in conclusion[,:]?|to summarize[,:]?|finally[,:]?)\s*",
    flags=re.IGNORECASE,
)


class EmbeddingResult(TypedDict):
    embedding_distance: float
    semantic_similarity: float


def _strip_llm_prefix(text: str) -> str:
    return _ANSWER_PREFIX_PATTERN.sub("", text.strip()).strip()


def _build_ngram_vector(text: str, n: int | None = None) -> dict[str, float]:
    n = n or settings.embedding_ngram_size
    normalized = _strip_llm_prefix(text).lower()
    if len(normalized) < n:
        return {normalized: 1.0} if normalized else {}
    ngrams = [normalized[i : i + n] for i in range(len(normalized) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    return {gram: count / total for gram, count in counts.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in shared)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def compute_embedding_distance(text_a: str, text_b: str) -> EmbeddingResult:
    if not text_a.strip() and not text_b.strip():
        return EmbeddingResult(embedding_distance=0.0, semantic_similarity=1.0)

    if not text_a.strip() or not text_b.strip():
        return EmbeddingResult(embedding_distance=1.0, semantic_similarity=0.0)

    vec_a = _build_ngram_vector(text_a)
    vec_b = _build_ngram_vector(text_b)

    similarity = _cosine_similarity(vec_a, vec_b)
    distance = 1.0 - similarity

    return EmbeddingResult(
        embedding_distance=round(distance, 4),
        semantic_similarity=round(similarity, 4),
    )
