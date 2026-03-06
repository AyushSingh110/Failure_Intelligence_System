import math
from collections import Counter
from typing import TypedDict

from config import get_settings

settings = get_settings()

# Stop words: grammatical fillers + question-framing words that appear
# symmetrically in both sentences and mask actual answer differences.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "not",
    "with", "this", "that", "it", "its", "i", "we", "they", "you",
    "what", "which", "where", "who", "how", "when",
    "capital", "country", "city", "place", "answer", "question",
    "correct", "final", "result", "output", "response",
})


class EnsembleResult(TypedDict):
    disagreement: bool
    similarity_score: float


def _tokenize(text: str) -> list[str]:
    return text.strip().lower().split()


def _content_tokens(text: str) -> list[str]:
    """
    Returns tokens with stop words removed.
    Falls back to all tokens if every word is a stop word,
    preventing empty-vector errors in cosine computation.
    """
    all_tokens     = _tokenize(text)
    content        = [t for t in all_tokens if t not in _STOP_WORDS]
    return content if content else all_tokens


def _build_term_frequency(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total  = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared      = set(vec_a.keys()) & set(vec_b.keys())
    dot         = sum(vec_a[t] * vec_b[t] for t in shared)
    magnitude_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    magnitude_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot / (magnitude_a * magnitude_b)


def compute_disagreement(
    primary_output:         str,
    secondary_output:       str,
    disagreement_threshold: float | None = None,
) -> EnsembleResult:
    threshold = disagreement_threshold or settings.ensemble_disagreement_threshold

    if not primary_output.strip() and not secondary_output.strip():
        return EnsembleResult(disagreement=False, similarity_score=1.0)

    if not primary_output.strip() or not secondary_output.strip():
        return EnsembleResult(disagreement=True, similarity_score=0.0)

    primary_tf   = _build_term_frequency(_content_tokens(primary_output))
    secondary_tf = _build_term_frequency(_content_tokens(secondary_output))

    similarity_score = _cosine_similarity(primary_tf, secondary_tf)

    return EnsembleResult(
        disagreement=similarity_score < threshold,
        similarity_score=round(similarity_score, 4),
    )
