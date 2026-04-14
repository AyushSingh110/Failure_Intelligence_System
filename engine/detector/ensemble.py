import math
from collections import Counter
from itertools import combinations
from typing import TypedDict

from config import get_settings

settings = get_settings()

# Stop words: grammatical fillers + question-framing words that appear
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "not",
    "with", "this", "that", "it", "its", "i", "we", "they", "you",
    "what", "which", "where", "who", "how", "when",
    "capital", "country", "city", "place", "answer", "question",
    "correct", "final", "result", "output", "response",
})


class EnsembleResult(TypedDict):
    disagreement:     bool
    similarity_score: float   # mean pairwise similarity across all model pairs
    pair_similarities: list[float]  # individual pairwise scores (for evidence)
    n_pairs:          int     # total pairs evaluated


def _tokenize(text: str) -> list[str]:
    return text.strip().lower().split()


def _content_tokens(text: str) -> list[str]:
    """
    Returns tokens with stop words removed.
    """
    all_tokens = _tokenize(text)
    content    = [t for t in all_tokens if t not in _STOP_WORDS]
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


def _pair_similarity(text_a: str, text_b: str) -> float:
    """
    Computes semantic similarity between two outputs.
    """
    import numpy as np
    from engine.detector.consistency import SHORT_ANSWER_THRESHOLD, SEMANTIC_SIMILARITY_THRESHOLD

    is_long = (
        len(text_a.strip()) >= SHORT_ANSWER_THRESHOLD or
        len(text_b.strip()) >= SHORT_ANSWER_THRESHOLD
    )

    if is_long:
        try:
            from engine.encoder import get_encoder
            encoder = get_encoder()
            if encoder.available:
                vecs = encoder.encode_batch([text_a.strip(), text_b.strip()])
                sim  = float(np.dot(vecs[0], vecs[1]))
                sim  = max(0.0, min(1.0, sim))

                # If the clustering algorithm would group these together
                # (sim >= SEMANTIC_SIMILARITY_THRESHOLD), treat as full agreement.
                # This prevents short/long paraphrase pairs from triggering
                # false ensemble disagreement.
                if sim >= SEMANTIC_SIMILARITY_THRESHOLD:
                    return 1.0

                return sim
        except Exception:
            pass   # fall through to TF cosine

    # Short answers or encoder unavailable → TF cosine
    tf_a = _build_term_frequency(_content_tokens(text_a))
    tf_b = _build_term_frequency(_content_tokens(text_b))
    return _cosine_similarity(tf_a, tf_b)

# Primary API — all model outputs

def compute_disagreement(
    model_outputs: list[str] | str,
    disagreement_threshold: float | str | None = None,
) -> EnsembleResult:
    """
    Computes ensemble disagreement across ALL provided model outputs.
    """
    if isinstance(model_outputs, str):
        secondary_output = (
            disagreement_threshold
            if isinstance(disagreement_threshold, str)
            else ""
        )
        if not model_outputs.strip() and secondary_output.strip():
            return EnsembleResult(
                disagreement=True,
                similarity_score=0.0,
                pair_similarities=[0.0],
                n_pairs=1,
            )
        if not model_outputs.strip() and not secondary_output.strip():
            return EnsembleResult(
                disagreement=False,
                similarity_score=1.0,
                pair_similarities=[],
                n_pairs=0,
            )
        model_outputs = [model_outputs, secondary_output]
        disagreement_threshold = None

    threshold = (
        float(disagreement_threshold)
        if disagreement_threshold is not None
        else settings.ensemble_disagreement_threshold
    )

    # Filter out blank outputs — a model that returned nothing is not useful
    valid_outputs = [o for o in model_outputs if o.strip()]

    # Edge cases 
    if len(valid_outputs) == 0:
        return EnsembleResult(
            disagreement=False,
            similarity_score=1.0,
            pair_similarities=[],
            n_pairs=0,
        )

    if len(valid_outputs) == 1:
        # Only one output — cannot compute disagreement
        return EnsembleResult(
            disagreement=False,
            similarity_score=1.0,
            pair_similarities=[],
            n_pairs=0,
        )

    # All pairwise similarities 
    pair_scores: list[float] = []

    for out_a, out_b in combinations(valid_outputs, 2):
        score = _pair_similarity(out_a, out_b)
        pair_scores.append(round(score, 4))

    mean_similarity = round(sum(pair_scores) / len(pair_scores), 4)

    # disagreement = True if ANY pair diverged below threshold
    any_disagreement = any(s < threshold for s in pair_scores)

    return EnsembleResult(
        disagreement=any_disagreement,
        similarity_score=mean_similarity,
        pair_similarities=pair_scores,
        n_pairs=len(pair_scores),
    )

# Legacy API — kept for backward compatibility

def compute_disagreement_pair(
    primary_output:         str,
    secondary_output:       str,
    disagreement_threshold: float | None = None,
) -> EnsembleResult:
    """
    Original two-output form. Wraps compute_disagreement([primary, secondary]).
    Kept for any internal callers that still pass exactly two strings.
    """
    return compute_disagreement(
        [primary_output, secondary_output],
        disagreement_threshold=disagreement_threshold,
    )
