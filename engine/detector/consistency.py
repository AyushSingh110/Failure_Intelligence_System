from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)

# Tuning constants
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.72
SHORT_ANSWER_THRESHOLD:        int   = 40
KEYWORD_THRESHOLD:             int   = 10  # outputs shorter than this = keyword answers

# Sentence boundary pattern
_SENTENCE_END = re.compile(r'[.!?](?:\s|$)')


#Public return type
class ConsistencyResult(TypedDict):
    agreement_score: float
    fsd_score:       float
    answer_counts:   dict[str, int]


#LLM normalization
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
    result = text.strip()
    for _ in range(3):
        before = result
        result = _PREFIX_PASS_1.sub("", result).strip()
        result = _PREFIX_PASS_2.sub("", result).strip()
        if result == before:
            break
    result = _TRAILING_PATTERN.sub("", result)
    return result.lower().strip()


# Encoding representative 
def _encoding_repr(text: str) -> str:
    """
    Returns the text used for semantic encoding.
    """
    if len(text) < SHORT_ANSWER_THRESHOLD:
        return text
    match = _SENTENCE_END.search(text)
    if match:
        return text[:match.end()].strip()
    return text[:150].strip()


# ── Keyword substring check ───────────────────────────────────────────────
def _keyword_matches(short_text: str, other_text: str) -> bool:
    """
    Returns True if short_text (a single keyword answer) appears as a
    whole-word match inside other_text.
    """
    if len(short_text) >= KEYWORD_THRESHOLD:
        return False
    pattern = r'\b' + re.escape(short_text) + r'\b'
    return bool(re.search(pattern, other_text, re.IGNORECASE))


#  Semantic clustering 
def _semantic_cluster(
    normalized_outputs: list[str],
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> dict[str, int] | None:
    """
    Groups outputs into semantic clusters using a two-rule approach:
    """
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
    except Exception as exc:
        logger.warning("Could not import encoder: %s", exc)
        return None

    if not encoder.available:
        return None

    # Pre-compute encoding representatives (first sentence for long outputs)
    enc_reprs = [_encoding_repr(o) for o in normalized_outputs]

    try:
        vecs = encoder.encode_batch(enc_reprs)
    except Exception as exc:
        logger.warning("Semantic clustering encoding failed: %s", exc)
        return None

    cluster_centroids: list[np.ndarray] = []
    cluster_labels:    list[str]        = []
    cluster_sizes:     list[int]        = []

    for i in range(len(normalized_outputs)):
        text = normalized_outputs[i]
        vec  = vecs[i]

        best_c   = -1
        best_sim = threshold - 1e-6

        for c_idx, label in enumerate(cluster_labels):

            # Rule 1: keyword substring check
            # Handles short keyword ("paris") vs long sentence ("the capital of france is paris.")
            if _keyword_matches(text, label) or _keyword_matches(label, text):
                best_c   = c_idx
                best_sim = 1.0
                break

            # Rule 2: cosine similarity on encoding representatives
            sim = float(np.dot(vec, cluster_centroids[c_idx]))
            if sim > best_sim:
                best_sim = sim
                best_c   = c_idx

        if best_c == -1:
            cluster_centroids.append(vec.copy())
            cluster_labels.append(text)
            cluster_sizes.append(1)
        else:
            old_n    = cluster_sizes[best_c]
            new_n    = old_n + 1
            new_cent = (cluster_centroids[best_c] * old_n + vec) / new_n
            norm     = np.linalg.norm(new_cent)
            if norm > 1e-8:
                new_cent = new_cent / norm
            cluster_centroids[best_c] = new_cent
            cluster_sizes[best_c]     = new_n

    # Truncate cluster labels to 300 chars to prevent MongoDB field issues
    # when jailbreak-obeying models return very long responses
    return {
        label[:300]: size
        for label, size in zip(cluster_labels, cluster_sizes)
    }


#Exact string matching (fallback) 
def _exact_cluster(normalized_outputs: list[str]) -> dict[str, int]:
    # Truncate keys to 300 chars to prevent MongoDB field length issues
    truncated = [o[:300] for o in normalized_outputs]
    return dict(Counter(truncated))


#Primary-outlier check
def is_primary_outlier(primary_output: str, shadow_outputs: list[str]) -> bool:
    """
    Returns True ONLY when the primary output meaningfully disagrees
    with the shadow model majority — meaning the primary is the outlier.

    Returns False when:
      - Fewer than 2 shadow outputs are available (can't establish majority)
      - Shadow models themselves disagree (shadow agreement < 0.60) — the
        question is genuinely ambiguous, not necessarily a primary failure
      - Primary matches the shadow majority semantically
    """
    if not shadow_outputs or len(shadow_outputs) < 2:
        return False

    # Step 1: Compute shadow-only consistency
    shadow_result    = compute_consistency(shadow_outputs)
    shadow_agreement = shadow_result["agreement_score"]

    # Step 2: Only meaningful when shadows mostly agree with each other
    if shadow_agreement < 0.60:
        # Shadows themselves are confused can't reliably identify a majority or outlier
        return False

    # Step 3: Find the majority shadow cluster label
    answer_counts  = shadow_result["answer_counts"]
    majority_label = max(answer_counts, key=answer_counts.get)

    # Step 4: Check if primary semantically matches the shadow majority
    normalized_primary = _normalize(primary_output)

    # Keyword substring check (handles "Paris" vs "The capital of France is Paris")
    if (_keyword_matches(normalized_primary, majority_label)
            or _keyword_matches(majority_label, normalized_primary)):
        return False   # primary agrees — not an outlier

    # Semantic embedding check
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
    except Exception:
        # Encoder unavailable — fall back to exact string check
        return normalized_primary.strip() != majority_label.strip()

    if not encoder.available:
        return normalized_primary.strip() != majority_label.strip()

    primary_repr  = _encoding_repr(normalized_primary)
    majority_repr = _encoding_repr(majority_label)

    try:
        vecs = encoder.encode_batch([primary_repr, majority_repr])
        sim  = float(np.dot(vecs[0], vecs[1]))
        # Primary is an outlier only if it is semantically far from shadow majority
        return sim < SEMANTIC_SIMILARITY_THRESHOLD
    except Exception:
        return normalized_primary.strip() != majority_label.strip()


#Public API 
def compute_consistency(model_outputs: list[str]) -> ConsistencyResult:
    if not model_outputs:
        return ConsistencyResult(agreement_score=0.0, fsd_score=0.0, answer_counts={})

    if len(model_outputs) == 1:
        normalized = _normalize(model_outputs[0])
        return ConsistencyResult(
            agreement_score=1.0,
            fsd_score=0.0,
            answer_counts={normalized: 1},
        )

    normalized_outputs = [_normalize(o) for o in model_outputs]
    total_samples      = len(normalized_outputs)

    # Always run semantic clustering when we have multiple different outputs.
    answer_counts = _semantic_cluster(normalized_outputs)

    if answer_counts is None:
        # Encoder unavailable — fall back to exact string matching
        logger.warning(
            "Semantic encoder unavailable — falling back to exact matching. "
            "Install sentence-transformers for accurate paraphrase detection."
        )
        answer_counts = _exact_cluster(normalized_outputs)

    sorted_counts = sorted(answer_counts.values(), reverse=True)
    top_count     = sorted_counts[0]
    second_count  = sorted_counts[1] if len(sorted_counts) > 1 else 0

    return ConsistencyResult(
        agreement_score=round(top_count / total_samples, 4),
        fsd_score=round((top_count - second_count) / total_samples, 4),
        answer_counts=answer_counts,
    )