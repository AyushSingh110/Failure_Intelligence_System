from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.72
SHORT_ANSWER_THRESHOLD:        int   = 40
KEYWORD_THRESHOLD:             int   = 10  # outputs shorter than this = keyword answers

# Sentence boundary pattern
_SENTENCE_END = re.compile(r'[.!?](?:\s|$)')


# ── Public return type ────────────────────────────────────────────────────
class ConsistencyResult(TypedDict):
    agreement_score: float
    fsd_score:       float
    answer_counts:   dict[str, int]


# ── LLM prefix / suffix normalization ────────────────────────────────────
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


# ── Encoding representative ───────────────────────────────────────────────
def _encoding_repr(text: str) -> str:
    """
    Returns the text used for semantic encoding.

    Short outputs (< SHORT_ANSWER_THRESHOLD): encode as-is.
    Long outputs: encode only the first sentence.

    Why: sentence transformers underestimate similarity between short keyword
    answers ("Paris") and long elaborated answers ("The capital of France is
    Paris. It's one of the most famous...") because extra content after the
    first sentence dilutes the core answer vector. Using first sentence
    ensures long outputs are compared on their actual answer content.
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

    This handles the case where a user's primary model returns a bare keyword
    ("Paris") while shadow models return full sentences ("The capital of France
    is Paris."). The encoder cannot reliably compare these due to embedding
    space asymmetry between keywords and propositions.

    Only applies when short_text is genuinely a keyword (< KEYWORD_THRESHOLD).
    Uses word-boundary regex to prevent partial matches:
      'paris' matches 'the capital of france is paris' ✓
      'paris' does NOT match 'comparison of france' ✓
      'ran' does NOT match 'france' ✓
    """
    if len(short_text) >= KEYWORD_THRESHOLD:
        return False
    pattern = r'\b' + re.escape(short_text) + r'\b'
    return bool(re.search(pattern, other_text, re.IGNORECASE))


# ── Semantic clustering ───────────────────────────────────────────────────
def _semantic_cluster(
    normalized_outputs: list[str],
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> dict[str, int] | None:
    """
    Groups outputs into semantic clusters using a two-rule approach:

    Rule 1 — Keyword substring: if one output is a short keyword (< 10 chars)
    and it appears as a whole word in the cluster label → merge.
    This handles: "Paris" vs "The capital of France is Paris."

    Rule 2 — Cosine similarity on encoding representatives: encode the first
    sentence of each output and compare. If similarity >= threshold → merge.
    This handles: two long paraphrases of the same answer.

    Full text is preserved as the cluster label in answer_counts.
    Only encoding representations are used for similarity computation.
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


# ── Exact string matching (fallback) ─────────────────────────────────────
def _exact_cluster(normalized_outputs: list[str]) -> dict[str, int]:
    # Truncate keys to 300 chars to prevent MongoDB field length issues
    truncated = [o[:300] for o in normalized_outputs]
    return dict(Counter(truncated))


# ── Public API ────────────────────────────────────────────────────────────
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
    # Previously this only ran for long-form outputs (>= 40 chars) but that
    # missed the case where "paris" (5 chars) and "the capital of france is
    # paris" (30 chars) are both short yet semantically identical.
    # The keyword check inside _semantic_cluster handles this correctly.
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