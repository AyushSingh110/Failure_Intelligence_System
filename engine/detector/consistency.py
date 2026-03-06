import re
from collections import Counter
from typing import TypedDict


class ConsistencyResult(TypedDict):
    agreement_score: float
    fsd_score: float
    answer_counts: dict[str, int]


# ── Regex patterns applied in order via _normalize() ─────────────────────

# Pass 1: Remove LLM reasoning closers and answer introducers
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

# Pass 2: Remove soft affirmatives and ordinal prefixes
_PREFIX_PASS_2 = re.compile(
    r"^("
    r"yes[,:]?\s*|no[,:]?\s*|"
    r"\d+[.)]\s*|[a-d][.)]\s*|\([a-d]\)\s*"
    r")",
    flags=re.IGNORECASE,
)

# Trailing noise: punctuation, whitespace, parenthetical notes
_TRAILING_PATTERN = re.compile(r"[\s.,!?;:)]+$")


def _normalize(text: str) -> str:
    """
    Multi-pass normalization.
    Each pass is applied up to 3 times to handle chained prefixes.
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


def compute_consistency(model_outputs: list[str]) -> ConsistencyResult:
    """
    Computes agreement, FSD, and answer distribution across sampled outputs.

    agreement_score: fraction of samples matching the plurality answer.
    fsd_score:       (top_count - second_count) / total — measures dominance
                     of the top answer. High FSD = confident plurality.
                     Low FSD = close race between two answers = ambiguity.
    """
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

    normalized_outputs    = [_normalize(o) for o in model_outputs]
    total_samples         = len(normalized_outputs)
    answer_counts: dict[str, int] = dict(Counter(normalized_outputs))
    sorted_counts         = sorted(answer_counts.values(), reverse=True)

    top_count      = sorted_counts[0]
    second_count   = sorted_counts[1] if len(sorted_counts) > 1 else 0
    agreement_score = top_count / total_samples
    fsd_score       = (top_count - second_count) / total_samples

    return ConsistencyResult(
        agreement_score=round(agreement_score, 4),
        fsd_score=round(fsd_score, 4),
        answer_counts=answer_counts,
    )
