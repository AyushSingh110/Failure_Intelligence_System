import math
from typing import TypedDict
from app.schemas import FailureSignalVector


# Feature weights — must mirror _extract_features() order exactly.
# Keys document which feature each weight applies to.
_FEATURE_WEIGHTS: dict[str, float] = {
    "ensemble_disagreement": 3.0,   # HIGH  — direct model conflict signal
    "high_failure_risk":     3.0,   # HIGH  — binary risk flag
    "entropy_score":         2.0,   # MED   — output instability
    "fsd_score":             2.0,   # MED   — first-second answer gap
    "agreement_score":       1.5,   # MED-LOW — correlated with entropy
    "ensemble_similarity":   1.0,   # LOW   — already captured by disagreement
    "latency_ms_norm":       0.5,   # LOW   — infrastructure noise
}

_WEIGHT_VECTOR: list[float] = list(_FEATURE_WEIGHTS.values())
_MAX_WEIGHTED_DISTANCE: float = math.sqrt(sum(w for w in _WEIGHT_VECTOR))


class WeightedSimilarityResult(TypedDict):
    similarity_score: float
    weighted_distance: float
    dominant_feature: str


def _extract_features(signal: FailureSignalVector) -> list[float]:
    """
    Extracts a normalized feature vector from a FailureSignalVector.
    Order must be identical to _WEIGHT_VECTOR.
    latency_ms is excluded from the schema — normalized to 0.0 when absent.
    """
    return [
        1.0 if signal.ensemble_disagreement else 0.0,
        1.0 if signal.high_failure_risk else 0.0,
        signal.entropy_score,
        signal.fsd_score,
        signal.agreement_score,
        signal.ensemble_similarity,
        0.0,   # latency_ms_norm — populated by weighted_distance() when available
    ]


def weighted_distance(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    """
    Computes normalized weighted distance between two raw signal dicts.
    Accepts plain dicts (e.g. from API JSON) for flexibility.

    Returns a float in [0.0, 1.0] where:
      0.0 = identical signals
      1.0 = maximally different signals
    """
    feature_keys = list(_FEATURE_WEIGHTS.keys())
    weights      = _WEIGHT_VECTOR

    total = 0.0
    for key, weight in zip(feature_keys, weights):
        val_a = float(a.get(key, 0.0))
        val_b = float(b.get(key, 0.0))
        total += weight * (val_a - val_b) ** 2

    raw_distance = math.sqrt(total)
    return round(min(raw_distance / _MAX_WEIGHTED_DISTANCE, 1.0), 4)


def compute_signal_similarity(
    signal_a: FailureSignalVector,
    signal_b: FailureSignalVector,
) -> float:
    """
    Returns a similarity score in [0.0, 1.0] between two FailureSignalVectors.
    Used by clustering.py to decide whether signals belong to the same archetype.
    """
    vec_a = _extract_features(signal_a)
    vec_b = _extract_features(signal_b)

    total = sum(
        w * (a - b) ** 2
        for w, a, b in zip(_WEIGHT_VECTOR, vec_a, vec_b)
    )
    raw_distance = math.sqrt(total)
    normalized   = raw_distance / _MAX_WEIGHTED_DISTANCE
    return round(max(0.0, 1.0 - normalized), 4)


def compute_signal_similarity_detailed(
    signal_a: FailureSignalVector,
    signal_b: FailureSignalVector,
) -> WeightedSimilarityResult:
    """
    Extended version that also returns the feature that contributed most
    to the distance — useful for explaining WHY two signals differ.
    """
    vec_a    = _extract_features(signal_a)
    vec_b    = _extract_features(signal_b)
    keys     = list(_FEATURE_WEIGHTS.keys())

    contributions = {
        key: w * (a - b) ** 2
        for key, w, a, b in zip(keys, _WEIGHT_VECTOR, vec_a, vec_b)
    }
    dominant_feature = max(contributions, key=lambda k: contributions[k])

    total        = math.sqrt(sum(contributions.values()))
    normalized   = total / _MAX_WEIGHTED_DISTANCE
    similarity   = round(max(0.0, 1.0 - normalized), 4)

    return WeightedSimilarityResult(
        similarity_score=similarity,
        weighted_distance=round(normalized, 4),
        dominant_feature=dominant_feature,
    )