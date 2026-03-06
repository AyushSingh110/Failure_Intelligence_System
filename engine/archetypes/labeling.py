from typing import TypedDict
from app.schemas import FailureSignalVector
from config import get_settings

settings = get_settings()


# ── Archetype label constants ─────────────────────────────────────────────
ARCHETYPE_HALLUCINATION_RISK    = "HALLUCINATION_RISK"
ARCHETYPE_OVERCONFIDENT_FAILURE = "OVERCONFIDENT_FAILURE"
ARCHETYPE_BLIND_SPOT            = "MODEL_BLIND_SPOT"
ARCHETYPE_RESOURCE_CONSTRAINT   = "RESOURCE_CONSTRAINT"
ARCHETYPE_UNSTABLE_OUTPUT       = "UNSTABLE_OUTPUT"
ARCHETYPE_LOW_CONFIDENCE        = "LOW_CONFIDENCE"
ARCHETYPE_STABLE                = "STABLE"

# Latency above this (ms) is considered elevated for RESOURCE_CONSTRAINT detection
_HIGH_LATENCY_MS = 3000.0

# Low entropy ceiling: below this = model is highly confident in its answer
_LOW_ENTROPY_CEILING = 0.25


class LabelResult(TypedDict):
    archetype: str
    confidence: str       # HIGH | MEDIUM | LOW — how many conditions fired
    conditions_met: list[str]


def assign_failure_label(signal_vector: dict) -> str:
    """
    Accepts a raw signal dict (from API / JSON).
    Returns a single taxonomy label string.
    """
    entropy      = float(signal_vector.get("entropy_score",         0.0))
    agreement    = float(signal_vector.get("agreement_score",       1.0))
    disagreement = bool(signal_vector.get("ensemble_disagreement",  False))
    risk         = bool(signal_vector.get("high_failure_risk",      False))
    latency      = float(signal_vector.get("latency_ms",            0.0))

    if disagreement and entropy >= settings.high_entropy_threshold:
        return ARCHETYPE_HALLUCINATION_RISK

    if risk and entropy < _LOW_ENTROPY_CEILING:
        return ARCHETYPE_OVERCONFIDENT_FAILURE

    if disagreement:
        return ARCHETYPE_BLIND_SPOT

    if latency >= _HIGH_LATENCY_MS and entropy >= settings.high_entropy_threshold:
        return ARCHETYPE_RESOURCE_CONSTRAINT

    if entropy >= settings.high_entropy_threshold:
        return ARCHETYPE_UNSTABLE_OUTPUT

    if agreement <= settings.low_agreement_threshold:
        return ARCHETYPE_LOW_CONFIDENCE

    return ARCHETYPE_STABLE


def label_failure_archetype(signal: FailureSignalVector) -> str:
    """
    Accepts a typed FailureSignalVector.
    Thin wrapper around assign_failure_label() for internal engine use.
    """
    return assign_failure_label({
        "entropy_score":        signal.entropy_score,
        "agreement_score":      signal.agreement_score,
        "ensemble_disagreement": signal.ensemble_disagreement,
        "high_failure_risk":    signal.high_failure_risk,
        "ensemble_similarity":  signal.ensemble_similarity,
        "latency_ms":           0.0,   # not present on schema; injected by routes if needed
    })


def label_failure_archetype_detailed(signal: FailureSignalVector) -> LabelResult:
    """
    Returns the archetype plus diagnostic metadata:
    which conditions fired and how confident the label is.
    Used by the /analyze endpoint for richer API responses.
    """
    entropy      = signal.entropy_score
    agreement    = signal.agreement_score
    disagreement = signal.ensemble_disagreement
    risk         = signal.high_failure_risk

    conditions: list[str] = []
    if entropy >= settings.high_entropy_threshold:
        conditions.append(f"entropy={entropy:.3f} ≥ {settings.high_entropy_threshold}")
    if agreement <= settings.low_agreement_threshold:
        conditions.append(f"agreement={agreement:.3f} ≤ {settings.low_agreement_threshold}")
    if disagreement:
        conditions.append("ensemble_disagreement=True")
    if risk:
        conditions.append("high_failure_risk=True")
    if entropy < _LOW_ENTROPY_CEILING and risk:
        conditions.append(f"entropy={entropy:.3f} < {_LOW_ENTROPY_CEILING} (overconfident)")

    archetype  = label_failure_archetype(signal)
    confidence = "HIGH" if len(conditions) >= 3 else ("MEDIUM" if conditions else "LOW")

    return LabelResult(
        archetype=archetype,
        confidence=confidence,
        conditions_met=conditions,
    )


def label_batch(signals: list[FailureSignalVector]) -> list[dict]:
    return [
        {
            "archetype": label_failure_archetype(s),
            "signal":    s.model_dump(),
        }
        for s in signals
    ]


def label_batch_detailed(signals: list[FailureSignalVector]) -> list[dict]:
    return [
        {
            **label_failure_archetype_detailed(s),
            "signal": s.model_dump(),
        }
        for s in signals
    ]
