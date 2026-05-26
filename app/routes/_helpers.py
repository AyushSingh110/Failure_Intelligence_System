from __future__ import annotations
import logging
from engine.detector.consistency import compute_consistency, is_primary_outlier
from engine.detector.entropy import compute_entropy_from_counts
from engine.detector.ensemble import compute_disagreement
from app.schemas import FailureSignalVector
from config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

def build_failure_signal(model_outputs: list[str]) -> FailureSignalVector:
    #Construct a FailureSignalVector from all model outputs.
    primary_output   = model_outputs[0]
    secondary_output = model_outputs[1] if len(model_outputs) > 1 else model_outputs[0]

    consistency   = compute_consistency(model_outputs)
    entropy_score = compute_entropy_from_counts(
        consistency["answer_counts"],
        len(model_outputs),
    )
    ensemble = compute_disagreement(model_outputs)

    ensemble_fires = (
        ensemble["disagreement"] is True
        and entropy_score > 0.0
    )

    shadow_outputs  = model_outputs[1:]
    primary_outlier = is_primary_outlier(primary_output, shadow_outputs)

    high_failure_risk = (
        primary_outlier
        or entropy_score >= settings.high_entropy_threshold
        or (ensemble_fires and primary_outlier)
    )

    return FailureSignalVector(
        agreement_score      = consistency["agreement_score"],
        fsd_score            = consistency["fsd_score"],
        answer_counts        = consistency["answer_counts"],
        entropy_score        = entropy_score,
        ensemble_disagreement= ensemble["disagreement"],
        ensemble_similarity  = ensemble["similarity_score"],
        high_failure_risk    = high_failure_risk,
    )


def get_signal_logs_collection():
    """Returns the signal_logs MongoDB collection or None when unavailable."""
    try:
        from storage.database import _db, _fallback_mode
        if _fallback_mode or _db is None:
            return None
        return _db["signal_logs"]
    except Exception:
        return None
