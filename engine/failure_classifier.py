from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Paths 
_HERE       = os.path.dirname(os.path.dirname(__file__))          # project root
_MODEL_PATH = os.path.join(_HERE, "models", "failure_classifier_v2.pkl")
_COLS_PATH  = os.path.join(_HERE, "models", "feature_columns_v2.pkl")

# Decision threshold 
CLASSIFIER_THRESHOLD: float = 0.423

# Feature names for one-hot encoding — must match synthetic_generator.py training
_CATS = ["archetype", "jury_verdict", "fix_strategy", "gt_source"]

# ── Lazy singleton ────────────────────────────────────────────────────────────
_model     = None
_feat_cols = None
_loaded    = False


def _load() -> None:
    global _model, _feat_cols, _loaded
    if _loaded:
        return
    _loaded = True  # prevent repeated attempts even if the first fails

    if not os.path.exists(_MODEL_PATH) or not os.path.exists(_COLS_PATH):
        logger.warning(
            "Failure classifier model not found at %s — "
            "falling back to POET rule-based detection.",
            _MODEL_PATH,
        )
        return

    try:
        import joblib
        _model     = joblib.load(_MODEL_PATH)
        _feat_cols = joblib.load(_COLS_PATH)
        logger.info(
            "FIE failure_classifier_v2 loaded (%d features, threshold=%.3f)",
            len(_feat_cols), CLASSIFIER_THRESHOLD,
        )
    except Exception as exc:
        logger.warning(
            "Could not load failure classifier (%s) — falling back to POET.", exc
        )
        _model     = None
        _feat_cols = None


def predict(
    agreement_score:     float,
    entropy_score:       float,
    jury_confidence:     float,
    fix_confidence:      float,
    gt_confidence:       float,
    high_failure_risk:   bool,    # POET's decision — used as an *input feature*
    fix_applied:         bool,
    requires_escalation: bool,
    gt_override:         bool,
    archetype:           str,
    jury_verdict_str:    str,
    fix_strategy:        str,
    gt_source:           str,
) -> tuple[bool, float]:
    """
    Runs the XGBoost v2 classifier.

    Returns:
        (is_failure, probability)
        is_failure  — True if the classifier predicts a real failure
        probability — raw XGBoost score in [0, 1]

    Falls back to (high_failure_risk, 1.0 or 0.0) if the model is unavailable.
    """
    _load()

    if _model is None or _feat_cols is None:
        fallback_prob = 1.0 if high_failure_risk else 0.0
        return high_failure_risk, fallback_prob

    try:
        row = {
            "agreement_score"    : agreement_score,
            "entropy_score"      : entropy_score,
            "jury_confidence"    : jury_confidence,
            "fix_confidence"     : fix_confidence,
            "gt_confidence"      : gt_confidence,
            "high_failure_risk"  : int(high_failure_risk),
            "fix_applied"        : int(fix_applied),
            "requires_escalation": int(requires_escalation),
            "gt_override"        : int(gt_override),
            "archetype"          : archetype        or "NONE",
            "jury_verdict"       : jury_verdict_str or "NONE",
            "fix_strategy"       : fix_strategy     or "NONE",
            "gt_source"          : gt_source        or "none",
        }

        df     = pd.DataFrame([row])
        df_enc = pd.get_dummies(df, columns=_CATS, prefix=_CATS)

        # Align to the exact feature space used during training.
        # Any unseen category → all-zeros for that group (safe fallback).
        df_enc = df_enc.reindex(columns=_feat_cols, fill_value=0).astype(float)

        prob       = float(_model.predict_proba(df_enc)[0, 1])
        is_failure = prob >= CLASSIFIER_THRESHOLD

        logger.debug(
            "Classifier: prob=%.4f threshold=%.3f is_failure=%s (POET was %s)",
            prob, CLASSIFIER_THRESHOLD, is_failure, high_failure_risk,
        )
        return is_failure, prob

    except Exception as exc:
        logger.warning("Classifier inference error (%s) — falling back to POET.", exc)
        fallback_prob = 1.0 if high_failure_risk else 0.0
        return high_failure_risk, fallback_prob
