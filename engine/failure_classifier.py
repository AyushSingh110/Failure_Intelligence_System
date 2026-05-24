from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Paths — v4 takes priority, falls back to v3, then v2
_HERE         = os.path.dirname(os.path.dirname(__file__))
_MODEL_V4     = os.path.join(_HERE, "models", "failure_classifier_v4.pkl")
_COLS_V4      = os.path.join(_HERE, "models", "feature_columns_v4.pkl")
_MODEL_V3     = os.path.join(_HERE, "models", "failure_classifier_v3.pkl")
_COLS_V3      = os.path.join(_HERE, "models", "feature_columns_v3.pkl")
_MODEL_V2     = os.path.join(_HERE, "models", "failure_classifier_v2.pkl")
_COLS_V2      = os.path.join(_HERE, "models", "feature_columns_v2.pkl")

if os.path.exists(_MODEL_V4):
    _MODEL_PATH, _COLS_PATH = _MODEL_V4, _COLS_V4
    _VERSION    = "v4"
elif os.path.exists(_MODEL_V3):
    _MODEL_PATH, _COLS_PATH = _MODEL_V3, _COLS_V3
    _VERSION    = "v3"
    logger.warning(
        "failure_classifier: v4 model not found at %s — falling back to v3. "
        "AUC drops from 0.840 to 0.749. To fix: ensure failure_classifier_v4.pkl "
        "and feature_columns_v4.pkl are present in the models/ directory.",
        _MODEL_V4,
    )
else:
    _MODEL_PATH, _COLS_PATH = _MODEL_V2, _COLS_V2
    _VERSION    = "v2"
    logger.warning(
        "failure_classifier: v4 and v3 models not found — falling back to v2. "
        "Significant accuracy loss. Ensure failure_classifier_v4.pkl is in models/.",
    )

_USING_V3 = _VERSION in ("v3", "v4")  # both v3 and v4 share the same feature schema

# Fallback threshold used only when fie_config is unavailable.
# The live value is always pulled from fie_config.get_threshold() at inference time.
CLASSIFIER_THRESHOLD: float = 0.522

# Known blind-spot question types — used only for the in_blind_spot flag.
# The actual threshold for each type comes from fie_config (MongoDB-backed), not here.
_BLIND_SPOT_TYPES: frozenset[str] = frozenset({"CODE", "REASONING", "OPINION"})

# v4 feature schema: adds question_type + provenance + reasoning failure type
_CATS_V3 = ["archetype", "jury_verdict", "fix_strategy", "gt_source", "question_type",
            "provenance_category", "provenance_label", "reasoning_failure_type"]
_CATS_V2 = ["archetype", "jury_verdict", "fix_strategy", "gt_source"]
_CATS    = _CATS_V3 if _USING_V3 else _CATS_V2


# ── Result dataclass for predict_full() ───────────────────────────────────────

@dataclass
class ClassifierResult:
    is_failure:     bool
    probability:    float   # temperature-calibrated probability
    raw_prob:       float   # raw XGBoost output (before calibration)
    threshold:      float   # effective threshold used (may differ from global)
    in_blind_spot:  bool    # True if question_type is a known weak spot
    low_confidence: bool    # True if prob is within _AMBIGUOUS_BAND of threshold
    question_type:  str


def _apply_temperature(raw_prob: float, temperature: float) -> float:
    """Platt-style temperature scaling: sigmoid(logit(p) / T)."""
    p = max(1e-7, min(1 - 1e-7, raw_prob))   # clamp to avoid log(0)
    logit = math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-logit / temperature))

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
            "FIE failure_classifier_%s loaded (%d features, threshold=%.3f)",
            _VERSION, len(_feat_cols), CLASSIFIER_THRESHOLD,
        )
    except Exception as exc:
        logger.warning(
            "Could not load failure classifier (%s) — falling back to POET.", exc
        )
        _model     = None
        _feat_cols = None


def _infer(
    agreement_score:     float,
    entropy_score:       float,
    jury_confidence:     float,
    fix_confidence:      float,
    gt_confidence:       float,
    high_failure_risk:   bool,
    fix_applied:         bool,
    requires_escalation: bool,
    gt_override:         bool,
    archetype:           str,
    jury_verdict_str:    str,
    fix_strategy:        str,
    gt_source:           str,
    question_type:       str,
    provenance_category:    str   = "GENERAL_KNOWLEDGE",
    provenance_label:       str   = "UNVERIFIED_MODEL_INFERENCE",
    reasoning_failure_type: str   = "NOT_APPLICABLE",
    reasoning_confidence:   float = 0.0,
) -> ClassifierResult:
    """Core inference — shared by predict() and predict_full()."""
    _load()

    qt = (question_type or "UNKNOWN").upper()

    # Pull live config from fie_config (MongoDB-backed, hot-reloadable)
    try:
        from engine.fie_config import get_threshold, get_temperature, get_ambiguous_band
        threshold    = get_threshold(qt)
        temperature  = get_temperature()
        ambig_band   = get_ambiguous_band()
    except Exception:
        threshold    = CLASSIFIER_THRESHOLD
        temperature  = 1.15
        ambig_band   = 0.06

    in_blind_spot = qt in _BLIND_SPOT_TYPES

    if _model is None or _feat_cols is None:
        fallback_prob = 1.0 if high_failure_risk else 0.0
        return ClassifierResult(
            is_failure    = high_failure_risk,
            probability   = fallback_prob,
            raw_prob      = fallback_prob,
            threshold     = threshold,
            in_blind_spot = in_blind_spot,
            low_confidence= False,
            question_type = qt,
        )

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
            "question_type"      : qt,
            "provenance_category"    : provenance_category    or "GENERAL_KNOWLEDGE",
            "provenance_label"       : provenance_label       or "UNVERIFIED_MODEL_INFERENCE",
            "reasoning_failure_type" : reasoning_failure_type or "NOT_APPLICABLE",
            "reasoning_confidence"   : float(reasoning_confidence or 0.0),
        }

        df     = pd.DataFrame([row])
        df_enc = pd.get_dummies(df, columns=_CATS, prefix=_CATS)
        df_enc = df_enc.reindex(columns=_feat_cols, fill_value=0).astype(float)

        raw_prob  = float(_model.predict_proba(df_enc)[0, 1])
        cal_prob  = _apply_temperature(raw_prob, temperature)

        is_failure     = cal_prob >= threshold
        low_confidence = abs(cal_prob - threshold) <= ambig_band

        logger.debug(
            "Classifier: raw=%.4f cal=%.4f threshold=%.3f temp=%.2f "
            "is_failure=%s blind_spot=%s low_conf=%s",
            raw_prob, cal_prob, threshold, temperature,
            is_failure, in_blind_spot, low_confidence,
        )

        return ClassifierResult(
            is_failure    = is_failure,
            probability   = cal_prob,
            raw_prob      = raw_prob,
            threshold     = threshold,
            in_blind_spot = in_blind_spot,
            low_confidence= low_confidence,
            question_type = qt,
        )

    except Exception as exc:
        logger.warning("Classifier inference error (%s) — falling back to POET.", exc)
        fallback_prob = 1.0 if high_failure_risk else 0.0
        return ClassifierResult(
            is_failure    = high_failure_risk,
            probability   = fallback_prob,
            raw_prob      = fallback_prob,
            threshold     = threshold,
            in_blind_spot = in_blind_spot,
            low_confidence= False,
            question_type = qt,
        )


def predict(
    agreement_score:     float,
    entropy_score:       float,
    jury_confidence:     float,
    fix_confidence:      float,
    gt_confidence:       float,
    high_failure_risk:   bool,
    fix_applied:         bool,
    requires_escalation: bool,
    gt_override:         bool,
    archetype:           str,
    jury_verdict_str:    str,
    fix_strategy:        str,
    gt_source:           str,
    question_type:          str   = "UNKNOWN",
    provenance_category:    str   = "GENERAL_KNOWLEDGE",
    provenance_label:       str   = "UNVERIFIED_MODEL_INFERENCE",
    reasoning_failure_type: str   = "NOT_APPLICABLE",
    reasoning_confidence:   float = 0.0,
) -> tuple[bool, float]:
    """
    Runs the XGBoost v4 classifier with temperature calibration.

    Returns:
        (is_failure, probability)
        is_failure  — True if the calibrated probability meets the effective threshold
        probability — temperature-calibrated probability in [0, 1]

    Falls back to (high_failure_risk, 1.0 or 0.0) if the model is unavailable.
    Signature is unchanged from v4 — all existing callers work without modification.
    """
    r = _infer(
        agreement_score, entropy_score, jury_confidence, fix_confidence,
        gt_confidence, high_failure_risk, fix_applied, requires_escalation,
        gt_override, archetype, jury_verdict_str, fix_strategy, gt_source, question_type,
        provenance_category, provenance_label, reasoning_failure_type, reasoning_confidence,
    )
    return r.is_failure, r.probability


def predict_full(
    agreement_score:     float,
    entropy_score:       float,
    jury_confidence:     float,
    fix_confidence:      float,
    gt_confidence:       float,
    high_failure_risk:   bool,
    fix_applied:         bool,
    requires_escalation: bool,
    gt_override:         bool,
    archetype:           str,
    jury_verdict_str:    str,
    fix_strategy:        str,
    gt_source:           str,
    question_type:          str   = "UNKNOWN",
    provenance_category:    str   = "GENERAL_KNOWLEDGE",
    provenance_label:       str   = "UNVERIFIED_MODEL_INFERENCE",
    reasoning_failure_type: str   = "NOT_APPLICABLE",
    reasoning_confidence:   float = 0.0,
) -> ClassifierResult:
    """
    Same as predict() but returns a ClassifierResult with full metadata:
      .is_failure     — bool
      .probability    — calibrated prob
      .raw_prob       — raw XGBoost output (before temperature scaling)
      .threshold      — effective threshold (may be lower for blind-spot types)
      .in_blind_spot  — True if question_type is CODE/REASONING/OPINION
      .low_confidence — True if prob is within _AMBIGUOUS_BAND of threshold
      .question_type  — normalized question type string

    Use this in routes that want to escalate low-confidence or blind-spot predictions
    to the DiagnosticJury instead of returning them directly.
    """
    return _infer(
        agreement_score, entropy_score, jury_confidence, fix_confidence,
        gt_confidence, high_failure_risk, fix_applied, requires_escalation,
        gt_override, archetype, jury_verdict_str, fix_strategy, gt_source, question_type,
        provenance_category, provenance_label, reasoning_failure_type, reasoning_confidence,
    )
