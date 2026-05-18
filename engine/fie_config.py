"""
Centralized runtime configuration for FIE.

Load order (highest priority wins):
  1. MongoDB `fie_config` collection — hot-reload, no server restart needed
  2. Hardcoded defaults below

Auto-calibration:
  recalibrate() reads all labeled signal_logs (feedback_received=True),
  computes the best-F1 threshold per question_type using sklearn's
  precision_recall_curve, and writes results back to MongoDB.
  Called automatically after every RECALIBRATION_INTERVAL new feedback entries.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model identity ─────────────────────────────────────────────────────────────
MODEL_VERSION  = "xgboost-v4"
MODEL_TRAINED  = "2026-05-05"

# ── How often to auto-recalibrate (number of new feedback entries) ─────────────
RECALIBRATION_INTERVAL = 50

# ── Default thresholds per question type ──────────────────────────────────────
# Factual: lower threshold = higher recall (catch more factual hallucinations)
# Opinion: higher threshold = only override POET when very confident
_DEFAULTS: dict[str, float] = {
    "FACTUAL":   0.40,
    "TEMPORAL":  0.50,
    "REASONING": 0.48,
    "CODE":      0.52,
    "OPINION":   0.60,
    "UNKNOWN":   0.45,   # fallback for unclassified
}

# ── Jury gate: minimum jury confidence before GT pipeline runs ─────────────────
JURY_CONFIDENCE_GATE: float = 0.45

# ── Classifier calibration params (hot-configurable) ─────────────────────────
# temperature: T > 1 pulls XGBoost probs toward 0.5 (reduces overconfidence).
#   Increase if the model is systematically overconfident on your traffic.
#   Decrease (toward 1.0) if you want sharper, more decisive predictions.
# ambiguous_band: predictions within this distance of the threshold are flagged
#   as low-confidence and eligible for jury escalation.
_TEMPERATURE_DEFAULT:    float = 1.15
_AMBIGUOUS_BAND_DEFAULT: float = 0.06

# ── Self-consistency threshold for REASONING/CODE ground truth ────────────────
# Mean pairwise cosine similarity between shadow outputs must exceed this value
# for the system to treat the majority answer as pseudo-GT and auto-correct.
# Below this → escalate to human review.
# Range [0, 1]. Higher = more conservative (escalate more).
_CONSISTENCY_THRESHOLD_DEFAULT: float = 0.72

# ── Adversarial scan threshold ────────────────────────────────────────────────
# Minimum best_conf for scan_prompt() to classify a prompt as an attack.
# Tuned against JailbreakBench v2 (PAIR v2 + framing filter active):
#   0.45 → Recall 88%, FPR 12%, F1 88%, AUC 0.923  (production baseline)
#   0.50 → Recall 73%, FPR  8%                      (stricter)
# Overridable via SCAN_THRESHOLD env var; hot-updatable via update_scan_threshold().
import os as _os
_SCAN_THRESHOLD_DEFAULT: float = float(_os.environ.get("SCAN_THRESHOLD", "0.45"))

# ── Pre-flight block mode ──────────────────────────────────────────────────────
# When True, prompts that exceed the scan threshold are BLOCKED before the
# primary LLM is called (inline protection mode).
# When False, attacks are detected and logged but the LLM call proceeds
# (warn-only / monitoring-only mode).
# Hot-updatable via update_preflight_config(); persisted to MongoDB.
_PREFLIGHT_BLOCK_ENABLED_DEFAULT: bool = _os.environ.get(
    "PREFLIGHT_BLOCK_ENABLED", "true"
).lower() not in ("0", "false", "no")

# ── In-memory live state ───────────────────────────────────────────────────────
_thresholds:              dict[str, float] = dict(_DEFAULTS)
_temperature:             float            = _TEMPERATURE_DEFAULT
_ambiguous_band:          float            = _AMBIGUOUS_BAND_DEFAULT
_consistency_threshold:   float            = _CONSISTENCY_THRESHOLD_DEFAULT
_scan_threshold:          float            = _SCAN_THRESHOLD_DEFAULT
_preflight_block_enabled: bool             = _PREFLIGHT_BLOCK_ENABLED_DEFAULT
_config_version:          str              = "default"
_feedback_count_at_last_calib: int         = 0
_lock = threading.Lock()


# ── MongoDB helpers ────────────────────────────────────────────────────────────

def _get_config_collection():
    try:
        from storage.database import _db, _fallback_mode
        if _fallback_mode or _db is None:
            return None
        return _db["fie_config"]
    except Exception:
        return None


def _get_signal_collection():
    try:
        from storage.database import _db, _fallback_mode
        if _fallback_mode or _db is None:
            return None
        return _db["signal_logs"]
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def load_from_db() -> None:
    """
    Pull the latest threshold config from MongoDB.
    Called once at startup; can be called again at any time to hot-reload.
    Loads: per-type thresholds, temperature, ambiguous_band, preflight_block_enabled.
    """
    global _thresholds, _temperature, _ambiguous_band, _config_version, _preflight_block_enabled
    col = _get_config_collection()
    if col is None:
        logger.debug("fie_config: MongoDB unavailable, using defaults.")
        return
    try:
        doc = col.find_one({"_id": "thresholds"})
        if doc:
            with _lock:
                for qt in _DEFAULTS:
                    key = f"threshold_{qt}"
                    if key in doc:
                        _thresholds[qt] = float(doc[key])
                if "temperature" in doc:
                    _temperature = float(doc["temperature"])
                if "ambiguous_band" in doc:
                    _ambiguous_band = float(doc["ambiguous_band"])
                if "consistency_threshold" in doc:
                    _consistency_threshold = float(doc["consistency_threshold"])
                if "scan_threshold" in doc:
                    _scan_threshold = float(doc["scan_threshold"])
                if "preflight_block_enabled" in doc:
                    _preflight_block_enabled = bool(doc["preflight_block_enabled"])
                _config_version = doc.get("version", "db-unknown")
            logger.info(
                "fie_config loaded from MongoDB | version=%s | thresholds=%s | "
                "temperature=%.2f | ambiguous_band=%.3f | preflight_block=%s",
                _config_version, _thresholds, _temperature, _ambiguous_band,
                _preflight_block_enabled,
            )
        else:
            logger.info("fie_config: no saved config in MongoDB, using defaults.")
    except Exception as exc:
        logger.warning("fie_config load failed: %s", exc)


def get_threshold(question_type: str = "UNKNOWN") -> float:
    """Returns the current classifier threshold for this question type."""
    with _lock:
        return _thresholds.get(question_type.upper(), _thresholds["UNKNOWN"])


def get_temperature() -> float:
    """Returns the current temperature scaling factor for XGBoost calibration."""
    with _lock:
        return _temperature


def get_ambiguous_band() -> float:
    """Returns the band around the threshold that marks low-confidence predictions."""
    with _lock:
        return _ambiguous_band


def get_consistency_threshold() -> float:
    """
    Returns the minimum mean pairwise cosine similarity required for shadow model
    outputs to be treated as consistent enough to use as pseudo-GT for
    REASONING/CODE questions.
    """
    with _lock:
        return _consistency_threshold


def get_scan_threshold() -> float:
    """
    Returns the current adversarial scan threshold for scan_prompt().
    Reads from MongoDB-backed live state; falls back to SCAN_THRESHOLD env var
    or the compiled default of 0.45.
    """
    with _lock:
        return _scan_threshold


def update_scan_threshold(value: float) -> float:
    """
    Hot-update the scan threshold at runtime and persist to MongoDB.
    Returns the new value.
    """
    global _scan_threshold
    value = float(value)
    with _lock:
        _scan_threshold = value

    cfg_col = _get_config_collection()
    if cfg_col is not None:
        try:
            cfg_col.update_one(
                {"_id": "thresholds"},
                {"$set": {"scan_threshold": value}},
                upsert=True,
            )
        except Exception as exc:
            logger.warning("fie_config.update_scan_threshold persist failed: %s", exc)

    logger.info("fie_config scan_threshold updated: %.4f", value)
    return value


def get_preflight_config() -> dict:
    """
    Returns the current pre-flight guard configuration.

    Keys
    ----
    block_enabled : bool   — True = hard block; False = warn-only
    scan_threshold: float  — confidence floor used by scan_prompt()
    """
    with _lock:
        return {
            "block_enabled":  _preflight_block_enabled,
            "scan_threshold": _scan_threshold,
        }


def update_preflight_config(block_enabled: Optional[bool] = None) -> dict:
    """
    Hot-update the pre-flight guard at runtime and persist to MongoDB.
    Use scan_threshold via update_scan_threshold(); this controls the block toggle.

    Parameters
    ----------
    block_enabled : bool, optional
        True = hard block mode (LLM never runs on detected attacks).
        False = warn-only mode (detects but allows through).

    Returns
    -------
    dict  Current live preflight config after the update.
    """
    global _preflight_block_enabled
    with _lock:
        if block_enabled is not None:
            _preflight_block_enabled = bool(block_enabled)
        result = {
            "block_enabled":  _preflight_block_enabled,
            "scan_threshold": _scan_threshold,
        }

    cfg_col = _get_config_collection()
    if cfg_col is not None:
        try:
            cfg_col.update_one(
                {"_id": "thresholds"},
                {"$set": {"preflight_block_enabled": _preflight_block_enabled}},
                upsert=True,
            )
        except Exception as exc:
            logger.warning("fie_config.update_preflight_config persist failed: %s", exc)

    logger.info(
        "fie_config preflight updated | block_enabled=%s | scan_threshold=%.4f",
        _preflight_block_enabled, _scan_threshold,
    )
    return result


def get_all_thresholds() -> dict[str, float]:
    with _lock:
        return dict(_thresholds)


def get_config_version() -> str:
    with _lock:
        return _config_version


def update_params(
    temperature:            Optional[float] = None,
    ambiguous_band:         Optional[float] = None,
    consistency_threshold:  Optional[float] = None,
) -> dict:
    """
    Hot-update calibration params at runtime and persist to MongoDB.
    Any param can be omitted — only provided values are changed.
    Returns the new live state for all three params.
    """
    global _temperature, _ambiguous_band, _consistency_threshold
    with _lock:
        if temperature is not None:
            _temperature = float(temperature)
        if ambiguous_band is not None:
            _ambiguous_band = float(ambiguous_band)
        if consistency_threshold is not None:
            _consistency_threshold = float(consistency_threshold)
        live = {
            "temperature":           _temperature,
            "ambiguous_band":        _ambiguous_band,
            "consistency_threshold": _consistency_threshold,
        }

    cfg_col = _get_config_collection()
    if cfg_col is not None:
        try:
            cfg_col.update_one(
                {"_id": "thresholds"},
                {"$set": live},
                upsert=True,
            )
        except Exception as exc:
            logger.warning("fie_config.update_params persist failed: %s", exc)

    logger.info("fie_config params updated: %s", live)
    return live


def recalibrate() -> dict:
    """
    Recomputes best-F1 threshold per question_type from labeled signal_logs.
    Writes results to MongoDB and updates the in-memory state.
    Returns a summary dict for logging / API responses.
    """
    global _thresholds, _config_version, _feedback_count_at_last_calib

    sig_col = _get_signal_collection()
    if sig_col is None:
        return {"status": "skipped", "reason": "MongoDB unavailable"}

    try:
        from sklearn.metrics import precision_recall_curve
        import numpy as np
    except ImportError:
        return {"status": "skipped", "reason": "scikit-learn not installed"}

    labeled = list(sig_col.find(
        {"feedback_received": True},
        {"classifier_probability": 1, "fie_was_correct": 1, "question_type": 1},
    ))

    if len(labeled) < 20:
        return {"status": "skipped", "reason": f"Only {len(labeled)} labeled examples, need ≥20"}

    new_thresholds: dict[str, float] = dict(_DEFAULTS)
    per_type_stats: dict[str, dict]  = {}

    # Group by question_type
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for doc in labeled:
        qt   = (doc.get("question_type") or "UNKNOWN").upper()
        prob = doc.get("classifier_probability")
        correct = doc.get("fie_was_correct")
        if prob is not None and correct is not None:
            groups[qt].append((prob, int(correct)))

    # Add all to UNKNOWN group for global fallback
    all_probs    = [p for docs in groups.values() for p, _ in docs]
    all_labels   = [l for docs in groups.values() for _, l in docs]

    for qt, items in groups.items():
        if len(items) < 10:
            per_type_stats[qt] = {"status": "skipped", "n": len(items)}
            continue
        probs  = [p for p, _ in items]
        labels = [l for _, l in items]
        try:
            prec, rec, threshs = precision_recall_curve(labels, probs)
            f1s  = 2 * prec * rec / (prec + rec + 1e-9)
            best_idx = int(np.argmax(f1s[:-1]))
            best_t   = float(threshs[best_idx])
            best_f1  = float(f1s[best_idx])
            # Clamp to [0.25, 0.75] to avoid degenerate thresholds
            best_t = max(0.25, min(0.75, best_t))
            new_thresholds[qt] = best_t
            per_type_stats[qt] = {
                "n": len(items), "best_threshold": round(best_t, 4),
                "best_f1": round(best_f1, 4),
            }
        except Exception as exc:
            per_type_stats[qt] = {"status": "error", "msg": str(exc)}

    # Global fallback from all data
    if len(all_probs) >= 20:
        try:
            prec, rec, threshs = precision_recall_curve(all_labels, all_probs)
            f1s  = 2 * prec * rec / (prec + rec + 1e-9)
            best_idx = int(np.argmax(f1s[:-1]))
            best_t   = float(threshs[best_idx])
            best_t   = max(0.25, min(0.75, best_t))
            new_thresholds["UNKNOWN"] = best_t
        except Exception:
            pass

    version = f"calibrated-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    # Persist to MongoDB (thresholds + current calibration params)
    cfg_col = _get_config_collection()
    if cfg_col is not None:
        try:
            with _lock:
                cur_temp     = _temperature
                cur_band     = _ambiguous_band
                cur_cons     = _consistency_threshold
                cur_scan     = _scan_threshold
                cur_preflight= _preflight_block_enabled
            doc = {
                "_id": "thresholds", "version": version,
                "updated_at": datetime.utcnow().isoformat(),
                "temperature": cur_temp, "ambiguous_band": cur_band,
                "consistency_threshold": cur_cons,
                "scan_threshold": cur_scan,
                "preflight_block_enabled": cur_preflight,
            }
            for qt, t in new_thresholds.items():
                doc[f"threshold_{qt}"] = t
            cfg_col.replace_one({"_id": "thresholds"}, doc, upsert=True)
        except Exception as exc:
            logger.warning("fie_config: failed to persist thresholds: %s", exc)

    with _lock:
        _thresholds    = new_thresholds
        _config_version = version
        _feedback_count_at_last_calib = len(labeled)

    logger.info("fie_config recalibrated | version=%s | %s", version, per_type_stats)
    return {
        "status":      "ok",
        "version":     version,
        "n_labeled":   len(labeled),
        "thresholds":  new_thresholds,
        "per_type":    per_type_stats,
    }


def maybe_recalibrate() -> None:
    """
    Called after each feedback submission.
    Triggers recalibration only when RECALIBRATION_INTERVAL new labels have
    accumulated since the last calibration — runs in a background thread.
    """
    global _feedback_count_at_last_calib
    sig_col = _get_signal_collection()
    if sig_col is None:
        return
    try:
        total_labeled = sig_col.count_documents({"feedback_received": True})
        with _lock:
            since_last = total_labeled - _feedback_count_at_last_calib
        if since_last >= RECALIBRATION_INTERVAL:
            t = threading.Thread(target=recalibrate, daemon=True)
            t.start()
            logger.info(
                "fie_config: auto-recalibration triggered (%d new labels)", since_last
            )
    except Exception as exc:
        logger.debug("maybe_recalibrate check failed: %s", exc)
