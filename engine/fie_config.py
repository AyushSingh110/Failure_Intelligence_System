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

# ── In-memory live state ───────────────────────────────────────────────────────
_thresholds: dict[str, float] = dict(_DEFAULTS)
_config_version: str          = "default"
_feedback_count_at_last_calib: int = 0
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
    """
    global _thresholds, _config_version
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
                _config_version = doc.get("version", "db-unknown")
            logger.info(
                "fie_config loaded from MongoDB | version=%s | thresholds=%s",
                _config_version, _thresholds,
            )
        else:
            logger.info("fie_config: no saved config in MongoDB, using defaults.")
    except Exception as exc:
        logger.warning("fie_config load failed: %s", exc)


def get_threshold(question_type: str = "UNKNOWN") -> float:
    """Returns the current classifier threshold for this question type."""
    with _lock:
        return _thresholds.get(question_type.upper(), _thresholds["UNKNOWN"])


def get_all_thresholds() -> dict[str, float]:
    with _lock:
        return dict(_thresholds)


def get_config_version() -> str:
    with _lock:
        return _config_version


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

    # Persist to MongoDB
    cfg_col = _get_config_collection()
    if cfg_col is not None:
        try:
            doc = {"_id": "thresholds", "version": version, "updated_at": datetime.utcnow().isoformat()}
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
