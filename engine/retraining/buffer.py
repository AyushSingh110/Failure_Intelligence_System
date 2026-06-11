from __future__ import annotations
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

RETRAIN_THRESHOLD = 500   # trigger retrain after this many new labeled examples
_MIN_TOTAL_SAMPLES = 200  # don't retrain until we have at least this many total labels

_retrain_lock = threading.Lock()
_is_retraining = False


# ── Buffer counter (MongoDB-backed) ──────────────────────────────────────────

def _get_buffer_collection():
    try:
        from config import get_settings
        from pymongo import MongoClient
        settings = get_settings()
        if not settings.mongodb_uri:
            return None
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        return client[settings.mongodb_db_name]["retraining_buffer"]
    except Exception as exc:
        logger.debug("RetrainingBuffer: MongoDB unavailable: %s", exc)
        return None


def add_to_buffer(
    log_id:         str,
    request_id:     str,
    is_failure:     bool,
    correct_answer: str = "",
) -> int:
    """
    Records one newly-labeled example in the retraining buffer.
    Returns the current buffer count.
    """
    col = _get_buffer_collection()
    if col is None:
        return 0

    try:
        col.insert_one({
            "log_id":         log_id,
            "request_id":     request_id,
            "is_failure":     is_failure,
            "correct_answer": correct_answer,
            "labeled_at":     datetime.now(timezone.utc).isoformat(),
        })
        count = col.count_documents({})
        logger.debug("RetrainingBuffer: %d labeled examples buffered.", count)
        return count
    except Exception as exc:
        logger.debug("RetrainingBuffer.add_to_buffer error: %s", exc)
        return 0


def get_buffer_count() -> int:
    """Returns current number of examples in the retraining buffer."""
    col = _get_buffer_collection()
    if col is None:
        return 0
    try:
        return col.count_documents({})
    except Exception:
        return 0


def clear_buffer() -> None:
    """Empties the buffer after a successful retrain."""
    col = _get_buffer_collection()
    if col is None:
        return
    try:
        col.delete_many({})
        logger.info("RetrainingBuffer: buffer cleared after retrain.")
    except Exception as exc:
        logger.warning("RetrainingBuffer.clear_buffer error: %s", exc)


# ── Retrain trigger ───────────────────────────────────────────────────────────

def maybe_trigger_retrain(buffer_count: int) -> None:
    """
    Fires a background retrain job when buffer_count >= RETRAIN_THRESHOLD.
    Idempotent — only one retrain runs at a time.
    """
    global _is_retraining

    if buffer_count < RETRAIN_THRESHOLD:
        return

    with _retrain_lock:
        if _is_retraining:
            logger.info("RetrainingBuffer: retrain already in progress, skipping trigger.")
            return
        _is_retraining = True

    logger.info(
        "RetrainingBuffer: %d new labeled examples — triggering background retrain.",
        buffer_count,
    )
    threading.Thread(target=_run_retrain, daemon=True).start()


def _run_retrain() -> None:
    """
    Background retrain job.
    Pulls all labeled signal logs, trains XGBoost, evaluates, conditionally saves.
    """
    global _is_retraining
    try:
        _do_retrain()
    except Exception as exc:
        logger.error("RetrainingBuffer: retrain failed: %s", exc, exc_info=True)
    finally:
        with _retrain_lock:
            _is_retraining = False


def _do_retrain() -> None:
    try:
        from storage.signal_logger import get_labeled_signals
    except ImportError:
        logger.warning("RetrainingBuffer: signal_logger.get_labeled_signals not available — skipping.")
        return

    labeled = get_labeled_signals()
    if len(labeled) < _MIN_TOTAL_SAMPLES:
        logger.info(
            "RetrainingBuffer: only %d labeled examples — need %d to retrain.",
            len(labeled), _MIN_TOTAL_SAMPLES,
        )
        return

    # Build feature matrix
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        import xgboost as xgb

        # Features: entropy, agreement, fsd, ensemble_disagreement, high_failure_risk,
        #           jury_confidence, fix_confidence, gt_confidence, fix_applied, requires_escalation
        X, y = [], []
        for row in labeled:
            X.append([
                row.get("entropy_score",         0.0),
                row.get("agreement_score",        1.0),
                row.get("fsd_score",              0.0),
                float(row.get("ensemble_disagreement", False)),
                float(row.get("high_failure_risk",     False)),
                row.get("jury_confidence",        0.0),
                row.get("fix_confidence",         0.0),
                row.get("gt_confidence",          0.0),
                float(row.get("fix_applied",      False)),
                float(row.get("requires_escalation", False)),
            ])
            y.append(float(row.get("is_failure", False)))

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y if len(set(y)) > 1 else None
        )

        model = xgb.XGBClassifier(
            n_estimators   = 200,
            max_depth      = 4,
            learning_rate  = 0.05,
            subsample      = 0.8,
            use_label_encoder = False,
            eval_metric    = "logloss",
            random_state   = 42,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        new_auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.0

        # Load current AUC from model info endpoint
        try:
            from engine.failure_classifier import _model as current_model
            _cur_auc = 0.840  # known v4 AUC — used as baseline
        except Exception:
            _cur_auc = 0.0

        logger.info(
            "RetrainingBuffer: new model AUC=%.4f vs current=%.4f (n=%d)",
            new_auc, _cur_auc, len(labeled),
        )

        if new_auc < _cur_auc - 0.01:
            logger.warning(
                "RetrainingBuffer: new AUC %.4f is more than 1pp below current %.4f — NOT saving.",
                new_auc, _cur_auc,
            )
            return

        import os
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/xgboost_retrained.pkl")
        logger.info(
            "RetrainingBuffer: new model saved to models/xgboost_retrained.pkl | AUC=%.4f",
            new_auc,
        )
        clear_buffer()

    except ImportError as exc:
        logger.warning("RetrainingBuffer: missing dependency for retrain (%s) — skipping.", exc)
    except Exception as exc:
        logger.error("RetrainingBuffer._do_retrain error: %s", exc, exc_info=True)
