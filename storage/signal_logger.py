from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# MongoDB collection name — separate from inferences to keep concerns clean
_COLLECTION_NAME = "signal_logs"


def _get_collection():
    """Returns the signal_logs MongoDB collection, or None if unavailable."""
    try:
        from storage.database import _db, _fallback_mode
        if _fallback_mode or _db is None:
            return None
        col = _db[_COLLECTION_NAME]
        return col
    except Exception as exc:
        logger.debug("Could not access signal_logs collection: %s", exc)
        return None


def log_signal(
    request_id:           str,
    prompt:               str,
    primary_output:       str,
    shadow_outputs:       list[str],
    shadow_confidences:   list[str],
    shadow_weights:       list[float],
    entropy_score:        float,
    agreement_score:      float,
    fsd_score:            float,
    ensemble_disagreement: bool,
    high_failure_risk:    bool,
    layers_fired:         list[str],
    layer_scores:         dict[str, float],
    jury_verdict:         str,
    jury_confidence:      float,
    gt_source:            str,
    gt_confidence:        float,
    gt_override_applied:  bool,
    gt_verified_answer:   str,
    requires_escalation:  bool,
    escalation_reason:    str,
    fix_applied:          bool,
    fix_strategy:         str,
    fix_confidence:       float,
    fix_output:           str,
) -> str:
    """
    Saves a complete raw signal snapshot to the signal_logs collection.
    Returns the log_id (UUID string) so the caller can store it and
    later call update_signal_feedback() when user feedback arrives.
    """
    log_id = str(uuid.uuid4())

    doc: dict[str, Any] = {
        "_id":                  log_id,
        "log_id":               log_id,
        "request_id":           request_id,
        "timestamp":            datetime.utcnow().isoformat(),

        # Core inputs
        "prompt":               prompt[:1000],      
        "primary_output":       primary_output[:500],

        # Shadow ensemble
        "shadow_outputs":       [o[:300] for o in shadow_outputs],
        "shadow_confidences":   shadow_confidences,
        "shadow_weights":       shadow_weights,

        # FSV — raw signal features
        "entropy_score":        round(entropy_score, 6),
        "agreement_score":      round(agreement_score, 6),
        "fsd_score":            round(fsd_score, 6),
        "ensemble_disagreement": ensemble_disagreement,
        "high_failure_risk":    high_failure_risk,

        # DomainCritic layer detail
        "layers_fired":         layers_fired,
        "layer_scores":         {k: round(v, 6) for k, v in layer_scores.items()},

        # Jury
        "jury_verdict":         jury_verdict,
        "jury_confidence":      round(jury_confidence, 6),

        # Ground truth pipeline
        "gt_source":            gt_source,
        "gt_confidence":        round(gt_confidence, 6),
        "gt_override_applied":  gt_override_applied,
        "gt_verified_answer":   gt_verified_answer[:300],
        "requires_escalation":  requires_escalation,
        "escalation_reason":    escalation_reason[:300],

        # Fix engine
        "fix_applied":          fix_applied,
        "fix_strategy":         fix_strategy,
        "fix_confidence":       round(fix_confidence, 6),
        "fix_output":           fix_output[:500],

        # Feedback — filled in later
        "feedback_received":    False,
        "fie_was_correct":      None,
        "correct_answer":       None,
    }

    try:
        col = _get_collection()
        if col is None:
            logger.debug("Signal log dropped — MongoDB unavailable (log_id=%s)", log_id)
            return log_id

        col.insert_one(doc)

        # Index on request_id and timestamp for fast lookups + range queries
        col.create_index("request_id", background=True)
        col.create_index("timestamp",  background=True)
        col.create_index("feedback_received", background=True)
        col.create_index("jury_verdict",      background=True)

        logger.debug("Signal logged | log_id=%s request_id=%s", log_id, request_id)

    except Exception as exc:
        logger.warning("Signal log write failed (non-fatal): %s", exc)

    return log_id


def update_signal_feedback(
    log_id:         str,
    fie_was_correct: bool,
    correct_answer:  Optional[str] = None,
) -> bool:
    """
    Called when a user submits feedback via POST /feedback/{request_id}.
    This is what turns a raw log into a LABELED TRAINING EXAMPLE.
    """
    try:
        col = _get_collection()
        if col is None:
            return False

        update = {
            "$set": {
                "feedback_received": True,
                "fie_was_correct":   fie_was_correct,
                "correct_answer":    correct_answer or "",
                "feedback_at":       datetime.utcnow().isoformat(),
            }
        }
        result = col.update_one({"log_id": log_id}, update)
        return result.modified_count > 0

    except Exception as exc:
        logger.warning("Failed to update signal feedback: %s", exc)
        return False


def find_log_by_request_id(request_id: str) -> Optional[dict]:
    """Returns the signal_log document for a given request_id, or None."""
    try:
        col = _get_collection()
        if col is None:
            return None
        return col.find_one({"request_id": request_id}, {"_id": 0})
    except Exception as exc:
        logger.warning("Signal log lookup failed: %s", exc)
        return None


def get_recent_logs(limit: int = 100) -> list[dict]:
    """Returns the N most recent signal logs, newest first."""
    try:
        col = _get_collection()
        if col is None:
            return []
        cursor = col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception as exc:
        logger.warning("Failed to fetch signal logs: %s", exc)
        return []


def get_calibration_stats() -> dict:
    """
    Computes calibration statistics from all labeled examples
    (documents where feedback_received=True).
    """
    try:
        col = _get_collection()
        if col is None:
            return {"error": "MongoDB unavailable"}

        labeled = list(col.find({"feedback_received": True}, {"_id": 0}))
        if not labeled:
            return {
                "total_labeled":    0,
                "message":          "No labeled examples yet. Submit feedback to start collecting.",
                "calibration":      {},
                "layer_precision":  {},
            }

        total          = len(labeled)
        correct_count  = sum(1 for d in labeled if d.get("fie_was_correct") is True)
        overall_acc    = round(correct_count / total, 4) if total else 0.0

        #Confidence bucket calibration 
        buckets: dict[str, dict] = {
            "0.9-1.0": {"predicted_mid": 0.95, "correct": 0, "total": 0},
            "0.7-0.9": {"predicted_mid": 0.80, "correct": 0, "total": 0},
            "0.5-0.7": {"predicted_mid": 0.60, "correct": 0, "total": 0},
            "0.0-0.5": {"predicted_mid": 0.25, "correct": 0, "total": 0},
        }
        for doc in labeled:
            conf    = doc.get("jury_confidence", 0.0)
            correct = doc.get("fie_was_correct", False)
            if conf >= 0.9:
                key = "0.9-1.0"
            elif conf >= 0.7:
                key = "0.7-0.9"
            elif conf >= 0.5:
                key = "0.5-0.7"
            else:
                key = "0.0-0.5"
            buckets[key]["total"] += 1
            if correct:
                buckets[key]["correct"] += 1

        calibration = {}
        for bucket, data in buckets.items():
            n = data["total"]
            if n == 0:
                continue
            actual_acc = round(data["correct"] / n, 4)
            calibration[bucket] = {
                "total_examples":     n,
                "fie_correct_count":  data["correct"],
                "predicted_accuracy": data["predicted_mid"],
                "actual_accuracy":    actual_acc,
                "calibration_error":  round(abs(data["predicted_mid"] - actual_acc), 4),
            }

        # Per-layer precision 
        # For each layer: when it fired, how often was it a real failure?
        layer_stats: dict[str, dict] = {}
        for doc in labeled:
            fired   = doc.get("layers_fired", [])
            correct = doc.get("fie_was_correct", False)  
            for layer in fired:
                if layer not in layer_stats:
                    layer_stats[layer] = {"fired": 0, "correct_fires": 0}
                layer_stats[layer]["fired"] += 1
                if correct:
                    layer_stats[layer]["correct_fires"] += 1

        layer_precision = {}
        for layer, stats in layer_stats.items():
            n = stats["fired"]
            layer_precision[layer] = {
                "times_fired":   n,
                "precision":     round(stats["correct_fires"] / n, 4) if n else 0.0,
                "current_weight": _KNOWN_WEIGHTS.get(layer, "unknown"),
            }

        return {
            "total_labeled":     total,
            "overall_accuracy":  overall_acc,
            "calibration":       calibration,
            "layer_precision":   layer_precision,
            "interpretation": (
                "calibration_error < 0.05 = well calibrated. "
                "layer precision = how often that layer firing was a real failure. "
                "Use layer precision values to update DomainCritic._WEIGHTS."
            ),
        }

    except Exception as exc:
        logger.warning("Calibration stats failed: %s", exc)
        return {"error": str(exc)}


# Current hardcoded weights 
_KNOWN_WEIGHTS = {
    "contradiction_signal":  0.40,
    "self_contradiction":    0.35,
    "hedge_detection":       0.15,
    "temporal_detection":    0.10,
    "external_verification": 0.45,
}
