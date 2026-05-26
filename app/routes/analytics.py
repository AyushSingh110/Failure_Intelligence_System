from __future__ import annotations
import logging
from collections import defaultdict
from fastapi import APIRouter, Header, Request
from app.limiter import rate_limit
from app.routes._helpers import get_signal_logs_collection
from app.schemas import TrendResponse, ClusterSummaryResponse, TelemetryPing
from app.auth_guard import require_admin
logger = logging.getLogger(__name__)
router = APIRouter()


# Trend and clusters

@router.get("/trend", response_model=TrendResponse)
def get_trend() -> TrendResponse:
    from engine.evolution.tracker import evolution_tracker
    return TrendResponse(**evolution_tracker.trend_summary())


@router.get("/clusters", response_model=ClusterSummaryResponse)
def get_clusters() -> ClusterSummaryResponse:
    from engine.archetypes.clustering import archetype_registry
    clusters = archetype_registry.summarize()
    return ClusterSummaryResponse(total_clusters=len(clusters), clusters=clusters)


@router.delete("/clusters/reset", response_model=dict)
def reset_clusters() -> dict:
    from engine.archetypes.clustering import ArchetypeClusterRegistry
    import engine.archetypes.clustering as clustering_module
    clustering_module.archetype_registry = ArchetypeClusterRegistry()
    return {"status": "reset", "message": "Archetype registry cleared"}


# Telemetry

@router.post("/telemetry", response_model=dict)
@rate_limit("30/minute")
def receive_telemetry(request: Request, body: TelemetryPing) -> dict:
    """
    Receives anonymized usage pings from fie-sdk clients (FIE_TELEMETRY=true).
    No prompt text, no API keys, no PII — only event type and boolean signals.
    """
    try:
        from storage.database import _db, _fallback_mode
        from datetime import datetime

        clean = body.model_dump()
        clean["received_at"] = datetime.utcnow().isoformat()

        if not _fallback_mode and _db is not None:
            _db["sdk_telemetry"].insert_one(clean)
    except Exception:
        pass  # telemetry failures must never surface to SDK users
    return {"status": "ok"}


# Analytics (admin)

@router.get("/analytics/usage", response_model=dict)
def analytics_usage(
    days:          int = 7,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """Request volume, latency, and failure detection rate over the past N days."""
    require_admin(authorization, x_api_key)
    col = get_signal_logs_collection()
    if col is None:
        return {"error": "MongoDB unavailable"}

    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    try:
        docs = list(col.find(
            {"timestamp": {"$gte": cutoff}},
            {"timestamp": 1, "high_failure_risk": 1, "fix_applied": 1,
             "question_type": 1, "model_version": 1},
        ))

        total    = len(docs)
        failures = sum(1 for d in docs if d.get("high_failure_risk"))
        fixes    = sum(1 for d in docs if d.get("fix_applied"))

        daily: dict = defaultdict(lambda: {"requests": 0, "failures": 0, "fixes": 0})
        for d in docs:
            day = (d.get("timestamp") or "")[:10]
            daily[day]["requests"] += 1
            if d.get("high_failure_risk"): daily[day]["failures"] += 1
            if d.get("fix_applied"):       daily[day]["fixes"] += 1

        qt_counts: dict = defaultdict(int)
        for d in docs:
            qt_counts[d.get("question_type", "UNKNOWN")] += 1

        return {
            "period_days":             days,
            "total_requests":          total,
            "failure_detections":      failures,
            "auto_fixes":              fixes,
            "failure_rate":            round(failures / total, 4) if total else 0.0,
            "fix_rate":                round(fixes / total, 4) if total else 0.0,
            "daily_breakdown":         dict(sorted(daily.items())),
            "question_type_breakdown": dict(qt_counts),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/model-performance", response_model=dict)
def analytics_model_performance(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """XGBoost vs POET agreement rate, accuracy from real user feedback."""
    require_admin(authorization, x_api_key)
    col = get_signal_logs_collection()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        all_docs     = list(col.find({}, {
            "high_failure_risk": 1, "classifier_probability": 1,
            "question_type": 1, "fie_was_correct": 1,
            "feedback_received": 1, "model_version": 1,
        }))
        labeled_docs = [d for d in all_docs if d.get("feedback_received")]

        total       = len(all_docs)
        n_labeled   = len(labeled_docs)
        correct     = sum(1 for d in labeled_docs if d.get("fie_was_correct"))
        overall_acc = round(correct / n_labeled, 4) if n_labeled else None

        with_prob    = sum(1 for d in all_docs if d.get("classifier_probability") is not None)
        xgb_coverage = round(with_prob / total, 4) if total else 0.0

        qt_stats: dict = defaultdict(lambda: {"total": 0, "labeled": 0, "correct": 0})
        for d in all_docs:
            qt = d.get("question_type", "UNKNOWN")
            qt_stats[qt]["total"] += 1
            if d.get("feedback_received"):
                qt_stats[qt]["labeled"] += 1
                if d.get("fie_was_correct"):
                    qt_stats[qt]["correct"] += 1

        qt_summary = {
            qt: {
                "total_requests": s["total"],
                "labeled":        s["labeled"],
                "accuracy":       round(s["correct"] / s["labeled"], 4) if s["labeled"] else None,
            }
            for qt, s in qt_stats.items()
        }

        ver_counts: dict = defaultdict(int)
        for d in all_docs:
            ver_counts[d.get("model_version", "unknown")] += 1

        return {
            "total_requests":     total,
            "total_labeled":      n_labeled,
            "overall_accuracy":   overall_acc,
            "xgboost_coverage":   xgb_coverage,
            "per_question_type":  qt_summary,
            "model_version_dist": dict(ver_counts),
            "note": (
                "accuracy = % of labeled examples where FIE verdict matched user feedback. "
                "xgboost_coverage = % of requests where classifier ran (vs POET fallback)."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/calibration", response_model=dict)
def analytics_calibration(
    question_type: str = "all",
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Confidence calibration curves from real user feedback.
    Pass ?question_type=FACTUAL for per-type curves.
    Returns points ready for a calibration plot (predicted vs actual accuracy).
    """
    require_admin(authorization, x_api_key)
    col = get_signal_logs_collection()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        query: dict = {"feedback_received": True}
        if question_type.upper() != "ALL":
            query["question_type"] = question_type.upper()

        labeled = list(col.find(query, {
            "classifier_probability": 1, "fie_was_correct": 1, "question_type": 1
        }))

        if not labeled:
            return {"error": "No labeled examples found", "question_type": question_type}

        n_bins  = 10
        bins: dict = {i: {"predicted_sum": 0.0, "correct": 0, "total": 0} for i in range(n_bins)}

        for doc in labeled:
            prob = doc.get("classifier_probability")
            if prob is None:
                continue
            b = min(int(prob * n_bins), n_bins - 1)
            bins[b]["predicted_sum"] += prob
            bins[b]["correct"]       += int(doc.get("fie_was_correct", False))
            bins[b]["total"]         += 1

        calibration_points = []
        ece      = 0.0
        n_total  = len(labeled)

        for b, data in bins.items():
            n = data["total"]
            if n == 0:
                continue
            pred_avg = data["predicted_sum"] / n
            actual   = data["correct"] / n
            ece     += (n / n_total) * abs(pred_avg - actual)
            calibration_points.append({
                "bin":               b,
                "predicted_avg":     round(pred_avg, 4),
                "actual_accuracy":   round(actual, 4),
                "calibration_error": round(abs(pred_avg - actual), 4),
                "n_examples":        n,
            })

        from engine.fie_config import get_all_thresholds, get_config_version
        return {
            "question_type":      question_type,
            "n_labeled":          n_total,
            "ece":                round(ece, 4),
            "interpretation":     "ECE < 0.05 = well calibrated. ECE > 0.10 = needs recalibration.",
            "calibration_points": calibration_points,
            "current_thresholds": get_all_thresholds(),
            "config_version":     get_config_version(),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/question-breakdown", response_model=dict)
def analytics_question_breakdown(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """Per-question-type breakdown: volume, failure rate, fix rate, escalation rate, avg XGB prob."""
    require_admin(authorization, x_api_key)
    col = get_signal_logs_collection()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        docs = list(col.find({}, {
            "question_type": 1, "high_failure_risk": 1, "fix_applied": 1,
            "requires_escalation": 1, "classifier_probability": 1, "gt_source": 1,
        }))

        stats: dict = defaultdict(lambda: {
            "total": 0, "failures": 0, "fixes": 0,
            "escalations": 0, "prob_sum": 0.0, "prob_count": 0,
            "gt_sources": defaultdict(int),
        })

        for d in docs:
            qt = d.get("question_type", "UNKNOWN")
            stats[qt]["total"] += 1
            if d.get("high_failure_risk"):    stats[qt]["failures"] += 1
            if d.get("fix_applied"):          stats[qt]["fixes"] += 1
            if d.get("requires_escalation"):  stats[qt]["escalations"] += 1
            prob = d.get("classifier_probability")
            if prob is not None:
                stats[qt]["prob_sum"]   += prob
                stats[qt]["prob_count"] += 1
            stats[qt]["gt_sources"][d.get("gt_source", "none")] += 1

        result = {}
        for qt, s in stats.items():
            n = s["total"]
            result[qt] = {
                "total_requests":  n,
                "failure_rate":    round(s["failures"] / n, 4) if n else 0.0,
                "fix_rate":        round(s["fixes"] / n, 4) if n else 0.0,
                "escalation_rate": round(s["escalations"] / n, 4) if n else 0.0,
                "avg_xgb_prob":    round(s["prob_sum"] / s["prob_count"], 4) if s["prob_count"] else None,
                "top_gt_sources":  dict(sorted(s["gt_sources"].items(), key=lambda x: -x[1])[:3]),
            }

        return {"breakdown": result, "total_logged": len(docs)}
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/paper-metrics", response_model=dict)
def analytics_paper_metrics(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    All metrics needed for the research paper results section in one call.
    Combine with notebook-generated AUC figures for the complete results table.
    """
    require_admin(authorization, x_api_key)
    col = get_signal_logs_collection()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        from datetime import datetime
        from storage.signal_logger import get_calibration_stats
        from engine.fie_config import (
            get_all_thresholds, get_config_version,
            MODEL_VERSION, MODEL_TRAINED, RECALIBRATION_INTERVAL,
        )

        calib_stats = get_calibration_stats()

        pipeline_docs     = list(col.find({}, {"gt_source": 1, "question_type": 1, "fix_applied": 1}))
        gt_source_counts: dict = defaultdict(int)
        qt_counts: dict        = defaultdict(int)
        for d in pipeline_docs:
            gt_source_counts[d.get("gt_source", "none")] += 1
            qt_counts[d.get("question_type", "UNKNOWN")] += 1

        labeled = list(col.find(
            {"feedback_received": True, "classifier_probability": {"$ne": None}},
            {"classifier_probability": 1, "fie_was_correct": 1},
        ))
        ece      = 0.0
        n_labeled= len(labeled)
        if n_labeled > 0:
            n_bins = 10
            bins: dict = {i: {"pred": 0.0, "correct": 0, "total": 0} for i in range(n_bins)}
            for doc in labeled:
                prob = doc.get("classifier_probability", 0.0) or 0.0
                b    = min(int(prob * n_bins), n_bins - 1)
                bins[b]["pred"]    += prob
                bins[b]["correct"] += int(doc.get("fie_was_correct", False))
                bins[b]["total"]   += 1
            for b, data in bins.items():
                n = data["total"]
                if n:
                    pred_avg = data["pred"] / n
                    actual   = data["correct"] / n
                    ece     += (n / n_labeled) * abs(pred_avg - actual)

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "model": {
                "version":                MODEL_VERSION,
                "trained":                MODEL_TRAINED,
                "threshold_mode":         "per_question_type_auto_calibrated",
                "thresholds":             get_all_thresholds(),
                "config_version":         get_config_version(),
                "recalibration_interval": RECALIBRATION_INTERVAL,
            },
            "live_accuracy": {
                "total_labeled":                    calib_stats.get("total_labeled", 0),
                "overall_accuracy":                 calib_stats.get("overall_accuracy"),
                "ece":                              round(ece, 4),
                "calibration_by_confidence_bucket": calib_stats.get("calibration", {}),
            },
            "layer_precision": calib_stats.get("layer_precision", {}),
            "pipeline_routing": {
                "total_requests":       len(pipeline_docs),
                "gt_source_counts":     dict(gt_source_counts),
                "question_type_counts": dict(qt_counts),
            },
            "how_to_cite": (
                "Use overall_accuracy, ece, and calibration_by_confidence_bucket "
                "for the calibration analysis section. "
                "Use pipeline_routing.gt_source_counts to show GT pipeline source distribution. "
                "Cross-reference with notebook AUC figures for the full results table."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/sdk-telemetry", response_model=dict)
def analytics_sdk_telemetry(
    days:          int = 30,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """Admin view of anonymized SDK usage telemetry from opted-in fie-sdk clients."""
    require_admin(authorization, x_api_key)
    try:
        from storage.database import _db, _fallback_mode
        from datetime import datetime, timedelta

        if _fallback_mode or _db is None:
            return {"error": "MongoDB unavailable"}

        col    = _db["sdk_telemetry"]
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        docs   = list(col.find({"received_at": {"$gte": cutoff}}, {"_id": 0}))
        total  = len(docs)

        if total == 0:
            return {
                "period_days": days,
                "total_pings": 0,
                "note": "No telemetry pings received. SDK users must set FIE_TELEMETRY=true to opt in.",
            }

        event_counts: dict   = defaultdict(int)
        version_counts: dict = defaultdict(int)
        qt_counts: dict      = defaultdict(int)
        mode_counts: dict    = defaultdict(int)
        for d in docs:
            event_counts[d.get("event", "unknown")]       += 1
            version_counts[d.get("sdk_version", "unknown")] += 1
            qt_counts[d.get("question_type", "UNKNOWN")]  += 1
            mode_counts[d.get("mode", "unknown")]          += 1

        monitor_pings = [d for d in docs if d.get("event") == "monitor_call"]
        n_monitor  = len(monitor_pings)
        n_failures = sum(1 for d in monitor_pings if d.get("high_failure_risk"))
        n_fixes    = sum(1 for d in monitor_pings if d.get("fix_applied"))

        return {
            "period_days":        days,
            "total_pings":        total,
            "event_breakdown":    dict(event_counts),
            "sdk_version_dist":   dict(version_counts),
            "question_type_dist": dict(qt_counts),
            "mode_dist":          dict(mode_counts),
            "field_failure_rate": round(n_failures / n_monitor, 4) if n_monitor else None,
            "field_fix_rate":     round(n_fixes    / n_monitor, 4) if n_monitor else None,
            "monitor_call_count": n_monitor,
            "note": (
                "All pings are anonymized — no prompts or API keys are stored. "
                "field_failure_rate = % of monitor calls where high_failure_risk=True from real SDK users."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}
