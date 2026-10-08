from __future__ import annotations
import logging
from collections import defaultdict
from fastapi import APIRouter, Depends, Request
from app.limiter import rate_limit
from app.schemas import TrendResponse, ClusterSummaryResponse, TelemetryPing
from app.auth_guard import Principal, ensure_principal, public, require_platform_admin, require_tenant
from app.tenancy import TenantScope, tenant_analytics
from storage.tenant_store import (
    calibration_stats_all_tenants,
    insert_sdk_telemetry,
    sdk_telemetry_since_all_tenants,
    signal_logs_collection_all_tenants,
)
logger = logging.getLogger(__name__)
router = APIRouter()


# Trend and clusters

# Each tenant has its own trend and its own clusters. A cluster centroid carries
# the model's normalised answers, so a shared registry showed one tenant's
# outputs to every caller.

@router.get("/trend", response_model=TrendResponse)
def get_trend(principal: Principal = Depends(require_tenant)) -> TrendResponse:
    analytics = tenant_analytics.for_scope(TenantScope(ensure_principal(principal)))
    return TrendResponse(**analytics.trend_summary())


@router.get("/clusters", response_model=ClusterSummaryResponse)
def get_clusters(principal: Principal = Depends(require_tenant)) -> ClusterSummaryResponse:
    analytics = tenant_analytics.for_scope(TenantScope(ensure_principal(principal)))
    clusters = analytics.summarize()
    return ClusterSummaryResponse(total_clusters=len(clusters), clusters=clusters)


@router.delete("/clusters/reset", response_model=dict)
def reset_clusters(principal: Principal = Depends(require_tenant)) -> dict:
    """Clears the caller's own clusters. No other tenant is affected."""
    tenant_analytics.for_scope(TenantScope(ensure_principal(principal))).reset_clusters()
    return {"status": "reset", "message": "Archetype registry cleared"}


# Telemetry

@router.post("/telemetry", response_model=dict, dependencies=[Depends(public)])
@rate_limit("30/minute")
def receive_telemetry(request: Request, body: TelemetryPing) -> dict:
    """
    Receives anonymized usage pings from fie-sdk clients (FIE_TELEMETRY=true).
    No prompt text, no API keys, no PII — only event type and boolean signals.
    """
    try:
        from datetime import datetime

        clean = body.model_dump()
        clean["received_at"] = datetime.utcnow().isoformat()
        insert_sdk_telemetry(clean)
    except Exception as exc:
        logger.warning(
            "degraded capability=receive_telemetry impact='this optional step was skipped' "
            "reason=%s", type(exc).__name__,
        )
    return {"status": "ok"}


# Analytics (admin)

@router.get("/analytics/usage", response_model=dict)
def analytics_usage(
    days:          int = 7,
    principal: Principal = Depends(require_platform_admin),
) -> dict:
    """Request volume, latency, and failure detection rate over the past N days."""
    col = signal_logs_collection_all_tenants()
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
        logger.warning(
            "degraded capability=analytics_usage impact='this optional step was skipped' "
            "reason=%s", type(exc).__name__,
        )
        return {"error": "analytics unavailable"}


@router.get("/analytics/model-performance", response_model=dict)
def analytics_model_performance(
    principal: Principal = Depends(require_platform_admin),
) -> dict:
    """XGBoost vs POET agreement rate, accuracy from real user feedback."""
    col = signal_logs_collection_all_tenants()
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
        logger.warning(
            "degraded capability=analytics_model_performance impact='this optional step was skipped' "
            "reason=%s", type(exc).__name__,
        )
        return {"error": "analytics unavailable"}


@router.get("/analytics/calibration", response_model=dict)
def analytics_calibration(
    question_type: str = "all",
    principal: Principal = Depends(require_platform_admin),
) -> dict:
    """
    Confidence calibration curves from real user feedback.
    Pass ?question_type=FACTUAL for per-type curves.
    Returns points ready for a calibration plot (predicted vs actual accuracy).
    """
    col = signal_logs_collection_all_tenants()
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
        logger.warning(
            "degraded capability=analytics_calibration impact='this optional step was skipped' "
            "reason=%s", type(exc).__name__,
        )
        return {"error": "analytics unavailable"}


@router.get("/analytics/question-breakdown", response_model=dict)
def analytics_question_breakdown(
    principal: Principal = Depends(require_platform_admin),
) -> dict:
    """Per-question-type breakdown: volume, failure rate, fix rate, escalation rate, avg XGB prob."""
    col = signal_logs_collection_all_tenants()
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
        logger.warning(
            "degraded capability=analytics_question_breakdown impact='this optional step was skipped' "
            "reason=%s", type(exc).__name__,
        )
        return {"error": "analytics unavailable"}


@router.get("/analytics/paper-metrics", response_model=dict)
def analytics_paper_metrics(
    principal: Principal = Depends(require_platform_admin),
) -> dict:
    """
    All metrics needed for the research paper results section in one call.
    Combine with notebook-generated AUC figures for the complete results table.
    """
    col = signal_logs_collection_all_tenants()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        from datetime import datetime
        from engine.fie_config import (
            get_all_thresholds, get_config_version,
            MODEL_VERSION, MODEL_TRAINED, RECALIBRATION_INTERVAL,
        )

        calib_stats = calibration_stats_all_tenants()

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
        logger.warning(
            "degraded capability=analytics_paper_metrics impact='this optional step was skipped' "
            "reason=%s", type(exc).__name__,
        )
        return {"error": "analytics unavailable"}


@router.get("/analytics/sdk-telemetry", response_model=dict)
def analytics_sdk_telemetry(
    days:          int = 30,
    principal: Principal = Depends(require_platform_admin),
) -> dict:
    """Admin view of anonymized SDK usage telemetry from opted-in fie-sdk clients."""
    try:
        from datetime import datetime, timedelta

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        docs   = sdk_telemetry_since_all_tenants(cutoff)
        if docs is None:
            return {"error": "MongoDB unavailable"}
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
        logger.warning(
            "degraded capability=analytics_sdk_telemetry impact='this optional step was skipped' "
            "reason=%s", type(exc).__name__,
        )
        return {"error": "analytics unavailable"}
