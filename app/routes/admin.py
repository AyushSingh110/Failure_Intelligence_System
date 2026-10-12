from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from app.auth_guard import Principal, ensure_principal, require_platform_admin, require_tenant
from app.security_events import emit
from app.tenancy import TenantScope
from storage.tenant_store import TenantStore
logger = logging.getLogger(__name__)
router = APIRouter()

# Guard config schemas
class GuardConfigUpdate(BaseModel):
    """Body for POST /admin/guard/config."""
    block_enabled:  Optional[bool]  = None
    scan_threshold: Optional[float] = None


# GET /admin/guard/config

@router.get("/admin/guard/config", response_model=dict)
def get_guard_config(
    principal: Principal = Depends(require_platform_admin),
) -> dict:

    #Return the current pre-flight guard configuration.
    from engine.fie_config import get_preflight_config, get_config_version
    cfg = get_preflight_config()
    return {
        "block_enabled":  cfg["block_enabled"],
        "scan_threshold": cfg["scan_threshold"],
        "config_version": get_config_version(),
        "note": (
            "block_enabled=true  → adversarial prompts are blocked before the LLM runs. "
            "block_enabled=false → warn-only, attacks are logged but LLM still runs. "
            "Update scan_threshold via this endpoint or SCAN_THRESHOLD env var."
        ),
    }


# POST /admin/guard/config
@router.post("/admin/guard/config", response_model=dict)
def update_guard_config(
    body:      GuardConfigUpdate,
    request:   Request,
    principal: Principal = Depends(require_platform_admin),
) -> dict:
    # This switch applies to every tenant. The admin right was read from the user
    # store for this request, and the change is recorded.

    if body.scan_threshold is not None:
        t = float(body.scan_threshold)
        if not (0.0 < t < 1.0):
            raise HTTPException(
                status_code=422,
                detail="scan_threshold must be between 0.0 and 1.0 (exclusive).",
            )

    from engine.fie_config import update_preflight_config, update_scan_threshold

    if body.scan_threshold is not None:
        update_scan_threshold(body.scan_threshold)

    result = update_preflight_config(block_enabled=body.block_enabled)

    logger.info(
        "GUARD_CONFIG_UPDATE | block_enabled=%s scan_threshold=%.4f",
        result["block_enabled"], result["scan_threshold"],
    )
    emit(
        "admin.config_change", severity="warning", outcome="changed", reason="guard_config",
        principal=principal, request=request,
        block_enabled=bool(result["block_enabled"]), scan_threshold=float(result["scan_threshold"]),
    )

    return {
        "status":         "updated",
        "block_enabled":  result["block_enabled"],
        "scan_threshold": result["scan_threshold"],
        "message": (
            f"Pre-flight guard is now in {'BLOCK' if result['block_enabled'] else 'WARN-ONLY'} mode "
            f"with scan_threshold={result['scan_threshold']:.4f}."
        ),
    }


@router.post("/notifications/digest", response_model=dict)
def send_weekly_digest(
    days:      int  = Query(default=7, ge=1, le=90),
    principal: Principal = Depends(require_tenant),
) -> dict:
    """
    Compile a usage digest for the authenticated tenant and email it via SendGrid.
    Call on a schedule (e.g. weekly cron) or on-demand.
    Returns a summary dict regardless of email delivery status.
    """
    from app.notifications import notify_weekly_digest

    principal = ensure_principal(principal)
    try:
        inferences = TenantStore(TenantScope(principal)).list_inferences()
    except Exception as exc:
        logger.error("digest: could not load inferences: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Could not load inferences")

    cutoff = datetime.utcnow() - timedelta(days=days)
    period = [
        r for r in inferences
        if r.timestamp and r.timestamp >= cutoff
    ] if inferences else []

    total       = len(period)
    high_risk   = sum(1 for r in period if (r.metrics.entropy if r.metrics else 0) > 0.75)
    attacks     = sum(1 for r in period if getattr(r, "is_adversarial", False))
    fix_applied = sum(1 for r in period if getattr(r, "fix_applied", False))
    escalations = sum(1 for r in period if getattr(r, "requires_escalation", False))

    archetype_counts: dict = {}
    for r in period:
        a = getattr(r, "archetype", "STABLE") or "STABLE"
        archetype_counts[a] = archetype_counts.get(a, 0) + 1
    top_archetype = max(archetype_counts, key=archetype_counts.get) if archetype_counts else "STABLE"

    notify_weekly_digest(
        tenant_id     = principal.tenant_id,
        total         = total,
        high_risk     = high_risk,
        attacks       = attacks,
        fix_applied   = fix_applied,
        escalations   = escalations,
        top_archetype = top_archetype,
        period_days   = days,
        to            = principal.subject,
    )

    return {
        "status":        "digest_sent",
        "period_days":   days,
        "total":         total,
        "high_risk":     high_risk,
        "attacks":       attacks,
        "fix_applied":   fix_applied,
        "escalations":   escalations,
        "top_archetype": top_archetype,
        "recipient":     principal.subject,
        "note": (
            "Email delivery requires SENDGRID_API_KEY and NOTIFICATION_EMAIL in .env. "
            "Stats are returned regardless."
        ),
    }
