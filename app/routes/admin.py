"""
Admin and notification routes.

Endpoints
---------
GET  /admin/guard/config    — view current pre-flight guard settings
POST /admin/guard/config    — hot-update block mode and/or scan threshold
POST /notifications/digest  — compile a usage digest and email it to the tenant
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Body, Header, HTTPException, Query
from pydantic import BaseModel

from app.auth_guard import resolve_user, require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Guard config schemas ───────────────────────────────────────────────────────

class GuardConfigUpdate(BaseModel):
    """Body for POST /admin/guard/config."""
    block_enabled:  Optional[bool]  = None
    scan_threshold: Optional[float] = None


# ── GET /admin/guard/config ───────────────────────────────────────────────────

@router.get("/admin/guard/config", response_model=dict)
def get_guard_config(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Return the current pre-flight guard configuration.

    Response fields
    ---------------
    block_enabled   True = hard block mode (LLM never runs on detected attacks).
                    False = warn-only mode (detects and logs, LLM still runs).
    scan_threshold  Confidence floor for scan_prompt() to classify an attack.
    source          Where the active config was loaded from.
    """
    require_admin(authorization, x_api_key)
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


# ── POST /admin/guard/config ──────────────────────────────────────────────────

@router.post("/admin/guard/config", response_model=dict)
def update_guard_config(
    body:          GuardConfigUpdate,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Hot-update the pre-flight guard at runtime — no restart needed.

    Both fields are optional; omit any field to leave it unchanged.

    Examples
    --------
    Switch to warn-only mode (keep detecting but stop blocking):
        {"block_enabled": false}

    Re-enable blocking and tighten the threshold:
        {"block_enabled": true, "scan_threshold": 0.55}

    Just lower the threshold (block mode unchanged):
        {"scan_threshold": 0.40}
    """
    require_admin(authorization, x_api_key)

    if body.scan_threshold is not None:
        t = float(body.scan_threshold)
        if not (0.0 < t < 1.0):
            raise HTTPException(
                status_code=422,
                detail="scan_threshold must be between 0.0 and 1.0 (exclusive).",
            )

    from engine.fie_config import update_preflight_config, update_scan_threshold, get_preflight_config

    if body.scan_threshold is not None:
        update_scan_threshold(body.scan_threshold)

    result = update_preflight_config(block_enabled=body.block_enabled)

    logger.info(
        "GUARD_CONFIG_UPDATE | block_enabled=%s scan_threshold=%.4f",
        result["block_enabled"], result["scan_threshold"],
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
    days:          int  = Query(default=7, ge=1, le=90),
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Compile a usage digest for the authenticated tenant and email it via SendGrid.
    Call on a schedule (e.g. weekly cron) or on-demand.
    Returns a summary dict regardless of email delivery status.
    """
    from app.notifications import notify_weekly_digest

    current_user = resolve_user(authorization, x_api_key)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        from storage.database import get_inferences_for_tenant
        inferences = get_inferences_for_tenant(current_user["tenant_id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

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
        tenant_id     = current_user["tenant_id"],
        total         = total,
        high_risk     = high_risk,
        attacks       = attacks,
        fix_applied   = fix_applied,
        escalations   = escalations,
        top_archetype = top_archetype,
        period_days   = days,
        to            = current_user.get("email"),
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
        "recipient":     current_user.get("email", "—"),
        "note": (
            "Email delivery requires SENDGRID_API_KEY and NOTIFICATION_EMAIL in .env. "
            "Stats are returned regardless."
        ),
    }
