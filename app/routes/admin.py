"""
Admin and notification routes.

Endpoints
---------
POST /notifications/digest  — compile a usage digest and email it to the tenant
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Header, HTTPException, Query

from app.auth_guard import resolve_user

logger = logging.getLogger(__name__)
router = APIRouter()


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
