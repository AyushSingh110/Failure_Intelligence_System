"""
/api/v1/flags — feedback loop review queue. Platform admin only.

GET  /flags                         → paginated list of flagged events
POST /flags/{id}/label              → label as true_positive or false_positive
GET  /flags/export                  → confirmed TPs
GET  /flags/hard-positives/stats    → collection counts
GET  /flags/hard-positives/export   → confirmed hard positives

WHY PLATFORM ADMIN ONLY
-----------------------
A flagged event holds a prompt hash and an excerpt, and no tenant. The queue
therefore cannot be shown per tenant without showing one tenant another's
excerpts. And a label is not a private note: a true positive adds the prompt to
the fast-block set and a false positive to the allow set, for every tenant. That
is shared platform state, so only a platform admin changes it, and every change
is recorded.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from app.auth_guard import Principal, require_platform_admin
from app.security_events import emit

logger = logging.getLogger(__name__)
router = APIRouter()


class LabelRequest(BaseModel):
    label: str   # "true_positive" | "false_positive"


@router.get("/flags")
def list_flags(
    limit:          int = 50,
    offset:         int = 0,
    unlabeled_only: bool = True,
    principal:      Principal = Depends(require_platform_admin),
):
    """
    Return flagged events awaiting human review.
    Each event has: id, kind, flag_type, confidence, matched, timestamp, label.
    """
    try:
        from fie.feedback_store import list_events
        return {"events": list_events(unlabeled_only=unlabeled_only,
                                      limit=max(1, min(limit, 500)), offset=max(0, offset))}
    except Exception as exc:
        logger.error("list_flags failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Could not list flagged events")


@router.post("/flags/{event_id}/label")
def label_flag(
    event_id:  str,
    body:      LabelRequest,
    request:   Request,
    principal: Principal = Depends(require_platform_admin),
):
    """
    Label a flagged event. Label must be 'true_positive' or 'false_positive'.

    Side effects, for every tenant:
      true_positive  → prompt hash added to in-process fast-block set immediately
      false_positive → prompt hash added to in-process whitelist immediately
    Both sides are persisted to MongoDB so they survive restart.
    """
    if body.label not in ("true_positive", "false_positive"):
        raise HTTPException(status_code=400, detail="label must be 'true_positive' or 'false_positive'")
    try:
        from fie.feedback_store import apply_label
        found = apply_label(event_id, body.label)   # type: ignore[arg-type]
        if not found:
            raise HTTPException(status_code=404, detail="event not found")
        # Stage or dismiss hard-positive candidate for PAIR retraining.
        try:
            from engine.hard_positive_collector import confirm_hard_positive, dismiss_candidate
            if body.label == "true_positive":
                confirm_hard_positive(event_id)
            else:
                dismiss_candidate(event_id)
        except Exception as _hpc_exc:
            logger.debug("hard_positive_collector wiring error (non-fatal): %s", type(_hpc_exc).__name__)
        emit("admin.flag_labelled", outcome="changed", reason="flag_label",
             principal=principal, request=request, label=body.label)
        return {"status": "ok", "event_id": event_id, "label": body.label}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("label_flag failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Could not label the event")


@router.get("/flags/export")
def export_tps(
    principal: Principal = Depends(require_platform_admin),
):
    """Export all confirmed true positives as a list (admin only). Used for PAIR retraining."""
    try:
        from fie.feedback_store import export_confirmed_tps
        tps = export_confirmed_tps()
        return {"count": len(tps), "true_positives": tps}
    except Exception as exc:
        logger.error("export_tps failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Could not export flagged events")


@router.get("/flags/hard-positives/stats")
def hard_positive_stats(
    principal: Principal = Depends(require_platform_admin),
):
    """Return hard-positive collection stats (admin only)."""
    try:
        from engine.hard_positive_collector import get_stats
        return get_stats()
    except Exception as exc:
        logger.error("hard_positive_stats failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Could not read collection stats")


@router.get("/flags/hard-positives/export")
def export_hard_positives(
    request:   Request,
    principal: Principal = Depends(require_platform_admin),
):
    """
    Export confirmed hard positives for PAIR retraining (admin only).
    Returns list of {event_id, prompt, flag_type, zone, confidence, confirmed_at}.
    These are raw prompts of every caller, so the read is recorded.
    """
    try:
        from engine.hard_positive_collector import export_for_retraining
        records = export_for_retraining()
        emit("admin.cross_tenant_read", outcome="allowed", reason="hard_positives",
             principal=principal, request=request, resource="hard_positives")
        return {"count": len(records), "hard_positives": records}
    except Exception as exc:
        logger.error("export_hard_positives failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Could not export hard positives")
