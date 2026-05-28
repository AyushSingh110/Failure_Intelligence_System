"""
/api/v1/flags — feedback loop review queue.

GET  /flags              → paginated list of unreviewed flagged events
GET  /flags/all          → all events (including labeled)
POST /flags/{id}/label   → label as true_positive or false_positive
GET  /flags/export       → download confirmed TPs as JSONL (admin only)
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from app.auth_guard import require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


class LabelRequest(BaseModel):
    label: str   # "true_positive" | "false_positive"


@router.get("/flags")
def list_flags(
    limit:          int = 50,
    offset:         int = 0,
    unlabeled_only: bool = True,
    authorization:  str = Header(default=""),
):
    """
    Return flagged events awaiting human review.
    Each event has: id, kind, flag_type, confidence, matched, timestamp, label.
    """
    _require_auth(authorization)
    try:
        from fie.feedback_store import list_events
        return {"events": list_events(unlabeled_only=unlabeled_only, limit=limit, offset=offset)}
    except Exception as exc:
        logger.error("list_flags failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/flags/{event_id}/label")
def label_flag(
    event_id:      str,
    body:          LabelRequest,
    authorization: str = Header(default=""),
):
    """
    Label a flagged event. Label must be 'true_positive' or 'false_positive'.

    Side effects:
      true_positive  → prompt hash added to in-process fast-block set immediately
      false_positive → prompt hash added to in-process whitelist immediately
    Both sides are persisted to MongoDB so they survive restart.
    """
    _require_auth(authorization)
    if body.label not in ("true_positive", "false_positive"):
        raise HTTPException(status_code=400, detail="label must be 'true_positive' or 'false_positive'")
    try:
        from fie.feedback_store import apply_label
        found = apply_label(event_id, body.label)   # type: ignore[arg-type]
        if not found:
            raise HTTPException(status_code=404, detail="event not found")
        return {"status": "ok", "event_id": event_id, "label": body.label}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("label_flag failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/flags/export")
def export_tps(authorization: str = Header(default="")):
    """Export all confirmed true positives as a list (admin only). Used for PAIR retraining."""
    require_admin(authorization)
    try:
        from fie.feedback_store import export_confirmed_tps
        tps = export_confirmed_tps()
        return {"count": len(tps), "true_positives": tps}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _require_auth(authorization: str) -> None:
    """Minimal JWT check — reuse existing auth_guard."""
    try:
        from app.auth_guard import verify_token
        verify_token(authorization.removeprefix("Bearer ").strip())
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")
