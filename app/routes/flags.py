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
        # Stage or dismiss hard-positive candidate for PAIR retraining.
        try:
            from engine.hard_positive_collector import confirm_hard_positive, dismiss_candidate
            if body.label == "true_positive":
                confirm_hard_positive(event_id)
            else:
                dismiss_candidate(event_id)
        except Exception as _hpc_exc:
            logger.debug("hard_positive_collector wiring error (non-fatal): %s", _hpc_exc)
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


@router.get("/flags/hard-positives/stats")
def hard_positive_stats(authorization: str = Header(default="")):
    """Return hard-positive collection stats (admin only)."""
    require_admin(authorization)
    try:
        from engine.hard_positive_collector import get_stats
        return get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/flags/hard-positives/export")
def export_hard_positives(authorization: str = Header(default="")):
    """
    Export confirmed hard positives for PAIR retraining (admin only).
    Returns list of {event_id, prompt, flag_type, zone, confidence, confirmed_at}.
    Pass to scripts/retrain_pair_v4.py --hard-positives-path <file>.
    """
    require_admin(authorization)
    try:
        from engine.hard_positive_collector import export_for_retraining
        records = export_for_retraining()
        return {"count": len(records), "hard_positives": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _require_auth(authorization: str) -> None:
    """Minimal JWT check — reuse existing auth_guard."""
    try:
        from app.auth_guard import verify_token
        verify_token(authorization.removeprefix("Bearer ").strip())
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")
