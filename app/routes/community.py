"""
/api/v1/community — opt-in feedback from the public demo.

POST /community/feedback   → report a wrong verdict (public, rate-limited)
GET  /community/stats      → aggregate counts (public)
GET  /community/export     → full dataset as JSONL (admin only)

The POST is deliberately UNAUTHENTICATED. Requiring an account to report a
false positive would collect nothing: the people best placed to notice
over-blocking are strangers trying the demo, and an account wall is exactly the
friction that stops them. The trade-off is managed with rate limiting and a
closed set of report kinds rather than with auth.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth_guard import require_admin
from app.limiter import rate_limit

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    # Constrained to the two labels that are actually actionable. Free-text
    # categories degrade into unusable training labels within a week.
    kind: str = Field(..., pattern="^(false_positive|missed_attack)$")
    # What FIE returned for this prompt. Optional, because a caller integrating
    # the SDK may not have it — but without it the report is far less useful,
    # since "you got this wrong" cannot be acted on without the verdict.
    is_attack:    bool | None = None
    attack_type:  str | None = None
    confidence:   float | None = None
    layers_fired: list[str] | None = None
    layer_scores: dict[str, float] | None = None


class FeedbackResponse(BaseModel):
    ok: bool
    message: str


@router.post("/community/feedback", response_model=FeedbackResponse)
@rate_limit("20/hour")
def submit_feedback(request: Request, body: FeedbackRequest) -> FeedbackResponse:
    """
    Record an opt-in correction to a scan verdict.

    Rate-limited per IP. The limit is generous for a human reporting genuine
    mistakes and restrictive for a script; note that the IP is used only by the
    limiter and is never stored with the report.
    """
    from engine.demo_feedback import record_feedback

    result = record_feedback(
        prompt=body.prompt,
        kind=body.kind,
        scan_result={
            "is_attack":    body.is_attack,
            "attack_type":  body.attack_type,
            "confidence":   body.confidence,
            "layers_fired": body.layers_fired,
            "layer_scores": body.layer_scores,
        },
    )
    return FeedbackResponse(**result)


@router.get("/community/stats")
def feedback_stats() -> dict:
    """
    Public counts of community reports.

    Deliberately public: the whole point of the dataset is that it is open, and
    publishing the running total is a small accountability mechanism — anyone
    can check that reports are actually being kept.
    """
    from engine.demo_feedback import stats
    return stats()


@router.get("/community/export")
def export_feedback(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Full report dataset, for building the public HuggingFace release.

    Admin-only despite the data being intended for publication: the raw feed is
    unreviewed, and mistaken submissions must be removable before anything is
    published. Review first, then release.
    """
    require_admin(authorization, x_api_key)
    from engine.demo_feedback import export_all

    records = export_all()
    if not records:
        raise HTTPException(status_code=404, detail="No community feedback recorded yet.")
    return {"count": len(records), "records": records}
