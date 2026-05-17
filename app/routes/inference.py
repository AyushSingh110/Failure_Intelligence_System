"""
Inference tracking and analysis routes.

Covers Phase 1 (track/analyze) and Phase 2 (analyze/v2, diagnose) of the FIE pipeline.
All CRUD operations on inference records also live here.
"""
from __future__ import annotations

import csv
import io
import logging

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.routes._helpers import build_failure_signal
from app.schemas import (
    InferenceRequest,
    AnalyzeRequest,
    ArchetypeAnalysisResponse,
    TrackResponse,
    FailureSignalVector,
    InferenceRecord,
    ClusterAssignment,
    LabelResult,
    DiagnosticRequest,
    DiagnosticResponse,
)
from storage.database import (
    save_inference,
    get_all_inferences,
    get_all_inferences as get_all_inferences_admin,
    get_inference_by_id,
    get_inference_by_id_for_tenant,
    get_inferences_for_tenant,
    delete_inference_for_tenant,
    clear_inferences_for_tenant,
)
from engine.detector.embedding import compute_embedding_distance
from engine.archetypes.labeling import label_failure_archetype
from engine.agents.failure_agent import failure_agent
from app.auth_guard import require_user, resolve_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Phase 1 ───────────────────────────────────────────────────────────────────

@router.post("/track", response_model=TrackResponse)
def track_inference(request: InferenceRequest) -> TrackResponse:
    success = save_inference(request)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store inference record")
    return TrackResponse(status="stored", request_id=request.request_id)


@router.post("/analyze", response_model=dict)
def analyze_outputs(body: AnalyzeRequest) -> dict:
    signal    = build_failure_signal(body.model_outputs)
    archetype = label_failure_archetype(signal)
    primary   = body.model_outputs[0]
    secondary = body.model_outputs[1] if len(body.model_outputs) > 1 else body.model_outputs[0]
    embedding = compute_embedding_distance(primary, secondary)
    return {
        "failure_signal_vector": signal.model_dump(),
        "archetype":             archetype,
        "embedding_distance":    embedding["embedding_distance"],
    }


@router.post("/track-and-analyze", response_model=dict)
def track_and_analyze(request: InferenceRequest, body: AnalyzeRequest) -> dict:
    signal  = build_failure_signal(body.model_outputs)
    success = save_inference(request)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store inference record")
    return {
        "status":                "stored",
        "request_id":            request.request_id,
        "failure_signal_vector": signal.model_dump(),
    }


# ── Inference record CRUD ─────────────────────────────────────────────────────

@router.get("/inferences", response_model=list[InferenceRequest])
def list_inferences(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> list[InferenceRequest]:
    user = require_user(authorization, x_api_key)
    if user.get("is_admin", False):
        return get_all_inferences_admin()
    return get_inferences_for_tenant(user["tenant_id"])


@router.get("/inferences/export/csv")
def export_inferences_csv(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
):
    """Download all inferences for the authenticated tenant as a CSV file."""
    user = require_user(authorization, x_api_key)
    records = (
        get_all_inferences_admin()
        if user.get("is_admin", False)
        else get_inferences_for_tenant(user["tenant_id"])
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "request_id", "timestamp", "model_name", "input_text", "output_text",
        "entropy", "agreement_score", "high_failure_risk", "is_adversarial",
        "archetype", "latency_ms", "tenant_id",
    ])
    for r in records:
        m = r.metrics or {}
        writer.writerow([
            r.request_id,
            r.timestamp.isoformat() if r.timestamp else "",
            r.model_name or "",
            (r.input_text  or "").replace("\n", " "),
            (r.output_text or "").replace("\n", " "),
            getattr(m, "entropy", "")        if hasattr(m, "entropy")        else m.get("entropy", "")        if isinstance(m, dict) else "",
            getattr(m, "agreement_score", "") if hasattr(m, "agreement_score") else m.get("agreement_score", "") if isinstance(m, dict) else "",
            getattr(m, "high_failure_risk", "") if hasattr(m, "high_failure_risk") else "",
            getattr(r, "is_adversarial", False),
            getattr(r, "archetype", ""),
            r.latency_ms or "",
            r.tenant_id  or "",
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fie_inferences.csv"},
    )


@router.get("/inferences/grouped/by-question", response_model=dict)
def get_inferences_grouped_by_question(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    user = require_user(authorization, x_api_key)
    all_records = (
        get_all_inferences_admin()
        if user.get("is_admin", False)
        else get_inferences_for_tenant(user["tenant_id"])
    )
    grouped: dict[str, list[dict]] = {}

    for record in all_records:
        question = record.input_text.strip()
        if not question:
            continue
        if question not in grouped:
            grouped[question] = []
        grouped[question].append({
            "model_name":    record.model_name,
            "model_version": record.model_version,
            "output_text":   record.output_text,
            "latency_ms":    record.latency_ms,
            "timestamp":     record.timestamp.isoformat(),
        })

    for question in grouped:
        grouped[question].sort(key=lambda r: r["model_name"])

    return grouped


@router.get("/inferences/{request_id}", response_model=InferenceRequest)
def get_inference(
    request_id:    str,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> InferenceRequest:
    user = require_user(authorization, x_api_key)
    record = (
        get_inference_by_id(request_id)
        if user.get("is_admin", False)
        else get_inference_by_id_for_tenant(request_id, user["tenant_id"])
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"Inference '{request_id}' not found")
    return record


@router.delete("/inferences/{request_id}", response_model=dict)
def delete_inference_record(
    request_id:    str,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """Delete a single inference record by request_id."""
    from storage.database import delete_inference
    user = require_user(authorization, x_api_key)
    success = (
        delete_inference(request_id)
        if user.get("is_admin", False)
        else delete_inference_for_tenant(request_id, user["tenant_id"])
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Inference '{request_id}' not found")
    return {"status": "deleted", "request_id": request_id}


@router.delete("/inferences", response_model=dict)
def clear_all_inferences(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """Delete ALL inference records. Admin deletes globally; tenant users delete their own."""
    from storage.database import delete_inference
    user = require_user(authorization, x_api_key)
    if user.get("is_admin", False):
        records = get_all_inferences_admin()
        count = sum(1 for r in records if delete_inference(r.request_id))
        return {"status": "cleared", "deleted_count": count}

    count = clear_inferences_for_tenant(user["tenant_id"])
    return {"status": "cleared", "deleted_count": count}


# ── Phase 2 ───────────────────────────────────────────────────────────────────

@router.post("/analyze/v2", response_model=ArchetypeAnalysisResponse)
def analyze_v2(body: AnalyzeRequest) -> ArchetypeAnalysisResponse:
    result = failure_agent.run_full(body.model_outputs)
    return ArchetypeAnalysisResponse(
        failure_signal_vector = FailureSignalVector(**result["failure_signal_vector"]),
        cluster_assignment    = ClusterAssignment(**result["cluster_assignment"]),
        label_detail          = LabelResult(**result["label_detail"]),
        embedding_distance    = result["embedding_distance"],
        trend_summary         = result["trend_summary"],
    )


@router.post("/diagnose", response_model=DiagnosticResponse)
def diagnose(
    body:          DiagnosticRequest,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> DiagnosticResponse:
    user     = resolve_user(authorization, x_api_key)
    response = failure_agent.run_diagnostic(body)
    if not (user and user.get("is_admin", False)):
        response.explanation_internal = None
    return response
