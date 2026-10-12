from __future__ import annotations
import csv
import io
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.routes._helpers import build_failure_signal
from app.schemas import (
    InferenceRequest,
    AnalyzeRequest,
    ArchetypeAnalysisResponse,
    TrackResponse,
    FailureSignalVector,
    ClusterAssignment,
    LabelResult,
    DiagnosticRequest,
    DiagnosticResponse,
)
from storage.tenant_store import (
    InferenceIdConflict,
    TenantStore,
    get_inference_all_tenants,
    list_inferences_all_tenants,
)
from engine.detector.embedding import compute_embedding_distance
from engine.archetypes.labeling import label_failure_archetype
from engine.agents.failure_agent import failure_agent
from app.auth_guard import (
    Principal,
    authorize_cross_tenant_read,
    ensure_principal,
    is_current_admin,
    require_tenant,
)
from app.security_events import emit
from app.tenancy import TenantScope, tenant_analytics

logger = logging.getLogger(__name__)
router = APIRouter()


def _enforce_body_tenant(http_request: Request, principal: Principal, record: InferenceRequest) -> None:
    """
    The tenant of a stored record is the caller's, taken from the credential.

    The body may omit `tenant_id` or repeat the caller's own. Any other value is
    refused with 403 and nothing is written. The answer is the same whether or
    not the named tenant exists.
    """
    if "tenant_id" in record.model_fields_set and record.tenant_id != principal.tenant_id:
        emit(
            "authz.tenant_mismatch", severity="warning", outcome="denied", reason="tenant_mismatch",
            principal=principal, request=http_request, target_tenant_id=record.tenant_id,
        )
        raise HTTPException(status_code=403, detail="tenant_id does not match the authenticated tenant")


def _store_tracked(store: TenantStore, record: InferenceRequest) -> None:
    try:
        success = store.save_inference(record)
    except InferenceIdConflict:
        raise HTTPException(status_code=409, detail="request_id is not available; choose a different one")
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store inference record")


#Phase 1
@router.post("/track", response_model=TrackResponse)
def track_inference(
    request:      InferenceRequest,
    http_request: Request,
    principal:    Principal = Depends(require_tenant),
) -> TrackResponse:
    principal = ensure_principal(principal)
    _enforce_body_tenant(http_request, principal, request)
    _store_tracked(TenantStore(TenantScope(principal)), request)
    return TrackResponse(status="stored", request_id=request.request_id)


@router.post("/analyze", response_model=dict)
def analyze_outputs(
    body:      AnalyzeRequest,
    principal: Principal = Depends(require_tenant),
) -> dict:
    ensure_principal(principal)
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
def track_and_analyze(
    request:      InferenceRequest,
    body:         AnalyzeRequest,
    http_request: Request,
    principal:    Principal = Depends(require_tenant),
) -> dict:
    principal = ensure_principal(principal)
    _enforce_body_tenant(http_request, principal, request)
    signal = build_failure_signal(body.model_outputs)
    _store_tracked(TenantStore(TenantScope(principal)), request)
    return {
        "status":                "stored",
        "request_id":            request.request_id,
        "failure_signal_vector": signal.model_dump(),
    }


#Phase 2 schemas

def _records_for(
    http_request: Request, principal: Principal, all_tenants: bool, resource: str,
    limit: int = 200, offset: int = 0,
) -> list[InferenceRequest]:
    """
    The caller's own records. A platform admin sees other tenants only by asking
    for it (`all_tenants=true`); that read is authorized against the user store
    and recorded.
    """
    if all_tenants:
        authorize_cross_tenant_read(http_request, principal, resource)
        return list_inferences_all_tenants(limit=limit, offset=offset)
    return TenantStore(TenantScope(principal)).list_inferences(limit=limit, offset=offset)


@router.get("/inferences", response_model=list[InferenceRequest])
def list_inferences(
    http_request: Request,
    limit:        int = 100,
    offset:       int = 0,
    all_tenants:  bool = False,
    principal:    Principal = Depends(require_tenant),
) -> list[InferenceRequest]:
    principal = ensure_principal(principal)
    limit  = max(1, min(limit, 500))
    offset = max(0, offset)
    return _records_for(http_request, principal, all_tenants, "inferences", limit=limit, offset=offset)


@router.get("/inferences/export/csv")
def export_inferences_csv(
    http_request: Request,
    all_tenants:  bool = False,
    principal:    Principal = Depends(require_tenant),
):
    """Download all inferences for the authenticated tenant as a CSV file."""
    principal = ensure_principal(principal)
    records = _records_for(http_request, principal, all_tenants, "inferences.csv")

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
            getattr(r, "tenant_id", "") or "",
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fie_inferences.csv"},
    )


@router.get("/inferences/grouped/by-question", response_model=dict)
def get_inferences_grouped_by_question(
    http_request: Request,
    all_tenants:  bool = False,
    principal:    Principal = Depends(require_tenant),
) -> dict:
    principal = ensure_principal(principal)
    all_records = _records_for(http_request, principal, all_tenants, "inferences.grouped")
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
    request_id:   str,
    http_request: Request,
    all_tenants:  bool = False,
    principal:    Principal = Depends(require_tenant),
) -> InferenceRequest:
    principal = ensure_principal(principal)
    if all_tenants:
        authorize_cross_tenant_read(http_request, principal, "inference")
        record = get_inference_all_tenants(request_id)
    else:
        record = TenantStore(TenantScope(principal)).get_inference(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Inference '{request_id}' not found")
    return record


@router.delete("/inferences/{request_id}", response_model=dict)
def delete_inference_record(
    request_id: str,
    principal:  Principal = Depends(require_tenant),
) -> dict:
    """Delete one of the caller's own inference records. There is no cross-tenant delete."""
    principal = ensure_principal(principal)
    if not TenantStore(TenantScope(principal)).delete_inference(request_id):
        raise HTTPException(status_code=404, detail=f"Inference '{request_id}' not found")
    return {"status": "deleted", "request_id": request_id}


@router.delete("/inferences", response_model=dict)
def clear_all_inferences(
    principal: Principal = Depends(require_tenant),
) -> dict:
    """Delete all of the caller's own inference records. Platform admins included: own tenant only."""
    principal = ensure_principal(principal)
    count = TenantStore(TenantScope(principal)).clear_inferences()
    return {"status": "cleared", "deleted_count": count}


# Phase 2

@router.post("/analyze/v2", response_model=ArchetypeAnalysisResponse)
def analyze_v2(
    body:      AnalyzeRequest,
    principal: Principal = Depends(require_tenant),
) -> ArchetypeAnalysisResponse:
    analytics = tenant_analytics.for_scope(TenantScope(ensure_principal(principal)))
    result = failure_agent.run_full(body.model_outputs, registry=analytics, tracker=analytics)
    return ArchetypeAnalysisResponse(
        failure_signal_vector = FailureSignalVector(**result["failure_signal_vector"]),
        cluster_assignment    = ClusterAssignment(**result["cluster_assignment"]),
        label_detail          = LabelResult(**result["label_detail"]),
        embedding_distance    = result["embedding_distance"],
        trend_summary         = result["trend_summary"],
    )


@router.post("/diagnose", response_model=DiagnosticResponse)
def diagnose(
    body:      DiagnosticRequest,
    principal: Principal = Depends(require_tenant),
) -> DiagnosticResponse:
    principal = ensure_principal(principal)
    analytics = tenant_analytics.for_scope(TenantScope(principal))
    response  = failure_agent.run_diagnostic(body, registry=analytics, tracker=analytics)
    if not is_current_admin(principal):
        response.explanation_internal = None
    return response
