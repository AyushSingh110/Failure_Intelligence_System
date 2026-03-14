from fastapi import APIRouter, HTTPException

from app.schemas import (
    InferenceRequest,
    AnalyzeRequest,
    ArchetypeAnalysisResponse,
    TrackResponse,
    TrendResponse,
    ClusterSummaryResponse,
    FailureSignalVector,
    InferenceRecord,
    ClusterAssignment,
    LabelResult,
)
from storage.database import save_inference, get_all_inferences, get_inference_by_id
from engine.detector.consistency import compute_consistency
from engine.detector.entropy import compute_entropy
from engine.detector.ensemble import compute_disagreement
from engine.detector.embedding import compute_embedding_distance
from engine.archetypes.labeling import label_failure_archetype
from engine.archetypes.clustering import archetype_registry
from engine.evolution.tracker import evolution_tracker
from engine.agents.failure_agent import failure_agent
from config import get_settings

router   = APIRouter()
settings = get_settings()


def _build_failure_signal(model_outputs: list[str]) -> FailureSignalVector:
    """
    Builds FSV from all model outputs.
    Primary  = model_outputs[0]
    Secondary = model_outputs[1] (or [0] if only one provided)
    Ensemble disagreement is computed across ALL pairwise combinations.
    """
    primary_output   = model_outputs[0]
    secondary_output = model_outputs[1] if len(model_outputs) > 1 else model_outputs[0]

    consistency   = compute_consistency(model_outputs)
    entropy_score = compute_entropy(model_outputs)
    ensemble      = compute_disagreement(model_outputs)   # full list — all pairs

    high_failure_risk = (
        entropy_score >= settings.high_entropy_threshold
        or consistency["agreement_score"] <= settings.low_agreement_threshold
        or ensemble["disagreement"] is True
    )

    return FailureSignalVector(
        agreement_score=consistency["agreement_score"],
        fsd_score=consistency["fsd_score"],
        answer_counts=consistency["answer_counts"],
        entropy_score=entropy_score,
        ensemble_disagreement=ensemble["disagreement"],
        ensemble_similarity=ensemble["similarity_score"],
        high_failure_risk=high_failure_risk,
    )


# ── Phase 1 endpoints ─────────────────────────────────────────────────────

@router.post("/track", response_model=TrackResponse)
def track_inference(request: InferenceRequest) -> TrackResponse:
    success = save_inference(request)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store inference record")
    return TrackResponse(status="stored", request_id=request.request_id)


@router.post("/analyze", response_model=dict)
def analyze_outputs(body: AnalyzeRequest) -> dict:
    signal    = _build_failure_signal(body.model_outputs)
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
    signal  = _build_failure_signal(body.model_outputs)
    success = save_inference(request)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store inference record")
    return {
        "status":                "stored",
        "request_id":            request.request_id,
        "failure_signal_vector": signal.model_dump(),
    }


@router.get("/inferences", response_model=list[InferenceRequest])
def list_inferences() -> list[InferenceRequest]:
    return get_all_inferences()


@router.get("/inferences/grouped/by-question", response_model=dict)
def get_inferences_grouped_by_question() -> dict:
    all_records = get_all_inferences()
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
def get_inference(request_id: str) -> InferenceRequest:
    record = get_inference_by_id(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Inference '{request_id}' not found")
    return record


# ── Phase 2 endpoints ─────────────────────────────────────────────────────

@router.post("/analyze/v2", response_model=ArchetypeAnalysisResponse)
def analyze_v2(body: AnalyzeRequest) -> ArchetypeAnalysisResponse:
    result = failure_agent.run_full(body.model_outputs)   # list only — no primary/secondary
    return ArchetypeAnalysisResponse(
        failure_signal_vector=FailureSignalVector(**result["failure_signal_vector"]),
        cluster_assignment=ClusterAssignment(**result["cluster_assignment"]),
        label_detail=LabelResult(**result["label_detail"]),
        embedding_distance=result["embedding_distance"],
        trend_summary=result["trend_summary"],
    )


@router.get("/trend", response_model=TrendResponse)
def get_trend() -> TrendResponse:
    summary = evolution_tracker.trend_summary()
    return TrendResponse(**summary)


@router.get("/clusters", response_model=ClusterSummaryResponse)
def get_clusters() -> ClusterSummaryResponse:
    clusters = archetype_registry.summarize()
    return ClusterSummaryResponse(
        total_clusters=len(clusters),
        clusters=clusters,
    )


@router.delete("/clusters/reset", response_model=dict)
def reset_clusters() -> dict:
    from engine.archetypes.clustering import ArchetypeClusterRegistry
    import engine.archetypes.clustering as clustering_module
    clustering_module.archetype_registry = ArchetypeClusterRegistry()
    return {"status": "reset", "message": "Archetype registry cleared"}


# ── Phase 3 endpoint ──────────────────────────────────────────────────────

from app.schemas import DiagnosticRequest, DiagnosticResponse

@router.post("/diagnose", response_model=DiagnosticResponse)
def diagnose(body: DiagnosticRequest) -> DiagnosticResponse:
    return failure_agent.run_diagnostic(body)