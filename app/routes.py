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


def _build_failure_signal(
    model_outputs:    list[str],
    primary_output:   str,
    secondary_output: str,
) -> FailureSignalVector:
    consistency   = compute_consistency(model_outputs)
    entropy_score = compute_entropy(model_outputs)
    ensemble      = compute_disagreement(primary_output, secondary_output)

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
    """
    Phase 1 analysis endpoint — used by the Streamlit dashboard Analyze page.
    Returns failure_signal_vector + archetype label + embedding_distance
    so the UI can display all three without needing a separate call.
    """
    signal    = _build_failure_signal(
        body.model_outputs,
        body.primary_output,
        body.secondary_output,
    )
    archetype = label_failure_archetype(signal)
    embedding = compute_embedding_distance(body.primary_output, body.secondary_output)
    return {
        "failure_signal_vector": signal.model_dump(),
        "archetype":             archetype,
        "embedding_distance":    embedding["embedding_distance"],
    }


@router.post("/track-and-analyze", response_model=dict)
def track_and_analyze(request: InferenceRequest, body: AnalyzeRequest) -> dict:
    """
    Combined endpoint: stores the inference AND returns its failure signal
    in one round trip — useful for real-time monitoring pipelines.
    """
    signal  = _build_failure_signal(
        body.model_outputs, body.primary_output, body.secondary_output,
    )
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


@router.get("/inferences/{request_id}", response_model=InferenceRequest)
def get_inference(request_id: str) -> InferenceRequest:
    record = get_inference_by_id(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Inference '{request_id}' not found")
    return record


# ── Phase 2 endpoints ─────────────────────────────────────────────────────

@router.post("/analyze/v2", response_model=ArchetypeAnalysisResponse)
def analyze_v2(body: AnalyzeRequest) -> ArchetypeAnalysisResponse:
    """
    Full Phase 2 analysis:
    - Extracts failure signal (Phase 1)
    - Assigns to archetype cluster
    - Returns detailed label with conditions
    - Updates evolution tracker
    """
    result = failure_agent.run_full(
        body.model_outputs,
        body.primary_output,
        body.secondary_output,
    )
    return ArchetypeAnalysisResponse(
        failure_signal_vector=FailureSignalVector(**result["failure_signal_vector"]),
        cluster_assignment=ClusterAssignment(**result["cluster_assignment"]),
        label_detail=LabelResult(**result["label_detail"]),
        embedding_distance=result["embedding_distance"],
        trend_summary=result["trend_summary"],
    )


@router.get("/trend", response_model=TrendResponse)
def get_trend() -> TrendResponse:
    """Returns the current EMA-based evolution trend from the tracker."""
    summary = evolution_tracker.trend_summary()
    return TrendResponse(**summary)


@router.get("/clusters", response_model=ClusterSummaryResponse)
def get_clusters() -> ClusterSummaryResponse:
    """Returns all known failure archetypes and their cluster statistics."""
    clusters = archetype_registry.summarize()
    return ClusterSummaryResponse(
        total_clusters=len(clusters),
        clusters=clusters,
    )


@router.delete("/clusters/reset", response_model=dict)
def reset_clusters() -> dict:
    """
    Resets the archetype registry. Useful for testing or after model redeployment.
    """
    from engine.archetypes.clustering import ArchetypeClusterRegistry
    import engine.archetypes.clustering as clustering_module
    clustering_module.archetype_registry = ArchetypeClusterRegistry()
    return {"status": "reset", "message": "Archetype registry cleared"}
