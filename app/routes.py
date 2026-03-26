import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

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
from engine.detector.entropy import compute_entropy, compute_entropy_from_counts
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
    entropy_score = compute_entropy_from_counts(
        consistency["answer_counts"],
        len(model_outputs),
    )
    ensemble      = compute_disagreement(model_outputs)   

    ensemble_fires = (
        ensemble["disagreement"] is True
        and entropy_score > 0.0   # only count ensemble disagreement if model is actually inconsistent
    )
    high_failure_risk = (
        entropy_score >= settings.high_entropy_threshold
        or consistency["agreement_score"] <= settings.low_agreement_threshold
        or ensemble_fires
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


@router.delete("/inferences/{request_id}", response_model=dict)
def delete_inference_record(request_id: str) -> dict:
    """Deletes a single inference record by request_id."""
    from storage.database import delete_inference
    success = delete_inference(request_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Inference '{request_id}' not found")
    return {"status": "deleted", "request_id": request_id}


@router.delete("/inferences", response_model=dict)
def clear_all_inferences() -> dict:
    """
    Deletes ALL inference records from MongoDB.
    Use for resetting between test runs.
    """
    from storage.database import get_all_inferences, delete_inference
    records = get_all_inferences()
    count   = 0
    for r in records:
        if delete_inference(r.request_id):
            count += 1
    return {"status": "cleared", "deleted_count": count}


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


# ── Phase 4 — Real-time monitor endpoint ──────────────────────────────────

from app.schemas import MonitorRequest, MonitorResponse, OllamaModelResult

@router.post("/monitor", response_model=MonitorResponse)
def monitor(body: MonitorRequest) -> MonitorResponse:
    """
    Real-time monitoring endpoint — the core of the production system.

    Flow:
      1. Receive prompt + primary model output from user
      2. Fan out prompt to all Ollama shadow models in parallel
      3. Combine primary output + shadow outputs into model_outputs list
      4. Run full FIE pipeline (Phase 1 + 2 + optionally Phase 3)
      5. Return full analysis with all model responses

    This endpoint is what the @monitor decorator in the SDK calls.
    Users never need to paste anything manually — it all happens here.
    """
    from app.schemas import DiagnosticRequest

    # ── Step 1: Call shadow models (Groq or Ollama) ─────────────────────
    # Groq: fast cloud API, ~1 second, free tier 14,400 req/day
    # Ollama: local, private, slower — uncomment below to re-enable
    ollama_available   = False  # kept for schema compatibility
    shadow_results_raw = []

    if settings.groq_enabled and settings.groq_api_key:
        # ── Groq shadow models ──────────────────────────────────────
        from engine.groq_service import get_groq_service
        groq = get_groq_service()
        if groq:
            groq_results = groq.fan_out(body.prompt)
            shadow_results_raw = groq_results
            successful = [r for r in groq_results if r.success]
            ollama_available = len(successful) > 0
            logger.info(
                "Groq shadow models: %d/%d responded successfully",
                len(successful), len(groq_results),
            )
        else:
            logger.warning("Groq service unavailable — running without shadow models")

    # ── Ollama fallback (commented out — uncomment for local/private use) ──
    # elif settings.ollama_enabled:
    #     from engine.ollama_service import ollama_service
    #     if ollama_service.is_available():
    #         shadow_results_raw = ollama_service.fan_out(body.prompt)
    #         ollama_available   = True
    #     else:
    #         print("[monitor] Ollama not running. Start: ollama serve")

    else:
        logger.warning(
            "No shadow model provider configured. "
            "Add GROQ_API_KEY=gsk_xxx to .env file."
        )

    # ── Step 2: Build full model_outputs list ──────────────────────────
    # Primary output always goes first (index 0)
    model_outputs = [body.primary_output]

    # Add successful shadow model outputs
    for r in shadow_results_raw:
        if r.success and r.output_text:
            model_outputs.append(r.output_text)

    # Convert to schema format
    shadow_model_results = [
        OllamaModelResult(
            model_name=r.model_name,
            output_text=r.output_text,
            latency_ms=r.latency_ms,
            success=r.success,
            error=r.error,
        )
        for r in shadow_results_raw
    ]

    # ── Step 3: Run FIE pipeline ───────────────────────────────────────
    signal    = _build_failure_signal(model_outputs)
    archetype = label_failure_archetype(signal)
    primary   = model_outputs[0]
    secondary = model_outputs[1] if len(model_outputs) > 1 else model_outputs[0]
    embedding = compute_embedding_distance(primary, secondary)

    # Update archetype registry and EMA tracker
    archetype_registry.assign(signal)
    evolution_tracker.record(signal)

    # ── Step 4: Optionally run DiagnosticJury ──────────────────────────
    jury_verdict = None
    if body.run_full_jury:
        diag_request = DiagnosticRequest(
            prompt=body.prompt,
            model_outputs=model_outputs,
            latency_ms=body.latency_ms,
        )
        diag_response = failure_agent.run_diagnostic(diag_request)
        jury_verdict  = diag_response.jury

    # ── Step 5: Build failure summary ─────────────────────────────────
    if jury_verdict and jury_verdict.failure_summary:
        failure_summary = jury_verdict.failure_summary
    elif signal.high_failure_risk:
        failure_summary = (
            f"High failure risk detected — archetype: {archetype}. "
            f"Entropy: {signal.entropy_score:.3f}, "
            f"Agreement: {signal.agreement_score:.3f}"
        )
    else:
        failure_summary = f"Model outputs are stable — archetype: {archetype}"

    # ── Step 5b: Auto-fix engine ────────────────────────────────────────
    # Only run if failure detected AND jury has a primary verdict
    fix_result_schema = None
    if jury_verdict and jury_verdict.primary_verdict:
        try:
            from engine.fix_engine import apply_fix
            from engine.fix_engine import prompt_requires_live_data
            from app.schemas import FixResult as FixResultSchema

            primary_v  = jury_verdict.primary_verdict
            root_cause = primary_v.root_cause
            confidence = primary_v.confidence_score
            layers_fired = set((primary_v.evidence or {}).get("layers_fired", []))
            factual_roots = {"KNOWLEDGE_BOUNDARY_FAILURE", "FACTUAL_HALLUCINATION"}
            externally_verified_factual_mismatch = (
                root_cause in factual_roots
                and "external_verification" in layers_fired
            )

            if not (signal.high_failure_risk or externally_verified_factual_mismatch):
                raise RuntimeError("Auto-fix skipped: no actionable failure signal.")

            # Get successful shadow outputs
            shadow_texts = [
                r.output_text
                for r in shadow_model_results
                if r.success and r.output_text
            ]

            prompt_is_temporal = prompt_requires_live_data(body.prompt)
            prefer_rag_grounding = (
                externally_verified_factual_mismatch
                and not signal.high_failure_risk
                and not prompt_is_temporal
            )

            fix = None
            if not prefer_rag_grounding:
                fix = apply_fix(
                    prompt         = body.prompt,
                    primary_output = body.primary_output,
                    shadow_outputs = shadow_texts,
                    root_cause     = root_cause,
                    confidence     = confidence,
                    model_fn       = None,  # no model_fn on server side
                )

            if fix is not None:
                fix_result_schema = FixResultSchema(
                    fixed_output      = fix.fixed_output,
                    fix_applied       = fix.fix_applied,
                    fix_strategy      = fix.fix_strategy,
                    fix_explanation   = fix.fix_explanation,
                    original_output   = fix.original_output,
                    root_cause        = fix.root_cause,
                    fix_confidence    = fix.fix_confidence,
                    improvement_score = fix.improvement_score,
                    warning           = fix.warning,
                )

            if fix is not None and fix.fix_applied:
                logger.info(
                    "Auto-fix applied | strategy=%s | confidence=%.3f",
                    fix.fix_strategy, fix.fix_confidence,
                )
                # Update failure summary to reflect the fix
                failure_summary = (
                    f"⚡ AUTO-FIXED: {fix.fix_strategy} applied. "
                    f"{fix.fix_explanation[:150]}"
                )
            if (
                prefer_rag_grounding
                or (
                    (fix is None or not fix.fix_applied)
                    and root_cause in factual_roots
                    and not prompt_is_temporal
                )
            ):
                from engine.rag_grounder import ground_with_wikipedia

                rag_result = ground_with_wikipedia(body.prompt, body.primary_output)
                if rag_result.success:
                    fix_result_schema = FixResultSchema(
                        fixed_output      = rag_result.grounded_answer,
                        fix_applied       = True,
                        fix_strategy      = "RAG_GROQ_GROUNDING",
                        fix_explanation   = (
                            "Shadow-model consensus was unavailable, so FIE used "
                            "Wikipedia-grounded retrieval and Groq-based correction."
                        ),
                        original_output   = body.primary_output,
                        root_cause        = root_cause,
                        fix_confidence    = rag_result.confidence,
                        improvement_score = rag_result.confidence,
                        warning           = f"Grounded source: {rag_result.source}",
                    )
                    failure_summary = (
                        "⚡ AUTO-FIXED: RAG_GROQ_GROUNDING applied. "
                        "Recovered a grounded factual answer using Wikipedia context."
                    )
                    logger.info(
                        "Auto-fix applied | strategy=RAG_GROQ_GROUNDING | confidence=%.3f",
                        rag_result.confidence,
                    )
                else:
                    logger.warning("RAG grounding unavailable: %s", rag_result.error)

        except Exception as exc:
            if "Auto-fix skipped" in str(exc):
                logger.debug(str(exc))
            else:
                logger.error("Fix engine failed: %s", exc, exc_info=True)

    # ── Step 6: Save to MongoDB ────────────────────────────────────
    # Build an InferenceRequest and persist it so the dashboard
    # can display real data from /inferences endpoint.
    try:
        import uuid
        from datetime import datetime
        from app.schemas import InferenceRequest, MathematicalMetrics

        inference_record = InferenceRequest(
            request_id    = str(uuid.uuid4())[:12],
            timestamp     = datetime.utcnow(),
            model_name    = body.primary_model_name,
            model_version = "monitor-v1",
            temperature   = 0.7,
            latency_ms    = body.latency_ms or 0.0,
            input_text    = body.prompt,
            output_text   = body.primary_output,
            metrics       = MathematicalMetrics(
                entropy          = signal.entropy_score,
                agreement_score  = signal.agreement_score,
                fsd_score        = signal.fsd_score,
                embedding_distance = embedding["embedding_distance"],
            ),
        )
        save_inference(inference_record)
    except Exception as exc:
        logger.warning("Failed to save inference record: %s", exc)

    return MonitorResponse(
        shadow_model_results  = shadow_model_results,
        all_model_outputs     = model_outputs,
        ollama_available      = ollama_available,
        failure_signal_vector = signal,
        archetype             = archetype,
        embedding_distance    = embedding["embedding_distance"],
        jury                  = jury_verdict,
        high_failure_risk     = signal.high_failure_risk,
        failure_summary       = failure_summary,
        fix_result            = fix_result_schema,
    )


@router.get("/monitor/status", response_model=dict)
def monitor_status() -> dict:
    """
    Returns the current status of the Ollama service.
    Used by the dashboard to show which models are available.
    """
    from engine.ollama_service import ollama_service
    available       = ollama_service.is_available()
    active_models   = ollama_service.get_available_models() if available else []
    configured      = ollama_service.models

    return {
        "ollama_running":    available,
        "configured_models": configured,
        "active_models":     active_models,
        "ready_models":      [m for m in configured if m in active_models],
        "missing_models":    [m for m in configured if m not in active_models],
    }
