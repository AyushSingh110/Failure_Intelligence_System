import logging
from fastapi import APIRouter, Header, HTTPException

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
from engine.detector.consistency import compute_consistency, is_primary_outlier
from engine.detector.entropy import compute_entropy, compute_entropy_from_counts
from engine.detector.ensemble import compute_disagreement
from engine.detector.embedding import compute_embedding_distance
from engine.archetypes.labeling import label_failure_archetype
from engine.archetypes.clustering import archetype_registry
from engine.evolution.tracker import evolution_tracker
from engine.agents.failure_agent import failure_agent
from config import get_settings
from app.auth_guard import require_user, require_admin, resolve_user

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
        and entropy_score > 0.0
    )

    # Check whether PRIMARY is the outlier vs shadow consensus.
    # This prevents false positives when shadow models disagree with each other
    # in phrasing/format but primary matches the shadow majority.
    # Example: primary="New Zealand", shadows=["New Zealand","New Zealand","Australia"]
    #   → shadow majority = "New Zealand" → primary matches → NOT an outlier
    # Example: primary="Australia", shadows=["New Zealand","New Zealand","New Zealand"]
    #   → shadow majority = "New Zealand" → primary disagrees → IS an outlier
    shadow_outputs   = model_outputs[1:]
    primary_outlier  = is_primary_outlier(primary_output, shadow_outputs)

    # high_failure_risk requires the PRIMARY to be the disagreeing party.
    # We keep the entropy gate as a secondary catch for catastrophic disagreement
    # (all 4 models give completely different answers — high entropy regardless of who's right).
    high_failure_risk = (
        primary_outlier                                          # primary disagrees with shadow majority
        or entropy_score >= settings.high_entropy_threshold     # total chaos across all models
        or (ensemble_fires and primary_outlier)                  # embedding-level primary mismatch
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
def list_inferences(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> list[InferenceRequest]:
    user = require_user(authorization, x_api_key)
    if user.get("is_admin", False):
        return get_all_inferences_admin()
    return get_inferences_for_tenant(user["tenant_id"])


@router.get("/inferences/grouped/by-question", response_model=dict)
def get_inferences_grouped_by_question(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
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
    request_id: str,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
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
    request_id: str,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """Deletes a single inference record by request_id."""
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
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Deletes ALL inference records from MongoDB.
    Use for resetting between test runs.
    """
    from storage.database import delete_inference
    user = require_user(authorization, x_api_key)
    if user.get("is_admin", False):
        records = get_all_inferences_admin()
        count = 0
        for r in records:
            if delete_inference(r.request_id):
                count += 1
        return {"status": "cleared", "deleted_count": count}

    count = clear_inferences_for_tenant(user["tenant_id"])
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
def diagnose(
    body: DiagnosticRequest,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> DiagnosticResponse:
    user = resolve_user(authorization, x_api_key)
    response = failure_agent.run_diagnostic(body)
    if not (user and user.get("is_admin", False)):
        response.explanation_internal = None
    return response


# ── Phase 4 — Real-time monitor endpoint ──────────────────────────────────

from app.schemas import MonitorRequest, MonitorResponse, OllamaModelResult
from engine.explainability.explanation_builder import attach_explanations_to_monitor

@router.post("/monitor", response_model=MonitorResponse)
def monitor(
    body: MonitorRequest,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> MonitorResponse:
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

    current_user = resolve_user(authorization, x_api_key)
    if current_user:
        try:
            from app.auth import increment_usage
            increment_usage(current_user["tenant_id"])
        except Exception as exc:
            logger.warning("Failed to update usage counters: %s", exc)

    # ── Step 1: Call shadow models (Groq or Ollama) ─────────────────────
    # Groq: fast cloud API, ~1 second, free tier 14,400 req/day
    # Ollama: local, private, slower — uncomment below to re-enable
    ollama_available   = False  # kept for schema compatibility
    shadow_results_raw = []

    if settings.groq_enabled and settings.groq_api_key:
        # ── Groq shadow models (Step 2: fan_out_with_confidence) ────
        from engine.groq_service import get_groq_service
        groq = get_groq_service()
        if groq:
            # STEP 2: Use fan_out_with_confidence so each shadow model
            # self-reports its certainty. Weights feed into Step 3.
            groq_results = groq.fan_out_with_confidence(body.prompt)
            shadow_results_raw = groq_results
            successful = [r for r in groq_results if r.success]
            ollama_available = len(successful) > 0
            logger.info(
                "Groq shadow models: %d/%d responded | confidences=%s",
                len(successful), len(groq_results),
                [r.model_confidence for r in successful],
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
    # STEP 2+3: Also capture confidence weights from enriched responses
    shadow_weights: list[float] = []
    for r in shadow_results_raw:
        if r.success and r.output_text:
            model_outputs.append(r.output_text)
            shadow_weights.append(getattr(r, "confidence_weight", 2.0))

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

    # ── Step 5b: Ground Truth Pipeline + Auto-fix engine ───────────────────
    # Steps 4–10: verify claim → wikidata → serper → cache → escalate/fix
    fix_result_schema = None
    gt_result_schema  = None
    requires_human_review = False
    escalation_reason_str = ""

    if jury_verdict and jury_verdict.primary_verdict:
        try:
            from engine.fix_engine import apply_fix, prompt_requires_live_data
            from app.schemas import FixResult as FixResultSchema, GroundTruthVerification

            primary_v  = jury_verdict.primary_verdict
            root_cause = primary_v.root_cause
            confidence = primary_v.confidence_score
            layers_fired = set((primary_v.evidence or {}).get("layers_fired", []))

            # Gate 1: Only proceed when FSV says primary is genuinely at risk.
            # Previously this allowed GT pipeline to run on correct answers when
            # root_cause was factual — causing wrong Wikidata fixes on stable outputs.
            if not signal.high_failure_risk:
                raise RuntimeError("Auto-fix skipped: primary output matches shadow consensus.")

            # Gate 2: Minimum jury confidence before taking any corrective action.
            # A low-confidence verdict (< 0.45) means the jury is unsure — acting
            # on it produces more wrong corrections than it prevents.
            if confidence < 0.45:
                raise RuntimeError(
                    f"Auto-fix skipped: jury confidence {confidence:.2f} below minimum 0.45."
                )

            # Successful shadow outputs and their confidence weights
            shadow_texts = [r.output_text for r in shadow_model_results if r.success and r.output_text]

            # ── Steps 4–10: Ground Truth Pipeline ─────────────────────────
            from engine.verifier.ground_truth_pipeline import run_ground_truth_pipeline
            gt = run_ground_truth_pipeline(
                prompt          = body.prompt,
                primary_output  = body.primary_output,
                root_cause      = root_cause,
                jury_confidence = jury_verdict.jury_confidence,
                shadow_outputs  = shadow_texts,
                shadow_weights  = shadow_weights,
            )
            gt_result_schema = GroundTruthVerification(
                verified_answer     = gt.verified_answer,
                confidence          = gt.confidence,
                source              = gt.source,
                from_cache          = gt.from_cache,
                requires_escalation = gt.requires_escalation,
                escalation_reason   = gt.escalation_reason,
                pipeline_trace      = gt.pipeline_trace,
            )
            logger.info(
                "GT pipeline done | source=%s | escalate=%s | confidence=%.3f",
                gt.source, gt.requires_escalation, gt.confidence,
            )

            # ── Step 10: Escalation check ──────────────────────────────────
            if gt.requires_escalation:
                requires_human_review = True
                escalation_reason_str = gt.escalation_reason
                failure_summary = (
                    "⚠️ HUMAN REVIEW REQUIRED: "
                    f"FIE could not establish reliable ground truth. {gt.escalation_reason[:150]}"
                )
                fix_result_schema = FixResultSchema(
                    fixed_output      = body.primary_output,
                    fix_applied       = False,
                    fix_strategy      = "HUMAN_ESCALATION",
                    fix_explanation   = gt.escalation_reason,
                    original_output   = body.primary_output,
                    root_cause        = root_cause,
                    fix_confidence    = 0.0,
                    improvement_score = 0.0,
                    warning=(
                        "FIE could not verify a reliable correction. "
                        "This inference has been queued for human review."
                    ),
                )

            # ── GT pipeline provided a verified answer → use it ────────────
            elif gt.verified_answer and gt.verified_answer != body.primary_output:
                fix_result_schema = FixResultSchema(
                    fixed_output      = gt.verified_answer,
                    fix_applied       = True,
                    fix_strategy      = f"GT_VERIFIED ({gt.source})",
                    fix_explanation   = (
                        f"Ground truth pipeline verified the correct answer from {gt.source}. "
                        f"Primary output contradicted verified source. "
                        f"Confidence: {gt.confidence:.0%}."
                    ),
                    original_output   = body.primary_output,
                    root_cause        = root_cause,
                    fix_confidence    = gt.confidence,
                    improvement_score = gt.confidence,
                    warning           = "",
                )
                failure_summary = (
                    f"⚡ AUTO-FIXED: GT_VERIFIED applied via {gt.source}. "
                    f"Confidence: {gt.confidence:.0%}."
                )
                logger.info(
                    "GT pipeline applied fix | source=%s | confidence=%.3f",
                    gt.source, gt.confidence,
                )

            # ── GT confirmed OR no override needed → run normal fix engine ──
            else:
                fix = apply_fix(
                    prompt          = body.prompt,
                    primary_output  = body.primary_output,
                    shadow_outputs  = shadow_texts,
                    root_cause      = root_cause,
                    confidence      = confidence,
                    model_fn        = None,
                    shadow_weights  = shadow_weights,   # STEP 3 — weighted consensus
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
                    # STEP 10: escalation from fix engine itself
                    if fix.requires_human_review:
                        requires_human_review = True
                        escalation_reason_str = fix.escalation_reason
                        failure_summary = (
                            "⚠️ HUMAN REVIEW REQUIRED: "
                            f"{fix.escalation_reason[:150]}"
                        )
                    elif fix.fix_applied:
                        failure_summary = (
                            f"⚡ AUTO-FIXED: {fix.fix_strategy} applied. "
                            f"{fix.fix_explanation[:150]}"
                        )
                        logger.info(
                            "Auto-fix applied | strategy=%s | confidence=%.3f",
                            fix.fix_strategy, fix.fix_confidence,
                        )

                # Fallback: Wikipedia RAG if fix engine produced no result
                if (fix_result_schema is None or not fix_result_schema.fix_applied) \
                        and root_cause in {"KNOWLEDGE_BOUNDARY_FAILURE", "FACTUAL_HALLUCINATION"} \
                        and not prompt_requires_live_data(body.prompt):
                    from engine.rag_grounder import ground_with_wikipedia
                    rag_result = ground_with_wikipedia(body.prompt, body.primary_output)
                    if rag_result.success:
                        fix_result_schema = FixResultSchema(
                            fixed_output      = rag_result.grounded_answer,
                            fix_applied       = True,
                            fix_strategy      = "RAG_GROQ_GROUNDING",
                            fix_explanation   = (
                                "Shadow-model consensus unavailable. FIE used "
                                "Wikipedia-grounded retrieval and Groq-based correction."
                            ),
                            original_output   = body.primary_output,
                            root_cause        = root_cause,
                            fix_confidence    = rag_result.confidence,
                            improvement_score = rag_result.confidence,
                            warning           = f"Grounded source: {rag_result.source}",
                        )
                        failure_summary = (
                            "⚡ AUTO-FIXED: RAG_GROQ_GROUNDING applied via Wikipedia."
                        )

        except Exception as exc:
            if "Auto-fix skipped" in str(exc):
                logger.debug(str(exc))
            else:
                logger.error("Fix engine failed: %s", exc, exc_info=True)

    # ── Step 6: Save to MongoDB ────────────────────────────────────
    # Build an InferenceRequest and persist it so the dashboard
    # can display real data from /inferences endpoint.
    stored_request_id = None
    response = MonitorResponse(
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
        ground_truth          = gt_result_schema,
        requires_human_review = requires_human_review,
        escalation_reason     = escalation_reason_str,
    )
    response = attach_explanations_to_monitor(response, request_id=stored_request_id)
    if not (current_user and current_user.get("is_admin", False)):
        response.explanation_internal = None

    # ── Signal logging — captures every raw value for future calibration ──
    try:
        from storage.signal_logger import log_signal

        # Extract layer-level scores from jury verdict if available
        _layers_fired:  list[str]        = []
        _layer_scores:  dict[str, float] = {}
        _jury_verdict   = ""
        _jury_conf      = 0.0
        if jury_verdict and jury_verdict.primary_verdict:
            pv = jury_verdict.primary_verdict
            ev = pv.evidence or {}
            _layers_fired = ev.get("layers_fired", [])
            _layer_scores = ev.get("layer_scores", {})
            _jury_verdict = pv.root_cause
            _jury_conf    = pv.confidence_score

        # GT pipeline values
        _gt_source    = gt_result_schema.source            if gt_result_schema else "none"
        _gt_conf      = gt_result_schema.confidence        if gt_result_schema else 0.0
        _gt_override  = (
            fix_result_schema is not None
            and fix_result_schema.fix_applied
            and gt_result_schema is not None
            and gt_result_schema.source not in ("none", "shadow_consensus")
        )
        _gt_answer    = gt_result_schema.verified_answer   if gt_result_schema else ""
        _req_esc      = requires_human_review
        _esc_reason   = escalation_reason_str

        # Fix engine values
        _fix_applied  = fix_result_schema.fix_applied      if fix_result_schema else False
        _fix_strategy = fix_result_schema.fix_strategy     if fix_result_schema else ""
        _fix_conf     = fix_result_schema.fix_confidence   if fix_result_schema else 0.0
        _fix_output   = fix_result_schema.fixed_output     if fix_result_schema else ""

        # Shadow confidence labels (not just weights)
        _shadow_confs = [
            getattr(r, "model_confidence", "MEDIUM")
            for r in shadow_results_raw if r.success
        ]
        _shadow_texts = [r.output_text for r in shadow_results_raw if r.success and r.output_text]

        _signal_log_id = log_signal(
            request_id            = "",   # filled in after we generate stored_request_id below
            prompt                = body.prompt,
            primary_output        = body.primary_output,
            shadow_outputs        = _shadow_texts,
            shadow_confidences    = _shadow_confs,
            shadow_weights        = shadow_weights,
            entropy_score         = signal.entropy_score,
            agreement_score       = signal.agreement_score,
            fsd_score             = signal.fsd_score,
            ensemble_disagreement = bool(signal.ensemble_disagreement),
            high_failure_risk     = signal.high_failure_risk,
            layers_fired          = _layers_fired,
            layer_scores          = _layer_scores,
            jury_verdict          = _jury_verdict,
            jury_confidence       = _jury_conf,
            gt_source             = _gt_source,
            gt_confidence         = _gt_conf,
            gt_override_applied   = _gt_override,
            gt_verified_answer    = _gt_answer,
            requires_escalation   = _req_esc,
            escalation_reason     = _esc_reason,
            fix_applied           = _fix_applied,
            fix_strategy          = _fix_strategy,
            fix_confidence        = _fix_conf,
            fix_output            = _fix_output,
        )
    except Exception as _log_exc:
        logger.debug("Signal logging failed (non-fatal): %s", _log_exc)
        _signal_log_id = ""

    try:
        import uuid
        from datetime import datetime
        from app.schemas import InferenceRequest, MathematicalMetrics

        stored_request_id = str(uuid.uuid4())[:12]

        # Back-fill the real request_id into the signal log we just wrote
        if _signal_log_id:
            try:
                from storage.signal_logger import _get_collection as _slc
                _sl_col = _slc()
                if _sl_col is not None:
                    _sl_col.update_one(
                        {"log_id": _signal_log_id},
                        {"$set": {"request_id": stored_request_id}},
                    )
            except Exception:
                pass

        if response.explanation_external is not None:
            response.explanation_external.request_id = stored_request_id
        if response.explanation_internal is not None:
            response.explanation_internal.request_id = stored_request_id

        inference_record = InferenceRequest(
            request_id    = stored_request_id,
            tenant_id     = current_user["tenant_id"] if current_user else "anonymous",
            timestamp     = datetime.utcnow(),
            model_name    = body.primary_model_name,
            model_version = "monitor-v1",
            temperature   = 0.7,
            latency_ms    = body.latency_ms or 0.0,
            input_text    = body.prompt,
            output_text   = body.primary_output,
            metrics       = MathematicalMetrics(
                entropy            = signal.entropy_score,
                agreement_score    = signal.agreement_score,
                fsd_score          = signal.fsd_score,
                embedding_distance = embedding["embedding_distance"],
            ),
            human_explanation   = response.human_explanation,
            explanation_external = response.explanation_external,
        )
        save_inference(inference_record)
    except Exception as exc:
        logger.warning("Failed to save inference record: %s", exc)
    return response


# ── Step 8: User Feedback endpoint ───────────────────────────────────────
from app.schemas import FeedbackRequest, FeedbackResponse

@router.post("/feedback/{request_id}", response_model=FeedbackResponse)
def submit_feedback(
    request_id:    str,
    body:          FeedbackRequest,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> FeedbackResponse:
    """
    Step 8 — Ground Truth Feedback Loop.

    Users submit whether the model's answer was correct.
    If is_correct=False and correct_answer is provided:
      1. The correct answer is saved to the ground_truth_cache so the same
         question is answered correctly in all future requests (Step 7).
      2. A feedback record is stored in MongoDB for analytics and
         threshold calibration (Step 9).

    This endpoint is the most important single feature for making FIE
    self-improving — every correction permanently improves the system.

    Authentication required. Users can only submit feedback for their
    own tenant's inferences.
    """
    from storage.database import save_feedback, get_inference_by_id_for_tenant
    from engine.ground_truth_cache import save_to_cache
    from datetime import datetime

    user = require_user(authorization, x_api_key)

    # Verify the request belongs to this tenant
    record = (
        get_inference_by_id(request_id)
        if user.get("is_admin", False)
        else get_inference_by_id_for_tenant(request_id, user["tenant_id"])
    )
    if record is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"Inference '{request_id}' not found for this account",
        )

    cache_updated = False

    # If user says it is wrong AND provides the correct answer → update GT cache
    if not body.is_correct and body.correct_answer:
        cache_updated = save_to_cache(
            question        = record.input_text,
            verified_answer = body.correct_answer.strip(),
            source          = "user_feedback",
            confidence      = 1.0,
            verified_by     = user.get("email", "user"),
        )
        logger.info(
            "GT cache updated from feedback | request_id=%s | correct=%s",
            request_id, body.correct_answer[:60],
        )

    # Update the signal log so this becomes a labeled training example
    try:
        from storage.signal_logger import find_log_by_request_id, update_signal_feedback
        sig_log = find_log_by_request_id(request_id)
        if sig_log:
            # fie_was_correct = True when:
            #   - user says it's correct (FIE didn't wrongly flag it), OR
            #   - user says it's wrong AND FIE had already flagged + corrected it
            fie_flagged   = sig_log.get("high_failure_risk", False)
            fie_corrected = sig_log.get("fix_applied", False)
            if body.is_correct:
                # User confirms the answer is right
                # FIE was correct IF it didn't wrongly apply a fix
                fie_was_correct = not fie_corrected
            else:
                # User says answer was wrong
                # FIE was correct IF it flagged it (high_failure_risk=True)
                fie_was_correct = fie_flagged
            update_signal_feedback(
                log_id          = sig_log["log_id"],
                fie_was_correct = fie_was_correct,
                correct_answer  = body.correct_answer or "",
            )
    except Exception as _fe:
        logger.debug("Signal feedback update failed (non-fatal): %s", _fe)

    # Store feedback record for analytics
    feedback_doc = {
        "request_id":    request_id,
        "tenant_id":     user["tenant_id"],
        "submitted_by":  user.get("email", "unknown"),
        "submitted_at":  datetime.utcnow().isoformat(),
        "is_correct":    body.is_correct,
        "correct_answer": body.correct_answer or "",
        "notes":         body.notes or "",
        "question":      record.input_text,
        "model_answer":  record.output_text,
        "model_name":    record.model_name,
    }
    save_feedback(feedback_doc)

    return FeedbackResponse(
        status        = "received",
        request_id    = request_id,
        cache_updated = cache_updated,
        message       = (
            "Thank you. The correct answer has been saved and will be used "
            "to verify future similar questions."
            if cache_updated else
            "Feedback recorded. Thank you for helping improve the system."
        ),
    )


@router.get("/monitor/calibration", response_model=dict)
def get_calibration_stats(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Returns calibration statistics computed from all labeled signal logs.

    Shows:
    - Overall FIE accuracy (across all feedback-labeled requests)
    - Per-confidence-bucket accuracy (are high-confidence verdicts actually more accurate?)
    - Per-layer precision (which DomainCritic layers are actually predictive?)

    Use this endpoint to know which hardcoded weights and thresholds need updating.
    """
    require_admin(authorization, x_api_key)
    from storage.signal_logger import get_calibration_stats
    return get_calibration_stats()


@router.get("/monitor/signal-logs", response_model=list)
def get_signal_logs(
    limit: int = 50,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> list:
    """
    Returns the N most recent raw signal logs.
    Useful for debugging, auditing, and manually reviewing what FIE is capturing.
    Admin only.
    """
    require_admin(authorization, x_api_key)
    from storage.signal_logger import get_recent_logs
    return get_recent_logs(limit=min(limit, 500))


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
