import logging
from fastapi import APIRouter, Header, HTTPException, Query, Request

logger = logging.getLogger(__name__)

# Rate limiter — imported from main; gracefully absent if slowapi not installed
try:
    from app.main import _limiter as limiter
    from slowapi import _rate_limit_exceeded_handler  # noqa: F401
    _has_limiter = limiter is not None
except Exception:
    limiter = None
    _has_limiter = False

def _rate_limit(rate: str):
    """Decorator factory: applies slowapi limit when available, no-op otherwise."""
    def decorator(func):
        if _has_limiter and limiter is not None:
            return limiter.limit(rate)(func)
        return func
    return decorator

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
    TelemetryPing,
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


# Phase 1 endpoints 

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


# ── Phase 2 endpoints 

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


#Phase 3 endpoint 

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
@_rate_limit("60/minute")
def monitor(
    request: Request,
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
            allowed = increment_usage(current_user["tenant_id"])
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Usage limit reached for your plan "
                        f"({current_user.get('calls_limit', 1000)} calls/month). "
                        "Upgrade your plan or contact support."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Failed to update usage counters: %s", exc)

    # ── Step 1: Call shadow models (Groq or Ollama) ─────────────────────
    # Groq: fast cloud API, ~1 second, free tier 14,400 req/day
    # Ollama: local, private, slower — uncomment below to re-enable
    ollama_available   = False  # kept for schema compatibility
    shadow_results_raw = []

    # Generate canary token injected into shadow model system prompts.
    # If the user's prompt extracts the system prompt, shadow outputs will
    # contain the canary — confirming a system-prompt-exfiltration attack.
    from engine.canary_tracker import generate_canary, build_canary_system_prompt
    _canary_token  = generate_canary()
    _canary_sysprompt = build_canary_system_prompt(_canary_token)

    if settings.groq_enabled and settings.groq_api_key:
        # ── Groq shadow models (Step 2: fan_out_with_confidence) ────
        from engine.groq_service import get_groq_service
        groq = get_groq_service()
        if groq:
            # STEP 2: Use fan_out_with_confidence so each shadow model
            # self-reports its certainty. Weights feed into Step 3.
            # Pass canary system prompt so exfiltration attacks are detectable.
            groq_results = groq.fan_out_with_confidence(
                body.prompt, system_message=_canary_sysprompt
            )
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

    # Step 2: Build full model_outputs list
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

    # Step 3: Run FIE pipeline 
    signal    = _build_failure_signal(model_outputs)
    archetype = label_failure_archetype(signal)
    primary   = model_outputs[0]
    secondary = model_outputs[1] if len(model_outputs) > 1 else model_outputs[0]
    embedding = compute_embedding_distance(primary, secondary)

    # Classify question type — gates the GT pipeline and XGBoost threshold
    from engine.question_classifier import classify as _classify_question, pipeline_gates
    _question_type = _classify_question(body.prompt)
    _pipeline_gates = pipeline_gates(_question_type)
    signal = signal.model_copy(update={"question_type": _question_type})

    # Update archetype registry and EMA tracker
    archetype_registry.assign(signal)
    evolution_tracker.record(signal)

    # Step 4: Optionally run DiagnosticJury
    jury_verdict = None
    if body.run_full_jury:
        diag_request = DiagnosticRequest(
            prompt=body.prompt,
            model_outputs=model_outputs,
            latency_ms=body.latency_ms,
            canary_token=_canary_token,
        )
        diag_response = failure_agent.run_diagnostic(diag_request)
        jury_verdict  = diag_response.jury

    # Step 4b: Multi-turn adversarial tracking ───────────────────────
    # Only runs when the caller provides a conversation_id.
    # Detects Crescendo-style attacks: no single turn is malicious but the
    # trajectory across turns escalates toward a harmful goal.
    multi_turn_result = None
    if body.conversation_id:
        try:
            from engine.multi_turn_tracker import check_multi_turn_escalation
            _jury_is_adversarial  = bool(jury_verdict and jury_verdict.is_adversarial)
            _jury_confidence      = jury_verdict.jury_confidence if jury_verdict else 0.0
            mt = check_multi_turn_escalation(
                conversation_id      = body.conversation_id,
                prompt               = body.prompt,
                question_type        = _question_type,
                is_adversarial       = _jury_is_adversarial,
                adversarial_confidence = _jury_confidence,
            )
            if mt.is_escalating:
                multi_turn_result = {
                    "is_escalating":  True,
                    "confidence":     mt.confidence,
                    "pattern":        mt.pattern,
                    "turn_count":     mt.turn_count,
                    "evidence":       mt.evidence,
                }
                logger.warning(
                    "MULTI_TURN_ESCALATION detected | conv=%s pattern=%s conf=%.3f",
                    body.conversation_id, mt.pattern, mt.confidence,
                )
        except Exception as exc:
            logger.warning("multi_turn_tracker failed (non-fatal): %s", exc)

    # Step 4c: Model extraction detection ───────────────────────────────────
    extraction_result = None
    try:
        from engine.model_extraction_tracker import check_model_extraction
        _tenant = current_user["tenant_id"] if current_user else "anonymous"
        ext = check_model_extraction(
            tenant_id       = _tenant,
            prompt          = body.prompt or "",
            conversation_id = body.conversation_id,
        )
        if ext.is_extracting:
            extraction_result = {
                "is_extracting": True,
                "confidence":    ext.confidence,
                "pattern":       ext.pattern,
                "evidence":      ext.evidence,
            }
            logger.warning(
                "MODEL_EXTRACTION | tenant=%s | pattern=%s | conf=%.3f",
                _tenant, ext.pattern, ext.confidence,
            )
    except Exception as exc:
        logger.debug("model_extraction_tracker failed (non-fatal): %s", exc)

    # Step 5: Build failure summary ─────────────────────────────────
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

    # Step 5b: Ground Truth Pipeline + Auto-fix engine ───────────────────
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

            # Gate 3: Question-type routing
            # OPINION/CODE/REASONING → skip external GT lookup entirely
            if not _pipeline_gates.get("run_wikidata", True) and not _pipeline_gates.get("run_serper", True):
                raise RuntimeError(
                    f"Auto-fix skipped: question_type={_question_type} does not use external GT."
                )

            # Steps 4–10: Ground Truth Pipeline 
            from engine.verifier.ground_truth_pipeline import run_ground_truth_pipeline
            gt = run_ground_truth_pipeline(
                prompt          = body.prompt,
                primary_output  = body.primary_output,
                root_cause      = root_cause,
                jury_confidence = jury_verdict.jury_confidence,
                shadow_outputs  = shadow_texts,
                shadow_weights  = shadow_weights,
                use_wikidata    = _pipeline_gates.get("run_wikidata", True),
                use_serper      = _pipeline_gates.get("run_serper",   True),
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

            # Step 10: Escalation check 
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

    # ── XGBoost classifier override (post-GT) ─────────────────────────
    # Runs AFTER the full GT+fix pipeline so all 434 features are populated.
    # gt_source, fix_strategy, fix_confidence, gt_confidence account for 47%
    # of the model's feature importance — they must not be zero at inference.
    # POET's boolean is still passed as an input feature (not discarded).
    _xgb_prob = None
    try:
        from engine.failure_classifier import predict as _clf_predict

        _jury_conf_xgb    = jury_verdict.jury_confidence if jury_verdict else 0.0
        _jury_verd_str    = (
            jury_verdict.primary_verdict.root_cause
            if jury_verdict and jury_verdict.primary_verdict
            else "NONE"
        )
        _gt_source_xgb    = gt_result_schema.source          if gt_result_schema  else "none"
        _gt_conf_xgb      = gt_result_schema.confidence      if gt_result_schema  else 0.0
        _fix_applied_xgb  = fix_result_schema.fix_applied    if fix_result_schema else False
        _fix_strategy_xgb = fix_result_schema.fix_strategy   if fix_result_schema else ""
        _fix_conf_xgb     = fix_result_schema.fix_confidence if fix_result_schema else 0.0
        _gt_override_xgb  = (
            fix_result_schema is not None
            and fix_result_schema.fix_applied
            and gt_result_schema is not None
            and gt_result_schema.source not in ("none", "shadow_consensus")
        )

        _xgb_is_failure, _xgb_prob = _clf_predict(
            agreement_score     = signal.agreement_score,
            entropy_score       = signal.entropy_score,
            jury_confidence     = _jury_conf_xgb,
            fix_confidence      = _fix_conf_xgb,
            gt_confidence       = _gt_conf_xgb,
            high_failure_risk   = signal.high_failure_risk,
            fix_applied         = _fix_applied_xgb,
            requires_escalation = requires_human_review,
            gt_override         = _gt_override_xgb,
            archetype           = archetype,
            jury_verdict_str    = _jury_verd_str,
            fix_strategy        = _fix_strategy_xgb,
            gt_source           = _gt_source_xgb,
            question_type       = _question_type,
        )

        # Use per-question-type threshold from fie_config (auto-calibrated)
        from engine.fie_config import get_threshold, get_config_version, MODEL_VERSION
        _xgb_threshold   = get_threshold(_question_type)
        _xgb_is_failure  = _xgb_prob >= _xgb_threshold
        _config_ver      = get_config_version()

        signal = signal.model_copy(update={"high_failure_risk": _xgb_is_failure})
        logger.info(
            "XGBoost post-GT: prob=%.4f threshold=%.3f is_failure=%s qt=%s gt_source=%s",
            _xgb_prob, _xgb_threshold, _xgb_is_failure, _question_type, _gt_source_xgb,
        )
    except Exception as _clf_exc:
        _config_ver = "default"
        logger.warning("XGBoost classifier unavailable, keeping POET decision: %s", _clf_exc)

    # ── Step 6: Save to MongoDB ────────────────────────────────────
    # Build an InferenceRequest and persist it so the dashboard
    # can display real data from /inferences endpoint.
    stored_request_id = None
    from engine.fie_config import MODEL_VERSION as _MODEL_VER
    response = MonitorResponse(
        shadow_model_results   = shadow_model_results,
        all_model_outputs      = model_outputs,
        ollama_available       = ollama_available,
        failure_signal_vector  = signal,
        archetype              = archetype,
        embedding_distance     = embedding["embedding_distance"],
        jury                   = jury_verdict,
        high_failure_risk      = signal.high_failure_risk,
        failure_summary        = failure_summary,
        fix_result             = fix_result_schema,
        ground_truth           = gt_result_schema,
        requires_human_review  = requires_human_review,
        escalation_reason      = escalation_reason_str,
        classifier_probability = _xgb_prob,
        model_version          = _MODEL_VER,
        config_version         = _config_ver,
        multi_turn_escalation  = multi_turn_result,
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
            request_id             = "",   # filled in after we generate stored_request_id below
            prompt                 = body.prompt,
            primary_output         = body.primary_output,
            shadow_outputs         = _shadow_texts,
            shadow_confidences     = _shadow_confs,
            shadow_weights         = shadow_weights,
            entropy_score          = signal.entropy_score,
            agreement_score        = signal.agreement_score,
            fsd_score              = signal.fsd_score,
            ensemble_disagreement  = bool(signal.ensemble_disagreement),
            high_failure_risk      = signal.high_failure_risk,
            classifier_probability = _xgb_prob,
            question_type          = _question_type,
            model_version          = _MODEL_VER,
            config_version         = _config_ver,
            layers_fired           = _layers_fired,
            layer_scores           = _layer_scores,
            jury_verdict           = _jury_verdict,
            jury_confidence        = _jury_conf,
            gt_source              = _gt_source,
            gt_confidence          = _gt_conf,
            gt_override_applied    = _gt_override,
            gt_verified_answer     = _gt_answer,
            requires_escalation    = _req_esc,
            escalation_reason      = _esc_reason,
            fix_applied            = _fix_applied,
            fix_strategy           = _fix_strategy,
            fix_confidence         = _fix_conf,
            fix_output             = _fix_output,
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

        # ── Email notifications (fire-and-forget, never blocks the pipeline) ──
        try:
            from app.notifications import notify_attack_detected, notify_human_review
            _notif_tenant = current_user["tenant_id"] if current_user else "anonymous"
            _notif_email  = current_user.get("email") if current_user else None

            _is_attack = bool(jury_verdict and jury_verdict.is_adversarial)
            if _is_attack:
                notify_attack_detected(
                    tenant_id   = _notif_tenant,
                    attack_type = (
                        jury_verdict.adversarial_verdict.attack_type
                        if jury_verdict and jury_verdict.adversarial_verdict else "UNKNOWN"
                    ),
                    confidence  = jury_verdict.jury_confidence if jury_verdict else 0.0,
                    prompt      = body.prompt or "",
                    model_name  = body.primary_model_name,
                    request_id  = stored_request_id,
                    to          = _notif_email,
                )
            elif requires_human_review:
                notify_human_review(
                    tenant_id         = _notif_tenant,
                    request_id        = stored_request_id,
                    escalation_reason = escalation_reason_str,
                    prompt            = body.prompt or "",
                    model_name        = body.primary_model_name,
                    to                = _notif_email,
                )
        except Exception as _notif_exc:
            logger.debug("Notification send failed (non-fatal): %s", _notif_exc)

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

    # Trigger auto-recalibration if enough new labels have accumulated
    try:
        from engine.fie_config import maybe_recalibrate
        maybe_recalibrate()
    except Exception:
        pass

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


@router.get("/monitor/model-info", response_model=dict)
def model_info() -> dict:
    """
    Returns the currently loaded classifier model, its training date,
    AUC on held-out set, current per-question-type thresholds, and
    config version. No auth required — safe to expose publicly.
    """
    from engine.fie_config import (
        get_all_thresholds, get_config_version,
        MODEL_VERSION, MODEL_TRAINED, RECALIBRATION_INTERVAL,
    )
    from engine.failure_classifier import _model, CLASSIFIER_THRESHOLD

    model_loaded = _model is not None
    return {
        "model_version":        MODEL_VERSION,
        "model_trained":        MODEL_TRAINED,
        "model_loaded":         model_loaded,
        "fallback_mode":        "POET rule-based" if not model_loaded else "XGBoost",
        "auc_held_out":         {"xgboost-v2": 0.728, "xgboost-v3": 0.677}.get(MODEL_VERSION, 0.749),
        "default_threshold":    CLASSIFIER_THRESHOLD,
        "note_threshold":       "Per-type thresholds (thresholds_per_type) are used in production. default_threshold is the fallback.",
        "thresholds_per_type":  get_all_thresholds(),
        "config_version":       get_config_version(),
        "recalibration_interval": RECALIBRATION_INTERVAL,
        "features": [
            "agreement_score", "entropy_score", "jury_confidence",
            "fix_confidence", "gt_confidence", "high_failure_risk",
            "fix_applied", "requires_escalation", "gt_override",
            "archetype", "jury_verdict", "fix_strategy", "gt_source",
        ],
    }


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


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics endpoints — telemetry, performance, calibration, research paper data
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sig_col():
    """Returns signal_logs collection or None."""
    try:
        from storage.database import _db, _fallback_mode
        if _fallback_mode or _db is None:
            return None
        return _db["signal_logs"]
    except Exception:
        return None


@router.get("/analytics/usage", response_model=dict)
def analytics_usage(
    days: int = 7,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Request volume, latency, and failure detection rate over the past N days.
    Broken down by day for trend charts. Useful for dashboards and reports.
    """
    require_admin(authorization, x_api_key)
    col = _get_sig_col()
    if col is None:
        return {"error": "MongoDB unavailable"}

    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    try:
        docs = list(col.find(
            {"timestamp": {"$gte": cutoff}},
            {"timestamp": 1, "high_failure_risk": 1, "fix_applied": 1,
             "question_type": 1, "model_version": 1},
        ))

        total = len(docs)
        failures = sum(1 for d in docs if d.get("high_failure_risk"))
        fixes    = sum(1 for d in docs if d.get("fix_applied"))

        # Daily breakdown
        from collections import defaultdict
        daily: dict = defaultdict(lambda: {"requests": 0, "failures": 0, "fixes": 0})
        for d in docs:
            day = (d.get("timestamp") or "")[:10]
            daily[day]["requests"] += 1
            if d.get("high_failure_risk"):
                daily[day]["failures"] += 1
            if d.get("fix_applied"):
                daily[day]["fixes"] += 1

        # Question type breakdown
        qt_counts: dict = defaultdict(int)
        for d in docs:
            qt_counts[d.get("question_type", "UNKNOWN")] += 1

        return {
            "period_days":       days,
            "total_requests":    total,
            "failure_detections": failures,
            "auto_fixes":        fixes,
            "failure_rate":      round(failures / total, 4) if total else 0.0,
            "fix_rate":          round(fixes / total, 4) if total else 0.0,
            "daily_breakdown":   dict(sorted(daily.items())),
            "question_type_breakdown": dict(qt_counts),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/model-performance", response_model=dict)
def analytics_model_performance(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    XGBoost vs POET agreement rate, accuracy from real user feedback,
    and per-question-type breakdown. Core metrics for the research paper.
    """
    require_admin(authorization, x_api_key)
    col = _get_sig_col()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        all_docs     = list(col.find({}, {"high_failure_risk": 1, "classifier_probability": 1,
                                          "question_type": 1, "fie_was_correct": 1,
                                          "feedback_received": 1, "model_version": 1}))
        labeled_docs = [d for d in all_docs if d.get("feedback_received")]

        total      = len(all_docs)
        n_labeled  = len(labeled_docs)
        correct    = sum(1 for d in labeled_docs if d.get("fie_was_correct"))
        overall_acc = round(correct / n_labeled, 4) if n_labeled else None

        # XGBoost coverage: % of requests where classifier ran (prob not None)
        with_prob = sum(1 for d in all_docs if d.get("classifier_probability") is not None)
        xgb_coverage = round(with_prob / total, 4) if total else 0.0

        # Per-question-type accuracy
        from collections import defaultdict
        qt_stats: dict = defaultdict(lambda: {"total": 0, "labeled": 0, "correct": 0})
        for d in all_docs:
            qt = d.get("question_type", "UNKNOWN")
            qt_stats[qt]["total"] += 1
            if d.get("feedback_received"):
                qt_stats[qt]["labeled"] += 1
                if d.get("fie_was_correct"):
                    qt_stats[qt]["correct"] += 1

        qt_summary = {}
        for qt, s in qt_stats.items():
            qt_summary[qt] = {
                "total_requests": s["total"],
                "labeled":        s["labeled"],
                "accuracy":       round(s["correct"] / s["labeled"], 4) if s["labeled"] else None,
            }

        # Model version distribution
        ver_counts: dict = defaultdict(int)
        for d in all_docs:
            ver_counts[d.get("model_version", "unknown")] += 1

        return {
            "total_requests":       total,
            "total_labeled":        n_labeled,
            "overall_accuracy":     overall_acc,
            "xgboost_coverage":     xgb_coverage,
            "per_question_type":    qt_summary,
            "model_version_dist":   dict(ver_counts),
            "note": (
                "accuracy = % of labeled examples where FIE verdict matched user feedback. "
                "xgboost_coverage = % of requests where classifier ran (vs POET fallback)."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/calibration", response_model=dict)
def analytics_calibration(
    question_type: str = "all",
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Confidence calibration curves from real user feedback.
    Pass ?question_type=FACTUAL to get per-type curves.
    Returns data points ready for a calibration plot (predicted vs actual accuracy).
    Essential for research paper Section 4: Calibration Analysis.
    """
    require_admin(authorization, x_api_key)
    col = _get_sig_col()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        query: dict = {"feedback_received": True}
        if question_type.upper() != "ALL":
            query["question_type"] = question_type.upper()

        labeled = list(col.find(query, {
            "classifier_probability": 1, "fie_was_correct": 1, "question_type": 1
        }))

        if not labeled:
            return {"error": "No labeled examples found", "question_type": question_type}

        # 10 equal-width bins from 0.0 → 1.0
        import math
        n_bins = 10
        bins: dict = {i: {"predicted_sum": 0.0, "correct": 0, "total": 0}
                      for i in range(n_bins)}

        for doc in labeled:
            prob = doc.get("classifier_probability")
            if prob is None:
                continue
            correct = doc.get("fie_was_correct", False)
            b = min(int(prob * n_bins), n_bins - 1)
            bins[b]["predicted_sum"] += prob
            bins[b]["correct"]       += int(correct)
            bins[b]["total"]         += 1

        calibration_points = []
        ece = 0.0  # Expected Calibration Error
        n_total = len(labeled)

        for b, data in bins.items():
            n = data["total"]
            if n == 0:
                continue
            pred_avg = data["predicted_sum"] / n
            actual   = data["correct"] / n
            ece     += (n / n_total) * abs(pred_avg - actual)
            calibration_points.append({
                "bin":              b,
                "predicted_avg":    round(pred_avg, 4),
                "actual_accuracy":  round(actual, 4),
                "calibration_error": round(abs(pred_avg - actual), 4),
                "n_examples":       n,
            })

        from engine.fie_config import get_all_thresholds, get_config_version
        return {
            "question_type":        question_type,
            "n_labeled":            n_total,
            "ece":                  round(ece, 4),
            "interpretation":       "ECE < 0.05 = well calibrated. ECE > 0.10 = needs recalibration.",
            "calibration_points":   calibration_points,
            "current_thresholds":   get_all_thresholds(),
            "config_version":       get_config_version(),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/question-breakdown", response_model=dict)
def analytics_question_breakdown(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Per-question-type breakdown: request volume, failure rate, fix success rate,
    escalation rate, and average classifier confidence.
    Useful for understanding where FIE adds the most value.
    """
    require_admin(authorization, x_api_key)
    col = _get_sig_col()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        docs = list(col.find({}, {
            "question_type": 1, "high_failure_risk": 1, "fix_applied": 1,
            "requires_escalation": 1, "classifier_probability": 1, "gt_source": 1,
        }))

        from collections import defaultdict
        stats: dict = defaultdict(lambda: {
            "total": 0, "failures": 0, "fixes": 0,
            "escalations": 0, "prob_sum": 0.0, "prob_count": 0,
            "gt_sources": defaultdict(int),
        })

        for d in docs:
            qt = d.get("question_type", "UNKNOWN")
            stats[qt]["total"] += 1
            if d.get("high_failure_risk"):
                stats[qt]["failures"] += 1
            if d.get("fix_applied"):
                stats[qt]["fixes"] += 1
            if d.get("requires_escalation"):
                stats[qt]["escalations"] += 1
            prob = d.get("classifier_probability")
            if prob is not None:
                stats[qt]["prob_sum"]   += prob
                stats[qt]["prob_count"] += 1
            src = d.get("gt_source", "none")
            stats[qt]["gt_sources"][src] += 1

        result = {}
        for qt, s in stats.items():
            n = s["total"]
            result[qt] = {
                "total_requests":    n,
                "failure_rate":      round(s["failures"] / n, 4) if n else 0.0,
                "fix_rate":          round(s["fixes"] / n, 4) if n else 0.0,
                "escalation_rate":   round(s["escalations"] / n, 4) if n else 0.0,
                "avg_xgb_prob":      round(s["prob_sum"] / s["prob_count"], 4)
                                     if s["prob_count"] else None,
                "top_gt_sources":    dict(sorted(
                    s["gt_sources"].items(), key=lambda x: -x[1]
                )[:3]),
            }

        return {
            "breakdown":    result,
            "total_logged": len(docs),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/paper-metrics", response_model=dict)
def analytics_paper_metrics(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    All metrics needed for the research paper results section in one call.

    Returns:
      - Live production calibration stats (from real user feedback)
      - Per-question-type accuracy breakdown
      - Model version info and current thresholds
      - Confidence calibration ECE
      - Pipeline routing stats (how often each GT source was used)

    Combine with notebook-generated POET/XGBoost AUC comparison for
    the complete results section.
    """
    require_admin(authorization, x_api_key)
    col = _get_sig_col()
    if col is None:
        return {"error": "MongoDB unavailable"}

    try:
        from storage.signal_logger import get_calibration_stats
        from engine.fie_config import (
            get_all_thresholds, get_config_version, MODEL_VERSION, MODEL_TRAINED,
            RECALIBRATION_INTERVAL,
        )

        calib_stats = get_calibration_stats()

        # Pipeline routing: how often each GT source was used
        pipeline_docs = list(col.find({}, {"gt_source": 1, "question_type": 1, "fix_applied": 1}))
        from collections import defaultdict
        gt_source_counts: dict  = defaultdict(int)
        qt_counts: dict         = defaultdict(int)
        for d in pipeline_docs:
            gt_source_counts[d.get("gt_source", "none")] += 1
            qt_counts[d.get("question_type", "UNKNOWN")] += 1

        # Calibration ECE from labeled examples
        labeled = list(col.find(
            {"feedback_received": True, "classifier_probability": {"$ne": None}},
            {"classifier_probability": 1, "fie_was_correct": 1},
        ))
        ece = 0.0
        n_labeled = len(labeled)
        if n_labeled > 0:
            n_bins = 10
            bins: dict = {i: {"pred": 0.0, "correct": 0, "total": 0} for i in range(n_bins)}
            for doc in labeled:
                prob = doc.get("classifier_probability", 0.0) or 0.0
                b    = min(int(prob * n_bins), n_bins - 1)
                bins[b]["pred"]    += prob
                bins[b]["correct"] += int(doc.get("fie_was_correct", False))
                bins[b]["total"]   += 1
            for b, data in bins.items():
                n = data["total"]
                if n:
                    pred_avg = data["pred"] / n
                    actual   = data["correct"] / n
                    ece += (n / n_labeled) * abs(pred_avg - actual)

        return {
            "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
            "model": {
                "version":        MODEL_VERSION,
                "trained":        MODEL_TRAINED,
                "threshold_mode": "per_question_type_auto_calibrated",
                "thresholds":     get_all_thresholds(),
                "config_version": get_config_version(),
                "recalibration_interval": RECALIBRATION_INTERVAL,
            },
            "live_accuracy": {
                "total_labeled":    calib_stats.get("total_labeled", 0),
                "overall_accuracy": calib_stats.get("overall_accuracy"),
                "ece":              round(ece, 4),
                "calibration_by_confidence_bucket": calib_stats.get("calibration", {}),
            },
            "layer_precision": calib_stats.get("layer_precision", {}),
            "pipeline_routing": {
                "total_requests":  len(pipeline_docs),
                "gt_source_counts": dict(gt_source_counts),
                "question_type_counts": dict(qt_counts),
            },
            "how_to_cite": (
                "Use overall_accuracy, ece, and calibration_by_confidence_bucket "
                "for the calibration analysis section. "
                "Use pipeline_routing.gt_source_counts to show GT pipeline source distribution. "
                "Cross-reference with notebook AUC figures for the full results table."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Telemetry receiver (opt-in pings from fie-sdk users) ──────────────────────

@router.post("/telemetry", response_model=dict)
@_rate_limit("30/minute")
def receive_telemetry(request: Request, body: TelemetryPing) -> dict:
    """
    Receives anonymized usage pings from fie-sdk clients when FIE_TELEMETRY=true.
    Stores aggregated counts in MongoDB `sdk_telemetry` collection.
    No prompt text, no API keys, no PII — only event type and boolean signals.
    Schema-validated and size-limited via TelemetryPing to prevent payload abuse.
    """
    try:
        from storage.database import _db, _fallback_mode
        from datetime import datetime

        clean = body.model_dump()
        clean["received_at"] = datetime.utcnow().isoformat()

        if not _fallback_mode and _db is not None:
            _db["sdk_telemetry"].insert_one(clean)
    except Exception:
        pass  # telemetry failures must never surface to SDK users

    return {"status": "ok"}


@router.get("/analytics/sdk-telemetry", response_model=dict)
def analytics_sdk_telemetry(
    days: int = 30,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Admin view of anonymized SDK usage telemetry.
    Shows how many users have FIE_TELEMETRY=true, what events they fire,
    failure rates, fix rates, and question type distribution from the field.
    Data comes only from fie-sdk clients that opted in via FIE_TELEMETRY=true.
    """
    require_admin(authorization, x_api_key)
    try:
        from storage.database import _db, _fallback_mode
        from datetime import datetime, timedelta
        from collections import defaultdict

        if _fallback_mode or _db is None:
            return {"error": "MongoDB unavailable"}

        col = _db["sdk_telemetry"]
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        docs = list(col.find({"received_at": {"$gte": cutoff}}, {"_id": 0}))

        total = len(docs)
        if total == 0:
            return {
                "period_days": days,
                "total_pings": 0,
                "note": "No telemetry pings received. SDK users must set FIE_TELEMETRY=true to opt in.",
            }

        # Event counts
        event_counts: dict = defaultdict(int)
        for d in docs:
            event_counts[d.get("event", "unknown")] += 1

        # SDK version distribution
        version_counts: dict = defaultdict(int)
        for d in docs:
            version_counts[d.get("sdk_version", "unknown")] += 1

        # Question type distribution
        qt_counts: dict = defaultdict(int)
        for d in docs:
            qt_counts[d.get("question_type", "UNKNOWN")] += 1

        # Mode distribution (monitor vs correct)
        mode_counts: dict = defaultdict(int)
        for d in docs:
            mode_counts[d.get("mode", "unknown")] += 1

        # Failure and fix rates from field
        monitor_pings = [d for d in docs if d.get("event") == "monitor_call"]
        n_monitor = len(monitor_pings)
        n_failures = sum(1 for d in monitor_pings if d.get("high_failure_risk"))
        n_fixes    = sum(1 for d in monitor_pings if d.get("fix_applied"))

        return {
            "period_days":          days,
            "total_pings":          total,
            "event_breakdown":      dict(event_counts),
            "sdk_version_dist":     dict(version_counts),
            "question_type_dist":   dict(qt_counts),
            "mode_dist":            dict(mode_counts),
            "field_failure_rate":   round(n_failures / n_monitor, 4) if n_monitor else None,
            "field_fix_rate":       round(n_fixes    / n_monitor, 4) if n_monitor else None,
            "monitor_call_count":   n_monitor,
            "note": (
                "All pings are anonymized — no prompts or API keys are stored. "
                "field_failure_rate = % of monitor calls where high_failure_risk=True from real SDK users."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Weekly digest notification ────────────────────────────────────────────────

@router.post("/notifications/digest", response_model=dict)
def send_weekly_digest(
    days:          int  = Query(default=7, ge=1, le=90),
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """
    Compile a usage digest for the authenticated tenant and email it via SendGrid.
    Call this on a schedule (e.g. weekly cron) or on-demand.
    Returns a summary dict regardless of email status.
    """
    from app.auth_guard import resolve_user
    from app.notifications import notify_weekly_digest

    current_user = resolve_user(authorization, x_api_key)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        from storage.database import get_inferences_for_tenant
        inferences = get_inferences_for_tenant(current_user["tenant_id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(days=days)

    period = [
        r for r in inferences
        if r.timestamp and r.timestamp >= cutoff
    ] if inferences else []

    total       = len(period)
    high_risk   = sum(1 for r in period if (r.metrics.entropy if r.metrics else 0) > 0.75)
    attacks     = sum(1 for r in period if getattr(r, "is_adversarial", False))
    fix_applied = sum(1 for r in period if getattr(r, "fix_applied", False))
    escalations = sum(1 for r in period if getattr(r, "requires_escalation", False))

    archetype_counts: dict = {}
    for r in period:
        a = getattr(r, "archetype", "STABLE") or "STABLE"
        archetype_counts[a] = archetype_counts.get(a, 0) + 1
    top_archetype = max(archetype_counts, key=archetype_counts.get) if archetype_counts else "STABLE"

    notify_weekly_digest(
        tenant_id     = current_user["tenant_id"],
        total         = total,
        high_risk     = high_risk,
        attacks       = attacks,
        fix_applied   = fix_applied,
        escalations   = escalations,
        top_archetype = top_archetype,
        period_days   = days,
        to            = current_user.get("email"),
    )

    return {
        "status":        "digest_sent",
        "period_days":   days,
        "total":         total,
        "high_risk":     high_risk,
        "attacks":       attacks,
        "fix_applied":   fix_applied,
        "escalations":   escalations,
        "top_archetype": top_archetype,
        "recipient":     current_user.get("email", "—"),
        "note": (
            "Email delivery requires SENDGRID_API_KEY and NOTIFICATION_EMAIL in .env. "
            "Stats are returned regardless."
        ),
    }
