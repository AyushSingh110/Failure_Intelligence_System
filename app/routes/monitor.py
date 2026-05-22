"""
Real-time monitoring routes — the core of the production pipeline.

Endpoints
---------
POST /monitor              — fan-out to shadow models, run full FIE pipeline, return analysis
GET  /monitor/status       — shadow model availability
GET  /monitor/model-info   — classifier version, thresholds, AUC
GET  /monitor/calibration  — per-bucket accuracy from user feedback (admin)
GET  /monitor/signal-logs  — recent raw signal logs (admin)
POST /feedback/{id}        — ground-truth feedback loop (Step 8)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request

from app.limiter import rate_limit
from app.routes._helpers import build_failure_signal, get_signal_logs_collection
from engine.agents.failure_agent import failure_agent
from app.schemas import (
    MonitorRequest,
    MonitorResponse,
    OllamaModelResult,
    FeedbackRequest,
    FeedbackResponse,
)
from app.auth_guard import require_user, require_admin, resolve_user
from config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()
router   = APIRouter()


# ── POST /monitor ─────────────────────────────────────────────────────────────

@router.post("/monitor", response_model=MonitorResponse)
@rate_limit("60/minute")
def monitor(
    request:       Request,
    body:          MonitorRequest,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> MonitorResponse:
    """
    Real-time monitoring endpoint — the core of the production system.

    Flow:
      1. Enforce per-tenant usage limits
      2. Auto-inject session context for multi-turn conversations
      3. Fan out prompt to all shadow models in parallel (Groq / Ollama)
      4. Build FailureSignalVector from all model outputs
      5. Run DiagnosticJury (if run_full_jury=True)
      6. Run multi-turn escalation and model-extraction detectors
      7. Run Ground Truth pipeline + auto-fix engine
      8. Run XGBoost classifier override
      9. Persist signal log + inference record
     10. Fire email notifications (fire-and-forget)
    """
    from app.schemas import DiagnosticRequest, InferenceRequest, MathematicalMetrics
    from engine.explainability.explanation_builder import attach_explanations_to_monitor
    from engine.detector.embedding import compute_embedding_distance
    from engine.archetypes.labeling import assign_failure_label
    from engine.archetypes.clustering import archetype_registry
    from engine.evolution.tracker import evolution_tracker

    # ── Pre-flight guard (runs before everything else) ─────────────────────
    # If the prompt is adversarial and block mode is active, skip shadow models,
    # skip the fix pipeline, skip billing — return a safe refusal immediately.
    if body.prompt:
        try:
            from fie.preflight import preflight_check
            _guard = preflight_check(body.prompt)
            if _guard.blocked:
                logger.warning(
                    "SERVER_PREFLIGHT_BLOCK | attack_type=%s confidence=%.3f layers=%s",
                    _guard.attack_type, _guard.confidence,
                    ",".join(_guard.layers_fired),
                )
                from app.schemas import FailureSignalVector
                return MonitorResponse(
                    shadow_model_results   = [],
                    all_model_outputs      = [body.primary_output],
                    ollama_available       = False,
                    failure_signal_vector  = FailureSignalVector(
                        agreement_score       = 1.0,
                        fsd_score             = 0.0,
                        answer_counts         = {},
                        entropy_score         = 0.0,
                        ensemble_disagreement = False,
                        ensemble_similarity   = 1.0,
                        high_failure_risk     = True,
                    ),
                    archetype              = "ADVERSARIAL_PROMPT",
                    embedding_distance     = 0.0,
                    high_failure_risk      = True,
                    failure_summary        = (
                        f"Prompt blocked by pre-flight guard: {_guard.attack_type} "
                        f"(confidence={_guard.confidence:.0%}). "
                        "Primary LLM was not invoked."
                    ),
                    requires_human_review  = False,
                    escalation_reason      = "",
                    guard_blocked          = True,
                    guard_attack_type      = _guard.attack_type,
                    guard_confidence       = _guard.confidence,
                )
        except Exception as _pf_exc:
            # Guard failure must never take the endpoint down — log and continue
            logger.warning("preflight_check failed (allowing request through): %s", _pf_exc)

    # ── Step 0: Usage enforcement ──────────────────────────────────────────
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

    # ── Step 1: Session context threading ─────────────────────────────────
    if body.session_id and not body.context:
        try:
            from engine.session_store import get_context
            _auto_ctx = get_context(body.session_id)
            if _auto_ctx:
                body = body.model_copy(update={"context": _auto_ctx})
                logger.info(
                    "SessionStore: auto-injected %d turns for session %s",
                    len(_auto_ctx), body.session_id,
                )
        except Exception as _sess_exc:
            logger.debug("SessionStore fetch failed (non-fatal): %s", _sess_exc)

    # ── Step 2: Shadow model fan-out ───────────────────────────────────────
    ollama_available   = False
    shadow_results_raw = []

    from engine.canary_tracker import generate_canary, build_canary_system_prompt
    _canary_token     = generate_canary()
    _canary_sysprompt = build_canary_system_prompt(_canary_token)

    if settings.groq_enabled and settings.groq_api_key:
        from engine.groq_service import get_groq_service
        groq = get_groq_service()
        if groq:
            groq_results       = groq.fan_out_with_confidence(
                body.prompt, system_message=_canary_sysprompt
            )
            shadow_results_raw = groq_results
            successful         = [r for r in groq_results if r.success]
            ollama_available   = len(successful) > 0
            logger.info(
                "Groq shadow models: %d/%d responded | confidences=%s",
                len(successful), len(groq_results),
                [r.model_confidence for r in successful],
            )
        else:
            logger.warning("Groq service unavailable — running without shadow models")
    else:
        logger.warning(
            "No shadow model provider configured. Add GROQ_API_KEY=gsk_xxx to .env"
        )

    # ── Step 3: Build model_outputs list ──────────────────────────────────
    model_outputs: list[str] = [body.primary_output]
    shadow_weights: list[float] = []
    for r in shadow_results_raw:
        if r.success and r.output_text:
            model_outputs.append(r.output_text)
            shadow_weights.append(getattr(r, "confidence_weight", 2.0))

    shadow_model_results = [
        OllamaModelResult(
            model_name = r.model_name,
            output_text= r.output_text,
            latency_ms = r.latency_ms,
            success    = r.success,
            error      = r.error,
        )
        for r in shadow_results_raw
    ]

    # ── Step 4: Signal analysis ────────────────────────────────────────────
    signal    = build_failure_signal(model_outputs)
    primary   = model_outputs[0]
    secondary = model_outputs[1] if len(model_outputs) > 1 else model_outputs[0]
    embedding = compute_embedding_distance(primary, secondary)

    from engine.question_classifier import classify as _classify_question, pipeline_gates
    _question_type  = _classify_question(body.prompt)
    _pipeline_gates = pipeline_gates(_question_type)
    signal = signal.model_copy(update={"question_type": _question_type})

    _has_context = bool(body.context)
    archetype = assign_failure_label({
        "entropy_score":             signal.entropy_score,
        "agreement_score":           signal.agreement_score,
        "ensemble_disagreement":     signal.ensemble_disagreement,
        "high_failure_risk":         signal.high_failure_risk,
        "latency_ms":                0.0,
        "question_type":             _question_type,
        "is_constitutional_refusal": body.is_constitutional_refusal,
        "has_conversation_context":  _has_context,
    })

    archetype_registry.assign(signal)
    evolution_tracker.record(signal)

    # Spike alert — EMA high-risk rate >= 40%
    try:
        from app.notifications import notify_degradation_spike
        _trend_snap = evolution_tracker.trend_summary()
        _risk_rate  = _trend_snap.get("ema_high_risk_rate", 0.0)
        if _risk_rate >= 0.40:
            _notif_t  = current_user["tenant_id"] if current_user else "anonymous"
            _notif_em = current_user.get("email")  if current_user else None
            notify_degradation_spike(
                tenant_id   = _notif_t,
                risk_pct    = _risk_rate * 100,
                ema_entropy = _trend_snap.get("ema_entropy", 0.0),
                velocity    = _trend_snap.get("degradation_velocity", 0.0),
                total       = _trend_snap.get("signals_count", 0),
                to          = _notif_em,
            )
    except Exception as _spike_exc:
        logger.debug("Spike notification failed (non-fatal): %s", _spike_exc)

    # ── Step 5: DiagnosticJury ─────────────────────────────────────────────
    jury_verdict = None
    if body.run_full_jury:
        diag_request = DiagnosticRequest(
            prompt        = body.prompt,
            model_outputs = model_outputs,
            latency_ms    = body.latency_ms,
            canary_token  = _canary_token,
        )
        diag_response = failure_agent.run_diagnostic(diag_request)
        jury_verdict  = diag_response.jury

    # FAISS auto-growth — persist confirmed adversarial detections
    if jury_verdict and jury_verdict.is_adversarial and jury_verdict.jury_confidence >= 0.85:
        try:
            from engine.archetypes.registry import adversarial_registry
            _adv_pv       = jury_verdict.primary_verdict
            _adv_label    = _adv_pv.root_cause if _adv_pv else "ADVERSARIAL_PROMPT"
            _adv_category = ((_adv_pv.evidence or {}).get("category", "UNKNOWN")) if _adv_pv else "UNKNOWN"
            adversarial_registry.add_confirmed_detection(
                prompt     = body.prompt,
                label      = _adv_label,
                category   = _adv_category,
                confidence = jury_verdict.jury_confidence,
            )
        except Exception as _faiss_exc:
            logger.debug("FAISS auto-growth failed (non-fatal): %s", _faiss_exc)

    # ── Step 5b: Multi-turn escalation ────────────────────────────────────
    multi_turn_result = None
    if body.conversation_id:
        try:
            from engine.multi_turn_tracker import check_multi_turn_escalation
            _jury_is_adversarial = bool(jury_verdict and jury_verdict.is_adversarial)
            _jury_confidence     = jury_verdict.jury_confidence if jury_verdict else 0.0
            mt = check_multi_turn_escalation(
                conversation_id        = body.conversation_id,
                prompt                 = body.prompt,
                question_type          = _question_type,
                is_adversarial         = _jury_is_adversarial,
                adversarial_confidence = _jury_confidence,
            )
            if mt.is_escalating:
                multi_turn_result = {
                    "is_escalating": True,
                    "confidence":    mt.confidence,
                    "pattern":       mt.pattern,
                    "turn_count":    mt.turn_count,
                    "evidence":      mt.evidence,
                }
                logger.warning(
                    "MULTI_TURN_ESCALATION | conv=%s pattern=%s conf=%.3f",
                    body.conversation_id, mt.pattern, mt.confidence,
                )
        except Exception as exc:
            logger.warning("multi_turn_tracker failed (non-fatal): %s", exc)

    # ── Step 5c: Model extraction detection ───────────────────────────────
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
                "MODEL_EXTRACTION | tenant=%s pattern=%s conf=%.3f",
                _tenant, ext.pattern, ext.confidence,
            )
    except Exception as exc:
        logger.debug("model_extraction_tracker failed (non-fatal): %s", exc)

    # ── Step 6: Failure summary ────────────────────────────────────────────
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

    # ── Step 7: Ground Truth pipeline + auto-fix ──────────────────────────
    fix_result_schema     = None
    gt_result_schema      = None
    gt_pipeline_result    = None   # raw dataclass — carries provenance_label/category
    requires_human_review = False
    escalation_reason_str = ""

    if jury_verdict and jury_verdict.primary_verdict:
        try:
            from engine.fix_engine import apply_fix, prompt_requires_live_data
            from app.schemas import FixResult as FixResultSchema, GroundTruthVerification

            primary_v    = jury_verdict.primary_verdict
            root_cause   = primary_v.root_cause
            confidence   = primary_v.confidence_score
            shadow_texts = [r.output_text for r in shadow_model_results if r.success and r.output_text]

            if not signal.high_failure_risk:
                raise RuntimeError("Auto-fix skipped: primary output matches shadow consensus.")

            if confidence < 0.45:
                raise RuntimeError(
                    f"Auto-fix skipped: jury confidence {confidence:.2f} below minimum 0.45."
                )

            if not _pipeline_gates.get("run_wikidata", True) and not _pipeline_gates.get("run_serper", True):
                raise RuntimeError(
                    f"Auto-fix skipped: question_type={_question_type} does not use external GT."
                )

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
                question_type   = _question_type,
            )
            gt_pipeline_result = gt   # keep raw result for provenance enrichment
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
                "GT pipeline done | source=%s escalate=%s confidence=%.3f",
                gt.source, gt.requires_escalation, gt.confidence,
            )

            if gt.requires_escalation:
                requires_human_review = True
                escalation_reason_str = gt.escalation_reason
                failure_summary = (
                    "HUMAN REVIEW REQUIRED: "
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
                    f"AUTO-FIXED: GT_VERIFIED applied via {gt.source}. "
                    f"Confidence: {gt.confidence:.0%}."
                )
                logger.info(
                    "GT pipeline applied fix | source=%s confidence=%.3f",
                    gt.source, gt.confidence,
                )

            else:
                fix = apply_fix(
                    prompt         = body.prompt,
                    primary_output = body.primary_output,
                    shadow_outputs = shadow_texts,
                    root_cause     = root_cause,
                    confidence     = confidence,
                    model_fn       = None,
                    shadow_weights = shadow_weights,
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
                    if fix.requires_human_review:
                        requires_human_review = True
                        escalation_reason_str = fix.escalation_reason
                        failure_summary = (
                            "HUMAN REVIEW REQUIRED: "
                            f"{fix.escalation_reason[:150]}"
                        )
                    elif fix.fix_applied:
                        failure_summary = (
                            f"AUTO-FIXED: {fix.fix_strategy} applied. "
                            f"{fix.fix_explanation[:150]}"
                        )
                        logger.info(
                            "Auto-fix applied | strategy=%s confidence=%.3f",
                            fix.fix_strategy, fix.fix_confidence,
                        )

                # Wikipedia RAG fallback
                if (
                    (fix_result_schema is None or not fix_result_schema.fix_applied)
                    and root_cause in {"KNOWLEDGE_BOUNDARY_FAILURE", "FACTUAL_HALLUCINATION"}
                    and not prompt_requires_live_data(body.prompt)
                ):
                    from engine.rag_grounder import ground_with_wikipedia
                    rag = ground_with_wikipedia(body.prompt, body.primary_output)
                    if rag.success:
                        fix_result_schema = FixResultSchema(
                            fixed_output      = rag.grounded_answer,
                            fix_applied       = True,
                            fix_strategy      = "RAG_GROQ_GROUNDING",
                            fix_explanation   = (
                                "Shadow-model consensus unavailable. FIE used "
                                "Wikipedia-grounded retrieval and Groq-based correction."
                            ),
                            original_output   = body.primary_output,
                            root_cause        = root_cause,
                            fix_confidence    = rag.confidence,
                            improvement_score = rag.confidence,
                            warning           = f"Grounded source: {rag.source}",
                        )
                        failure_summary = "AUTO-FIXED: RAG_GROQ_GROUNDING applied via Wikipedia."

        except Exception as exc:
            if "Auto-fix skipped" in str(exc):
                logger.debug(str(exc))
            else:
                logger.error("Fix engine failed: %s", exc, exc_info=True)

    # ── Step 8: XGBoost classifier (post-GT) ──────────────────────────────
    _xgb_prob   = None
    _config_ver = "default"
    try:
        from engine.failure_classifier import predict as _clf_predict
        from engine.fie_config import get_threshold, get_config_version, MODEL_VERSION

        _jury_conf_xgb    = jury_verdict.jury_confidence if jury_verdict else 0.0
        _jury_verd_str    = (
            jury_verdict.primary_verdict.root_cause
            if jury_verdict and jury_verdict.primary_verdict else "NONE"
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

        # Provenance at this point — GT pipeline result takes priority,
        # otherwise derive from question classifier.
        from engine.question_classifier import classify_provenance_category as _cprov
        _prov_cat_xgb = (
            gt_pipeline_result.provenance_category
            if gt_pipeline_result is not None
            else _cprov(_question_type, body.prompt)
        )
        _prov_lbl_xgb = (
            gt_pipeline_result.provenance_label
            if gt_pipeline_result is not None
            else "UNVERIFIED_MODEL_INFERENCE"
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
            provenance_category = _prov_cat_xgb,
            provenance_label    = _prov_lbl_xgb,
        )

        _xgb_threshold  = get_threshold(_question_type)
        _xgb_is_failure = _xgb_prob >= _xgb_threshold
        _config_ver     = get_config_version()
        signal = signal.model_copy(update={"high_failure_risk": _xgb_is_failure})
        logger.info(
            "XGBoost post-GT: prob=%.4f threshold=%.3f is_failure=%s qt=%s gt_source=%s",
            _xgb_prob, _xgb_threshold, _xgb_is_failure, _question_type, _gt_source_xgb,
        )
    except Exception as _clf_exc:
        logger.warning("XGBoost classifier unavailable, keeping POET decision: %s", _clf_exc)

    # ── Provenance enrichment — copy GT pipeline labels onto FSV ──────────
    # gt_pipeline_result is the raw dataclass with provenance_label/category.
    # When the GT pipeline didn't run (adversarial block, etc.) we derive
    # provenance from question_type so the field is always populated.
    try:
        from engine.question_classifier import classify_provenance_category
        _prov_cat = classify_provenance_category(_question_type, body.prompt)
        if gt_pipeline_result is not None:
            _prov_label = gt_pipeline_result.provenance_label
            _prov_cat   = gt_pipeline_result.provenance_category
        elif _prov_cat in ("LIVE_WORLD_STATE", "USER_SPECIFIC_STATE", "MIXED_SYNTHESIS"):
            # Live/user data but GT pipeline didn't run → mark as requiring verification
            _prov_label = "REQUIRES_TOOL_VERIFICATION"
        else:
            _prov_label = "UNVERIFIED_MODEL_INFERENCE"
        signal = signal.model_copy(update={
            "provenance_category": _prov_cat,
            "provenance_label":    _prov_label,
        })
    except Exception as _prov_exc:
        logger.debug("Provenance enrichment failed (non-fatal): %s", _prov_exc)

    # ── Step 9a: Build and annotate response ──────────────────────────────
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

    stored_request_id = None
    response = attach_explanations_to_monitor(response, request_id=stored_request_id)
    if not (current_user and current_user.get("is_admin", False)):
        response.explanation_internal = None

    # ── Step 9b: Signal logging ────────────────────────────────────────────
    _signal_log_id = ""
    try:
        from storage.signal_logger import log_signal

        _layers_fired: list[str]        = []
        _layer_scores: dict[str, float] = {}
        _jury_verdict_str = ""
        _jury_conf        = 0.0
        if jury_verdict and jury_verdict.primary_verdict:
            pv              = jury_verdict.primary_verdict
            ev              = pv.evidence or {}
            _layers_fired   = ev.get("layers_fired", [])
            _layer_scores   = ev.get("layer_scores", {})
            _jury_verdict_str = pv.root_cause
            _jury_conf      = pv.confidence_score

        _gt_source   = gt_result_schema.source          if gt_result_schema  else "none"
        _gt_conf     = gt_result_schema.confidence      if gt_result_schema  else 0.0
        _gt_override = (
            fix_result_schema is not None
            and fix_result_schema.fix_applied
            and gt_result_schema is not None
            and gt_result_schema.source not in ("none", "shadow_consensus")
        )
        _gt_answer   = gt_result_schema.verified_answer if gt_result_schema  else ""
        _fix_applied = fix_result_schema.fix_applied    if fix_result_schema else False
        _fix_strat   = fix_result_schema.fix_strategy   if fix_result_schema else ""
        _fix_conf    = fix_result_schema.fix_confidence if fix_result_schema else 0.0
        _fix_output  = fix_result_schema.fixed_output   if fix_result_schema else ""
        _shadow_confs= [
            getattr(r, "model_confidence", "MEDIUM") for r in shadow_results_raw if r.success
        ]
        _shadow_texts= [r.output_text for r in shadow_results_raw if r.success and r.output_text]

        _signal_log_id = log_signal(
            request_id             = "",
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
            jury_verdict           = _jury_verdict_str,
            jury_confidence        = _jury_conf,
            gt_source              = _gt_source,
            gt_confidence          = _gt_conf,
            gt_override_applied    = _gt_override,
            gt_verified_answer     = _gt_answer,
            requires_escalation    = requires_human_review,
            escalation_reason      = escalation_reason_str,
            fix_applied            = _fix_applied,
            fix_strategy           = _fix_strat,
            fix_confidence         = _fix_conf,
            fix_output             = _fix_output,
        )
    except Exception as _log_exc:
        logger.debug("Signal logging failed (non-fatal): %s", _log_exc)

    # ── Step 9c: Persist inference record ─────────────────────────────────
    try:
        stored_request_id = str(uuid.uuid4())[:12]

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

        # Email notifications (fire-and-forget)
        try:
            from app.notifications import notify_attack_detected, notify_human_review
            _notif_tenant = current_user["tenant_id"] if current_user else "anonymous"
            _notif_email  = current_user.get("email") if current_user else None
            _is_attack    = bool(jury_verdict and jury_verdict.is_adversarial)

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
            human_explanation    = response.human_explanation,
            explanation_external = response.explanation_external,
        )
        from storage.database import save_inference
        save_inference(inference_record)
    except Exception as exc:
        logger.warning("Failed to save inference record: %s", exc)

    # ── Step 10: Update session store ─────────────────────────────────────
    if body.session_id:
        try:
            from engine.session_store import store_turn
            store_turn(body.session_id, "user",      body.prompt)
            store_turn(body.session_id, "assistant", body.primary_output)
        except Exception as _sess_save_exc:
            logger.debug("SessionStore save failed (non-fatal): %s", _sess_save_exc)

    return response


# ── GET /monitor/status ───────────────────────────────────────────────────────

@router.get("/monitor/status", response_model=dict)
def monitor_status() -> dict:
    """Current Ollama service status and model availability."""
    from engine.ollama_service import ollama_service
    available     = ollama_service.is_available()
    active_models = ollama_service.get_available_models() if available else []
    configured    = ollama_service.models
    return {
        "ollama_running":    available,
        "configured_models": configured,
        "active_models":     active_models,
        "ready_models":      [m for m in configured if m in active_models],
        "missing_models":    [m for m in configured if m not in active_models],
    }


# ── GET /monitor/model-info ───────────────────────────────────────────────────

@router.get("/monitor/model-info", response_model=dict)
def model_info() -> dict:
    """Classifier version, thresholds, AUC, config version. No auth required."""
    from engine.fie_config import (
        get_all_thresholds, get_config_version,
        MODEL_VERSION, MODEL_TRAINED, RECALIBRATION_INTERVAL,
    )
    from engine.failure_classifier import _model, CLASSIFIER_THRESHOLD

    model_loaded = _model is not None
    return {
        "model_version":           MODEL_VERSION,
        "model_trained":           MODEL_TRAINED,
        "model_loaded":            model_loaded,
        "fallback_mode":           "POET rule-based" if not model_loaded else "XGBoost",
        "auc_held_out":            {"xgboost-v2": 0.728, "xgboost-v3": 0.677}.get(MODEL_VERSION, 0.749),
        "default_threshold":       CLASSIFIER_THRESHOLD,
        "note_threshold":          (
            "Per-type thresholds (thresholds_per_type) are used in production. "
            "default_threshold is the fallback."
        ),
        "thresholds_per_type":     get_all_thresholds(),
        "config_version":          get_config_version(),
        "recalibration_interval":  RECALIBRATION_INTERVAL,
        "features": [
            "agreement_score", "entropy_score", "jury_confidence",
            "fix_confidence", "gt_confidence", "high_failure_risk",
            "fix_applied", "requires_escalation", "gt_override",
            "archetype", "jury_verdict", "fix_strategy", "gt_source",
        ],
    }


# ── GET /monitor/calibration ──────────────────────────────────────────────────

@router.get("/monitor/calibration", response_model=dict)
def get_calibration_stats(
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> dict:
    """Calibration statistics from all labeled signal logs. Admin only."""
    require_admin(authorization, x_api_key)
    from storage.signal_logger import get_calibration_stats
    return get_calibration_stats()


# ── GET /monitor/signal-logs ──────────────────────────────────────────────────

@router.get("/monitor/signal-logs", response_model=list)
def get_signal_logs(
    limit:         int = 50,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> list:
    """N most recent raw signal logs for debugging and auditing. Admin only."""
    require_admin(authorization, x_api_key)
    from storage.signal_logger import get_recent_logs
    return get_recent_logs(limit=min(limit, 500))


# ── POST /feedback/{request_id} ───────────────────────────────────────────────

@router.post("/feedback/{request_id}", response_model=FeedbackResponse)
def submit_feedback(
    request_id:    str,
    body:          FeedbackRequest,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> FeedbackResponse:
    """
    Step 8 — Ground Truth Feedback Loop.

    When is_correct=False and correct_answer is provided:
      1. Saves the correct answer to the GT cache (permanent, affects all future requests).
      2. Stores a feedback record for analytics and threshold recalibration.
      3. Adds to the XGBoost retraining buffer.

    Every correction permanently improves the system.
    """
    from storage.database import save_feedback, get_inference_by_id_for_tenant
    from engine.ground_truth_cache import save_to_cache

    user = require_user(authorization, x_api_key)

    record = (
        get_inference_by_id(request_id)
        if user.get("is_admin", False)
        else get_inference_by_id_for_tenant(request_id, user["tenant_id"])
    )
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Inference '{request_id}' not found for this account",
        )

    cache_updated = False
    if not body.is_correct and body.correct_answer:
        cache_updated = save_to_cache(
            question        = record.input_text,
            verified_answer = body.correct_answer.strip(),
            source          = "user_feedback",
            confidence      = 1.0,
            verified_by     = user.get("email", "user"),
        )
        logger.info(
            "GT cache updated from feedback | request_id=%s correct=%s",
            request_id, body.correct_answer[:60],
        )

    # Update signal log with labeled outcome
    try:
        from storage.signal_logger import find_log_by_request_id, update_signal_feedback
        sig_log = find_log_by_request_id(request_id)
        if sig_log:
            fie_flagged   = sig_log.get("high_failure_risk", False)
            fie_corrected = sig_log.get("fix_applied", False)
            fie_was_correct = (not fie_corrected) if body.is_correct else fie_flagged
            update_signal_feedback(
                log_id          = sig_log["log_id"],
                fie_was_correct = fie_was_correct,
                correct_answer  = body.correct_answer or "",
            )
    except Exception as _fe:
        logger.debug("Signal feedback update failed (non-fatal): %s", _fe)

    feedback_doc = {
        "request_id":     request_id,
        "tenant_id":      user["tenant_id"],
        "submitted_by":   user.get("email", "unknown"),
        "submitted_at":   datetime.utcnow().isoformat(),
        "is_correct":     body.is_correct,
        "correct_answer": body.correct_answer or "",
        "notes":          body.notes or "",
        "question":       record.input_text,
        "model_answer":   record.output_text,
        "model_name":     record.model_name,
    }
    save_feedback(feedback_doc)

    try:
        from engine.fie_config import maybe_recalibrate
        maybe_recalibrate()
    except Exception:
        pass

    try:
        from engine.retraining.buffer import add_to_buffer, maybe_trigger_retrain
        from storage.signal_logger import find_log_by_request_id as _flbr
        _sig = _flbr(request_id)
        if _sig:
            _buf_count = add_to_buffer(
                log_id         = _sig.get("log_id", ""),
                request_id     = request_id,
                is_failure     = not body.is_correct,
                correct_answer = body.correct_answer or "",
            )
            maybe_trigger_retrain(_buf_count)
    except Exception as _buf_exc:
        logger.debug("Retraining buffer update failed (non-fatal): %s", _buf_exc)

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


# ── local import needed by submit_feedback ────────────────────────────────────
from storage.database import get_inference_by_id  # noqa: E402
