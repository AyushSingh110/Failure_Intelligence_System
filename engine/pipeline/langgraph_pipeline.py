"""
Full LangGraph orchestration pipeline for the Failure Intelligence Engine.

Every request to /monitor is routed through this graph.  Each node is
responsible for exactly one concern; all shared state flows through
MonitorState.  Nodes return only the keys they write — LangGraph merges
the partial dict back into the state automatically.

Graph topology
--------------

  START
    │
    ▼
  load_session          ← fetch prior turns from session store (optional)
    │
    ▼
  shadow_inference      ← fan-out to Groq shadow models + canary injection
    │
    ▼
  adversarial_guard     ← 7-layer offline scan_prompt (no LLM call, ~15 ms)
    │
    ├─ BLOCKED ──────────────────────────────────────────────► END
    │
    └─ SAFE ──►  signal_extract   ← FSV + question classify + archetype + EMA
                    │
                    ▼
                 jury_deliberate  ← DiagnosticJury (optional, gated by run_full_jury)
                    │
                    ▼
                 security_checks  ← multi-turn Crescendo + model-extraction
                    │
                    ▼
                 should_verify?   ← routing: high_failure_risk + jury confidence gate
                    │
                    ├─ NO ───────────────────────────────────► finalize
                    │
                    └─ YES ──►  gt_verify   ← Wikidata / Serper / self-consistency
                                   │
                                   ├─ ESCALATE ──►  escalate  ──► finalize
                                   │
                                   └─ FIX ──────►  auto_correct ► finalize
                                                       │
                                                       ▼
                                                    finalize     ← xgboost classify +
                                                                   signal log + persist
                                                       │
                                                       ▼
                                                     END

Design notes
------------
- Every node is a plain function returning ``dict``.  No async — the
  underlying services (Groq, MongoDB, Wikidata) all use the ``requests``
  library synchronously.
- The compiled graph is a module-level singleton (thread-safe, stateless
  between invocations).
- ``pipeline_trace`` is a list[str] that every node appends to; it gives
  the XAI explanation panel a step-by-step audit trail.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class MonitorState(TypedDict, total=False):
    # ── Request inputs ───────────────────────────────────────────────────
    prompt:                   str
    primary_output:           str
    primary_model_name:       str
    run_full_jury:            bool
    session_id:               Optional[str]
    conversation_id:          Optional[str]
    context:                  list[dict]
    is_constitutional_refusal: bool
    tenant_id:                Optional[str]
    tenant_email:             Optional[str]
    latency_ms:               Optional[float]
    request_id:               str

    # ── Session ──────────────────────────────────────────────────────────
    session_context:          list[dict]

    # ── Shadow models ────────────────────────────────────────────────────
    model_outputs:            list[str]
    shadow_results_raw:       list[Any]
    shadow_weights:           list[float]
    shadow_texts:             list[str]
    canary_token:             str
    ollama_available:         bool

    # ── Adversarial guard ────────────────────────────────────────────────
    is_adversarial:           bool
    attack_type:              Optional[str]
    attack_category:          Optional[str]
    attack_confidence:        float
    attack_layers_fired:      list[str]
    attack_evidence:          dict
    guard_blocked:            bool

    # ── Signal extraction ────────────────────────────────────────────────
    failure_signal:           Optional[Any]          # FailureSignalVector schema
    question_type:            str
    archetype:                str
    embedding_distance:       float

    # ── Jury deliberation ────────────────────────────────────────────────
    jury_verdict:             Optional[Any]           # JuryVerdict schema
    jury_confidence:          float
    failure_summary:          str
    root_cause:               str

    # ── Security checks ──────────────────────────────────────────────────
    multi_turn_result:        Optional[dict]
    extraction_result:        Optional[dict]

    # ── Ground truth verification ────────────────────────────────────────
    gt_result:                Optional[Any]           # GroundTruthVerification schema
    verified_answer:          Optional[str]
    gt_confidence:            float
    gt_source:                str

    # ── Auto-correction ──────────────────────────────────────────────────
    fix_result:               Optional[Any]           # FixResult schema
    corrected_answer:         Optional[str]

    # ── XGBoost post-GT classifier ───────────────────────────────────────
    xgb_probability:          Optional[float]
    xgb_is_failure:           bool
    config_version:           str

    # ── Escalation ───────────────────────────────────────────────────────
    requires_human_review:    bool
    escalation_reason:        str

    # ── Audit trail ──────────────────────────────────────────────────────
    pipeline_trace:           list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trace(state: MonitorState, msg: str) -> list[str]:
    trace = list(state.get("pipeline_trace") or [])
    trace.append(msg)
    return trace


# ── Node 1: load_session ──────────────────────────────────────────────────────

def load_session(state: MonitorState) -> dict:
    """Fetch prior conversation turns from the session store when session_id is set."""
    session_id = state.get("session_id")
    if not session_id or state.get("context"):
        return {"session_context": state.get("context") or [],
                "pipeline_trace": _trace(state, "load_session: skipped (no session_id or context already set)")}

    try:
        from engine.session_store import get_context
        ctx = get_context(session_id) or []
        logger.info("load_session: fetched %d turns for session=%s", len(ctx), session_id)
        return {
            "session_context": ctx,
            "pipeline_trace":  _trace(state, f"load_session: loaded {len(ctx)} turns from store"),
        }
    except Exception as exc:
        logger.warning("load_session failed (non-fatal): %s", exc)
        return {
            "session_context": [],
            "pipeline_trace":  _trace(state, f"load_session: error — {exc}"),
        }


# ── Node 2: shadow_inference ──────────────────────────────────────────────────

def shadow_inference(state: MonitorState) -> dict:
    """Fan out prompt to Groq shadow models and collect outputs + confidence weights."""
    from engine.canary_tracker import generate_canary, build_canary_system_prompt
    from config import get_settings

    settings       = get_settings()
    canary_token   = generate_canary()
    canary_sysprom = build_canary_system_prompt(canary_token)

    model_outputs       = [state["primary_output"]]
    shadow_results_raw  = []
    shadow_weights:     list[float] = []
    ollama_available    = False
    trace               = _trace(state, "shadow_inference: starting Groq fan-out")

    if settings.groq_enabled and settings.groq_api_key:
        try:
            from engine.groq_service import get_groq_service
            groq = get_groq_service()
            if groq:
                ctx = state.get("session_context") or state.get("context") or []
                results = groq.fan_out_with_confidence(
                    state["prompt"],
                    system_message=canary_sysprom,
                )
                shadow_results_raw = results
                successful = [r for r in results if r.success]
                ollama_available = len(successful) > 0

                for r in results:
                    if r.success and r.output_text:
                        model_outputs.append(r.output_text)
                        shadow_weights.append(getattr(r, "confidence_weight", 2.0))

                logger.info(
                    "shadow_inference: %d/%d shadow models responded",
                    len(successful), len(results),
                )
                trace = _trace(
                    {**state, "pipeline_trace": trace},
                    f"shadow_inference: {len(successful)}/{len(results)} models responded",
                )
            else:
                trace = _trace({**state, "pipeline_trace": trace}, "shadow_inference: Groq unavailable")
        except Exception as exc:
            logger.warning("shadow_inference Groq call failed: %s", exc)
            trace = _trace({**state, "pipeline_trace": trace}, f"shadow_inference: Groq error — {exc}")
    else:
        trace = _trace({**state, "pipeline_trace": trace}, "shadow_inference: Groq not configured")

    shadow_texts = [r.output_text for r in shadow_results_raw if getattr(r, "success", False) and r.output_text]

    return {
        "canary_token":      canary_token,
        "model_outputs":     model_outputs,
        "shadow_results_raw": shadow_results_raw,
        "shadow_weights":    shadow_weights,
        "shadow_texts":      shadow_texts,
        "ollama_available":  ollama_available,
        "pipeline_trace":    trace,
    }


# ── Node 3: adversarial_guard ─────────────────────────────────────────────────

def adversarial_guard(state: MonitorState) -> dict:
    """
    Run offline 7-layer adversarial scan.  Fast path — no LLM call, ~15 ms.
    Sets guard_blocked=True when a confirmed attack is detected.
    """
    try:
        from fie.adversarial import scan_prompt
        result = scan_prompt(state["prompt"], primary_output=state.get("primary_output", ""))

        blocked = result.is_attack
        trace_msg = (
            f"adversarial_guard: {'BLOCKED' if blocked else 'SAFE'} "
            f"| type={result.attack_type} | conf={result.confidence:.3f} "
            f"| layers={result.layers_fired}"
        )
        logger.info(trace_msg)

        return {
            "is_adversarial":      result.is_attack,
            "attack_type":         result.attack_type,
            "attack_category":     result.category,
            "attack_confidence":   result.confidence,
            "attack_layers_fired": result.layers_fired,
            "attack_evidence":     result.evidence or {},
            "guard_blocked":       blocked,
            "pipeline_trace":      _trace(state, trace_msg),
        }
    except Exception as exc:
        logger.error("adversarial_guard failed: %s", exc, exc_info=True)
        return {
            "is_adversarial":    False,
            "guard_blocked":     False,
            "attack_confidence": 0.0,
            "pipeline_trace":    _trace(state, f"adversarial_guard: error — {exc}"),
        }


# ── Routing: blocked? ─────────────────────────────────────────────────────────

def _route_after_guard(state: MonitorState) -> str:
    return "END" if state.get("guard_blocked") else "signal_extract"


# ── Node 4: signal_extract ────────────────────────────────────────────────────

def signal_extract(state: MonitorState) -> dict:
    """
    Phase 1 + 2:
      - Build FailureSignalVector (entropy, agreement, ensemble disagreement)
      - Classify question type
      - Assign failure archetype
      - Update FAISS registry and EMA tracker
    """
    from engine.detector.consistency import compute_consistency, is_primary_outlier
    from engine.detector.entropy import compute_entropy_from_counts
    from engine.detector.ensemble import compute_disagreement
    from engine.detector.embedding import compute_embedding_distance
    from engine.archetypes.labeling import assign_failure_label
    from engine.archetypes.clustering import archetype_registry
    from engine.evolution.tracker import evolution_tracker
    from engine.question_classifier import classify as classify_question, pipeline_gates
    from app.schemas import FailureSignalVector
    from config import get_settings

    settings      = get_settings()
    model_outputs = state["model_outputs"]
    primary       = model_outputs[0]
    secondary     = model_outputs[1] if len(model_outputs) > 1 else primary

    consistency   = compute_consistency(model_outputs)
    entropy_score = compute_entropy_from_counts(consistency["answer_counts"], len(model_outputs))
    ensemble      = compute_disagreement(model_outputs)
    embedding     = compute_embedding_distance(primary, secondary)

    primary_outlier   = is_primary_outlier(primary, model_outputs[1:])
    high_failure_risk = (
        primary_outlier
        or entropy_score >= settings.high_entropy_threshold
        or (ensemble["disagreement"] and primary_outlier)
    )

    question_type  = classify_question(state["prompt"])
    gates          = pipeline_gates(question_type)
    has_context    = bool(state.get("session_context") or state.get("context"))

    signal = FailureSignalVector(
        agreement_score       = consistency["agreement_score"],
        fsd_score             = consistency["fsd_score"],
        answer_counts         = consistency["answer_counts"],
        entropy_score         = entropy_score,
        ensemble_disagreement = ensemble["disagreement"],
        ensemble_similarity   = ensemble["similarity_score"],
        high_failure_risk     = high_failure_risk,
        question_type         = question_type,
    )

    archetype = assign_failure_label({
        "entropy_score":             signal.entropy_score,
        "agreement_score":           signal.agreement_score,
        "ensemble_disagreement":     signal.ensemble_disagreement,
        "high_failure_risk":         signal.high_failure_risk,
        "latency_ms":                state.get("latency_ms") or 0.0,
        "question_type":             question_type,
        "is_constitutional_refusal": state.get("is_constitutional_refusal", False),
        "has_conversation_context":  has_context,
    })

    archetype_registry.assign(signal)
    evolution_tracker.record(signal)

    # Degradation spike alert
    try:
        from app.notifications import notify_degradation_spike
        snap      = evolution_tracker.trend_summary()
        risk_rate = snap.get("ema_high_risk_rate", 0.0)
        if risk_rate >= 0.40:
            notify_degradation_spike(
                tenant_id   = state.get("tenant_id") or "anonymous",
                risk_pct    = risk_rate * 100,
                ema_entropy = snap.get("ema_entropy", 0.0),
                velocity    = snap.get("degradation_velocity", 0.0),
                total       = snap.get("signals_count", 0),
                to          = state.get("tenant_email"),
            )
    except Exception as exc:
        logger.debug("Degradation spike notification failed (non-fatal): %s", exc)

    trace_msg = (
        f"signal_extract: qt={question_type} archetype={archetype} "
        f"entropy={entropy_score:.3f} agreement={signal.agreement_score:.3f} "
        f"high_risk={high_failure_risk}"
    )
    logger.info(trace_msg)

    return {
        "failure_signal":    signal,
        "question_type":     question_type,
        "archetype":         archetype,
        "embedding_distance": embedding["embedding_distance"],
        "pipeline_trace":    _trace(state, trace_msg),
    }


# ── Node 5: jury_deliberate ───────────────────────────────────────────────────

def jury_deliberate(state: MonitorState) -> dict:
    """
    Phase 3 — DiagnosticJury: adversarial specialist + linguistic auditor + domain critic.
    Skipped when run_full_jury=False (fast path for real-time monitoring).
    """
    if not state.get("run_full_jury"):
        return {
            "jury_verdict":   None,
            "jury_confidence": 0.0,
            "failure_summary": f"Model outputs are stable — archetype: {state.get('archetype', 'UNKNOWN')}",
            "root_cause":      "NONE",
            "pipeline_trace":  _trace(state, "jury_deliberate: skipped (run_full_jury=False)"),
        }

    try:
        from app.schemas import DiagnosticRequest
        from engine.agents.failure_agent import failure_agent

        req      = DiagnosticRequest(
            prompt=state["prompt"],
            model_outputs=state["model_outputs"],
            latency_ms=state.get("latency_ms"),
            canary_token=state.get("canary_token", ""),
        )
        response     = failure_agent.run_diagnostic(req)
        jury         = response.jury
        root_cause   = ""
        jury_conf    = jury.jury_confidence if jury else 0.0
        summary      = jury.failure_summary if jury else ""

        if jury and jury.primary_verdict:
            root_cause = jury.primary_verdict.root_cause or ""

        # FAISS auto-growth for confirmed adversarial detections
        if jury and jury.is_adversarial and jury_conf >= 0.85:
            try:
                from engine.archetypes.registry import adversarial_registry
                pv    = jury.primary_verdict
                label = pv.root_cause if pv else "ADVERSARIAL_PROMPT"
                cat   = ((pv.evidence or {}).get("category", "UNKNOWN")) if pv else "UNKNOWN"
                adversarial_registry.add_confirmed_detection(
                    prompt=state["prompt"], label=label,
                    category=cat, confidence=jury_conf,
                )
            except Exception as exc:
                logger.debug("FAISS auto-growth failed (non-fatal): %s", exc)

        trace_msg = (
            f"jury_deliberate: is_adversarial={jury.is_adversarial if jury else False} "
            f"confidence={jury_conf:.3f} root_cause={root_cause}"
        )
        logger.info(trace_msg)

        return {
            "jury_verdict":    jury,
            "jury_confidence": jury_conf,
            "failure_summary": summary,
            "root_cause":      root_cause,
            "pipeline_trace":  _trace(state, trace_msg),
        }
    except Exception as exc:
        logger.error("jury_deliberate failed: %s", exc, exc_info=True)
        return {
            "jury_verdict":    None,
            "jury_confidence": 0.0,
            "failure_summary": "Jury error — manual review recommended.",
            "root_cause":      "UNKNOWN",
            "pipeline_trace":  _trace(state, f"jury_deliberate: error — {exc}"),
        }


# ── Node 6: security_checks ───────────────────────────────────────────────────

def security_checks(state: MonitorState) -> dict:
    """Multi-turn Crescendo tracking + model-extraction detection."""
    updates: dict = {"pipeline_trace": list(state.get("pipeline_trace") or [])}

    # Multi-turn Crescendo
    if state.get("conversation_id"):
        try:
            from engine.multi_turn_tracker import check_multi_turn_escalation
            jury   = state.get("jury_verdict")
            mt = check_multi_turn_escalation(
                conversation_id        = state["conversation_id"],
                prompt                 = state["prompt"],
                question_type          = state.get("question_type", "UNKNOWN"),
                is_adversarial         = bool(jury and jury.is_adversarial),
                adversarial_confidence = jury.jury_confidence if jury else 0.0,
            )
            if mt.is_escalating:
                updates["multi_turn_result"] = {
                    "is_escalating": True,
                    "confidence":    mt.confidence,
                    "pattern":       mt.pattern,
                    "turn_count":    mt.turn_count,
                    "evidence":      mt.evidence,
                }
                logger.warning(
                    "MULTI_TURN_ESCALATION | conv=%s pattern=%s conf=%.3f",
                    state["conversation_id"], mt.pattern, mt.confidence,
                )
                updates["pipeline_trace"] = _trace(
                    {**state, "pipeline_trace": updates["pipeline_trace"]},
                    f"security_checks: multi_turn escalation pattern={mt.pattern}",
                )
        except Exception as exc:
            logger.warning("multi_turn_tracker failed (non-fatal): %s", exc)

    # Model extraction
    try:
        from engine.model_extraction_tracker import check_model_extraction
        ext = check_model_extraction(
            tenant_id       = state.get("tenant_id") or "anonymous",
            prompt          = state.get("prompt") or "",
            conversation_id = state.get("conversation_id"),
        )
        if ext.is_extracting:
            updates["extraction_result"] = {
                "is_extracting": True,
                "confidence":    ext.confidence,
                "pattern":       ext.pattern,
                "evidence":      ext.evidence,
            }
            logger.warning(
                "MODEL_EXTRACTION | tenant=%s pattern=%s conf=%.3f",
                state.get("tenant_id"), ext.pattern, ext.confidence,
            )
            updates["pipeline_trace"] = _trace(
                {**state, "pipeline_trace": updates["pipeline_trace"]},
                f"security_checks: model extraction pattern={ext.pattern}",
            )
    except Exception as exc:
        logger.debug("model_extraction_tracker failed (non-fatal): %s", exc)

    if "multi_turn_result" not in updates:
        updates["multi_turn_result"] = None
    if "extraction_result" not in updates:
        updates["extraction_result"] = None

    return updates


# ── Routing: should we run GT verification? ───────────────────────────────────

def _route_to_verify(state: MonitorState) -> str:
    signal = state.get("failure_signal")
    jury   = state.get("jury_verdict")
    if not signal:
        return "finalize"
    high_risk   = getattr(signal, "high_failure_risk", False)
    jury_conf   = state.get("jury_confidence", 0.0)
    gate_passed = jury_conf >= 0.45 if state.get("run_full_jury") else high_risk
    return "gt_verify" if gate_passed and high_risk else "finalize"


# ── Node 7: gt_verify ─────────────────────────────────────────────────────────

def gt_verify(state: MonitorState) -> dict:
    """
    Phase 4 — Ground Truth Pipeline.
    Routes by question type: FACTUAL→Wikidata+Serper, TEMPORAL→Serper,
    REASONING/CODE→self-consistency, OPINION/IDENTITY→skip.
    """
    try:
        from engine.question_classifier import pipeline_gates
        from engine.verifier.ground_truth_pipeline import run_ground_truth_pipeline

        jury     = state.get("jury_verdict")
        root_c   = state.get("root_cause", "UNKNOWN")
        jury_conf = state.get("jury_confidence", 0.0)
        qt        = state.get("question_type", "UNKNOWN")
        gates     = pipeline_gates(qt)

        # Skip external lookup for opinion/identity/code/reasoning
        if not gates.get("run_wikidata", True) and not gates.get("run_serper", True):
            return {
                "gt_result":    None,
                "pipeline_trace": _trace(state, f"gt_verify: skipped for question_type={qt}"),
            }

        shadow_texts = state.get("shadow_texts") or []
        shadow_weights = state.get("shadow_weights") or []

        gt = run_ground_truth_pipeline(
            prompt          = state["prompt"],
            primary_output  = state["primary_output"],
            root_cause      = root_c,
            jury_confidence = jury_conf,
            shadow_outputs  = shadow_texts,
            shadow_weights  = shadow_weights,
            use_wikidata    = gates.get("run_wikidata", True),
            use_serper      = gates.get("run_serper", True),
            question_type   = qt,
        )

        trace_msg = (
            f"gt_verify: source={gt.source} confidence={gt.confidence:.3f} "
            f"escalate={gt.requires_escalation}"
        )
        logger.info(trace_msg)

        return {
            "gt_result":      gt,
            "verified_answer": gt.verified_answer or None,
            "gt_confidence":  gt.confidence,
            "gt_source":      gt.source,
            "pipeline_trace": _trace(state, trace_msg),
        }
    except Exception as exc:
        logger.error("gt_verify failed: %s", exc, exc_info=True)
        return {
            "gt_result":      None,
            "pipeline_trace": _trace(state, f"gt_verify: error — {exc}"),
        }


# ── Routing: escalate or fix? ─────────────────────────────────────────────────

def _route_after_gt(state: MonitorState) -> str:
    gt = state.get("gt_result")
    if gt is None:
        return "finalize"
    if gt.requires_escalation:
        return "escalate"
    return "auto_correct"


# ── Node 8: escalate ──────────────────────────────────────────────────────────

def escalate(state: MonitorState) -> dict:
    """Mark the inference as requiring human review and build the escalation response."""
    from app.schemas import FixResult as FixResultSchema, GroundTruthVerification

    gt     = state.get("gt_result")
    reason = gt.escalation_reason if gt else "GT pipeline found no reliable source."

    fix_result = FixResultSchema(
        fixed_output      = state.get("primary_output", ""),
        fix_applied       = False,
        fix_strategy      = "HUMAN_ESCALATION",
        fix_explanation   = reason,
        original_output   = state.get("primary_output", ""),
        root_cause        = state.get("root_cause", "UNKNOWN"),
        fix_confidence    = 0.0,
        improvement_score = 0.0,
        warning           = "FIE could not verify a reliable correction. Queued for human review.",
    )

    trace_msg = f"escalate: human review required — {reason[:120]}"
    logger.warning(trace_msg)

    return {
        "fix_result":           fix_result,
        "requires_human_review": True,
        "escalation_reason":    reason,
        "failure_summary":      f"HUMAN REVIEW REQUIRED: {reason[:150]}",
        "pipeline_trace":       _trace(state, trace_msg),
    }


# ── Node 9: auto_correct ──────────────────────────────────────────────────────

def auto_correct(state: MonitorState) -> dict:
    """
    Apply the best available correction strategy:
      1. GT verified answer (Wikidata / Serper override)
      2. Fix engine (weighted shadow consensus via Groq)
      3. RAG Wikipedia grounding (factual fallback)
    """
    from app.schemas import FixResult as FixResultSchema

    gt     = state.get("gt_result")
    root_c = state.get("root_cause", "UNKNOWN")

    # ── Strategy 1: GT verified answer ──────────────────────────────────
    if gt and gt.verified_answer and gt.verified_answer != state.get("primary_output"):
        fix = FixResultSchema(
            fixed_output      = gt.verified_answer,
            fix_applied       = True,
            fix_strategy      = f"GT_VERIFIED ({gt.source})",
            fix_explanation   = (
                f"Ground truth pipeline verified the correct answer from {gt.source}. "
                f"Primary output contradicted the verified source. "
                f"Confidence: {gt.confidence:.0%}."
            ),
            original_output   = state.get("primary_output", ""),
            root_cause        = root_c,
            fix_confidence    = gt.confidence,
            improvement_score = gt.confidence,
            warning           = "",
        )
        trace_msg = f"auto_correct: GT_VERIFIED via {gt.source} (conf={gt.confidence:.3f})"
        logger.info(trace_msg)
        return {
            "fix_result":      fix,
            "corrected_answer": gt.verified_answer,
            "failure_summary": f"AUTO-FIXED: GT_VERIFIED applied via {gt.source}.",
            "pipeline_trace":  _trace(state, trace_msg),
        }

    # ── Strategy 2: Fix engine (shadow consensus) ────────────────────────
    try:
        from engine.fix_engine import apply_fix
        fix_obj = apply_fix(
            prompt         = state["prompt"],
            primary_output = state["primary_output"],
            shadow_outputs = state.get("shadow_texts") or [],
            root_cause     = root_c,
            confidence     = state.get("jury_confidence", 0.0),
            model_fn       = None,
            shadow_weights = state.get("shadow_weights") or [],
        )
        if fix_obj is not None and fix_obj.fix_applied:
            fix = FixResultSchema(
                fixed_output      = fix_obj.fixed_output,
                fix_applied       = fix_obj.fix_applied,
                fix_strategy      = fix_obj.fix_strategy,
                fix_explanation   = fix_obj.fix_explanation,
                original_output   = fix_obj.original_output,
                root_cause        = fix_obj.root_cause,
                fix_confidence    = fix_obj.fix_confidence,
                improvement_score = fix_obj.improvement_score,
                warning           = fix_obj.warning,
            )
            if fix_obj.requires_human_review:
                trace_msg = f"auto_correct: fix_engine escalated — {fix_obj.escalation_reason[:80]}"
                return {
                    "fix_result":           fix,
                    "requires_human_review": True,
                    "escalation_reason":    fix_obj.escalation_reason,
                    "failure_summary":      f"HUMAN REVIEW REQUIRED: {fix_obj.escalation_reason[:150]}",
                    "pipeline_trace":       _trace(state, trace_msg),
                }
            trace_msg = f"auto_correct: fix_engine applied {fix_obj.fix_strategy} (conf={fix_obj.fix_confidence:.3f})"
            logger.info(trace_msg)
            return {
                "fix_result":      fix,
                "corrected_answer": fix_obj.fixed_output,
                "failure_summary": f"AUTO-FIXED: {fix_obj.fix_strategy}.",
                "pipeline_trace":  _trace(state, trace_msg),
            }
    except Exception as exc:
        logger.warning("auto_correct fix_engine failed: %s", exc)

    # ── Strategy 3: RAG Wikipedia grounding ─────────────────────────────
    if root_c in {"KNOWLEDGE_BOUNDARY_FAILURE", "FACTUAL_HALLUCINATION"}:
        try:
            from engine.fix_engine import prompt_requires_live_data
            from engine.rag_grounder import ground_with_wikipedia

            if not prompt_requires_live_data(state["prompt"]):
                rag = ground_with_wikipedia(state["prompt"], state["primary_output"])
                if rag.success:
                    fix = FixResultSchema(
                        fixed_output      = rag.grounded_answer,
                        fix_applied       = True,
                        fix_strategy      = "RAG_GROQ_GROUNDING",
                        fix_explanation   = "Wikipedia-grounded retrieval + Groq correction.",
                        original_output   = state.get("primary_output", ""),
                        root_cause        = root_c,
                        fix_confidence    = rag.confidence,
                        improvement_score = rag.confidence,
                        warning           = f"Grounded source: {rag.source}",
                    )
                    trace_msg = f"auto_correct: RAG_GROQ_GROUNDING applied (conf={rag.confidence:.3f})"
                    logger.info(trace_msg)
                    return {
                        "fix_result":      fix,
                        "corrected_answer": rag.grounded_answer,
                        "failure_summary": "AUTO-FIXED: RAG_GROQ_GROUNDING applied via Wikipedia.",
                        "pipeline_trace":  _trace(state, trace_msg),
                    }
        except Exception as exc:
            logger.warning("auto_correct RAG fallback failed: %s", exc)

    return {
        "fix_result":     None,
        "pipeline_trace": _trace(state, "auto_correct: no fix strategy succeeded"),
    }


# ── Node 10: finalize ─────────────────────────────────────────────────────────

def finalize(state: MonitorState) -> dict:
    """
    Post-pipeline tasks (run regardless of path):
      - XGBoost post-GT classifier override
      - Persist inference record to MongoDB
      - Write signal log for future calibration
      - Store this turn in session store
      - Send email notifications (fire-and-forget)
    """
    from engine.fie_config import get_threshold, get_config_version, MODEL_VERSION

    signal     = state.get("failure_signal")
    jury       = state.get("jury_verdict")
    gt         = state.get("gt_result")
    fix        = state.get("fix_result")
    qt         = state.get("question_type", "UNKNOWN")
    archetype  = state.get("archetype", "UNKNOWN")

    # ── XGBoost post-GT classifier ───────────────────────────────────────
    xgb_prob    = None
    config_ver  = "default"
    xgb_failure = state.get("failure_signal") and getattr(state["failure_signal"], "high_failure_risk", False)

    try:
        from engine.failure_classifier import predict as clf_predict

        jury_conf    = state.get("jury_confidence", 0.0)
        jury_rc      = state.get("root_cause", "NONE")
        gt_source    = gt.source if gt else "none"
        gt_conf_val  = gt.confidence if gt else 0.0
        fix_applied  = fix.fix_applied if fix else False
        fix_strategy = fix.fix_strategy if fix else ""
        fix_conf     = fix.fix_confidence if fix else 0.0
        gt_override  = (
            fix is not None and fix.fix_applied
            and gt is not None and gt.source not in ("none", "shadow_consensus")
        )

        xgb_failure, xgb_prob = clf_predict(
            agreement_score     = getattr(signal, "agreement_score", 0.0) if signal else 0.0,
            entropy_score       = getattr(signal, "entropy_score", 0.0) if signal else 0.0,
            jury_confidence     = jury_conf,
            fix_confidence      = fix_conf,
            gt_confidence       = gt_conf_val,
            high_failure_risk   = getattr(signal, "high_failure_risk", False) if signal else False,
            fix_applied         = fix_applied,
            requires_escalation = state.get("requires_human_review", False),
            gt_override         = gt_override,
            archetype           = archetype,
            jury_verdict_str    = jury_rc,
            fix_strategy        = fix_strategy,
            gt_source           = gt_source,
            question_type       = qt,
        )
        threshold   = get_threshold(qt)
        xgb_failure = xgb_prob >= threshold
        config_ver  = get_config_version()

        if signal is not None:
            signal = signal.model_copy(update={"high_failure_risk": xgb_failure})

        logger.info(
            "finalize.xgboost: prob=%.4f threshold=%.3f failure=%s qt=%s",
            xgb_prob, threshold, xgb_failure, qt,
        )
    except Exception as exc:
        config_ver = "default"
        logger.warning("XGBoost classifier unavailable: %s", exc)

    # ── Failure summary (fallback if not already set by earlier nodes) ───
    failure_summary = state.get("failure_summary") or (
        f"High failure risk — archetype: {archetype}. "
        f"Entropy: {getattr(signal, 'entropy_score', 0):.3f}, "
        f"Agreement: {getattr(signal, 'agreement_score', 0):.3f}"
        if xgb_failure else
        f"Model outputs are stable — archetype: {archetype}"
    )

    # ── Signal logging ───────────────────────────────────────────────────
    signal_log_id = ""
    try:
        from storage.signal_logger import log_signal

        layers_fired  = []
        layer_scores  = {}
        jury_verdict_str = ""
        jury_conf_log = 0.0

        if jury and jury.primary_verdict:
            pv = jury.primary_verdict
            ev = pv.evidence or {}
            layers_fired     = ev.get("layers_fired", [])
            layer_scores     = ev.get("layer_scores", {})
            jury_verdict_str = pv.root_cause
            jury_conf_log    = pv.confidence_score

        signal_log_id = log_signal(
            request_id             = state.get("request_id", ""),
            prompt                 = state["prompt"],
            primary_output         = state["primary_output"],
            shadow_outputs         = state.get("shadow_texts") or [],
            shadow_confidences     = [],
            shadow_weights         = state.get("shadow_weights") or [],
            entropy_score          = getattr(signal, "entropy_score", 0.0) if signal else 0.0,
            agreement_score        = getattr(signal, "agreement_score", 0.0) if signal else 0.0,
            fsd_score              = getattr(signal, "fsd_score", 0.0) if signal else 0.0,
            ensemble_disagreement  = bool(getattr(signal, "ensemble_disagreement", False)) if signal else False,
            high_failure_risk      = xgb_failure,
            classifier_probability = xgb_prob,
            question_type          = qt,
            model_version          = MODEL_VERSION,
            config_version         = config_ver,
            layers_fired           = layers_fired,
            layer_scores           = layer_scores,
            jury_verdict           = jury_verdict_str,
            jury_confidence        = jury_conf_log,
            gt_source              = gt.source if gt else "none",
            gt_confidence          = gt.confidence if gt else 0.0,
            gt_override_applied    = bool(fix and fix.fix_applied and gt and gt.source not in ("none", "shadow_consensus")),
            gt_verified_answer     = gt.verified_answer if gt else "",
            requires_escalation    = state.get("requires_human_review", False),
            escalation_reason      = state.get("escalation_reason", ""),
            fix_applied            = fix.fix_applied if fix else False,
            fix_strategy           = fix.fix_strategy if fix else "",
            fix_confidence         = fix.fix_confidence if fix else 0.0,
            fix_output             = fix.fixed_output if fix else "",
        )
    except Exception as exc:
        logger.debug("Signal logging failed (non-fatal): %s", exc)

    # ── Persist inference record ─────────────────────────────────────────
    stored_request_id = state.get("request_id") or str(uuid.uuid4())[:12]
    try:
        from storage.database import save_inference
        from app.schemas import InferenceRequest, MathematicalMetrics

        record = InferenceRequest(
            request_id    = stored_request_id,
            tenant_id     = state.get("tenant_id") or "anonymous",
            timestamp     = datetime.now(timezone.utc),
            model_name    = state.get("primary_model_name") or "unknown",
            model_version = "monitor-v2",
            temperature   = 0.7,
            latency_ms    = state.get("latency_ms") or 0.0,
            input_text    = state["prompt"],
            output_text   = state["primary_output"],
            metrics       = MathematicalMetrics(
                entropy            = getattr(signal, "entropy_score", 0.0) if signal else 0.0,
                agreement_score    = getattr(signal, "agreement_score", 0.0) if signal else 0.0,
                fsd_score          = getattr(signal, "fsd_score", 0.0) if signal else 0.0,
                embedding_distance = state.get("embedding_distance", 0.0),
            ),
        )
        save_inference(record)
    except Exception as exc:
        logger.warning("Inference record persistence failed: %s", exc)

    # ── Session store — save this turn for future context ────────────────
    if state.get("session_id"):
        try:
            from engine.session_store import store_turn
            store_turn(state["session_id"], "user",      state["prompt"])
            store_turn(state["session_id"], "assistant", state["primary_output"])
        except Exception as exc:
            logger.debug("SessionStore save failed (non-fatal): %s", exc)

    # ── Email notifications (fire-and-forget) ────────────────────────────
    try:
        from app.notifications import notify_attack_detected, notify_human_review
        tenant_id = state.get("tenant_id") or "anonymous"
        email     = state.get("tenant_email")

        if state.get("is_adversarial"):
            notify_attack_detected(
                tenant_id   = tenant_id,
                attack_type = state.get("attack_type") or "UNKNOWN",
                confidence  = state.get("attack_confidence", 0.0),
                prompt      = state["prompt"],
                model_name  = state.get("primary_model_name") or "",
                request_id  = stored_request_id,
                to          = email,
            )
        elif state.get("requires_human_review"):
            notify_human_review(
                tenant_id         = tenant_id,
                request_id        = stored_request_id,
                escalation_reason = state.get("escalation_reason", ""),
                prompt            = state["prompt"],
                model_name        = state.get("primary_model_name") or "",
                to                = email,
            )
    except Exception as exc:
        logger.debug("Notification send failed (non-fatal): %s", exc)

    trace_msg = (
        f"finalize: xgb_prob={xgb_prob:.4f if xgb_prob else 'N/A'} "
        f"xgb_failure={xgb_failure} request_id={stored_request_id}"
    )

    return {
        "failure_signal":    signal,
        "xgb_probability":   xgb_prob,
        "xgb_is_failure":    xgb_failure,
        "config_version":    config_ver,
        "failure_summary":   failure_summary,
        "request_id":        stored_request_id,
        "pipeline_trace":    _trace(state, trace_msg),
    }


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(MonitorState)

    g.add_node("load_session",     load_session)
    g.add_node("shadow_inference", shadow_inference)
    g.add_node("adversarial_guard", adversarial_guard)
    g.add_node("signal_extract",   signal_extract)
    g.add_node("jury_deliberate",  jury_deliberate)
    g.add_node("security_checks",  security_checks)
    g.add_node("gt_verify",        gt_verify)
    g.add_node("escalate",         escalate)
    g.add_node("auto_correct",     auto_correct)
    g.add_node("finalize",         finalize)

    g.set_entry_point("load_session")

    g.add_edge("load_session",     "shadow_inference")
    g.add_edge("shadow_inference", "adversarial_guard")

    g.add_conditional_edges(
        "adversarial_guard",
        _route_after_guard,
        {"END": END, "signal_extract": "signal_extract"},
    )

    g.add_edge("signal_extract",  "jury_deliberate")
    g.add_edge("jury_deliberate", "security_checks")

    g.add_conditional_edges(
        "security_checks",
        _route_to_verify,
        {"gt_verify": "gt_verify", "finalize": "finalize"},
    )

    g.add_conditional_edges(
        "gt_verify",
        _route_after_gt,
        {"escalate": "escalate", "auto_correct": "auto_correct", "finalize": "finalize"},
    )

    g.add_edge("escalate",     "finalize")
    g.add_edge("auto_correct", "finalize")
    g.add_edge("finalize",     END)

    return g.compile()


# ── Singleton compiled graph ──────────────────────────────────────────────────

_graph: Any = None


def get_pipeline():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline(initial_state: MonitorState) -> MonitorState:
    """
    Execute the full FIE monitoring pipeline via LangGraph.

    Parameters
    ----------
    initial_state : MonitorState
        Populated by the /monitor route handler before invocation.
        Must include at minimum: prompt, primary_output, run_full_jury.

    Returns
    -------
    MonitorState
        The final merged state after all nodes have executed.
        Safe to call concurrently — the graph is stateless between invocations.
    """
    # Ensure required defaults
    initial_state.setdefault("pipeline_trace", [])
    initial_state.setdefault("request_id", str(uuid.uuid4())[:12])
    initial_state.setdefault("is_adversarial", False)
    initial_state.setdefault("guard_blocked", False)
    initial_state.setdefault("requires_human_review", False)
    initial_state.setdefault("escalation_reason", "")
    initial_state.setdefault("jury_confidence", 0.0)
    initial_state.setdefault("failure_summary", "")
    initial_state.setdefault("root_cause", "NONE")
    initial_state.setdefault("model_outputs", [initial_state.get("primary_output", "")])
    initial_state.setdefault("shadow_texts", [])
    initial_state.setdefault("shadow_weights", [])
    initial_state.setdefault("canary_token", "")
    initial_state.setdefault("context", [])
    initial_state.setdefault("session_context", [])

    result = get_pipeline().invoke(initial_state)

    logger.info(
        "pipeline complete | rid=%s archetype=%s adversarial=%s xgb_prob=%s",
        result.get("request_id"),
        result.get("archetype"),
        result.get("is_adversarial"),
        f"{result.get('xgb_probability', 0):.4f}" if result.get("xgb_probability") else "N/A",
    )
    return result
