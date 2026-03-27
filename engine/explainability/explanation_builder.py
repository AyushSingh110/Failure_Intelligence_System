from __future__ import annotations

from typing import Optional

from app.schemas import (
    DiagnosticResponse,
    ExplanationAttribution,
    ExplanationBundle,
    ExplanationEvidence,
    ExplanationSignal,
    ExplanationStep,
    FixResult,
    HumanExplanation,
    JuryVerdict,
    MonitorResponse,
)
from engine.explainability.humanizer import build_human_explanation
from engine.explainability.redaction import filter_safe_evidence, sanitize_text_for_external


def _safe_round(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return round(float(value), 4)


def _build_core_signals(
    *,
    jury: Optional[JuryVerdict],
    fix_result: Optional[FixResult],
    embedding_distance: float,
    failure_signal_vector,
) -> list[ExplanationSignal]:
    signals = [
        ExplanationSignal(
            name="entropy_score",
            source="failure_signal_vector",
            value=str(failure_signal_vector.entropy_score),
            normalized_score=_safe_round(failure_signal_vector.entropy_score),
            direction="supports_failure" if failure_signal_vector.entropy_score > 0 else "neutral",
            contribution_weight=0.18,
            summary="Measures how divergent the combined model outputs are.",
        ),
        ExplanationSignal(
            name="agreement_score",
            source="failure_signal_vector",
            value=str(failure_signal_vector.agreement_score),
            normalized_score=_safe_round(1.0 - failure_signal_vector.agreement_score),
            direction="supports_failure" if failure_signal_vector.agreement_score < 0.8 else "supports_stability",
            contribution_weight=0.16,
            summary="Captures how closely the model outputs align with each other.",
        ),
        ExplanationSignal(
            name="embedding_distance",
            source="embedding_comparison",
            value=str(embedding_distance),
            normalized_score=_safe_round(embedding_distance),
            direction="supports_failure" if embedding_distance > 0.2 else "neutral",
            contribution_weight=0.12,
            summary="Represents semantic distance between the primary and reference outputs.",
        ),
    ]

    if jury is not None:
        signals.append(
            ExplanationSignal(
                name="jury_confidence",
                source="diagnostic_jury",
                value=str(jury.jury_confidence),
                normalized_score=_safe_round(jury.jury_confidence),
                direction="supports_diagnosis" if jury.primary_verdict else "neutral",
                contribution_weight=0.24,
                summary="Represents how strongly the diagnostic jury supported the selected diagnosis.",
            )
        )

    if fix_result is not None:
        signals.append(
            ExplanationSignal(
                name="fix_confidence",
                source="fix_engine",
                value=str(fix_result.fix_confidence),
                normalized_score=_safe_round(fix_result.fix_confidence),
                direction="supports_fix" if fix_result.fix_applied else "neutral",
                contribution_weight=0.16,
                summary="Measures how confident the system was in the selected mitigation strategy.",
            )
        )

    return signals


def _build_evidence(
    *,
    jury: Optional[JuryVerdict],
    fix_result: Optional[FixResult],
) -> list[ExplanationEvidence]:
    evidence: list[ExplanationEvidence] = []

    if jury and jury.primary_verdict:
        evidence.append(
            ExplanationEvidence(
                type="jury_verdict",
                source=jury.primary_verdict.agent_name,
                content_preview=jury.failure_summary,
                confidence=_safe_round(jury.primary_verdict.confidence_score),
                supports=jury.primary_verdict.root_cause,
                safe_to_expose=True,
            )
        )
        if jury.primary_verdict.evidence:
            evidence.append(
                ExplanationEvidence(
                    type="agent_evidence",
                    source=jury.primary_verdict.agent_name,
                    content_preview=str(jury.primary_verdict.evidence)[:220],
                    confidence=_safe_round(jury.primary_verdict.confidence_score),
                    supports=jury.primary_verdict.root_cause,
                    safe_to_expose=False,
                )
            )

    if fix_result is not None:
        evidence.append(
            ExplanationEvidence(
                type="fix_strategy",
                source="fix_engine",
                content_preview=fix_result.fix_explanation,
                confidence=_safe_round(fix_result.fix_confidence),
                supports=fix_result.fix_strategy,
                safe_to_expose=True,
            )
        )
        if fix_result.warning:
            evidence.append(
                ExplanationEvidence(
                    type="fix_warning",
                    source="fix_engine",
                    content_preview=fix_result.warning,
                    confidence=_safe_round(fix_result.fix_confidence),
                    supports=fix_result.fix_strategy,
                    safe_to_expose=False,
                )
            )

    return evidence


def _build_attributions(
    *,
    signals: list[ExplanationSignal],
    jury: Optional[JuryVerdict],
    fix_result: Optional[FixResult],
) -> list[ExplanationAttribution]:
    ranked = sorted(signals, key=lambda item: item.normalized_score * item.contribution_weight, reverse=True)
    top = ranked[:4]
    attributions = [
        ExplanationAttribution(
            factor=item.name,
            impact_score=_safe_round(item.normalized_score * item.contribution_weight),
            polarity=item.direction,
            details=item.summary,
        )
        for item in top
    ]

    if jury and jury.primary_verdict:
        attributions.append(
            ExplanationAttribution(
                factor="primary_verdict",
                impact_score=_safe_round(jury.primary_verdict.confidence_score),
                polarity="supports_diagnosis",
                details=f"The jury prioritized {jury.primary_verdict.root_cause} as the primary explanation.",
            )
        )

    if fix_result and fix_result.fix_applied:
        attributions.append(
            ExplanationAttribution(
                factor="selected_fix",
                impact_score=_safe_round(fix_result.fix_confidence),
                polarity="supports_fix",
                details=f"The system selected {fix_result.fix_strategy} as the safest available mitigation.",
            )
        )

    return attributions


def _build_steps(
    *,
    archetype: str,
    jury: Optional[JuryVerdict],
    fix_result: Optional[FixResult],
    failure_summary: str,
) -> list[ExplanationStep]:
    steps = [
        ExplanationStep(
            stage="signal_extraction",
            decision=f"Archetype candidate identified as {archetype}.",
            reason="Failure signals were computed from the primary and shadow model outputs.",
            confidence=0.7,
            inputs_used=["model_outputs", "failure_signal_vector"],
        )
    ]

    if jury and jury.primary_verdict:
        steps.append(
            ExplanationStep(
                stage="jury_aggregation",
                decision=f"Primary verdict selected: {jury.primary_verdict.root_cause}.",
                reason=jury.failure_summary,
                confidence=_safe_round(jury.primary_verdict.confidence_score),
                inputs_used=[verdict.agent_name for verdict in jury.verdicts],
            )
        )

    if fix_result is not None:
        steps.append(
            ExplanationStep(
                stage="fix_selection",
                decision=f"Mitigation strategy: {fix_result.fix_strategy}.",
                reason=fix_result.fix_explanation,
                confidence=_safe_round(fix_result.fix_confidence),
                inputs_used=["jury.primary_verdict", "fix_engine"],
            )
        )
    else:
        steps.append(
            ExplanationStep(
                stage="fix_selection",
                decision="No mitigation strategy applied.",
                reason=failure_summary,
                confidence=0.5,
                inputs_used=["failure_summary"],
            )
        )

    return steps


def _compute_explanation_confidence(
    *,
    jury: Optional[JuryVerdict],
    fix_result: Optional[FixResult],
    failure_signal_vector,
    evidence: list[ExplanationEvidence],
) -> tuple[float, list[str]]:
    jury_alignment = jury.jury_confidence if jury else 0.35
    evidence_strength = max((item.confidence for item in evidence), default=0.3)
    signal_consistency = (
        (failure_signal_vector.entropy_score + (1.0 - failure_signal_vector.agreement_score)) / 2.0
    )
    fix_selection_consistency = fix_result.fix_confidence if fix_result and fix_result.fix_applied else 0.45
    source_reliability = 0.8 if any(item.type == "fix_strategy" for item in evidence) else 0.55

    uncertainty_notes: list[str] = []
    penalty = 0.0

    if jury is None or jury.primary_verdict is None:
        penalty += 0.12
        uncertainty_notes.append("No jury verdict was available, so the explanation relies mostly on surface signals.")
    elif len([v for v in jury.verdicts if not v.skipped]) > 1 and jury.jury_confidence < 0.5:
        penalty += 0.10
        uncertainty_notes.append("Multiple active agents contributed with only moderate agreement.")

    if fix_result is None:
        penalty += 0.08
        uncertainty_notes.append("No corrective strategy was selected, so the explanation is diagnostic-only.")
    elif not fix_result.fix_applied:
        penalty += 0.06
        uncertainty_notes.append("A fix was considered but not confidently applied.")

    score = (
        0.25 * jury_alignment
        + 0.20 * evidence_strength
        + 0.20 * signal_consistency
        + 0.20 * fix_selection_consistency
        + 0.15 * source_reliability
        - penalty
    )
    return max(0.0, min(round(score, 4), 1.0)), uncertainty_notes


def build_explanation_bundle(
    *,
    request_id: Optional[str],
    archetype: str,
    embedding_distance: float,
    failure_signal_vector,
    jury: Optional[JuryVerdict],
    fix_result: Optional[FixResult],
    failure_summary: str,
    mode: str,
) -> ExplanationBundle:
    """Build a structured explanation object without altering existing backend flow."""
    signals = _build_core_signals(
        jury=jury,
        fix_result=fix_result,
        embedding_distance=embedding_distance,
        failure_signal_vector=failure_signal_vector,
    )
    evidence = _build_evidence(jury=jury, fix_result=fix_result)
    attributions = _build_attributions(signals=signals, jury=jury, fix_result=fix_result)
    decision_trace = _build_steps(
        archetype=archetype,
        jury=jury,
        fix_result=fix_result,
        failure_summary=failure_summary,
    )
    explanation_confidence, uncertainty_notes = _compute_explanation_confidence(
        jury=jury,
        fix_result=fix_result,
        failure_signal_vector=failure_signal_vector,
        evidence=evidence,
    )

    final_label = jury.primary_verdict.root_cause if jury and jury.primary_verdict else archetype
    final_fix_strategy = fix_result.fix_strategy if fix_result else "NO_FIX"
    summary = failure_summary

    if mode == "external":
        summary = sanitize_text_for_external(summary)
        evidence = [
            ExplanationEvidence(**item)
            for item in filter_safe_evidence([entry.model_dump() for entry in evidence])
        ]
        uncertainty_notes = [sanitize_text_for_external(note) for note in uncertainty_notes]

    return ExplanationBundle(
        explanation_version="v1",
        mode=mode,
        request_id=request_id,
        final_label=final_label,
        final_fix_strategy=final_fix_strategy,
        explanation_confidence=explanation_confidence,
        summary=summary,
        decision_trace=decision_trace,
        signals=signals,
        evidence=evidence,
        attributions=attributions,
        alternatives_considered=_collect_alternatives(jury),
        uncertainty_notes=uncertainty_notes,
        internal_only=(mode == "internal"),
    )


def _collect_alternatives(jury: Optional[JuryVerdict]) -> list[str]:
    if jury is None:
        return []
    primary = jury.primary_verdict.root_cause if jury.primary_verdict else None
    alternatives = [
        verdict.root_cause
        for verdict in jury.verdicts
        if not verdict.skipped and verdict.root_cause != primary
    ]
    return alternatives[:4]


def attach_explanations_to_diagnostic(response: DiagnosticResponse) -> DiagnosticResponse:
    response.explanation_internal = build_explanation_bundle(
        request_id=None,
        archetype=response.archetype,
        embedding_distance=response.embedding_distance,
        failure_signal_vector=response.failure_signal_vector,
        jury=response.jury,
        fix_result=None,
        failure_summary=response.jury.failure_summary,
        mode="internal",
    )
    response.explanation_external = build_explanation_bundle(
        request_id=None,
        archetype=response.archetype,
        embedding_distance=response.embedding_distance,
        failure_signal_vector=response.failure_signal_vector,
        jury=response.jury,
        fix_result=None,
        failure_summary=response.jury.failure_summary,
        mode="external",
    )
    response.human_explanation = build_human_explanation(response.explanation_external)
    return response


def attach_explanations_to_monitor(
    response: MonitorResponse,
    *,
    request_id: Optional[str],
) -> MonitorResponse:
    response.explanation_internal = build_explanation_bundle(
        request_id=request_id,
        archetype=response.archetype,
        embedding_distance=response.embedding_distance,
        failure_signal_vector=response.failure_signal_vector,
        jury=response.jury,
        fix_result=response.fix_result,
        failure_summary=response.failure_summary,
        mode="internal",
    )
    response.explanation_external = build_explanation_bundle(
        request_id=request_id,
        archetype=response.archetype,
        embedding_distance=response.embedding_distance,
        failure_signal_vector=response.failure_signal_vector,
        jury=response.jury,
        fix_result=response.fix_result,
        failure_summary=response.failure_summary,
        mode="external",
    )
    response.human_explanation = build_human_explanation(response.explanation_external)
    return response
