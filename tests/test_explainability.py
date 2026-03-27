from app.schemas import (
    AgentVerdict,
    DiagnosticResponse,
    ExplanationBundle,
    FailureSignalVector,
    FixResult,
    JuryVerdict,
    MonitorResponse,
)
from engine.explainability.explanation_builder import (
    attach_explanations_to_diagnostic,
    attach_explanations_to_monitor,
)


def _build_signal() -> FailureSignalVector:
    return FailureSignalVector(
        agreement_score=0.35,
        fsd_score=0.62,
        answer_counts={"answer a": 1, "answer b": 2},
        entropy_score=0.58,
        ensemble_disagreement=True,
        ensemble_similarity=0.41,
        high_failure_risk=True,
    )


def _build_jury() -> JuryVerdict:
    primary = AgentVerdict(
        agent_name="AdversarialSpecialist",
        root_cause="PROMPT_INJECTION",
        confidence_score=0.84,
        mitigation_strategy="Sanitize input.",
        evidence={"detection_layers_fired": ["prompt_guard", "faiss"]},
    )
    return JuryVerdict(
        verdicts=[
            primary,
            AgentVerdict(
                agent_name="DomainCritic",
                root_cause="FACTUAL_HALLUCINATION",
                confidence_score=0.42,
                mitigation_strategy="Use grounding.",
            ),
        ],
        primary_verdict=primary,
        jury_confidence=0.63,
        is_adversarial=True,
        failure_summary="Adversarial attack detected with high confidence.",
    )


def test_attach_explanations_to_diagnostic_adds_internal_and_external_views():
    response = DiagnosticResponse(
        failure_signal_vector=_build_signal(),
        archetype="HALLUCINATION_RISK",
        embedding_distance=0.44,
        jury=_build_jury(),
    )

    enriched = attach_explanations_to_diagnostic(response)

    assert isinstance(enriched.explanation_internal, ExplanationBundle)
    assert isinstance(enriched.explanation_external, ExplanationBundle)
    assert enriched.explanation_internal.mode == "internal"
    assert enriched.explanation_external.mode == "external"
    assert enriched.explanation_internal.final_label == "PROMPT_INJECTION"
    assert enriched.explanation_external.internal_only is False
    assert enriched.human_explanation is not None
    assert enriched.human_explanation.summary


def test_attach_explanations_to_monitor_keeps_fix_context():
    response = MonitorResponse(
        shadow_model_results=[],
        all_model_outputs=["bad answer", "safe answer"],
        ollama_available=True,
        failure_signal_vector=_build_signal(),
        archetype="HALLUCINATION_RISK",
        embedding_distance=0.44,
        jury=_build_jury(),
        high_failure_risk=True,
        failure_summary="AUTO-FIXED: SANITIZE_AND_RERUN applied.",
        fix_result=FixResult(
            fixed_output="I can help you with legitimate questions and tasks.",
            fix_applied=True,
            fix_strategy="SANITIZE_AND_RERUN",
            fix_explanation="Adversarial content was removed before retrying.",
            original_output="unsafe answer",
            root_cause="PROMPT_INJECTION",
            fix_confidence=0.81,
            improvement_score=0.78,
            warning="Grounded source: internal safety handling",
        ),
    )

    enriched = attach_explanations_to_monitor(response, request_id="req-123")

    assert enriched.explanation_internal is not None
    assert enriched.explanation_external is not None
    assert enriched.explanation_internal.request_id == "req-123"
    assert enriched.explanation_internal.final_fix_strategy == "SANITIZE_AND_RERUN"
    assert enriched.explanation_external.evidence
    assert all(item.safe_to_expose for item in enriched.explanation_external.evidence)
    assert enriched.human_explanation is not None
    assert enriched.human_explanation.recommended_action
