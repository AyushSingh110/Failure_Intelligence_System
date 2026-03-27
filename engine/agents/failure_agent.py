#failure_agent.py
from __future__ import annotations

import logging
from typing import Optional

from app.schemas import (
    AgentVerdict,
    FailureSignalVector,
    ClusterAssignment,
    LabelResult,
    JuryVerdict,
    DiagnosticRequest,
    DiagnosticResponse,
)
from engine.detector.consistency import compute_consistency
from engine.detector.entropy import compute_entropy, compute_entropy_from_counts
from engine.detector.ensemble import compute_disagreement
from engine.detector.embedding import compute_embedding_distance
from engine.archetypes.labeling import (
    label_failure_archetype,
    label_failure_archetype_detailed,
)
from engine.archetypes.clustering import archetype_registry
from engine.evolution.tracker import evolution_tracker
from engine.agents.base_agent import BaseJuryAgent, DiagnosticContext
from engine.agents.linguistic_auditor import linguistic_auditor
from engine.agents.adversarial_specialist import adversarial_specialist
from engine.agents.domain_critic import domain_critic
from config import get_settings
from engine.explainability.explanation_builder import attach_explanations_to_diagnostic

logger   = logging.getLogger(__name__)
settings = get_settings()

_ADVERSARIAL_ROOTS = frozenset({
    "PROMPT_INJECTION",
    "JAILBREAK_ATTEMPT",
    "INSTRUCTION_OVERRIDE",
    "TOKEN_SMUGGLING",
    "INTENTIONAL_PROMPT_ATTACK",
})

_TEMPORAL_ROOTS = frozenset({
    "TEMPORAL_KNOWLEDGE_CUTOFF",
})


# ── DiagnosticJury ─────────────────────────────────────────────────────────

class DiagnosticJury:

    def __init__(self) -> None:
        self._agents: list[BaseJuryAgent] = [
            adversarial_specialist,
            linguistic_auditor,
            domain_critic,
        ]

    def deliberate(self, context: DiagnosticContext) -> JuryVerdict:
        verdicts: list[AgentVerdict] = []

        for agent in self._agents:
            try:
                verdict = agent.analyze(context)
                verdicts.append(verdict)
            except Exception as exc:
                logger.error(
                    "Agent %s raised an unexpected exception: %s",
                    agent.agent_name, exc, exc_info=True,
                )
                verdicts.append(AgentVerdict(
                    agent_name=agent.agent_name,
                    root_cause="AGENT_ERROR",
                    confidence_score=0.0,
                    mitigation_strategy="",
                    skipped=True,
                    skip_reason=f"Agent raised exception: {exc}",
                ))

        return self._aggregate(verdicts)

    def _aggregate(self, verdicts: list[AgentVerdict]) -> JuryVerdict:
        active  = [v for v in verdicts if not v.skipped]
        skipped = [v for v in verdicts if v.skipped]

        jury_confidence = (
            round(sum(v.confidence_score for v in active) / len(active), 4)
            if active else 0.0
        )

        is_adversarial = any(v.root_cause in _ADVERSARIAL_ROOTS for v in active)
        is_temporal    = any(v.root_cause in _TEMPORAL_ROOTS for v in active)
        is_complex     = any(v.root_cause == "PROMPT_COMPLEXITY_OOD" for v in active)

        if is_adversarial:
            primary = max(
                (v for v in active if v.root_cause in _ADVERSARIAL_ROOTS),
                key=lambda v: v.confidence_score,
            )
        elif is_temporal:
            primary = max(
                (v for v in active if v.root_cause in _TEMPORAL_ROOTS),
                key=lambda v: v.confidence_score,
            )
        else:
            primary = (
                max(active, key=lambda v: v.confidence_score)
                if active else None
            )

        failure_summary = self._build_summary(primary, active, is_adversarial, is_complex)

        logger.debug(
            "Jury deliberation complete | active=%d skipped=%d primary=%s confidence=%.3f",
            len(active), len(skipped),
            primary.root_cause if primary else "NONE",
            jury_confidence,
        )

        return JuryVerdict(
            verdicts=verdicts,
            primary_verdict=primary,
            jury_confidence=jury_confidence,
            is_adversarial=is_adversarial,
            is_complex_prompt=is_complex,
            failure_summary=failure_summary,
        )

    @staticmethod
    def _build_summary(
        primary:        Optional[AgentVerdict],
        active:         list[AgentVerdict],
        is_adversarial: bool,
        is_complex:     bool,
    ) -> str:
        if not active:
            return "No diagnostic agents reached a conclusion. Manual review recommended."
        if not primary:
            return "All agents skipped. Failure cause undetermined."

        conf_pct = int(primary.confidence_score * 100)

        if is_adversarial:
            return (
                f"Adversarial attack detected ({primary.root_cause}) "
                f"with {conf_pct}% confidence. "
                f"{primary.mitigation_strategy[:100]}"
            )

        if is_complex:
            n_dims = len(primary.evidence.get("dimensions_fired", [])) if primary.evidence else 0
            return (
                f"Prompt complexity is the likely failure cause ({n_dims} complexity "
                f"signal(s) detected, confidence {conf_pct}%). "
                f"Recommend decomposing into simpler sub-prompts."
            )

        if len(active) > 1:
            causes = ", ".join(v.root_cause for v in active)
            return (
                f"Multiple potential causes identified: {causes}. "
                f"Primary diagnosis: {primary.root_cause} ({conf_pct}% confidence)."
            )

        return (
            f"Diagnosis: {primary.root_cause} "
            f"(confidence {conf_pct}%, agent: {primary.agent_name})."
        )


# ── FailureAgent ───────────────────────────────────────────────────────────

class FailureAgent:

    def __init__(self) -> None:
        self._jury = DiagnosticJury()

    # ── Phase 1 ────────────────────────────────────────────────────────

    def run(
        self,
        model_outputs: list[str],
        primary_output: Optional[str] = None,
        secondary_output: Optional[str] = None,
    ) -> dict:
        """Phase 1: extract signal, return signal + archetype label."""
        primary_output = primary_output or model_outputs[0]
        secondary_output = secondary_output or (
            model_outputs[1] if len(model_outputs) > 1 else model_outputs[0]
        )

        signal    = self._build_signal(model_outputs)
        archetype = label_failure_archetype(signal)
        embedding = compute_embedding_distance(primary_output, secondary_output)

        return {
            "failure_signal_vector": signal.model_dump(),
            "archetype":             archetype,
            "embedding_distance":    embedding["embedding_distance"],
        }

    # ── Phase 2 ────────────────────────────────────────────────────────

    def run_full(
        self,
        model_outputs: list[str],
        primary_output: Optional[str] = None,
        secondary_output: Optional[str] = None,
    ) -> dict:
        """Phase 2: extract signal, assign to cluster, update tracker."""
        primary_output = primary_output or model_outputs[0]
        secondary_output = secondary_output or (
            model_outputs[1] if len(model_outputs) > 1 else model_outputs[0]
        )

        signal    = self._build_signal(model_outputs)
        embedding = compute_embedding_distance(primary_output, secondary_output)

        assignment:   ClusterAssignment = archetype_registry.assign(signal)
        label_detail: LabelResult       = label_failure_archetype_detailed(signal)
        evolution_tracker.record(signal)
        trend = evolution_tracker.trend_summary()

        return {
            "failure_signal_vector": signal.model_dump(),
            "cluster_assignment":    dict(assignment),
            "label_detail":          dict(label_detail),
            "embedding_distance":    embedding["embedding_distance"],
            "trend_summary":         trend,
        }

    # ── Phase 3 ────────────────────────────────────────────────────────

    def run_diagnostic(self, request: DiagnosticRequest) -> DiagnosticResponse:
        """Phase 3: full Phase 1 + Phase 2 + DiagnosticJury reasoning."""
        # Derive primary and secondary from model_outputs list.
        # model_outputs[0] = primary (model under test)
        # model_outputs[1] = secondary/reference model (if present)
        primary_output   = request.model_outputs[0]
        secondary_output = (
            request.model_outputs[1]
            if len(request.model_outputs) > 1
            else request.model_outputs[0]
        )

        # ── Phase 1 ────────────────────────────────────────────────────
        signal    = self._build_signal(request.model_outputs)
        archetype = label_failure_archetype(signal)
        embedding = compute_embedding_distance(primary_output, secondary_output)

        # ── Phase 2 ────────────────────────────────────────────────────
        archetype_registry.assign(signal)
        evolution_tracker.record(signal)

        # ── Phase 3 ────────────────────────────────────────────────────
        context = DiagnosticContext.build(
            prompt=request.prompt,
            primary_output=primary_output,
            secondary_output=secondary_output,
            model_outputs=request.model_outputs,
            fsv=signal,
            latency_ms=request.latency_ms,
        )
        jury_verdict = self._jury.deliberate(context)

        response = DiagnosticResponse(
            failure_signal_vector=signal,
            archetype=archetype,
            embedding_distance=embedding["embedding_distance"],
            jury=jury_verdict,
        )
        return attach_explanations_to_diagnostic(response)

    # ── Shared signal builder ──────────────────────────────────────────

    def _build_signal(self, model_outputs: list[str]) -> FailureSignalVector:
        """
        Builds a FailureSignalVector from all provided model outputs.

        All 3 detectors receive the full list:
          - consistency + entropy : use the full list for clustering
          - ensemble              : compares ALL pairwise combinations
          - embedding             : compares outputs[0] vs outputs[1]
        """
        consistency   = compute_consistency(model_outputs)
        # Use compute_entropy_from_counts to guarantee entropy and consistency
        # always use the exact same semantic clusters — prevents the case where
        # they compute clusters independently and get different results.
        entropy_score = compute_entropy_from_counts(
            consistency["answer_counts"],
            len(model_outputs),
        )
        ensemble      = compute_disagreement(model_outputs)   # ← list, not two strings

        # Guard: never set high_failure_risk from ensemble alone when entropy=0.0.
        # entropy=0.0 means all outputs are semantically consistent — the model is stable.
        # Without this guard, short/long paraphrase pairs ("Paris" vs "The capital of
        # France is Paris.") can trigger ensemble_disagreement=True even after the
        # _pair_similarity fix, which would cascade to OVERCONFIDENT_FAILURE.
        ensemble_fires = (
            ensemble["disagreement"] is True
            and entropy_score > 0.0
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


# ── Singletons ─────────────────────────────────────────────────────────────
failure_agent   = FailureAgent()
diagnostic_jury = failure_agent._jury
