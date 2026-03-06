from app.schemas import FailureSignalVector, ClusterAssignment, LabelResult
from engine.detector.consistency import compute_consistency
from engine.detector.entropy import compute_entropy
from engine.detector.ensemble import compute_disagreement
from engine.detector.embedding import compute_embedding_distance
from engine.archetypes.labeling import (
    label_failure_archetype,
    label_failure_archetype_detailed,
)
from engine.archetypes.clustering import archetype_registry
from engine.evolution.tracker import evolution_tracker
from config import get_settings

settings = get_settings()


class FailureAgent:

    def run(
        self,
        model_outputs:    list[str],
        primary_output:   str,
        secondary_output: str,
    ) -> dict:
        """
        Phase 1 path: extract signal, return signal + archetype label.
        Does NOT update the registry or tracker (use run_full for that).
        """
        signal    = self._build_signal(model_outputs, primary_output, secondary_output)
        archetype = label_failure_archetype(signal)
        embedding = compute_embedding_distance(primary_output, secondary_output)

        return {
            "failure_signal_vector": signal.model_dump(),
            "archetype":             archetype,
            "embedding_distance":    embedding["embedding_distance"],
        }

    def run_full(
        self,
        model_outputs:    list[str],
        primary_output:   str,
        secondary_output: str,
    ) -> dict:
        """
        Phase 2 path: extract signal, assign to cluster, update tracker.
        Returns full diagnostic output including cluster assignment and trend.
        """
        signal    = self._build_signal(model_outputs, primary_output, secondary_output)
        embedding = compute_embedding_distance(primary_output, secondary_output)

        # Phase 2: cluster assignment
        assignment: ClusterAssignment = archetype_registry.assign(signal)   # type: ignore[assignment]
        cluster_dict = dict(assignment)

        # Phase 2: label with conditions
        label_detail: LabelResult = label_failure_archetype_detailed(signal)  # type: ignore[assignment]
        label_dict = dict(label_detail)

        # Phase 2: update evolution tracker
        evolution_tracker.record(signal)
        trend = evolution_tracker.trend_summary()

        return {
            "failure_signal_vector": signal.model_dump(),
            "cluster_assignment":    cluster_dict,
            "label_detail":          label_dict,
            "embedding_distance":    embedding["embedding_distance"],
            "trend_summary":         trend,
        }

    def _build_signal(
        self,
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


failure_agent = FailureAgent()
