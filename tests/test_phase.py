import sys
import os
import math

# Path setup — run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# ── Shared fixture ────────────────────────────────────────────────────────

def _make_signal(
    agreement:    float = 0.9,
    fsd:          float = 0.8,
    entropy:      float = 0.1,
    similarity:   float = 0.9,
    disagreement: bool  = False,
    risk:         bool  = False,
) :
    from app.schemas import FailureSignalVector
    return FailureSignalVector(
        agreement_score=agreement,
        fsd_score=fsd,
        answer_counts={"paris": 3},
        entropy_score=entropy,
        ensemble_disagreement=disagreement,
        ensemble_similarity=similarity,
        high_failure_risk=risk,
    )


# ═══════════════════════════════════════════════════════════════
# PHASE 1: DETECTOR TESTS
# ═══════════════════════════════════════════════════════════════

class TestConsistency:

    def test_basic_agreement(self):
        from engine.detector.consistency import compute_consistency
        result = compute_consistency(["Paris", "Paris", "Paris", "London"])
        assert result["agreement_score"] == 0.75, \
            f"Expected 0.75, got {result['agreement_score']}"
        assert result["fsd_score"] == 0.5, \
            f"Expected FSD=0.5, got {result['fsd_score']}"

    def test_llm_prefix_stripping(self):
        """
        Critical test: LLM prefixes must not inflate disagreement.
        "The answer is Paris" and "Paris" are the SAME answer.
        """
        from engine.detector.consistency import compute_consistency
        outputs = [
            "The answer is Paris",
            "Paris",
            "Therefore, Paris",
            "Result: Paris",
        ]
        result = compute_consistency(outputs)
        assert result["agreement_score"] == 1.0, (
            f"All 4 outputs mean 'Paris' — agreement should be 1.0, "
            f"got {result['agreement_score']}.\n"
            f"answer_counts: {result['answer_counts']}\n"
            "FIX: Check _normalize() regex is stripping all prefixes correctly."
        )

    def test_chained_prefix_stripping(self):
        """'Therefore, the answer is Paris' needs two-pass stripping."""
        from engine.detector.consistency import compute_consistency
        result = compute_consistency([
            "Therefore, the answer is Paris",
            "Paris",
        ])
        assert result["agreement_score"] == 1.0, (
            f"Chained prefix not stripped, got {result['answer_counts']}"
        )

    def test_all_different(self):
        from engine.detector.consistency import compute_consistency
        result = compute_consistency(["Paris", "London", "Berlin", "Rome"])
        assert result["agreement_score"] == 0.25
        assert result["fsd_score"] == 0.0

    def test_single_output(self):
        from engine.detector.consistency import compute_consistency
        result = compute_consistency(["Paris"])
        assert result["agreement_score"] == 1.0
        assert result["fsd_score"] == 0.0

    def test_empty_outputs(self):
        from engine.detector.consistency import compute_consistency
        result = compute_consistency([])
        assert result["agreement_score"] == 0.0
        assert result["answer_counts"] == {}


class TestEntropy:

    def test_identical_outputs_zero_entropy(self):
        from engine.detector.entropy import compute_entropy
        score = compute_entropy(["Paris", "Paris", "Paris"])
        assert score == 0.0, f"Identical outputs → entropy must be 0.0, got {score}"

    def test_all_different_max_entropy(self):
        from engine.detector.entropy import compute_entropy
        score = compute_entropy(["Paris", "London", "Berlin", "Rome"])
        assert score == 1.0, f"All different → entropy must be 1.0, got {score}"

    def test_partial_entropy_between_0_and_1(self):
        from engine.detector.entropy import compute_entropy
        score = compute_entropy(["Paris", "Paris", "London"])
        assert 0.0 < score < 1.0, f"Expected 0 < score < 1, got {score}"

    def test_single_output_zero_entropy(self):
        from engine.detector.entropy import compute_entropy
        assert compute_entropy(["Paris"]) == 0.0

    def test_empty_zero_entropy(self):
        from engine.detector.entropy import compute_entropy
        assert compute_entropy([]) == 0.0


class TestEnsemble:

    def test_disagreement_detected(self):
        from engine.detector.ensemble import compute_disagreement
        result = compute_disagreement(
            "The capital of France is Paris",
            "The capital of France is Lyon",
        )
        assert result["disagreement"] is True, (
            f"Different answers should trigger disagreement. "
            f"similarity={result['similarity_score']}"
        )

    def test_agreement_detected(self):
        from engine.detector.ensemble import compute_disagreement
        result = compute_disagreement("Paris", "Paris")
        assert result["disagreement"] is False
        assert result["similarity_score"] == 1.0

    def test_empty_primary_is_disagreement(self):
        from engine.detector.ensemble import compute_disagreement
        result = compute_disagreement("", "Paris")
        assert result["disagreement"] is True
        assert result["similarity_score"] == 0.0

    def test_both_empty_no_disagreement(self):
        from engine.detector.ensemble import compute_disagreement
        result = compute_disagreement("", "")
        assert result["disagreement"] is False


# ═══════════════════════════════════════════════════════════════
# PHASE 2: SIMILARITY TESTS
# ═══════════════════════════════════════════════════════════════

class TestSimilarity:

    def test_identical_signals_score_1(self):
        from engine.archetypes.similarity import compute_signal_similarity
        signal = _make_signal()
        score  = compute_signal_similarity(signal, signal)
        assert score == 1.0, f"Identical signals must score 1.0, got {score}"

    def test_opposite_signals_score_low(self):
        from engine.archetypes.similarity import compute_signal_similarity
        stable  = _make_signal(agreement=1.0, entropy=0.0, disagreement=False, risk=False)
        failing = _make_signal(agreement=0.0, entropy=1.0, disagreement=True,  risk=True)
        score   = compute_signal_similarity(stable, failing)
        assert score < 0.30, (
            f"Maximally opposite signals should score < 0.30, got {score}"
        )

    def test_weighted_disagreement_dominates(self):
        """
        Two signals identical except one has ensemble_disagreement=True.
        The disagreement feature has the highest weight (3.0) so it should
        cause a significant similarity drop compared to a low-weight difference.
        """
        from engine.archetypes.similarity import compute_signal_similarity
        base      = _make_signal(agreement=0.9, entropy=0.1, disagreement=False, risk=False)
        with_dis  = _make_signal(agreement=0.9, entropy=0.1, disagreement=True,  risk=False)
        only_lat  = _make_signal(agreement=0.9, entropy=0.1, disagreement=False, risk=False, similarity=0.5)
        score_dis = compute_signal_similarity(base, with_dis)
        score_lat = compute_signal_similarity(base, only_lat)
        assert score_dis < score_lat, (
            f"High-weight disagreement should reduce similarity MORE than "
            f"low-weight similarity diff. "
            f"score_dis={score_dis}, score_lat={score_lat}"
        )

    def test_weighted_distance_dict_api(self):
        from engine.archetypes.similarity import weighted_distance
        a = {"ensemble_disagreement": 1.0, "high_failure_risk": 1.0,
             "entropy_score": 0.9, "fsd_score": 0.1, "agreement_score": 0.1,
             "ensemble_similarity": 0.2, "latency_ms_norm": 0.0}
        b = {"ensemble_disagreement": 0.0, "high_failure_risk": 0.0,
             "entropy_score": 0.1, "fsd_score": 0.9, "agreement_score": 0.9,
             "ensemble_similarity": 0.9, "latency_ms_norm": 0.0}
        dist = weighted_distance(a, b)
        assert 0.0 <= dist <= 1.0
        assert dist > 0.5, f"Very different signals should have high distance, got {dist}"


# ═══════════════════════════════════════════════════════════════
# PHASE 2: LABELING TESTS
# ═══════════════════════════════════════════════════════════════

class TestLabeling:

    def test_hallucination_risk(self):
        from engine.archetypes.labeling import label_failure_archetype
        signal = _make_signal(entropy=0.9, disagreement=True, risk=True)
        assert label_failure_archetype(signal) == "HALLUCINATION_RISK"

    def test_overconfident_failure(self):
        """Low entropy + high risk = confident but wrong."""
        from engine.archetypes.labeling import label_failure_archetype
        signal = _make_signal(entropy=0.1, disagreement=False, risk=True)
        assert label_failure_archetype(signal) == "OVERCONFIDENT_FAILURE", (
            "Low entropy + high_failure_risk should be OVERCONFIDENT_FAILURE"
        )

    def test_blind_spot(self):
        from engine.archetypes.labeling import label_failure_archetype
        signal = _make_signal(entropy=0.3, disagreement=True, risk=False)
        assert label_failure_archetype(signal) == "MODEL_BLIND_SPOT"

    def test_unstable_output(self):
        from engine.archetypes.labeling import label_failure_archetype
        signal = _make_signal(entropy=0.9, disagreement=False, risk=False)
        assert label_failure_archetype(signal) == "UNSTABLE_OUTPUT"

    def test_low_confidence(self):
        from engine.archetypes.labeling import label_failure_archetype
        signal = _make_signal(agreement=0.3, entropy=0.2, disagreement=False, risk=False)
        assert label_failure_archetype(signal) == "LOW_CONFIDENCE"

    def test_stable(self):
        from engine.archetypes.labeling import label_failure_archetype
        signal = _make_signal(agreement=0.9, entropy=0.1, disagreement=False, risk=False)
        assert label_failure_archetype(signal) == "STABLE"

    def test_assign_failure_label_dict_api(self):
        from engine.archetypes.labeling import assign_failure_label
        label = assign_failure_label({
            "entropy_score": 0.9,
            "agreement_score": 0.2,
            "ensemble_disagreement": True,
            "high_failure_risk": True,
        })
        assert label == "HALLUCINATION_RISK"

    def test_detailed_label_has_conditions(self):
        from engine.archetypes.labeling import label_failure_archetype_detailed
        signal = _make_signal(entropy=0.9, disagreement=True, risk=True)
        result = label_failure_archetype_detailed(signal)
        assert result["archetype"] == "HALLUCINATION_RISK"
        assert len(result["conditions_met"]) >= 2
        assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")


# ═══════════════════════════════════════════════════════════════
# PHASE 2: CLUSTERING TESTS
# ═══════════════════════════════════════════════════════════════

class TestClustering:

    def _fresh_registry(self):
        from engine.archetypes.clustering import ArchetypeClusterRegistry
        return ArchetypeClusterRegistry()

    def test_first_signal_is_novel(self):
        registry   = self._fresh_registry()
        signal     = _make_signal()
        assignment = registry.assign(signal)
        assert assignment["status"] == "NOVEL_ANOMALY", (
            "First signal must be NOVEL_ANOMALY — no existing clusters to match"
        )
        assert registry.cluster_count() == 1

    def test_identical_signal_merges(self):
        registry   = self._fresh_registry()
        signal     = _make_signal()
        registry.assign(signal)
        assignment = registry.assign(signal)   # same signal again
        assert assignment["status"] == "KNOWN_FAILURE", (
            f"Identical signal must merge into existing cluster. "
            f"Got: {assignment['status']}, similarity={assignment['similarity_score']}"
        )
        assert registry.cluster_count() == 1, "No new cluster should have been created"

    def test_alien_signal_is_novel_anomaly(self):
        registry = self._fresh_registry()
        stable   = _make_signal(agreement=1.0, entropy=0.0, disagreement=False, risk=False)
        failing  = _make_signal(agreement=0.0, entropy=1.0, disagreement=True,  risk=True)
        registry.assign(stable)
        assignment = registry.assign(failing)
        assert assignment["status"] == "NOVEL_ANOMALY", (
            f"Maximally different signal should be NOVEL_ANOMALY. "
            f"Got: {assignment['status']}, similarity={assignment['similarity_score']}"
        )
        assert registry.cluster_count() == 2

    def test_adaptive_threshold_grows(self):
        from engine.archetypes.clustering import _adaptive_threshold
        t_at_1  = _adaptive_threshold(1)
        t_at_10 = _adaptive_threshold(10)
        t_at_50 = _adaptive_threshold(50)
        assert t_at_1 < t_at_10 < t_at_50, (
            f"Threshold must grow with cluster count: "
            f"n=1→{t_at_1}, n=10→{t_at_10}, n=50→{t_at_50}"
        )

    def test_cluster_output_structure(self):
        registry = self._fresh_registry()
        registry.assign(_make_signal())
        summary  = registry.summarize()
        assert len(summary) == 1
        cluster = summary[0]
        assert "cluster_id"       in cluster
        assert "archetype"        in cluster
        assert "cluster_size"     in cluster
        assert "is_novel_anomaly" in cluster
        assert "centroid"         in cluster

    def test_novel_anomaly_promoted_on_second_member(self):
        """A novel anomaly becomes a confirmed archetype when a second signal joins."""
        registry = self._fresh_registry()
        signal   = _make_signal()
        registry.assign(signal)
        registry.assign(signal)   # second identical signal
        cluster = registry.get_all_clusters()[0]
        assert cluster.is_novel is False, (
            "After a second member joins, cluster should no longer be novel"
        )


# ═══════════════════════════════════════════════════════════════
# PHASE 2: TRACKER TESTS
# ═══════════════════════════════════════════════════════════════

class TestTracker:

    def _fresh_tracker(self):
        from engine.evolution.tracker import SignalEvolutionTracker
        return SignalEvolutionTracker(decay_alpha=0.5)   # fast decay for test visibility

    def test_ema_updates_on_record(self):
        tracker = self._fresh_tracker()
        assert tracker.high_risk_rate() == 0.0
        tracker.record(_make_signal(risk=True))
        assert tracker.high_risk_rate() > 0.0

    def test_high_risk_rate_spikes_on_recent_failures(self):
        """
        With alpha=0.5, a burst of 3 failures at the end should give
        a higher high_risk_rate than 3 failures spread over 10 signals.
        """
        tracker_recent  = self._fresh_tracker()
        tracker_spread  = self._fresh_tracker()

        stable  = _make_signal(risk=False)
        failing = _make_signal(risk=True)

        # Spread: failure, 3 stable, failure, 3 stable, failure
        for _ in range(3):
            tracker_spread.record(failing)
            tracker_spread.record(stable)
            tracker_spread.record(stable)

        # Recent burst: 7 stable, then 3 failures
        for _ in range(7):
            tracker_recent.record(stable)
        for _ in range(3):
            tracker_recent.record(failing)

        assert tracker_recent.high_risk_rate() > tracker_spread.high_risk_rate(), (
            f"Recent burst should give higher risk rate than spread failures. "
            f"recent={tracker_recent.high_risk_rate()}, "
            f"spread={tracker_spread.high_risk_rate()}"
        )

    def test_degradation_velocity_positive(self):
        """Rising failure rate → positive velocity."""
        from engine.evolution.tracker import SignalEvolutionTracker
        tracker = SignalEvolutionTracker(decay_alpha=0.5)
        stable  = _make_signal(risk=False)
        failing = _make_signal(risk=True)
        for _ in range(5):
            tracker.record(stable)
        for _ in range(5):
            tracker.record(failing)
        velocity = tracker.degradation_velocity()
        assert velocity > 0.0, (
            f"Failures in second half → velocity should be positive, got {velocity}"
        )

    def test_degradation_velocity_negative(self):
        """Falling failure rate → negative velocity (recovery)."""
        from engine.evolution.tracker import SignalEvolutionTracker
        tracker = SignalEvolutionTracker(decay_alpha=0.5)
        stable  = _make_signal(risk=False)
        failing = _make_signal(risk=True)
        for _ in range(5):
            tracker.record(failing)
        for _ in range(5):
            tracker.record(stable)
        velocity = tracker.degradation_velocity()
        assert velocity < 0.0, (
            f"Recovery in second half → velocity should be negative, got {velocity}"
        )

    def test_is_degrading_flag(self):
        from engine.evolution.tracker import SignalEvolutionTracker
        tracker = SignalEvolutionTracker(decay_alpha=0.9)
        failing = _make_signal(risk=True)
        for _ in range(20):
            tracker.record(failing)
        assert tracker.is_degrading() is True, (
            "20 consecutive failures should trigger is_degrading=True"
        )

    def test_trend_summary_keys(self):
        tracker  = self._fresh_tracker()
        tracker.record(_make_signal())
        summary  = tracker.trend_summary()
        required = {
            "signals_recorded", "decay_alpha", "ema_entropy",
            "ema_agreement", "ema_disagreement_rate",
            "ema_high_risk_rate", "degradation_velocity", "is_degrading",
        }
        missing = required - set(summary.keys())
        assert not missing, f"trend_summary() missing keys: {missing}"

    def test_external_history_velocity(self):
        """degradation_velocity() accepts an external list for batch/testing use."""
        from engine.evolution.tracker import SignalEvolutionTracker
        tracker  = SignalEvolutionTracker()
        history  = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
        velocity = tracker.degradation_velocity(history=history)
        assert velocity > 0.0


# ═══════════════════════════════════════════════════════════════
# INTEGRATION TESTS: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_failure_agent_phase1_run(self):
        from engine.agents.failure_agent import FailureAgent
        agent  = FailureAgent()
        result = agent.run(
            model_outputs=["Paris", "Paris", "Paris"],
            primary_output="Paris",
            secondary_output="Paris",
        )
        assert "failure_signal_vector" in result
        assert "archetype"             in result
        assert "embedding_distance"    in result
        fsv = result["failure_signal_vector"]
        assert fsv["agreement_score"] == 1.0
        assert fsv["entropy_score"]   == 0.0
        assert result["archetype"]    == "STABLE"

    def test_failure_agent_phase2_run_full(self):
        from engine.agents.failure_agent import FailureAgent
        agent  = FailureAgent()
        result = agent.run_full(
            model_outputs=["Paris", "London", "Berlin", "Rome"],
            primary_output="The answer is Paris",
            secondary_output="I believe it is London",
        )
        assert "cluster_assignment" in result
        assert "label_detail"       in result
        assert "trend_summary"      in result
        ca = result["cluster_assignment"]
        assert ca["status"]   in ("NOVEL_ANOMALY", "KNOWN_FAILURE", "AMBIGUOUS")
        assert ca["archetype"] != ""

    def test_full_pipeline_stable_scenario(self):
        """
        Scenario: 5 identical correct answers from both models.
        Expected: STABLE archetype, low entropy, high agreement, no disagreement.
        """
        from engine.agents.failure_agent import FailureAgent
        agent  = FailureAgent()
        result = agent.run_full(
            model_outputs=["Paris", "Paris", "Paris", "Paris", "Paris"],
            primary_output="Paris",
            secondary_output="Paris",
        )
        fsv = result["failure_signal_vector"]
        assert fsv["entropy_score"]   == 0.0,  "All identical → entropy must be 0"
        assert fsv["agreement_score"] == 1.0,  "All identical → agreement must be 1"
        assert fsv["ensemble_disagreement"] is False
        assert fsv["high_failure_risk"]     is False
        assert result["label_detail"]["archetype"] == "STABLE"

    def test_full_pipeline_high_risk_scenario(self):
        """
        Scenario: All different outputs, models disagree.
        Expected: HALLUCINATION_RISK or UNSTABLE_OUTPUT, high entropy, low agreement.
        """
        from engine.agents.failure_agent import FailureAgent
        agent  = FailureAgent()
        result = agent.run_full(
            model_outputs=["Paris", "London", "Berlin", "Rome"],
            primary_output="Paris is the capital of France",
            secondary_output="Actually Lyon is the major city",
        )
        fsv = result["failure_signal_vector"]
        assert fsv["entropy_score"] == 1.0,   "All different → entropy must be 1.0"
        assert fsv["agreement_score"] == 0.25, "4 different answers → 0.25 agreement"
        assert fsv["high_failure_risk"] is True
        assert result["label_detail"]["archetype"] in (
            "HALLUCINATION_RISK", "UNSTABLE_OUTPUT", "MODEL_BLIND_SPOT"
        )

    def test_consistency_and_entropy_agree(self):
        """
        Mathematical consistency check:
        high agreement ↔ low entropy (they must move inversely).
        """
        from engine.detector.consistency import compute_consistency
        from engine.detector.entropy import compute_entropy

        all_same  = ["Paris"] * 5
        all_diff  = ["Paris", "London", "Berlin", "Rome", "Madrid"]

        same_agreement = compute_consistency(all_same)["agreement_score"]
        diff_agreement = compute_consistency(all_diff)["agreement_score"]
        same_entropy   = compute_entropy(all_same)
        diff_entropy   = compute_entropy(all_diff)

        assert same_agreement > diff_agreement, "Same outputs should have higher agreement"
        assert same_entropy   < diff_entropy,   "Same outputs should have lower entropy"
        assert same_entropy   == 0.0
        assert diff_entropy   == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
