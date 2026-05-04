"""
Core unit tests for FIE pipeline components.
Run with: pytest tests/test_core.py -v
No server required — all tests are offline.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Question Classifier ───────────────────────────────────────────────────────

class TestQuestionClassifier:
    from engine.question_classifier import classify

    def test_factual(self):
        from engine.question_classifier import classify
        assert classify("Who invented the telephone?") == "FACTUAL"

    def test_temporal(self):
        from engine.question_classifier import classify
        assert classify("What is the current weather in Delhi?") == "TEMPORAL"
        assert classify("Who is currently the US president?") == "TEMPORAL"

    def test_code(self):
        from engine.question_classifier import classify
        assert classify("Write a Python function to sort a list") == "CODE"
        assert classify("How do I implement a binary search?") == "CODE"

    def test_opinion(self):
        from engine.question_classifier import classify
        assert classify("Should I use Python or JavaScript?") == "OPINION"
        assert classify("What is your recommendation for a database?") == "OPINION"

    def test_reasoning(self):
        from engine.question_classifier import classify
        assert classify("Why does entropy increase over time?") == "REASONING"

    def test_returns_string(self):
        from engine.question_classifier import classify
        result = classify("Some random question")
        assert isinstance(result, str)
        assert result in {"FACTUAL", "TEMPORAL", "CODE", "OPINION", "REASONING", "UNKNOWN"}

    def test_empty_string(self):
        from engine.question_classifier import classify
        result = classify("")
        assert isinstance(result, str)


# ── Failure Classifier (unit — no model file needed) ─────────────────────────

class TestFailureClassifierFallback:

    def test_fallback_high_risk(self):
        """When model files are absent, predict() falls back to POET (high_failure_risk)."""
        from unittest.mock import patch
        import engine.failure_classifier as fc

        # Also patch _loaded=True so _load() returns immediately without touching _model
        with patch.object(fc, "_loaded", True), \
             patch.object(fc, "_model", None), \
             patch.object(fc, "_feat_cols", None):
            is_failure, prob = fc.predict(
                agreement_score=0.5, entropy_score=0.8,
                jury_confidence=0.6, fix_confidence=0.0,
                gt_confidence=0.0, high_failure_risk=True,
                fix_applied=False, requires_escalation=False,
                gt_override=False, archetype="HALLUCINATION_RISK",
                jury_verdict_str="FACTUAL_HALLUCINATION",
                fix_strategy="NO_FIX", gt_source="none",
            )
            assert is_failure is True
            assert prob == 1.0

    def test_fallback_low_risk(self):
        from unittest.mock import patch
        import engine.failure_classifier as fc

        with patch.object(fc, "_loaded", True), \
             patch.object(fc, "_model", None), \
             patch.object(fc, "_feat_cols", None):
            is_failure, prob = fc.predict(
                agreement_score=1.0, entropy_score=0.0,
                jury_confidence=0.0, fix_confidence=0.0,
                gt_confidence=0.0, high_failure_risk=False,
                fix_applied=False, requires_escalation=False,
                gt_override=False, archetype="STABLE",
                jury_verdict_str="NONE", fix_strategy="NO_FIX",
                gt_source="none",
            )
            assert is_failure is False
            assert prob == 0.0

    def test_threshold_value(self):
        from engine.failure_classifier import CLASSIFIER_THRESHOLD
        assert CLASSIFIER_THRESHOLD == pytest.approx(0.522, abs=0.001)

    def test_predict_returns_tuple(self):
        from unittest.mock import patch
        import engine.failure_classifier as fc

        with patch.object(fc, "_loaded", True), \
             patch.object(fc, "_model", None), \
             patch.object(fc, "_feat_cols", None):
            result = fc.predict(
                agreement_score=0.75, entropy_score=0.4,
                jury_confidence=0.5, fix_confidence=0.0,
                gt_confidence=0.0, high_failure_risk=False,
                fix_applied=False, requires_escalation=False,
                gt_override=False, archetype="STABLE",
                jury_verdict_str="NONE", fix_strategy="NO_FIX",
                gt_source="none",
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], float)


# ── FIE Config ────────────────────────────────────────────────────────────────

class TestFIEConfig:

    def test_default_thresholds(self):
        from engine.fie_config import get_threshold, get_all_thresholds
        thresholds = get_all_thresholds()
        assert "FACTUAL" in thresholds
        assert "CODE" in thresholds
        assert "OPINION" in thresholds
        assert thresholds["FACTUAL"] < thresholds["OPINION"]

    def test_get_threshold_factual(self):
        from engine.fie_config import get_threshold
        t = get_threshold("FACTUAL")
        assert 0.0 < t < 1.0

    def test_get_threshold_unknown(self):
        from engine.fie_config import get_threshold
        t = get_threshold("NONEXISTENT_TYPE")
        assert t == get_threshold("UNKNOWN")

    def test_get_threshold_case_insensitive(self):
        from engine.fie_config import get_threshold
        assert get_threshold("factual") == get_threshold("FACTUAL")

    def test_model_version_is_v4(self):
        from engine.fie_config import MODEL_VERSION
        assert MODEL_VERSION == "xgboost-v4"


# ── SDK Local Predictor ───────────────────────────────────────────────────────

class TestLocalPredictor:

    def test_opinion_always_safe(self):
        from fie.local_predictor import predict_local
        result = predict_local("Should I use React or Vue?", "I think React is better.")
        assert result.is_suspicious is False
        assert result.question_type == "OPINION"

    def test_code_always_safe(self):
        from fie.local_predictor import predict_local
        result = predict_local("Write a sort function", "def sort(lst): return sorted(lst)")
        assert result.is_suspicious is False
        assert result.question_type == "CODE"

    def test_heavy_hedging_flagged(self):
        from fie.local_predictor import predict_local
        response = (
            "I'm not sure, but I think it was probably Thomas Edison. "
            "I may be wrong about this. I cannot confirm this."
        )
        result = predict_local("Who invented the telephone?", response)
        assert result.confidence > 0.0
        assert result.signals["hedge_phrases_found"] >= 2

    def test_confident_correct_answer_safe(self):
        from fie.local_predictor import predict_local
        result = predict_local(
            "What is the capital of France?",
            "The capital of France is Paris."
        )
        assert result.confidence < 0.3

    def test_temporal_signal_detected(self):
        from fie.local_predictor import predict_local
        response = "As of my training data cutoff, I don't have access to real-time prices."
        result = predict_local("What is Bitcoin's price?", response)
        assert result.signals["temporal_signals_found"] >= 1

    def test_returns_dict(self):
        from fie.local_predictor import predict_local
        result = predict_local("Who invented the telephone?", "Alexander Graham Bell")
        d = result.to_dict()
        assert "high_failure_risk" in d
        assert "classifier_probability" in d
        assert d["mode"] == "local"

    def test_empty_response(self):
        from fie.local_predictor import predict_local
        result = predict_local("What is 2+2?", "")
        assert isinstance(result.is_suspicious, bool)


# ── SDK Config ────────────────────────────────────────────────────────────────

class TestSDKConfig:

    def test_get_config_defaults(self):
        from fie.config import get_config
        cfg = get_config(fie_url="http://test:8000", api_key="test-key")
        assert cfg.fie_url == "http://test:8000"
        assert cfg.api_key == "test-key"

    def test_url_trailing_slash_stripped(self):
        from fie.config import get_config
        cfg = get_config(fie_url="http://test:8000/", api_key="k")
        assert not cfg.fie_url.endswith("/")


# ── Entropy / Detector helpers ────────────────────────────────────────────────

class TestEntropyDetector:

    def test_zero_entropy_full_agreement(self):
        from engine.detector.entropy import compute_entropy
        outputs = ["Paris", "Paris", "Paris", "Paris"]
        score = compute_entropy(outputs)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_max_entropy_all_different(self):
        from engine.detector.entropy import compute_entropy
        outputs = ["A", "B", "C", "D"]
        score = compute_entropy(outputs)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_agreement(self):
        from engine.detector.entropy import compute_entropy
        outputs = ["Paris", "Paris", "Paris", "London"]
        score = compute_entropy(outputs)
        assert 0.0 < score < 1.0
