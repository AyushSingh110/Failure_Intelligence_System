"""
D-064 (OD-S0-4): an unchecked factual reasoning step is NOT_CHECKED, the dead
ground-truth call is gone, and consumers can tell unchecked from incorrect.
"""
from __future__ import annotations

import types

import pytest

from engine.reasoning.chain_decomposer import ReasoningStep


@pytest.fixture
def no_gt_and_no_judge(monkeypatch):
    """Any ground-truth pipeline or Groq call fails the test."""
    import engine.groq_service as gs
    import engine.verifier.ground_truth_pipeline as gtp

    def forbidden(*a, **k):
        raise AssertionError("factual step verification must not call out")
    monkeypatch.setattr(gtp, "run_ground_truth_pipeline", forbidden)
    monkeypatch.setattr(gs, "get_groq_service", forbidden)


def _steps():
    return [
        ReasoningStep(index=1, text="Thomas Edison invented the telephone.", step_type="FACTUAL_PREMISE"),
        ReasoningStep(index=2, text="2 + 2 = 5", step_type="ARITHMETIC"),
        ReasoningStep(index=3, text="3 * 4 = 12", step_type="ARITHMETIC"),
    ]


@pytest.mark.parametrize("use_groq", [True, False])
def test_factual_step_is_not_checked_and_no_call_is_made(no_gt_and_no_judge, use_groq):
    from engine.reasoning.step_verifier import verify_steps
    steps = _steps()
    results = verify_steps(steps[:1], "Who invented the telephone?", [], [], use_groq=use_groq)
    assert steps[0].verification_status == "NOT_CHECKED"
    assert results[0].confidence == 0.0
    assert results[0].method in ("not_checked", "offline_skipped")


def test_unchecked_is_distinguishable_from_incorrect(no_gt_and_no_judge):
    from engine.reasoning.step_verifier import verify_steps
    steps = _steps()
    verify_steps(steps, "q", [], [], use_groq=True)
    factual, wrong, right = steps
    assert (factual.is_correct, factual.verification_status) == (True, "NOT_CHECKED")
    assert (wrong.is_correct, wrong.verification_status) == (False, "CONTRADICTED")
    assert (right.is_correct, right.verification_status) == (True, "SUPPORTED")


def test_unverified_steps_default_to_not_checked():
    from app.schemas import ReasoningStepResult
    assert ReasoningStep(index=1, text="x", step_type="FACTUAL_PREMISE").verification_status == "NOT_CHECKED"
    assert ReasoningStepResult(index=1, text="x", step_type="FACTUAL_PREMISE").verification_status == "NOT_CHECKED"


def test_monitor_response_carries_the_step_status(monkeypatch):
    """The /monitor reasoning block exposes verification_status; is_correct keeps its meaning."""
    import engine.reasoning.reasoning_verifier as rv

    from .e_r1_part_a import BELL, Shadow
    from .harness import Scenario, run_monitor

    step = ReasoningStep(index=1, text="Thomas Edison invented the telephone.", step_type="FACTUAL_PREMISE",
                         is_verified=True, is_correct=True, confidence=0.0)
    monkeypatch.setattr("engine.question_classifier.classify", lambda prompt: "REASONING")
    monkeypatch.setattr(rv, "verify_reasoning", lambda **kw: types.SimpleNamespace(
        failure_detected=False, failure_type="NO_FAILURE_DETECTED", confidence=0.0, total_steps=1,
        first_failed_step=None, steps=[step], pipeline_trace=[], socratic=None))
    sc = Scenario("Who invented the telephone, step by step?", BELL, [Shadow(BELL), Shadow(BELL)])
    r = run_monitor(sc).response
    [s] = r.reasoning_verification.steps
    assert (s.is_correct, s.verification_status) == (True, "NOT_CHECKED")
