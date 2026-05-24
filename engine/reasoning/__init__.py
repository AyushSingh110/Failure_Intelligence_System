"""
engine.reasoning
================
Step-level reasoning failure detection for FIE.

Works fully offline (no GPU, no API key required):
  - chain_decomposer  : splits reasoning output into numbered atomic steps
  - step_verifier     : checks each step (factual / logical / arithmetic)
  - socratic_probe    : generates adversarial follow-up challenges
  - reasoning_verifier: orchestrates all three → returns ReasoningVerificationResult

When Groq is available the components use LLM-assisted decomposition and
probing.  When Groq is unavailable every component falls back to a
rule-based / math-eval path so the pipeline never blocks.
"""
from engine.reasoning.chain_decomposer   import decompose_reasoning_chain, ReasoningStep
from engine.reasoning.step_verifier      import verify_steps,             StepVerificationResult
from engine.reasoning.socratic_probe     import run_socratic_probe,       SocraticProbeResult
from engine.reasoning.reasoning_verifier import verify_reasoning,         ReasoningVerificationResult

__all__ = [
    "decompose_reasoning_chain", "ReasoningStep",
    "verify_steps",              "StepVerificationResult",
    "run_socratic_probe",        "SocraticProbeResult",
    "verify_reasoning",          "ReasoningVerificationResult",
]
