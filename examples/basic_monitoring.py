"""
Basic LLM monitoring with FIE SDK.
Tracks every prompt+output and returns failure archetype + confidence.
"""
import os
from fie_sdk import FIEClient

client = FIEClient(api_key=os.getenv("FIE_API_KEY", "fie-your-key"))

result = client.monitor(
    prompt="What is the capital of Australia?",
    model_outputs=[
        "The capital of Australia is Canberra.",
        "Sydney is the capital of Australia.",
        "The capital of Australia is Canberra.",
    ],
)

print(f"Archetype:       {result.archetype}")
print(f"Failure summary: {result.failure_summary}")
print(f"High risk:       {result.high_failure_risk}")
print(f"Entropy:         {result.failure_signal_vector.entropy_score:.3f}")
print(f"Agreement:       {result.failure_signal_vector.agreement_score:.3f}")
