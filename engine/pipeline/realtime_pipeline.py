"""
Thin wrapper kept for backwards compatibility.
All logic has moved to langgraph_pipeline.run_pipeline().
"""
from engine.pipeline.langgraph_pipeline import run_pipeline


def run_realtime_pipeline(prompt: str, model_outputs: list[str] | None = None) -> dict:
    outputs = model_outputs or []
    return run_pipeline(
        prompt=prompt,
        model_outputs=outputs,
        primary_output=outputs[0] if outputs else "",
    )
