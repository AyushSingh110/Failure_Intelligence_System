"""
FIE detection layers.

Each module implements one layer of the pre-flight guard and exposes a
`_run_*` function with the shared contract:

    (prompt: str, ...) -> tuple[attack_type: str | None, confidence: float, evidence: dict]

Returning `attack_type=None` means "this layer saw nothing", which contributes
zero to weighted aggregation. Raising means the layer FAILED — the runner marks
the scan degraded rather than treating the silence as evidence of safety. That
distinction is the whole reason layers report status instead of just a score.

Orchestration (thread pool, deadlines, weighted vote, meta-classifier blending,
three-zone routing) lives in `fie.adversarial`, not here. A layer never needs to
know about any other layer.

Adding a layer
--------------
1. Create `fie/layers/your_layer.py` with a `_run_your_layer(prompt)` function.
2. Register a wrapper and a weight in `fie/adversarial.py`.
3. Add a per-type threshold to `_ATTACK_THRESHOLDS` if it emits a new type.
4. Add prompts to `tests/test_detection_golden.py` and regenerate the golden
   file — a new layer changes scores, and the diff is the evidence of what it
   changed.

See CONTRIBUTING.md ("Adding a Detection Layer") for the full walkthrough.
"""
from __future__ import annotations

__all__ = [
    "copyright",
    "direct_harm",
    "gcg",
    "indirect",
    "many_shot",
    "pair",
    "patterns",
    "perplexity",
    "prompt_guard",
]
