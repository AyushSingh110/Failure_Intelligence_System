"""
fie lite mode — 4-layer adversarial scan with zero heavy dependencies.

Runs layers 1, 2, 5, 8 only (regex, GCG, many-shot, multilingual) with no
sentence-transformers or sklearn required. Trade-off: ~15-20% lower recall,
but imports in <100 ms and works in minimal environments (Lambda, edge, etc.).

Usage:
    from fie._lite import scan_prompt_lite
    result = scan_prompt_lite("your prompt here")

    # Or via fie.adversarial with env var:
    import os
    os.environ["FIE_LITE"] = "1"
    from fie.adversarial import scan_prompt

    # Or install lite extras only:
    pip install fie-sdk   # (no [ml] extras — lite runs on base install)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LiteScanResult:
    """Lightweight scan result — subset of ScanResult fields."""
    is_attack:    bool
    attack_type:  str | None
    confidence:   float
    layers_fired: list[str]
    evidence:     dict = field(default_factory=dict)


def scan_prompt_lite(prompt: str, threshold: float = 0.65) -> LiteScanResult:
    """
    Scan a prompt using 4 layers that require no ML model downloads.

    Layers included:
      1. regex        — pattern-matching (injection phrases, role-play triggers)
      2. gcg_suffix   — GCG adversarial suffix detection (entropy + token anomaly)
      3. many_shot    — Many-shot jailbreak detection (repeated Q/A patterns)
      4. multilingual — Tier 1+2 multilingual injection (script anomaly + phrases)

    Layers excluded (require sentence-transformers / sklearn):
      semantic, pair, perplexity, indirect, direct_harm, virtualization, fiction_harm

    Returns LiteScanResult. All fields compatible with ScanResult equivalents.
    """
    from fie.adversarial import (
        _layer_regex,
        _layer_gcg,
        _layer_many_shot,
        _layer_multilingual,
    )

    lite_layers = [
        ("regex",        lambda: _layer_regex(prompt)),
        ("gcg_suffix",   lambda: _layer_gcg(prompt)),
        ("many_shot",    lambda: _layer_many_shot(prompt)),
        ("multilingual", lambda: _layer_multilingual(prompt)),
    ]

    fired_types: list[str]  = []
    fired_names: list[str]  = []
    combined_ev: dict       = {}
    best_conf   = 0.0
    best_type   = None

    for name, fn in lite_layers:
        try:
            attack_type, confidence, evidence = fn()
        except Exception:
            continue

        if attack_type is not None:
            fired_names.append(name)
            combined_ev[name] = evidence
            if confidence > best_conf:
                best_conf = confidence
                best_type = attack_type

    is_attack = best_type is not None and best_conf >= threshold

    return LiteScanResult(
        is_attack    = is_attack,
        attack_type  = best_type if is_attack else None,
        confidence   = round(best_conf, 4) if is_attack else 0.0,
        layers_fired = fired_names,
        evidence     = combined_ev,
    )
