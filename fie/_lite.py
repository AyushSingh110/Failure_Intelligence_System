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
import logging
from typing import Any
logger = logging.getLogger(__name__)

# The four layers this scanner runs. The other eight are reported as `disabled`.
_LITE_LAYERS: tuple[str, ...] = ("regex", "gcg_suffix", "many_shot", "multilingual")


@dataclass
class LiteScanResult:
    """Lightweight scan result — subset of ScanResult fields."""
    is_attack:    bool
    attack_type:  str | None
    confidence:   float
    layers_fired: list[str]
    evidence:     dict = field(default_factory=dict)
    # Layers that raised during this scan. Non-empty means reduced coverage:
    # is_attack=False is weaker evidence of safety than usual.
    degraded_layers: list[str] = field(default_factory=list)

    # ── Schema version 2: the same fields ScanResult carries ──────────────────
    # `coverage` is a fie.adversarial.ScanCoverage. A lite scan runs four of the
    # twelve layers, so its status is always "partial": the eight layers it
    # never runs are `disabled`, as are the meta-classifier and the tiebreaker.
    # There is no uncertain band here, so `zone` is "allow" or "clear_block".
    zone:           str | None = None
    decided_by:     str | None = None
    coverage:       Any = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.zone is None:
            self.zone = "clear_block" if self.is_attack else "allow"

    def to_dict(self) -> dict:
        """JSON-safe form. A subset of ScanResult.to_dict(), same keys, same order."""
        from fie.adversarial import _json_safe

        return {
            "is_attack":       bool(self.is_attack),
            "attack_type":     self.attack_type,
            "confidence":      float(self.confidence),
            "layers_fired":    sorted(self.layers_fired or []),
            "evidence":        _json_safe(self.evidence if isinstance(self.evidence, dict) else {}),
            "degraded_layers": sorted(self.degraded_layers or []),
            "zone":            self.zone,
            "decided_by":      self.decided_by,
            "coverage":        self.coverage.to_dict() if self.coverage is not None else None,
            "schema_version":  self.schema_version,
        }


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
        ALL_LAYERS,
        ScanCoverage,
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

    fired_names: list[str]  = []
    degraded_names: list[str] = []
    combined_ev: dict       = {}
    best_conf   = 0.0
    best_type   = None
    states: dict[str, str] = {name: "disabled" for name in ALL_LAYERS}
    translation = "not_needed"

    for name, fn in lite_layers:
        try:
            out = fn()
            attack_type, confidence, evidence = out
            states[name] = "ok"
            if name == "multilingual":
                noted = (getattr(out, "notes", None) or {}).get("translation")
                if noted in ("ok", "unavailable"):
                    translation = noted
        except Exception as exc:
            # Lite mode already runs a reduced layer set, so a further layer
            # loss matters more here than in the full pipeline. Record it on the
            # result so the caller can see the scan was incomplete.
            logger.warning(
                "degraded capability=layer:%s impact='lite scan ran with "
                "reduced coverage' reason=%s: %s",
                name, type(exc).__name__, exc,
            )
            degraded_names.append(name)
            states[name] = "error"
            if name == "multilingual":
                translation = "unavailable"
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
        degraded_layers = degraded_names,
        zone         = "clear_block" if is_attack else "allow",
        decided_by   = "pipeline",
        coverage     = ScanCoverage(
            status     = "partial",
            layers     = states,
            classifier = states["pair_classifier"],
            optional   = {"meta_classifier": "disabled", "tiebreaker": "disabled",
                          "translation": translation},
        ),
    )
