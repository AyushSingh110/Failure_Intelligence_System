"""Pre-flight adversarial guard. Runs scan_prompt before the LLM call; returns GuardedResponse on block."""
from __future__ import annotations

import dataclasses
import logging
import os

logger = logging.getLogger("fie.preflight")

# ── Env-var defaults (read once at import time) ────────────────────────────────
_ENV_BLOCK_ENABLED: bool = os.environ.get(
    "PREFLIGHT_BLOCK_ENABLED", "true"
).lower() not in ("0", "false", "no")

_DEFAULT_REFUSAL: str = os.environ.get(
    "PREFLIGHT_REFUSAL_MESSAGE",
    (
        "I'm unable to process this request. "
        "It was flagged by the security layer as potentially adversarial. "
        "Please rephrase your message."
    ),
)


# ── GuardResult ───────────────────────────────────────────────────────────────

@dataclasses.dataclass
class GuardResult:
    blocked:         bool
    attack_type:     str
    confidence:      float
    layers_fired:    list[str]
    refusal_message: str


# ── GuardedResponse ───────────────────────────────────────────────────────────

class GuardedResponse(str):
    """str subclass returned when a prompt is blocked. Transparent to callers that forward return values."""

    blocked:      bool
    attack_type:  str
    confidence:   float
    layers_fired: list[str]

    def __new__(
        cls,
        refusal_message: str,
        attack_type:     str,
        confidence:      float,
        layers_fired:    list[str],
    ) -> "GuardedResponse":
        instance = super().__new__(cls, refusal_message)
        instance.blocked      = True
        instance.attack_type  = attack_type
        instance.confidence   = confidence
        instance.layers_fired = layers_fired
        return instance

    def __repr__(self) -> str:
        return (
            f"GuardedResponse(blocked=True, attack_type={self.attack_type!r}, "
            f"confidence={self.confidence:.3f})"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_block_enabled() -> bool:
    """
    Returns whether block mode is currently active.

    Reads from engine.fie_config (MongoDB-backed, hot-configurable) first.
    Falls back to PREFLIGHT_BLOCK_ENABLED env var, then True.
    """
    try:
        from engine.fie_config import get_preflight_config
        return get_preflight_config()["block_enabled"]
    except Exception:
        return _ENV_BLOCK_ENABLED


def _safe_scan(prompt: str, session_id: str | None = None) -> tuple[bool, str, float, list[str]]:
    """
    Run scan_prompt() and return (is_attack, attack_type, confidence, layers_fired).
    Never raises — returns (False, "", 0.0, []) on any failure.
    """
    try:
        from fie.adversarial import scan_prompt
        result = scan_prompt(prompt, session_id=session_id)
        return result.is_attack, result.attack_type, result.confidence, result.layers_fired
    except Exception as exc:
        logger.debug("preflight scan failed (allowing request through): %s", exc)
        return False, "", 0.0, []


# ── Public API ────────────────────────────────────────────────────────────────

def preflight_check(prompt: str, session_id: str | None = None) -> GuardResult:
    """Scan prompt before the LLM call. Returns GuardResult with blocked=True if an attack is detected."""
    if not prompt or not prompt.strip():
        return GuardResult(
            blocked=False, attack_type="", confidence=0.0,
            layers_fired=[], refusal_message="",
        )

    is_attack, attack_type, confidence, layers_fired = _safe_scan(prompt, session_id=session_id)

    if not is_attack:
        return GuardResult(
            blocked=False, attack_type="", confidence=confidence,
            layers_fired=layers_fired, refusal_message="",
        )

    block_enabled = _get_block_enabled()

    if block_enabled:
        logger.warning(
            "PREFLIGHT_BLOCK | attack_type=%s confidence=%.3f layers=%s",
            attack_type, confidence, ",".join(layers_fired),
        )
        return GuardResult(
            blocked         = True,
            attack_type     = attack_type,
            confidence      = confidence,
            layers_fired    = layers_fired,
            refusal_message = _DEFAULT_REFUSAL,
        )

    # Warn-only mode — log but let the request through
    logger.warning(
        "PREFLIGHT_WARN (block_enabled=False) | attack_type=%s confidence=%.3f layers=%s",
        attack_type, confidence, ",".join(layers_fired),
    )
    return GuardResult(
        blocked         = False,
        attack_type     = attack_type,
        confidence      = confidence,
        layers_fired    = layers_fired,
        refusal_message = "",
    )


def make_guarded_response(guard: GuardResult) -> GuardedResponse:
    """Convenience constructor — turn a GuardResult into a GuardedResponse."""
    return GuardedResponse(
        refusal_message = guard.refusal_message or _DEFAULT_REFUSAL,
        attack_type     = guard.attack_type,
        confidence      = guard.confidence,
        layers_fired    = guard.layers_fired,
    )
