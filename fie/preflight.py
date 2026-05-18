"""
Pre-flight adversarial guard — intercepts prompts BEFORE the primary LLM runs.

This module is the SDK-side half of the inline protection layer.  It is
imported by fie/monitor.py and called at the very start of every wrapper so
that adversarial prompts never reach the primary model.

Design decisions
----------------
* GuardedResponse has __str__ returning the refusal message, so callers
  that just forward the result to their users see a meaningful safe reply.
* Callers that want to detect a block check `hasattr(result, "blocked")`.
* Block mode is opt-out via env var or hot-config — teams can flip to
  warn-only without redeploying.
* preflight_check() never raises — failures fall through as "not blocked"
  so a bug here can never take the primary model offline.

Configuration (in order of priority)
--------------------------------------
1. MongoDB `fie_config` collection  →  hot-reload, no restart needed
2. PREFLIGHT_BLOCK_ENABLED env var  →  "true" / "false"
3. Compiled default                 →  block_enabled = True
"""
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
    """Raw result from preflight_check() — used internally by the wrappers."""
    blocked:         bool
    attack_type:     str
    confidence:      float
    layers_fired:    list[str]
    refusal_message: str


# ── GuardedResponse ───────────────────────────────────────────────────────────

class GuardedResponse(str):
    """
    Returned to the caller instead of calling the primary LLM when a prompt is
    blocked by the pre-flight guard.

    Inherits from str so it behaves transparently in code that just forwards
    the return value as text.  Callers that want to detect a block explicitly:

        result = my_llm_fn(prompt=user_input)
        if isinstance(result, GuardedResponse) and result.blocked:
            log_security_event(result.attack_type, result.confidence)

    Attributes
    ----------
    blocked      True always (a GuardedResponse is always a block event).
    attack_type  Category of detected attack (e.g. "PROMPT_INJECTION").
    confidence   Detection confidence in [0, 1].
    layers_fired List of detection layers that fired.
    """

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


def _safe_scan(prompt: str) -> tuple[bool, str, float, list[str]]:
    """
    Run scan_prompt() and return (is_attack, attack_type, confidence, layers_fired).
    Never raises — returns (False, "", 0.0, []) on any failure.
    """
    try:
        from fie.adversarial import scan_prompt
        result = scan_prompt(prompt)
        return result.is_attack, result.attack_type, result.confidence, result.layers_fired
    except Exception as exc:
        logger.debug("preflight scan failed (allowing request through): %s", exc)
        return False, "", 0.0, []


# ── Public API ────────────────────────────────────────────────────────────────

def preflight_check(prompt: str) -> GuardResult:
    """
    Scan a prompt for adversarial content BEFORE the primary LLM is invoked.

    Parameters
    ----------
    prompt : str
        The raw user prompt.

    Returns
    -------
    GuardResult
        .blocked = True  → caller should return a GuardedResponse, skip LLM.
        .blocked = False → safe to proceed with the primary model call.

    Notes
    -----
    * Uses the same threshold as scan_prompt() (SCAN_THRESHOLD / fie_config).
    * In warn-only mode (block_enabled=False) the scan still runs and logs,
      but blocked is always False so the LLM call proceeds.
    * This function never raises — failures are logged and return blocked=False.
    """
    if not prompt or not prompt.strip():
        return GuardResult(
            blocked=False, attack_type="", confidence=0.0,
            layers_fired=[], refusal_message="",
        )

    is_attack, attack_type, confidence, layers_fired = _safe_scan(prompt)

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
