"""Pre-flight adversarial guard. Runs scan_prompt before the LLM call; returns GuardedResponse on block."""
from __future__ import annotations

import dataclasses
import logging
import os

from fie._degrade import attempt

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
    # True when the scanner itself failed and the prompt was never actually
    # examined. Distinct from blocked=False, which means "scanned, looked safe".
    # Callers enforcing their own policy must branch on this: an unscanned
    # prompt is not a safe prompt. Defaults False so existing constructors and
    # any code unpacking this dataclass keep working unchanged.
    scan_failed:     bool = False
    # Carried over from the scan result (ScanResult.zone, ScanResult.coverage.status).
    # Both are None when no scan produced them: an empty prompt, or scan_failed.
    # `blocked` is the enforcement decision; `zone` is the scanner's routing
    # outcome, so in warn-only mode a result can be blocked=False with a block zone.
    # `coverage_status` other than "full" means blocked=False rests on fewer layers
    # than the full pipeline.
    zone:            str | None = None
    coverage_status: str | None = None
    schema_version:  int = 2

    def to_dict(self) -> dict:
        """JSON-safe form, fixed key order."""
        return {
            "blocked":         bool(self.blocked),
            "attack_type":     self.attack_type,
            "confidence":      float(self.confidence),
            "layers_fired":    sorted(self.layers_fired or []),
            "refusal_message": self.refusal_message,
            "scan_failed":     bool(self.scan_failed),
            "zone":            self.zone,
            "coverage_status": self.coverage_status,
            "schema_version":  self.schema_version,
        }


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

def _server_preflight_block_enabled() -> bool:
    """Read block_enabled from the server hot-config. Raises if unavailable."""
    from engine.fie_config import get_preflight_config
    return get_preflight_config()["block_enabled"]


def _get_block_enabled() -> bool:
    """
    Returns whether block mode is currently active.

    Reads from engine.fie_config (MongoDB-backed, hot-configurable) first.
    Falls back to PREFLIGHT_BLOCK_ENABLED env var, then True.
    """
    # SDK-only installs have no engine package; that is the common case, not an
    # error. Falls back to the env var, which itself defaults to True (blocking)
    # — losing hot-config must never silently turn the guard off.
    return attempt(
        lambda: _server_preflight_block_enabled(),
        default    = _ENV_BLOCK_ENABLED,
        capability = "fie_config",
        impact     = f"using env/default block_enabled={_ENV_BLOCK_ENABLED} "
                     "instead of hot config",
        logger_    = logger,
    )


class _ScanOutcome(tuple):
    """
    The five values _safe_scan() has always returned, with the scan result's
    zone and coverage status riding along as attributes. It is a tuple and
    unpacks as one, so code that calls or replaces _safe_scan() is unaffected.
    """
    zone:            str | None = None
    coverage_status: str | None = None


# What to do when the scanner itself fails (not when it returns "safe").
#
#   "open"   — allow the prompt through. Availability over security.
#   "closed" — block it. Security over availability. RECOMMENDED for anything
#              exposed to untrusted input.
#
# The default is "open" to preserve historical behaviour, but note what that
# means: if scan_prompt() raises, every prompt is forwarded to your model
# unscanned, and the caller sees blocked=False — indistinguishable from a clean
# scan unless it checks `scan_failed`. An attacker who can reliably crash the
# scanner has disabled the guard. Set FIE_SCAN_FAILURE_MODE=closed in production.
_SCAN_FAILURE_MODE: str = os.environ.get("FIE_SCAN_FAILURE_MODE", "open").strip().lower()
_FAIL_SECURE: bool = _SCAN_FAILURE_MODE == "closed"


def _safe_scan(
    prompt: str,
    session_id: str | None = None,
    domain: str | None = None,
) -> tuple[bool, str, float, list[str], bool]:
    """
    Run scan_prompt(). Never raises.

    Returns (is_attack, attack_type, confidence, layers_fired, scan_failed).

    `scan_failed` is the important addition: without it, a crashed scanner and a
    genuinely clean prompt both returned (False, "", 0.0, []) and no caller could
    tell them apart. Callers must treat scan_failed=True as "not scanned", never
    as "safe".
    """
    try:
        from fie.adversarial import scan_prompt
        result = scan_prompt(prompt, session_id=session_id, domain=domain)
        outcome = _ScanOutcome((
            result.is_attack,
            result.attack_type or "",
            result.confidence,
            result.layers_fired,
            False,
        ))
        zone = getattr(result, "zone", None)
        coverage_status = getattr(getattr(result, "coverage", None), "status", None)
        outcome.zone = zone if isinstance(zone, str) else None
        outcome.coverage_status = coverage_status if isinstance(coverage_status, str) else None
        return outcome
    except Exception as exc:
        # ERROR, not debug: the guard just stopped guarding. This is precisely
        # the event an operator needs paged on, and it was previously invisible
        # at default log levels.
        logger.error(
            "preflight scan FAILED — prompt was not scanned "
            "(failure_mode=%s, action=%s) reason=%s: %s",
            _SCAN_FAILURE_MODE,
            "block" if _FAIL_SECURE else "allow through",
            type(exc).__name__, exc,
            exc_info=True,
        )
        if _FAIL_SECURE:
            return True, "SCAN_FAILED", 1.0, ["preflight_fail_secure"], True
        return False, "", 0.0, [], True


# ── Public API ────────────────────────────────────────────────────────────────

def preflight_check(
    prompt: str,
    session_id: str | None = None,
    domain: str | None = None,
) -> GuardResult:
    """Scan prompt before the LLM call. Returns GuardResult with blocked=True if an attack is detected."""
    if not prompt or not prompt.strip():
        return GuardResult(
            blocked=False, attack_type="", confidence=0.0,
            layers_fired=[], refusal_message="",
        )

    outcome = _safe_scan(prompt, session_id=session_id, domain=domain)
    is_attack, attack_type, confidence, layers_fired, scan_failed = outcome
    zone            = getattr(outcome, "zone", None)
    coverage_status = getattr(outcome, "coverage_status", None)

    if not is_attack:
        return GuardResult(
            blocked=False, attack_type="", confidence=confidence,
            layers_fired=layers_fired, refusal_message="",
            scan_failed=scan_failed,
            zone=zone, coverage_status=coverage_status,
        )

    # A fail-secure block is not a detection — warn-only mode must not be able
    # to wave it through, because there is no verdict to warn about. Blocking
    # here is the whole point of FIE_SCAN_FAILURE_MODE=closed.
    if scan_failed:
        return GuardResult(
            blocked         = True,
            attack_type     = attack_type,
            confidence      = confidence,
            layers_fired    = layers_fired,
            refusal_message = _DEFAULT_REFUSAL,
            scan_failed     = True,
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
            zone            = zone,
            coverage_status = coverage_status,
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
        zone            = zone,
        coverage_status = coverage_status,
    )


def make_guarded_response(guard: GuardResult) -> GuardedResponse:
    """Convenience constructor — turn a GuardResult into a GuardedResponse."""
    return GuardedResponse(
        refusal_message = guard.refusal_message or _DEFAULT_REFUSAL,
        attack_type     = guard.attack_type,
        confidence      = guard.confidence,
        layers_fired    = guard.layers_fired,
    )
