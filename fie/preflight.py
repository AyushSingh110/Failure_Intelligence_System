"""Pre-flight adversarial guard. Runs scan_prompt before the LLM call; returns GuardedResponse on block."""
from __future__ import annotations

import dataclasses
import logging
import os

from fie import policy as _policy
from fie._degrade import attempt

logger = logging.getLogger("fie.preflight")
_policy_log = logging.getLogger("fie.policy")

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
    # The policy outcome (fie.policy). `blocked` above is the enforcement bit;
    # these say which policy produced it and why.
    #   action   "allow" | "review" | "block" | "error"; None when nothing was
    #            decided (an empty prompt, or a result built by other code)
    #   policy   "<name>@<version>" of the policy applied, or "invalid"
    #   reasons  fixed reason codes, deciding reason first; "warn_only" is
    #            appended when warn-only mode waived a detector block
    # An action is not a detector verdict: a block can come from missing coverage
    # or a scanner failure. `attack_type` is only ever what a detector reported.
    action:          str | None = None
    policy:          str | None = None
    reasons:         tuple = ()

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
            "action":          self.action,
            "policy":          self.policy,
            "reasons":         list(self.reasons or ()),
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
    The five values _safe_scan() has always returned, with the scan result, its
    zone and its coverage status riding along as attributes. It is a tuple and
    unpacks as one, so code that calls or replaces _safe_scan() is unaffected.
    """
    zone:            str | None = None
    coverage_status: str | None = None
    result:          object = None


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
        outcome.result = result
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

def _audit(decision: "_policy.PolicyDecision", blocked: bool, reasons: tuple) -> None:
    """
    One line per decision that is not a plain detector allow. Fixed codes, layer
    names and the policy id only: no prompt text, no exception text, no
    configuration value.
    """
    if decision.action == "allow" and reasons and reasons[0] == "detector_allow":
        return
    _policy_log.log(
        logging.WARNING if blocked else logging.INFO,
        "POLICY_DECISION | action=%s blocked=%s policy=%s digest=%s reasons=%s zone=%s "
        "coverage=%s missing_required=%s",
        decision.action, blocked, decision.policy, decision.policy_digest, ",".join(reasons),
        decision.zone, decision.coverage_status, ",".join(decision.missing_required) or "-",
    )


def _result_for_policy(is_attack: bool, attack_type: str, confidence: float, layers_fired: list):
    """
    A neutral scan result for a _safe_scan() replacement that returned only the
    five legacy values. It says what those values say and claims no coverage.
    """
    from fie.adversarial import ScanResult
    return ScanResult(
        is_attack=bool(is_attack), attack_type=attack_type or None, category=None,
        confidence=confidence, layers_fired=list(layers_fired or []), matched_text=None, mitigation="",
    )


def preflight_check(
    prompt: str,
    session_id: str | None = None,
    domain: str | None = None,
    *,
    policy: "_policy.Policy | None" = None,
) -> GuardResult:
    """
    Scan prompt before the LLM call and apply the runtime policy to the result.

    Returns a GuardResult. `blocked` is the enforcement decision; `action`,
    `policy` and `reasons` say which policy produced it and why (fie.policy).

    policy  A fie.policy.Policy to apply to this call, passed by the integrator's
            own code. Keyword-only. When omitted, the process policy is used
            (fie.policy.active_policy(): FIE_POLICY, default compat@1, which
            behaves exactly as this function did before policies existed).
            Never pass a value that arrived in a request.

    The policy is resolved once, before the scan, and that one object decides.
    If the process policy configuration is invalid, nothing is scanned and the
    result is a blocking `error` with reason `invalid_policy_config`.
    """
    if policy is not None and not isinstance(policy, _policy.Policy):
        raise TypeError("preflight_check(policy=...) requires a fie.policy.Policy instance")

    if not prompt or not prompt.strip():
        return GuardResult(
            blocked=False, attack_type="", confidence=0.0,
            layers_fired=[], refusal_message="",
        )

    # ── Resolve the policy: once, before scanning ─────────────────────────────
    if policy is None:
        try:
            policy = _policy.active_policy()
        except _policy.PolicyConfigError:
            # No valid policy exists, so there is nothing to evaluate a scan
            # under. Not a scanner failure: the scanner is not run. Warn-only
            # mode cannot waive this.
            decision = _policy.invalid_config_decision()
            _audit(decision, True, decision.reasons)
            return GuardResult(
                blocked         = True,
                attack_type     = "",
                confidence      = 0.0,
                layers_fired    = [],
                refusal_message = _DEFAULT_REFUSAL,
                action          = decision.action,
                policy          = decision.policy,
                reasons         = decision.reasons,
            )

    outcome = _safe_scan(prompt, session_id=session_id, domain=domain)
    is_attack, attack_type, confidence, layers_fired, scan_failed = outcome
    zone            = getattr(outcome, "zone", None)
    coverage_status = getattr(outcome, "coverage_status", None)

    # ── Decide, with the object resolved above ────────────────────────────────
    if scan_failed:
        decision = _policy.decide(None, policy, scan_failed=True)
    else:
        result = getattr(outcome, "result", None)
        if result is None:
            result = _result_for_policy(is_attack, attack_type, confidence, layers_fired)
        decision = _policy.decide(result, policy)

    blocked = decision.blocked
    reasons = decision.reasons

    # ── Warn-only: an enforcement override, applied after the decision ────────
    # It waives a block that came from a DETECTION and nothing else. A block for
    # missing coverage, a scanner failure or a configuration error is not a
    # detection: there is no verdict to warn about, so it stands. The action and
    # the deciding reason are never changed.
    if (
        blocked
        and decision.action in ("block", "review")
        and reasons[0] in _policy.DETECTOR_BLOCK_REASONS
        and not _get_block_enabled()
    ):
        blocked = False
        reasons = reasons + ("warn_only",)

    _audit(decision, blocked, reasons)

    if scan_failed:
        # The scanner produced nothing. What is reported follows the decision:
        # blocked -> the fail-secure marker; allowed -> empty, with scan_failed set.
        if blocked:
            attack_type, confidence, layers_fired = "SCAN_FAILED", 1.0, ["preflight_fail_secure"]
        else:
            attack_type, confidence, layers_fired = "", 0.0, []
        return GuardResult(
            blocked         = blocked,
            attack_type     = attack_type,
            confidence      = confidence,
            layers_fired    = layers_fired,
            refusal_message = _DEFAULT_REFUSAL if blocked else "",
            scan_failed     = True,
            action          = decision.action,
            policy          = decision.policy,
            reasons         = reasons,
        )

    if is_attack:
        if blocked:
            logger.warning(
                "PREFLIGHT_BLOCK | attack_type=%s confidence=%.3f layers=%s",
                attack_type, confidence, ",".join(layers_fired),
            )
        else:
            # Warn-only mode — log but let the request through
            logger.warning(
                "PREFLIGHT_WARN (block_enabled=False) | attack_type=%s confidence=%.3f layers=%s",
                attack_type, confidence, ",".join(layers_fired),
            )

    return GuardResult(
        blocked         = blocked,
        # Only ever what a detector reported. A policy block of a prompt the
        # detector allowed carries no attack type: none was found.
        attack_type     = attack_type if is_attack else "",
        confidence      = confidence,
        layers_fired    = layers_fired,
        refusal_message = _DEFAULT_REFUSAL if blocked else "",
        scan_failed     = False,
        zone            = zone,
        coverage_status = coverage_status,
        action          = decision.action,
        policy          = decision.policy,
        reasons         = reasons,
    )


def make_guarded_response(guard: GuardResult) -> GuardedResponse:
    """Convenience constructor — turn a GuardResult into a GuardedResponse."""
    return GuardedResponse(
        refusal_message = guard.refusal_message or _DEFAULT_REFUSAL,
        attack_type     = guard.attack_type,
        confidence      = guard.confidence,
        layers_fired    = guard.layers_fired,
    )
