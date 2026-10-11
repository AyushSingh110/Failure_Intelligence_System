"""
Runtime policy: what to do with a scan result.

Three things are kept apart here on purpose.

    Detection   what the detectors concluded     ScanResult.is_attack, .zone, .decided_by
    Coverage    which detectors actually ran     ScanResult.coverage
    Policy      what the application should do   PolicyDecision.action   (this module)

`decide(result, policy)` turns the first two into the third. It is a pure
function: it reads its arguments and nothing else. It does not call a detector,
does not change or annotate the result, and is never stored on one.

    from fie.adversarial import scan_prompt
    from fie.policy import STRICT, decide

    result   = scan_prompt(prompt)
    decision = decide(result, STRICT)
    if decision.blocked:
        ...                      # decision.reasons says why, in fixed codes

An action is not a verdict. `action == "block"` does not mean a detector fired:
a policy can block because coverage it requires is missing, or because the
scanner failed. `reasons[0]` is the reason that decided; `detector_is_attack`
and `zone` on the decision repeat what the detector said, next to the action.

Which policy a process uses is a separate question, answered by
`active_policy()`. The default, `compat@1`, reproduces the behaviour FIE had
before this module existed: in particular it still ALLOWS a prompt that was
scanned without the classifier, now with an explicit reason on the decision.
Stricter presets apply only when an operator selects them.

Where this is enforced: `fie.preflight.preflight_check`, and nowhere else. Code
that calls `scan_prompt()` directly and branches on `is_attack` gets no policy.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from fie.adversarial import ALL_LAYERS, ZONES

logger = logging.getLogger(__name__)

DECISION_SCHEMA_VERSION: int = 1

#   allow    the policy permits forwarding the input. Not a statement that the
#            input is safe or that the scan was complete.
#   review   the policy will not decide alone: uncertain evidence, or coverage it
#            requires is missing. Bound to `blocked` by Policy.on_review.
#   block    the policy forbids forwarding.
#   error    the input could not be evaluated to the standard the policy
#            requires, or no valid policy exists. Stops the input, and says the
#            guard is unhealthy rather than that this input was refused.
ACTIONS: tuple[str, ...] = ("allow", "review", "block", "error")

# Why a decision came out as it did. Fixed codes; `reasons[0]` decided the action.
REASONS: tuple[str, ...] = (
    # from the detector's verdict
    "detector_block",               # zone clear_block
    "detector_uncertain",           # zone uncertain_block, no tiebreaker verdict
    "tiebreaker_confirmed",         # zone uncertain_block, the tiebreaker confirmed
    "labelled_attack",              # a stored human label: attack
    "labelled_safe",                # a stored human label: false positive
    "tiebreaker_cleared",           # zone uncertain_allow, the tiebreaker cleared
    "uncertain_allowed_by_config",  # zone uncertain_allow, FIE_UNCERTAIN_ALLOW
    "detector_allow",               # zone allow
    # from coverage
    "missing_required_layer",       # a layer the policy requires did not run
    "coverage_unknown",             # the result does not say what ran
    "degraded_optional_layer",      # a layer the policy does not require did not run
    "layer_disabled_by_caller",     # a layer the policy does not require was disabled
    # operational
    "scan_failed",                  # the scanner produced no result
    "invalid_policy_config",        # no valid policy exists. Never produced by decide()
    # enforcement
    "warn_only",                    # appended by preflight_check when it waives a block
)

# A block whose first reason is one of these came from a detection. Warn-only
# mode may waive these and nothing else: it exists to observe detections, not to
# switch off a coverage requirement or hide an unhealthy guard.
DETECTOR_BLOCK_REASONS: frozenset[str] = frozenset({
    "detector_block", "detector_uncertain", "tiebreaker_confirmed", "labelled_attack",
})

INVALID_POLICY_ID: str = "invalid"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_BLOCKING_ZONES = frozenset({"uncertain_block", "clear_block"})
_NOT_RUN = frozenset({"unavailable_dependency", "unavailable_model", "unavailable_load_failed", "error", "timeout"})

_ON_UNCERTAIN        = ("block", "review")
_ON_MISSING_REQUIRED = ("allow", "review", "block", "error")
_ON_SCAN_FAILURE     = ("allow", "block", "error")
_ON_REVIEW           = ("block", "allow")


class PolicyConfigError(Exception):
    """
    The process policy configuration is not valid.

    The message is fixed text. It never contains the offending value: that value
    came from the environment and may hold anything.
    """

    def __init__(self) -> None:
        super().__init__("FIE_POLICY is not a known policy name")


@dataclass(frozen=True)
class Policy:
    """
    How scan results are turned into actions. An immutable, validated value.

    name                 short lowercase token; "invalid" is reserved
    version              integer >= 1. A change in behaviour is a new version
    required_layers      layers whose absence matters (see on_missing_required)
    on_uncertain         what an uncertain, fail-secure detector block becomes:
                         "block" | "review"
    on_missing_required  what an ALLOW verdict becomes when a required layer did
                         not run: "allow" | "review" | "block" | "error".
                         A detector block is never changed by missing coverage
    on_scan_failure      what happens when the scanner produced no result:
                         "allow" | "block" | "error"
    on_review            whether a `review` action stops the input:
                         "block" | "allow"

    `id` is "<name>@<version>". `digest` is a short hash of every field, so two
    policies with the same id and different settings can be told apart.
    """
    name:                str
    version:             int
    required_layers:     tuple
    on_uncertain:        str
    on_missing_required: str
    on_scan_failure:     str
    on_review:           str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name) or self.name == INVALID_POLICY_ID:
            raise ValueError("Policy.name must be a short lowercase token and must not be the reserved name")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("Policy.version must be an integer >= 1")
        if not isinstance(self.required_layers, tuple) or any(
            not isinstance(layer, str) or layer not in ALL_LAYERS for layer in self.required_layers
        ):
            raise ValueError("Policy.required_layers must be a tuple of known layer names")
        for value, allowed, label in (
            (self.on_uncertain, _ON_UNCERTAIN, "on_uncertain"),
            (self.on_missing_required, _ON_MISSING_REQUIRED, "on_missing_required"),
            (self.on_scan_failure, _ON_SCAN_FAILURE, "on_scan_failure"),
            (self.on_review, _ON_REVIEW, "on_review"),
        ):
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"Policy.{label} must be one of {allowed}")
        # Stored in pipeline order, without duplicates: the same policy always has the same digest.
        object.__setattr__(self, "required_layers",
                           tuple(layer for layer in ALL_LAYERS if layer in self.required_layers))
        # The object is immutable, so its identity is computed once, here, and
        # not on every decision.
        canonical = json.dumps({
            "name": self.name, "version": self.version, "required_layers": list(self.required_layers),
            "on_uncertain": self.on_uncertain, "on_missing_required": self.on_missing_required,
            "on_scan_failure": self.on_scan_failure, "on_review": self.on_review,
        }, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "_id", f"{self.name}@{self.version}")
        object.__setattr__(self, "_digest", hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8])

    @property
    def id(self) -> str:
        return self._id

    @property
    def digest(self) -> str:
        return self._digest


@dataclass(frozen=True)
class PolicyDecision:
    """
    The outcome of applying one Policy to one scan.

    action              "allow" | "review" | "block" | "error"
    blocked             the enforcement bit that follows from the action
    reasons             fixed codes; reasons[0] decided the action
    policy              "<name>@<version>" of the policy that decided, or "invalid"
    policy_digest       that policy's digest; None when policy == "invalid"
    detector_is_attack  what the detector said (None when there is no result)
    zone                the detector's routing zone (None when there is no result)
    coverage_status     "full" | "partial" | "bypassed" | None
    missing_required    required layers that did not run, in pipeline order

    The detector's verdict is repeated here so that an action is never read as a
    verdict: `detector_is_attack=False, action="block"` is a policy block of
    something the detector allowed, and both facts are on the same object.
    """
    action:             str
    blocked:            bool
    reasons:            tuple
    policy:             str
    policy_digest:      str | None = None
    detector_is_attack: bool | None = None
    zone:               str | None = None
    coverage_status:    str | None = None
    missing_required:   tuple = ()
    schema_version:     int = DECISION_SCHEMA_VERSION

    def to_dict(self) -> dict:
        """JSON-safe form, fixed key order."""
        return {
            "action":             self.action,
            "blocked":            self.blocked,
            "reasons":            list(self.reasons),
            "policy":             self.policy,
            "policy_digest":      self.policy_digest,
            "detector_is_attack": self.detector_is_attack,
            "zone":               self.zone,
            "coverage_status":    self.coverage_status,
            "missing_required":   list(self.missing_required),
            "schema_version":     self.schema_version,
        }


# ── Presets ───────────────────────────────────────────────────────────────────
# compat@1   today's behaviour, exactly. The default.
# strict@1   an allow that rests on a scan without a required layer is blocked;
#            a scanner failure is blocked.
# review@1   uncertainty and missing coverage are surfaced as `review`; a scanner
#            failure is an `error`.
#
# compat@1 has two forms that differ only in what a scanner failure does, because
# that is an existing, separately configured behaviour (the pre-flight guard's
# scan-failure mode).
# They share the id; their digests differ.

_REQUIRED: tuple[str, ...] = ("pair_classifier",)

_COMPAT_OPEN = Policy(name="compat", version=1, required_layers=_REQUIRED, on_uncertain="block",
                      on_missing_required="allow", on_scan_failure="allow", on_review="block")
_COMPAT_CLOSED = Policy(name="compat", version=1, required_layers=_REQUIRED, on_uncertain="block",
                        on_missing_required="allow", on_scan_failure="block", on_review="block")
STRICT = Policy(name="strict", version=1, required_layers=_REQUIRED, on_uncertain="block",
                on_missing_required="block", on_scan_failure="block", on_review="block")
REVIEW = Policy(name="review", version=1, required_layers=_REQUIRED, on_uncertain="review",
                on_missing_required="review", on_scan_failure="error", on_review="block")


def compat_policy(*, scan_failure: str) -> Policy:
    """`compat@1` in the form where a scanner failure is "allow" (open) or "block" (closed)."""
    if scan_failure == "allow":
        return _COMPAT_OPEN
    if scan_failure == "block":
        return _COMPAT_CLOSED
    raise ValueError('compat_policy(scan_failure=...) must be "allow" or "block"')


# ── The decision ──────────────────────────────────────────────────────────────

def _is_blocked(action: str, policy: Policy) -> bool:
    return action in ("block", "error") or (action == "review" and policy.on_review == "block")


def decide(result: Any, policy: Policy, *, scan_failed: bool = False) -> PolicyDecision:
    """
    Apply `policy` to one scan result. Pure.

    result       a ScanResult or LiteScanResult, or None when the scanner
                 produced nothing
    policy       the Policy to apply. Required and explicit: this function never
                 looks one up
    scan_failed  True when the scanner raised. A result passed with it is ignored

    The first matching row decides the action and `reasons[0]`:

        1  no result                              -> policy.on_scan_failure
        2  a stored label decided (bypassed)      -> block or allow, by the label
        3  clear block                            -> block
        4  uncertain, tiebreaker confirmed        -> block
        5  uncertain, no tiebreaker verdict       -> policy.on_uncertain
        6  allow verdict, required layer not run  -> policy.on_missing_required
        7  allow verdict, coverage not reported   -> policy.on_missing_required
        8  uncertain, cleared or allowed          -> allow
        9  allow                                  -> allow

    Rows 3 and 4 come before row 6: a detector that fired is positive evidence
    whatever else did not run, so missing coverage never downgrades a block.
    Rows 6 and 7 always put a coverage reason first: an allow that rests on an
    incomplete scan is never reported as a plain `detector_allow`.

    Reads nothing but its arguments, changes nothing, stores nothing.
    """
    if not isinstance(policy, Policy):
        raise TypeError("decide() requires an explicit Policy instance")

    if scan_failed or result is None:                                              # row 1
        action = policy.on_scan_failure
        return PolicyDecision(action=action, blocked=_is_blocked(action, policy), reasons=("scan_failed",),
                              policy=policy.id, policy_digest=policy.digest)

    is_attack = bool(getattr(result, "is_attack", False))
    zone = getattr(result, "zone", None)
    if not isinstance(zone, str) or zone not in ZONES:
        zone = "clear_block" if is_attack else "allow"
    decided_by = getattr(result, "decided_by", None)
    coverage = getattr(result, "coverage", None)
    status = getattr(coverage, "status", None)
    layers = getattr(coverage, "layers", None) if coverage is not None else None
    if not isinstance(layers, dict):
        coverage, status, layers = None, None, None

    def done(action: str, reasons: list, missing: tuple = ()) -> PolicyDecision:
        return PolicyDecision(
            action=action, blocked=_is_blocked(action, policy), reasons=tuple(reasons),
            policy=policy.id, policy_digest=policy.digest, detector_is_attack=is_attack, zone=zone,
            coverage_status=status if isinstance(status, str) else None, missing_required=missing,
        )

    if status == "bypassed":                                                       # row 2
        return done("block", ["labelled_attack"]) if is_attack else done("allow", ["labelled_safe"])

    missing: tuple = ()
    context: list = []
    if layers is not None:
        required = policy.required_layers
        missing = tuple(name for name in required if layers.get(name) != "ok")
        degraded = disabled = False
        for name, state in layers.items():
            if state == "ok" or name in required:
                continue
            if state == "disabled":
                disabled = True
            elif state in _NOT_RUN:
                degraded = True
        if degraded:
            context.append("degraded_optional_layer")
        if disabled:
            context.append("layer_disabled_by_caller")

    if zone in _BLOCKING_ZONES:
        if zone == "clear_block":                                                  # row 3
            action, first = "block", "detector_block"
        elif decided_by == "tiebreaker":                                           # row 4
            action, first = "block", "tiebreaker_confirmed"
        else:                                                                      # row 5
            action, first = policy.on_uncertain, "detector_uncertain"
        return done(action, [first, *(["missing_required_layer"] if missing else []), *context], missing)

    if zone == "uncertain_allow":
        own = "tiebreaker_cleared" if decided_by == "tiebreaker" else "uncertain_allowed_by_config"
    else:
        own = "detector_allow"

    if missing:                                                                    # row 6
        return done(policy.on_missing_required, ["missing_required_layer", *context, own], missing)
    if layers is None:                                                             # row 7
        return done(policy.on_missing_required, ["coverage_unknown", own])
    return done("allow", [own, *context])                                          # rows 8, 9


def invalid_config_decision(result: Any = None) -> PolicyDecision:
    """
    The fixed decision for "no valid policy exists". Pure.

    Always a blocking `error` with the single reason `invalid_policy_config`, the
    policy id "invalid" and no digest. When a scan result is given (the CLI still
    scans, to report what the detector found), its verdict, zone and coverage
    status are copied across; nothing else about it is consulted.
    """
    is_attack = zone = status = None
    if result is not None:
        is_attack = bool(getattr(result, "is_attack", False))
        zone = getattr(result, "zone", None)
        zone = zone if isinstance(zone, str) and zone in ZONES else None
        status = getattr(getattr(result, "coverage", None), "status", None)
        status = status if isinstance(status, str) else None
    return PolicyDecision(action="error", blocked=True, reasons=("invalid_policy_config",),
                          policy=INVALID_POLICY_ID, policy_digest=None, detector_is_attack=is_attack,
                          zone=zone, coverage_status=status)


# ── Process configuration ─────────────────────────────────────────────────────
# Separate from decide(), which takes its policy as an argument.
#
#   set_policy(p)                an explicit Policy, used as given
#   FIE_POLICY=strict|review     the preset of that name
#   FIE_POLICY=compat, or unset  compat@1, in the form matching the pre-flight
#                                guard's existing fail-secure flag
#   anything else                PolicyConfigError. Never a fallback
#
# FIE_POLICY is read once, on the first active_policy() call, and the SELECTION
# is cached for the life of the process, including a failure. Changing the
# variable afterwards does nothing; there is no hot reload. To change it:
# restart, or set_policy(), or reset_policy().

_SELECTABLE: dict[str, str] = {"": "compat", "compat": "compat", "strict": "strict", "review": "review"}
_INVALID = "<invalid>"

_config_lock = threading.Lock()
_explicit_policy: Policy | None = None
_selection: str | None = None          # None: not resolved yet
_config_error_logged: bool = False


def _resolve_selection() -> str:
    global _selection, _config_error_logged
    with _config_lock:
        if _selection is None:
            raw = (os.environ.get("FIE_POLICY") or "").strip().lower()
            _selection = _SELECTABLE.get(raw, _INVALID)
            if _selection == _INVALID and not _config_error_logged:
                _config_error_logged = True
                # The value is deliberately not logged.
                logger.error(
                    "policy configuration invalid: FIE_POLICY is not one of compat, strict, review. "
                    "No policy is in force: every guarded call returns a blocking error "
                    "(reason=invalid_policy_config) until the process is restarted with a valid value "
                    "or set_policy()/reset_policy() is called. Logged once per process."
                )
        return _selection


def active_policy() -> Policy:
    """
    The policy this process applies when a caller does not pass one.

    Returns the exact Policy object in force at the moment of the call. Raises
    PolicyConfigError when FIE_POLICY names no known policy.

    With the `compat` selection the returned form follows the pre-flight guard's
    own fail-secure flag (`fie.preflight._FAIL_SECURE`), read at each call. That
    flag is set from the guard's scan-failure-mode environment variable, once,
    when `fie.preflight` is imported, as it always has been; this module does not
    read that variable. An
    explicit policy, `strict` and `review` state their own scanner-failure
    setting and do not consult the flag.
    """
    explicit = _explicit_policy
    if explicit is not None:
        return explicit
    selection = _selection if _selection is not None else _resolve_selection()
    if selection == "strict":
        return STRICT
    if selection == "review":
        return REVIEW
    if selection == "compat":
        from fie import preflight as _preflight
        return _COMPAT_CLOSED if _preflight._FAIL_SECURE else _COMPAT_OPEN
    raise PolicyConfigError()


def set_policy(policy: Policy) -> None:
    """
    Use `policy` for this process, from the next call on, in place of FIE_POLICY.

    Accepts only a Policy instance, so a policy can never be selected by a string
    that arrived from somewhere else. Also clears the effect of an invalid
    FIE_POLICY.
    """
    global _explicit_policy
    if not isinstance(policy, Policy):
        raise TypeError("set_policy() requires a Policy instance")
    with _config_lock:
        _explicit_policy = policy


def reset_policy() -> None:
    """
    Forget set_policy() and the cached FIE_POLICY selection. The next
    active_policy() call reads FIE_POLICY again. For tests, and for recovering
    from an invalid value without a restart.
    """
    global _explicit_policy, _selection
    with _config_lock:
        _explicit_policy = None
        _selection = None
