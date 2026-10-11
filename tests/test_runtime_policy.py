"""
Runtime policy (WP-004).

Detection says what the detectors concluded. Coverage says which of them ran.
Policy says what to do about it. These tests pin the policy layer:

  * `fie.policy.decide` is a pure function of (scan result, Policy, scan_failed);
  * the decision table of PLAN_004 section 6, cell by cell;
  * the process configuration lifecycle (FIE_POLICY, set_policy, reset_policy);
  * `preflight_check`, the one enforcement point, and the CLI, which displays;
  * and, above all, that nothing the detector returns has moved.

Reference values for today's behaviour were captured from the unmodified code
at commit 924c70d (2026-10-10) and are literals here or in the WP-003 contract
file. `compat@1`, the default policy, must reproduce them exactly.

WP-004 enforces policy in `preflight_check` only. Code that calls
`scan_prompt()` directly is NOT covered; one test below documents that gap.
"""
from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

os.environ.setdefault("FIE_NO_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scan_result_contract import (  # noqa: E402  - shared reference values and loader fixtures
    ALL_LAYERS,
    PLANTED,
    PLANTED_EMAIL,
    PLANTED_MESSAGE,
    PLANTED_PATH,
    PLANTED_TOKEN,
    PRE_LOADED,
    PRE_MISSING,
    PROMPTS,
    TO_DICT_KEYS,
    _make_classifier_unavailable,
    _pair_module_restored,
    _tiebreaker_down,
    _verdict,
    _expected,
)

ACTIONS = ("allow", "review", "block", "error")
REASONS = {
    "detector_block", "detector_uncertain", "tiebreaker_confirmed", "labelled_attack", "labelled_safe",
    "tiebreaker_cleared", "uncertain_allowed_by_config", "detector_allow", "missing_required_layer",
    "coverage_unknown", "degraded_optional_layer", "layer_disabled_by_caller", "scan_failed",
    "invalid_policy_config", "warn_only",
}
DETECTOR_BLOCK_REASONS = {"detector_block", "detector_uncertain", "tiebreaker_confirmed", "labelled_attack"}
DECISION_KEYS = ("action", "blocked", "reasons", "policy", "policy_digest", "detector_is_attack", "zone",
                 "coverage_status", "missing_required", "schema_version")
GUARD_KEYS = ("blocked", "attack_type", "confidence", "layers_fired", "refusal_message", "scan_failed",
              "zone", "coverage_status", "schema_version", "action", "policy", "reasons")
OLD_GUARD_FIELDS = GUARD_KEYS[:9]
PRESETS = ("compat", "strict", "review")
STDERR_NOTICE = ("fie: policy configuration is invalid (FIE_POLICY is not a known policy name); "
                 "no policy was applied")
PLANTED_ENV = f"strict {PLANTED_PATH} {PLANTED_TOKEN} {PLANTED_EMAIL}"
DEFAULT = "<default>"

# ── GuardResult values captured from the unmodified code (924c70d) ───────────
# (blocked, attack_type, confidence, layers_fired, refusal, scan_failed, zone, coverage_status)
PRE = {
    ("loaded", False): {
        "benign_capital":    (False, "", 0.0, [], "", False, "allow", "full"),
        "benign_python":     (False, "", 0.0, [], "", False, "allow", "full"),
        "copyright":         (True, "COPYRIGHT_REPRODUCTION", 0.72, ["copyright"], DEFAULT, False, "clear_block", "full"),
        "dan":               (True, "JAILBREAK_ATTEMPT", 0.6813, ["regex"], DEFAULT, False, "clear_block", "full"),
        "harmful_pair_only": (True, "PROMPT_EXTRACTION", 0.7006, ["regex"], DEFAULT, False, "uncertain_block", "full"),
        "injection":         (True, "PROMPT_EXTRACTION", 0.82, ["regex"], DEFAULT, False, "clear_block", "full"),
        "many_shot":         (True, "DIRECT_HARMFUL_REQUEST", 0.85, ["direct_harm"], DEFAULT, False, "clear_block", "full"),
        "soft_harm":         (True, "JAILBREAK_ATTEMPT", 0.95, ["pair_classifier"], DEFAULT, False, "clear_block", "full"),
    },
    ("missing", False): {
        "benign_capital":    (False, "", 0.0, [], "", False, "allow", "partial"),
        "benign_python":     (False, "", 0.0, [], "", False, "allow", "partial"),
        "copyright":         (True, "COPYRIGHT_REPRODUCTION", 0.72, ["copyright"], DEFAULT, False, "clear_block", "partial"),
        "dan":               (True, "JAILBREAK_ATTEMPT", 0.82, ["regex"], DEFAULT, False, "clear_block", "partial"),
        "harmful_pair_only": (True, "PROMPT_EXTRACTION", 0.82, ["regex"], DEFAULT, False, "clear_block", "partial"),
        "injection":         (True, "PROMPT_EXTRACTION", 0.82, ["regex"], DEFAULT, False, "clear_block", "partial"),
        "many_shot":         (True, "DIRECT_HARMFUL_REQUEST", 0.85, ["direct_harm"], DEFAULT, False, "clear_block", "partial"),
        "soft_harm":         (False, "", 0.0, [], "", False, "allow", "partial"),
    },
}
# Warn-only, as captured: a detected attack keeps its fields, is not blocked, and has no refusal text.
for _state in ("loaded", "missing"):
    PRE[(_state, True)] = {
        name: (row if not row[0] else (False, row[1], row[2], row[3], "", row[5], row[6], row[7]))
        for name, row in PRE[(_state, False)].items()
    }
PRE_EMPTY = (False, "", 0.0, [], "", False, None, None)
PRE_SCAN_RAISES = {   # fail_secure -> row; identical with warn-only on and off
    False: (False, "", 0.0, [], "", True, None, None),
    True:  (True, "SCAN_FAILED", 1.0, ["preflight_fail_secure"], DEFAULT, True, None, None),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _old_fields(guard) -> tuple:
    import fie.preflight as pf

    refusal = DEFAULT if guard.refusal_message == pf._DEFAULT_REFUSAL else guard.refusal_message
    return (guard.blocked, guard.attack_type, round(float(guard.confidence), 4), sorted(guard.layers_fired),
            refusal, guard.scan_failed, guard.zone, guard.coverage_status)


def _preset(name: str, *, fail_secure: bool = False):
    import fie.policy as policy

    if name == "compat":
        return policy.compat_policy(scan_failure="block" if fail_secure else "allow")
    return {"strict": policy.STRICT, "review": policy.REVIEW}[name]


def _result(zone="allow", decided_by="pipeline", *, clf="ok", other=None, bypassed=False, coverage=True):
    """A synthetic ScanResult in the WP-003 shape. `other` degrades or disables the regex layer."""
    from fie.adversarial import ScanCoverage, ScanResult

    is_attack = zone in ("uncertain_block", "clear_block")
    cov = None
    if bypassed:
        cov = ScanCoverage(status="bypassed", layers=dict.fromkeys(ALL_LAYERS, "bypassed"), classifier="bypassed",
                           optional={"meta_classifier": "bypassed", "tiebreaker": "not_needed",
                                     "translation": "not_needed"})
    elif coverage:
        layers = dict.fromkeys(ALL_LAYERS, "ok")
        layers["pair_classifier"] = clf
        if other:
            layers["regex"] = other
        cov = ScanCoverage(status="full" if set(layers.values()) == {"ok"} else "partial", layers=layers,
                           classifier=clf, optional={"meta_classifier": "ok", "tiebreaker": "not_needed",
                                                     "translation": "not_needed"})
    return ScanResult(is_attack=is_attack, attack_type="JAILBREAK_ATTEMPT" if is_attack else None, category=None,
                      confidence=0.9 if is_attack else 0.0, layers_fired=[], matched_text=None, mitigation="",
                      zone=zone, decided_by=decided_by, coverage=cov)


VERDICTS = [   # every valid (zone, decided_by) pair a scan can produce
    ("allow", "pipeline"), ("clear_block", "pipeline"),
    ("uncertain_block", "fail_secure"), ("uncertain_block", "tiebreaker"),
    ("uncertain_allow", "tiebreaker"), ("uncertain_allow", "config"),
]
CLF_STATES = ["ok", "unavailable_dependency", "unavailable_model", "unavailable_load_failed", "error", "timeout",
              "disabled"]
OTHER_STATES = [None, "error", "timeout", "unavailable_model", "disabled"]
DETECTOR_REASON = {
    ("allow", "pipeline"): "detector_allow", ("clear_block", "pipeline"): "detector_block",
    ("uncertain_block", "fail_secure"): "detector_uncertain", ("uncertain_block", "tiebreaker"): "tiebreaker_confirmed",
    ("uncertain_allow", "tiebreaker"): "tiebreaker_cleared", ("uncertain_allow", "config"): "uncertain_allowed_by_config",
}
PRESET_FIELDS = {   # written out here; the implementation's presets are compared with this
    "compat": dict(on_uncertain="block", on_missing_required="allow", on_review="block"),
    "strict": dict(on_uncertain="block", on_missing_required="block", on_scan_failure="block", on_review="block"),
    "review": dict(on_uncertain="review", on_missing_required="review", on_scan_failure="error", on_review="block"),
}


def _oracle(preset: str, zone: str, decided_by: str, clf: str, other, coverage: bool) -> tuple:
    """The decision table of PLAN_004 section 6.1, written independently of fie/policy.py."""
    cfg = PRESET_FIELDS[preset]
    missing = coverage and clf != "ok"
    context = []
    if other in ("error", "timeout", "unavailable_model"):
        context.append("degraded_optional_layer")
    if other == "disabled":
        context.append("layer_disabled_by_caller")
    if zone == "clear_block":
        action, first = "block", "detector_block"
    elif zone == "uncertain_block" and decided_by == "tiebreaker":
        action, first = "block", "tiebreaker_confirmed"
    elif zone == "uncertain_block":
        action, first = cfg["on_uncertain"], "detector_uncertain"
    elif missing or not coverage:
        action = cfg["on_missing_required"]
        first = "missing_required_layer" if coverage else "coverage_unknown"
        return action, (first, *context, DETECTOR_REASON[(zone, decided_by)]), ("pair_classifier",) if missing else ()
    else:
        return "allow", (DETECTOR_REASON[(zone, decided_by)], *context), ()
    reasons = (first, *(["missing_required_layer"] if missing else []), *context)
    return action, reasons, ("pair_classifier",) if missing else ()


def _blocked(action: str, on_review: str = "block") -> bool:
    return action in ("block", "error") or (action == "review" and on_review == "block")


def _cli(command: str, prompt: str, capsys, output="json", quiet=False):
    import fie.__main__ as cli

    fn = cli._cmd_detect if command == "detect" else cli._cmd_explain
    code = fn(argparse.Namespace(prompt=prompt, output=output, threshold=0.5, quiet=quiet))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class _EnvTrap(dict):
    """Stands in for os.environ and fails the test on any read."""

    def _trap(self, *args, **kwargs):
        raise AssertionError("decide() read the process environment")

    __getitem__ = __contains__ = get = keys = items = values = __iter__ = _trap


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def no_outbound_network(monkeypatch):
    import socket

    attempts: list[str] = []
    real_connect = socket.socket.connect

    def guarded(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in ("127.0.0.1", "::1", "localhost"):
            attempts.append(str(host))
            raise OSError("outbound network is not allowed in the runtime-policy tests")
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield attempts
    assert attempts == [], f"a network connection was opened: {attempts}"


def _reset_policy_state() -> None:
    try:
        import fie.policy as policy
    except ImportError:
        return
    policy.reset_policy()
    if hasattr(policy, "_config_error_logged"):
        policy._config_error_logged = False


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Default configuration, block mode, no tiebreaker, no translator, no labels, no cached results."""
    import fie.adversarial as adv
    import fie.feedback_store as fb
    import fie.llama_guard as lg
    import fie.multilingual as ml
    import fie.preflight as pf

    for name in ("FIE_POLICY", "FIE_SCAN_FAILURE_MODE", "FIE_UNCERTAIN_ALLOW", "FIE_DISABLE_META",
                 "FIE_PAIR_VERSION", "FIE_EMBED_BACKEND", "PREFLIGHT_BLOCK_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FIE_NO_AUTO_DOWNLOAD", "1")
    monkeypatch.setattr(fb, "_LOCAL_PATH", tmp_path / "flagged_events.jsonl")
    monkeypatch.setattr(fb, "_KNOWN_ATTACK_HASHES", set())
    monkeypatch.setattr(fb, "_WHITELIST_HASHES", set())
    monkeypatch.setattr(lg, "query_llama_guard", _tiebreaker_down)
    monkeypatch.setattr(ml, "translate_to_english", lambda text, timeout=3.0: None)
    monkeypatch.setattr(pf, "_FAIL_SECURE", False)
    monkeypatch.setattr(pf, "_get_block_enabled", lambda: True)
    adv._scan_cache._cache.clear()
    ml._TRANSLATION_CACHE.clear()
    _reset_policy_state()
    yield
    adv._scan_cache._cache.clear()
    ml._TRANSLATION_CACHE.clear()
    _reset_policy_state()


@pytest.fixture(scope="module")
def loaded():
    import fie.adversarial as adv

    adv.warmup()
    state = adv.health()
    if not state["pair_classifier"]["loaded"] or not state["meta_classifier"]["loaded"]:
        pytest.skip("the PAIR classifier or the meta-classifier is not loaded")
    return adv


@pytest.fixture
def missing(loaded, monkeypatch, tmp_path):
    """The classifier is unavailable because its package cannot be imported (a base install)."""
    with _pair_module_restored() as pair:
        _make_classifier_unavailable(pair, monkeypatch, tmp_path, "dependency")
        yield


@pytest.fixture
def scan_spy(monkeypatch):
    """Counts calls to fie.adversarial.scan_prompt without changing what it returns."""
    import fie.adversarial as adv

    calls: list[str] = []
    real = adv.scan_prompt

    def spy(prompt, *args, **kwargs):
        calls.append(prompt)
        return real(prompt, *args, **kwargs)

    monkeypatch.setattr(adv, "scan_prompt", spy)
    return calls


@pytest.fixture
def scanner_raises(monkeypatch):
    import fie.adversarial as adv

    def _crash(*args, **kwargs):
        raise RuntimeError(PLANTED_MESSAGE)

    monkeypatch.setattr(adv, "scan_prompt", _crash)


@pytest.fixture
def warn_only(monkeypatch):
    import fie.preflight as pf

    monkeypatch.setattr(pf, "_get_block_enabled", lambda: False)


def _use(monkeypatch, preset: str) -> None:
    monkeypatch.setenv("FIE_POLICY", preset)
    _reset_policy_state()


# ══ 1. The module: closed sets, validation, presets ══════════════════════════

def test_public_names_exist_and_are_exported():
    import fie
    import fie.policy as policy

    for name in ("Policy", "PolicyDecision", "decide", "invalid_config_decision", "compat_policy", "STRICT",
                 "REVIEW", "active_policy", "set_policy", "reset_policy", "PolicyConfigError"):
        assert hasattr(policy, name), name
    assert {"Policy", "PolicyDecision", "decide"} <= set(fie.__all__)
    assert fie.decide is policy.decide and fie.Policy is policy.Policy
    assert tuple(policy.ACTIONS) == ACTIONS
    assert set(policy.REASONS) == REASONS and len(policy.REASONS) == 15


def test_policy_module_imports_nothing_it_should_not():
    tree = ast.parse((ROOT / "fie" / "policy.py").read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    assert found <= {"__future__", "dataclasses", "hashlib", "json", "logging", "os", "re", "threading", "typing",
                     "fie"}, found
    source = (ROOT / "fie" / "policy.py").read_text(encoding="utf-8")
    assert "FIE_SCAN_FAILURE_MODE" not in source, "fie.policy must never read that variable; it reads the flag"


def test_shipped_presets_are_pinned():
    import fie.policy as policy

    open_form, closed_form = policy.compat_policy(scan_failure="allow"), policy.compat_policy(scan_failure="block")
    for p, name, scan_failure in ((open_form, "compat", "allow"), (closed_form, "compat", "block"),
                                  (policy.STRICT, "strict", "block"), (policy.REVIEW, "review", "error")):
        want = {"on_scan_failure": scan_failure, **PRESET_FIELDS[name]}
        assert (p.name, p.version, p.required_layers) == (name, 1, ("pair_classifier",))
        assert {k: getattr(p, k) for k in want} == want
        assert p.id == f"{name}@1" and re.fullmatch(r"[0-9a-f]{8}", p.digest)
    assert open_form.id == closed_form.id == "compat@1" and open_form.digest != closed_form.digest
    assert policy.compat_policy(scan_failure="allow") is open_form, "the presets are prebuilt constants"
    # Version discipline: these digests change only if a preset's behaviour does.
    assert (open_form.digest, closed_form.digest, policy.STRICT.digest, policy.REVIEW.digest) == PINNED_DIGESTS
    with pytest.raises(ValueError):
        policy.compat_policy(scan_failure="error")


PINNED_DIGESTS = ("989f7491", "f5fea0f8", "4cb035e4", "31e8f0cd")   # compat open, compat closed, strict, review


@pytest.mark.parametrize("bad", [
    dict(name="Strict"), dict(name="invalid"), dict(name=""), dict(name="a b"), dict(name=PLANTED_PATH),
    dict(name="x" * 33), dict(version=0), dict(version=True), dict(version="1"),
    dict(required_layers=("pair",)), dict(required_layers="pair_classifier"),
    dict(on_uncertain="allow"), dict(on_missing_required="deny"), dict(on_scan_failure="review"),
    dict(on_review="review"), dict(on_review=None),
])
def test_policy_rejects_invalid_values(bad):
    from fie.policy import Policy

    good = dict(name="custom", version=1, required_layers=("pair_classifier",), on_uncertain="block",
                on_missing_required="block", on_scan_failure="block", on_review="block")
    Policy(**good)
    with pytest.raises(ValueError) as err:
        Policy(**{**good, **bad})
    for needle in PLANTED:
        assert needle not in str(err.value)


def test_policy_is_an_immutable_value():
    import fie.policy as policy

    with pytest.raises(Exception):
        policy.STRICT.on_missing_required = "allow"
    twin = policy.Policy(name="strict", version=1, required_layers=("pair_classifier",), on_uncertain="block",
                         on_missing_required="block", on_scan_failure="block", on_review="block")
    assert twin == policy.STRICT and twin.digest == policy.STRICT.digest
    assert copy.deepcopy(policy.STRICT) == policy.STRICT


# ══ 2. decide() is pure ══════════════════════════════════════════════════════

def test_decide_requires_an_explicit_policy():
    from fie.policy import decide

    for bad in (None, "strict", {"name": "strict"}, 1):
        with pytest.raises(TypeError):
            decide(_result(), bad)
    with pytest.raises(TypeError):
        decide(_result())                                   # no default


def test_decide_reads_no_environment_and_no_process_configuration(monkeypatch):
    import fie.policy as policy
    import fie.preflight as pf

    cases = [(_result(z, d, clf=c), s) for (z, d) in VERDICTS for c in ("ok", "unavailable_model")
             for s in (False,)] + [(None, True)]
    before = {name: [policy.decide(r, _preset(name), scan_failed=s) for r, s in cases] for name in PRESETS}

    # Every process-level knob now points somewhere else.
    monkeypatch.setenv("FIE_POLICY", "review")
    monkeypatch.setenv("FIE_SCAN_FAILURE_MODE", "closed")
    monkeypatch.setenv("FIE_UNCERTAIN_ALLOW", "1")
    monkeypatch.setenv("PREFLIGHT_BLOCK_ENABLED", "false")
    monkeypatch.setattr(pf, "_FAIL_SECURE", True)
    monkeypatch.setattr(pf, "_get_block_enabled", lambda: False)
    policy.set_policy(policy.STRICT)
    monkeypatch.setattr(os, "environ", _EnvTrap())
    after = {name: [policy.decide(r, _preset(name), scan_failed=s) for r, s in cases] for name in PRESETS}
    assert after == before


def test_decide_does_not_touch_the_result_or_the_cache(loaded):
    import fie.policy as policy

    r = loaded.scan_prompt(PROMPTS["injection"], use_llama_guard=False)
    snapshot, attrs = r.to_dict(), set(vars(r))
    for name in PRESETS:
        d = policy.decide(r, _preset(name))
        assert d.policy == _preset(name).id
    assert r.to_dict() == snapshot and set(vars(r)) == attrs
    assert tuple(r.to_dict()) == TO_DICT_KEYS
    for cached, _ts in loaded._scan_cache._cache.values():
        assert not isinstance(cached, policy.PolicyDecision)
        assert not any(isinstance(v, policy.PolicyDecision) for v in vars(cached).values())
    assert loaded.scan_prompt(PROMPTS["injection"], use_llama_guard=False) is r


def test_decision_names_exactly_the_policy_it_was_given():
    import fie.policy as policy

    custom = policy.Policy(name="acme-gateway", version=7, required_layers=("pair_classifier", "regex"),
                           on_uncertain="review", on_missing_required="error", on_scan_failure="error",
                           on_review="allow")
    for p in (policy.compat_policy(scan_failure="allow"), policy.compat_policy(scan_failure="block"),
              policy.STRICT, policy.REVIEW, custom):
        d = policy.decide(_result(), p)
        assert (d.policy, d.policy_digest) == (p.id, p.digest)
    d = policy.decide(_result("allow", clf="ok", other="error"), custom)
    assert (d.action, d.reasons[0], d.missing_required) == ("error", "missing_required_layer", ("regex",))
    d = policy.decide(_result("uncertain_block", "fail_secure"), custom)
    assert (d.action, d.blocked) == ("review", False), "on_review='allow' leaves a review unblocked"


# ══ 3. The decision table ════════════════════════════════════════════════════

CROSS = [(p, z, d, c, o, cov) for p in PRESETS for (z, d) in VERDICTS for c in CLF_STATES for o in OTHER_STATES
         for cov in (True,)] + [(p, z, d, "ok", None, False) for p in PRESETS for (z, d) in VERDICTS]


def test_decision_table_cross_product():
    """630 scan shapes + 18 legacy results, each against the independently written table."""
    from fie.policy import decide

    checked = 0
    for preset, zone, decided_by, clf, other, coverage in CROSS:
        result = _result(zone, decided_by, clf=clf, other=other, coverage=coverage)
        for fail_secure in ((False, True) if preset == "compat" else (False,)):
            p = _preset(preset, fail_secure=fail_secure)
            got = decide(result, p)
            action, reasons, missing = _oracle(preset, zone, decided_by, clf, other, coverage)
            where = (preset, zone, decided_by, clf, other, coverage)
            assert (got.action, got.reasons, got.missing_required) == (action, reasons, missing), where
            assert got.blocked == _blocked(action), where
            assert (got.detector_is_attack, got.zone) == (result.is_attack, zone), where
            assert got.coverage_status == (result.coverage.status if coverage else None), where
            checked += 1
    assert checked == (3 * 6 * 7 * 5 + 18) + (6 * 7 * 5 + 6)


# PLAN_004 section 6.3, row by row: (zone, decided_by, kwargs) -> action under compat / strict / review
TABLE_6_3 = [
    ("full coverage, allow",               ("allow", "pipeline", {}),                               ("allow", "allow", "allow")),
    ("full coverage, clear block",         ("clear_block", "pipeline", {}),                         ("block", "block", "block")),
    ("full coverage, uncertain",           ("uncertain_block", "fail_secure", {}),                  ("block", "block", "review")),
    ("classifier missing (dependency)",    ("allow", "pipeline", {"clf": "unavailable_dependency"}), ("allow", "block", "review")),
    ("classifier missing (model)",         ("allow", "pipeline", {"clf": "unavailable_model"}),      ("allow", "block", "review")),
    ("classifier missing (load failed)",   ("allow", "pipeline", {"clf": "unavailable_load_failed"}), ("allow", "block", "review")),
    ("classifier missing, detector block", ("clear_block", "pipeline", {"clf": "unavailable_model"}), ("block", "block", "block")),
    ("classifier timeout, allow",          ("allow", "pipeline", {"clf": "timeout"}),               ("allow", "block", "review")),
    ("classifier error, allow",            ("allow", "pipeline", {"clf": "error"}),                 ("allow", "block", "review")),
    ("other layer degraded, allow",        ("allow", "pipeline", {"other": "error"}),               ("allow", "allow", "allow")),
    ("caller disabled another layer",      ("allow", "pipeline", {"other": "disabled"}),            ("allow", "allow", "allow")),
    ("caller disabled the classifier",     ("allow", "pipeline", {"clf": "disabled"}),              ("allow", "block", "review")),
    ("labelled false positive",            ("allow", "feedback_override", {"bypassed": True}),      ("allow", "allow", "allow")),
    ("labelled attack",                    ("clear_block", "feedback_override", {"bypassed": True}), ("block", "block", "block")),
    ("tiebreaker confirmed",               ("uncertain_block", "tiebreaker", {}),                   ("block", "block", "block")),
    ("tiebreaker cleared",                 ("uncertain_allow", "tiebreaker", {}),                   ("allow", "allow", "allow")),
    ("FIE_UNCERTAIN_ALLOW",                ("uncertain_allow", "config", {}),                       ("allow", "allow", "allow")),
    ("legacy result, coverage unknown",    ("allow", "pipeline", {"coverage": False}),              ("allow", "block", "review")),
]


@pytest.mark.parametrize("label,spec,actions", TABLE_6_3, ids=[row[0] for row in TABLE_6_3])
def test_decision_table_section_6_3(label, spec, actions):
    from fie.policy import decide

    zone, decided_by, kwargs = spec
    for preset, action in zip(PRESETS, actions):
        d = decide(_result(zone, decided_by, **kwargs), _preset(preset))
        assert d.action == action, (label, preset)
        assert d.blocked == (action != "allow"), (label, preset)


def test_reasons_of_the_named_rows():
    from fie.policy import decide

    strict = _preset("strict")
    assert decide(_result("allow", clf="unavailable_model"), _preset("compat")).reasons == (
        "missing_required_layer", "detector_allow")
    assert decide(_result("clear_block", clf="unavailable_model", other="timeout"), strict).reasons == (
        "detector_block", "missing_required_layer", "degraded_optional_layer")
    assert decide(_result("allow", other="disabled"), strict).reasons == ("detector_allow", "layer_disabled_by_caller")
    assert decide(_result("allow", "feedback_override", bypassed=True), strict).reasons == ("labelled_safe",)
    assert decide(_result("clear_block", "feedback_override", bypassed=True), strict).reasons == ("labelled_attack",)
    assert decide(_result("allow", coverage=False), strict).reasons == ("coverage_unknown", "detector_allow")
    assert decide(_result("uncertain_allow", "config", clf="error"), strict).reasons == (
        "missing_required_layer", "uncertain_allowed_by_config")
    labelled = decide(_result("allow", "feedback_override", bypassed=True), strict)
    assert labelled.missing_required == () and labelled.coverage_status == "bypassed"


@pytest.mark.parametrize("preset,fail_secure,action,blocked", [
    ("compat", False, "allow", False), ("compat", True, "block", True),
    ("strict", False, "block", True), ("strict", True, "block", True),
    ("review", False, "error", True), ("review", True, "error", True),
])
def test_scanner_failure_row(preset, fail_secure, action, blocked):
    from fie.policy import decide

    p = _preset(preset, fail_secure=fail_secure)
    for result in (None, _result("clear_block")):
        d = decide(result, p, scan_failed=True)
        assert (d.action, d.blocked, d.reasons) == (action, blocked, ("scan_failed",))
        assert (d.detector_is_attack, d.zone, d.coverage_status, d.missing_required) == (None, None, None, ())
    assert decide(None, p).reasons == ("scan_failed",), "no result is a scanner failure"


def test_invariants_over_every_input():
    """Section 6.1, invariants 1 to 5."""
    from fie.policy import decide

    policies = [_preset("compat"), _preset("compat", fail_secure=True), _preset("strict"), _preset("review")]
    shapes = [_result(z, d, clf=c, other=o) for (z, d) in VERDICTS for c in CLF_STATES for o in OTHER_STATES]
    shapes += [_result(z, d, coverage=False) for (z, d) in VERDICTS]
    shapes += [_result("allow", "feedback_override", bypassed=True),
               _result("clear_block", "feedback_override", bypassed=True)]
    for p, r in itertools.product(policies, shapes):
        d = decide(r, p)
        cov = r.coverage
        # 1. a detector block is never downgraded by missing coverage
        if r.zone == "clear_block" or (r.zone == "uncertain_block" and r.decided_by == "tiebreaker"):
            assert d.action == "block" and d.blocked
        # 2. missing required coverage never yields a bare allow
        required_missing = cov is None or (cov.status != "bypassed" and cov.layers["pair_classifier"] != "ok")
        if not r.is_attack and required_missing:
            assert d.reasons[0] in ("missing_required_layer", "coverage_unknown")
            assert d.reasons[0] != "detector_allow"
        if d.reasons[0] == "detector_allow":
            assert cov is not None and cov.layers["pair_classifier"] == "ok"
        # 3. the enforcement bit follows the action
        assert d.action in ACTIONS and d.blocked == _blocked(d.action, p.on_review)
        # 4. reasons: non-empty, closed set, no duplicates
        assert d.reasons and set(d.reasons) <= REASONS and len(set(d.reasons)) == len(d.reasons)
        # 5. these two are never produced by decide()
        assert "invalid_policy_config" not in d.reasons and "warn_only" not in d.reasons
        assert tuple(d.to_dict()) == DECISION_KEYS and json.loads(json.dumps(d.to_dict())) == d.to_dict()


def test_decide_accepts_a_lite_result():
    from fie._lite import scan_prompt_lite
    from fie.policy import decide

    safe, attack = scan_prompt_lite(PROMPTS["benign_capital"]), scan_prompt_lite(PROMPTS["injection"])
    for preset, action in (("compat", "allow"), ("strict", "block"), ("review", "review")):
        d = decide(safe, _preset(preset))
        assert (d.action, d.reasons[0], d.missing_required) == (action, "missing_required_layer", ("pair_classifier",))
        assert "layer_disabled_by_caller" in d.reasons
        assert decide(attack, _preset(preset)).action == "block"


def test_baseline_replay_matches_plan_section_6_4():
    """Every BL-0002 record, replayed through decide(): the cost of each preset, in numbers."""
    from fie.policy import decide

    records = ROOT / "evals" / "baselines" / "BL-0002_pair-v6.3b_fie-70de7053" / "records"
    want = {
        "canonical": {"compat": {"allow": 328, "block": 1688}, "strict": {"allow": 328, "block": 1688},
                      "review": {"allow": 328, "review": 294, "block": 1394}},
        "lite":      {"compat": {"allow": 1768, "block": 80}, "strict": {"block": 1848},
                      "review": {"review": 1769, "block": 79}},
    }
    groups = {"canonical": sorted(records.glob("std.*.jsonl")), "lite": [records / "lite.std.jsonl"]}
    for group, files in groups.items():
        counts = {name: {} for name in PRESETS}
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                clf = "unavailable_dependency" if "pair_classifier" in rec["degraded"] else "ok"
                decided_by = "fail_secure" if rec["zone"] == "uncertain_block" else "pipeline"
                result = _result(rec["zone"], decided_by, clf=clf)
                assert result.is_attack == rec["flagged"]
                for name in PRESETS:
                    action = decide(result, _preset(name)).action
                    counts[name][action] = counts[name].get(action, 0) + 1
        assert counts == want[group], group


# ══ 4. Process configuration and the effective policy ════════════════════════

def test_default_selection_is_compat(monkeypatch):
    import fie.policy as policy

    assert policy.active_policy() is policy.compat_policy(scan_failure="allow")
    for raw, want in (("compat", "compat@1"), (" Strict ", "strict@1"), ("REVIEW", "review@1"), ("", "compat@1")):
        monkeypatch.setenv("FIE_POLICY", raw)
        policy.reset_policy()
        assert policy.active_policy().id == want


def test_fie_policy_is_read_once(monkeypatch):
    import fie.policy as policy

    monkeypatch.setenv("FIE_POLICY", "strict")
    policy.reset_policy()
    assert policy.active_policy() is policy.STRICT
    monkeypatch.setenv("FIE_POLICY", "review")
    assert policy.active_policy() is policy.STRICT, "no hot reload: valid -> valid"
    monkeypatch.setenv("FIE_POLICY", PLANTED_ENV)
    assert policy.active_policy() is policy.STRICT, "no hot reload: valid -> invalid"
    monkeypatch.delenv("FIE_POLICY")
    assert policy.active_policy() is policy.STRICT
    policy.reset_policy()
    assert policy.active_policy().id == "compat@1", "reset_policy() re-reads the environment"


def test_set_policy_precedence_and_types(monkeypatch):
    import fie.policy as policy

    monkeypatch.setenv("FIE_POLICY", "review")
    policy.reset_policy()
    policy.set_policy(policy.STRICT)
    assert policy.active_policy() is policy.STRICT, "set_policy outranks FIE_POLICY"
    for bad in ("strict", None, {"name": "strict"}):
        with pytest.raises(TypeError):
            policy.set_policy(bad)
    assert policy.active_policy() is policy.STRICT
    policy.reset_policy()
    assert policy.active_policy() is policy.REVIEW


def test_concurrent_first_resolution_reads_the_environment_once(monkeypatch):
    import fie.policy as policy

    monkeypatch.setenv("FIE_POLICY", "strict")
    policy.reset_policy()
    reads = []
    real_get = os.environ.get

    def counting_get(key, default=None):
        if key == "FIE_POLICY":
            reads.append(threading.get_ident())
        return real_get(key, default)

    monkeypatch.setattr(os.environ, "get", counting_get, raising=False)
    start = threading.Barrier(16)

    def resolve(_):
        start.wait()
        return policy.active_policy()

    with ThreadPoolExecutor(max_workers=16) as pool:
        got = list(pool.map(resolve, range(16)))
    assert all(p is policy.STRICT for p in got) and len(reads) == 1


def test_compat_form_follows_the_preflight_flag_at_each_call(monkeypatch):
    """(a) (b) (c): one source for the scanner-failure setting, with today's timing."""
    import fie.policy as policy
    import fie.preflight as pf

    open_form, closed_form = policy.compat_policy(scan_failure="allow"), policy.compat_policy(scan_failure="block")
    assert policy.active_policy() is open_form
    monkeypatch.setattr(pf, "_FAIL_SECURE", True)
    assert policy.active_policy() is closed_form, "the flag is followed on the next call, without a reset"
    monkeypatch.setattr(pf, "_FAIL_SECURE", False)
    assert policy.active_policy() is open_form

    # The environment variable is not this module's to read: changing it after import does nothing.
    for value in ("closed", "open", PLANTED_ENV):
        monkeypatch.setenv("FIE_SCAN_FAILURE_MODE", value)
        policy.reset_policy()
        assert policy.active_policy() is open_form
    monkeypatch.setattr(pf, "_FAIL_SECURE", True)
    monkeypatch.setenv("FIE_SCAN_FAILURE_MODE", "open")
    assert policy.active_policy() is closed_form


def test_explicit_policies_ignore_the_flag(monkeypatch, scanner_raises):
    """(d) (e): strict, review, set_policy and per-call policies state their own scanner-failure setting."""
    import fie.policy as policy
    import fie.preflight as pf

    for flag in (False, True):
        monkeypatch.setattr(pf, "_FAIL_SECURE", flag)
        for name, action in (("strict", "block"), ("review", "error")):
            _use(monkeypatch, name)
            g = pf.preflight_check(PROMPTS["injection"])
            assert (g.action, g.blocked, g.reasons, g.policy) == (action, True, ("scan_failed",), f"{name}@1")
        monkeypatch.delenv("FIE_POLICY")
        policy.reset_policy()
        policy.set_policy(policy.REVIEW)
        assert pf.preflight_check(PROMPTS["injection"]).action == "error"
        policy.reset_policy()

    monkeypatch.setattr(pf, "_FAIL_SECURE", True)
    explicit_open = policy.compat_policy(scan_failure="allow")
    g = pf.preflight_check(PROMPTS["injection"], policy=explicit_open)
    assert (g.action, g.blocked, g.scan_failed) == ("allow", False, True), "an explicit Policy is never altered"


def test_preflight_resolves_once_before_scanning(monkeypatch, loaded):
    """(f): one resolution per guarded call, before the scan; a per-call policy skips resolution."""
    import fie.policy as policy
    import fie.preflight as pf

    order = []
    real_active, real_scan = policy.active_policy, loaded.scan_prompt
    monkeypatch.setattr(policy, "active_policy", lambda: order.append("policy") or real_active())
    monkeypatch.setattr(loaded, "scan_prompt", lambda *a, **k: order.append("scan") or real_scan(*a, **k))
    pf.preflight_check(PROMPTS["benign_capital"])
    assert order == ["policy", "scan"]
    order.clear()
    pf.preflight_check(PROMPTS["benign_python"], policy=policy.STRICT)
    assert order == ["scan"], "a per-call policy does not consult active_policy()"
    with pytest.raises(TypeError):
        pf.preflight_check(PROMPTS["benign_python"], policy="strict")
    with pytest.raises(TypeError):
        pf.preflight_check(PROMPTS["benign_python"], None, None, policy.STRICT)   # keyword-only


@pytest.mark.parametrize("case", ["compat-open", "compat-closed", "strict", "review", "set_policy", "per-call"])
def test_identity_equals_effect(monkeypatch, loaded, case):
    """(g): the id and digest on the result are those of the object that decided, and replaying it agrees."""
    import fie.policy as policy
    import fie.preflight as pf

    kwargs = {}
    if case == "compat-closed":
        monkeypatch.setattr(pf, "_FAIL_SECURE", True)
    elif case in ("strict", "review"):
        _use(monkeypatch, case)
    elif case == "set_policy":
        policy.set_policy(policy.REVIEW)
    elif case == "per-call":
        kwargs["policy"] = policy.STRICT
    effective = kwargs.get("policy") or policy.active_policy()

    seen = []
    real_decide = policy.decide
    monkeypatch.setattr(policy, "decide", lambda r, p, **k: seen.append((r, p, k)) or real_decide(r, p, **k))
    for name in ("benign_capital", "harmful_pair_only", "injection"):
        seen.clear()
        g = pf.preflight_check(PROMPTS[name], **kwargs)
        assert len(seen) == 1, "exactly one decision per guarded call"
        result, used, extra = seen[0]
        assert used is effective, "decide() received the resolved object itself"
        replay = real_decide(result, effective, **extra)
        assert (g.action, g.policy, g.reasons) == (replay.action, effective.id, replay.reasons)
        assert replay.policy_digest == effective.digest


def test_no_policy_substitution_during_a_call(monkeypatch, loaded):
    """(h): a flag flip or set_policy during the scan affects the next call, never this one."""
    import fie.policy as policy
    import fie.preflight as pf

    def crash_after_reconfiguring(*args, **kwargs):
        pf._FAIL_SECURE = True
        policy.set_policy(policy.REVIEW)
        raise RuntimeError("scanner crashed")

    monkeypatch.setattr(loaded, "scan_prompt", crash_after_reconfiguring)
    first = pf.preflight_check(PROMPTS["injection"])
    assert (first.action, first.blocked, first.policy) == ("allow", False, "compat@1"), \
        "the call finished under the policy it resolved before scanning"
    second = pf.preflight_check(PROMPTS["injection"])
    assert (second.action, second.blocked, second.policy) == ("error", True, "review@1")


# ══ 5. Invalid configuration ═════════════════════════════════════════════════

def _invalid(monkeypatch):
    monkeypatch.setenv("FIE_POLICY", PLANTED_ENV)
    _reset_policy_state()


def test_invalid_config_blocks_preflight_without_scanning(monkeypatch, scan_spy, caplog, warn_only):
    import fie.policy as policy
    import fie.preflight as pf

    _invalid(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="fie"):
        with pytest.raises(policy.PolicyConfigError) as err:
            policy.active_policy()
        assert str(err.value) == "FIE_POLICY is not a known policy name"
        results = [pf.preflight_check(PROMPTS[name]) for name in ("benign_capital", "injection", "soft_harm")]
    assert scan_spy == [], "no scan may run when there is no policy to evaluate it under"
    for g in results:
        assert (g.blocked, g.action, g.reasons, g.policy) == (True, "error", ("invalid_policy_config",), "invalid")
        assert (g.scan_failed, g.attack_type, g.confidence, g.layers_fired) == (False, "", 0.0, [])
        assert (g.zone, g.coverage_status, g.refusal_message) == (None, None, pf._DEFAULT_REFUSAL)
    errors = [r for r in caplog.records if r.name == "fie.policy" and r.levelno == logging.ERROR]
    assert len(errors) == 1, "one ERROR per process, however many calls"
    text = "\n".join(r.getMessage() for r in caplog.records) + json.dumps([g.to_dict() for g in results])
    for needle in (*PLANTED, PLANTED_ENV, "strict "):
        assert needle not in text
    assert "invalid_policy_config" in text


def test_invalid_config_decision_is_fixed_and_pure():
    import fie.policy as policy

    bare = policy.invalid_config_decision()
    assert bare.to_dict() == {
        "action": "error", "blocked": True, "reasons": ["invalid_policy_config"], "policy": "invalid",
        "policy_digest": None, "detector_is_attack": None, "zone": None, "coverage_status": None,
        "missing_required": [], "schema_version": 1}
    r = _result("clear_block", clf="unavailable_model")
    with_result = policy.invalid_config_decision(r)
    assert (with_result.detector_is_attack, with_result.zone, with_result.coverage_status) == (
        True, "clear_block", "partial")
    assert (with_result.action, with_result.reasons, with_result.policy, with_result.policy_digest) == (
        "error", ("invalid_policy_config",), "invalid", None)
    assert policy.invalid_config_decision(None) == bare


def test_invalid_config_lifecycle_and_recovery(monkeypatch, loaded):
    import fie.policy as policy
    import fie.preflight as pf

    _invalid(monkeypatch)
    assert pf.preflight_check(PROMPTS["benign_capital"]).action == "error"
    assert pf.preflight_check("   ").action is None, "the empty-prompt rule comes first"
    assert pf.preflight_check("   ").blocked is False
    explicit = pf.preflight_check(PROMPTS["benign_capital"], policy=policy.STRICT)
    assert (explicit.action, explicit.policy) == ("allow", "strict@1"), "a per-call policy is unaffected"

    monkeypatch.setenv("FIE_POLICY", "strict")
    assert pf.preflight_check(PROMPTS["benign_capital"]).action == "error", "the failure is cached"
    policy.reset_policy()
    assert pf.preflight_check(PROMPTS["benign_capital"]).policy == "strict@1"

    _invalid(monkeypatch)
    assert pf.preflight_check(PROMPTS["benign_capital"]).action == "error"
    policy.set_policy(policy.REVIEW)
    assert pf.preflight_check(PROMPTS["benign_capital"]).policy == "review@1", "set_policy masks a cached failure"


def test_configuration_failure_is_not_scanner_failure(monkeypatch, scanner_raises):
    import fie.preflight as pf

    _use(monkeypatch, "strict")
    scan = pf.preflight_check(PROMPTS["injection"])
    _invalid(monkeypatch)
    config = pf.preflight_check(PROMPTS["injection"])
    assert (scan.reasons[0], scan.scan_failed, scan.policy) == ("scan_failed", True, "strict@1")
    assert (config.reasons[0], config.scan_failed, config.policy) == ("invalid_policy_config", False, "invalid")
    assert (scan.attack_type, scan.layers_fired) == ("SCAN_FAILED", ["preflight_fail_secure"])
    assert (config.attack_type, config.layers_fired) == ("", [])
    assert scan.blocked and config.blocked


@pytest.mark.parametrize("command", ["detect", "explain"])
@pytest.mark.parametrize("name,is_attack", [("benign_capital", False), ("injection", True)])
def test_cli_under_invalid_config_still_scans_once(monkeypatch, loaded, capsys, scan_spy, command, name, is_attack):
    reference = loaded.scan_prompt(PROMPTS[name])
    scan_spy.clear()
    valid_code, valid_out, valid_err = _cli(command, PROMPTS[name], capsys)
    assert valid_err == "" and len(scan_spy) == 1
    scan_spy.clear()

    _invalid(monkeypatch)
    code, out, err = _cli(command, PROMPTS[name], capsys)
    assert len(scan_spy) == 1, "the CLI is a display, not an enforcement point: it still scans, once"
    assert err.strip() == STDERR_NOTICE and err.count("\n") == 1
    assert code == valid_code == (1 if is_attack else 0)
    data, valid = json.loads(out), json.loads(valid_out)
    assert list(data) == list(valid) and list(data)[-1] == "policy"
    assert {k: v for k, v in data.items() if k != "policy"} == {k: v for k, v in valid.items() if k != "policy"}
    assert data["policy"] == {
        "action": "error", "blocked": True, "reasons": ["invalid_policy_config"], "policy": "invalid",
        "policy_digest": None, "detector_is_attack": reference.is_attack, "zone": reference.zone,
        "coverage_status": reference.coverage.status, "missing_required": [], "schema_version": 1}
    assert loaded.scan_prompt(PROMPTS[name]).to_dict() == reference.to_dict()
    for needle in (*PLANTED, PLANTED_ENV):
        assert needle not in out + err

    for kwargs in (dict(output="text"), dict(output="text", quiet=True)):
        code, out, err = _cli("detect", PROMPTS[name], capsys, **kwargs)
        assert err.strip() == STDERR_NOTICE and code == (1 if is_attack else 0)
        if kwargs.get("quiet"):
            assert out == "", "--quiet still prints nothing to stdout"
        else:
            assert "error (invalid_policy_config)" in out
    _, out, err = _cli(command, PROMPTS[name], capsys, output="text")
    assert "error (invalid_policy_config)" in out and err.strip() == STDERR_NOTICE


# ══ 6. Compatibility: C1 to C5 ═══════════════════════════════════════════════

@pytest.mark.parametrize("warn", [False, True], ids=["block-mode", "warn-only"])
@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_compat_reproduces_today_with_the_classifier_loaded(monkeypatch, loaded, name, warn):
    """C1 and C2: the nine existing GuardResult values, captured from the unmodified code."""
    import fie.preflight as pf

    monkeypatch.setattr(pf, "_get_block_enabled", lambda: not warn)
    assert _old_fields(pf.preflight_check(PROMPTS[name])) == PRE[("loaded", warn)][name]


@pytest.mark.parametrize("warn", [False, True], ids=["block-mode", "warn-only"])
@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_compat_reproduces_today_with_the_classifier_missing(monkeypatch, missing, name, warn):
    import fie.preflight as pf

    monkeypatch.setattr(pf, "_get_block_enabled", lambda: not warn)
    assert _old_fields(pf.preflight_check(PROMPTS[name])) == PRE[("missing", warn)][name]


@pytest.mark.parametrize("warn", [False, True], ids=["block-mode", "warn-only"])
@pytest.mark.parametrize("fail_secure", [False, True], ids=["open", "closed"])
def test_compat_reproduces_today_when_the_scanner_raises(monkeypatch, scanner_raises, fail_secure, warn):
    import fie.preflight as pf

    monkeypatch.setattr(pf, "_FAIL_SECURE", fail_secure)
    monkeypatch.setattr(pf, "_get_block_enabled", lambda: not warn)
    g = pf.preflight_check(PROMPTS["injection"])
    assert _old_fields(g) == PRE_SCAN_RAISES[fail_secure]
    assert _old_fields(pf.preflight_check("   ")) == PRE_EMPTY
    assert g.schema_version == 2


def test_compat_decisions_carry_the_new_fields(loaded):
    import fie.preflight as pf

    safe = pf.preflight_check(PROMPTS["benign_capital"])
    clear = pf.preflight_check(PROMPTS["injection"])
    uncertain = pf.preflight_check(PROMPTS["harmful_pair_only"])
    assert (safe.action, safe.policy, safe.reasons) == ("allow", "compat@1", ("detector_allow",))
    assert (clear.action, clear.policy, clear.reasons) == ("block", "compat@1", ("detector_block",))
    assert (uncertain.action, uncertain.reasons) == ("block", ("detector_uncertain",))


def test_guard_result_additions_are_appended_and_defaulted():
    """C3."""
    from fie.preflight import GuardResult, preflight_check

    for legacy in (GuardResult(False, "", 0.0, [], ""),
                   GuardResult(blocked=True, attack_type="X", confidence=0.9, layers_fired=["regex"],
                               refusal_message="no", scan_failed=False, zone="clear_block", coverage_status="full")):
        assert (legacy.action, legacy.policy, legacy.reasons, legacy.schema_version) == (None, None, (), 2)
        d = legacy.to_dict()
        assert tuple(d) == GUARD_KEYS and tuple(d)[:9] == OLD_GUARD_FIELDS
        assert (d["action"], d["policy"], d["reasons"]) == (None, None, [])
        assert json.loads(json.dumps(d)) == d
    positional = GuardResult(True, "X", 0.9, ["regex"], "no", False, "clear_block", "full", 2)
    assert positional.coverage_status == "full" and positional.action is None

    empty = preflight_check("   ")
    assert (empty.blocked, empty.action, empty.policy, empty.reasons) == (False, None, None, ())


@pytest.mark.parametrize("command,old_keys", [
    ("detect", ("is_attack", "attack_type", "category", "confidence", "layers_fired", "matched_text", "mitigation",
                "evidence", "zone", "decided_by", "coverage", "models", "degraded_layers", "schema_version")),
    ("explain", ("is_attack", "attack_type", "confidence", "layer_scores", "layers_fired", "evidence", "mitigation",
                 "zone", "decided_by", "coverage", "models", "degraded_layers", "schema_version")),
])
def test_cli_json_appends_exactly_one_key(monkeypatch, loaded, capsys, command, old_keys):
    """C4."""
    import fie.policy as policy

    for preset in PRESETS:
        _use(monkeypatch, preset)
        for name in ("injection", "benign_capital", "harmful_pair_only"):
            code, out, err = _cli(command, PROMPTS[name], capsys)
            data = json.loads(out)
            r = loaded.scan_prompt(PROMPTS[name])
            assert tuple(data) == old_keys + ("policy",) and err == ""
            full = r.to_dict()
            for key in old_keys:
                want = full[key] if key in ("zone", "decided_by", "coverage", "models", "degraded_layers",
                                            "schema_version") else json.loads(json.dumps(getattr(r, key)))
                assert data[key] == want, key
            assert data["policy"] == policy.decide(r, policy.active_policy()).to_dict()
            assert tuple(data["policy"]) == DECISION_KEYS
            assert code == (1 if r.is_attack else 0)


def test_cli_exit_code_is_the_detector_verdict_not_the_policy(monkeypatch, missing, capsys):
    """W4-15: under strict with the classifier missing, a benign verdict exits 0 while the policy blocks."""
    _use(monkeypatch, "strict")
    for command in ("detect", "explain"):
        code, out, err = _cli(command, PROMPTS["soft_harm"], capsys)
        data = json.loads(out)
        assert (code, data["is_attack"], data["policy"]["action"], data["policy"]["blocked"]) == (0, False, "block", True)
        assert data["policy"]["reasons"][0] == "missing_required_layer" and err == ""
    code, out, _ = _cli("detect", PROMPTS["soft_harm"], capsys, output="text")
    line = [ln for ln in out.splitlines() if ln.strip().startswith("Policy")]
    assert code == 0 and len(line) == 1 and "block" in line[0] and "missing_required_layer" in line[0]
    code, out, err = _cli("detect", PROMPTS["soft_harm"], capsys, output="text", quiet=True)
    assert (code, out, err) == (0, "", "")


def test_cli_text_has_no_policy_line_when_the_policy_adds_nothing(loaded, capsys):
    for command in ("detect", "explain"):
        for name in ("benign_capital", "injection"):
            _, out, err = _cli(command, PROMPTS[name], capsys, output="text")
            assert not [ln for ln in out.splitlines() if ln.strip().startswith("Policy")] and err == ""


@pytest.mark.parametrize("config", [None, "compat", "strict", "review", PLANTED_ENV])
def test_scan_result_is_the_same_under_every_policy_configuration(monkeypatch, loaded, config):
    """C5: scan_prompt does not know the policy exists."""
    if config is not None:
        monkeypatch.setenv("FIE_POLICY", config)
        _reset_policy_state()
    for name in sorted(PROMPTS):
        loaded._scan_cache._cache.clear()
        r = loaded.scan_prompt(PROMPTS[name], use_llama_guard=False)
        assert _verdict(r) == _expected(PRE_LOADED[name])
        assert r.zone == PRE_LOADED[name][5] and tuple(r.to_dict()) == TO_DICT_KEYS
        assert not hasattr(r, "action") and not hasattr(r, "policy") and not hasattr(r, "reasons")


@pytest.mark.parametrize("config", ["strict", "review", PLANTED_ENV])
def test_scan_result_is_the_same_with_the_classifier_missing(monkeypatch, missing, config):
    import fie.adversarial as adv

    monkeypatch.setenv("FIE_POLICY", config)
    _reset_policy_state()
    for name in sorted(PROMPTS):
        adv._scan_cache._cache.clear()
        r = adv.scan_prompt(PROMPTS[name], use_llama_guard=False)
        assert _verdict(r) == _expected(PRE_MISSING[name]) and r.degraded_layers == ["pair_classifier"]


def test_detector_source_files_are_untouched():
    """The detector is the same file it was at the WP-003 commit."""
    import subprocess

    out = subprocess.run(["git", "status", "--porcelain", "--", "fie/adversarial.py", "fie/layers", "fie/_lite.py",
                          "fie/onnx_encoder.py", "fie/multilingual.py", "fie/monitor.py", "fie/integrations",
                          "evals", "app", "engine", "storage"],
                         cwd=str(ROOT), capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    assert out.stdout.strip() == "", out.stdout


# ══ 7. Coverage states through the real scanner ══════════════════════════════

MISSING_EXPECT = {"compat": ("allow", False), "strict": ("block", True), "review": ("review", True)}


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("reason", ["dependency", "model", "load_failed"])
def test_missing_classifier_through_preflight(monkeypatch, loaded, tmp_path, preset, reason):
    import fie.preflight as pf

    _use(monkeypatch, preset)
    action, blocked = MISSING_EXPECT[preset]
    with _pair_module_restored() as pair:
        _make_classifier_unavailable(pair, monkeypatch, tmp_path, reason)
        allow = pf.preflight_check(PROMPTS["soft_harm"])            # only the classifier catches this one
        block = pf.preflight_check(PROMPTS["injection"])
    assert (allow.action, allow.blocked, allow.reasons[0]) == (action, blocked, "missing_required_layer")
    assert (allow.attack_type, allow.confidence, allow.zone, allow.coverage_status) == ("", 0.0, "allow", "partial")
    assert allow.refusal_message == (pf._DEFAULT_REFUSAL if blocked else "")
    assert (block.action, block.blocked, block.reasons[:2]) == (
        "block", True, ("detector_block", "missing_required_layer")), "a detector block is never downgraded"
    for needle in (*PLANTED, str(tmp_path), "joblib"):
        assert needle not in json.dumps(allow.to_dict())


@pytest.mark.parametrize("preset", PRESETS)
def test_classifier_error_timeout_and_loading_race(monkeypatch, loaded, preset):
    import time

    import fie.layers.pair as pair
    import fie.preflight as pf

    _use(monkeypatch, preset)
    action, blocked = MISSING_EXPECT[preset]

    class _BrokenEncoder:
        def encode(self, *args, **kwargs):
            raise RuntimeError(PLANTED_MESSAGE)

    with monkeypatch.context() as m:                                # inference raises -> layer state `error`
        m.setattr(pair, "_pair_embedder", _BrokenEncoder())
        g = pf.preflight_check(PROMPTS["benign_python"])
        assert (g.action, g.blocked, g.reasons[0]) == (action, blocked, "missing_required_layer")
        assert not any(needle in json.dumps(g.to_dict()) for needle in PLANTED)
    loaded._scan_cache._cache.clear()

    with monkeypatch.context() as m:                                # another thread still loading -> `timeout`
        m.setattr(loaded, "_load_pair_classifier", lambda: False)
        m.setattr(loaded, "_pair_unavailable_state", lambda: "loading")
        g = pf.preflight_check(PROMPTS["benign_capital"])
        assert (g.action, g.blocked, g.reasons[0]) == (action, blocked, "missing_required_layer")
    loaded._scan_cache._cache.clear()

    def _slow(prompt):
        time.sleep(2.5)
        return None, 0.0, {}

    with monkeypatch.context() as m:                                # a non-required layer times out
        m.setattr(loaded, "_LAYER_DEADLINE_S", 1.0)
        m.setattr(loaded, "_layer_gcg", _slow)
        g = pf.preflight_check(PROMPTS["benign_capital"])
        assert (g.action, g.blocked, g.reasons) == ("allow", False, ("detector_allow", "degraded_optional_layer"))
    time.sleep(1.8)


@pytest.mark.parametrize("preset", PRESETS)
def test_labelled_prompts_are_decided_by_the_label(monkeypatch, missing, preset):
    import fie.feedback_store as fb
    import fie.preflight as pf

    _use(monkeypatch, preset)
    monkeypatch.setattr(fb, "_WHITELIST_HASHES", {fb._prompt_hash(PROMPTS["injection"])})
    monkeypatch.setattr(fb, "_KNOWN_ATTACK_HASHES", {fb._prompt_hash(PROMPTS["benign_capital"])})
    safe, attack = pf.preflight_check(PROMPTS["injection"]), pf.preflight_check(PROMPTS["benign_capital"])
    assert (safe.action, safe.blocked, safe.reasons, safe.coverage_status) == (
        "allow", False, ("labelled_safe",), "bypassed"), "coverage rules do not apply to a labelled prompt"
    assert (attack.action, attack.blocked, attack.reasons) == ("block", True, ("labelled_attack",))
    assert attack.attack_type == "CONFIRMED_ATTACK"


@pytest.mark.parametrize("preset", PRESETS)
def test_disabled_layers(loaded, preset):
    from fie.policy import decide

    p = _preset(preset)
    other = decide(loaded.scan_prompt(PROMPTS["benign_capital"], use_llama_guard=False,
                                      disabled_layers={"copyright"}), p)
    assert (other.action, other.reasons) == ("allow", ("detector_allow", "layer_disabled_by_caller"))
    required = decide(loaded.scan_prompt(PROMPTS["soft_harm"], use_llama_guard=False,
                                         disabled_layers={"pair_classifier"}), p)
    assert (required.action, required.reasons[0]) == (MISSING_EXPECT[preset][0], "missing_required_layer")


# ══ 8. Warn-only ═════════════════════════════════════════════════════════════

@pytest.mark.parametrize("preset", PRESETS)
def test_warn_only_waives_detector_blocks_only(monkeypatch, loaded, warn_only, preset):
    import fie.preflight as pf

    _use(monkeypatch, preset)
    clear = pf.preflight_check(PROMPTS["injection"])
    assert (clear.action, clear.blocked, clear.reasons) == ("block", False, ("detector_block", "warn_only"))
    assert (clear.attack_type, clear.refusal_message) == ("PROMPT_EXTRACTION", "")
    uncertain = pf.preflight_check(PROMPTS["harmful_pair_only"])
    want = "review" if preset == "review" else "block"
    assert (uncertain.action, uncertain.blocked, uncertain.reasons) == (
        want, False, ("detector_uncertain", "warn_only")), "an uncertain-band block is waived today, and still is"
    safe = pf.preflight_check(PROMPTS["benign_capital"])
    assert (safe.action, safe.blocked, safe.reasons) == ("allow", False, ("detector_allow",))


@pytest.mark.parametrize("preset", ["strict", "review"])
def test_warn_only_cannot_waive_missing_coverage(monkeypatch, missing, warn_only, preset):
    import fie.preflight as pf

    _use(monkeypatch, preset)
    g = pf.preflight_check(PROMPTS["soft_harm"])
    assert g.blocked is True and g.action == MISSING_EXPECT[preset][0]
    assert g.reasons[0] == "missing_required_layer" and "warn_only" not in g.reasons
    detected = pf.preflight_check(PROMPTS["injection"])
    assert (detected.blocked, detected.reasons) == (False, ("detector_block", "missing_required_layer", "warn_only"))


@pytest.mark.parametrize("preset,action", [("strict", "block"), ("review", "error")])
def test_warn_only_cannot_waive_a_scanner_failure(monkeypatch, scanner_raises, warn_only, preset, action):
    import fie.preflight as pf

    _use(monkeypatch, preset)
    g = pf.preflight_check(PROMPTS["injection"])
    assert (g.action, g.blocked, g.reasons) == (action, True, ("scan_failed",))


def test_warn_only_is_not_consulted_unless_there_is_something_to_waive(monkeypatch, missing):
    import fie.preflight as pf

    asked = []
    monkeypatch.setattr(pf, "_get_block_enabled", lambda: asked.append(1) or False)
    _use(monkeypatch, "strict")
    assert pf.preflight_check(PROMPTS["soft_harm"]).blocked is True and asked == []
    assert pf.preflight_check(PROMPTS["benign_capital"]).blocked is True and asked == []
    assert pf.preflight_check(PROMPTS["injection"]).blocked is False and asked == [1]


# ══ 9. Verdict and action stay apart; scope; leaks; concurrency ══════════════

def test_verdict_and_action_are_different_things(monkeypatch, missing):
    import fie.adversarial as adv
    import fie.preflight as pf
    from fie.policy import decide

    _use(monkeypatch, "strict")
    r = adv.scan_prompt(PROMPTS["soft_harm"], use_llama_guard=False)
    d = decide(r, _preset("strict"))
    assert (r.is_attack, r.zone) == (False, "allow") and (d.detector_is_attack, d.zone) == (False, "allow")
    assert (d.action, d.blocked) == ("block", True), "the policy blocks what the detector allowed, and says both"
    assert d.to_dict()["detector_is_attack"] is False and d.to_dict()["action"] == "block"
    assert {f for f in ("action", "policy", "reasons", "blocked")} & set(type(r).__dataclass_fields__) == set()
    g = pf.preflight_check(PROMPTS["soft_harm"])
    assert (g.blocked, g.action, g.attack_type, g.zone) == (True, "block", "", "allow"), \
        "no detector finding is invented for a policy block"


def test_no_request_path_can_select_a_policy():
    import inspect

    import fie.policy as policy
    import fie.preflight as pf

    sig = inspect.signature(pf.preflight_check)
    assert list(sig.parameters) == ["prompt", "session_id", "domain", "policy"]
    assert sig.parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "disabled_layers" not in sig.parameters
    assert list(inspect.signature(policy.decide).parameters) == ["result", "policy", "scan_failed"]
    offenders = []
    for folder in ("app", "fie/integrations"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if re.search(r"policy\s*=|set_policy|fie\.policy|from fie import .*Policy", text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"server or integration code selects a policy: {offenders}"
    assert "FIE_POLICY" not in (ROOT / "fie" / "monitor.py").read_text(encoding="utf-8")


def test_known_gap_direct_scan_prompt_callers_are_not_covered(monkeypatch, missing):
    """
    CURRENT BEHAVIOUR, documented on purpose (PLAN_004 section 8.6, follow-up F3).

    The OpenAI wrapper calls scan_prompt() directly. With FIE_POLICY=strict and the
    classifier missing, preflight_check blocks this prompt, and the wrapper still
    forwards it. WP-004 is not universal enforcement. The package that migrates the
    integrations must change this test deliberately.
    """
    import fie.preflight as pf
    from fie.integrations.openai import Client

    _use(monkeypatch, "strict")
    assert pf.preflight_check(PROMPTS["soft_harm"]).blocked is True

    forwarded = []

    class _Completions:
        def create(self, **kwargs):
            forwarded.append(kwargs)
            return type("R", (), {"choices": []})()

    client = Client.__new__(Client)
    client._openai = type("O", (), {"chat": type("C", (), {"completions": _Completions()})()})()
    client._block_attacks, client._fie_client, client._mode = True, None, "monitor"
    client._monitored_create(messages=[{"role": "user", "content": PROMPTS["soft_harm"]}], model="any")
    assert len(forwarded) == 1, "the integration forwarded a prompt the configured policy blocks"


def test_audit_line_and_outputs_carry_no_internal_detail(monkeypatch, loaded, tmp_path, caplog):
    import fie.preflight as pf

    _use(monkeypatch, "strict")
    outputs = []
    with caplog.at_level(logging.DEBUG, logger="fie.policy"):
        with _pair_module_restored() as pair:
            _make_classifier_unavailable(pair, monkeypatch, tmp_path, "model")
            outputs.append(pf.preflight_check(PROMPTS["soft_harm"]).to_dict())
        with monkeypatch.context() as m:
            m.setattr(loaded, "scan_prompt", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(PLANTED_MESSAGE)))
            outputs.append(pf.preflight_check(PROMPTS["injection"]).to_dict())
        outputs.append(pf.preflight_check(PROMPTS["benign_capital"]).to_dict())
    lines = [r.getMessage() for r in caplog.records if r.name == "fie.policy"]
    assert len(lines) == 2, "one audit line per decision that is not a plain detector allow"
    assert "action=block" in lines[0] and "missing_required_layer" in lines[0] and "strict@1" in lines[0]
    assert "pair_classifier" in lines[0] and "scan_failed" in lines[1]
    text = "\n".join(lines) + json.dumps(outputs)
    for needle in (*PLANTED, str(tmp_path), "no-models-here", PROMPTS["soft_harm"], PROMPTS["injection"]):
        assert needle not in text


def test_concurrent_guard_calls_agree_with_serial_ones(monkeypatch, loaded):
    import fie.preflight as pf

    for preset in PRESETS:
        _use(monkeypatch, preset)
        loaded._scan_cache._cache.clear()
        serial = {name: pf.preflight_check(PROMPTS[name]).to_dict() for name in sorted(PROMPTS)}
        loaded._scan_cache._cache.clear()
        names = sorted(PROMPTS) * 4
        with ThreadPoolExecutor(max_workers=8) as pool:
            parallel = list(pool.map(lambda n: (n, pf.preflight_check(PROMPTS[n]).to_dict()), names))
        for name, got in parallel:
            assert got == serial[name], (preset, name)
            assert got["policy"] == f"{preset}@1"
