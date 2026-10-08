"""
The subject adapter: the ONLY module in the harness that imports or touches `fie`.

Everything the harness knows about the guardrail goes through here, so that a
refactor inside `fie` can break exactly one obvious place. The adapter reads
several private names, because `fie` does not yet expose them publicly:

    which model file was loaded      -> only a log line says so
    the routing zone of a result     -> only an evidence string says so
    the result cache                 -> has no public clear()

Each private dependency is listed in PRIVATE_CONTRACT and checked by
tests/evals/test_subject_contract.py. WP-003 (truthful scan result) is expected
to replace most of them with public fields.

Nothing here modifies a file under `fie/`. The profile is applied at run time, in
this process only: see evals/registry/profiles.json for the list of deviations
from an unmodified user environment.

`fie` is imported lazily, inside prepare(), so the caller can arm the hermetic
guard and verify the model files first.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from pathlib import Path

from evals.integrity import ModelIntegrityError

ZONE_ALLOW = "allow"
ZONE_UNCERTAIN = "uncertain_block"
ZONE_CLEAR = "clear_block"
ZONE_ERROR = "error"

# The evidence marker scan_prompt() attaches when the UNCERTAIN band is blocked
# because no tiebreaker was reachable (fie/adversarial.py, fail-secure branch).
_UNCERTAIN_MARKER = ("llama_guard", "unavailable_blocked")

FIXED_BENIGN_TRANSLATION = (
    "This is an ordinary everyday question about cooking, travel, homework or work."
)

# (module, attribute) pairs the adapter depends on. Checked by the contract test.
PRIVATE_CONTRACT = (
    ("fie.adversarial", "scan_prompt"),
    ("fie.adversarial", "warmup"),
    ("fie.adversarial", "_scan_cache"),
    ("fie.adversarial", "_server_config"),
    ("fie.adversarial", "_ATTACK_THRESHOLDS"),
    ("fie.adversarial", "_LAYER_WEIGHTS"),
    ("fie.adversarial", "_FAST_PATH_LAYERS"),
    ("fie.adversarial", "_DOMAIN_MULTIPLIERS"),
    ("fie.adversarial", "SCAN_THRESHOLD"),
    ("fie.adversarial", "_LAYER_POOL_SIZE"),
    ("fie.adversarial", "_LAYER_DEADLINE_S"),
    ("fie.multilingual", "translate_to_english"),
    ("fie.multilingual", "_TRANSLATION_CACHE"),
    ("fie.layers.pair", "_pair_state"),
    ("fie.layers.pair", "_meta_state"),
    ("fie.layers.pair", "_meta_threshold"),
    ("fie.layers.pair", "_meta_clf_features"),
    ("fie.feedback_store", "_KNOWN_ATTACK_HASHES"),
    ("fie.feedback_store", "_WHITELIST_HASHES"),
    ("fie.feedback_store", "_LOCAL_PATH"),
    ("fie.framing_filter", "FRAMING_DAMPEN_FACTOR"),
    ("fie.llama_guard", "_GROQ_API_KEY"),
)

_LOAD_LOG_RE = re.compile(r"status=ready model=(\S+) threshold=(\S+) backend=(\S+)")


class SubjectError(Exception):
    """The profile's preconditions do not hold in this process."""


class _M:
    """Imported fie modules and the runtime knobs currently applied."""
    adv = ml = pair = fb = ff = lg = None
    translation = "unavailable"
    langdetect_seed: int | None = 0
    load_log: list[str] = []
    state: dict = {}


class _Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _M.load_log.append(record.getMessage())
        except Exception:
            pass


# ── runtime knobs ────────────────────────────────────────────────────────────

def set_translation(mode: str) -> None:
    """
    Replace fie's online translator for this process.

    "unavailable"   returns None — identical to the translator being unreachable
    "fixed_benign"  returns one fixed English sentence — a counterfactual, not a
                    measurement of any translation service
    """
    if mode not in ("unavailable", "fixed_benign"):
        raise SubjectError(f"unknown translation mode: {mode!r}")

    if mode == "unavailable":
        def stub(text, timeout=3.0):
            return None
    else:
        def stub(text, timeout=3.0):
            return FIXED_BENIGN_TRANSLATION
    _M.ml.translate_to_english = stub
    _M.ml._TRANSLATION_CACHE.clear()
    _M.translation = mode


def set_langdetect_seed(seed: int | None) -> None:
    """
    Seed the language detector, or (None) leave it unseeded as the product ships.

    Unseeded, langdetect draws random n-gram samples, and at least one benchmark
    verdict changes from run to run. The canonical profile fixes the seed; the
    stability suite removes it again to measure the shipped behaviour.
    """
    try:
        from langdetect import DetectorFactory
    except ImportError:
        _M.langdetect_seed = seed
        return
    DetectorFactory.seed = seed
    _M.langdetect_seed = seed


def clear_caches() -> None:
    """fie's result cache key ignores configuration; clear it between suites."""
    _M.adv._scan_cache._cache.clear()
    _M.ml._TRANSLATION_CACHE.clear()


# ── preparation and cross-checks ─────────────────────────────────────────────

def _check_environment(profile: dict) -> None:
    problems = []
    for name in ("GROQ_API_KEY", "FIE_UNCERTAIN_ALLOW", "FIE_DISABLE_META", "REDIS_URL",
                 "LIBRETRANSLATE_URL", "SCAN_THRESHOLD", "FRAMING_DAMPEN_FACTOR",
                 "FIE_EMBED_BACKEND"):
        if os.environ.get(name):
            problems.append(f"{name} is set in the worker environment")
    want = profile.get("pair_version")
    if want and os.environ.get("FIE_PAIR_VERSION") != want:
        problems.append(f"FIE_PAIR_VERSION is {os.environ.get('FIE_PAIR_VERSION')!r}, "
                        f"profile requires {want!r}")
    for name in ("FIE_NO_TELEMETRY", "FIE_NO_AUTO_DOWNLOAD", "FIE_FEEDBACK_PATH"):
        if not os.environ.get(name):
            problems.append(f"{name} is not set")
    if problems:
        raise SubjectError("profile preconditions not met: " + "; ".join(problems))


def prepare(profile: dict, repo_root: str | Path, verified: dict, declared: dict) -> dict:
    """
    Import fie under `profile`, warm it up, and confirm that what loaded is what
    was verified. Call only after the guard is armed and the models are hashed.

    verified   role table from integrity.verify_roles()
    declared   integrity.declared_metadata() — values from hash-checked files
    """
    _check_environment(profile)

    # The cross-check reads the loader's own log line. Make sure logging is not
    # globally disabled, or that check would (correctly) refuse to proceed.
    logging.disable(logging.NOTSET)
    logging.getLogger().setLevel(logging.WARNING)
    pair_logger = logging.getLogger("fie.layers.pair")
    pair_logger.setLevel(logging.INFO)
    capture = _Capture(level=logging.INFO)
    pair_logger.addHandler(capture)

    t0 = time.perf_counter()
    import fie
    import fie.adversarial as adv
    import fie.feedback_store as fb
    import fie.framing_filter as ff
    import fie.llama_guard as lg
    import fie.multilingual as ml
    import fie.layers.pair as pair
    import_s = time.perf_counter() - t0

    root = Path(repo_root).resolve()
    origin = Path(fie.__file__).resolve()
    if root not in origin.parents:
        raise SubjectError(
            f"`fie` was imported from {origin}, not from the working tree {root}. "
            "A stale installed copy must never be measured.")

    _M.adv, _M.ml, _M.pair, _M.fb, _M.ff, _M.lg = adv, ml, pair, fb, ff, lg
    set_translation(profile.get("translation", "unavailable"))
    set_langdetect_seed(profile.get("langdetect_seed", 0))

    t0 = time.perf_counter()
    warm = adv.warmup()
    warmup_s = time.perf_counter() - t0

    pair_logger.removeHandler(capture)
    logging.getLogger("fie").setLevel(logging.ERROR)     # per-block warnings are noise here
    pair_logger.setLevel(logging.ERROR)

    state = {
        "fie_origin_in_worktree": True,
        "warmup": {k: v for k, v in warm.items() if k != "elapsed_ms"},
        "pair_state": pair._pair_state(),
        "meta_state": pair._meta_state(),
        "server_config_attached": adv._server_config() is not None,
        "load_log": [m for m in _M.load_log if "pair_classifier" in m or "encoder=" in m][:12],
        "import_s": round(import_s, 3),
        "warmup_s": round(warmup_s, 3),
    }
    _M.state = state
    _cross_check(profile, verified, declared, state)
    return state


def _cross_check(profile: dict, verified: dict, declared: dict, state: dict) -> None:
    """The model that loaded must be the model that was verified. Otherwise abort."""
    expect = profile.get("expect", {})
    problems = []
    adv, pair, fb, lg = _M.adv, _M.pair, _M.fb, _M.lg

    pair_loaded = bool(state["pair_state"]["loaded"])
    meta_loaded = bool(state["meta_state"]["loaded"])
    if pair_loaded != bool(expect.get("pair_loaded")):
        problems.append(f"PAIR classifier loaded={pair_loaded}, profile expects "
                        f"{expect.get('pair_loaded')} (loader error: {state['pair_state'].get('error')})")
    if meta_loaded != bool(expect.get("meta_loaded")):
        problems.append(f"meta-classifier loaded={meta_loaded}, profile expects {expect.get('meta_loaded')}")

    if expect.get("pair_loaded") and pair_loaded:
        ready = [_LOAD_LOG_RE.search(m) for m in _M.load_log]
        ready = [m for m in ready if m]
        if not ready:
            problems.append("the loader did not log which PAIR file it loaded")
        else:
            file_name, _, backend = ready[-1].groups()
            want_file = verified.get("pair_classifier", {}).get("file")
            if file_name != want_file:
                problems.append(f"loader reports model file {file_name!r}, verified file is {want_file!r}")
            if expect.get("pair_file") and file_name != expect["pair_file"]:
                problems.append(f"loader reports {file_name!r}, profile expects {expect['pair_file']!r}")
            if expect.get("encoder_backend") and backend != expect["encoder_backend"]:
                problems.append(f"embedder backend is {backend!r}, profile expects {expect['encoder_backend']!r}")
        d_pair = declared.get("pair", {})
        if not _same(state["pair_state"]["threshold"], d_pair.get("threshold")):
            problems.append(f"loaded PAIR threshold {state['pair_state']['threshold']} differs from the "
                            f"verified metadata file ({d_pair.get('threshold')})")
        if expect.get("pair_declared_version") and d_pair.get("declared_version") != expect["pair_declared_version"]:
            problems.append(f"metadata declares version {d_pair.get('declared_version')!r}, profile "
                            f"expects {expect['pair_declared_version']!r}")

    if expect.get("meta_loaded") and meta_loaded:
        d_meta = declared.get("meta_classifier", {})
        if not _same(pair._meta_threshold(), d_meta.get("threshold")):
            problems.append(f"loaded meta-classifier threshold {pair._meta_threshold()} differs from the "
                            f"verified metadata file ({d_meta.get('threshold')})")
        if list(pair._meta_clf_features) != list(d_meta.get("features", [])):
            problems.append("meta-classifier feature list differs from the verified metadata file")

    if not expect.get("pair_loaded") and not pair_loaded:
        err = state["pair_state"].get("error") or ""
        if "missing dependency" not in err:
            problems.append(f"PAIR is absent for an unexpected reason: {err!r}")

    if "server_config_attached" in expect and state["server_config_attached"] != expect["server_config_attached"]:
        problems.append(f"server hot-config attached={state['server_config_attached']}, "
                        f"profile expects {expect['server_config_attached']}")
    if lg._GROQ_API_KEY:
        problems.append("the tiebreaker module holds an API key")
    if fb._KNOWN_ATTACK_HASHES or fb._WHITELIST_HASHES:
        problems.append("the feedback store's allow/deny sets are not empty")
    want_path = os.environ.get("FIE_FEEDBACK_PATH", "")
    if os.path.normcase(str(fb._LOCAL_PATH)) != os.path.normcase(str(Path(want_path))):
        problems.append(f"feedback store writes to {fb._LOCAL_PATH}, not the run's scratch path")

    if problems:
        raise ModelIntegrityError(
            "the subject that loaded is not the subject that was verified:\n  - "
            + "\n  - ".join(problems))


def _same(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


# ── configuration read-out ───────────────────────────────────────────────────

def configuration() -> dict:
    """The subject-side half of the CONFIGURATION fingerprint block."""
    adv, pair, ff, fb = _M.adv, _M.pair, _M.ff, _M.fb
    cfg = adv._server_config()
    overrides = {}
    if cfg is not None:
        try:
            overrides = dict(cfg.get_attack_thresholds())
        except Exception as exc:                              # pragma: no cover
            overrides = {"error": type(exc).__name__}
    return {
        "pair": {"loaded": bool(_M.state["pair_state"]["loaded"]),
                 "threshold": float(_M.state["pair_state"]["threshold"])},
        "meta": {"loaded": bool(_M.state["meta_state"]["loaded"]),
                 "threshold": float(pair._meta_threshold()),
                 "features": list(pair._meta_clf_features)},
        "thresholds": {"attack": {k: float(v) for k, v in adv._ATTACK_THRESHOLDS.items()},
                       "scan": float(adv.SCAN_THRESHOLD)},
        "layer_weights": {k: float(v) for k, v in adv._LAYER_WEIGHTS.items()},
        "fast_path_layers": sorted(adv._FAST_PATH_LAYERS),
        "domain_multipliers": {k: float(v) for k, v in adv._DOMAIN_MULTIPLIERS.items()},
        "framing_dampen_factor": float(ff.FRAMING_DAMPEN_FACTOR),
        "operator_overrides": overrides,
        "server_config_attached": cfg is not None,
        "layer_pool_size": int(adv._LAYER_POOL_SIZE),
        "layer_deadline_s": float(adv._LAYER_DEADLINE_S),
        "feedback_store": {"known_attack_hashes": len(fb._KNOWN_ATTACK_HASHES),
                           "whitelist_hashes": len(fb._WHITELIST_HASHES)},
        "tiebreaker": "disabled",
        "scan_args": {"use_llama_guard": False, "domain": None, "session_id": None,
                      "threshold": None, "disabled_layers": []},
        "cache_policy": "result and translation caches cleared before every suite",
    }


# ── scanning ─────────────────────────────────────────────────────────────────

def derive_zone(is_attack: bool, evidence) -> str:
    """
    Routing zone of a result. Valid only while the tiebreaker is off and
    FIE_UNCERTAIN_ALLOW is unset — both are asserted in prepare().
    """
    if not is_attack:
        return ZONE_ALLOW
    key, value = _UNCERTAIN_MARKER
    if isinstance(evidence, dict) and evidence.get(key) == value:
        return ZONE_UNCERTAIN
    return ZONE_CLEAR


def _json_safe(obj, depth: int = 0):
    """Evidence may hold arbitrary objects; make it serializable without guessing."""
    if depth > 12:
        return "<truncated>"
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(str(v) for v in obj)
    return f"<{type(obj).__name__}>"


def scan(text: str) -> tuple[dict, dict]:
    """
    Scan one input. Returns (verdict fields, evidence).

    A scan that raises is a finding, not a crash: it is returned with
    status "error:<ExceptionClass>" and the suite is marked incomplete.
    """
    try:
        r = _M.adv.scan_prompt(text, use_llama_guard=False)
    except Exception as exc:
        return ({"flagged": False, "zone": ZONE_ERROR, "type": None, "conf": 0.0,
                 "layers_fired": [], "layer_scores": {}, "degraded": [],
                 "status": f"error:{type(exc).__name__}"},
                {"error": str(exc)[:300]})
    evidence = r.evidence if isinstance(r.evidence, dict) else {}
    fields = {
        "flagged": bool(r.is_attack),
        "zone": derive_zone(bool(r.is_attack), evidence),
        "type": r.attack_type,
        "conf": float(r.confidence),
        "layers_fired": sorted(r.layers_fired or []),
        "layer_scores": {str(k): float(v) for k, v in (r.layer_scores or {}).items()},
        "degraded": sorted(r.degraded_layers or []),
        "status": "ok",
    }
    return fields, _json_safe(evidence)


# ── contract ─────────────────────────────────────────────────────────────────

def contract_report() -> dict:
    """
    Check every private name the adapter relies on. Imports fie; run it in a
    child process with telemetry disabled.
    """
    import importlib

    missing = []
    for module_name, attr in PRIVATE_CONTRACT:
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            missing.append(f"{module_name}.{attr}")
    import fie.adversarial as adv
    import fie.layers.pair as pair
    shapes = {
        "scan_cache_has_dict": isinstance(getattr(adv._scan_cache, "_cache", None), dict),
        "attack_thresholds_is_dict": isinstance(adv._ATTACK_THRESHOLDS, dict),
        "pair_state_keys": sorted(pair._pair_state().keys()),
        "meta_state_keys": sorted(pair._meta_state().keys()),
        "scan_prompt_accepts_use_llama_guard":
            "use_llama_guard" in adv.scan_prompt.__code__.co_varnames,
        "scanresult_fields": sorted(adv.ScanResult.__dataclass_fields__.keys()),
        "pair_logger_format_present": "status=ready model=%s threshold=%.2f" in
            Path(pair.__file__).read_text(encoding="utf-8"),
        "uncertain_marker_present": '"llama_guard": "unavailable_blocked"' in
            Path(adv.__file__).read_text(encoding="utf-8"),
    }
    return {"missing": missing, "shapes": shapes}
