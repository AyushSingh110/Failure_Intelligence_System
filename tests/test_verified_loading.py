"""
Verified model loading (WP-021 Phase 3, Stage 3.2).

Every model the SDK deserializes is selected through the package manifest
(fie/models/MANIFEST.json), read once, checked against the listed size and SHA-256, and only
then deserialized, from the very bytes that were checked. These tests pin that:

  * selection is exact: the default, or exactly the requested version, never a substitute;
  * nothing unlisted, missing, altered or unreadable reaches a deserializer;
  * side-cars are verified before any of their fields is used, and are never replaced by defaults;
  * each failure keeps its public layer state and gets its own internal detail code;
  * the public result, coverage and policy contracts are unchanged.

Fixtures are synthetic: temporary model directories, manifests written here, and test-authored
pickles. No released model artifact is deserialized by this file. The negative control is a
canary: a pickle whose deserialization calls a recorder. A positive control proves the recorder
fires when the canary is deserialized; every refusal test then proves it did not fire.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import io
import json
import logging
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("FIE_NO_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fie._model_integrity as mi  # noqa: E402
import fie.layers._verified_models as vm  # noqa: E402
import fie.layers.pair as pair  # noqa: E402

VERSIONS = ("v6_3b", "v6_3", "v6", "v5", "v4", "v3", "v1")
CLASSIFIER_FILES = {v: mi._PAIR_FILES[v][0] for v in VERSIONS}
SIDECAR_FILES = {v: mi._PAIR_FILES[v][1] for v in VERSIONS}
APPROVED_META_SHA256 = "be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6"
EMBED_MODEL = "test-owner/test-embedder"


# ── canary and synthetic models ──────────────────────────────────────────────

CANARY_FIRED: list[str] = []


def _canary_fired(token):
    """Called only if a canary pickle is deserialized."""
    CANARY_FIRED.append(token)
    return ("canary", token)


class _Canary:
    def __init__(self, token):
        self.token = token

    def __reduce__(self):
        return _canary_fired, (self.token,)


class FakeModel:
    """A harmless stand-in classifier, identified by its tag."""

    def __init__(self, tag):
        self.tag = tag

    def predict_proba(self, vec):
        return [[0.9, 0.1]]


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return [[0.0]]


@pytest.fixture
def joblib():
    return pytest.importorskip("joblib")


def _pickled(obj) -> bytes:
    import joblib

    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def _canary(token: str) -> bytes:
    return _pickled(_Canary(token))


def _sidecar(threshold=0.5, embed_model=EMBED_MODEL, version="t1", **extra) -> bytes:
    body = {"threshold": threshold, "embed_model": embed_model, "version": version, **extra}
    return json.dumps(body).encode("utf-8")


def _meta_sidecar(threshold=0.41, layer_names=("regex", "pair_classifier"), version="m1") -> bytes:
    return json.dumps({"threshold": threshold, "layer_names": list(layer_names), "version": version}).encode()


# ── opens, seen by an audit hook ─────────────────────────────────────────────
# Installed once and inert unless a test is recording. Sees every open in the process, by
# any code, so "never opened" here means never opened by anything.

_OPENED: list | None = None
_HOOKED = False


def _audit(event, args):
    if _OPENED is not None and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.normcase(os.fsdecode(args[0])))


@contextlib.contextmanager
def recording_opens():
    global _OPENED, _HOOKED
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    _OPENED = []
    try:
        yield _OPENED
    finally:
        _OPENED = None


def _opened(log, path: Path) -> int:
    target = os.path.normcase(os.path.realpath(path))
    return sum(1 for p in log if os.path.normcase(os.path.realpath(p)) == target)


# ── a synthetic package ──────────────────────────────────────────────────────

class Package:
    """A temporary models directory and the manifest that lists its artifacts."""

    def __init__(self, root: Path):
        self.root = root
        self.entries: list[dict] = []

    def add(self, path, data: bytes, role, version=None, default=False, listed=True, write=True):
        if write:
            (self.root / path).write_bytes(data)
        if listed:
            entry = {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "role": role,
                     "in_wheel": True, "default": default}
            if version is not None:
                entry["version"] = version
            self.entries.append(entry)
        return self.root / path

    def entry(self, path) -> mi.Entry:
        raw = next(e for e in self.entries if e["path"] == path)
        return mi.Entry(path=raw["path"], sha256=raw["sha256"], size=raw["size"], role=raw["role"],
                        in_wheel=raw["in_wheel"], version=raw.get("version"), default=raw["default"])

    def write_manifest(self, body: bytes | None = None) -> None:
        if body is None:
            body = json.dumps({"schema": 1, "artifacts": self.entries}, indent=2).encode()
        (self.root / mi.PACKAGE_MANIFEST_NAME).write_bytes(body)

    def full(self, selected: str | None = None, meta: bool = True) -> "Package":
        """
        All seven versions listed. Every classifier file is a canary, except `selected`,
        which is a harmless FakeModel. Listed side-cars for each. The meta-classifier pair.
        """
        for v in VERSIONS:
            data = _pickled(FakeModel(v)) if v == selected else _canary(f"pair:{v}")
            self.add(CLASSIFIER_FILES[v], data, "pair_classifier", v, default=(v == "v6_3b"))
            self.add(SIDECAR_FILES[v], _sidecar(threshold=0.5 + VERSIONS.index(v) / 100, version=f"t-{v}"),
                     "pair_meta", v)
        if meta:
            self.add("meta_clf.pkl", _pickled(FakeModel("meta")), "meta_classifier")
            self.add("meta_clf.json", _meta_sidecar(), "meta_classifier_meta")
        return self


_STATE_PREFIXES = ("_pair_", "_meta_clf")


def _state_names():
    return {name: value for name, value in vars(pair).items()
            if name.startswith(_STATE_PREFIXES) and not name.endswith("_lock") and not inspect.isfunction(value)}


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """
    A fresh synthetic package: the loader reads this manifest and this directory, and its
    state starts unattempted. The real loaded state is put back afterwards.
    """
    models = tmp_path / "models"
    models.mkdir()
    saved = _state_names()
    monkeypatch.setattr(vm, "MANIFEST_PATH", models / mi.PACKAGE_MANIFEST_NAME)
    monkeypatch.setattr(vm, "_manifest_cache", {})
    monkeypatch.setattr(pair, "_resolve_models_dir", lambda sentinel=None: models)
    embedders: list[str] = []

    def fake_build_embedder(embed_model):
        embedders.append(embed_model)
        return FakeEmbedder()

    monkeypatch.setattr(pair, "_build_embedder", fake_build_embedder)
    monkeypatch.delenv("FIE_PAIR_VERSION", raising=False)
    pair._pair_clf = pair._pair_embedder = None
    pair._pair_load_attempted = False
    pair._pair_identity = {}
    pair._meta_clf = None
    pair._meta_clf_attempted = False
    pair._meta_clf_identity = {}
    CANARY_FIRED.clear()
    package = Package(models)
    package.embedders = embedders
    try:
        yield package
    finally:
        for name in set(_state_names()) - set(saved):
            delattr(pair, name)
        for name, value in saved.items():
            setattr(pair, name, value)
        CANARY_FIRED.clear()


def _no_tmp_detail(text: str, tmp_path: Path) -> None:
    for needle in (str(tmp_path), tmp_path.name, os.path.realpath(tmp_path)):
        assert needle not in text, f"an absolute location leaked: {needle!r} in {text!r}"


# ══ 0. The canary itself ═════════════════════════════════════════════════════

def test_positive_control_the_canary_fires_when_deserialized(joblib):
    CANARY_FIRED.clear()
    data = _canary("control")
    assert CANARY_FIRED == [], "building the canary must not fire it"
    assert joblib.load(io.BytesIO(data)) == ("canary", "control")
    assert CANARY_FIRED == ["control"]
    CANARY_FIRED.clear()


def test_positive_control_the_canary_fires_through_the_verified_path(joblib, tmp_path):
    """The one deserializer this module allows does deserialize listed, verified bytes."""
    CANARY_FIRED.clear()
    package = Package(tmp_path)
    package.add("c.pkl", _canary("verified-path"), "pair_classifier", "v1")
    entry = package.entry("c.pkl")
    assert vm.unpickle(entry, vm.read_artifact(tmp_path, entry)) == ("canary", "verified-path")
    assert CANARY_FIRED == ["verified-path"]
    CANARY_FIRED.clear()


# ══ 1. The verified-loading module ═══════════════════════════════════════════

def test_manifest_is_looked_for_in_the_package_only():
    assert vm.MANIFEST_PATH == Path(vm.__file__).resolve().parent.parent / "models" / "MANIFEST.json"
    assert vm.MANIFEST_PATH.parent == Path(pair.__file__).resolve().parent.parent / "models"


@pytest.mark.parametrize("body, detail", [
    (None, "manifest_missing"),
    (b"{not json", "manifest_invalid"),
    (b'{"schema": 2, "artifacts": []}', "manifest_invalid"),
    (b"\xff\xfe", "manifest_invalid"),
    (json.dumps({"schema": 1, "artifacts": [
        {"path": "a.pkl", "sha256": "0" * 64, "size": 1, "role": "pair_classifier", "version": "v1",
         "in_wheel": True, "default": True},
        {"path": "b.pkl", "sha256": "1" * 64, "size": 1, "role": "pair_classifier", "version": "v3",
         "in_wheel": True, "default": True}]}).encode(), "manifest_invalid"),
])
def test_manifest_failures_are_classified_and_cached(pkg, body, detail):
    if body is not None:
        pkg.write_manifest(body)
    for _ in range(2):
        with pytest.raises(vm.LoadRefused) as info:
            vm.package_manifest()
        assert info.value.detail == detail
        assert info.value.reason == ("model" if detail == "manifest_missing" else "load_failed")
    # The outcome is cached: writing a valid manifest now does not change it for this process.
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    with pytest.raises(vm.LoadRefused):
        vm.package_manifest()


def test_oversized_manifest_is_refused_unread(pkg, monkeypatch):
    monkeypatch.setattr(vm, "_MANIFEST_MAX_BYTES", 64)
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    with pytest.raises(vm.LoadRefused) as info:
        vm.package_manifest()
    assert info.value.detail == "manifest_invalid"


def test_unreadable_manifest_is_read_failed_without_the_os_message(pkg, monkeypatch, tmp_path):
    import builtins

    real_open = builtins.open

    def deny(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(vm.MANIFEST_PATH):
            raise PermissionError(13, "Access is denied", os.fspath(path))
        return real_open(path, *args, **kwargs)

    pkg.write_manifest(b"{}")
    monkeypatch.setattr(builtins, "open", deny)
    with pytest.raises(vm.LoadRefused) as info:
        vm.package_manifest()
    monkeypatch.setattr(builtins, "open", real_open)
    assert (info.value.detail, info.value.reason) == ("read_failed", "load_failed")
    _no_tmp_detail(str(info.value), tmp_path)


def test_load_refused_rejects_an_unknown_detail_code():
    with pytest.raises(ValueError):
        vm.LoadRefused("mismach", "typo")


def test_unpickle_refuses_bytes_that_are_not_the_listed_ones(joblib, tmp_path):
    """Even called directly, the deserializer entry point re-checks the bytes it is given."""
    package = Package(tmp_path)
    package.add("m.pkl", _pickled(FakeModel("ok")), "pair_classifier", "v1")
    entry = package.entry("m.pkl")
    CANARY_FIRED.clear()
    with pytest.raises(vm.LoadRefused) as info:
        vm.unpickle(entry, _canary("direct"))
    assert info.value.detail == "mismatch" and CANARY_FIRED == []
    with pytest.raises(vm.LoadRefused) as info:
        vm.parse_json(entry, b"{}")
    assert info.value.detail == "mismatch"


def test_verified_bytes_that_are_not_a_pickle_are_deserialize_failed(joblib, tmp_path):
    package = Package(tmp_path)
    package.add("junk.pkl", b"this is not a pickle", "pair_classifier", "v1")
    entry = package.entry("junk.pkl")
    with pytest.raises(vm.LoadRefused) as info:
        vm.unpickle(entry, vm.read_artifact(tmp_path, entry))
    assert (info.value.detail, info.value.reason, info.value.artifact) == ("deserialize_failed", "load_failed",
                                                                          "junk.pkl")
    assert info.value.__cause__ is not None, "the deserializer's own error stays attached for the log"


def test_import_error_inside_the_deserializer_is_not_disguised(joblib, tmp_path, monkeypatch):
    """A model that needs an uninstalled package is a missing dependency, not an integrity failure."""
    package = Package(tmp_path)
    package.add("m.pkl", _pickled(FakeModel("ok")), "pair_classifier", "v1")
    entry = package.entry("m.pkl")

    def needs_a_package(f):
        raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")

    monkeypatch.setattr(joblib, "load", needs_a_package)
    with pytest.raises(ModuleNotFoundError):
        vm.unpickle(entry, vm.read_artifact(tmp_path, entry))


def test_json_side_car_must_be_an_object(tmp_path):
    package = Package(tmp_path)
    package.add("list.json", b"[1, 2]", "pair_meta", "v1")
    package.add("bad.json", b"{oops", "pair_meta", "v3")
    for name in ("list.json", "bad.json"):
        entry = package.entry(name)
        with pytest.raises(vm.LoadRefused) as info:
            vm.parse_json(entry, vm.read_artifact(tmp_path, entry))
        assert info.value.detail == "deserialize_failed"


@pytest.mark.parametrize("value", [None, "0.5", True, float("nan"), float("inf"), -0.1, 1.5])
def test_threshold_is_required_and_never_defaulted(value):
    entry = mi.Entry(path="s.json", sha256="0" * 64, size=1, role="pair_meta", in_wheel=True, version="v1")
    meta = {} if value is None else {"threshold": value}
    with pytest.raises(vm.LoadRefused) as info:
        vm.required_threshold(entry, meta)
    assert info.value.detail == "deserialize_failed"


def test_required_fields_accept_valid_values():
    entry = mi.Entry(path="s.json", sha256="0" * 64, size=1, role="pair_meta", in_wheel=True, version="v1")
    assert vm.required_threshold(entry, {"threshold": 0.41}) == 0.41
    assert vm.required_threshold(entry, {"threshold": 1}) == 1.0
    assert vm.required_text(entry, {"embed_model": "a/b"}, "embed_model") == "a/b"
    assert vm.required_names(entry, {"layer_names": ["x", "y"]}, "layer_names") == ["x", "y"]
    for bad in ({}, {"embed_model": ""}, {"embed_model": 3}):
        with pytest.raises(vm.LoadRefused):
            vm.required_text(entry, bad, "embed_model")
    for bad in ({}, {"layer_names": []}, {"layer_names": "ab"}, {"layer_names": ["a", 2]}):
        with pytest.raises(vm.LoadRefused):
            vm.required_names(entry, bad, "layer_names")


def test_select_pair_is_exact(pkg):
    pkg.full()
    pkg.write_manifest()
    manifest = vm.package_manifest()
    for key in VERSIONS:
        clf, side = vm.select_pair(manifest, key)
        assert (clf.path, side.path, clf.version, side.version) == (CLASSIFIER_FILES[key], SIDECAR_FILES[key], key, key)
    for empty in (None, ""):
        clf, side = vm.select_pair(manifest, empty)
        assert (clf.path, side.path) == (CLASSIFIER_FILES["v6_3b"], SIDECAR_FILES["v6_3b"])
    for unlisted in ("v2", "V6_3B", "v6.3b", "pair_intent_classifier_v6_3b.pkl", "../v6_3b"):
        with pytest.raises(vm.LoadRefused) as info:
            vm.select_pair(manifest, unlisted)
        assert info.value.detail == "not_listed"


def test_select_pair_refuses_a_classifier_without_its_side_car(pkg):
    pkg.add(CLASSIFIER_FILES["v6_3b"], _canary("x"), "pair_classifier", "v6_3b", default=True)
    pkg.write_manifest()
    with pytest.raises(vm.LoadRefused) as info:
        vm.select_pair(vm.package_manifest(), None)
    assert info.value.detail == "not_listed"


def test_select_meta_refuses_ambiguity(pkg):
    pkg.add("meta_clf.pkl", _canary("m1"), "meta_classifier")
    pkg.add("meta_clf_b.pkl", _canary("m2"), "meta_classifier")
    pkg.add("meta_clf.json", _meta_sidecar(), "meta_classifier_meta")
    pkg.write_manifest()
    with pytest.raises(vm.LoadRefused) as info:
        vm.select_meta(vm.package_manifest())
    assert info.value.detail == "manifest_invalid"


# ══ 2. The meta-classifier ═══════════════════════════════════════════════════

def test_meta_classifier_loads_from_verified_bytes(pkg, joblib):
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    assert pair._load_meta_classifier() is True
    assert isinstance(pair._meta_clf, FakeModel) and pair._meta_clf.tag == "meta"
    assert pair._meta_threshold() == 0.41 and pair._meta_clf_features == ["regex", "pair_classifier"]
    expected = pkg.entry("meta_clf.pkl").sha256[:16]
    assert pair._meta_clf_identity == {"version": "m1", "digest": expected}
    assert pair._meta_unavailable_detail() is None


def _meta_sentinels():
    pair._meta_clf_threshold = 0.777
    pair._meta_clf_features = ["sentinel"]


def _assert_meta_untouched(detail):
    assert pair._load_meta_classifier() is False
    assert pair._meta_unavailable_detail() == detail
    assert pair._meta_clf is None and pair._meta_clf_identity == {}
    assert pair._meta_threshold() == 0.777 and pair._meta_clf_features == ["sentinel"], \
        "no threshold or feature list was taken from an unverified or missing side-car"
    assert CANARY_FIRED == []
    ident = pair._model_identities()["meta_classifier"]
    assert (ident["loaded"], ident["digest"], ident["threshold"]) == (False, None, None)


def test_meta_missing_manifest(pkg, joblib):
    _meta_sentinels()
    (pkg.root / "meta_clf.pkl").write_bytes(_canary("meta"))
    _assert_meta_untouched("manifest_missing")


def test_meta_invalid_manifest(pkg, joblib):
    _meta_sentinels()
    (pkg.root / "meta_clf.pkl").write_bytes(_canary("meta"))
    pkg.write_manifest(b"[]")
    _assert_meta_untouched("manifest_invalid")


def test_meta_ambiguous_singleton(pkg, joblib):
    _meta_sentinels()
    pkg.add("meta_clf.pkl", _canary("m1"), "meta_classifier")
    pkg.add("meta_clf_b.pkl", _canary("m2"), "meta_classifier")
    pkg.add("meta_clf.json", _meta_sidecar(), "meta_classifier_meta")
    pkg.write_manifest()
    _assert_meta_untouched("manifest_invalid")


def test_meta_tampered_bytes_are_refused_before_deserialization(pkg, joblib):
    _meta_sentinels()
    pkg.add("meta_clf.pkl", _pickled(FakeModel("meta")), "meta_classifier")
    pkg.add("meta_clf.json", _meta_sidecar(), "meta_classifier_meta")
    pkg.write_manifest()
    (pkg.root / "meta_clf.pkl").write_bytes(_canary("tampered-meta"))
    _assert_meta_untouched("mismatch")


@pytest.mark.parametrize("change", ["modified", "missing", "invalid_fields"])
def test_meta_side_car_is_never_replaced_by_defaults(pkg, joblib, change):
    _meta_sentinels()
    pkg.add("meta_clf.pkl", _canary("meta-must-not-load"), "meta_classifier")
    if change == "invalid_fields":
        pkg.add("meta_clf.json", json.dumps({"threshold": 0.41}).encode(), "meta_classifier_meta")
    else:
        pkg.add("meta_clf.json", _meta_sidecar(), "meta_classifier_meta")
    pkg.write_manifest()
    if change == "modified":
        (pkg.root / "meta_clf.json").write_bytes(_meta_sidecar(threshold=0.01))
    elif change == "missing":
        (pkg.root / "meta_clf.json").unlink()
    _assert_meta_untouched({"modified": "mismatch", "missing": "missing",
                            "invalid_fields": "deserialize_failed"}[change])


def test_meta_missing_dependency_is_reported_as_such(pkg, monkeypatch):
    _meta_sentinels()
    monkeypatch.setitem(sys.modules, "joblib", None)
    assert pair._load_meta_classifier() is False
    assert pair._meta_unavailable_detail() == "dependency"


def test_meta_refusal_is_logged_with_its_code_and_no_location(pkg, joblib, caplog, tmp_path):
    pkg.add("meta_clf.pkl", _pickled(FakeModel("meta")), "meta_classifier")
    pkg.add("meta_clf.json", _meta_sidecar(), "meta_classifier_meta")
    pkg.write_manifest()
    (pkg.root / "meta_clf.pkl").write_bytes(b"altered")
    with caplog.at_level(logging.WARNING, logger="fie.layers.pair"):
        assert pair._load_meta_classifier() is False
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "reason=load_failed detail=mismatch artifact=meta_clf.pkl" in text
    _no_tmp_detail(text, tmp_path)


def test_meta_is_attempted_once(pkg, joblib):
    pkg.full(selected="v6_3b")
    with recording_opens() as log:
        assert pair._load_meta_classifier() is False      # no manifest yet
        pkg.write_manifest()
        assert pair._load_meta_classifier() is False      # not retried within the process
    assert _opened(log, pkg.root / "meta_clf.pkl") == 0


# ══ 3. The PAIR classifier ═══════════════════════════════════════════════════

NOT_LOADED = {"loaded": False, "version": None, "digest": None, "threshold": None, "backend": None}


def _assert_pair_refused(state, detail, tmp_path):
    assert pair._load_pair_classifier() is False
    assert pair._pair_unavailable_state() == state
    assert pair._pair_unavailable_detail_code() == detail
    assert pair._pair_clf is None and pair._pair_embedder is None and pair._pair_identity == {}
    assert pair._model_identities()["pair_classifier"] == NOT_LOADED
    assert pair._pair_state()["loaded"] is False
    assert CANARY_FIRED == [], "a refused artifact was deserialized"
    _no_tmp_detail(pair._pair_load_error, tmp_path)


@pytest.fixture
def unpickle_spy(monkeypatch):
    calls: list[str] = []
    real = vm.unpickle

    def spy(entry, data):
        calls.append(entry.path)
        return real(entry, data)

    monkeypatch.setattr(vm, "unpickle", spy)
    return calls


@pytest.mark.parametrize("value", [None, "", "   "])
def test_default_is_selected_when_no_version_is_requested(pkg, joblib, monkeypatch, caplog, value):
    if value is not None:
        monkeypatch.setenv("FIE_PAIR_VERSION", value)
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    with caplog.at_level(logging.INFO, logger="fie.layers.pair"):
        assert pair._load_pair_classifier() is True
    assert pair._pair_clf.tag == "v6_3b" and CANARY_FIRED == []
    entry = pkg.entry(CLASSIFIER_FILES["v6_3b"])
    assert pair._pair_identity == {"version": "t-v6_3b", "digest": entry.sha256[:16]}
    assert pair._pair_threshold == 0.5 and pkg.embedders == [EMBED_MODEL]
    assert pair._pair_unavailable_state() is None and pair._pair_unavailable_detail_code() is None
    ready = [r.getMessage() for r in caplog.records if "status=ready" in r.getMessage()]
    assert ready == [f"layer=pair_classifier status=ready model=pair_intent_classifier_v6_3b.pkl threshold=0.50 "
                     f"backend=FakeEmbedder embed_model={EMBED_MODEL}"]


@pytest.mark.parametrize("key", VERSIONS)
def test_each_supported_version_selects_exactly_its_entry(pkg, joblib, monkeypatch, key):
    monkeypatch.setenv("FIE_PAIR_VERSION", key.upper() if key == "v6_3" else key)   # matching is case-insensitive
    pkg.full(selected=key)
    pkg.write_manifest()
    with recording_opens() as log:
        assert pair._load_pair_classifier() is True
    assert pair._pair_clf.tag == key and CANARY_FIRED == []
    assert pair._pair_identity["digest"] == pkg.entry(CLASSIFIER_FILES[key]).sha256[:16]
    assert pair._pair_threshold == 0.5 + VERSIONS.index(key) / 100
    for other in VERSIONS:
        expected = 1 if other == key else 0
        assert _opened(log, pkg.root / CLASSIFIER_FILES[other]) == expected, other
        assert _opened(log, pkg.root / SIDECAR_FILES[other]) == expected, other


def test_unlisted_version_v2_is_refused_even_though_its_file_exists(pkg, joblib, monkeypatch, tmp_path):
    monkeypatch.setenv("FIE_PAIR_VERSION", "v2")
    pkg.full(selected="v6_3b")
    pkg.add("pair_intent_classifier_v2.pkl", _canary("v2"), "pair_classifier", "v2", listed=False)
    pkg.add("pair_intent_meta_v2.json", _sidecar(), "pair_meta", "v2", listed=False)
    pkg.write_manifest()
    with recording_opens() as log:
        _assert_pair_refused("unavailable_load_failed", "not_listed", tmp_path)
    for name in ("pair_intent_classifier_v2.pkl", "pair_intent_meta_v2.json", CLASSIFIER_FILES["v6_3b"]):
        assert _opened(log, pkg.root / name) == 0, f"{name} was opened"


def test_listed_version_with_a_removed_file_is_not_replaced_by_the_default(pkg, joblib, monkeypatch, tmp_path):
    monkeypatch.setenv("FIE_PAIR_VERSION", "v5")
    pkg.full(selected="v6_3b")             # the default is valid and loadable: it must still not be used
    pkg.write_manifest()
    (pkg.root / CLASSIFIER_FILES["v5"]).unlink()
    with recording_opens() as log:
        _assert_pair_refused("unavailable_model", "missing", tmp_path)
    assert _opened(log, pkg.root / CLASSIFIER_FILES["v6_3b"]) == 0


def test_tampered_classifier_is_refused_before_deserialization(pkg, joblib, tmp_path, unpickle_spy):
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    (pkg.root / CLASSIFIER_FILES["v6_3b"]).write_bytes(_canary("tampered"))
    _assert_pair_refused("unavailable_load_failed", "mismatch", tmp_path)
    assert unpickle_spy == [] and pkg.embedders == []


def test_unlisted_model_files_are_never_opened(pkg, joblib):
    pkg.full(selected="v6_3b")
    unlisted = ("pair_intent_classifier_v64_harmaug.pkl", "pair_intent_meta_v64_harmaug.json",
                "pair_intent_classifier_v2.pkl", "pair_intent_classifier_v6_noaug.pkl", "meta_clf_old.pkl")
    for name in unlisted:
        pkg.add(name, _canary(name) if name.endswith(".pkl") else _sidecar(), "pair_classifier", listed=False)
    pkg.write_manifest()
    with recording_opens() as log:
        assert pair._load_pair_classifier() is True
        assert pair._load_meta_classifier() is True
    assert CANARY_FIRED == []
    for name in unlisted:
        assert _opened(log, pkg.root / name) == 0, name


def test_the_bytes_deserialized_are_the_bytes_verified(pkg, joblib, monkeypatch):
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    seen = []
    real_load = joblib.load

    def spy(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO), "a model was deserialized from something other than verified bytes"
        seen.append(hashlib.sha256(source.getvalue()).hexdigest())
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", spy)
    with recording_opens() as log:
        assert pair._load_pair_classifier() is True
    assert seen == [pkg.entry(CLASSIFIER_FILES["v6_3b"]).sha256]
    assert _opened(log, pkg.root / CLASSIFIER_FILES["v6_3b"]) == 1, "the file was opened again after verification"


def test_replacing_the_file_after_verification_changes_nothing(pkg, joblib, monkeypatch):
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    target = pkg.root / CLASSIFIER_FILES["v6_3b"]
    real = mi.read_verified

    def read_then_swap(models_dir, entry):
        data = real(models_dir, entry)
        if entry.path == CLASSIFIER_FILES["v6_3b"]:
            target.write_bytes(_canary("swapped-after-verification"))
        return data

    monkeypatch.setattr(mi, "read_verified", read_then_swap)
    assert pair._load_pair_classifier() is True
    assert pair._pair_clf.tag == "v6_3b" and CANARY_FIRED == []
    assert pair._pair_identity["digest"] == pkg.entry(CLASSIFIER_FILES["v6_3b"]).sha256[:16]
    assert hashlib.sha256(target.read_bytes()).hexdigest() != pkg.entry(CLASSIFIER_FILES["v6_3b"]).sha256


def _sentinel_threshold():
    pair._pair_threshold = 0.777


@pytest.mark.parametrize("field, value", [("threshold", 0.01), ("embed_model", "attacker/lookalike-encoder")])
def test_altered_pair_side_car_affects_nothing(pkg, joblib, tmp_path, unpickle_spy, field, value):
    _sentinel_threshold()
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    body = json.loads((pkg.root / SIDECAR_FILES["v6_3b"]).read_bytes())
    body[field] = value
    (pkg.root / SIDECAR_FILES["v6_3b"]).write_bytes(json.dumps(body).encode())
    _assert_pair_refused("unavailable_load_failed", "mismatch", tmp_path)
    assert pkg.embedders == [], "the encoder fallback was built from an unverified side-car"
    assert unpickle_spy == [] and pair._pair_threshold == 0.777


def test_missing_pair_side_car_does_not_mean_a_default_threshold(pkg, joblib, tmp_path, unpickle_spy):
    _sentinel_threshold()
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    (pkg.root / SIDECAR_FILES["v6_3b"]).unlink()
    _assert_pair_refused("unavailable_model", "missing", tmp_path)
    assert pkg.embedders == [] and unpickle_spy == [] and pair._pair_threshold == 0.777


@pytest.mark.parametrize("drop", ["threshold", "embed_model"])
def test_listed_side_car_without_a_required_field_is_refused(pkg, joblib, tmp_path, unpickle_spy, drop):
    _sentinel_threshold()
    pkg.add(CLASSIFIER_FILES["v6_3b"], _canary("v6_3b"), "pair_classifier", "v6_3b", default=True)
    body = {"threshold": 0.5, "embed_model": EMBED_MODEL}
    del body[drop]
    pkg.add(SIDECAR_FILES["v6_3b"], json.dumps(body).encode(), "pair_meta", "v6_3b")
    pkg.write_manifest()
    _assert_pair_refused("unavailable_load_failed", "deserialize_failed", tmp_path)
    assert pkg.embedders == [] and unpickle_spy == [] and pair._pair_threshold == 0.777


def test_the_side_car_bytes_parsed_are_the_bytes_verified(pkg, joblib, monkeypatch):
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    parsed = []
    real = vm.parse_json

    def spy(entry, data):
        parsed.append((entry.path, hashlib.sha256(data).hexdigest()))
        return real(entry, data)

    monkeypatch.setattr(vm, "parse_json", spy)
    with recording_opens() as log:
        assert pair._load_pair_classifier() is True
    side = pkg.entry(SIDECAR_FILES["v6_3b"])
    assert parsed == [(side.path, side.sha256)]
    assert _opened(log, pkg.root / side.path) == 1


def _deny_reading(monkeypatch, target: Path):
    import builtins

    real_open = builtins.open

    def deny(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and os.path.normcase(os.fspath(path)) == \
                os.path.normcase(os.path.realpath(target)):
            raise PermissionError(13, "Access is denied", os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny)


@pytest.mark.parametrize("case, state, detail", [
    ("manifest_missing", "unavailable_model", "manifest_missing"),
    ("manifest_invalid", "unavailable_load_failed", "manifest_invalid"),
    ("missing", "unavailable_model", "missing"),
    ("not_listed", "unavailable_load_failed", "not_listed"),
    ("mismatch", "unavailable_load_failed", "mismatch"),
    ("read_failed", "unavailable_load_failed", "read_failed"),
    ("path_violation", "unavailable_load_failed", "path_violation"),
    ("deserialize_failed", "unavailable_load_failed", "deserialize_failed"),
    ("dependency", "unavailable_dependency", "dependency"),
])
def test_each_failure_keeps_its_public_state_and_its_own_detail(pkg, joblib, monkeypatch, caplog, tmp_path,
                                                                 case, state, detail):
    target = pkg.root / CLASSIFIER_FILES["v6_3b"]
    if case != "manifest_missing":
        pkg.full(selected="v6_3b")
        pkg.write_manifest(b'{"schema": 1, "artifacts": {}}' if case == "manifest_invalid" else None)
    if case == "missing":
        target.unlink()
    elif case == "not_listed":
        monkeypatch.setenv("FIE_PAIR_VERSION", "v9")
    elif case == "mismatch":
        target.write_bytes(_canary("mismatch"))
    elif case == "read_failed":
        _deny_reading(monkeypatch, target)
    elif case == "path_violation":
        target.unlink()
        target.mkdir()                     # something is there, and it is not a regular file
    elif case == "deserialize_failed":
        pkg.entries = [e for e in pkg.entries if e["path"] != target.name]
        pkg.add(target.name, b"listed, verified, and not a pickle", "pair_classifier", "v6_3b", default=True)
        pkg.write_manifest()
    elif case == "dependency":
        monkeypatch.setitem(sys.modules, "joblib", None)
    with caplog.at_level(logging.WARNING, logger="fie.layers.pair"):
        _assert_pair_refused(state, detail, tmp_path)
    if case == "dependency":
        assert pair._pair_load_error.startswith("missing dependency: "), "the harness and warmup rely on this text"
        return
    records = [r for r in caplog.records if "layer=pair_classifier" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert f"detail={detail}" in message and f"reason={state[len('unavailable_'):]}" in message
    assert records[0].levelno == (logging.WARNING if state == "unavailable_model" else logging.ERROR)
    _no_tmp_detail(message, tmp_path)


def test_a_programming_error_is_not_reported_as_an_integrity_failure(pkg, joblib, monkeypatch, caplog, tmp_path):
    pkg.full(selected="v6_3b")
    pkg.write_manifest()

    def defect(models_dir, entry):
        raise TypeError("a defect, not a file problem")

    monkeypatch.setattr(vm, "read_artifact", defect)
    with caplog.at_level(logging.ERROR, logger="fie.layers.pair"):
        assert pair._load_pair_classifier() is False
    assert pair._pair_unavailable_state() == "unavailable_load_failed"
    assert pair._pair_unavailable_detail_code() == "unexpected"
    assert pair._pair_load_error == "TypeError: a defect, not a file problem"
    assert any(r.exc_info for r in caplog.records), "the traceback is kept for a defect"


def test_concurrent_first_loads_verify_and_deserialize_once(pkg, joblib, unpickle_spy):
    import threading

    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(pair._load_pair_classifier())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    # Callers that arrive while another thread is inside the first load get False without
    # waiting (an existing contract: that scan reports the layer as not ready). Whatever the
    # interleaving, the artifact is verified and deserialized exactly once.
    assert len(results) == 8 and True in results
    assert pair._load_pair_classifier() is True
    assert unpickle_spy == [CLASSIFIER_FILES["v6_3b"]]


# ══ 4. The deserializer boundary ═════════════════════════════════════════════
# A static check over every module in fie/: a call that deserializes (pickle, joblib, dill,
# cloudpickle, torch.load, marshal) may appear only in _verified_models.py. The session-data
# store in session_tracker.py is not a model loader; it is listed separately so that its one
# known call is pinned as well, and any second call there is caught.

FIE = ROOT / "fie"
VERIFIED_LOADER = "fie/layers/_verified_models.py"
SESSION_DATA = "fie/session_tracker.py"          # out of scope for Stage 3.2: session data, not a model
_DESERIALIZERS = {
    "pickle": {"load", "loads", "Unpickler"}, "_pickle": {"load", "loads", "Unpickler"},
    "cPickle": {"load", "loads", "Unpickler"}, "joblib": {"load"}, "dill": {"load", "loads", "Unpickler"},
    "cloudpickle": {"load", "loads"}, "torch": {"load"}, "marshal": {"load", "loads"},
    "numpy": {"load"}, "pandas": {"read_pickle"},
}


def _deserializer_calls(source: str) -> list[str]:
    """Deserializing calls in `source`, as 'module.function', following import aliases."""
    tree = ast.parse(source)
    module_alias: dict[str, str] = {}      # local name -> module
    function_alias: dict[str, str] = {}    # local name -> "module.function"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in _DESERIALIZERS:
                    module_alias[a.asname or root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in _DESERIALIZERS:
            root = node.module.split(".")[0]
            for a in node.names:
                if a.name in _DESERIALIZERS[root]:
                    function_alias[a.asname or a.name] = f"{root}.{a.name}"
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in module_alias:
            root = module_alias[f.value.id]
            if f.attr in _DESERIALIZERS[root]:
                found.append(f"{root}.{f.attr}")
        elif isinstance(f, ast.Name) and f.id in function_alias:
            found.append(function_alias[f.id])
    return sorted(found)


def test_the_boundary_check_can_see_a_direct_call():
    """Positive control for the static check: each form of a direct call is detected."""
    assert _deserializer_calls("import joblib\njoblib.load(p)\n") == ["joblib.load"]
    assert _deserializer_calls("import joblib as jl\njl.load(p)\n") == ["joblib.load"]
    assert _deserializer_calls("from joblib import load\nload(p)\n") == ["joblib.load"]
    assert _deserializer_calls("from pickle import loads as l\nl(b)\n") == ["pickle.loads"]
    assert _deserializer_calls("import pickle\npickle.Unpickler(f).load()\n") == ["pickle.Unpickler"]
    assert _deserializer_calls("def f():\n    import torch\n    return torch.load(p)\n") == ["torch.load"]
    assert _deserializer_calls("import json\njson.load(f)\n") == []


def test_model_deserialization_happens_only_in_the_verified_loader():
    found = {}
    for path in sorted(FIE.rglob("*.py")):
        calls = _deserializer_calls(path.read_text(encoding="utf-8"))
        if calls:
            found[path.relative_to(ROOT).as_posix()] = calls
    assert found.pop(SESSION_DATA, None) == ["pickle.loads"], "session data: exactly the one known call"
    assert found == {VERIFIED_LOADER: ["joblib.load"]}


def test_the_verified_loader_deserializes_only_inside_unpickle():
    """Inside the module, the one call sits in unpickle(), which re-checks the bytes first."""
    tree = ast.parse((ROOT / VERIFIED_LOADER).read_text(encoding="utf-8"))
    owners = [fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
              if _deserializer_calls(ast.unparse(fn)) and fn.name != "<module>"]
    assert owners == ["unpickle"]


# ══ 5. The real artifacts against their approved entries ═════════════════════
# Hashes only: these tests read the released files and compare digests. Nothing is deserialized.

TRACKED = ROOT / "scripts" / "model_manifest.json"
RUNTIME = ROOT / "fie" / "models" / "MANIFEST.json"
_MISSING_RUNTIME = ("fie/models/MANIFEST.json is absent, so the released artifacts cannot be checked here. "
                    "Generate it (it downloads nothing when every artifact is already present and verified): "
                    "python scripts/download_models.py")


@pytest.fixture(scope="module")
def runtime_manifest():
    if not RUNTIME.is_file():
        pytest.fail(_MISSING_RUNTIME)       # a limitation to fix, not a reason to pass silently
    return RUNTIME.read_bytes()


def test_runtime_manifest_is_the_derivation_of_the_tracked_inventory(runtime_manifest):
    assert runtime_manifest == mi.derive_package_manifest(TRACKED.read_bytes()), \
        "fie/models/MANIFEST.json differs from what scripts/model_manifest.json derives; regenerate it, never edit it"
    mi.parse_manifest(runtime_manifest)


def test_released_pair_and_meta_artifacts_match_their_approved_entries(runtime_manifest):
    tracked = {a["path"]: a for a in json.loads(TRACKED.read_bytes())["artifacts"]}
    manifest = mi.parse_manifest(runtime_manifest)
    meta, meta_side = vm.select_meta(manifest)
    selected = [meta, meta_side]
    for key in VERSIONS:
        selected.extend(vm.select_pair(manifest, key))
    assert len(selected) == 16
    for entry in selected:
        approved = tracked[mi.PACKAGE_PREFIX + entry.path]
        assert (entry.sha256, entry.size) == (approved["sha256"], approved["size"]), entry.path
        data = vm.read_artifact(pair._PKG_MODELS, entry)       # raises unless size and SHA-256 match
        assert hashlib.sha256(data).hexdigest() == approved["sha256"]
    assert meta.path == "meta_clf.pkl" and meta.sha256 == APPROVED_META_SHA256
    default, _ = vm.select_pair(manifest, None)
    assert (default.version, default.path) == ("v6_3b", "pair_intent_classifier_v6_3b.pkl")


# ══ 6. The repository-root models/ directory has no say ══════════════════════

_REAL_RESOLVE = pair._resolve_models_dir


def test_the_loader_reads_only_the_package_directory():
    assert not hasattr(pair, "_REPO_MODELS")
    for sentinel in ("pair_intent_classifier.pkl", "pair_intent_classifier_v6_3b.pkl", "meta_clf.pkl",
                     "no_such_file.pkl", None):
        assert _REAL_RESOLVE(sentinel) == pair._PKG_MODELS == ROOT / "fie" / "models"


def test_repository_root_models_cannot_select_or_override_a_package_artifact(pkg, joblib, monkeypatch, tmp_path):
    """
    A source-checkout layout: <root>/fie/models (the package) and <root>/models (the repository
    root, which really does hold old pickles in this repository). Files planted at the
    repository root, under every name the loader could want, change nothing: the package file
    loads, a listed file missing from the package stays missing, and the root is never opened.
    """
    root = tmp_path / "checkout"
    package_models = root / "fie" / "models"
    repo_models = root / "models"
    package_models.mkdir(parents=True)
    repo_models.mkdir()
    package = Package(package_models).full(selected="v6_3b")
    package.write_manifest()
    for name in list(CLASSIFIER_FILES.values()) + ["meta_clf.pkl"]:
        (repo_models / name).write_bytes(_canary(f"root:{name}"))
    # The strongest case: a root copy carrying exactly the listed bytes of a classifier that
    # is missing from the package. Picking it up would even pass verification.
    (repo_models / CLASSIFIER_FILES["v5"]).write_bytes((package_models / CLASSIFIER_FILES["v5"]).read_bytes())
    (package_models / CLASSIFIER_FILES["v5"]).unlink()
    for name in list(SIDECAR_FILES.values()) + ["meta_clf.json"]:
        (repo_models / name).write_bytes(_sidecar(threshold=0.01))

    monkeypatch.setattr(pair, "_PKG_MODELS", package_models)
    monkeypatch.setattr(pair, "_resolve_models_dir", _REAL_RESOLVE)
    monkeypatch.setattr(vm, "MANIFEST_PATH", package_models / mi.PACKAGE_MANIFEST_NAME)

    with recording_opens() as log:
        assert pair._load_pair_classifier() is True
        assert pair._load_meta_classifier() is True
    assert pair._pair_clf.tag == "v6_3b" and pair._meta_clf.tag == "meta" and CANARY_FIRED == []
    assert pair._pair_threshold == 0.5 and pair._meta_threshold() == 0.41

    pair._pair_clf = pair._pair_embedder = None
    pair._pair_load_attempted = False
    monkeypatch.setenv("FIE_PAIR_VERSION", "v5")
    with recording_opens() as log2:
        _assert_pair_refused("unavailable_model", "missing", tmp_path)
    for name in os.listdir(repo_models):
        assert _opened(log + log2, repo_models / name) == 0, f"the repository-root {name} was opened"


# ══ 7. warmup() classifies by code ═══════════════════════════════════════════

@pytest.fixture
def warmup_only(monkeypatch):
    """warmup() with the meta-classifier and the layer probe stubbed: only the PAIR outcome is real."""
    import fie.adversarial as adv

    monkeypatch.setattr(adv, "_load_meta_classifier", lambda: False)
    monkeypatch.setattr(adv, "_run_all_layers_parallel", lambda *args, **kwargs: None)
    return adv


@pytest.mark.parametrize("case, expected, detail", [
    ("ready", "ready", None),
    ("manifest_missing", "unavailable", "manifest_missing"),
    ("missing", "unavailable", "missing"),
    ("dependency", "unavailable", "dependency"),
    ("mismatch", "failed", "mismatch"),
    ("not_listed", "failed", "not_listed"),
    ("manifest_invalid", "failed", "manifest_invalid"),
    ("deserialize_failed", "failed", "deserialize_failed"),
    ("unexpected", "failed", "unexpected"),
])
def test_warmup_classifies_by_detail_code(pkg, joblib, warmup_only, monkeypatch, caplog, case, expected, detail):
    if case != "manifest_missing":
        pkg.full(selected="v6_3b")
        pkg.write_manifest(b"[]" if case == "manifest_invalid" else None)
    target = pkg.root / CLASSIFIER_FILES["v6_3b"]
    if case == "missing":
        target.unlink()
    elif case == "dependency":
        monkeypatch.setitem(sys.modules, "joblib", None)
    elif case == "mismatch":
        target.write_bytes(_canary("warmup"))
    elif case == "not_listed":
        monkeypatch.setenv("FIE_PAIR_VERSION", "v2")
    elif case == "deserialize_failed":
        pkg.entries = [e for e in pkg.entries if e["path"] != target.name]
        pkg.add(target.name, b"not a pickle", "pair_classifier", "v6_3b", default=True)
        pkg.write_manifest()
    elif case == "unexpected":
        monkeypatch.setattr(vm, "read_artifact", lambda models_dir, entry: (_ for _ in ()).throw(TypeError("defect")))
    with caplog.at_level(logging.ERROR, logger="fie.adversarial"):
        status = warmup_only.warmup()
    assert status["pair_classifier"] == expected
    assert pair._pair_unavailable_detail_code() == detail
    failed_lines = [r.getMessage() for r in caplog.records if "warmup: pair_classifier failed" in r.getMessage()]
    assert failed_lines == ([f"warmup: pair_classifier failed detail={detail}"] if expected == "failed" else [])
    assert CANARY_FIRED == []


def test_warmup_does_not_read_the_loader_message_text():
    import fie.adversarial as adv

    tree = ast.parse(inspect.getsource(adv.warmup))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    texts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "_pair_state" not in names and "_pair_unavailable_detail_code" in names
    assert not any("no PAIR classifier found" in t or "missing dependency" in t for t in texts)


# ══ 8. Public contracts on an integrity refusal ══════════════════════════════

@pytest.fixture
def scanner(monkeypatch, tmp_path):
    """The real scanner, isolated: no tiebreaker, no translator, no labels, no cache."""
    import fie.adversarial as adv
    import fie.feedback_store as fb
    import fie.llama_guard as lg
    import fie.multilingual as ml
    import fie.preflight as pf

    def tiebreaker_down(prompt):
        raise RuntimeError("tiebreaker unreachable (test stand-in)")

    for name in ("FIE_POLICY", "FIE_UNCERTAIN_ALLOW", "FIE_DISABLE_META", "FIE_EMBED_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fb, "_LOCAL_PATH", tmp_path / "flagged_events.jsonl")
    monkeypatch.setattr(fb, "_KNOWN_ATTACK_HASHES", set())
    monkeypatch.setattr(fb, "_WHITELIST_HASHES", set())
    monkeypatch.setattr(lg, "query_llama_guard", tiebreaker_down)
    monkeypatch.setattr(ml, "translate_to_english", lambda text, timeout=3.0: None)
    monkeypatch.setattr(pf, "_FAIL_SECURE", False)
    monkeypatch.setattr(pf, "_get_block_enabled", lambda: True)
    adv._scan_cache._cache.clear()
    yield adv
    adv._scan_cache._cache.clear()


SOFT_HARM = "How can I secretly track my partner's phone without them knowing?"


@pytest.mark.parametrize("tamper", ["classifier", "side_car"])
def test_an_integrity_refusal_is_reported_truthfully(pkg, joblib, scanner, tmp_path, tamper):
    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    name = CLASSIFIER_FILES["v6_3b"] if tamper == "classifier" else SIDECAR_FILES["v6_3b"]
    (pkg.root / name).write_bytes(_canary("scan") if tamper == "classifier" else _sidecar(threshold=0.01))

    r = scanner.scan_prompt("What is the capital of France?", use_llama_guard=False)
    assert r.coverage.layers["pair_classifier"] == "unavailable_load_failed"
    assert r.coverage.classifier == "unavailable_load_failed" and r.coverage.status == "partial"
    assert r.degraded_layers == ["pair_classifier"]
    ident = r.models["pair_classifier"]
    assert (ident.loaded, ident.version, ident.digest, ident.threshold) == (False, None, None, None)
    assert scanner.classifier_state() == "unavailable_load_failed"
    text = json.dumps(r.to_dict()) + repr(r.coverage) + repr(r.models)
    for needle in ("mismatch", name, "pair_intent", ".pkl", ".json", str(tmp_path), tmp_path.name):
        assert needle not in text, f"a public field carries internal detail: {needle!r}"
    assert CANARY_FIRED == []


@pytest.mark.parametrize("preset, action, blocked", [("strict", "block", True), ("compat", "allow", False)])
def test_policy_on_an_integrity_refusal(pkg, joblib, scanner, preset, action, blocked):
    import fie.policy as policy
    import fie.preflight as pf

    pkg.full(selected="v6_3b")
    pkg.write_manifest()
    (pkg.root / CLASSIFIER_FILES["v6_3b"]).write_bytes(_canary("policy"))
    chosen = policy.STRICT if preset == "strict" else policy.compat_policy(scan_failure="allow")
    result = pf.preflight_check(SOFT_HARM, policy=chosen)
    assert (result.action, result.blocked) == (action, blocked)
    assert result.reasons[0] == "missing_required_layer" and result.coverage_status == "partial"
    assert pair._pair_unavailable_detail_code() == "mismatch" and CANARY_FIRED == []
