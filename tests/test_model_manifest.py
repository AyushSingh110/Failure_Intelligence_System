"""
tests/test_model_manifest.py — the model inventory, the manifest derived from it, and the
download script that writes it (WP-021 phase 3, stage 3.1).

Three things are covered:

  1. validate_production(): the rules scripts/model_manifest.json must satisfy.
  2. derive_package_manifest(): fie/models/MANIFEST.json as a pure function of that inventory.
  3. scripts/download_models.py: still downloads and verifies every entry, and writes the
     derived manifest only after every artifact has been verified.

Every model file here is a few synthetic bytes in a temporary directory. The real model files
are never opened; the tracked inventory is read as JSON only. No test touches the network: the
download function is replaced by one that writes prepared bytes.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from fie import _model_integrity as mi

REPO = Path(__file__).resolve().parent.parent
TRACKED = REPO / "scripts" / "model_manifest.json"
APPROVED_META_CLF_SHA256 = "be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6"
PAIR_KEYS = ["v6_3b", "v6_3", "v6", "v5", "v4", "v3", "v1"]
LEGACY = ["fie/models/failure_classifier_v4.pkl", "fie/models/feature_columns_v4.pkl"]
SERVER = ["models/failure_classifier_slim.pkl", "models/feature_columns_slim.pkl",
          "models/failure_classifier_slim_meta.json", "storage/faiss_adversarial.index",
          "storage/faiss_adversarial_meta.json"]


def content(path: str) -> bytes:
    """The synthetic bytes standing in for the artifact at `path`."""
    return f"synthetic artifact for {path}\n".encode("utf-8") * 3


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry(path: str, role: str, in_wheel: bool, version=None, default=False) -> dict:
    out = {"path": path, "sha256": sha(content(path)), "size": len(content(path)), "role": role}
    if version is not None:
        out["version"] = version
    out.update({"default": default, "in_wheel": in_wheel, "used_by": f"free text about {path}"})
    return out


def production() -> dict:
    """A complete, valid inventory with the real layout and synthetic digests."""
    artifacts = [entry("fie/models/minilm-onnx/model.onnx", "encoder", False),
                 entry("fie/models/minilm-onnx/tokenizer.json", "tokenizer", False)]
    for key in PAIR_KEYS:
        suffix = "" if key == "v1" else f"_{key}"
        artifacts.append(entry(f"fie/models/pair_intent_classifier{suffix}.pkl", "pair_classifier", True, key,
                               default=(key == "v6_3b")))
        artifacts.append(entry(f"fie/models/pair_intent_meta{suffix}.json", "pair_meta", True, key))
    artifacts += [entry("fie/models/meta_clf.pkl", "meta_classifier", True),
                  entry("fie/models/meta_clf.json", "meta_classifier_meta", True)]
    artifacts += [entry(p, "legacy", False) for p in LEGACY]
    artifacts += [entry(p, "server", False) for p in SERVER]
    return {"_comment": "synthetic", "schema": 1, "release_tag": "models-test", "repo": "owner/repo",
            "artifacts": artifacts}


def find(data: dict, path: str) -> dict:
    return next(a for a in data["artifacts"] if a["path"] == path)


def changed(target_path: str, /, **fields) -> dict:
    """The valid inventory with one entry altered (a field set to ... is removed)."""
    data = production()
    target = find(data, target_path)
    for key, value in fields.items():
        if value is ...:
            target.pop(key, None)
        else:
            target[key] = value
    return data


def rejected(data, fragment: str) -> mi.ManifestError:
    with pytest.raises(mi.ManifestError) as caught:
        mi.validate_production(data)
    assert fragment in str(caught.value), str(caught.value)
    assert caught.value.code == "manifest_invalid"
    return caught.value


DEFAULT_CLF = "fie/models/pair_intent_classifier_v6_3b.pkl"
V5_CLF, V5_META = "fie/models/pair_intent_classifier_v5.pkl", "fie/models/pair_intent_meta_v5.json"
META_CLF = "fie/models/meta_clf.pkl"


# ══ 1. the production inventory ══════════════════════════════════════════════

def test_the_synthetic_inventory_is_valid():
    m = mi.validate_production(production())
    assert len(m.entries) == 25
    assert mi.validate_production(json.dumps(production())) == m
    assert mi.validate_production(json.dumps(production()).encode("utf-8")) == m


def test_role_vocabulary_is_exactly_the_approved_eight():
    assert mi.ROLES == {"pair_classifier", "pair_meta", "meta_classifier", "meta_classifier_meta",
                        "encoder", "tokenizer", "server", "legacy"}
    assert {a["role"] for a in production()["artifacts"]} == mi.ROLES


@pytest.mark.parametrize("role", ["classifier", "model", "pair_classifier_meta", "Server", "experiment", "sdk"])
def test_an_unapproved_role_is_rejected(role):
    rejected(changed("models/failure_classifier_slim.pkl", role=role), "is not one of")


def test_supported_selection_keys_are_exactly_the_approved_seven():
    assert list(mi.PAIR_VERSIONS) == PAIR_KEYS and mi.DEFAULT_PAIR_VERSION == "v6_3b"
    m = mi.validate_production(production())
    assert sorted(e.version for e in m.for_role("pair_classifier")) == sorted(PAIR_KEYS)
    assert sorted(e.version for e in m.for_role("pair_meta")) == sorted(PAIR_KEYS)


@pytest.mark.parametrize("field", ["path", "sha256", "size", "role", "in_wheel"])
def test_fields_required_on_every_entry(field):
    with pytest.raises(mi.ManifestError) as caught:
        mi.validate_production(changed(META_CLF, **{field: ...}))
    assert "missing required field(s)" in str(caught.value)


@pytest.mark.parametrize("path", [DEFAULT_CLF, V5_META, META_CLF, LEGACY[0], SERVER[0],
                                  "fie/models/minilm-onnx/model.onnx"])
def test_default_must_be_stated_on_every_entry(path):
    rejected(changed(path, default=...), "'default' must be stated on every entry")


@pytest.mark.parametrize("path", [V5_CLF, V5_META])
def test_version_is_required_for_classifiers_and_their_side_cars(path):
    rejected(changed(path, version=...), "is not a supported selection key")


@pytest.mark.parametrize("version", ["v2", "v7", "v6.2", "v6.3b", "V5", "v6_3B", "default", "v64_harmaug"])
def test_a_version_outside_the_approved_keys_is_rejected(version):
    """Among others: the side-cars' declared versions (v6.2, v6.3b) are not selection keys."""
    rejected(changed(V5_CLF, version=version), "is not a supported selection key")


def test_a_selection_key_must_sit_on_its_own_file():
    """`v6` selects pair_intent_classifier_v6.pkl and nothing else."""
    data = production()
    find(data, V5_CLF)["version"], find(data, "fie/models/pair_intent_classifier_v4.pkl")["version"] = "v4", "v5"
    rejected(data, "belongs to")
    rejected(changed(V5_META, version="v4"), "more than once")


def test_the_original_classifier_is_v1():
    m = mi.validate_production(production())
    assert mi.select_entry(m, "pair_classifier", "v1").path == "fie/models/pair_intent_classifier.pkl"
    assert mi.select_entry(m, "pair_meta", "v1").path == "fie/models/pair_intent_meta.json"
    assert mi.select_entry(m, "pair_classifier", "v6").path == "fie/models/pair_intent_classifier_v6.pkl"


@pytest.mark.parametrize("path", [META_CLF, "fie/models/meta_clf.json", "fie/models/minilm-onnx/model.onnx",
                                  "fie/models/minilm-onnx/tokenizer.json", LEGACY[0], SERVER[0]])
def test_no_version_is_invented_for_roles_selected_by_role(path):
    rejected(changed(path, version="v1"), "carries no version")


def test_duplicate_role_and_version_identities_are_rejected():
    data = production()
    data["artifacts"].append({**find(data, V5_CLF), "path": "fie/models/copy_of_v5.pkl"})
    with pytest.raises(mi.ManifestError) as caught:
        mi.validate_production(data)
    assert "lists version 'v5' more than once" in str(caught.value)


def test_duplicate_paths_are_rejected():
    data = production()
    data["artifacts"].append(copy.deepcopy(find(data, SERVER[0])))
    with pytest.raises(mi.ManifestError) as caught:
        mi.validate_production(data)
    assert "duplicate path" in str(caught.value)


# ── the default ─────────────────────────────────────────────────────────────

def test_exactly_one_default_and_it_is_v6_3b():
    m = mi.validate_production(production())
    assert [e.path for e in m.entries if e.default] == [DEFAULT_CLF]
    assert mi.default_entry(m, "pair_classifier").version == "v6_3b"


def test_no_default_is_rejected():
    rejected(changed(DEFAULT_CLF, default=False), "exactly one entry is the default")


def test_a_second_default_classifier_is_rejected():
    with pytest.raises(mi.ManifestError) as caught:
        mi.validate_production(changed(V5_CLF, default=True))
    assert "default" in str(caught.value)


def test_another_version_cannot_take_the_default():
    data = changed(DEFAULT_CLF, default=False)
    find(data, V5_CLF)["default"] = True
    rejected(data, "only the 'v6_3b' pair_classifier is the default")


@pytest.mark.parametrize("path", [META_CLF, "fie/models/pair_intent_meta_v6_3b.json",
                                  "fie/models/minilm-onnx/model.onnx", LEGACY[0], SERVER[0]])
def test_only_a_pair_classifier_can_be_the_default(path):
    """The meta-classifier is a singleton chosen by role. It is not a second default."""
    with pytest.raises(mi.ManifestError) as caught:
        mi.validate_production(changed(path, default=True))
    assert "default" in str(caught.value)


# ── classifiers and side-cars ───────────────────────────────────────────────

def test_every_supported_version_must_be_listed():
    data = production()
    data["artifacts"] = [a for a in data["artifacts"] if a.get("version") != "v4"]
    rejected(data, "missing ['v4']")


def test_a_classifier_without_its_side_car_is_rejected():
    data = production()
    data["artifacts"] = [a for a in data["artifacts"] if a["path"] != V5_META]
    rejected(data, "without one: ['v5']")


def test_a_side_car_without_its_classifier_is_rejected():
    data = production()
    data["artifacts"] = [a for a in data["artifacts"] if a["path"] != V5_CLF]
    rejected(data, "missing ['v5']")


# ── singletons ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role, path", [
    ("meta_classifier", "fie/models/meta_clf.pkl"), ("meta_classifier_meta", "fie/models/meta_clf.json"),
    ("encoder", "fie/models/minilm-onnx/model.onnx"), ("tokenizer", "fie/models/minilm-onnx/tokenizer.json")])
def test_sole_entry_returns_the_one_entry_of_a_singleton_role(role, path):
    m = mi.validate_production(production())
    found = mi.sole_entry(m, role)
    assert found.path == path and found.version is None and found.default is False


def test_sole_entry_with_no_candidate_is_not_listed():
    m = mi.validate_production(production())
    with pytest.raises(mi.NotListed) as caught:
        mi.sole_entry(m, "no_such_role")
    assert (caught.value.code, caught.value.role, caught.value.version) == ("not_listed", "no_such_role", None)


def test_sole_entry_with_several_candidates_refuses_to_pick_one():
    """Seven classifiers: sole_entry must not return the first."""
    m = mi.validate_production(production())
    with pytest.raises(mi.ManifestError) as caught:
        mi.sole_entry(m, "pair_classifier")
    assert "must have exactly one entry, found 7" in str(caught.value)
    assert caught.value.code == "manifest_invalid"
    with pytest.raises(mi.ManifestError):
        mi.sole_entry(m, "server")


def test_sole_entry_on_a_generic_manifest_with_two_of_a_role():
    data = {"schema": 1, "artifacts": [
        {"path": "a.pkl", "sha256": sha(b"a"), "size": 1, "role": "meta_classifier", "in_wheel": True},
        {"path": "b.pkl", "sha256": sha(b"b"), "size": 1, "role": "meta_classifier", "in_wheel": True}]}
    with pytest.raises(mi.ManifestError) as caught:
        mi.sole_entry(mi.parse_manifest(data), "meta_classifier")
    assert "a.pkl" in str(caught.value) and "b.pkl" in str(caught.value)


@pytest.mark.parametrize("role", ["meta_classifier", "meta_classifier_meta", "encoder", "tokenizer"])
def test_a_missing_singleton_is_rejected(role):
    data = production()
    data["artifacts"] = [a for a in data["artifacts"] if a["role"] != role]
    rejected(data, f"role {role!r} must have exactly one entry, found 0")


def test_a_second_meta_classifier_is_rejected():
    data = production()
    data["artifacts"].append(entry("fie/models/meta_clf_v2.pkl", "meta_classifier", True))
    rejected(data, "role 'meta_classifier' must have exactly one entry, found 2")


# ── paths, roots, and what may ship ─────────────────────────────────────────

@pytest.mark.parametrize("path", ["engine/models/x.pkl", "data/model.pkl", "x.pkl", "fie/other/x.pkl",
                                  "scripts/x.pkl", "Models/x.pkl", "dist/x.pkl"])
def test_a_path_outside_the_approved_roots_is_rejected(path):
    data = production()
    data["artifacts"].append(entry(path, "server", False))
    rejected(data, "outside the approved roots")


@pytest.mark.parametrize("path", ["../fie/models/x.pkl", "/fie/models/x.pkl", "fie/models/../../x.pkl",
                                  "fie\\models\\x.pkl", "C:/fie/models/x.pkl"])
def test_an_unsafe_path_is_rejected_by_the_underlying_rules(path):
    data = production()
    data["artifacts"].append(entry(path, "server", False))
    with pytest.raises(mi.ManifestError):
        mi.validate_production(data)


def test_in_wheel_by_role_in_the_valid_inventory():
    m = mi.validate_production(production())
    shipped = {e.role for e in m.entries if e.in_wheel}
    kept_out = {e.role for e in m.entries if not e.in_wheel}
    assert shipped == {"pair_classifier", "pair_meta", "meta_classifier", "meta_classifier_meta"}
    assert kept_out == {"encoder", "tokenizer", "server", "legacy"}
    assert all(e.path.startswith("fie/models/") for e in m.entries if e.in_wheel)
    assert sum(e.in_wheel for e in m.entries) == 16


@pytest.mark.parametrize("path", [SERVER[0], SERVER[3], LEGACY[0], LEGACY[1],
                                  "fie/models/minilm-onnx/model.onnx", "fie/models/minilm-onnx/tokenizer.json"])
def test_what_does_not_ship_cannot_be_marked_as_shipping(path):
    rejected(changed(path, in_wheel=True), "must have in_wheel false")


@pytest.mark.parametrize("path", [DEFAULT_CLF, V5_CLF, V5_META, META_CLF, "fie/models/meta_clf.json"])
def test_what_ships_cannot_be_silently_dropped_from_the_wheel(path):
    rejected(changed(path, in_wheel=False), "must have in_wheel true")


def test_server_entries_live_outside_the_package_and_nothing_else_does():
    rejected(changed(SERVER[0], role="meta_classifier"), "'server' entries live under")
    data = production()
    data["artifacts"].append(entry("fie/models/server_copy.pkl", "server", False))
    rejected(data, "'server' entries live under")


def test_legacy_is_exactly_the_two_approved_paths():
    assert mi.LEGACY_PATHS == set(LEGACY)
    data = production()
    data["artifacts"].append(entry("fie/models/another_old_model.pkl", "legacy", False))
    rejected(data, "role 'legacy' belongs to exactly")


@pytest.mark.parametrize("path", LEGACY)
@pytest.mark.parametrize("role", ["pair_classifier", "meta_classifier", "server", "encoder"])
def test_a_legacy_file_cannot_be_given_another_role(path, role):
    rejected(changed(path, role=role), "role 'legacy' belongs to exactly")


@pytest.mark.parametrize("path", LEGACY)
def test_a_legacy_file_cannot_be_promoted_into_the_wheel(path):
    rejected(changed(path, in_wheel=True), "role 'legacy' must have in_wheel false")
    promoted = changed(path, role="meta_classifier", in_wheel=True)
    with pytest.raises(mi.ManifestError):
        mi.validate_production(promoted)


def test_the_derived_manifest_cannot_be_listed_in_its_own_source():
    data = production()
    data["artifacts"].append(entry("fie/models/MANIFEST.json", "pair_meta", True, "v5"))
    with pytest.raises(mi.ManifestError):
        mi.validate_production(data)
    data = production()
    find(data, "fie/models/meta_clf.json")["path"] = "fie/models/MANIFEST.json"
    rejected(data, "cannot list itself")


def test_schema_must_be_one():
    data = production()
    data["schema"] = 2
    with pytest.raises(mi.ManifestError) as caught:
        mi.validate_production(data)
    assert "schema must be 1" in str(caught.value)
    del data["schema"]
    with pytest.raises(mi.ManifestError):
        mi.validate_production(data)


def test_invalid_values_are_rejected_not_normalised():
    for field, value in (("in_wheel", "true"), ("in_wheel", 1), ("default", "false"), ("default", 0),
                         ("size", "11349"), ("sha256", sha(b"x").upper()), ("role", " server")):
        with pytest.raises(mi.ManifestError):
            mi.validate_production(changed(SERVER[0], **{field: value}))


def test_validation_does_not_read_any_file(tmp_path, monkeypatch):
    """The rules are about the inventory's own content. No artifact is opened."""
    opened = []
    monkeypatch.setattr(mi, "open", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError("opened")),
                        raising=False)
    mi.validate_production(production())
    mi.derive_package_manifest(production())
    assert opened == []


# ══ 2. the derived package manifest ══════════════════════════════════════════

def derived(data=None) -> dict:
    return json.loads(mi.derive_package_manifest(data or production()).decode("utf-8"))


def test_derivation_keeps_exactly_the_package_entries():
    data = production()
    expected = sorted(a["path"][len("fie/models/"):] for a in data["artifacts"] if a["path"].startswith("fie/models/"))
    got = [a["path"] for a in derived(data)["artifacts"]]
    assert got == expected and len(got) == 20


def test_derivation_excludes_server_entries():
    paths = [a["path"] for a in derived()["artifacts"]]
    assert not any(p.startswith(("models/", "storage/")) for p in paths)
    assert not any("failure_classifier_slim" in p or "faiss" in p for p in paths)
    assert all(a["role"] != "server" for a in derived()["artifacts"])


def test_derivation_retains_entries_that_are_not_shipped():
    by_path = {a["path"]: a for a in derived()["artifacts"]}
    for path in ("minilm-onnx/model.onnx", "minilm-onnx/tokenizer.json", "failure_classifier_v4.pkl",
                 "feature_columns_v4.pkl"):
        assert by_path[path]["in_wheel"] is False
    assert sum(a["in_wheel"] for a in by_path.values()) == 16 and len(by_path) == 20


def test_derivation_drops_free_text_and_keeps_only_runtime_fields():
    out = derived()
    assert sorted(out) == ["artifacts", "schema"] and out["schema"] == 1
    for a in out["artifacts"]:
        assert "used_by" not in a
        assert set(a) in ({"path", "sha256", "size", "role", "default", "in_wheel"},
                          {"path", "sha256", "size", "role", "default", "in_wheel", "version"})
        assert ("version" in a) == (a["role"] in ("pair_classifier", "pair_meta"))
    text = mi.derive_package_manifest(production()).decode("utf-8")
    assert "free text" not in text and "models-test" not in text and "owner/repo" not in text


def test_derivation_carries_digest_size_role_version_and_default_unchanged():
    data = production()
    by_path = {a["path"]: a for a in derived(data)["artifacts"]}
    for src in data["artifacts"]:
        if not src["path"].startswith("fie/models/"):
            continue
        out = by_path[src["path"][len("fie/models/"):]]
        for field in ("sha256", "size", "role", "default", "in_wheel"):
            assert out[field] == src[field]
        assert out.get("version") == src.get("version")
    assert [a["path"] for a in by_path.values() if a["default"]] == ["pair_intent_classifier_v6_3b.pkl"]


def test_derivation_is_byte_identical_across_repeated_calls():
    first = mi.derive_package_manifest(production())
    assert all(mi.derive_package_manifest(production()) == first for _ in range(5))
    assert mi.derive_package_manifest(json.dumps(production())) == first
    assert mi.derive_package_manifest(json.dumps(production(), indent=7).encode("utf-8")) == first


def test_derivation_does_not_depend_on_the_order_or_formatting_of_the_source():
    data = production()
    shuffled = copy.deepcopy(data)
    shuffled["artifacts"].reverse()
    shuffled["artifacts"] = [dict(reversed(list(a.items()))) for a in shuffled["artifacts"]]
    crlf = json.dumps(data, indent=2).replace("\n", "\r\n").encode("utf-8")
    first = mi.derive_package_manifest(data)
    assert mi.derive_package_manifest(shuffled) == first and mi.derive_package_manifest(crlf) == first


def test_derivation_serialisation_is_stable():
    raw = mi.derive_package_manifest(production())
    text = raw.decode("ascii")                               # ASCII only
    assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    paths = [a["path"] for a in json.loads(text)["artifacts"]]
    assert paths == sorted(paths)
    assert all(not line.startswith("\t") and (len(line) - len(line.lstrip(" "))) % 2 == 0 for line in text.splitlines())


def test_derivation_contains_nothing_environmental():
    text = mi.derive_package_manifest(production()).decode("utf-8")
    for needle in (str(REPO), REPO.name, str(Path.home()), "\\", "2026", "generated", "timestamp", "fie/models/"):
        assert needle not in text, needle


def test_the_derived_manifest_never_lists_itself():
    assert all(a["path"] != "MANIFEST.json" for a in derived()["artifacts"])
    assert "MANIFEST.json" not in mi.derive_package_manifest(production()).decode("utf-8")
    assert mi.PACKAGE_MANIFEST_NAME == "MANIFEST.json" and mi.PACKAGE_PREFIX == "fie/models/"


def test_the_derived_manifest_is_itself_a_valid_manifest_for_the_runtime_helpers():
    m = mi.parse_manifest(mi.derive_package_manifest(production()))
    assert mi.default_entry(m, "pair_classifier").path == "pair_intent_classifier_v6_3b.pkl"
    assert mi.select_entry(m, "pair_classifier", "v6").path == "pair_intent_classifier_v6.pkl"
    assert mi.select_entry(m, "pair_meta", "v1").path == "pair_intent_meta.json"
    assert mi.sole_entry(m, "meta_classifier").path == "meta_clf.pkl"
    assert mi.sole_entry(m, "encoder").path == "minilm-onnx/model.onnx"
    with pytest.raises(mi.NotListed):
        mi.select_entry(m, "pair_classifier", "v2")
    with pytest.raises(mi.NotListed):
        mi.sole_entry(m, "server")


def test_the_derived_manifest_verifies_files_in_a_package_directory(tmp_path):
    data = production()
    models = tmp_path / "fie" / "models"
    for a in data["artifacts"]:
        if a["path"].startswith("fie/models/"):
            target = tmp_path / a["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content(a["path"]))
    m = mi.parse_manifest(mi.derive_package_manifest(data))
    for e in m.entries:
        assert mi.read_verified(models, e) == content("fie/models/" + e.path)
    (models / "meta_clf.pkl").write_bytes(b"tampered")
    with pytest.raises(mi.IntegrityMismatch):
        mi.read_verified(models, mi.sole_entry(m, "meta_classifier"))


def test_an_invalid_inventory_derives_nothing():
    for bad in (changed(DEFAULT_CLF, default=False), changed(LEGACY[0], in_wheel=True), "{not json",
                changed(V5_CLF, version="v2")):
        with pytest.raises(mi.ManifestError):
            mi.derive_package_manifest(bad)


# ══ 3. the tracked inventory itself ══════════════════════════════════════════

@pytest.fixture(scope="module")
def tracked() -> dict:
    return json.loads(TRACKED.read_text(encoding="utf-8"))


def test_tracked_inventory_passes_the_production_rules(tracked):
    m = mi.validate_production(TRACKED.read_bytes())
    assert len(m.entries) == len(tracked["artifacts"]) == 25


def test_tracked_inventory_keeps_its_top_level_keys_and_gains_the_schema(tracked):
    assert list(tracked) == ["_comment", "schema", "release_tag", "repo", "artifacts"]
    assert tracked["schema"] == 1 and tracked["release_tag"] == "models-v1.18.0"
    assert tracked["repo"] == "AyushSingh110/Failure_Intelligence_System"


def test_tracked_meta_classifier_is_the_approved_file(tracked):
    """W21-7. The digest is the owner-approved one; no version is invented for it."""
    meta = find(tracked, "fie/models/meta_clf.pkl")
    assert meta["sha256"] == APPROVED_META_CLF_SHA256 and meta["size"] == 212131
    assert meta["role"] == "meta_classifier" and "version" not in meta
    assert meta["in_wheel"] is True and meta["default"] is False
    side = find(tracked, "fie/models/meta_clf.json")
    assert side["role"] == "meta_classifier_meta" and "version" not in side and side["in_wheel"] is True


def test_tracked_roles_are_as_decided(tracked):
    roles = {a["path"]: a["role"] for a in tracked["artifacts"]}
    assert roles["fie/models/minilm-onnx/model.onnx"] == "encoder"
    assert roles["fie/models/minilm-onnx/tokenizer.json"] == "tokenizer"
    assert {p for p, r in roles.items() if r == "legacy"} == set(LEGACY)
    assert {p for p, r in roles.items() if r == "server"} == set(SERVER)
    assert sorted(a["version"] for a in tracked["artifacts"] if a["role"] == "pair_classifier") == sorted(PAIR_KEYS)
    assert [a["path"] for a in tracked["artifacts"] if a["default"]] == [DEFAULT_CLF]
    assert all(isinstance(a["default"], bool) and isinstance(a["in_wheel"], bool) for a in tracked["artifacts"])


def test_tracked_wheel_contents_are_the_sixteen_sdk_files(tracked):
    shipped = sorted(a["path"] for a in tracked["artifacts"] if a["in_wheel"])
    assert len(shipped) == 16 and all(p.startswith("fie/models/") and "/minilm-onnx/" not in p for p in shipped)
    assert not any(p in shipped for p in LEGACY + SERVER)


def test_tracked_inventory_lists_none_of_the_experiment_files_or_encoder_by_products(tracked):
    paths = {a["path"] for a in tracked["artifacts"]}
    for name in ("v2", "v3_w3x", "v6_1_contaminated", "v6_noaug", "v64_harmaug", "v64_harmaug_capped",
                 "v65_benignaug"):
        assert f"fie/models/pair_intent_classifier_{name}.pkl" not in paths
        assert f"fie/models/pair_intent_meta_{name}.json" not in paths
    for name in ("config.json", "ort_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"):
        assert f"fie/models/minilm-onnx/{name}" not in paths


def test_tracked_selection_keys_agree_with_the_loader():
    """Read from the loader's source text: each approved key names the same file there."""
    source = (REPO / "fie" / "layers" / "pair.py").read_text(encoding="utf-8")
    in_loader = dict(re.findall(r'\("(v[0-9a-z_]+)",\s*_models_dir\s*/\s*"(pair_intent_classifier[0-9a-z_]*\.pkl)"', source))
    side_cars = dict(re.findall(r'_models_dir\s*/\s*"(pair_intent_classifier[0-9a-z_]*\.pkl)",\s*_models_dir\s*/\s*'
                                r'"(pair_intent_meta[0-9a-z_]*\.json)"', source))
    assert set(PAIR_KEYS) <= set(in_loader)
    for key in PAIR_KEYS:
        clf, meta = mi._PAIR_FILES[key]
        assert in_loader[key] == clf and side_cars[clf] == meta


def test_tracked_inventory_is_still_readable_by_the_harness():
    """evals reads path, sha256 and size from the same file; the new fields must not disturb it."""
    from evals import integrity

    loaded = integrity.load_manifest(REPO)
    assert loaded["release_tag"] == "models-v1.18.0" and len(loaded["artifacts"]) == 25
    assert loaded["artifacts"]["fie/models/meta_clf.pkl"] == {"sha256": APPROVED_META_CLF_SHA256, "size": 212131}


def test_package_manifest_derived_from_the_tracked_inventory(tracked):
    raw = mi.derive_package_manifest(TRACKED.read_bytes())
    assert raw == mi.derive_package_manifest(TRACKED.read_bytes()) == mi.derive_package_manifest(tracked)
    out = json.loads(raw)
    assert len(out["artifacts"]) == 20 and sum(a["in_wheel"] for a in out["artifacts"]) == 16
    meta = next(a for a in out["artifacts"] if a["path"] == "meta_clf.pkl")
    assert meta == {"path": "meta_clf.pkl", "sha256": APPROVED_META_CLF_SHA256, "size": 212131,
                    "role": "meta_classifier", "default": False, "in_wheel": True}
    m = mi.parse_manifest(raw)
    assert mi.default_entry(m, "pair_classifier").sha256 == find(tracked, DEFAULT_CLF)["sha256"]


def test_no_generated_manifest_is_tracked_in_git():
    import subprocess

    out = subprocess.run(["git", "ls-files", "fie/models/MANIFEST.json"], cwd=str(REPO), capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    assert out.stdout.strip() == ""


# ══ 4. the download script ═══════════════════════════════════════════════════

SCRIPT = REPO / "scripts" / "download_models.py"


class Downloader:
    """scripts/download_models.py pointed at a temporary tree, with the network replaced."""

    def __init__(self, tmp_path: Path, monkeypatch, data: dict | None = None):
        spec = importlib.util.spec_from_file_location(f"download_models_{id(self)}", SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.root, self.data = tmp_path, data if data is not None else production()
        (tmp_path / "scripts").mkdir()
        self.manifest_path = tmp_path / "scripts" / "model_manifest.json"
        self.manifest_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        self.remote = {a["path"].rsplit("/", 1)[-1]: content(a["path"]) for a in self.data["artifacts"]}
        self.requested: list[str] = []
        self.monkeypatch = monkeypatch
        monkeypatch.setattr(self.mod, "ROOT", str(tmp_path))
        monkeypatch.setattr(self.mod, "MANIFEST_PATH", str(self.manifest_path))
        monkeypatch.setattr(self.mod, "_download", self._download)
        monkeypatch.setattr(self.mod.urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("live network call")))
        monkeypatch.delenv("FIE_MODELS_RELEASE", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def _download(self, url: str, dest: str, token: str = "") -> None:
        name = url.rsplit("/", 1)[-1]
        self.requested.append(url)
        if name not in self.remote:
            raise self.mod.urllib.error.URLError(f"no such asset: {name}")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.remote[name])

    def run(self, *argv: str) -> int:
        self.monkeypatch.setattr(sys, "argv", ["download_models.py", *argv])
        return self.mod.main()

    def place_all(self) -> None:
        for a in self.data["artifacts"]:
            target = self.root / a["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content(a["path"]))

    @property
    def derived(self) -> Path:
        return self.root / "fie" / "models" / "MANIFEST.json"


@pytest.fixture
def dl(tmp_path, monkeypatch):
    return Downloader(tmp_path, monkeypatch)


def test_downloads_and_verifies_every_entry_whatever_in_wheel_says(dl, capsys):
    assert dl.run("--strict") == 0
    for a in dl.data["artifacts"]:
        assert (dl.root / a["path"]).read_bytes() == content(a["path"])
    assert len(dl.requested) == 25
    fetched = {u.rsplit("/", 1)[-1] for u in dl.requested}
    for path in SERVER + LEGACY + ["fie/models/minilm-onnx/model.onnx", "fie/models/minilm-onnx/tokenizer.json"]:
        assert path.rsplit("/", 1)[-1] in fetched                    # the in_wheel: false ones too
    assert "verified 25/25 artifacts" in capsys.readouterr().out


def test_release_selection_is_unchanged(dl):
    assert dl.run() == 0
    assert all(u.startswith("https://github.com/owner/repo/releases/download/models-test/") for u in dl.requested)
    dl.requested.clear()
    for a in dl.data["artifacts"]:
        (dl.root / a["path"]).unlink()
    assert dl.run("--tag", "models-other") == 0
    assert all("/releases/download/models-other/" in u for u in dl.requested)


def test_the_derived_manifest_is_written_after_a_successful_run(dl, capsys):
    assert not dl.derived.exists()
    assert dl.run("--strict") == 0
    assert dl.derived.read_bytes() == mi.derive_package_manifest(dl.manifest_path.read_bytes())
    assert "wrote fie/models/MANIFEST.json" in capsys.readouterr().out
    m = mi.parse_manifest(dl.derived.read_bytes())
    assert mi.read_verified(dl.root / "fie" / "models", mi.sole_entry(m, "meta_classifier")) == content("fie/models/meta_clf.pkl")


def test_a_second_run_with_everything_present_downloads_nothing_and_rewrites_the_same_bytes(dl):
    assert dl.run("--strict") == 0
    first = dl.derived.read_bytes()
    dl.requested.clear()
    assert dl.run("--strict") == 0
    assert dl.requested == [] and dl.derived.read_bytes() == first


def test_check_is_read_only_when_everything_is_present(dl):
    dl.place_all()
    before = sorted(p.relative_to(dl.root).as_posix() for p in dl.root.rglob("*") if p.is_file())
    assert dl.run("--check") == 0
    after = sorted(p.relative_to(dl.root).as_posix() for p in dl.root.rglob("*") if p.is_file())
    assert after == before and dl.requested == [] and not dl.derived.exists()


def test_check_does_not_overwrite_an_existing_derived_manifest(dl):
    dl.place_all()
    dl.derived.write_bytes(b"left by an earlier run")
    assert dl.run("--check") == 0
    assert dl.derived.read_bytes() == b"left by an earlier run" and dl.requested == []


def test_check_with_a_missing_artifact_exits_1_and_downloads_nothing(dl, capsys):
    dl.place_all()
    (dl.root / SERVER[0]).unlink()
    assert dl.run("--check") == 1
    out = capsys.readouterr().out
    assert f"[missing] {SERVER[0]}" in out and dl.requested == [] and not dl.derived.exists()
    assert not (dl.root / SERVER[0]).exists()


def test_check_with_a_wrong_digest_exits_1(dl, capsys):
    dl.place_all()
    (dl.root / META_CLF).write_bytes(b"not the approved file")
    assert dl.run("--check") == 1
    assert f"[stale]   {META_CLF}" in capsys.readouterr().out
    assert (dl.root / META_CLF).read_bytes() == b"not the approved file" and not dl.derived.exists()


def test_a_stale_local_file_is_replaced_by_the_verified_one(dl):
    dl.place_all()
    (dl.root / META_CLF).write_bytes(b"stale")
    assert dl.run("--strict") == 0
    assert (dl.root / META_CLF).read_bytes() == content(META_CLF) and len(dl.requested) == 1
    assert dl.derived.exists()


@pytest.mark.parametrize("path", [META_CLF, SERVER[0], LEGACY[0], "fie/models/minilm-onnx/model.onnx"])
def test_a_downloaded_file_with_the_wrong_digest_fails_and_writes_no_manifest(dl, capsys, path):
    dl.remote[path.rsplit("/", 1)[-1]] = b"a different file published under the same name"
    assert dl.run() == 1
    assert dl.run("--strict") == 1
    assert "does not match manifest" in capsys.readouterr().out
    assert not (dl.root / path).exists()                              # the corrupt download is deleted, as before
    assert not dl.derived.exists()


@pytest.mark.parametrize("path", [META_CLF, SERVER[4], LEGACY[1], "fie/models/minilm-onnx/tokenizer.json"])
def test_an_unreachable_artifact_strict_exits_1_and_writes_no_manifest(dl, path):
    del dl.remote[path.rsplit("/", 1)[-1]]
    assert dl.run("--strict") == 1
    assert not dl.derived.exists() and not (dl.root / path).exists()


def test_an_unreachable_artifact_non_strict_keeps_exit_0_and_writes_no_manifest(dl, capsys):
    """The existing lenient behaviour is kept; but a partly verified inventory gets no manifest."""
    del dl.remote["meta_clf.pkl"]
    assert dl.run() == 0
    out = capsys.readouterr().out
    assert "some artifacts unavailable" in out and "verified 24/25 artifacts" in out
    assert not dl.derived.exists()


@pytest.mark.parametrize("argv, expected", [(("--strict",), 1), ((), 0)])
def test_a_failed_run_does_not_replace_an_existing_derived_manifest(dl, argv, expected):
    assert dl.run("--strict") == 0
    good = dl.derived.read_bytes()
    (dl.root / SERVER[1]).unlink()
    del dl.remote[SERVER[1].rsplit("/", 1)[-1]]
    assert dl.run(*argv) == expected
    assert dl.derived.read_bytes() == good


def test_a_corrupt_download_does_not_replace_an_existing_derived_manifest(dl):
    assert dl.run("--strict") == 0
    good = dl.derived.read_bytes()
    (dl.root / META_CLF).unlink()
    dl.remote["meta_clf.pkl"] = b"corrupt"
    assert dl.run("--strict") == 1
    assert dl.derived.read_bytes() == good


def test_the_manifest_is_not_written_before_the_last_artifact_is_verified(dl):
    seen = []
    real = dl._download

    def watching(url, dest, token=""):
        seen.append(dl.derived.exists())
        real(url, dest, token)

    dl.monkeypatch.setattr(dl.mod, "_download", watching)
    assert dl.run("--strict") == 0
    assert seen == [False] * 25 and dl.derived.exists()


def test_server_entries_are_still_downloaded_and_strictly_verified(dl):
    assert dl.run("--strict") == 0
    for path in SERVER:
        assert (dl.root / path).read_bytes() == content(path)
    for path in SERVER:
        (dl.root / path).write_bytes(b"tampered on the server")
        assert dl.run("--check") == 1
        (dl.root / path).write_bytes(content(path))
    assert dl.run("--check") == 0


def test_an_inventory_that_breaks_the_rules_fails_the_run_and_writes_nothing(tmp_path, monkeypatch, capsys):
    bad = changed(LEGACY[0], in_wheel=True)
    dl = Downloader(tmp_path, monkeypatch, bad)
    assert dl.run("--strict") == 1
    out = capsys.readouterr().out
    assert "verified 25/25 artifacts" in out and "could not derive fie/models/MANIFEST.json" in out
    assert not dl.derived.exists()


def test_no_temporary_file_is_left_beside_the_manifest(dl):
    assert dl.run("--strict") == 0
    names = sorted(p.name for p in (dl.root / "fie" / "models").iterdir())
    assert "MANIFEST.json" in names and not [n for n in names if n.startswith("tmp")]


def test_the_script_does_not_import_the_fie_package():
    """It runs before the SDK's dependencies are installed, so it loads the one module by path."""
    code = (f"import sys, importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('dl', r'{SCRIPT}')\n"
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            f"m._integrity()\n"
            f"print(sorted(k for k in sys.modules if k == 'fie' or k.startswith('fie.')))")
    import subprocess

    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert proc.stdout.strip().splitlines()[-1] == "[]"
