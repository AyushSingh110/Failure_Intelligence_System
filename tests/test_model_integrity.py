"""
tests/test_model_integrity.py — the model-integrity primitives (WP-021 phase 3, stage 3.1).

fie/_model_integrity.py checks a model artifact against a manifest and hands back the bytes
it checked. These tests cover that module on its own. It is not yet used by the model loader.

Every fixture here is synthetic: small byte strings written to a temporary directory, and
manifests built in the test. No real model file is read, and nothing is deserialized.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fie import _model_integrity as mi

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "fie" / "_model_integrity.py"

CLF_V1 = b"synthetic classifier bytes, version one\n" * 4
CLF_V2 = b"synthetic classifier bytes, version two\n" * 5
META = b"synthetic meta-classifier bytes\x00\x01\x02"
ENCODER = b"synthetic encoder bytes"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact(path: str, data: bytes, role: str, version=None, in_wheel=True, default=None) -> dict:
    out = {"path": path, "sha256": sha(data), "size": len(data), "role": role, "in_wheel": in_wheel}
    if version is not None:
        out["version"] = version
    if default is not None:
        out["default"] = default
    return out


def manifest_dict(*artifacts) -> dict:
    return {"schema": 1, "artifacts": list(artifacts)}


def standard() -> dict:
    return manifest_dict(
        artifact("clf_v1.pkl", CLF_V1, "pair_classifier", "v1"),
        artifact("clf_v2.pkl", CLF_V2, "pair_classifier", "v2", default=True),
        artifact("meta.pkl", META, "meta_classifier", "m1", default=True),
        artifact("enc/model.onnx", ENCODER, "encoder", in_wheel=False),
    )


@pytest.fixture
def models(tmp_path):
    root = tmp_path / "models"
    (root / "enc").mkdir(parents=True)
    (root / "clf_v1.pkl").write_bytes(CLF_V1)
    (root / "clf_v2.pkl").write_bytes(CLF_V2)
    (root / "meta.pkl").write_bytes(META)
    (root / "enc" / "model.onnx").write_bytes(ENCODER)
    return root


def with_entry(**changes) -> dict:
    """The standard manifest with its first entry altered (a key set to ... is removed)."""
    data = standard()
    for key, value in changes.items():
        if value is ...:
            data["artifacts"][0].pop(key, None)
        else:
            data["artifacts"][0][key] = value
    return data


# ── parsing a valid manifest ────────────────────────────────────────────────

def test_valid_manifest_parses_to_typed_entries():
    m = mi.parse_manifest(standard())
    assert m.schema == 1 and len(m.entries) == 4
    first = m.entries[0]
    assert first == mi.Entry(path="clf_v1.pkl", sha256=sha(CLF_V1), size=len(CLF_V1), role="pair_classifier",
                             in_wheel=True, version="v1", default=False)
    assert m.entries[1].default is True
    assert m.entries[3] == mi.Entry(path="enc/model.onnx", sha256=sha(ENCODER), size=len(ENCODER), role="encoder",
                                    in_wheel=False, version=None, default=False)


@pytest.mark.parametrize("encode", [json.dumps, lambda d: json.dumps(d).encode("utf-8"), lambda d: d])
def test_text_bytes_and_decoded_object_parse_the_same(encode):
    assert mi.parse_manifest(encode(standard())) == mi.parse_manifest(standard())


def test_manifest_lookups():
    m = mi.parse_manifest(standard())
    assert m.by_path("meta.pkl").role == "meta_classifier"
    assert m.by_path("absent.pkl") is None
    assert [e.version for e in m.for_role("pair_classifier")] == ["v1", "v2"]
    assert m.for_role("nothing") == ()


def test_entries_are_immutable():
    entry = mi.parse_manifest(standard()).entries[0]
    with pytest.raises(AttributeError):
        entry.sha256 = "0" * 64


def test_extra_keys_are_tolerated_and_ignored():
    data = with_entry(used_by="a free-text note")
    assert mi.parse_manifest(data).entries[0] == mi.parse_manifest(standard()).entries[0]


# ── an invalid manifest is rejected whole ───────────────────────────────────

@pytest.mark.parametrize("source, fragment", [
    ("{not json", "not valid JSON"),
    ("", "not valid JSON"),
    (b"\xff\xfe\x00", "not UTF-8"),
    ("[]", "must be a JSON object"),
    ('"a string"', "must be a JSON object"),
    ({"artifacts": []}, "schema must be 1"),
    ({"schema": 2, "artifacts": []}, "schema must be 1"),
    ({"schema": True, "artifacts": []}, "schema must be 1"),
    ({"schema": "1", "artifacts": []}, "schema must be 1"),
    ({"schema": 1}, "'artifacts' must be a list"),
    ({"schema": 1, "artifacts": {}}, "'artifacts' must be a list"),
    ({"schema": 1, "artifacts": ["clf.pkl"]}, "artifacts[0]: must be an object"),
])
def test_malformed_manifest_is_a_manifest_error(source, fragment):
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(source)
    assert fragment in str(caught.value)
    assert caught.value.code == "manifest_invalid"


@pytest.mark.parametrize("field", ["path", "sha256", "size", "role", "in_wheel"])
def test_each_required_field_is_required(field):
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(with_entry(**{field: ...}))
    assert "missing required field(s)" in str(caught.value) and field in str(caught.value)


@pytest.mark.parametrize("value", [
    "", "abc", "g" * 64, "A" * 64, sha(CLF_V1).upper(), sha(CLF_V1)[:63], sha(CLF_V1) + "0",
    " " + sha(CLF_V1)[1:], "sha256:" + sha(CLF_V1), None, 123, ["x"],
])
def test_invalid_sha256_format_is_rejected(value):
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(with_entry(sha256=value))
    assert "64 lowercase hexadecimal" in str(caught.value)
    assert caught.value.path == "clf_v1.pkl"


@pytest.mark.parametrize("value", [-1, 1.5, "160", None, True, False, [160]])
def test_invalid_size_is_rejected(value):
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(with_entry(size=value))
    assert "'size' must be a non-negative integer" in str(caught.value)


@pytest.mark.parametrize("field, value, fragment", [
    ("role", "", "'role' must be a short token"),
    ("role", "two words", "'role' must be a short token"),
    ("role", None, "'role' must be a short token"),
    ("role", "x" * 65, "'role' must be a short token"),
    ("in_wheel", "true", "'in_wheel' must be true or false"),
    ("in_wheel", 1, "'in_wheel' must be true or false"),
    ("in_wheel", None, "'in_wheel' must be true or false"),
    ("version", "", "'version' must be a short token or absent"),
    ("version", 6, "'version' must be a short token or absent"),
    ("version", "v 6", "'version' must be a short token or absent"),
    ("default", "yes", "'default' must be true or false"),
    ("default", 1, "'default' must be true or false"),
])
def test_invalid_metadata_is_rejected(field, value, fragment):
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(with_entry(**{field: value}))
    assert fragment in str(caught.value)


def test_one_bad_entry_invalidates_the_whole_manifest():
    data = standard()
    data["artifacts"][3]["sha256"] = "not-a-digest"
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(data)
    assert "artifacts[3]" in str(caught.value)


# ── paths ───────────────────────────────────────────────────────────────────

UNSAFE_PATHS = [
    "../clf_v1.pkl", "..", "sub/../../clf.pkl", "a/./b.pkl", "/etc/passwd", "/clf.pkl", "clf.pkl/",
    "a//b.pkl", "sub\\clf.pkl", "..\\clf.pkl", "C:/models/clf.pkl", "C:\\models\\clf.pkl", "clf.pkl:stream",
    "clf\x00.pkl", "has space.pkl", "", ".", "~/clf.pkl", "$HOME/clf.pkl", "clf*.pkl",
]


@pytest.mark.parametrize("path", UNSAFE_PATHS)
def test_unsafe_path_in_a_manifest_is_rejected(path):
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(with_entry(path=path))
    assert "path" in str(caught.value)


@pytest.mark.parametrize("path", [None, 5, ["clf.pkl"]])
def test_path_must_be_a_string(path):
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(with_entry(path=path))
    assert "'path' must be a non-empty string" in str(caught.value)


@pytest.mark.parametrize("second", ["clf_v1.pkl", "CLF_V1.PKL", "Clf_v1.pkl"])
def test_duplicate_paths_are_rejected_also_when_only_case_differs(second):
    data = manifest_dict(artifact("clf_v1.pkl", CLF_V1, "pair_classifier", "v1"),
                         artifact(second, CLF_V2, "pair_classifier", "v2"))
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(data)
    assert "duplicate path" in str(caught.value)


@pytest.mark.parametrize("path", UNSAFE_PATHS)
def test_a_hand_built_entry_with_an_unsafe_path_cannot_be_resolved(models, path):
    entry = mi.Entry(path=path, sha256=sha(CLF_V1), size=len(CLF_V1), role="pair_classifier", in_wheel=True)
    with pytest.raises(mi.PathViolation) as caught:
        mi.resolve_path(models, entry)
    assert caught.value.code == "path_violation"
    with pytest.raises(mi.PathViolation):
        mi.read_verified(models, entry)


def test_a_file_outside_the_model_directory_is_never_read(models, tmp_path):
    outside = tmp_path / "secret.pkl"
    outside.write_bytes(CLF_V1)                           # correct bytes, wrong place
    entry = mi.Entry(path="../secret.pkl", sha256=sha(CLF_V1), size=len(CLF_V1), role="pair_classifier", in_wheel=True)
    with pytest.raises(mi.PathViolation):
        mi.read_verified(models, entry)


def test_a_symbolic_link_leaving_the_model_directory_is_refused(models, tmp_path):
    outside = tmp_path / "elsewhere.pkl"
    outside.write_bytes(CLF_V1)
    link = models / "linked.pkl"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links cannot be created here")
    entry = mi.Entry(path="linked.pkl", sha256=sha(CLF_V1), size=len(CLF_V1), role="pair_classifier", in_wheel=True)
    with pytest.raises(mi.PathViolation) as caught:
        mi.read_verified(models, entry)
    assert "outside the model directory" in str(caught.value)


def test_a_directory_junction_leaving_the_model_directory_is_refused(models, tmp_path):
    """The Windows counterpart of the symbolic-link case; a junction needs no special privilege."""
    if os.name != "nt":
        pytest.skip("directory junctions are a Windows feature; the symbolic-link test covers other systems")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.pkl").write_bytes(CLF_V1)
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(models / "junction"), str(outside)],
                          capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip("a directory junction cannot be created here")
    assert (models / "junction" / "secret.pkl").read_bytes() == CLF_V1         # reachable through the junction
    entry = mi.Entry(path="junction/secret.pkl", sha256=sha(CLF_V1), size=len(CLF_V1), role="pair_classifier",
                     in_wheel=True)
    with pytest.raises(mi.PathViolation) as caught:
        mi.read_verified(models, entry)
    assert "outside the model directory" in str(caught.value)


def test_a_directory_is_not_an_artifact(models):
    entry = mi.Entry(path="enc", sha256=sha(b""), size=0, role="encoder", in_wheel=False)
    with pytest.raises(mi.PathViolation) as caught:
        mi.read_verified(models, entry)
    assert "not a regular file" in str(caught.value)


def test_safe_nested_path_resolves_inside_the_directory(models):
    entry = mi.parse_manifest(standard()).by_path("enc/model.onnx")
    resolved = mi.resolve_path(models, entry)
    assert resolved == Path(os.path.realpath(models / "enc" / "model.onnx"))
    assert Path(os.path.realpath(models)) in resolved.parents


# ── selection ───────────────────────────────────────────────────────────────

def test_default_is_the_explicitly_marked_entry_not_the_first_or_newest():
    m = mi.parse_manifest(standard())
    assert mi.default_entry(m, "pair_classifier").path == "clf_v2.pkl"
    reordered = standard()
    reordered["artifacts"][0]["default"], reordered["artifacts"][1]["default"] = True, False
    assert mi.default_entry(mi.parse_manifest(reordered), "pair_classifier").path == "clf_v1.pkl"


def test_each_role_has_its_own_default():
    m = mi.parse_manifest(standard())
    assert mi.default_entry(m, "meta_classifier").path == "meta.pkl"


def test_exact_version_selection():
    m = mi.parse_manifest(standard())
    assert mi.select_entry(m, "pair_classifier", "v1").path == "clf_v1.pkl"
    assert mi.select_entry(m, "pair_classifier", "v2").path == "clf_v2.pkl"
    assert mi.select_entry(m, "meta_classifier", "m1").path == "meta.pkl"


@pytest.mark.parametrize("role, version", [
    ("pair_classifier", "v3"), ("pair_classifier", "V1"), ("pair_classifier", "v1 "), ("pair_classifier", ""),
    ("pair_classifier", "m1"), ("meta_classifier", "v1"), ("unknown_role", "v1"),
])
def test_an_absent_version_is_an_error_and_nothing_is_substituted(role, version):
    m = mi.parse_manifest(standard())
    with pytest.raises(mi.NotListed) as caught:
        mi.select_entry(m, role, version)
    assert caught.value.code == "not_listed"
    assert (caught.value.role, caught.value.version) == (role, version)


def test_select_entry_never_returns_the_default_for_a_missing_version():
    """The old loader fell back to the default. This helper must not."""
    m = mi.parse_manifest(standard())
    default = mi.default_entry(m, "pair_classifier")
    try:
        result = mi.select_entry(m, "pair_classifier", "v99")
    except mi.NotListed:
        result = None
    assert result is None and default.path == "clf_v2.pkl"


def test_a_role_without_a_marked_default_has_no_default():
    data = manifest_dict(artifact("clf_v1.pkl", CLF_V1, "pair_classifier", "v1"),
                         artifact("clf_v2.pkl", CLF_V2, "pair_classifier", "v2"))
    m = mi.parse_manifest(data)
    with pytest.raises(mi.NotListed) as caught:
        mi.default_entry(m, "pair_classifier")
    assert "marks no default" in str(caught.value) and caught.value.version is None
    with pytest.raises(mi.NotListed):
        mi.default_entry(m, "encoder")


def test_two_defaults_for_one_role_are_rejected():
    data = manifest_dict(artifact("clf_v1.pkl", CLF_V1, "pair_classifier", "v1", default=True),
                         artifact("clf_v2.pkl", CLF_V2, "pair_classifier", "v2", default=True))
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(data)
    assert "more than one default" in str(caught.value)
    assert "clf_v1.pkl" in str(caught.value) and "clf_v2.pkl" in str(caught.value)


def test_a_default_without_a_version_is_rejected():
    data = manifest_dict(artifact("clf.pkl", CLF_V1, "pair_classifier", default=True))
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(data)
    assert "a default entry must name its version" in str(caught.value)


def test_the_same_version_twice_for_one_role_is_rejected():
    data = manifest_dict(artifact("a.pkl", CLF_V1, "pair_classifier", "v1"),
                         artifact("b.pkl", CLF_V2, "pair_classifier", "v1"))
    with pytest.raises(mi.ManifestError) as caught:
        mi.parse_manifest(data)
    assert "lists version 'v1' more than once" in str(caught.value)


def test_the_same_version_under_two_roles_is_allowed():
    data = manifest_dict(artifact("a.pkl", CLF_V1, "pair_classifier", "v1"),
                         artifact("b.pkl", META, "meta_classifier", "v1"))
    m = mi.parse_manifest(data)
    assert mi.select_entry(m, "meta_classifier", "v1").path == "b.pkl"


# ── reading and verifying ───────────────────────────────────────────────────

def test_read_verified_returns_the_exact_bytes(models):
    m = mi.parse_manifest(standard())
    for path, expected in (("clf_v1.pkl", CLF_V1), ("clf_v2.pkl", CLF_V2), ("meta.pkl", META),
                           ("enc/model.onnx", ENCODER)):
        data = mi.read_verified(models, m.by_path(path))
        assert data == expected and isinstance(data, bytes)
        assert sha(data) == m.by_path(path).sha256 and len(data) == m.by_path(path).size


def test_the_file_is_opened_exactly_once(models):
    """The bytes returned are the bytes hashed: there is no second read of the path."""
    entry = mi.parse_manifest(standard()).by_path("clf_v1.pkl")
    target = os.path.realpath(models / "clf_v1.pkl")
    opened = []

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) == target:
                opened.append(args[0])

    sys.addaudithook(hook)
    data = mi.read_verified(models, entry)
    count = len(opened)
    target = "<disarmed>"                                  # audit hooks cannot be removed
    assert data == CLF_V1 and count == 1


def test_replacing_the_file_after_the_read_does_not_change_what_was_returned(models):
    entry = mi.parse_manifest(standard()).by_path("clf_v1.pkl")
    data = mi.read_verified(models, entry)
    (models / "clf_v1.pkl").write_bytes(b"swapped after verification")
    assert data == CLF_V1 and sha(data) == entry.sha256
    with pytest.raises(mi.IntegrityMismatch):
        mi.read_verified(models, entry)


def test_changed_content_of_the_same_size_is_rejected_and_nothing_is_returned(models):
    entry = mi.parse_manifest(standard()).by_path("clf_v1.pkl")
    tampered = bytes([CLF_V1[0] ^ 0x01]) + CLF_V1[1:]
    assert len(tampered) == len(CLF_V1)
    (models / "clf_v1.pkl").write_bytes(tampered)
    result = "not assigned"
    with pytest.raises(mi.IntegrityMismatch) as caught:
        result = mi.read_verified(models, entry)
    assert result == "not assigned"
    err = caught.value
    assert (err.code, err.what, err.path) == ("mismatch", "sha256", "clf_v1.pkl")
    assert err.expected == sha(CLF_V1) and err.actual == sha(tampered)


@pytest.mark.parametrize("replacement", [CLF_V1[:-1], CLF_V1 + b"x", b"", CLF_V1 * 50])
def test_wrong_size_is_rejected(models, replacement):
    entry = mi.parse_manifest(standard()).by_path("clf_v1.pkl")
    (models / "clf_v1.pkl").write_bytes(replacement)
    with pytest.raises(mi.IntegrityMismatch) as caught:
        mi.read_verified(models, entry)
    err = caught.value
    assert (err.what, err.expected, err.actual) == ("size", len(CLF_V1), len(replacement))


def test_a_larger_file_is_not_read_into_memory(models, monkeypatch):
    entry = mi.parse_manifest(standard()).by_path("clf_v1.pkl")
    (models / "clf_v1.pkl").write_bytes(CLF_V1 * 1000)
    seen = []
    real = mi.verify_bytes
    monkeypatch.setattr(mi, "verify_bytes", lambda e, d: (seen.append(len(d)), real(e, d)))
    with pytest.raises(mi.IntegrityMismatch) as caught:
        mi.read_verified(models, entry)
    assert seen == [] and caught.value.actual == len(CLF_V1) * 1000


def test_a_manifest_with_the_wrong_digest_rejects_the_untouched_file(models):
    data = standard()
    data["artifacts"][0]["sha256"] = sha(b"some other file")
    entry = mi.parse_manifest(data).by_path("clf_v1.pkl")
    with pytest.raises(mi.IntegrityMismatch) as caught:
        mi.read_verified(models, entry)
    assert caught.value.what == "sha256" and caught.value.actual == sha(CLF_V1)


def test_a_manifest_with_the_wrong_size_rejects_the_untouched_file(models):
    data = standard()
    data["artifacts"][0]["size"] = len(CLF_V1) + 7
    entry = mi.parse_manifest(data).by_path("clf_v1.pkl")
    with pytest.raises(mi.IntegrityMismatch) as caught:
        mi.read_verified(models, entry)
    assert (caught.value.what, caught.value.actual) == ("size", len(CLF_V1))


def test_a_listed_file_that_is_absent_is_missing_not_mismatched(models):
    entry = mi.parse_manifest(standard()).by_path("clf_v1.pkl")
    (models / "clf_v1.pkl").unlink()
    with pytest.raises(mi.ArtifactMissing) as caught:
        mi.read_verified(models, entry)
    assert (caught.value.code, caught.value.path) == ("missing", "clf_v1.pkl")
    assert not isinstance(caught.value, mi.IntegrityMismatch)


def test_a_missing_model_directory_is_missing(tmp_path):
    entry = mi.parse_manifest(standard()).by_path("clf_v1.pkl")
    with pytest.raises(mi.ArtifactMissing):
        mi.read_verified(tmp_path / "no-such-directory", entry)


def test_an_unlisted_file_beside_the_listed_ones_cannot_be_reached(models):
    (models / "stray.pkl").write_bytes(b"an unlisted file")
    m = mi.parse_manifest(standard())
    assert m.by_path("stray.pkl") is None
    with pytest.raises(mi.NotListed):
        mi.select_entry(m, "pair_classifier", "stray")
    assert all(e.path != "stray.pkl" for e in m.entries)


def test_verify_bytes_checks_size_then_digest():
    entry = mi.parse_manifest(standard()).by_path("meta.pkl")
    assert mi.verify_bytes(entry, META) is None
    with pytest.raises(mi.IntegrityMismatch) as caught:
        mi.verify_bytes(entry, META + b"!")
    assert caught.value.what == "size"
    with pytest.raises(mi.IntegrityMismatch) as caught:
        mi.verify_bytes(entry, META[::-1])
    assert caught.value.what == "sha256"


# ── the four failure kinds are distinct ─────────────────────────────────────

def test_error_classes_and_codes_are_distinct():
    classes = [mi.ManifestError, mi.PathViolation, mi.NotListed, mi.ArtifactMissing, mi.IntegrityMismatch]
    assert len({c.code for c in classes}) == 5
    assert {c.code for c in classes} == {"manifest_invalid", "path_violation", "not_listed", "missing", "mismatch"}
    for cls in classes:
        assert issubclass(cls, mi.ModelIntegrityError)
        assert sum(issubclass(cls, other) for other in classes) == 1          # none is a kind of another


def test_error_messages_do_not_contain_the_absolute_location(models):
    entry = mi.parse_manifest(standard()).by_path("clf_v1.pkl")
    (models / "clf_v1.pkl").write_bytes(b"x")
    with pytest.raises(mi.IntegrityMismatch) as caught:
        mi.read_verified(models, entry)
    assert str(models) not in str(caught.value) and os.path.realpath(models) not in str(caught.value)
    (models / "clf_v1.pkl").unlink()
    with pytest.raises(mi.ArtifactMissing) as caught:
        mi.read_verified(models, entry)
    assert str(models) not in str(caught.value)


# ── what the module is, and is not, in this stage ───────────────────────────

def test_module_imports_only_the_standard_library_and_no_deserializer():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}
    assert not imported & {"pickle", "joblib", "marshal", "shelve", "dill", "cloudpickle", "numpy", "sklearn"}


def test_module_contains_no_deserialization_call():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
             for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not calls & {"load", "Unpickler", "eval", "exec", "__import__"}
    bare = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "compile" not in bare                            # re.compile is used; the builtin is not
    assert "loads" in calls                                 # json.loads, for the manifest, and nothing else
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "loads"]
    assert all(isinstance(n.func.value, ast.Name) and n.func.value.id == "json" for n in loads)


def test_importing_the_module_loads_no_model_library():
    code = ("import sys, json\nfrom fie import _model_integrity\n"
            "print(json.dumps(sorted(m for m in ('joblib','pickle','sklearn','numpy','xgboost','onnxruntime') "
            "if m in sys.modules)))")
    env = {**os.environ, "FIE_NO_TELEMETRY": "1", "FIE_NO_AUTO_DOWNLOAD": "1"}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


def test_the_loader_does_not_use_the_module_yet():
    """Stage 3.1: the primitives exist and nothing in the package calls them."""
    users = [p.relative_to(REPO).as_posix() for p in (REPO / "fie").rglob("*.py")
             if p != MODULE and "_model_integrity" in p.read_text(encoding="utf-8", errors="replace")]
    assert users == []
    code = "import sys, fie\nprint('fie._model_integrity' in sys.modules)"
    env = {**os.environ, "FIE_NO_TELEMETRY": "1", "FIE_NO_AUTO_DOWNLOAD": "1"}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)
    assert proc.returncode == 0 and proc.stdout.strip().splitlines()[-1] == "False"
