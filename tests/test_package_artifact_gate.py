"""
The package archive gate, scripts/verify_package_artifacts.py (WP-021 Stage 3.3c).

The gate judges built archives by their actual members: under fie/models/ exactly the
`in_wheel: true` artifacts of the inventory, byte for byte, plus fie/models/MANIFEST.json equal
to the derivation of that inventory; and, for an sdist, the inputs a wheel build needs.

Every archive here is synthetic: written by the test into a temporary directory, with synthetic
member bytes and a synthetic inventory built from the project's own rules. Nothing is
deserialized, and the real model files are never touched.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_build_artifact_allowlist import synthetic_inventory  # noqa: E402  - the same inventory rules

CHECKER_PATH = ROOT / "scripts" / "verify_package_artifacts.py"
_spec = importlib.util.spec_from_file_location("_fie_verify_package_artifacts_under_test", CHECKER_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
mi = gate.load_integrity(ROOT / "fie" / "_model_integrity.py")


# ── synthetic distributions ──────────────────────────────────────────────────

@pytest.fixture
def inv(tmp_path):
    """A synthetic production inventory on disk, and its entries with their bytes."""
    entries = synthetic_inventory()
    body = {"schema": 1, "release_tag": "synthetic", "repo": "synthetic/synthetic",
            "artifacts": [{k: v for k, v in e.items() if k != "_bytes"} for e in entries]}
    path = tmp_path / "model_manifest.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {"path": path, "entries": entries, "bytes": path.read_bytes(),
            "derived": mi.derive_package_manifest(path.read_bytes())}


def good_members(inv) -> dict[str, bytes]:
    members = {e["path"]: e["_bytes"] for e in inv["entries"] if e["in_wheel"]}
    members["fie/models/MANIFEST.json"] = inv["derived"]
    members["fie/__init__.py"] = b'"""synthetic"""\n'
    members["fie/adversarial.py"] = b"# unrelated package source\n"
    return members


def sdist_inputs(inv) -> dict[str, bytes]:
    return {"pyproject.toml": b"[build-system]\n", "hatch_build.py": b"# hook\n",
            "fie/_model_integrity.py": b"# integrity\n", "scripts/model_manifest.json": inv["bytes"],
            "README.md": b"unrelated\n", "docs/media/big.bin": b"\0" * 64}


def make_wheel(path: Path, members: dict[str, bytes], duplicates: dict[str, bytes] | None = None,
               symlinks: tuple = ()) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
        z.writestr("fie_sdk-0.0.0.dist-info/METADATA", "Name: fie-sdk\n")
        for name in symlinks:
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o120777 << 16)
            z.writestr(info, "../../etc/passwd")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                     # zipfile warns on duplicate names
            for name, data in (duplicates or {}).items():
                z.writestr(name, data)
    return path


def make_sdist(path: Path, members: dict[str, bytes], *, top="fie_sdk-0.0.0", extra_raw=(), symlinks=()) -> Path:
    with tarfile.open(path, "w:gz") as t:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))

        for name, data in members.items():
            add(f"{top}/{name}", data)
        for name, data in extra_raw:                            # names written exactly as given
            add(name, data)
        for name in symlinks:
            info = tarfile.TarInfo(f"{top}/{name}")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../../etc/passwd"
            t.addfile(info)
    return path


def check(path: Path, kind: str, inv) -> dict:
    return gate.verify(path, kind, gate.expectations(inv["bytes"], mi), mi)


def failures(result) -> str:
    return "\n".join(result["failures"])


# ══ 1. What is expected comes from the inventory ═════════════════════════════

def test_expectations_are_the_in_wheel_entries_and_the_derived_manifest(inv):
    exp = gate.expectations(inv["bytes"], mi)
    want = {e["path"]: {"size": e["size"], "sha256": e["sha256"]} for e in inv["entries"] if e["in_wheel"]}
    assert exp["artifacts"] == want and len(want) == 16
    assert exp["manifest_bytes"] == inv["derived"]
    assert exp["manifest"]["sha256"] == hashlib.sha256(inv["derived"]).hexdigest()


def test_changing_the_inventory_changes_what_is_expected(inv, tmp_path):
    """No second list: a new approved digest is expected as soon as the inventory says so."""
    wheel = make_wheel(tmp_path / "w.whl", good_members(inv))
    assert check(wheel, "wheel", inv)["passed"]
    body = json.loads(inv["bytes"])
    target = next(a for a in body["artifacts"] if a["path"].endswith("pair_intent_classifier_v6_3b.pkl"))
    new_bytes = b"a newly approved classifier, synthetic\n"
    target["sha256"], target["size"] = hashlib.sha256(new_bytes).hexdigest(), len(new_bytes)
    new_inv = {"bytes": json.dumps(body).encode()}
    new_inv["derived"] = mi.derive_package_manifest(new_inv["bytes"])
    assert not check(wheel, "wheel", new_inv)["passed"], "the old bytes no longer match the inventory"
    members = good_members(inv) | {target["path"]: new_bytes, "fie/models/MANIFEST.json": new_inv["derived"]}
    assert check(make_wheel(tmp_path / "w2.whl", members), "wheel", new_inv)["passed"]


def test_real_inventory_expects_sixteen_artifacts_and_the_established_manifest():
    exp = gate.expectations((ROOT / "scripts" / "model_manifest.json").read_bytes(), mi)
    assert len(exp["artifacts"]) == 16 and all(p.startswith("fie/models/") for p in exp["artifacts"])
    assert (exp["manifest"]["sha256"], exp["manifest"]["size"]) == (
        "fbab28ab75acad2cfa8d178617caad112c32adafc6048d2a8483cd4a393bf2f5", 5112)


# ══ 2. Valid archives pass ═══════════════════════════════════════════════════

def test_valid_wheel_and_sdist_pass(inv, tmp_path):
    wheel = check(make_wheel(tmp_path / "w.whl", good_members(inv)), "wheel", inv)
    sdist = check(make_sdist(tmp_path / "s.tar.gz", good_members(inv) | sdist_inputs(inv)), "sdist", inv)
    assert wheel["passed"] and sdist["passed"], failures(wheel) + failures(sdist)
    assert set(wheel["fie_models"]) == set(sdist["fie_models"]) == \
        {e["path"] for e in inv["entries"] if e["in_wheel"]} | {"fie/models/MANIFEST.json"}


def test_unrelated_files_outside_the_model_area_are_not_judged(inv, tmp_path):
    members = good_members(inv) | sdist_inputs(inv) | {"brag-output/clip.mp4": b"\0" * 128,
                                                        "models/server_classifier.pkl": b"server",
                                                        "tests/fixture.pkl": b"fixture"}
    assert check(make_sdist(tmp_path / "s.tar.gz", members), "sdist", inv)["passed"]


# ══ 3. Negative controls ═════════════════════════════════════════════════════

@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("case, mutate, expected", [
    ("missing_artifact", lambda m: m.pop("fie/models/meta_clf.pkl"), "missing required member: fie/models/meta_clf.pkl"),
    ("missing_manifest", lambda m: m.pop("fie/models/MANIFEST.json"), "missing required member: fie/models/MANIFEST.json"),
    ("legacy", lambda m: m.update({"fie/models/failure_classifier_v4.pkl": b"legacy"}),
     "unexpected member under fie/models/: fie/models/failure_classifier_v4.pkl"),
    ("experiment_classifier", lambda m: m.update({"fie/models/pair_intent_classifier_v64_harmaug.pkl": b"x"}),
     "unexpected member under fie/models/: fie/models/pair_intent_classifier_v64_harmaug.pkl"),
    ("experiment_side_car", lambda m: m.update({"fie/models/pair_intent_meta_v2.json": b"{}"}),
     "unexpected member under fie/models/: fie/models/pair_intent_meta_v2.json"),
    ("nested", lambda m: m.update({"fie/models/sub/extra.json": b"{}"}),
     "unexpected member under fie/models/: fie/models/sub/extra.json"),
    ("encoder_by_product", lambda m: m.update({"fie/models/minilm-onnx/vocab.txt": b"a"}),
     "unexpected member under fie/models/: fie/models/minilm-onnx/vocab.txt"),
    ("gitkeep", lambda m: m.update({"fie/models/.gitkeep": b""}), "unexpected member under fie/models/: fie/models/.gitkeep"),
    ("wrong_bytes", lambda m: m.update({"fie/models/pair_intent_classifier_v5.pkl":
                                        bytes([m["fie/models/pair_intent_classifier_v5.pkl"][0] ^ 1])
                                        + m["fie/models/pair_intent_classifier_v5.pkl"][1:]}),
     "fie/models/pair_intent_classifier_v5.pkl: SHA-256 does not match the inventory"),
    ("wrong_size", lambda m: m.update({"fie/models/pair_intent_meta_v5.json": m["fie/models/pair_intent_meta_v5.json"] + b" "}),
     "fie/models/pair_intent_meta_v5.json: size"),
    ("stale_manifest", lambda m: m.update({"fie/models/MANIFEST.json": m["fie/models/MANIFEST.json"].replace(b"\n", b"\r\n")}),
     "fie/models/MANIFEST.json: bytes differ from the manifest derived from the inventory"),
    ("case_variant", lambda m: m.update({"FIE/models/meta_clf.pkl": b"shadow"}), "unexpected member under fie/models/: FIE/models/meta_clf.pkl"),
])
def test_negative_controls_fail(inv, tmp_path, kind, case, mutate, expected):
    members = good_members(inv)
    mutate(members)
    if kind == "wheel":
        result = check(make_wheel(tmp_path / f"{case}.whl", members), "wheel", inv)
    else:
        result = check(make_sdist(tmp_path / f"{case}.tar.gz", members | sdist_inputs(inv)), "sdist", inv)
    assert not result["passed"]
    assert expected in failures(result), failures(result)


def test_duplicate_wheel_member_fails(inv, tmp_path):
    members = good_members(inv)
    wheel = make_wheel(tmp_path / "dup.whl", members,
                       duplicates={"fie/models/meta_clf.pkl": members["fie/models/meta_clf.pkl"]})
    result = check(wheel, "wheel", inv)
    assert not result["passed"] and "duplicate member: fie/models/meta_clf.pkl" in failures(result)


def test_duplicate_sdist_member_fails(inv, tmp_path):
    members = good_members(inv) | sdist_inputs(inv)
    sdist = make_sdist(tmp_path / "dup.tar.gz", members,
                       extra_raw=[("fie_sdk-0.0.0/fie/models/meta_clf.json", members["fie/models/meta_clf.json"])])
    result = check(sdist, "sdist", inv)
    assert not result["passed"] and "duplicate member: fie/models/meta_clf.json" in failures(result)


def test_members_differing_only_by_case_fail(inv, tmp_path):
    members = good_members(inv) | {"fie/models/Meta_clf.pkl": b"case twin"}
    result = check(make_wheel(tmp_path / "case.whl", members), "wheel", inv)
    assert "differ only by case" in failures(result)


@pytest.mark.parametrize("bad_name", ["fie/models/../models/meta_clf.pkl", "/fie/models/abs.pkl",
                                      "fie\\models\\win.pkl", "C:/fie/models/drive.pkl", "fie/./models/dot.pkl"])
def test_traversal_and_non_canonical_paths_fail(inv, tmp_path, bad_name):
    wheel = make_wheel(tmp_path / "trav.whl", good_members(inv) | {bad_name: b"x"})
    result = check(wheel, "wheel", inv)
    assert not result["passed"]
    if "\\" in bad_name and sys.platform == "win32":
        # On Windows, zipfile itself turns the backslashes into "/" when writing and reading, so
        # the member arrives as fie/models/win.pkl: refused as an unexpected model member instead.
        assert "unexpected member under fie/models/: fie/models/win.pkl" in failures(result)
    else:
        assert "not a plain relative path" in failures(result)


def test_links_in_the_model_area_fail(inv, tmp_path):
    wheel = check(make_wheel(tmp_path / "l.whl", good_members(inv), symlinks=("fie/models/link.pkl",)), "wheel", inv)
    assert not wheel["passed"] and "fie/models/link.pkl" in failures(wheel)
    sdist = check(make_sdist(tmp_path / "l.tar.gz", good_members(inv) | sdist_inputs(inv),
                             symlinks=("fie/models/link.pkl",)), "sdist", inv)
    assert not sdist["passed"] and "not a regular file (link): fie/models/link.pkl" in failures(sdist)


def test_sdist_must_have_one_top_level_directory(inv, tmp_path):
    sdist = make_sdist(tmp_path / "two.tar.gz", good_members(inv) | sdist_inputs(inv),
                       extra_raw=[("other-0.0.0/fie/models/meta_clf.pkl", b"x"), ("loose.txt", b"x")])
    text = failures(check(sdist, "sdist", inv))
    assert "exactly one top-level directory" in text and "file outside the top-level directory: loose.txt" in text


@pytest.mark.parametrize("missing", ["pyproject.toml", "hatch_build.py", "fie/_model_integrity.py",
                                     "scripts/model_manifest.json"])
def test_sdist_missing_a_build_input_fails(inv, tmp_path, missing):
    members = good_members(inv) | sdist_inputs(inv)
    members.pop(missing)
    result = check(make_sdist(tmp_path / "s.tar.gz", members), "sdist", inv)
    assert not result["passed"] and f"missing build input: {missing}" in failures(result)


def test_sdist_with_a_different_inventory_fails(inv, tmp_path):
    body = json.loads(inv["bytes"])
    body["artifacts"][0]["sha256"] = "0" * 64
    members = good_members(inv) | sdist_inputs(inv) | {"scripts/model_manifest.json": json.dumps(body).encode()}
    result = check(make_sdist(tmp_path / "s.tar.gz", members), "sdist", inv)
    assert "describes a different model set" in failures(result)


# ══ 4. The command line ═══════════════════════════════════════════════════════

def _cli(*args):
    return subprocess.run([sys.executable, "-I", str(CHECKER_PATH), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)


def test_cli_exit_codes_and_report(inv, tmp_path):
    good_w = make_wheel(tmp_path / "g.whl", good_members(inv))
    good_s = make_sdist(tmp_path / "g.tar.gz", good_members(inv) | sdist_inputs(inv))
    bad_w = make_wheel(tmp_path / "b.whl", good_members(inv) | {"fie/models/.gitkeep": b""})
    report = tmp_path / "report.json"
    ok = _cli("--inventory", inv["path"], "--sdist", good_s, "--wheel", good_w, "--report", report)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "PASSED (2 archive(s))" in ok.stdout
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["passed"] and len(data["results"]) == 2 and len(data["expected"]["artifacts"]) == 16
    bad = _cli("--inventory", inv["path"], "--sdist", good_s, "--wheel", bad_w)
    assert bad.returncode == 1 and "[FAIL] wheel b.whl" in bad.stdout and "fie/models/.gitkeep" in bad.stdout
    assert str(tmp_path) not in bad.stdout + bad.stderr, "diagnostics name archives by file name only"


def test_cli_unusable_inventory_or_arguments_exit_2(inv, tmp_path):
    wheel = make_wheel(tmp_path / "g.whl", good_members(inv))
    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    assert _cli("--inventory", broken, "--wheel", wheel).returncode == 2
    assert _cli("--inventory", inv["path"]).returncode == 2
    assert _cli("--inventory", inv["path"], "--wheel", tmp_path / "absent.whl").returncode == 2


def test_checker_imports_neither_fie_nor_ml_packages(inv, tmp_path):
    wheel = make_wheel(tmp_path / "g.whl", good_members(inv))
    code = ("import json, runpy, sys\n"
            f"sys.argv = ['v', '--inventory', {str(inv['path'])!r}, '--wheel', {str(wheel)!r}]\n"
            "try:\n"
            f"    runpy.run_path({str(CHECKER_PATH)!r}, run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    code = e.code\n"
            "watched = ('fie', 'numpy', 'joblib', 'sklearn', 'xgboost', 'onnxruntime', 'pickle', 'torch')\n"
            "print(json.dumps({'code': code, 'loaded': sorted(m for m in watched if m in sys.modules)}))\n")
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {"code": 0, "loaded": []}
