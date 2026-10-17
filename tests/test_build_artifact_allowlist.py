"""
The model packaging allowlist (WP-021 Stage 3.3b, decisions D-048 to D-052).

hatch_build.py packages exactly the `in_wheel: true` artifacts of scripts/model_manifest.json,
after verifying their size and SHA-256, into the sdist and the wheel, together with
fie/models/MANIFEST.json derived from the inventory at build time. Nothing else under
fie/models/ is packed, and a missing or altered approved artifact fails the build.

Three layers of tests:

  * plan(): the selection and verification logic, on synthetic inventories and synthetic file
    contents in temporary directories. Runs everywhere; needs no build tool.
  * pyproject.toml: the build configuration that makes the hook the only way model files enter
    a distribution. Runs everywhere.
  * real builds: `python -m build --no-isolation` on a synthetic project that uses the real
    pyproject.toml, .gitignore, hatch_build.py and fie/_model_integrity.py. Needs `build` and
    hatchling installed locally; skipped where they are not (the CI test jobs do not install
    them), never downloaded.

No model is ever deserialized here: synthetic artifacts are short byte strings, and the real
artifacts are only hashed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fie import _model_integrity as mi  # noqa: E402

HOOK_PATH = ROOT / "hatch_build.py"
TRACKED = ROOT / "scripts" / "model_manifest.json"
# The derived manifest of today's inventory (Stage 3.1 evidence). Changes only together with a
# reviewed change to scripts/model_manifest.json.
ESTABLISHED_MANIFEST = ("fbab28ab75acad2cfa8d178617caad112c32adafc6048d2a8483cd4a393bf2f5", 5112)
EXPERIMENT_NAMES = ("v2", "v3_w3x", "v6_1_contaminated", "v6_noaug", "v64_harmaug", "v64_harmaug_capped",
                    "v65_benignaug")


def _load_hook():
    spec = importlib.util.spec_from_file_location("_fie_hatch_build_under_test", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── a synthetic project ──────────────────────────────────────────────────────

def synthetic_inventory() -> list[dict]:
    """A valid production inventory built from the project's own rules, with synthetic bytes."""
    entries = []

    def add(path, role, version=None, default=False):
        data = f"synthetic artifact for packaging tests, never deserialized: {path}\n".encode()
        item = {"path": path, "sha256": _sha(data), "size": len(data), "role": role,
                "in_wheel": mi._IN_WHEEL[role], "default": default, "_bytes": data}
        if version is not None:
            item["version"] = version
        entries.append(item)

    for key in mi.PAIR_VERSIONS:
        clf, side = mi._PAIR_FILES[key]
        add(mi.PACKAGE_PREFIX + clf, "pair_classifier", key, default=(key == mi.DEFAULT_PAIR_VERSION))
        add(mi.PACKAGE_PREFIX + side, "pair_meta", key)
    add(mi.PACKAGE_PREFIX + "meta_clf.pkl", "meta_classifier")
    add(mi.PACKAGE_PREFIX + "meta_clf.json", "meta_classifier_meta")
    add(mi.PACKAGE_PREFIX + "minilm-onnx/model.onnx", "encoder")
    add(mi.PACKAGE_PREFIX + "minilm-onnx/tokenizer.json", "tokenizer")
    for path in sorted(mi.LEGACY_PATHS):
        add(path, "legacy")
    add("models/server_classifier.pkl", "server")
    return entries


def make_project(root: Path, *, write_off_wheel=False) -> list[dict]:
    """The real build configuration and hook around a synthetic package and inventory."""
    for rel in ("pyproject.toml", ".gitignore", "hatch_build.py", "fie/_model_integrity.py", "LICENSE"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    (root / "fie" / "__init__.py").write_text('"""synthetic package for packaging tests"""\n', encoding="utf-8")
    (root / "README.md").write_text("synthetic\n", encoding="utf-8")
    entries = synthetic_inventory()
    for e in entries:
        if e["in_wheel"] or write_off_wheel:
            (root / e["path"]).parent.mkdir(parents=True, exist_ok=True)
            (root / e["path"]).write_bytes(e["_bytes"])
    inventory = {"schema": 1, "release_tag": "synthetic", "repo": "synthetic/synthetic",
                 "artifacts": [{k: v for k, v in e.items() if k != "_bytes"} for e in entries]}
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "model_manifest.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return entries


def approved_paths(entries) -> set[str]:
    return {e["path"] for e in entries if e["in_wheel"]}


def add_extras(root: Path) -> dict[str, str]:
    """Everything that must never ship, under the names it has in a real checkout."""
    extras = {}
    for name in EXPERIMENT_NAMES:
        extras[f"fie/models/pair_intent_classifier_{name}.pkl"] = "experiment"
        extras[f"fie/models/pair_intent_meta_{name}.json"] = "experiment"
    for name in ("config.json", "ort_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"):
        extras[f"fie/models/minilm-onnx/{name}"] = "encoder by-product"
    extras["fie/models/sub/nested_extra.json"] = "nested, not ignored by .gitignore"
    extras["fie/models/notes.txt"] = "not ignored by .gitignore"
    extras["fie/models/stray.pkl"] = "unlisted pickle"
    for rel in extras:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(f"extra file, must not ship: {rel}\n".encode())
    (root / "fie" / "models" / ".gitkeep").write_bytes(b"")
    (root / "fie" / "models" / "MANIFEST.json").write_bytes(b'{"stale": "generated earlier, must not ship"}\n')
    return extras


# ══ 1. plan(): selection and verification ════════════════════════════════════

def test_selection_is_exactly_the_in_wheel_entries(tmp_path):
    entries = make_project(tmp_path, write_off_wheel=True)
    result = hook.plan(tmp_path)
    assert set(result["include"].values()) == approved_paths(entries)
    assert len(result["include"]) == 16
    for source, target in result["include"].items():
        assert Path(source) == tmp_path.joinpath(*target.split("/")) and target.startswith("fie/models/")


def test_real_inventory_selects_its_sixteen_in_wheel_entries():
    """The selection comes from the tracked inventory; this guards today's count and roles."""
    tracked = json.loads(TRACKED.read_text(encoding="utf-8"))["artifacts"]
    in_wheel = [a for a in tracked if a["in_wheel"]]
    assert len(in_wheel) == 16
    assert {a["role"] for a in in_wheel} == {"pair_classifier", "pair_meta", "meta_classifier", "meta_classifier_meta"}
    assert all(a["path"].startswith("fie/models/") for a in in_wheel)
    assert not any(a["role"] in ("legacy", "server", "encoder", "tokenizer") for a in in_wheel)


def test_real_tree_verifies_against_the_tracked_inventory():
    """The released artifacts in this checkout pass the build's own check (bytes hashed only)."""
    tracked = {a["path"]: a for a in json.loads(TRACKED.read_text(encoding="utf-8"))["artifacts"]}
    try:
        result = hook.plan(ROOT)
    except hook.ModelArtifactError as exc:
        pytest.fail(f"the approved artifacts are not all present and verified in this checkout: {exc}. "
                    "Fetch them with `python scripts/download_models.py`")
    assert sorted(result["include"].values()) == sorted(p for p, a in tracked.items() if a["in_wheel"])
    assert result["manifest"] == mi.derive_package_manifest(TRACKED.read_bytes())


@pytest.mark.parametrize("damage", ["flip_byte", "append_byte", "truncate", "replace"])
def test_altered_approved_artifact_fails(tmp_path, damage):
    make_project(tmp_path)
    victim = tmp_path / "fie/models/pair_intent_classifier_v6_3b.pkl"
    data = victim.read_bytes()
    victim.write_bytes({"flip_byte": bytes([data[0] ^ 1]) + data[1:], "append_byte": data + b"x",
                        "truncate": data[:-1], "replace": b"other"}[damage])
    with pytest.raises(hook.ModelArtifactError) as info:
        hook.plan(tmp_path)
    message = str(info.value)
    assert "fie/models/pair_intent_classifier_v6_3b.pkl: mismatch" in message
    assert "1 of 16 approved model artifacts" in message
    assert str(tmp_path) not in message and tmp_path.name not in message


def test_missing_approved_artifact_fails(tmp_path):
    make_project(tmp_path)
    (tmp_path / "fie/models/meta_clf.json").unlink()
    (tmp_path / "fie/models/pair_intent_meta_v5.json").unlink()
    with pytest.raises(hook.ModelArtifactError) as info:
        hook.plan(tmp_path)
    message = str(info.value)
    assert "2 of 16" in message and "fie/models/meta_clf.json: missing" in message
    assert "fie/models/pair_intent_meta_v5.json: missing" in message
    assert str(tmp_path) not in message


def test_no_model_files_at_all_fails(tmp_path):
    make_project(tmp_path)
    shutil.rmtree(tmp_path / "fie" / "models")
    with pytest.raises(hook.ModelArtifactError) as info:
        hook.plan(tmp_path)
    assert "16 of 16" in str(info.value)


def test_off_wheel_entries_are_neither_required_nor_included(tmp_path):
    entries = make_project(tmp_path, write_off_wheel=False)
    assert set(hook.plan(tmp_path)["include"].values()) == approved_paths(entries)      # absent: fine
    make_project(tmp_path, write_off_wheel=True)
    result = hook.plan(tmp_path)
    off = {e["path"] for e in entries if not e["in_wheel"] and e["path"].startswith("fie/models/")}
    assert not off & set(result["include"].values())
    assert set(result["off_wheel_present"]) == off
    assert {"fie/models/failure_classifier_v4.pkl", "fie/models/feature_columns_v4.pkl"} <= off


def test_extras_experiments_and_gitkeep_are_never_selected(tmp_path):
    entries = make_project(tmp_path, write_off_wheel=True)
    extras = add_extras(tmp_path)
    result = hook.plan(tmp_path)
    assert set(result["include"].values()) == approved_paths(entries)
    assert set(result["unlisted"]) == set(extras), "every unlisted file is reported"
    assert "fie/models/.gitkeep" not in result["unlisted"] and "fie/models/MANIFEST.json" not in result["unlisted"]
    assert "fie/models/.gitkeep" not in result["include"].values()


def test_manifest_is_derived_even_when_a_stale_one_is_present(tmp_path):
    make_project(tmp_path)
    stale = b'{"stale": true}\n'
    (tmp_path / "fie/models/MANIFEST.json").write_bytes(stale)
    result = hook.plan(tmp_path)
    expected = mi.derive_package_manifest((tmp_path / "scripts/model_manifest.json").read_bytes())
    assert result["manifest"] == expected != stale
    assert "fie/models/MANIFEST.json" not in result["include"].values(), "the working-tree file is never packed"


def test_derivation_is_deterministic_and_keeps_the_established_bytes():
    source = TRACKED.read_bytes()
    outputs = {mi.derive_package_manifest(source) for _ in range(3)}
    assert len(outputs) == 1
    (derived,) = outputs
    assert (_sha(derived), len(derived)) == ESTABLISHED_MANIFEST
    assert b"\r" not in derived and derived.endswith(b"\n")


def test_invalid_inventory_fails_without_packaging_anything(tmp_path):
    make_project(tmp_path)
    inv = json.loads((tmp_path / "scripts/model_manifest.json").read_text(encoding="utf-8"))
    inv["artifacts"][0]["in_wheel"] = False          # a pair classifier may not leave the wheel
    (tmp_path / "scripts/model_manifest.json").write_text(json.dumps(inv), encoding="utf-8")
    with pytest.raises(hook.ModelArtifactError) as info:
        hook.plan(tmp_path)
    assert "not a valid production inventory" in str(info.value)


def test_missing_inventory_or_integrity_module_fails(tmp_path):
    make_project(tmp_path)
    (tmp_path / "scripts/model_manifest.json").unlink()
    with pytest.raises(hook.ModelArtifactError):
        hook.plan(tmp_path)
    make_project(tmp_path)
    (tmp_path / "fie/_model_integrity.py").unlink()
    with pytest.raises(hook.ModelArtifactError):
        hook.plan(tmp_path)


def test_hook_imports_neither_fie_nor_ml_packages(tmp_path):
    make_project(tmp_path)
    code = ("import importlib.util, json, sys\n"
            f"spec = importlib.util.spec_from_file_location('h', {str(HOOK_PATH)!r})\n"
            "h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)\n"
            f"r = h.plan({str(tmp_path)!r})\n"
            "watched = ('fie', 'numpy', 'joblib', 'sklearn', 'xgboost', 'onnxruntime', 'pickle', 'torch')\n"
            "print(json.dumps({'loaded': sorted(m for m in watched if m in sys.modules), 'n': len(r['include'])}))\n")
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {"loaded": [], "n": 16}


# ══ 2. The build configuration ═══════════════════════════════════════════════

def _pyproject() -> dict:
    try:
        import tomllib as toml
    except ImportError:
        try:
            import tomli as toml
        except ImportError:
            from pip._vendor import tomli as toml
    return toml.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path + (k,), v
            yield from _walk(v, path + (k,))


# The sdist allowlist (WP-021 W21-29, Stage 3.3d). Pinned: an addition or a removal fails.
SDIST_ONLY_INCLUDE = ["/fie", "/hatch_build.py", "/scripts/model_manifest.json", "/README.md", "/LICENSE"]
# The one JSON file a selection key may name: the inventory, and only in the sdist's
# only-include. It is the build hook's input, not a model artifact.
_INVENTORY_ENTRY = (("build", "targets", "sdist", "only-include"), "/scripts/model_manifest.json")
_SELECTION_KEYS = ("artifacts", "include", "only-include", "force-include", "sources", "shared-data")


def _model_pattern_violations(hatch: dict) -> list:
    """Every selection key that names model files: anything with `models`, `.pkl` or `.json`."""
    violations = []
    for key_path, value in _walk(hatch):
        if key_path[-1] not in _SELECTION_KEYS:
            continue
        if key_path == _INVENTORY_ENTRY[0] and isinstance(value, list):
            value = [v for v in value if v != _INVENTORY_ENTRY[1]]
        text = json.dumps(value)
        if "models" in text or ".pkl" in text or ".json" in text:
            violations.append((key_path, value))
    return violations


def test_no_pattern_selects_model_files():
    """
    No `artifacts`, `include`, `only-include` or `force-include` anywhere may name model files.
    The single exception is the inventory, /scripts/model_manifest.json, in the sdist's
    only-include; and that list is pinned exactly.
    """
    hatch = _pyproject()["tool"]["hatch"]
    assert _model_pattern_violations(hatch) == []
    assert hatch["build"]["targets"]["sdist"]["only-include"] == SDIST_ONLY_INCLUDE


@pytest.mark.parametrize("key_path, value", [
    (("build", "targets", "sdist", "only-include"), SDIST_ONLY_INCLUDE + ["/fie/models"]),
    (("build", "targets", "sdist", "only-include"), SDIST_ONLY_INCLUDE + ["/fie/models/meta_clf.pkl"]),
    (("build", "targets", "sdist", "only-include"), SDIST_ONLY_INCLUDE + ["/fie/models/MANIFEST.json"]),
    (("build", "targets", "sdist", "only-include"), SDIST_ONLY_INCLUDE + ["/data/meta_clf_trainset.json"]),
    (("build", "targets", "wheel", "only-include"), ["/scripts/model_manifest.json"]),   # exception is sdist-only
    (("build", "artifacts"), ["fie/models/*.pkl"]),
    (("build", "targets", "sdist", "include"), ["*.json"]),
    (("build", "targets", "wheel", "force-include"), {"x/meta_clf.pkl": "fie/models/meta_clf.pkl"}),
])
def test_guard_still_rejects_model_patterns(key_path, value):
    """Negative controls: the inventory exception admits that one path and nothing else."""
    hatch = json.loads(json.dumps(_pyproject()["tool"]["hatch"]))
    node = hatch
    for key in key_path[:-1]:
        node = node.setdefault(key, {})
    node[key_path[-1]] = value
    assert _model_pattern_violations(hatch), f"{key_path} = {value} was not caught"


def test_fie_models_is_excluded_and_the_hook_is_registered():
    cfg = _pyproject()
    build = cfg["tool"]["hatch"]["build"]
    assert "/fie/models/**" in build["exclude"]
    assert build["hooks"]["custom"] == {"path": "hatch_build.py"}
    assert cfg["build-system"]["requires"] == ["hatchling==1.29.0"]
    assert cfg["build-system"]["build-backend"] == "hatchling.build"
    assert cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["fie"]


def test_hook_class_is_a_hatchling_build_hook():
    pytest.importorskip("hatchling")
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface

    assert issubclass(hook.ModelAllowlistHook, BuildHookInterface)


# ══ 3. Real builds of a synthetic project (offline, local tools only) ═════════

def _build_tools():
    if importlib.util.find_spec("build") is None or importlib.util.find_spec("hatchling") is None:
        pytest.skip("`build` and hatchling are not installed here; they are never downloaded by these tests")
    from importlib.metadata import version

    if version("hatchling") != "1.29.0":
        pytest.skip(f"hatchling {version('hatchling')} installed; these tests pin the measured 1.29.0")


def _run_build(kind: str, src: Path, out: Path):
    env = {**os.environ, "PIP_NO_INDEX": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("SOURCE_DATE_EPOCH", None)
    return subprocess.run([sys.executable, "-m", "build", "--no-isolation", f"--{kind}", "--outdir", str(out), str(src)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300)


def _wheel_members(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        return {n: _sha(z.read(n)) for n in z.namelist() if not n.endswith("/")}


def _sdist_members(path: Path) -> dict[str, str]:
    with tarfile.open(path) as t:
        return {m.name.split("/", 1)[1]: _sha(t.extractfile(m).read()) for m in t.getmembers() if m.isfile()}


def _models(members: dict[str, str]) -> dict[str, str]:
    return {n: h for n, h in members.items() if n.startswith("fie/models/")}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One synthetic project with every kind of extra, built as sdist, wheel, and wheel-from-sdist."""
    _build_tools()
    root = tmp_path_factory.mktemp("allowlist") / "project"
    root.mkdir()
    entries = make_project(root, write_off_wheel=True)
    extras = add_extras(root)
    out = root.parent / "dist"
    sdist = _run_build("sdist", root, out / "sdist")
    wheel = _run_build("wheel", root, out / "wheel")
    assert sdist.returncode == 0, sdist.stdout[-3000:] + sdist.stderr[-3000:]
    assert wheel.returncode == 0, wheel.stdout[-3000:] + wheel.stderr[-3000:]
    sdist_path = next((out / "sdist").glob("*.tar.gz"))
    unpacked = root.parent / "unpacked"
    with tarfile.open(sdist_path) as t:
        t.extractall(unpacked)
    from_sdist = _run_build("wheel", next(unpacked.iterdir()), out / "from_sdist")
    assert from_sdist.returncode == 0, from_sdist.stdout[-3000:] + from_sdist.stderr[-3000:]
    expected = {e["path"]: e["sha256"] for e in entries if e["in_wheel"]}
    expected["fie/models/MANIFEST.json"] = _sha(mi.derive_package_manifest((root / "scripts/model_manifest.json").read_bytes()))
    return {"expected": expected, "extras": extras, "log": sdist.stdout + sdist.stderr,
            "sdist": _sdist_members(sdist_path), "wheel": _wheel_members(next((out / "wheel").glob("*.whl"))),
            "from_sdist": _wheel_members(next((out / "from_sdist").glob("*.whl")))}


@pytest.mark.parametrize("archive", ["sdist", "wheel", "from_sdist"])
def test_archive_model_members_are_exactly_the_allowlist(built, archive):
    assert _models(built[archive]) == built["expected"], "model members must be the 16 approved files + MANIFEST.json"


def test_wheel_from_sdist_has_the_same_model_members_as_the_direct_wheel(built):
    assert _models(built["from_sdist"]) == _models(built["wheel"]) == _models(built["sdist"])


@pytest.mark.parametrize("archive", ["sdist", "wheel", "from_sdist"])
def test_nothing_unapproved_ships(built, archive):
    names = set(built[archive])
    for rel in list(built["extras"]) + ["fie/models/.gitkeep", "fie/models/failure_classifier_v4.pkl",
                                         "fie/models/feature_columns_v4.pkl", "fie/models/minilm-onnx/model.onnx",
                                         "fie/models/minilm-onnx/tokenizer.json", "models/server_classifier.pkl"]:
        assert rel not in names, f"{rel} shipped in the {archive}"


def test_sdist_carries_the_hook_inputs(built):
    for rel in ("hatch_build.py", "pyproject.toml", ".gitignore", "scripts/model_manifest.json",
                "fie/_model_integrity.py"):
        assert rel in built["sdist"], rel


def test_build_warns_about_unlisted_files(built):
    assert "are not in scripts/model_manifest.json and are NOT packaged" in built["log"]
    assert "fie/models/pair_intent_classifier_v64_harmaug.pkl" in built["log"]


@pytest.mark.parametrize("damage", ["missing", "tampered"])
@pytest.mark.parametrize("kind", ["sdist", "wheel"])
def test_build_fails_closed(tmp_path, kind, damage):
    _build_tools()
    root = tmp_path / "project"
    root.mkdir()
    make_project(root)
    victim = root / "fie/models/meta_clf.pkl"
    if damage == "missing":
        victim.unlink()
    else:
        victim.write_bytes(b"tampered")
    proc = _run_build(kind, root, tmp_path / "out")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "model packaging refused" in output and "fie/models/meta_clf.pkl" in output
    assert ("missing" if damage == "missing" else "mismatch") in output
    assert not list((tmp_path / "out").glob("*")) if (tmp_path / "out").exists() else True


# ══ 4. The narrowed sdist (WP-021 W21-29, Stage 3.3d) ═════════════════════════
# The sdist is a minimal build input: the `fie` package, the build hook, the inventory,
# README and LICENSE, plus the approved model artifacts and MANIFEST.json the hook adds, and
# the PKG-INFO and .gitignore hatchling writes. Everything else in a real checkout stays out.

SDIST_SIZE_CEILING = 2_000_000          # bytes; today's narrowed sdist is about 0.27 MB

# What a real checkout holds besides the package, under the names it really has.
REPOSITORY_ONLY_FILES = {
    "brag-output/brag.mp4": b"\0" * 4096,
    "brag-output/composition/assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3": b"ID3" + b"\0" * 4096,
    "brag-output-2026-09-29-113038/composition/index.html": b"<html></html>\n",
    "docs/media/fie-explainer.mp4": b"\0" * 4096,
    "docs/ARCHITECTURE.md": b"# docs\n",
    "docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3a/builds.json": b"{}\n",
    "docs/fie_rebuild_2026/experiments/E51/run.py": b"# experiment\n",
    "evals/baselines/BL-0002_pair-v6.3b_fie-70de7053/records/std.advbench.jsonl": b"{}\n",
    "evals/__init__.py": b"",
    "data/meta_clf_trainset.jsonl": b"{}\n",
    "notebooks/train_pair_classifier.ipynb": b"{}\n",
    "tests/test_something.py": b"def test_x():\n    pass\n",
    "app/main.py": b"# server\n",
    "engine/core.py": b"# server\n",
    "Frontend/src/App.jsx": b"// frontend\n",
    "deploy/huggingface/push_space.py": b"# deployment\n",
    "Dockerfile": b"FROM python:3.11\n",
    "requirements.txt": b"requests\n",
    ".github/workflows/ci.yml": b"name: CI\n",
    "scripts/download_models.py": b"# downloader\n",
    "scripts/release_check.py": b"# release gate\n",
    "scripts/train_pair_harmaug.py": b"# research\n",
    "evaluation/phase2/exp6_before_after/run_eval.py": b"# research\n",
    "CHANGELOG.md": b"# changes\n",
}


def plant_repository_files(root: Path) -> None:
    for rel, data in REPOSITORY_ONLY_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)


def expected_sdist_members(root: Path, entries) -> set[str]:
    """Exactly what the narrowed sdist may hold for the synthetic project at `root`."""
    package = {p.relative_to(root).as_posix() for p in (root / "fie").rglob("*")
               if p.is_file() and not p.relative_to(root).as_posix().startswith("fie/models/")}
    return (package | approved_paths(entries) | {"fie/models/MANIFEST.json"}
            | {"hatch_build.py", "pyproject.toml", "scripts/model_manifest.json", "README.md", "LICENSE",
               "PKG-INFO", ".gitignore"})


@pytest.fixture(scope="module")
def narrowed(tmp_path_factory):
    """A synthetic checkout with every kind of repository-only and unapproved file, built."""
    _build_tools()
    root = tmp_path_factory.mktemp("narrowed") / "project"
    root.mkdir()
    entries = make_project(root, write_off_wheel=True)
    extras = add_extras(root)                     # experiment files, by-products, nested, .gitkeep, stale manifest
    plant_repository_files(root)
    big = root / "brag-output-2026-09-29-113038" / "brag.mp4"
    big.write_bytes(os.urandom(5_000_000))        # incompressible: would breach the ceiling if packed
    out = root.parent / "dist"
    sdist = _run_build("sdist", root, out / "sdist")
    assert sdist.returncode == 0, sdist.stdout[-3000:] + sdist.stderr[-3000:]
    sdist_path = next((out / "sdist").glob("*.tar.gz"))
    unpacked = root.parent / "unpacked"
    with tarfile.open(sdist_path) as t:
        t.extractall(unpacked)
    wheel = _run_build("wheel", next(unpacked.iterdir()), out / "from_sdist")
    assert wheel.returncode == 0, wheel.stdout[-3000:] + wheel.stderr[-3000:]
    return {"root": root, "entries": entries, "extras": extras, "sdist_path": sdist_path,
            "sdist": _sdist_members(sdist_path), "wheel_path": next((out / "from_sdist").glob("*.whl")),
            "expected": expected_sdist_members(root, entries)}


def test_narrowed_sdist_holds_exactly_the_allowlist(narrowed):
    assert set(narrowed["sdist"]) == narrowed["expected"]


def test_repository_only_files_do_not_leak_into_the_sdist(narrowed):
    names = set(narrowed["sdist"])
    leaked = sorted(n for n in names if n in REPOSITORY_ONLY_FILES or n.endswith((".mp4", ".mp3", ".ipynb", ".jsonl")))
    assert leaked == []
    for top in ("brag-output", "docs", "evals", "data", "notebooks", "tests", "app", "engine", "Frontend",
                "deploy", ".github", "evaluation"):
        assert not any(n == top or n.startswith(top + "/") for n in names), top
    assert not any(n.startswith("brag-output") for n in names)


def test_experiment_files_and_evidence_do_not_leak_into_the_sdist(narrowed):
    names = set(narrowed["sdist"])
    for rel in list(narrowed["extras"]) + ["fie/models/.gitkeep", "fie/models/failure_classifier_v4.pkl",
                                            "fie/models/feature_columns_v4.pkl", "fie/models/minilm-onnx/model.onnx",
                                            "docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3a/builds.json",
                                            "docs/fie_rebuild_2026/experiments/E51/run.py"]:
        assert rel not in names, rel


def test_narrowed_sdist_carries_the_approved_models_and_build_inputs(narrowed):
    for e in narrowed["entries"]:
        if e["in_wheel"]:
            assert narrowed["sdist"].get(e["path"]) == e["sha256"], e["path"]
    derived = mi.derive_package_manifest((narrowed["root"] / "scripts/model_manifest.json").read_bytes())
    assert narrowed["sdist"]["fie/models/MANIFEST.json"] == _sha(derived)
    for rel in ("hatch_build.py", "pyproject.toml", "scripts/model_manifest.json", "fie/_model_integrity.py"):
        assert rel in narrowed["sdist"], rel


def test_narrowed_sdist_is_below_the_size_ceiling(narrowed):
    assert narrowed["sdist_path"].stat().st_size < SDIST_SIZE_CEILING


def test_wheel_from_the_narrowed_sdist_passes_the_archive_checker(narrowed):
    proc = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "verify_package_artifacts.py"),
                           "--inventory", str(narrowed["root"] / "scripts" / "model_manifest.json"),
                           "--sdist", str(narrowed["sdist_path"]), "--wheel", str(narrowed["wheel_path"])],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "verify_package_artifacts: PASSED (2 archive(s))" in proc.stdout


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_narrowed_sdist_build_still_fails_closed(tmp_path, damage):
    _build_tools()
    root = tmp_path / "project"
    root.mkdir()
    make_project(root)
    plant_repository_files(root)
    victim = root / "fie/models/pair_intent_classifier_v6_3b.pkl"
    if damage == "missing":
        victim.unlink()
    else:
        victim.write_bytes(b"tampered")
    proc = _run_build("sdist", root, tmp_path / "out")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "model packaging refused" in output and "fie/models/pair_intent_classifier_v6_3b.pkl" in output
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.tar.gz"))
