"""
Boundaries.

  * The harness is not on the request-serving path: nothing under fie/, engine/,
    app/ or storage/ imports it, and it is not packaged in the wheel.
  * The harness is safer than what it measures: no pickle, no eval/exec, no
    import-by-name from a registry, no network client of its own.
"""
from __future__ import annotations

import ast
import re

from _helpers import REPO_ROOT

PRODUCTION_DIRS = ("fie", "engine", "app", "storage")
EVALS = REPO_ROOT / "evals"


def _py_files(folder):
    return [p for p in sorted(folder.rglob("*.py")) if "__pycache__" not in p.parts]


def _imports(path):
    """Top-level names of every module a file imports, anywhere in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_no_production_module_imports_the_harness():
    offenders = []
    for folder in PRODUCTION_DIRS:
        for path in _py_files(REPO_ROOT / folder):
            if "evals" in _imports(path):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"production code must not depend on the harness: {offenders}"


def test_the_harness_is_not_packaged_in_the_wheel():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"\[tool\.hatch\.build\.targets\.wheel\]\s*packages\s*=\s*\[([^\]]*)\]", text)
    assert match, "wheel package list not found in pyproject.toml"
    packages = re.findall(r'"([^"]+)"', match.group(1))
    assert packages == ["fie"], packages
    assert "evals" not in packages


def test_the_harness_does_not_deserialize_code():
    banned = {"pickle", "joblib", "marshal", "shelve", "dill", "cloudpickle"}
    offenders = {str(p.relative_to(REPO_ROOT)): sorted(_imports(p) & banned)
                 for p in _py_files(EVALS) if _imports(p) & banned}
    assert not offenders, offenders


def test_the_harness_has_no_network_client_of_its_own():
    """Only hermetic.py may touch `socket`, and only to deny it."""
    clients = {"requests", "urllib3", "httpx", "aiohttp", "http", "ftplib", "smtplib", "websocket"}
    for path in _py_files(EVALS):
        used = _imports(path)
        if path.name == "hermetic.py":
            assert used & clients <= {"urllib3"}, used & clients      # the pre-arm import only
            continue
        assert not (used & (clients | {"socket", "_socket"})), f"{path.name}: {used & clients}"


def test_the_harness_never_evaluates_or_imports_by_name():
    for path in _py_files(EVALS):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                assert func.id not in ("eval", "exec", "compile", "__import__"), f"{path.name}: {func.id}()"
            elif isinstance(func, ast.Attribute) and func.attr == "import_module":
                # Allowed only with names from a fixed constant in subject.py.
                assert path.name == "subject.py", f"{path.name}: import_module()"


def test_only_two_harness_modules_start_processes():
    """The orchestrator starts git and the worker; the guard self-test probes denial."""
    users = {p.name for p in _py_files(EVALS) if "subprocess" in _imports(p)}
    assert users == {"orchestrator.py", "hermetic.py"}, users


def test_registries_are_plain_data():
    """A registry is JSON. Nothing in it names a Python object to call."""
    import json
    for path in sorted((EVALS / "registry").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        json.loads(text)
        for needle in ("__import__", "eval(", "exec(", "os.system", "subprocess", "importlib"):
            assert needle not in text, f"{path.name} contains {needle!r}"
    assert not list((EVALS / "registry").glob("*.py"))


def test_worker_imports_nothing_heavy_at_module_level():
    """The lite profile makes numpy unimportable; the worker must still start."""
    heavy = {"numpy", "sklearn", "joblib", "onnxruntime", "pandas", "xgboost", "fie"}
    for name in ("worker.py", "hermetic.py", "canonical.py", "datasets.py", "integrity.py",
                 "transforms.py", "fingerprint.py", "latency.py"):
        tree = ast.parse((EVALS / name).read_text(encoding="utf-8"))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        assert not (top & heavy), f"{name} imports {top & heavy} at module level"
