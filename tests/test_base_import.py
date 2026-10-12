"""
tests/test_base_import.py — `import fie` must not need optional packages (WP-021 phase 1).

E51 measured that a base install of fie-sdk could not be imported: `fie/__init__.py` imported
`fie.integrations`, which imported the FastAPI middleware, which needs Starlette. Starlette is
not a base dependency. The repository's tests never saw it, because the development
environment has Starlette installed.

These tests take the optional packages away. Each case runs in a fresh interpreter in which an
import hook makes the named packages unimportable, exactly as if they were not installed.

They import the working tree. They are unit tests of the import contract; the proof for a
built wheel is scripts/release_check.py, which installs the wheel into a clean environment.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# What `pip install fie-sdk` gives you, as import names (pyproject.toml [project].dependencies).
BASE_IMPORT_NAMES = {"requests", "deep_translator", "langdetect"}

# Everything an extra can bring. None of it may be needed by `import fie`.
OPTIONAL = ["starlette", "fastapi", "openai", "anthropic", "joblib", "sklearn", "xgboost", "numpy",
            "scipy", "pandas", "onnxruntime", "tokenizers", "torch", "sentence_transformers"]

_PRELUDE = """
import importlib.abc, json, sys

class _Hide(importlib.abc.MetaPathFinder):
    '''Makes top-level packages unimportable, as if they were not installed.'''
    def __init__(self, hidden, broken):
        self.hidden, self.broken = set(hidden), dict(broken)
    def find_spec(self, name, path=None, target=None):
        if name in self.broken:
            kind, message, missing = self.broken[name]
            if kind == "ModuleNotFoundError":
                raise ModuleNotFoundError(message, name=missing)
            raise {"ImportError": ImportError, "RuntimeError": RuntimeError}[kind](message)
        top = name.split(".")[0]
        if top in self.hidden:
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)
        return None

sys.meta_path.insert(0, _Hide(%(hidden)r, %(broken)r))

def describe(exc):
    return {"type": type(exc).__name__, "name": getattr(exc, "name", None), "message": str(exc),
            "cause": type(exc.__cause__).__name__ if exc.__cause__ is not None else None}
"""


def run(body: str, hidden=(), broken=None) -> dict:
    """Run `body` in a fresh interpreter and return the JSON object it prints last."""
    env = {**os.environ, "FIE_NO_TELEMETRY": "1", "FIE_NO_AUTO_DOWNLOAD": "1"}
    code = _PRELUDE % {"hidden": list(hidden), "broken": broken or {}} + body
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO),
                          env=env, timeout=300)
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ── import fie without anything optional ────────────────────────────────────

@pytest.fixture(scope="module")
def bare():
    """`import fie` with every optional package hidden."""
    return run("""
import fie
import fie.integrations as integ
unresolved = []
for n in fie.__all__:
    try:
        getattr(fie, n)
    except BaseException as exc:
        unresolved.append([n, type(exc).__name__])
print(json.dumps({
    "all": list(fie.__all__), "unresolved": unresolved,
    "integrations_all": list(integ.__all__),
    "has_openai": "openai" in vars(integ), "has_anthropic": "anthropic" in vars(integ),
    "fastapi_bound": "fastapi" in vars(integ),
    "fastapi_module_loaded": "fie.integrations.fastapi" in sys.modules,
    "optional_loaded": sorted(m for m in %r if m in sys.modules),
    "dir_lists_fastapi": "fastapi" in dir(integ),
    "wrapper_modules": [integ.openai.__name__, integ.anthropic.__name__],
}))
""" % (OPTIONAL,), hidden=OPTIONAL)


def test_import_fie_succeeds_without_optional_packages(bare):
    assert bare["all"], "fie.__all__ is empty"


def test_every_public_name_resolves(bare):
    assert bare["unresolved"] == []
    assert "integrations" in bare["all"]


def test_no_optional_package_is_imported(bare):
    assert bare["optional_loaded"] == []


def test_sdk_wrappers_import_without_their_sdks(bare):
    assert bare["has_openai"] and bare["has_anthropic"]
    assert bare["wrapper_modules"] == ["fie.integrations.openai", "fie.integrations.anthropic"]


def test_fastapi_integration_is_not_loaded_by_import_fie(bare):
    assert bare["fastapi_bound"] is False
    assert bare["fastapi_module_loaded"] is False


def test_integrations_all_is_the_two_wrappers(bare):
    assert bare["integrations_all"] == ["openai", "anthropic"]


def test_dir_does_not_offer_fastapi_when_starlette_is_absent(bare):
    assert bare["dir_lists_fastapi"] is False


def test_star_import_and_attribute_walk_work_without_starlette():
    out = run("""
import inspect
ns = {}
exec("from fie.integrations import *", ns)
import fie.integrations as integ
members = [name for name, _ in inspect.getmembers(integ)]
print(json.dumps({"star": sorted(k for k in ns if not k.startswith("__")), "walked": "openai" in members}))
""", hidden=OPTIONAL)
    assert out["star"] == ["anthropic", "openai"]
    assert out["walked"] is True


# ── asking for the FastAPI integration without Starlette ────────────────────

REQUESTS = {
    "from_submodule": "from fie.integrations.fastapi import FIEMiddleware",
    "from_package": "from fie.integrations import fastapi",
    "attribute": "import fie; fie.integrations.fastapi",
    "import_module": "import fie.integrations.fastapi",
}


@pytest.mark.parametrize("form", sorted(REQUESTS))
def test_requesting_fastapi_without_starlette_fails_clearly(form):
    out = run(f"""
import fie
try:
    exec({REQUESTS[form]!r})
    result = {{"raised": False}}
except BaseException as exc:
    result = {{"raised": True, **describe(exc)}}
result["fie_still_usable"] = callable(fie.scan_prompt)
print(json.dumps(result))
""", hidden=OPTIONAL)
    assert out["raised"] is True
    assert out["type"] == "ModuleNotFoundError"
    assert out["name"] == "starlette"
    assert 'pip install "fie-sdk[fastapi]"' in out["message"]
    assert out["cause"] == "ModuleNotFoundError"          # the original error is chained, not dropped
    assert out["fie_still_usable"] is True


def test_the_failed_request_can_be_repeated():
    """A failed import must not leave a half-initialised module behind."""
    out = run("""
import fie
seen = []
for _ in range(2):
    try:
        from fie.integrations.fastapi import FIEMiddleware
        seen.append("imported")
    except ModuleNotFoundError as exc:
        seen.append(exc.name)
print(json.dumps({"seen": seen, "left_behind": "fie.integrations.fastapi" in sys.modules}))
""", hidden=OPTIONAL)
    assert out == {"seen": ["starlette", "starlette"], "left_behind": False}


# ── a real defect is not reported as "not installed" ────────────────────────

@pytest.mark.parametrize("kind, message, missing", [
    ("ImportError", "DLL load failed while importing _speedups", None),
    ("ModuleNotFoundError", "No module named 'anyio'", "anyio"),
    ("ModuleNotFoundError", "No module named 'starlette.requests'", "starlette.requests"),
    ("RuntimeError", "boom inside starlette", None),
])
def test_other_import_failures_propagate_unchanged(kind, message, missing):
    pytest.importorskip("starlette")
    out = run("""
import fie
try:
    from fie.integrations.fastapi import FIEMiddleware
    result = {"raised": False}
except BaseException as exc:
    result = {"raised": True, **describe(exc)}
print(json.dumps(result))
""", broken={"starlette.requests": [kind, message, missing]})
    assert out["raised"] is True
    assert out["type"] == kind
    assert out["message"] == message
    assert out["name"] == missing
    assert "fie-sdk[fastapi]" not in out["message"]
    assert out["cause"] is None                              # not wrapped, not relabelled


def test_unknown_attribute_is_still_an_attribute_error():
    out = run("""
import fie.integrations as integ
try:
    integ.no_such_integration
    result = {"raised": False}
except BaseException as exc:
    result = {"raised": True, **describe(exc)}
print(json.dumps(result))
""", hidden=OPTIONAL)
    assert out["type"] == "AttributeError"
    assert "no_such_integration" in out["message"]


# ── with Starlette installed, the integration is still lazy ─────────────────

def test_fastapi_integration_loads_on_request_when_starlette_is_installed():
    pytest.importorskip("starlette")
    out = run("""
import fie
import fie.integrations as integ
before = {"starlette": "starlette" in sys.modules, "module": "fie.integrations.fastapi" in sys.modules,
          "listed": "fastapi" in dir(integ)}
module = integ.fastapi
after = {"starlette": "starlette" in sys.modules, "bound": vars(integ).get("fastapi") is module,
         "same": integ.fastapi is module, "has_middleware": hasattr(module, "FIEMiddleware")}
print(json.dumps({"before": before, "after": after}))
""")
    assert out["before"] == {"starlette": False, "module": False, "listed": True}
    assert out["after"] == {"starlette": True, "bound": True, "same": True, "has_middleware": True}


# ── the static form: no eager optional import anywhere in the closure ───────

def _module_file(name: str) -> Path | None:
    path = REPO / Path(*name.split("."))
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return path.with_suffix(".py") if path.with_suffix(".py").is_file() else None


def _import_closure() -> tuple[set, dict]:
    """Modules reached from `fie` through module-level imports, and the third-party names they need."""
    stdlib = set(sys.stdlib_module_names)
    seen: set = set()
    third_party: dict = {}

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        source = _module_file(name)
        if source is None:
            return
        package = name if source.name == "__init__.py" else name.rpartition(".")[0]

        def walk(body, guarded: bool) -> None:
            for node in body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        targets = [alias.name for alias in node.names]
                    else:
                        base = node.module or ""
                        if node.level:
                            parts = package.split(".")
                            base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                        targets = [base] + [f"{base}.{alias.name}" for alias in node.names]
                    for target in targets:
                        top = target.split(".")[0]
                        if top == "fie":
                            if not guarded and _module_file(target):
                                visit(target)
                        elif top and top not in stdlib and not guarded:
                            third_party.setdefault(top, set()).add(f"{name}:{node.lineno}")
                elif isinstance(node, ast.Try):
                    walk(node.body, True)                 # a guarded import may fail without breaking `import fie`
                    for handler in node.handlers:
                        walk(handler.body, guarded)
                    walk(node.orelse, guarded)
                    walk(node.finalbody, guarded)
                elif isinstance(node, (ast.If, ast.With)):
                    walk(node.body, guarded)
                    walk(getattr(node, "orelse", []), guarded)

        walk(ast.parse(source.read_text(encoding="utf-8")).body, False)

    visit("fie")
    return seen, third_party


def test_import_closure_needs_only_base_dependencies():
    """Fails the moment a module that `import fie` reaches imports an optional package at module level."""
    seen, third_party = _import_closure()
    assert "fie.adversarial" in seen and "fie.integrations" in seen, "the walk did not reach the package"
    extra = {name: sorted(where) for name, where in third_party.items() if name not in BASE_IMPORT_NAMES}
    assert extra == {}, f"module-level imports of non-base packages reachable from `import fie`: {extra}"


def test_fastapi_integration_is_outside_the_import_closure():
    seen, _ = _import_closure()
    assert "fie.integrations.fastapi" not in seen
    assert {"fie.integrations.openai", "fie.integrations.anthropic"} <= seen
