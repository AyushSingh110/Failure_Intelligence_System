"""
tests/test_release_check.py — the release gate's own judgement (WP-021 phase 2).

scripts/release_check.py installs a built wheel into fresh environments and decides whether it
may be released. These tests cover the deciding, not the installing: the pure functions that
turn command results into PASS, FAIL, NOT_EVALUATED or TOOL_FAILED, and those into an exit
code. No environment is created and nothing is installed here; CI's package job runs the gate
itself against the built wheel.

The rule under test: nothing but a real pass is a pass.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release_check.py"
_spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

GOOD, BAD = {"status": rc.PASS}, {"status": rc.FAIL}
SKIPPED, BROKEN = {"status": rc.NOT_EVALUATED}, {"status": rc.TOOL_FAILED}


def test_builtin_selftest_passes():
    assert rc.selftest() == 0


def test_selftest_runs_as_a_command():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--selftest"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "FAIL " not in proc.stdout


def test_both_targets_are_checked_by_default():
    assert [rc.TARGETS[t]["label"] for t in ("base", "fastapi")] == ["T-base", "T-fastapi"]
    assert rc.TARGETS["base"]["extra"] is None and rc.TARGETS["fastapi"]["extra"] == "fastapi"


@pytest.mark.parametrize("criteria, code", [
    ({"a": GOOD, "b": GOOD}, 0),
    ({"a": GOOD, "b": BAD}, 1),
    ({"a": BAD, "b": BROKEN, "c": SKIPPED}, 1),
    ({"a": GOOD, "b": BROKEN}, 2),
    ({"a": GOOD, "b": SKIPPED}, 2),
    ({}, 2),
])
def test_only_all_passes_gives_exit_zero(criteria, code):
    assert rc.overall(criteria)[0] == code


@pytest.mark.parametrize("base, fastapi, code", [
    (GOOD, GOOD, 0), (GOOD, BAD, 1), (BAD, GOOD, 1), (GOOD, SKIPPED, 2), (GOOD, BROKEN, 2), (BAD, BROKEN, 1),
])
def test_every_target_must_pass(base, fastapi, code):
    result, verdict = rc.overall_targets({"T-base": {"criteria": {"x": base}}, "T-fastapi": {"criteria": {"x": fastapi}}})
    assert result == code
    assert "T-base" in verdict and "T-fastapi" in verdict


def test_a_missing_target_is_not_a_pass():
    assert rc.overall_targets({})[0] == 2
    assert rc.overall_targets({"T-base": {"criteria": {"x": GOOD}}, "T-fastapi": {"criteria": {}}})[0] == 2


def test_the_e51_import_failure_is_a_failed_criterion_naming_the_module():
    result = rc.classify_import(1, {"import_ok": False, "exception": "ModuleNotFoundError",
                                    "missing_module": "starlette", "message": "No module named 'starlette'"})
    assert result["status"] == rc.FAIL and result["missing_module"] == "starlette"
    assert "starlette" in result["detail"]


def test_a_failed_tool_says_nothing_about_packages():
    result = rc.classify_installed_set(2, "", "fie-sdk", {})
    assert result["status"] == rc.TOOL_FAILED
    assert result["missing"] == rc.NOT_EVALUATED and result["undeclared"] == rc.NOT_EVALUATED


def _inspect(packages: dict, requires: dict) -> str:
    return json.dumps({"installed": [{"metadata": {"name": n, "version": v, "requires_dist": requires.get(n, [])}}
                                     for n, v in packages.items()]})


REQUIRES = {"fie-sdk": ["requests>=2.28.0", "starlette>=0.52.1; extra == 'fastapi'", "fastapi>=0.100; extra == 'server'"],
            "starlette": ["anyio<5,>=3.6.2"]}


def test_base_target_rejects_a_starlette_that_was_not_asked_for():
    out = rc.classify_installed_set(0, _inspect({"fie-sdk": "1", "requests": "2", "starlette": "0.52.1", "anyio": "4"},
                                                REQUIRES), "fie-sdk", {})
    assert out["status"] == rc.FAIL and out["undeclared"] == ["anyio", "starlette"]
    assert rc.classify_frameworks_absent(out["distributions"])["status"] == rc.FAIL


def test_fastapi_target_requires_the_extras_packages():
    without = rc.classify_installed_set(0, _inspect({"fie-sdk": "1", "requests": "2"}, REQUIRES), "fie-sdk", {},
                                        ("fastapi",))
    assert without["status"] == rc.FAIL and without["missing"] == ["starlette"]
    with_it = rc.classify_installed_set(0, _inspect({"fie-sdk": "1", "requests": "2", "starlette": "0.52.1", "anyio": "4"},
                                                    REQUIRES), "fie-sdk", {}, ("fastapi",))
    assert with_it["status"] == rc.PASS


def test_ac10_accepts_only_the_documented_error():
    documented = {"raised": True, "type": "ModuleNotFoundError", "name": "starlette",
                  "message": 'fie.integrations.fastapi needs Starlette, which is not installed. '
                             'Install it with: pip install "fie-sdk[fastapi]"'}
    assert rc.classify_fastapi_request_absent(0, {"a": documented})["status"] == rc.PASS
    for wrong in ({"raised": False}, {**documented, "type": "ImportError"}, {**documented, "name": "anyio"},
                  {**documented, "message": "No module named 'starlette'"}):
        assert rc.classify_fastapi_request_absent(0, {"a": documented, "b": wrong})["status"] == rc.FAIL


def test_middleware_must_block_the_injection_and_pass_the_benign_request():
    blocked = {"status": 400, "body": {"error": "adversarial_input_blocked", "attack_type": "PROMPT_EXTRACTION",
                                       "confidence": 0.82, "message": "Request blocked by FIE adversarial scan."}}
    passed = {"status": 200, "body": {"reached": "handler"}}
    ok = {"from_environment": True, "is_middleware": True, "same_object": True,
          "starlette_imported_by_import_fie": False, "starlette_version": "0.52.1", "injection": blocked, "benign": passed}
    assert rc.classify_middleware(0, ok)["status"] == rc.PASS
    assert rc.classify_middleware(0, {**ok, "injection": passed})["status"] == rc.FAIL       # fail-open would show here
    assert rc.classify_middleware(0, {**ok, "benign": blocked})["status"] == rc.FAIL
    assert rc.classify_middleware(0, {**ok, "from_environment": False})["status"] == rc.FAIL
    assert rc.classify_middleware(1, None)["status"] == rc.TOOL_FAILED


def test_usage_errors_are_not_a_pass(tmp_path):
    for args in ([], ["--wheel", str(tmp_path / "missing.whl"), "--out", str(tmp_path / "o")],
                 ["--wheel", "x.whl", "--out", str(tmp_path / "o"), "--targets", "nonsense"]):
        proc = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120)
        assert proc.returncode == 2, (args, proc.stdout)
