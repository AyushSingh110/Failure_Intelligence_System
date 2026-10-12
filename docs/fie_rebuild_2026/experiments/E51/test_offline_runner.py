"""
Regression tests for the offline runner's gate classification (E51, after offline attempt 4).

Run explicitly; this file is outside `tests/` and is not part of the repository's test suite:

    python -m pytest docs/fie_rebuild_2026/experiments/E51/test_offline_runner.py -p no:cacheprovider

Pure functions only: no subprocess, no pip, no network, no environment is touched.

Attempt 4: `pip inspect` died with a UnicodeEncodeError (exit 2) and the runner reported all 12
pinned distributions as missing. A tool that failed has said nothing about what is installed.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_runner as r  # noqa: E402

WANTED = {"beautifulsoup4": "4.15.0", "certifi": "2026.7.22", "charset-normalizer": "3.5.2",
          "deep-translator": "1.11.4", "fie-sdk": "1.18.0", "idna": "3.20", "langdetect": "1.0.9",
          "requests": "2.34.2", "six": "1.17.0", "soupsieve": "2.10", "typing-extensions": "4.16.0",
          "urllib3": "2.8.0"}
PREEXISTING = {"pip": "23.0.1", "setuptools": "79.0.1"}
DIFF_KEYS = ("missing", "wrong_version", "extra_beyond_pins_and_preexisting", "preexisting_changed")


def report(packages: dict) -> str:
    return json.dumps({"version": "1", "pip_version": "23.0.1", "installed": [
        {"metadata": {"name": n, "version": v}, "installer": "pip", "metadata_location": f"x/{n}-{v}.dist-info"}
        for n, v in packages.items()]})


def full_set() -> dict:
    return {**WANTED, **PREEXISTING}


def assert_not_evaluated(result: dict) -> None:
    for key in DIFF_KEYS + ("installed", "distribution_count"):
        assert result[key] == r.NOT_EVALUATED, key


# ── the tool failed: nothing may be said about the packages ─────────────────

def test_nonzero_exit_with_traceback_is_tool_failed():
    result = r.classify_inspect(2, "", WANTED, PREEXISTING)
    assert result["status"] == "tool_failed" and result["exit"] == 2
    assert_not_evaluated(result)


def test_nonzero_exit_is_tool_failed_even_if_stdout_holds_a_complete_report():
    result = r.classify_inspect(1, report(full_set()), WANTED, PREEXISTING)
    assert result["status"] == "tool_failed"
    assert_not_evaluated(result)


@pytest.mark.parametrize("code", [-9, -8])          # the runner's codes for a timeout and a failure to start
def test_timeout_and_execution_error_are_tool_failed(code):
    assert r.classify_inspect(code, "", WANTED, PREEXISTING)["status"] == "tool_failed"


def test_attempt4_never_reports_packages_as_missing():
    """The exact shape of attempt 4: exit 2, stdout cut off inside the first distribution."""
    truncated = '{\n  "version": "1",\n  "pip_version": "23.0.1",\n  "installed": [\n    {\n      "metadata": {\n'
    result = r.classify_inspect(2, truncated, WANTED, PREEXISTING)
    assert result["status"] == "tool_failed"
    assert result["missing"] == r.NOT_EVALUATED and result["missing"] != sorted(WANTED)


# ── exit 0, but the output cannot be used ───────────────────────────────────

@pytest.mark.parametrize("stdout", [
    "",                                                              # empty
    "   \n",                                                         # whitespace
    '{"version": "1", "installed": [{"metadata": {"name": "urllib3", ',   # truncated
    "ERROR: something went wrong",                                   # not JSON
    "[]",                                                            # JSON, wrong shape
    '{"version": "1", "pip_version": "23.0.1"}',                     # no installed list
    '{"installed": []}',                                             # empty installed list
    '{"installed": "urllib3"}',                                      # installed is not a list
    '{"installed": [{"metadata": {"name": "urllib3"}}]}',            # an entry without a version
    '{"installed": [{"installer": "pip"}]}',                         # an entry without metadata
])
def test_unusable_output_is_output_unparseable(stdout):
    result = r.classify_inspect(0, stdout, WANTED, PREEXISTING)
    assert result["status"] == "output_unparseable"
    assert result["parse_error"]
    assert_not_evaluated(result)


# ── parsed: match or a real mismatch ────────────────────────────────────────

def test_exact_set_with_unchanged_preexisting_is_match():
    result = r.classify_inspect(0, report(full_set()), WANTED, PREEXISTING)
    assert result["status"] == "match" and result["distribution_count"] == 14
    assert all(result[k] == [] for k in DIFF_KEYS)


def test_missing_distribution_is_mismatch_naming_exactly_that_one():
    packages = full_set()
    del packages["langdetect"]
    result = r.classify_inspect(0, report(packages), WANTED, PREEXISTING)
    assert result["status"] == "mismatch" and result["missing"] == ["langdetect"]
    assert result["wrong_version"] == [] and result["extra_beyond_pins_and_preexisting"] == []


def test_wrong_version_is_mismatch():
    result = r.classify_inspect(0, report({**full_set(), "requests": "2.31.0"}), WANTED, PREEXISTING)
    assert result["status"] == "mismatch" and result["wrong_version"] == ["requests"] and result["missing"] == []


def test_unexpected_extra_is_mismatch():
    result = r.classify_inspect(0, report({**full_set(), "wheel": "0.48.0"}), WANTED, PREEXISTING)
    assert result["status"] == "mismatch" and result["extra_beyond_pins_and_preexisting"] == ["wheel"]


def test_changed_preexisting_distribution_is_mismatch():
    result = r.classify_inspect(0, report({**full_set(), "pip": "26.0.1"}), WANTED, PREEXISTING)
    assert result["status"] == "mismatch" and result["preexisting_changed"] == ["pip"]


def test_removed_preexisting_distribution_is_mismatch():
    packages = full_set()
    del packages["setuptools"]
    result = r.classify_inspect(0, report(packages), WANTED, PREEXISTING)
    assert result["status"] == "mismatch" and result["preexisting_changed"] == ["setuptools"]


def test_names_are_normalised():
    packages = {**full_set()}
    packages["typing_extensions"] = packages.pop("typing-extensions")
    packages["Charset.Normalizer"] = packages.pop("charset-normalizer")
    wanted = {**WANTED}
    wanted["Deep_Translator"] = wanted.pop("deep-translator")
    assert r.classify_inspect(0, report(packages), wanted, PREEXISTING)["status"] == "match"
    assert r.norm("typing_extensions") == r.norm("Typing.Extensions") == "typing-extensions"


def test_legacy_egg_info_metadata_is_recorded_not_failed():
    data = json.loads(report(full_set()))
    for d in data["installed"]:
        if d["metadata"]["name"] == "langdetect":
            d["installer"] = None
            d["metadata_location"] = "x/langdetect-1.0.9-py3.10.egg-info"
    result = r.classify_inspect(0, json.dumps(data), WANTED, PREEXISTING)
    assert result["status"] == "match"
    assert result["installed"]["langdetect"]["metadata_kind"] == "egg-info"


# ── pip check ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code, status", [(0, "pass"), (1, "broken_requirements"), (2, "tool_failed"),
                                          (-9, "tool_failed"), (-8, "tool_failed"), (3, "tool_failed")])
def test_pip_check_exit_codes(code, status):
    assert r.classify_pip_check(code) == {"status": status, "exit": code}


# ── the gate: only a confirmed match passes (G7 unchanged) ──────────────────

@pytest.mark.parametrize("status", ["tool_failed", "output_unparseable", "mismatch", "", "unknown", None])
def test_no_status_other_than_match_passes(status):
    assert r.gate_verdict(status, [], []) != r.VERDICT_PASS


def test_match_passes_only_when_every_other_gate_is_clean():
    assert r.gate_verdict("match", [], []) == r.VERDICT_PASS
    assert r.gate_verdict("match", ["pip check (tool_failed)"], []) == r.VERDICT_TOOL_FAILED
    assert r.gate_verdict("match", [], ["pip check"]) == r.VERDICT_NOT_VERIFIED


def test_tool_failure_is_never_labelled_as_a_package_finding():
    assert r.gate_verdict("tool_failed", ["pip inspect (tool_failed)"], []) == r.VERDICT_TOOL_FAILED
    assert r.gate_verdict("mismatch", [], ["installed set differs from the pins"]) == r.VERDICT_NOT_VERIFIED
    assert r.VERDICT_TOOL_FAILED != r.VERDICT_NOT_VERIFIED


# ── the command and the evidence ────────────────────────────────────────────

def test_inspect_command_keeps_isolation_and_adds_utf8_mode():
    plan = {"base_interpreter": "base/python.exe"}
    bundled = r.inspect_command(plan, {"bundled_pip": True, "launcher": "env/Scripts/python.exe"})
    assert bundled == ["env/Scripts/python.exe", "-I", "-X", "utf8", "-m", "pip", "inspect",
                       "--disable-pip-version-check"]
    other = r.inspect_command(plan, {"bundled_pip": False, "launcher": "env/Scripts/python.exe"})
    assert other[:6] == ["base/python.exe", "-I", "-X", "utf8", "-m", "pip"] and "inspect" in other


def test_install_command_is_unchanged_for_mode_a():
    info = {"install_mode": "A-source-build", "launcher": "L", "find_links": ["WA", "WF"], "pins": "P",
            "bundled_pip_version": "23.0.1"}
    cmd, installer = r.install_command({"base_interpreter": "B"}, info)
    assert cmd == ["L", "-I", "-m", "pip", "install", "--isolated", "--no-index", "--find-links", "WA",
                   "--find-links", "WF", "--only-binary", ":all:", "--no-binary", "langdetect",
                   "--require-hashes", "-r", "P", "--no-cache-dir", "--disable-pip-version-check", "-vv"]
    assert installer == "pip 23.0.1 bundled with the venv"


def test_exclusive_evidence_is_not_overwritten(tmp_path):
    ev = r.Evidence(tmp_path / "attempt")
    ev.save("cell/gates/pip_inspect.stderr.txt", "the original traceback", exclusive=True)
    with pytest.raises(FileExistsError):
        ev.save("cell/gates/pip_inspect.stderr.txt", "a later step", exclusive=True)
    assert (tmp_path / "attempt" / "cell" / "gates" / "pip_inspect.stderr.txt").read_text() == "the original traceback"


def test_gate_run_writes_output_before_it_is_parsed(tmp_path, monkeypatch):
    ev = r.Evidence(tmp_path / "attempt")
    monkeypatch.setattr(r, "run", lambda cmd, env=None, cwd=None, timeout=0: (2, "partial {", "Traceback: boom", 0.1))
    code, out, err, record = r.gate_run(ev, "cell", "pip_inspect", ["python", "-I", "-m", "pip", "inspect"])
    gates = tmp_path / "attempt" / "cell" / "gates"
    assert (gates / "pip_inspect.stdout.txt").read_text() == "partial {"
    assert (gates / "pip_inspect.stderr.txt").read_text() == "Traceback: boom"
    saved = json.loads((gates / "pip_inspect.run.json").read_text())
    assert saved["exit_code"] == 2 and saved["interpreter"] == "python" and "pip inspect" in saved["command"]
    with pytest.raises(FileExistsError):                       # a second run of the same gate cannot replace it
        r.gate_run(ev, "cell", "pip_inspect", ["python"])


# ── import failures and completion status (after offline attempt 5) ─────────
# Attempt 5: `import fie` raised ModuleNotFoundError, no scan ran, and the runner printed
# "RESULT: COMPLETED". The probe recorded the exception class but not the missing module.

import probe  # noqa: E402

BLOCKED_IMPORT = {"import_ok": False, "import_completed": False, "exception": "ModuleNotFoundError",
                  "missing_module": "starlette", "exception_message": "No module named 'starlette'",
                  "scan_records": 0}
GOOD_IMPORT = {"import_ok": True, "import_completed": True}


def raised_import_error():
    try:
        import e51_module_that_does_not_exist  # noqa: F401
    except ModuleNotFoundError as exc:
        return exc
    raise AssertionError("the import unexpectedly succeeded")


def test_missing_module_is_recorded_as_import_failure_with_its_name():
    record = probe.import_failure_record(raised_import_error(), 0.25)
    assert record["import_ok"] is False and record["import_completed"] is False
    assert record["exception"] == "ModuleNotFoundError"
    assert record["missing_module"] == "e51_module_that_does_not_exist"
    assert "e51_module_that_does_not_exist" in record["exception_message"]
    assert record["scan_records"] == 0


def test_other_import_time_exception_has_no_missing_module():
    try:
        raise RuntimeError("boom at import")
    except RuntimeError as exc:
        record = probe.import_failure_record(exc, 0.1)
    assert record["exception"] == "RuntimeError" and record["missing_module"] is None
    assert record["import_completed"] is False and record["scan_records"] == 0


def test_exception_message_is_sanitized(monkeypatch):
    home = str(Path.home())
    monkeypatch.setattr(probe, "_env_root", home + "\\scratch\\env\\cell")
    message = probe._sanitize(f"cannot open {home}\\scratch\\env\\cell\\Lib\\x.py or {home}\\secret.txt")
    assert home not in message and "<ENV>" in message and "<HOME>" in message
    assert len(probe._sanitize("x" * 5000)) == 300


def test_import_blocked_run_is_not_a_completed_scan():
    run = r.classify_probe_run(3, BLOCKED_IMPORT, None)
    assert run["status"] == "import_blocked" and run["scan_records"] == 0
    assert run["import_completed"] is False and run["missing_module"] == "starlette"


def test_exit_zero_with_no_records_is_not_a_completed_scan():
    assert r.classify_probe_run(0, GOOD_IMPORT, 0)["status"] == "no_scan_records"
    assert r.classify_probe_run(0, GOOD_IMPORT, None)["status"] == "no_scan_records"
    assert r.classify_probe_run(0, None, 48)["status"] == "no_scan_records"       # no proof the import completed


def test_successful_import_with_records_is_scan_completed():
    run = r.classify_probe_run(0, GOOD_IMPORT, 1848)
    assert run == {"status": "scan_completed", "exit": 0, "scan_records": 1848, "import_completed": True,
                   "exception": None, "missing_module": None, "exception_message": None}


def test_other_probe_exit_is_probe_failed():
    assert r.classify_probe_run(2, GOOD_IMPORT, None)["status"] == "probe_failed"
    assert r.classify_probe_run(-9, None, None)["status"] == "probe_failed"


def test_attempt5_cell_is_import_blocked_and_never_completed():
    """Exactly attempt 5: S-run1 and PC-run1 exit 3, no repeat runs, zero records."""
    runs = {"S-run1": r.classify_probe_run(3, BLOCKED_IMPORT, None),
            "PC-run1": r.classify_probe_run(3, BLOCKED_IMPORT, None)}
    cell = r.classify_cell(runs)
    assert cell["outcome"] == r.OUTCOME_IMPORT_BLOCKED == "IMPORT_BLOCKED_NO_SCAN_COMPLETED"
    assert cell["scan_records_total"] == 0 and cell["runs_completed"] == []
    assert cell["import_error"]["missing_module"] == "starlette"
    result = r.overall_result(None, {"modeA-pypi-base-r3": cell["outcome"]}, ["modeA-pypi-base-r3"])
    assert not result.startswith("COMPLETED") and "IMPORT_BLOCKED_NO_SCAN_COMPLETED" in result


@pytest.mark.parametrize("runs", [
    {},                                                                            # nothing ran
    {"S-run1": ("no_scan_records", 0)},
    {"S-run1": ("probe_failed", 0), "PC-run1": ("probe_failed", 0)},
    {"S-run1": ("import_blocked", 0), "S-run2": ("import_blocked", 0),
     "PC-run1": ("import_blocked", 0), "PC-run2": ("import_blocked", 0)},
])
def test_zero_scan_records_can_never_be_labelled_completed(runs):
    results = {t: {"status": s, "scan_records": n, "exception": None, "missing_module": None,
                   "exception_message": None} for t, (s, n) in runs.items()}
    cell = r.classify_cell(results)
    assert cell["scan_records_total"] == 0 and cell["outcome"] != r.OUTCOME_SCANS
    assert not r.overall_result(None, {"c": cell["outcome"]}, ["c"]).startswith("COMPLETED")


def test_all_four_runs_completed_is_scans_completed():
    done = r.classify_probe_run(0, GOOD_IMPORT, 48)
    big = r.classify_probe_run(0, GOOD_IMPORT, 1848)
    cell = r.classify_cell({"S-run1": done, "S-run2": done, "PC-run1": big, "PC-run2": big})
    assert cell["outcome"] == r.OUTCOME_SCANS and cell["scan_records_total"] == 2 * 48 + 2 * 1848
    assert cell["import_error"] is None
    assert r.overall_result(None, {"c": cell["outcome"]}, ["c"]).startswith("COMPLETED")


def test_some_runs_completed_is_partial_not_completed():
    done = r.classify_probe_run(0, GOOD_IMPORT, 48)
    cell = r.classify_cell({"S-run1": done, "S-run2": done, "PC-run1": r.classify_probe_run(2, GOOD_IMPORT, None)})
    assert cell["outcome"] == r.OUTCOME_PARTIAL
    assert not r.overall_result(None, {"c": cell["outcome"]}, ["c"]).startswith("COMPLETED")


def test_overall_result_needs_every_planned_cell():
    assert not r.overall_result(None, {"a": r.OUTCOME_SCANS}, ["a", "b"]).startswith("COMPLETED")
    assert "b: NOT_RUN" in r.overall_result(None, {"a": r.OUTCOME_SCANS}, ["a", "b"])
    assert not r.overall_result(None, {}, []).startswith("COMPLETED")
    mixed = r.overall_result(None, {"a": r.OUTCOME_SCANS, "b": r.OUTCOME_IMPORT_BLOCKED}, ["a", "b"])
    assert not mixed.startswith("COMPLETED")
    assert r.overall_result("the machine does not look offline", {"a": r.OUTCOME_SCANS}, ["a"]).startswith("STOPPED")


def test_tool_failure_stays_distinct_from_package_mismatch_and_from_import_block():
    tool = r.classify_inspect(2, "", WANTED, PREEXISTING)
    packages = full_set()
    del packages["requests"]
    real = r.classify_inspect(0, report(packages), WANTED, PREEXISTING)
    assert (tool["status"], real["status"]) == ("tool_failed", "mismatch")
    assert tool["missing"] == r.NOT_EVALUATED and real["missing"] == ["requests"]
    assert r.gate_verdict(tool["status"], ["pip inspect (tool_failed)"], []) == r.VERDICT_TOOL_FAILED
    assert r.gate_verdict(real["status"], [], ["installed set differs from the pins"]) == r.VERDICT_NOT_VERIFIED
    assert r.OUTCOME_IMPORT_BLOCKED not in (r.VERDICT_TOOL_FAILED, r.VERDICT_NOT_VERIFIED, r.VERDICT_PASS)


def test_cell_outcome_and_blocked_files_are_not_overwritten(tmp_path):
    ev = r.Evidence(tmp_path / "attempt")
    for name in ("cell/outcome.json", "cell/blocked.json", "cell/runs.json", "cell/S-run1/run.json"):
        ev.save(name, {"first": True}, exclusive=True)
        with pytest.raises(FileExistsError):
            ev.save(name, {"first": False}, exclusive=True)
        assert json.loads((tmp_path / "attempt" / name).read_text()) == {"first": True}
