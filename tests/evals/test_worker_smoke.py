"""Worker and orchestrator: lifecycle, hermetic accounting, secrets, resume, output location."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evals import canonical, orchestrator
from _helpers import REPO_ROOT, child_env, child_json

PLANTED = {
    "GROQ_API_KEY": "gsk_PLANTED_SECRET_ALPHA", "MONGODB_URI": "mongodb+srv://PLANTED_SECRET_BETA",
    "JWT_SECRET_KEY": "PLANTED_SECRET_GAMMA", "PYPI_TOKEN": "pypi-PLANTED_SECRET_DELTA",
    "FIE_UNCERTAIN_ALLOW": "1", "FIE_API_KEY": "fie-PLANTED_SECRET_EPSILON",
}


def evals_cli(args: list[str], tmp_path: Path, extra_env: dict | None = None,
              timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-m", "evals", *args],
        cwd=str(REPO_ROOT), env=child_env(tmp_path, extra_env),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """One small real run, shared by the tests below. The caller's shell holds planted secrets."""
    from _helpers import models_present
    if not models_present():
        pytest.skip("model artifacts not present — run: python scripts/download_models.py --strict")
    tmp = tmp_path_factory.mktemp("smoke")
    out = tmp / "runs"
    proc = evals_cli(["run", "--suites", "std.xstest", "--limit", "3", "--out", str(out)], tmp, PLANTED)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    return run_dirs[0], proc, tmp


def test_smoke_run_writes_records_in_order(smoke_run):
    run_dir, _, _ = smoke_run
    recs = canonical.read_jsonl(run_dir / "records" / "std.xstest.jsonl")
    assert [r["idx"] for r in recs] == list(range(6))
    assert [r["dataset"] for r in recs] == ["xstest_safe"] * 3 + ["xstest_unsafe"] * 3
    assert [r["source_idx"] for r in recs] == [0, 1, 2, 0, 1, 2]
    assert [r["expected"] for r in recs] == ["benign"] * 3 + ["attack"] * 3
    for r in recs:
        assert set(r) == {"suite", "idx", "dataset", "source_idx", "variant", "input_sha256",
                          "expected", "flagged", "zone", "type", "conf", "layers_fired",
                          "layer_scores", "degraded", "status"}
        assert r["status"] == "ok" and r["variant"] == "base" and len(r["input_sha256"]) == 64
        assert r["zone"] in ("allow", "uncertain_block", "clear_block")
        assert (r["zone"] == "allow") == (not r["flagged"])
        assert r["layers_fired"] == sorted(r["layers_fired"]) and len(r["layer_scores"]) == 12
        assert "prompt" not in r and "text" not in r
    raw = (run_dir / "records" / "std.xstest.jsonl").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n") and raw.count(b"\n") == 6


def test_smoke_run_is_marked_non_canonical(smoke_run):
    run_dir, _, _ = smoke_run
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["canonical"] is False
    reasons = " ".join(meta["not_canonical_because"])
    assert "--limit" in reasons and "explicit suite selection" in reasons


def test_worker_guard_accounting_is_complete(smoke_run):
    run_dir, _, _ = smoke_run
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    worker = meta["workers"]["sdk-offline-failsecure"]
    guard = worker["guard"]
    assert worker["guard_ok"] is True and worker["guard_problems"] == []
    assert guard["installed"] and guard["selftest"]["passed"]
    assert guard["violations"] == 0 and guard["canary_events"] == 5
    assert guard["audit_events_seen"] > 0
    assert worker["models_reverified_after_run"] is True
    assert worker["subject_state"]["server_config_attached"] is False
    assert worker["pickle_classes"], "unpickled classes were not recorded"
    assert meta["orchestrator_guard"]["violations"] == 0
    events = [json.loads(l) for l in
              (run_dir / "work" / "sdk-offline-failsecure" / "guard-events.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    assert events and all(e["canary"] for e in events)


def test_no_planted_secret_reaches_any_artifact_or_log(smoke_run):
    run_dir, proc, _ = smoke_run
    needles = [v for v in PLANTED.values() if "PLANTED" in v] + ["PLANTED_SECRET"]
    assert not any(n in proc.stdout + proc.stderr for n in needles)
    scanned = 0
    for path in run_dir.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            scanned += 1
            for needle in needles:
                assert needle.encode() not in data, f"{needle} found in {path.relative_to(run_dir)}"
    assert scanned > 8
    fp = canonical.read_json(run_dir / "fingerprint.json")
    present = fp["configuration"]["sdk-offline-failsecure"]["env_present"]
    assert present["GROQ_API_KEY"] is False, "the worker must not even see the variable"
    assert fp["configuration"]["sdk-offline-failsecure"]["env"]["FIE_UNCERTAIN_ALLOW"] is None


def test_deterministic_files_hold_no_run_metadata(smoke_run):
    run_dir, _, tmp = smoke_run
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    # The run id, its UTC stamp, absolute paths and the machine's names must not
    # appear. A bare calendar date may: registry text cites when counts were approved.
    forbidden = [meta["run_id"], meta["run_id"].split("_")[0], meta["started_utc"],
                 str(REPO_ROOT), str(tmp),
                 os.environ.get("COMPUTERNAME") or "\0never\0", os.environ.get("USERNAME") or "\0never\0"]
    clock = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")           # any date-with-time stamp
    files = orchestrator.deterministic_files(run_dir)
    assert "fingerprint.json" in files and "records/std.xstest.jsonl" in files
    for rel in files:
        text = (run_dir / rel).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token!r} found in deterministic file {rel}"
            assert token.replace("\\", "\\\\") not in text
        assert not clock.search(text), f"a timestamp appears in deterministic file {rel}"
        for word in ("duration", "wall_s", '"pid"', "elapsed"):
            assert word not in text, f"{word!r} found in deterministic file {rel}"
    manifest = orchestrator.read_manifest(run_dir / "MANIFEST.sha256")
    assert sorted(manifest) == files
    assert all(manifest[rel] == canonical.sha256_bytes(canonical.read_lf(run_dir / rel)) for rel in files)


def test_fingerprint_has_three_blocks_four_keys_and_all_hashes(smoke_run):
    run_dir, _, _ = smoke_run
    fp = canonical.read_json(run_dir / "fingerprint.json")
    assert set(fp) == {"schema_version", "keys", "identity", "configuration", "environment"}
    assert set(fp["keys"]) == {"dataset_key", "config_key", "subject_key", "env_key"}
    ident = fp["identity"]
    models = ident["models"]["sdk-offline-failsecure"]
    assert set(models) == {"pair_classifier", "pair_meta", "meta_classifier",
                           "meta_classifier_meta", "encoder", "tokenizer"}
    assert all(len(m["sha256"]) == 64 for m in models.values())
    assert len(ident["subject"]["fie_tree_sha256"]) == 64
    assert {"xstest_safe", "xstest_unsafe"} <= set(ident["datasets"])
    assert all(len(d["content_sha256"]) == 64 for d in ident["datasets"].values())
    assert ident["harness"]["version"] and ident["harness"]["schema_version"] == 1
    assert ident["declared"]["sdk-offline-failsecure"]["pair"]["declared_version"] == "v6.3b"
    cfg = fp["configuration"]["sdk-offline-failsecure"]
    assert cfg["profile"]["id"] == "sdk-offline-failsecure"
    assert "canonical reproducibility/evaluation profile" in cfg["profile"]["title"]
    assert cfg["langdetect_seed"] == 0 and cfg["env"]["PYTHONHASHSEED"] == "0"
    assert cfg["subject"]["tiebreaker"] == "disabled"
    assert fp["environment"]["python"]["version"] and fp["environment"]["packages"]["numpy"]


def test_resume_refuses_a_changed_plan(smoke_run):
    run_dir, _, tmp = smoke_run
    proc = evals_cli(["run", "--suites", "std.xstest", "--limit", "4", "--resume", str(run_dir)], tmp)
    assert proc.returncode == 8, proc.stdout + proc.stderr
    assert "cannot resume" in proc.stderr


def test_pin_refuses_a_non_canonical_run(smoke_run):
    """A limited, hand-picked run must never become a baseline."""
    run_dir, _, tmp = smoke_run
    baselines = REPO_ROOT / "evals" / "baselines"
    before = sorted(p.name for p in baselines.iterdir()) if baselines.exists() else []
    proc = evals_cli(["pin", str(run_dir), "--canonical"], tmp)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "refusing to pin" in proc.stderr
    assert "the run is not canonical" in proc.stderr
    assert "--determinism-proof is required" in proc.stderr
    after = sorted(p.name for p in baselines.iterdir()) if baselines.exists() else []
    assert after == before, "pin created something despite refusing"


def test_pin_needs_exactly_one_kind(smoke_run):
    run_dir, _, tmp = smoke_run
    assert evals_cli(["pin", str(run_dir)], tmp).returncode == 1
    assert evals_cli(["pin", str(run_dir), "--canonical", "--reference"], tmp).returncode == 1


def test_output_inside_the_repository_is_refused(tmp_path):
    proc = evals_cli(["run", "--suites", "std.xstest", "--limit", "1",
                      "--out", str(REPO_ROOT / "fie" / "runs")], tmp_path)
    assert proc.returncode == 1
    assert "refusing to write run output inside the repository" in proc.stderr
    assert not (REPO_ROOT / "fie" / "runs").exists()


def test_unknown_suite_and_bad_model_hash_are_usage_errors(tmp_path):
    assert evals_cli(["run", "--suites", "no.such.suite"], tmp_path).returncode == 1
    assert evals_cli(["run", "--model-sha256", "pair_classifier=short"], tmp_path).returncode == 1


def test_a_wrong_model_hash_aborts_with_exit_3_before_any_worker(tmp_path, need_models):
    proc = evals_cli(["run", "--suites", "std.xstest", "--limit", "1", "--out", str(tmp_path / "r"),
                      "--model-sha256", "pair_classifier=" + "0" * 64], tmp_path)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "model integrity check failed" in proc.stderr
    assert not (tmp_path / "r").exists(), "a run directory was created despite the integrity failure"


# ── the run fails on the guard's RECORD, not on the exception ────────────────

def test_a_swallowed_network_attempt_stops_the_suite_and_leaves_a_partial_file(tmp_path):
    """
    The product catches exceptions around its network calls. Reproduce that with
    a stand-in subject that swallows the guard's exception: the suite must still
    stop, with exit code 5, leaving only a .partial file.
    """
    run_dir = tmp_path / "run"
    code = f'''
import json, socket, time
from pathlib import Path
from evals import hermetic
hermetic.install()
from evals import canonical, worker

class Subject:
    def set_translation(self, mode): pass
    def clear_caches(self): pass
    def scan(self, text):
        if "second" in text:
            try:
                socket.create_connection(("192.0.2.1", 9), timeout=1)   # denied by the guard
            except OSError:
                pass                                                     # swallowed, as fie does
        return ({{"flagged": False, "zone": "allow", "type": None, "conf": 0.0, "layers_fired": [],
                 "layer_scores": {{}}, "degraded": [], "status": "ok"}}, {{}})

items = [{{"idx": i, "dataset": "d", "source_idx": i, "variant": "base", "expected": "benign",
          "text": t, "input_sha256": "0" * 64, "runtime": {{}}}}
         for i, t in enumerate(["first", "second", "third"])]
run_dir = Path({str(run_dir)!r})
try:
    worker._run_scan_suite({{"id": "t.suite"}}, items, run_dir, Subject(), canonical,
                           worker.check_guard, time, "unavailable")
    result = {{"stopped": False}}
except worker._Stop as stop:
    result = {{"stopped": True, "code": stop.code, "status": stop.status, "detail": stop.detail}}
print(json.dumps(result))
'''
    out = child_json(code, tmp_path)
    assert out["stopped"] is True and out["code"] == 5 and out["status"] == "hermetic_violation"
    assert "192.0.2.1" in out["detail"]
    assert not (run_dir / "records" / "t.suite.jsonl").exists(), "an invalid suite was finalized"
    partial = run_dir / "records" / "t.suite.jsonl.partial"
    assert partial.exists() and len(partial.read_bytes().splitlines()) == 2


# ── the orchestrator's verdict on a worker's hermetic proof ──────────────────

def _guard_meta(**over):
    guard = {"installed": True, "selftest": {"passed": True}, "violations": 0,
             "audit_events_seen": 1234}
    guard.update(over)
    return {"guard": guard}


def _log(tmp_path: Path, lines: list[dict], profile: str = "p") -> Path:
    work = tmp_path / "work" / profile
    work.mkdir(parents=True, exist_ok=True)
    (work / "guard-events.jsonl").write_text(
        "".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    return tmp_path


def test_guard_verdict_accepts_a_clean_worker(tmp_path):
    run_dir = _log(tmp_path, [{"canary": True, "event": "socket.__new__"}])
    assert orchestrator.guard_verdict(run_dir, "p", _guard_meta()) == (True, [])


@pytest.mark.parametrize("meta,fragment", [
    (None, "no guard summary"),
    ({}, "no guard summary"),
    (_guard_meta(installed=False), "not installed"),
    (_guard_meta(selftest={"passed": False}), "self-test did not pass"),
    (_guard_meta(violations=2), "2 denied operation(s)"),
    (_guard_meta(audit_events_seen=0), "never ran"),
])
def test_guard_verdict_rejects_an_unproven_worker(tmp_path, meta, fragment):
    run_dir = _log(tmp_path, [{"canary": True}])
    ok, problems = orchestrator.guard_verdict(run_dir, "p", meta)
    assert ok is False and any(fragment in p for p in problems), problems


def test_guard_verdict_rejects_a_non_canary_event_in_the_log(tmp_path):
    run_dir = _log(tmp_path, [{"canary": True}, {"canary": False, "event": "socket.connect"}])
    ok, problems = orchestrator.guard_verdict(run_dir, "p", _guard_meta())
    assert ok is False and any("non-canary" in p for p in problems)


def test_guard_verdict_rejects_a_missing_event_log(tmp_path):
    ok, problems = orchestrator.guard_verdict(tmp_path, "p", _guard_meta())
    assert ok is False and any("event log is missing" in p for p in problems)
