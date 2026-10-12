"""
The orchestrator: preconditions, worker lifecycle, final assembly.

It NEVER imports `fie`. It checks datasets and model files, records the subject's
identity, builds a sanitized environment from an allowlist, and starts one worker
per profile in a fresh interpreter. When the workers finish it validates their
guard summaries and assembles the deterministic artifacts.

Run directory (under evals/runs/, which is git-ignored):

    DETERMINISTIC RESULT ARTIFACT          RUN METADATA
    fingerprint.json                       run.json
    records/<suite>.jsonl                  timing/<suite>.jsonl
    evidence/<suite>.jsonl                 nondeterministic/<suite>.jsonl
    summary.json                           latency.json, RUN_NOTES.md
    REPORT.md                              work/, logs/, status/
    MANIFEST.sha256

No timestamp, duration, run id, host name, absolute path or process id appears in
the left column.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evals import SCHEMA_VERSION, canonical, datasets, fingerprint, hermetic, integrity, transforms
from evals.datasets import DatasetIntegrityError
from evals.integrity import ModelIntegrityError

EXIT_OK, EXIT_USAGE, EXIT_INCOMPLETE = 0, 1, 2
EXIT_MODEL, EXIT_DATASET, EXIT_HERMETIC = 3, 4, 5
EXIT_BASELINE_MISMATCH, EXIT_NONDETERMINISTIC, EXIT_FINGERPRINT = 6, 7, 8

DETERMINISTIC_DIRS = ("records", "evidence")
DETERMINISTIC_FILES = ("fingerprint.json", "summary.json", "REPORT.md")
# Files the determinism criterion applies to. Evidence is hashed and compared,
# but a difference there is reported, not fatal (WP-001 decision OD-5).
GATED_PREFIXES = ("records/", "fingerprint.json", "summary.json")

PLAN_KINDS = {
    "baseline":      "every suite, each under its own profile",
    "deterministic": "every suite that produces deterministic records (no latency, no stability)",
    "standard":      "the standard benchmark suites only",
    "reference":     "the standard benchmark suites under a reference profile",
    "custom":        "an explicit subset; never canonical",
}


class RunError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def say(message: str) -> None:
    print(message, flush=True)


# ── guard ────────────────────────────────────────────────────────────────────

_GIT: str | None = None


def arm_guard(event_log: Path | None = None) -> dict:
    """
    Arm the orchestrator's guard. It may start exactly two programs: this
    interpreter (the worker) and git. Everything else, and all networking, is denied.
    """
    global _GIT
    _GIT = shutil.which("git")
    allow = [sys.executable] + ([_GIT] if _GIT else [])
    hermetic.install(mode="orchestrator", event_log=event_log, allow_exec=tuple(allow))
    result = hermetic.selftest()
    if not result["passed"]:
        raise RunError(EXIT_HERMETIC, f"orchestrator guard self-test failed: {result['checks']}")
    return result


def _git(repo: Path, *args: str) -> str | None:
    if not _GIT:
        return None
    try:
        proc = subprocess.run([_GIT, *args], executable=_GIT, cwd=str(repo), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_identity(repo: Path) -> dict:
    """
    Code revision and cleanliness of the MEASURED paths.

    A run is traceable when the subject's files are exactly a commit's files.
    Uncommitted changes elsewhere (for example the harness itself, whose content
    is pinned separately by its own tree hash) are recorded but do not make the
    subject untraceable.
    """
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty_subject": None, "dirty_other": None, "subject_paths_changed": []}
    status = _git(repo, "status", "--porcelain", "--untracked-files=all") or ""
    subject_changed, other_changed = [], 0
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"').split(" -> ")[-1]
        if any(path == s or path.startswith(s.rstrip("/") + "/") for s in fingerprint.SUBJECT_PATHS):
            subject_changed.append(path)
        else:
            other_changed += 1
    return {
        "commit": commit.strip(),
        "dirty_subject": bool(subject_changed),
        "dirty_other": bool(other_changed),
        "subject_paths_changed": sorted(subject_changed)[:50],
    }


# ── planning ─────────────────────────────────────────────────────────────────

def select_suites(suites_reg: dict, kind: str, primary: str, suite_ids: list[str] | None) -> dict:
    """Return {profile_id: [suite, ...]} in registry order."""
    chosen = []
    for suite in suites_reg["suites"]:
        if suite_ids is not None:
            if suite["id"] in suite_ids:
                chosen.append(suite)
        elif kind == "baseline":
            chosen.append(suite)
        elif kind == "deterministic":
            if suite["kind"] == "scan":
                chosen.append(suite)
        elif kind in ("standard", "reference"):
            if suite["group"] == "standard":
                chosen.append(suite)
    if suite_ids is not None:
        unknown = sorted(set(suite_ids) - {s["id"] for s in suites_reg["suites"]})
        if unknown:
            raise RunError(EXIT_USAGE, f"unknown suite id(s): {unknown}")
    by_profile: dict[str, list] = {}
    for suite in chosen:
        profile = primary if suite["profile"] == transforms.PRIMARY else suite["profile"]
        by_profile.setdefault(profile, []).append(suite)
    return by_profile


# ── execution ────────────────────────────────────────────────────────────────

def execute(kind: str = "baseline", primary: str | None = None, suite_ids: list[str] | None = None,
            out: str | None = None, limit: int | None = None, resume: str | None = None,
            hash_seed: str = "0", model_overrides: dict | None = None,
            quiet: bool = False) -> tuple[int, Path]:
    """Run a plan. Returns (exit code, run directory)."""
    repo = repo_root()
    started_wall = time.time()
    t0 = time.perf_counter()

    try:
        profiles_reg = transforms.load_profiles(repo)
        suites_reg = transforms.load_suites(repo)
    except transforms.RegistryError as exc:
        raise RunError(EXIT_USAGE, str(exc))
    primary = primary or profiles_reg["canonical_profile"]
    if primary not in profiles_reg["profiles"]:
        raise RunError(EXIT_USAGE, f"unknown profile: {primary}")
    if suite_ids is not None:
        kind = "custom"
    by_profile = select_suites(suites_reg, kind, primary, suite_ids)
    if not by_profile:
        raise RunError(EXIT_USAGE, "the plan selects no suites")

    # Datasets and fixtures (exit 4) -----------------------------------------
    try:
        ds_registry = datasets.load_registry(repo)
        fixtures = transforms.load_fixtures(repo, suites_reg)
        needed: list[str] = []
        for suites in by_profile.values():
            for suite in suites:
                for ds_id in transforms.suite_datasets(suite):
                    if ds_id not in needed:
                        needed.append(ds_id)
        ds_identity = datasets.identity_block(repo, needed, ds_registry)
    except DatasetIntegrityError as exc:
        raise RunError(EXIT_DATASET, str(exc))

    # Models (exit 3) --------------------------------------------------------
    manifest = integrity.load_manifest(repo)
    model_tables: dict[str, dict] = {}
    for profile_id in by_profile:
        prof = profiles_reg["profiles"][profile_id]
        model_tables[profile_id] = integrity.verify_roles(
            prof["models"], repo, manifest, required=prof["required_roles"],
            overrides=(model_overrides or {}) if profile_id == primary else {})

    subject = fingerprint.subject_identity(repo)
    git = git_identity(repo)
    harness = fingerprint.harness_identity(repo)

    plan_doc = {
        "kind": kind, "primary_profile": primary, "limit": limit, "hash_seed": hash_seed,
        "suite_ids": [s["id"] for suites in by_profile.values() for s in suites],
        "explicit_suite_selection": suite_ids is not None,
        "model_overrides": bool(model_overrides),
        "subject": subject,
        "model_hashes": {p: {r: v["sha256"] for r, v in t.items()} for p, t in model_tables.items()},
        "dataset_hashes": {k: v["content_sha256"] for k, v in ds_identity.items()},
        "harness_tree_sha256": harness["tree_sha256"],
    }

    # Run directory ----------------------------------------------------------
    if resume:
        run_dir = Path(resume).resolve()
        try:
            stored = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RunError(EXIT_USAGE, f"cannot resume {run_dir}: {exc}")
        if stored != plan_doc:
            diff = sorted(k for k in set(stored) | set(plan_doc) if stored.get(k) != plan_doc.get(k))
            raise RunError(EXIT_FINGERPRINT,
                           f"cannot resume: the plan or the subject changed since the run started "
                           f"(differs in: {diff})")
    else:
        stamp = datetime.fromtimestamp(started_wall, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}_{subject['fie_tree_sha256'][:8]}_{canonical.hash_obj(plan_doc)[:8]}"
        base = Path(out).resolve() if out else repo / "evals" / "runs"
        _check_output_location(base, repo)
        run_dir = base / name
        run_dir.mkdir(parents=True, exist_ok=False)
        canonical.write_bytes(run_dir / "plan.json",
                              (json.dumps(plan_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    (run_dir / "logs").mkdir(exist_ok=True)

    if not quiet:
        say(f"evals: plan '{kind}' | primary profile '{primary}' | run directory {run_dir}")

    # Workers ----------------------------------------------------------------
    worker_results: dict[str, dict] = {}
    worst = EXIT_OK
    for profile_id, suites in by_profile.items():
        prof = profiles_reg["profiles"][profile_id]
        work = run_dir / "work" / profile_id
        work.mkdir(parents=True, exist_ok=True)
        scratch = work / "scratch"
        worker_plan = {
            "repo_root": str(repo), "run_dir": str(run_dir), "profile_id": profile_id,
            "profile": prof, "suites": suites, "limit": limit,
            "model_expect": plan_doc["model_hashes"][profile_id],
            "model_overrides": (model_overrides or {}) if profile_id == primary else {},
            "dataset_expect": plan_doc["dataset_hashes"],
        }
        plan_path = work / "plan.json"
        canonical.write_bytes(plan_path, (json.dumps(worker_plan, sort_keys=True) + "\n").encode("utf-8"))
        guard_log = work / "guard-events.jsonl"
        profile_env = dict(prof["env"])
        profile_env["FIE_FEEDBACK_PATH"] = str(scratch / "flagged_events.jsonl")
        env = hermetic.sanitized_env(os.environ, profile_env, scratch, hash_seed=hash_seed)

        if not quiet:
            n_items = len(suites)
            say(f"evals: starting worker '{profile_id}' ({n_items} suite(s))")
        tw = time.perf_counter()
        log_path = run_dir / "logs" / f"worker-{profile_id}.log"
        with open(log_path, "ab") as log:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-B", "-m", "evals.worker", str(plan_path), str(guard_log)],
                executable=sys.executable, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT)
        worker_results[profile_id] = {"exit_code": proc.returncode,
                                      "wall_s": round(time.perf_counter() - tw, 2)}
        worst = max(worst, proc.returncode) if proc.returncode in (0, 2) else proc.returncode
        if not quiet:
            say(f"evals: worker '{profile_id}' exited {proc.returncode} "
                f"in {worker_results[profile_id]['wall_s']} s")
        if proc.returncode not in (EXIT_OK, EXIT_INCOMPLETE):
            _report_worker_failure(run_dir, profile_id, proc.returncode)
            break
        shutil.rmtree(scratch, ignore_errors=True)        # holds only redirected side effects

    code, run_meta = finalize(run_dir, repo, profiles_reg, suites_reg, by_profile, plan_doc,
                              model_tables, ds_identity, fixtures, subject, git, harness,
                              worker_results, worst, started_wall, time.perf_counter() - t0, manifest)
    if not quiet:
        say(f"evals: finished with exit code {code} | canonical={run_meta['canonical']}")
    return code, run_dir


def _check_output_location(base: Path, repo: Path) -> None:
    """Output goes under evals/runs/ or outside the repository — never over project files."""
    repo = repo.resolve()
    inside = repo == base or repo in base.parents
    allowed = repo / "evals" / "runs"
    if inside and not (base == allowed or allowed in base.parents):
        raise RunError(EXIT_USAGE,
                       f"refusing to write run output inside the repository at {base}; "
                       f"use evals/runs/ or a directory outside the repository")


def _report_worker_failure(run_dir: Path, profile_id: str, code: int) -> None:
    meta_path = run_dir / "work" / profile_id / "worker.json"
    detail = ""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        detail = f"{meta.get('status')}: {meta.get('detail')}"
    else:
        log = run_dir / "logs" / f"worker-{profile_id}.log"
        if log.exists():
            detail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
    say(f"evals: worker '{profile_id}' FAILED (exit {code})\n{detail}")


# ── final assembly ───────────────────────────────────────────────────────────

def load_worker(run_dir: Path, profile_id: str) -> tuple[dict | None, dict | None]:
    work = run_dir / "work" / profile_id
    meta = blocks = None
    if (work / "worker.json").exists():
        meta = json.loads((work / "worker.json").read_text(encoding="utf-8"))
    if (work / "blocks.json").exists():
        blocks = canonical.read_json(work / "blocks.json")
    return meta, blocks


def guard_verdict(run_dir: Path, profile_id: str, meta: dict | None) -> tuple[bool, list[str]]:
    """
    A worker's hermetic proof. It holds only if the guard was armed, its self-test
    passed, it recorded no non-canary event, and the audit hook demonstrably ran.
    """
    problems = []
    if meta is None or "guard" not in meta:
        return False, ["the worker left no guard summary"]
    guard = meta["guard"]
    if not guard.get("installed"):
        problems.append("guard was not installed")
    if not (guard.get("selftest") or {}).get("passed"):
        problems.append("guard self-test did not pass")
    if guard.get("violations", 1) != 0:
        problems.append(f"{guard.get('violations')} denied operation(s) recorded")
    if not guard.get("audit_events_seen"):
        problems.append("the audit hook saw no events: it never ran")
    log = run_dir / "work" / profile_id / "guard-events.jsonl"
    if log.exists():
        real = [ln for ln in log.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not json.loads(ln).get("canary")]
        if real:
            problems.append(f"{len(real)} non-canary event(s) in the guard event log")
    else:
        problems.append("guard event log is missing")
    return not problems, problems


def finalize(run_dir: Path, repo: Path, profiles_reg: dict, suites_reg: dict, by_profile: dict,
             plan_doc: dict, model_tables: dict, ds_identity: dict, fixtures, subject: dict,
             git: dict, harness: dict, worker_results: dict, worst: int, started_wall: float,
             duration_s: float, manifest: dict) -> tuple[int, dict]:
    kind, primary = plan_doc["kind"], plan_doc["primary_profile"]
    reasons: list[str] = []          # why the run is not canonical, if it is not
    code = worst

    workers_meta: dict[str, dict] = {}
    configuration: dict[str, dict] = {}
    environment: dict | None = None
    hermetic_ok = True
    for profile_id in by_profile:
        meta, blocks = load_worker(run_dir, profile_id)
        ok, problems = guard_verdict(run_dir, profile_id, meta)
        if profile_id not in worker_results:
            problems = ["worker was not started"]
            ok = False
        workers_meta[profile_id] = {
            "exit_code": worker_results.get(profile_id, {}).get("exit_code"),
            "wall_s": worker_results.get(profile_id, {}).get("wall_s"),
            "status": (meta or {}).get("status"), "detail": (meta or {}).get("detail"),
            "guard_ok": ok, "guard_problems": problems,
            "guard": (meta or {}).get("guard"), "suites": (meta or {}).get("suites", {}),
            "timings": (meta or {}).get("timings", {}),
            "pickle_classes": (meta or {}).get("pickle_classes", []),
            "subject_state": (meta or {}).get("subject_state"),
            "blocked_import_hits": (meta or {}).get("blocked_import_hits", []),
            "models_reverified_after_run": (meta or {}).get("models_reverified_after_run", False),
        }
        if not ok:
            hermetic_ok = False
            reasons.append(f"hermetic proof failed for '{profile_id}': {'; '.join(problems)}")
        if blocks:
            configuration[profile_id] = blocks["configuration"]
            if environment is None:
                environment = blocks["environment"]
            elif blocks["environment"] != environment:
                reasons.append(f"worker '{profile_id}' reports a different environment block")
    if not hermetic_ok and code in (EXIT_OK, EXIT_INCOMPLETE):
        code = EXIT_HERMETIC

    # Which scan suites finished ---------------------------------------------
    scan_suites = [s for suites in by_profile.values() for s in suites if s["kind"] == "scan"]
    complete = [s for s in scan_suites if (run_dir / "records" / f"{s['id']}.jsonl").exists()]
    incomplete = [s["id"] for s in scan_suites if s not in complete]
    other = [s for suites in by_profile.values() for s in suites if s["kind"] != "scan"]
    incomplete += [s["id"] for s in other if not (run_dir / "status" / f"{s['id']}.done").exists()]
    if incomplete and code == EXIT_OK:
        code = EXIT_INCOMPLETE

    # Canonical? --------------------------------------------------------------
    if incomplete:
        reasons.append(f"incomplete suite(s): {incomplete}")
    if code != EXIT_OK:
        reasons.append(f"exit code {code}")
    if kind == "custom" or plan_doc["explicit_suite_selection"]:
        reasons.append("explicit suite selection")
    if plan_doc["limit"] is not None:
        reasons.append("--limit was used")
    if plan_doc["model_overrides"]:
        reasons.append("a model hash was given on the command line")
    if plan_doc["hash_seed"] != "0":
        reasons.append(f"PYTHONHASHSEED={plan_doc['hash_seed']}")
    if git["commit"] is None:
        reasons.append("not a git checkout: the code revision is unknown")
    elif git["dirty_subject"]:
        reasons.append(f"uncommitted changes in measured paths: {git['subject_paths_changed'][:5]}")
    for profile_id, table in model_tables.items():
        if any(not v["in_manifest"] for v in table.values()):
            reasons.append(f"profile '{profile_id}' uses a model that is not in the manifest")
    canonical_run = not reasons

    # Fingerprint -------------------------------------------------------------
    fp = None
    if environment is not None and configuration:
        identity = {
            "subject": subject,
            "git": {"commit": git["commit"], "dirty_subject": git["dirty_subject"],
                    "dirty_other": git["dirty_other"]},
            "models": {p: {r: {"file": v["file"], "sha256": v["sha256"], "size": v["size"],
                               "in_manifest": v["in_manifest"]} for r, v in t.items()}
                       for p, t in model_tables.items()},
            "model_manifest": {"path": integrity.MANIFEST_PATH, "release_tag": manifest["release_tag"]},
            "declared": {p: (canonical.read_json(run_dir / "work" / p / "blocks.json")["declared"])
                         for p in configuration},
            "datasets": ds_identity,
            "suites": {s["id"]: transforms.suite_definition_hash(s) for s in complete},
            "fixtures": dict(fixtures.hashes),
            "harness": harness,
        }
        fp = fingerprint.build(identity, configuration, environment)
        canonical.write_json(run_dir / "fingerprint.json", fp)

    # Summary and report (modules arrive in step 8) ---------------------------
    summary = None
    if fp is not None and complete:
        try:
            from evals import metrics, report
        except ImportError:
            metrics = report = None
        if metrics is not None:
            summary = metrics.summarize(run_dir, repo, complete, by_profile, profiles_reg, suites_reg,
                                        fp, canonical_run, workers_meta, plan_doc)
            canonical.write_json(run_dir / "summary.json", summary)
            canonical.write_bytes(run_dir / "REPORT.md",
                                  report.render(summary, fp, profiles_reg, suites_reg).encode("utf-8"))
            notes = report.render_run_notes(run_dir, summary, workers_meta, by_profile)
            if notes is not None:
                latency_doc, text = notes
                canonical.write_bytes(run_dir / "latency.json",
                                      (json.dumps(latency_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"))
                canonical.write_bytes(run_dir / "RUN_NOTES.md", text.encode("utf-8"))
                if latency_doc.get("stability") is not None:
                    canonical.write_bytes(
                        run_dir / "known_unstable.json",
                        (json.dumps({"note": "Prompts whose verdict changed between identical "
                                             "unseeded passes. Run metadata, not byte-compared.",
                                     "datasets": latency_doc["stability"]},
                                    indent=2, sort_keys=True) + "\n").encode("utf-8"))

    digest = write_manifest(run_dir)

    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "kind": kind, "kind_description": PLAN_KINDS.get(kind, ""),
        "primary_profile": primary,
        "started_utc": datetime.fromtimestamp(started_wall, timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 2),
        "argv": sys.argv[1:],
        "exit_code": code,
        "canonical": canonical_run,
        "not_canonical_because": reasons,
        "artifact_digest": digest,
        "git": git,
        "workers": workers_meta,
        "incomplete_suites": incomplete,
        "unmanifested_model_files_present_not_loaded": integrity.unmanifested_model_files(repo, manifest),
        "orchestrator_guard": hermetic.summary(),
        "keys": fp["keys"] if fp else None,
    }
    canonical.write_bytes(run_dir / "run.json",
                          (json.dumps(run_meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return code, run_meta


def deterministic_files(run_dir: Path) -> list[str]:
    files = [name for name in DETERMINISTIC_FILES if (run_dir / name).exists()]
    for folder in DETERMINISTIC_DIRS:
        d = run_dir / folder
        if d.is_dir():
            files += [f"{folder}/{p.name}" for p in sorted(d.glob("*.jsonl"))]
    return sorted(files)


def write_manifest(run_dir: Path) -> str:
    """
    MANIFEST.sha256 lists every deterministic file with the SHA-256 of its
    LF-normalized bytes, sorted by path. Its own hash is the run's artifact digest.
    """
    lines = [f"{canonical.sha256_bytes(canonical.read_lf(run_dir / rel))}  {rel}"
             for rel in deterministic_files(run_dir)]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    canonical.write_bytes(run_dir / "MANIFEST.sha256", data)
    return canonical.sha256_bytes(data)


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    for line in canonical.read_lf(path).decode("utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            out[rel] = digest
    return out


def flagged_counts(run_dir: Path) -> dict[str, list[int]]:
    """{dataset: [flagged, n]} for base-variant records. Used by the count checkpoint."""
    counts: dict[str, list[int]] = {}
    rec_dir = run_dir / "records"
    for path in sorted(rec_dir.glob("std.*.jsonl")):
        for rec in canonical.read_jsonl(path):
            if rec["variant"] != "base":
                continue
            c = counts.setdefault(rec["dataset"], [0, 0])
            c[0] += int(bool(rec["flagged"]))
            c[1] += 1
    return counts
