"""Implementations of the `python -m evals` commands. All run under the orchestrator's guard."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from evals import canonical, datasets, fingerprint, hermetic, integrity, orchestrator, transforms
from evals.datasets import DatasetIntegrityError
from evals.integrity import ModelIntegrityError
from evals.orchestrator import (EXIT_BASELINE_MISMATCH, EXIT_DATASET, EXIT_HERMETIC, EXIT_MODEL,
                                EXIT_NONDETERMINISTIC, EXIT_OK, EXIT_USAGE, RunError, say)

BASELINES = "evals/baselines"


def dispatch(args) -> int:
    try:
        orchestrator.arm_guard()
        if args.command == "selftest":
            return cmd_selftest()
        if args.command == "run":
            return cmd_run(args)
        if args.command == "baseline":
            return cmd_baseline(args)
        if args.command == "verify-determinism":
            return cmd_verify(args)
        if args.command == "pin":
            return cmd_pin(args)
    except RunError as exc:
        print(f"evals: {exc}", file=sys.stderr)
        return exc.code
    except ModelIntegrityError as exc:
        print(f"evals: {exc}", file=sys.stderr)
        return EXIT_MODEL
    except DatasetIntegrityError as exc:
        print(f"evals: {exc}", file=sys.stderr)
        return EXIT_DATASET
    return EXIT_USAGE


# ── selftest ─────────────────────────────────────────────────────────────────

def cmd_selftest() -> int:
    repo = orchestrator.repo_root()
    guard = hermetic.summary()
    say(f"guard        armed, self-test passed ({guard['canary_events']} canaries denied and recorded)")
    for row in datasets.verify_all(repo):
        say(f"dataset      {row['id']:<18} rows={row['rows']:<4} unique={row['unique']:<4} "
            f"content={row['content_sha256'][:12]}  ok")
    profiles = transforms.load_profiles(repo)
    manifest = integrity.load_manifest(repo)
    for profile_id, prof in profiles["profiles"].items():
        table = integrity.verify_roles(prof["models"], repo, manifest, required=prof["required_roles"])
        if not table:
            say(f"models       {profile_id:<24} loads no model files")
        for role, info in table.items():
            say(f"models       {profile_id:<24} {role:<22} {info['sha256'][:12]}  ok")
    transforms.load_fixtures(repo, transforms.load_suites(repo))
    say("fixtures     ok")
    if hermetic.violation_count():
        say(f"guard        {hermetic.violation_count()} unexpected denied operation(s)")
        return EXIT_HERMETIC
    say("selftest     passed")
    return EXIT_OK


# ── run / baseline ───────────────────────────────────────────────────────────

def _print_counts(run_dir: Path) -> dict:
    counts = orchestrator.flagged_counts(run_dir)
    if counts:
        say("")
        say(f"  {'dataset':<18}{'flagged':>9}  /  n")
        for ds_id, (flagged, n) in counts.items():
            say(f"  {ds_id:<18}{flagged:>9}  /  {n}")
    return counts


def cmd_run(args) -> int:
    overrides = {}
    for item in args.model_sha256:
        role, sep, digest = item.partition("=")
        if not sep or len(digest) != 64:
            raise RunError(EXIT_USAGE, f"--model-sha256 expects ROLE=<64 hex digits>, got {item!r}")
        overrides[role] = digest.lower()
    suite_ids = [s.strip() for s in args.suites.split(",") if s.strip()] if args.suites else None
    kind = args.plan
    profiles = transforms.load_profiles(orchestrator.repo_root())
    primary = args.profile or profiles["canonical_profile"]
    if primary != profiles["canonical_profile"] and kind == "baseline":
        kind = "reference"          # a non-canonical primary profile runs the standard suites
    code, run_dir = orchestrator.execute(
        kind=kind, primary=primary, suite_ids=suite_ids, out=args.out, limit=args.limit,
        resume=args.resume, hash_seed=args.hash_seed, model_overrides=overrides or None)
    counts = _print_counts(run_dir)
    expected = profiles["profiles"][primary].get("expected_counts")
    if expected and counts and args.limit is None and code == EXIT_OK:
        mismatch = _count_mismatches(counts, expected)
        if mismatch:
            say("\n  counts differ from the profile's approved counts:")
            for line in mismatch:
                say("    " + line)
            code = EXIT_BASELINE_MISMATCH
        else:
            say(f"\n  counts match the approved counts for profile '{primary}'")
    say(f"\n  report: {run_dir / 'REPORT.md'}")
    return code


def _count_mismatches(counts: dict, expected: dict) -> list[str]:
    out = []
    for ds_id, want in expected.items():
        if ds_id not in counts:
            continue
        got = counts[ds_id][0]
        if got != want:
            out.append(f"{ds_id}: flagged {got} / {counts[ds_id][1]}, expected {want}")
    return out


def _canonical_baseline(repo: Path) -> Path | None:
    pointer = repo / BASELINES / "CANONICAL"
    if not pointer.exists():
        return None
    name = pointer.read_text(encoding="utf-8").strip()
    path = repo / BASELINES / name
    return path if name and path.is_dir() else None


def cmd_baseline(args) -> int:
    """Run every suite, then check the standard counts against the canonical baseline."""
    repo = orchestrator.repo_root()
    profiles = transforms.load_profiles(repo)
    primary = profiles["canonical_profile"]
    code, run_dir = orchestrator.execute(kind="baseline", primary=primary, out=args.out,
                                         resume=args.resume)
    counts = _print_counts(run_dir)
    if code != EXIT_OK:
        say(f"\n  run did not complete cleanly (exit {code}); see {run_dir / 'run.json'}")
        return code

    expected = profiles["profiles"][primary]["expected_counts"]
    mismatch = _count_mismatches(counts, expected)
    if mismatch:
        say("\n  COUNTS DIFFER from the approved canonical counts:")
        for line in mismatch:
            say("    " + line)
        return EXIT_BASELINE_MISMATCH
    say("\n  standard counts match the approved canonical counts")

    pinned = _canonical_baseline(repo)
    if pinned is None:
        say("  no canonical baseline is pinned yet")
    else:
        same, differing = _compare_records(pinned, run_dir)
        if differing:
            say(f"  per-prompt records differ from {pinned.name} in: {differing}")
            return EXIT_BASELINE_MISMATCH
        say(f"  per-prompt records are byte-identical to {pinned.name} ({same} file(s))")
    say(f"\n  report: {run_dir / 'REPORT.md'}")
    return EXIT_OK


def _compare_records(baseline: Path, run_dir: Path) -> tuple[int, list[str]]:
    same, differing = 0, []
    for path in sorted((baseline / "records").glob("*.jsonl")):
        other = run_dir / "records" / path.name
        if not other.exists() or canonical.read_lf(path) != canonical.read_lf(other):
            differing.append(path.name)
        else:
            same += 1
    return same, differing


# ── determinism ──────────────────────────────────────────────────────────────

def _diff_runs(a: Path, b: Path) -> tuple[list[str], list[str]]:
    """Return (gated files that differ, non-gated deterministic files that differ)."""
    ma = orchestrator.read_manifest(a / "MANIFEST.sha256")
    mb = orchestrator.read_manifest(b / "MANIFEST.sha256")
    gated, other = [], []
    for rel in sorted(set(ma) | set(mb)):
        if ma.get(rel) == mb.get(rel):
            continue
        (gated if rel.startswith(orchestrator.GATED_PREFIXES) else other).append(rel)
    return gated, other


def _record_differences(a: Path, b: Path, limit: int = 40) -> list[dict]:
    out = []
    for path in sorted((a / "records").glob("*.jsonl")):
        other = b / "records" / path.name
        if not other.exists():
            out.append({"file": path.name, "problem": "missing in the second run"})
            continue
        ra, rb = canonical.read_jsonl(path), canonical.read_jsonl(other)
        for x, y in zip(ra, rb):
            if x != y:
                fields = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
                out.append({"file": path.name, "idx": x.get("idx"), "dataset": x.get("dataset"),
                            "variant": x.get("variant"), "fields": fields,
                            "first": {k: x.get(k) for k in fields if k != "layer_scores"},
                            "second": {k: y.get(k) for k in fields if k != "layer_scores"}})
                if len(out) >= limit:
                    return out
        if len(ra) != len(rb):
            out.append({"file": path.name, "problem": f"{len(ra)} vs {len(rb)} records"})
    return out


def cmd_verify(args) -> int:
    """
    Two runs in separate worker processes, compared byte for byte; then a third
    with a random hash seed, compared record by record.
    """
    repo = orchestrator.repo_root()
    profiles = transforms.load_profiles(repo)
    primary = args.profile or profiles["canonical_profile"]
    reference = primary != profiles["canonical_profile"]
    kind = "reference" if reference else ("deterministic" if args.all else "standard")
    scope = "all" if (args.all or reference) else "standard"

    say(f"verify-determinism: scope={scope} profile={primary}")
    code_a, run_a = orchestrator.execute(kind=kind, primary=primary, out=args.out)
    code_b, run_b = orchestrator.execute(kind=kind, primary=primary, out=args.out)
    if code_a != EXIT_OK or code_b != EXIT_OK:
        say(f"verify-determinism: a run failed (exit {code_a}, {code_b})")
        return code_a or code_b

    gated, other = _diff_runs(run_a, run_b)
    meta_a = json.loads((run_a / "run.json").read_text(encoding="utf-8"))
    meta_b = json.loads((run_b / "run.json").read_text(encoding="utf-8"))
    keys_equal = meta_a["keys"] == meta_b["keys"]

    # Random hash seed: the standard suites again, in a process whose string
    # hashing is randomized. Records must not depend on it.
    seed_kind = "reference" if reference else "standard"
    code_c, run_c = orchestrator.execute(kind=seed_kind, primary=primary, out=args.out,
                                         hash_seed="random")
    seed_diffs = _record_differences(run_c, run_a) if code_c == EXIT_OK else [{"problem": f"exit {code_c}"}]

    proof = {
        "verified": not gated and keys_equal and not seed_diffs,
        "scope": scope, "plan_kind": kind, "profile": primary,
        "runs": [run_a.name, run_b.name],
        "keys_equal": keys_equal, "keys": meta_a["keys"],
        "artifact_digest": meta_a["artifact_digest"],
        "artifact_digest_second_run": meta_b["artifact_digest"],
        "gated_files_compared": len([r for r in orchestrator.read_manifest(run_a / "MANIFEST.sha256")
                                     if r.startswith(orchestrator.GATED_PREFIXES)]),
        "gated_files_differing": gated,
        "evidence_or_report_files_differing": other,
        "record_differences": _record_differences(run_a, run_b) if gated else [],
        "hash_seed_experiment": {
            "run": run_c.name, "pythonhashseed": "random", "plan_kind": seed_kind,
            "record_differences": seed_diffs,
            "records_identical_to_first_run": not seed_diffs,
        },
        "harness_tree_sha256": fingerprint.harness_identity(repo)["tree_sha256"],
        "verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    proof_path = run_a / "DETERMINISM.json"
    canonical.write_bytes(proof_path, (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    say(f"\n  runs compared              {run_a.name}  vs  {run_b.name}")
    say(f"  four keys equal            {keys_equal}")
    say(f"  gated files compared       {proof['gated_files_compared']}")
    say(f"  gated files differing      {len(gated)}  {gated[:6]}")
    say(f"  evidence/report differing  {len(other)}  {other[:6]}  (reported, not fatal)")
    say(f"  random hash seed           {len(seed_diffs)} record difference(s) vs the first run")
    say(f"  proof                      {proof_path}")
    if gated or not keys_equal:
        say("\n  NOT DETERMINISTIC. Differing records:")
        for d in proof["record_differences"][:20]:
            say(f"    {d}")
        return EXIT_NONDETERMINISTIC
    if seed_diffs:
        say("\n  RECORDS DEPEND ON THE HASH SEED:")
        for d in seed_diffs[:20]:
            say(f"    {d}")
        return EXIT_NONDETERMINISTIC
    say("\n  deterministic: byte-identical results, summary and fingerprint")
    return EXIT_OK


# ── pin ──────────────────────────────────────────────────────────────────────

def cmd_pin(args) -> int:
    """
    Copy a canonical run's deterministic files into evals/baselines/. Never
    overwrites: a baseline is immutable, a correction is a new baseline.
    """
    if args.canonical == args.reference:
        raise RunError(EXIT_USAGE, "pin needs exactly one of --canonical or --reference")
    repo = orchestrator.repo_root()
    run_dir = Path(args.run_dir).resolve()
    try:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        fp = canonical.read_json(run_dir / "fingerprint.json")
    except (OSError, ValueError) as exc:
        raise RunError(EXIT_USAGE, f"not a completed run directory: {exc}")

    problems = []
    if not meta.get("canonical"):
        problems.append(f"the run is not canonical: {meta.get('not_canonical_because')}")
    want_kind = "baseline" if args.canonical else "reference"
    if meta.get("kind") != want_kind:
        problems.append(f"the run's plan is '{meta.get('kind')}', pinning needs '{want_kind}'")
    profiles = transforms.load_profiles(repo)
    if args.canonical and meta.get("primary_profile") != profiles["canonical_profile"]:
        problems.append("a canonical baseline must use the canonical reproducibility profile")
    if orchestrator.write_manifest(run_dir) != meta.get("artifact_digest"):
        problems.append("the run's deterministic files changed after it finished")

    if not args.determinism_proof:
        problems.append("--determinism-proof is required (run `verify-determinism --all` first)")
    else:
        try:
            proof = json.loads(Path(args.determinism_proof).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RunError(EXIT_USAGE, f"cannot read determinism proof: {exc}")
        if not proof.get("verified"):
            problems.append("the determinism proof did not verify")
        if proof.get("scope") != "all":
            problems.append("the determinism proof covers the standard suites only; `--all` is required")
        if proof.get("artifact_digest") != meta.get("artifact_digest"):
            problems.append("the determinism proof is for different artifacts than this run "
                            f"(proof {str(proof.get('artifact_digest'))[:12]}…, "
                            f"run {str(meta.get('artifact_digest'))[:12]}…)")
    if problems:
        raise RunError(EXIT_USAGE, "refusing to pin:\n  - " + "\n  - ".join(problems))

    primary = meta["primary_profile"]
    declared = fp["identity"]["declared"][primary]["pair"]["declared_version"]
    tree8 = fp["identity"]["subject"]["fie_tree_sha256"][:8]
    prefix = "BL" if args.canonical else "REF"
    base = repo / BASELINES
    base.mkdir(parents=True, exist_ok=True)
    number = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in base.glob(f"{prefix}-*")
                      if p.is_dir() and p.name.split("_")[0].split("-")[1].isdigit()] or [0])
    name = f"{prefix}-{number:04d}_pair-{declared}_fie-{tree8}"
    dest = base / name
    if dest.exists():
        raise RunError(EXIT_USAGE, f"{dest} already exists; baselines are never overwritten")
    dest.mkdir()

    for rel in ("fingerprint.json", "summary.json", "REPORT.md", "MANIFEST.sha256"):
        shutil.copyfile(run_dir / rel, dest / rel)
    if args.canonical:
        (dest / "records").mkdir()
        for path in sorted((run_dir / "records").glob("*.jsonl")):
            shutil.copyfile(path, dest / "records" / path.name)
    # Run metadata travels with the baseline but is not part of the byte-compared set.
    extra = {}
    for rel in ("RUN_NOTES.md", "latency.json", "known_unstable.json"):
        if (run_dir / rel).exists():
            shutil.copyfile(run_dir / rel, dest / rel)
            extra[rel] = True
    pin_doc = {
        "baseline_id": name, "kind": "canonical baseline" if args.canonical else "reference reproduction",
        "pinned_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": meta["run_id"], "artifact_digest": meta["artifact_digest"],
        "keys": meta["keys"], "git": meta["git"],
        "determinism_proof": proof,
        "note": ("fingerprint.json, summary.json, REPORT.md, MANIFEST.sha256 and records/ are the "
                 "deterministic artifact. Everything else in this directory is run metadata and is "
                 "not byte-compared. MANIFEST.sha256 also lists the run's evidence files, which are "
                 "not copied here."),
        "metadata_files": sorted(extra),
    }
    canonical.write_bytes(dest / "PIN.json",
                          (json.dumps(pin_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if args.canonical:
        canonical.write_bytes(base / "CANONICAL", (name + "\n").encode("utf-8"))
    say(f"pinned {name}")
    if args.canonical:
        say(f"CANONICAL -> {name}")
    return EXIT_OK
