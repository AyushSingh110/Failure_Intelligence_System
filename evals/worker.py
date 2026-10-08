"""
The worker: one fresh interpreter, one profile, the suites assigned to it.

Started by the orchestrator as

    python -X utf8 -B -m evals.worker <plan.json> <guard-events.jsonl>

with a sanitized environment. Lifecycle:

    1  arm the hermetic guard, run its self-test      (before anything else)
    2  make the profile's blocked packages unimportable
    3  verify datasets and fixtures
    4  verify model files                              (before `fie` is imported)
    5  import fie, warm up, confirm what loaded is what was verified
    6  write the configuration and environment blocks
    7  run the suites in order, checking the guard after every scan
    8  re-verify model files, write the guard summary, exit

Exit codes: 0 ok · 2 a scan raised or a suite is incomplete · 3 model integrity
· 4 dataset integrity · 5 hermetic violation or guard self-test failure.

Only the standard library is imported here, so the lite profile can make numpy
unimportable. `fie` is reached only through evals.subject.
"""
import os
import sys

from evals import hermetic

EXIT_OK, EXIT_INCOMPLETE, EXIT_MODEL, EXIT_DATASET, EXIT_HERMETIC = 0, 2, 3, 4, 5


def _arm(argv):
    if len(argv) != 3:
        print("usage: python -m evals.worker <plan.json> <guard-events.jsonl>", file=sys.stderr)
        raise SystemExit(1)
    hermetic.install(mode="worker", event_log=argv[2])
    return hermetic.selftest()


class _Stop(Exception):
    def __init__(self, code: int, status: str, detail: str) -> None:
        super().__init__(detail)
        self.code, self.status, self.detail = code, status, detail


def check_guard(where: str) -> None:
    """
    Fail the run if the guard recorded any non-canary denied operation.

    This, not the exception the guard raises, is what stops a run: the product
    catches exceptions around its own network calls and carries on.
    """
    if hermetic.violation_count():
        first = hermetic.violations()[0]
        raise _Stop(EXIT_HERMETIC, "hermetic_violation",
                    f"{hermetic.violation_count()} denied operation(s) {where}; first: "
                    f"{first['event']} {first['target']} on thread {first['thread']}")


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    selftest = _arm(argv)                       # step 1 — nothing else has been imported

    import json
    import time
    from pathlib import Path

    from evals import canonical, datasets, fingerprint, integrity, transforms
    from evals.datasets import DatasetIntegrityError
    from evals.integrity import ModelIntegrityError

    plan = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    repo = Path(plan["repo_root"])
    run_dir = Path(plan["run_dir"])
    profile_id, profile = plan["profile_id"], plan["profile"]
    work = run_dir / "work" / profile_id
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    meta = {
        "profile_id": profile_id, "status": "started", "detail": "",
        "guard_selftest": selftest, "suites": {}, "timings": {},
    }

    blocker = None

    def save_meta() -> None:
        meta["guard"] = hermetic.summary()
        meta["blocked_import_hits"] = sorted(blocker.hits) if blocker else []
        meta["pickle_classes"] = hermetic.observed_pickle_classes()
        canonical.write_bytes(work / "worker.json",
                              (json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False)
                               + "\n").encode("utf-8"))

    exit_code = EXIT_OK
    try:
        if not selftest["passed"]:
            raise _Stop(EXIT_HERMETIC, "guard_selftest_failed", json.dumps(selftest["checks"]))

        blocker = hermetic.block_imports(profile.get("blocked_imports", []))      # step 2

        try:                                                                      # step 3
            ds_registry = datasets.load_registry(repo)
            suites_registry = transforms.load_suites(repo)
            needed = []
            for suite in plan["suites"]:
                for ds_id in transforms.suite_datasets(suite):
                    if ds_id not in needed:
                        needed.append(ds_id)
            loaded = {ds_id: datasets.load_dataset(ds_id, repo, ds_registry) for ds_id in needed}
            fixtures = transforms.load_fixtures(repo, suites_registry)
            for ds_id, ds in loaded.items():
                want = plan["dataset_expect"].get(ds_id)
                if want != ds.content_sha256:
                    raise DatasetIntegrityError(
                        f"dataset '{ds_id}' changed between orchestrator and worker")
        except (DatasetIntegrityError, transforms.RegistryError) as exc:
            raise _Stop(EXIT_DATASET, "dataset_integrity", str(exc))

        try:                                                                      # step 4
            verified = integrity.verify_roles(
                profile["models"], repo, required=profile["required_roles"],
                overrides=plan.get("model_overrides") or {})
            for role, info in verified.items():
                if plan["model_expect"].get(role) != info["sha256"]:
                    raise ModelIntegrityError(
                        f"role '{role}' changed between orchestrator and worker")
            declared = integrity.declared_metadata(profile["models"], repo)
        except ModelIntegrityError as exc:
            raise _Stop(EXIT_MODEL, "model_integrity", str(exc))
        meta["models_verified_before_import"] = verified
        if "fie" in sys.modules:
            raise _Stop(EXIT_MODEL, "model_integrity", "fie was imported before verification")

        from evals import subject                                                 # step 5
        try:
            state = subject.prepare(profile, repo, verified, declared)
        except ModelIntegrityError as exc:
            raise _Stop(EXIT_MODEL, "model_integrity", str(exc))
        except subject.SubjectError as exc:
            raise _Stop(EXIT_INCOMPLETE, "profile_precondition", str(exc))
        meta["subject_state"] = state
        check_guard("during import and warm-up")

        t0 = time.perf_counter()
        subject.scan("What is the capital of France?")
        meta["timings"]["first_scan_s"] = round(time.perf_counter() - t0, 4)
        meta["timings"]["import_s"] = state["import_s"]
        meta["timings"]["warmup_s"] = state["warmup_s"]
        subject.clear_caches()

        blocks = {                                                                # step 6
            "configuration": fingerprint.profile_configuration(
                profile_id, profile, subject.configuration(),
                profile.get("translation", "unavailable"), profile.get("langdetect_seed", 0)),
            "environment": fingerprint.environment_block(),
            "models": {role: {"file": v["file"], "path": v["path"], "sha256": v["sha256"],
                              "size": v["size"], "in_manifest": v["in_manifest"]}
                       for role, v in verified.items()},
            "declared": declared,
        }
        canonical.write_json(work / "blocks.json", blocks)
        save_meta()

        default_translation = profile.get("translation", "unavailable")
        default_seed = profile.get("langdetect_seed", 0)
        any_errors = False

        for suite in plan["suites"]:                                              # step 7
            sid = suite["id"]
            if (run_dir / "status" / f"{sid}.done").exists():
                meta["suites"][sid] = {"status": "skipped_already_complete"}
                continue
            items = transforms.build_items(suite, loaded, fixtures, plan.get("limit"))
            subject.clear_caches()
            subject.set_translation(default_translation)
            subject.set_langdetect_seed(default_seed)
            t_suite = time.perf_counter()

            if suite["kind"] == "scan":
                errors = _run_scan_suite(suite, items, run_dir, subject, canonical, check_guard,
                                         time, default_translation)
            elif suite["kind"] == "latency":
                errors = _run_latency_suite(suite, items, run_dir, subject, check_guard, time, json)
            else:
                errors = _run_stability_suite(suite, loaded, run_dir, subject, check_guard, json,
                                              plan.get("limit"))
            subject.set_translation(default_translation)
            subject.set_langdetect_seed(default_seed)

            wall = round(time.perf_counter() - t_suite, 3)
            meta["suites"][sid] = {"status": "complete" if not errors else "complete_with_errors",
                                   "kind": suite["kind"], "items": len(items),
                                   "scan_errors": errors, "wall_s": wall}
            any_errors = any_errors or bool(errors)
            (run_dir / "status").mkdir(exist_ok=True)
            (run_dir / "status" / f"{sid}.done").write_text(
                "complete\n" if not errors else "complete_with_errors\n", encoding="utf-8")
            save_meta()

        try:                                                                      # step 8
            integrity.reverify(verified, repo)
        except ModelIntegrityError as exc:
            raise _Stop(EXIT_MODEL, "model_changed_during_run", str(exc))
        check_guard("by the end of the run")
        meta["models_reverified_after_run"] = True
        meta["status"] = "complete" if not any_errors else "complete_with_scan_errors"
        exit_code = EXIT_OK if not any_errors else EXIT_INCOMPLETE

    except _Stop as stop:
        meta["status"], meta["detail"] = stop.status, stop.detail
        exit_code = stop.code
        print(f"[evals.worker:{profile_id}] {stop.status}: {stop.detail}", file=sys.stderr)

    meta["timings"]["worker_wall_s"] = round(time.perf_counter() - started, 3)
    save_meta()
    return exit_code


def _run_scan_suite(suite, items, run_dir, subject, canonical, check_guard, time,
                    default_translation) -> int:
    """Deterministic records, evidence beside them, timings kept apart."""
    sid = suite["id"]
    rec_dir, ev_dir, tm_dir = run_dir / "records", run_dir / "evidence", run_dir / "timing"
    for d in (rec_dir, ev_dir, tm_dir):
        d.mkdir(parents=True, exist_ok=True)
    rec_tmp = rec_dir / f"{sid}.jsonl.partial"
    ev_tmp = ev_dir / f"{sid}.jsonl.partial"
    tm_tmp = tm_dir / f"{sid}.jsonl.partial"
    errors = 0
    current_translation = default_translation        # set by the caller before this suite
    with open(rec_tmp, "wb") as rec_f, open(ev_tmp, "wb") as ev_f, open(tm_tmp, "wb") as tm_f:
        for item in items:
            want = item["runtime"].get("translation", default_translation)
            if want != current_translation:
                # Runtime knobs change only at part boundaries; clear caches so
                # nothing computed under the previous setting is reused.
                subject.set_translation(want)
                subject.clear_caches()
                current_translation = want
            t0 = time.perf_counter()
            fields, evidence = subject.scan(item["text"])
            ms = (time.perf_counter() - t0) * 1000.0
            if fields["status"] != "ok":
                errors += 1
            record = {
                "suite": sid, "idx": item["idx"], "dataset": item["dataset"],
                "source_idx": item["source_idx"], "variant": item["variant"],
                "input_sha256": item["input_sha256"], "expected": item["expected"],
            }
            record.update(fields)
            rec_f.write(canonical.record_line(record))
            ev_f.write(canonical.record_line({"suite": sid, "idx": item["idx"], "evidence": evidence}))
            tm_f.write(canonical.record_line({"idx": item["idx"], "chars": len(item["text"]),
                                              "ms": round(ms, 3)}))
            check_guard(f"while scanning {sid} item {item['idx']}")
    for tmp in (rec_tmp, ev_tmp, tm_tmp):
        os.replace(tmp, tmp.with_name(tmp.name[: -len(".partial")]))
    return errors


def _run_latency_suite(suite, items, run_dir, subject, check_guard, time, json) -> int:
    """
    Warm latency. The result cache is cleared before every scan so each timing is
    a real scan. One discarded pass, then `passes` measured passes. Timing never
    enters a deterministic artifact.
    """
    sid = suite["id"]
    out_dir = run_dir / "nondeterministic"
    out_dir.mkdir(parents=True, exist_ok=True)
    passes = int(suite.get("params", {}).get("passes", 3))
    errors = 0
    with open(out_dir / f"{sid}.jsonl", "wb") as f:
        for p in range(passes + 1):                     # pass 0 is the discarded warm-up
            for item in items:
                subject.clear_caches()
                t0 = time.perf_counter()
                fields, _ = subject.scan(item["text"])
                ms = (time.perf_counter() - t0) * 1000.0
                if fields["status"] != "ok":
                    errors += 1
                if p:
                    f.write((json.dumps({"pass": p, "idx": item["idx"], "dataset": item["dataset"],
                                         "variant": item["variant"], "chars": len(item["text"]),
                                         "ms": round(ms, 3)}, sort_keys=True) + "\n").encode("utf-8"))
            check_guard(f"during latency pass {p}")
    return errors


def _run_stability_suite(suite, loaded, run_dir, subject, check_guard, json, limit) -> int:
    """
    The shipped behaviour: language detector UNSEEDED, several identical passes.
    Counts verdicts that change between passes. Non-deterministic by design, so
    its output is run metadata, never a deterministic artifact.
    """
    sid = suite["id"]
    out_dir = run_dir / "nondeterministic"
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = 0
    subject.set_langdetect_seed(None)
    with open(out_dir / f"{sid}.jsonl", "wb") as f:
        for part in suite["parts"]:
            ds = loaded[part["dataset"]]
            prompts = ds.prompts[:limit] if limit else ds.prompts
            for p in range(int(part.get("passes", 3))):
                subject.clear_caches()
                flagged = []
                for idx, prompt in enumerate(prompts):
                    fields, _ = subject.scan(prompt)
                    if fields["status"] != "ok":
                        errors += 1
                    if fields["flagged"]:
                        flagged.append(idx)
                f.write((json.dumps({"dataset": ds.id, "pass": p, "n": len(prompts),
                                     "flagged_idx": flagged}, sort_keys=True) + "\n").encode("utf-8"))
                check_guard(f"during stability pass {p} of {ds.id}")
    return errors


if __name__ == "__main__":
    sys.exit(main())
