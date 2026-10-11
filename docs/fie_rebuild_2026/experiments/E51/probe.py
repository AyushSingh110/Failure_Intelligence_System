"""
E51 probe. Runs INSIDE a clean environment, measures the INSTALLED fie. Standard library only.

    <env>\\Scripts\\python.exe -I -B probe.py --mode S|PC --inputs <jsonl> --out <dir>
        --header <json file> --verified <json file with verified model SHA-256 list> --repo <repo root>

Modes (PLAN_020 section 4):
    S    as-shipped configuration: no FIE_* variable is set by the caller and nothing in fie is
         patched. scan_prompt(prompt) is called with its defaults.
    PC   comparable: the base-install profile's settings from evals/registry/profiles.json are
         applied in-process (translator returns nothing, language detector seeded, result cache
         cleared), and scan_prompt(prompt, use_llama_guard=False) is called, as the harness does.

What is loaded around fie in every mode, and is NOT isolation:
    socket recorder     records and refuses Python-level connections and name lookups
    pickle tripwire     ends the process before an unverified pickle file is read
    process recorder    ends the process if the measured code starts another process
Machine-level isolation is the owner's airplane mode, verified by egress_selftest.py.

The probe refuses to run unless `fie` was imported from this environment's site-packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
import time
import warnings
from pathlib import Path

EXIT_NOT_INSTALLED_COPY = 90
EXIT_TRIPWIRE = 97
EXIT_PROCESS = 98

_lock = threading.Lock()
_phase = "startup"
_egress: list[dict] = []
_opened_models: list[dict] = []
_violations: list[dict] = []
_out_dir: Path | None = None
_verified: set[str] = set()
_env_root = ""


def _rel(path: str) -> str:
    p = str(path)
    for root, tag in ((_env_root, "<ENV>"), (str(Path.home()), "<HOME>")):
        if root and p.lower().startswith(root.lower()):
            return tag + p[len(root):].replace("\\", "/")
    return Path(p).name


def _caller() -> str:
    frame = sys._getframe(2)
    for _ in range(12):
        mod = frame.f_globals.get("__name__", "?")
        if mod not in (__name__, "socket", "__main__", "ssl"):
            return mod
        if frame.f_back is None:
            break
        frame = frame.f_back
    return "?"


def _flush(name: str, obj) -> None:
    if _out_dir is not None:
        (_out_dir / name).write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8", newline="\n")


def install_instrumentation() -> None:
    real_connect = socket.socket.connect

    def refuse(kind):
        def _refused(*args, **kwargs):
            target = args[1] if kind in ("connect", "connect_ex") and len(args) > 1 else (args[0] if args else None)
            with _lock:
                _egress.append({"phase": _phase, "kind": kind, "target": repr(target)[:200], "caller": _caller()})
            raise OSError(f"E51: network use refused by the probe ({kind})")
        return _refused

    socket.socket.connect = refuse("connect")
    socket.socket.connect_ex = refuse("connect_ex")
    socket.getaddrinfo = refuse("getaddrinfo")
    socket.gethostbyname = refuse("gethostbyname")
    socket.create_connection = refuse("create_connection")
    del real_connect

    hashing = threading.local()

    def audit(event, args):
        if event == "open":
            if getattr(hashing, "active", False):                  # our own read, below
                return
            path = args[0]
            if isinstance(path, (str, bytes, os.PathLike)):
                name = os.fsdecode(path)
                if name.lower().endswith((".pkl", ".joblib", ".pickle")) and os.path.isfile(name):
                    hashing.active = True
                    try:
                        with open(name, "rb") as fh:               # hashing; no unpickling
                            digest = hashlib.sha256(fh.read()).hexdigest()
                    finally:
                        hashing.active = False
                    ok = digest in _verified
                    with _lock:
                        _opened_models.append({"path": _rel(name), "sha256": digest, "verified": ok,
                                               "phase": _phase})
                    if not ok:
                        _flush("TRIPWIRE.json", {"reason": "unverified model file opened", "path": _rel(name),
                                                 "sha256": digest, "phase": _phase})
                        os._exit(EXIT_TRIPWIRE)
        elif event in ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.startfile"):
            with _lock:
                _violations.append({"phase": _phase, "event": event, "detail": repr(args)[:200]})
            _flush("PROCESS_VIOLATION.json", {"event": event, "phase": _phase})
            os._exit(EXIT_PROCESS)

    sys.addaudithook(audit)


def derive_zone(is_attack: bool, evidence) -> str:
    """The harness's rule (evals/subject.py), for artifacts whose result has no `zone`."""
    if not is_attack:
        return "allow"
    if isinstance(evidence, dict) and evidence.get("llama_guard") == "unavailable_blocked":
        return "uncertain_block"
    return "clear_block"


def record_of(result) -> dict:
    evidence = getattr(result, "evidence", None)
    evidence = evidence if isinstance(evidence, dict) else {}
    is_attack = bool(getattr(result, "is_attack", False))
    zone = getattr(result, "zone", None)
    coverage = getattr(result, "coverage", None)
    out = {
        "flagged": is_attack,
        "zone": zone if isinstance(zone, str) else derive_zone(is_attack, evidence),
        "zone_source": "result.zone" if isinstance(zone, str) else "derived (result has no zone)",
        "type": getattr(result, "attack_type", None),
        "conf": float(getattr(result, "confidence", 0.0) or 0.0),
        "layers_fired": sorted(getattr(result, "layers_fired", None) or []),
        "layer_scores": {str(k): float(v) for k, v in (getattr(result, "layer_scores", None) or {}).items()},
        "degraded": (sorted(getattr(result, "degraded_layers", None) or [])
                     if hasattr(result, "degraded_layers") else None),
        "has_degraded_layers_field": hasattr(result, "degraded_layers"),
        "decided_by": getattr(result, "decided_by", None),
        "coverage": coverage.to_dict() if coverage is not None and hasattr(coverage, "to_dict") else None,
        "status": "ok",
    }
    return out


def main() -> int:
    global _phase, _out_dir, _verified, _env_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["S", "PC"], required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--header", required=True)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _out_dir = out
    _env_root = sys.prefix
    header = json.loads(Path(args.header).read_text(encoding="utf-8"))
    _verified = set(json.loads(Path(args.verified).read_text(encoding="utf-8")))
    inputs = [json.loads(line) for line in Path(args.inputs).read_text(encoding="utf-8").splitlines() if line.strip()]

    fie_env = sorted(k for k in os.environ if k.upper().startswith(("FIE_", "HF_", "TRANSFORMERS_", "GROQ", "PIP_"))
                     or k.upper() in ("SCAN_THRESHOLD", "PREFLIGHT_BLOCK_ENABLED"))
    header.update({
        "mode": args.mode, "env_overrides": {k: os.environ[k] for k in fie_env},
        "in_process_changes": [], "instrumentation": ["socket_recorder", "pickle_tripwire", "process_recorder"],
        "isolation": "owner_network_off", "python": sys.version.split()[0],
        "sys_executable": _rel(sys.executable), "base_executable": Path(getattr(sys, "_base_executable", "")).name,
        "isolated_flag": bool(sys.flags.isolated), "inputs": len(inputs),
        "inputs_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
    })
    if args.mode == "S" and header["env_overrides"]:
        _flush("ABORT.json", {"reason": "mode S must run with no FIE_* variable", "found": fie_env})
        return 2

    # On Windows the standard library's platform.uname() can start a short-lived `ver` process
    # the first time it is called. Call it once before the process recorder is armed (the harness
    # does the same), so that a later call from the measured code is served from its cache and
    # the recorder reports only processes the measured code itself starts.
    import platform
    platform.uname()
    header["prearmed"] = ["platform.uname"]

    install_instrumentation()
    caught_warnings: list = []
    warnings.simplefilter("always")
    _show = warnings.showwarning
    warnings.showwarning = lambda message, category, *a, **k: caught_warnings.append(category.__name__)

    # ── import ───────────────────────────────────────────────────────────────
    _phase = "import"
    repo = os.path.normcase(os.path.abspath(args.repo))
    repo_on_path = [p for p in sys.path if p and os.path.normcase(os.path.abspath(p)).startswith(repo)]
    info: dict = {"repo_on_sys_path": bool(repo_on_path), "cwd_is_repo": os.path.normcase(os.getcwd()).startswith(repo)}
    before = set(sys.modules)
    t0 = time.perf_counter()
    try:
        import fie
        info.update({"import_ok": True, "import_s": round(time.perf_counter() - t0, 3)})
    except BaseException as exc:                                   # the result, not a crash
        info.update({"import_ok": False, "exception": type(exc).__name__, "import_s": round(time.perf_counter() - t0, 3)})
        _flush("import.json", {**header, **info})
        _flush("egress_python.json", {**header, "observations": _egress})
        return 3
    fie_file = os.path.normcase(os.path.abspath(fie.__file__))
    site = os.path.normcase(os.path.join(sys.prefix, "Lib", "site-packages"))
    info.update({
        "fie_file": _rel(fie.__file__), "fie_in_env_site_packages": fie_file.startswith(site),
        "fie_version": getattr(fie, "__version__", None),
        "fie_all": sorted(getattr(fie, "__all__", [])),
        "modules_imported": len(set(sys.modules) - before),
        "third_party_imported": sorted({m.split(".")[0] for m in set(sys.modules) - before} &
                                       {"requests", "numpy", "sklearn", "joblib", "torch", "onnxruntime", "xgboost",
                                        "pandas", "langdetect", "deep_translator", "sentence_transformers"}),
    })
    _flush("import.json", {**header, **info})
    if not info["fie_in_env_site_packages"] or info["repo_on_sys_path"] or info["cwd_is_repo"]:
        _flush("ABORT.json", {"reason": "not measuring the installed copy", **info})
        return EXIT_NOT_INSTALLED_COPY
    time.sleep(1.5)                                                # let import-time background threads act

    import importlib
    adv = importlib.import_module("fie.adversarial")
    try:
        pol = importlib.import_module("fie.policy")
    except Exception:
        pol = None

    # ── comparable-mode settings (the harness's, applied in-process) ─────────
    if args.mode == "PC":
        ml = importlib.import_module("fie.multilingual")
        ml.translate_to_english = lambda text, timeout=3.0: None
        changes = ["fie.multilingual.translate_to_english -> returns None"]
        try:
            from langdetect import DetectorFactory
            DetectorFactory.seed = 0
            changes.append("langdetect.DetectorFactory.seed = 0")
        except ImportError:
            changes.append("langdetect not importable; seed not set")
        cache = getattr(getattr(adv, "_scan_cache", None), "_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        if isinstance(getattr(ml, "_TRANSLATION_CACHE", None), dict):
            ml._TRANSLATION_CACHE.clear()
        changes += ["result and translation caches cleared before the suite", "scan_prompt(use_llama_guard=False)"]
        header["in_process_changes"] = changes

    # ── state ────────────────────────────────────────────────────────────────
    _phase = "warmup"
    state: dict = {"apis_present": {n: hasattr(adv, n) for n in ("warmup", "health", "classifier_state",
                                                                 "model_identities", "ScanCoverage")},
                   "fie.policy_present": pol is not None}
    t0 = time.perf_counter()
    try:
        state["warmup"] = {k: v for k, v in adv.warmup().items() if k != "elapsed_ms"} if hasattr(adv, "warmup") else None
    except Exception as exc:
        state["warmup"] = {"exception": type(exc).__name__}
    state["warmup_s"] = round(time.perf_counter() - t0, 3)
    try:
        if hasattr(adv, "classifier_state"):
            state["classifier_state"] = adv.classifier_state()
        if hasattr(adv, "health"):
            h = adv.health()
            pc = dict(h.get("pair_classifier") or {})
            if pc.get("error"):                                    # free text can hold a local path
                text = str(pc["error"])
                pc["error"] = ("missing dependency" if "missing dependency" in text else
                               "no classifier found" if "no PAIR classifier found" in text else "other")
            state["health"] = {"pair_classifier": pc, "meta_classifier": h.get("meta_classifier"),
                               "layers": h.get("layers")}
        if hasattr(adv, "model_identities"):
            state["model_identities"] = {r: i.to_dict() for r, i in adv.model_identities().items()}
    except Exception as exc:
        state["state_exception"] = type(exc).__name__
    _flush("state.json", {**header, **state})

    # ── scans ────────────────────────────────────────────────────────────────
    _phase = "scan"
    records, policy_rows, timings = [], [], []
    presets = {}
    if pol is not None:
        presets = {"compat": pol.compat_policy(scan_failure="allow"), "strict": pol.STRICT, "review": pol.REVIEW}
    rec_path = out / "records.jsonl"
    with open(rec_path, "w", encoding="utf-8", newline="\n") as rec_f:
        for item in inputs:
            t0 = time.perf_counter()
            try:
                if args.mode == "PC":
                    result = adv.scan_prompt(item["text"], use_llama_guard=False)
                else:
                    result = adv.scan_prompt(item["text"])
                rec = record_of(result)
            except Exception as exc:
                result = None
                rec = {"flagged": False, "zone": "error", "type": None, "conf": 0.0, "layers_fired": [],
                       "layer_scores": {}, "degraded": [], "status": f"error:{type(exc).__name__}"}
            timings.append((time.perf_counter() - t0) * 1000.0)
            row = {"idx": item["idx"], "dataset": item["dataset"], "expected": item.get("expected"),
                   "input_sha256": item["input_sha256"], **rec}
            if "name" in item:
                row["name"] = item["name"]
            rec_f.write(json.dumps(row, sort_keys=True) + "\n")
            records.append(row)
            if presets and result is not None:
                policy_rows.append({"idx": item["idx"], **{name: {"action": d.action, "blocked": d.blocked,
                                                                   "reasons": list(d.reasons)}
                                                           for name, d in ((n, pol.decide(result, p))
                                                                           for n, p in presets.items())}})

    # ── pre-flight guard, default configuration (mode S only) ────────────────
    guard_rows = []
    if args.mode == "S":
        _phase = "preflight"
        try:
            pf = importlib.import_module("fie.preflight")
            for item in inputs[:8]:
                g = pf.preflight_check(item["text"])
                guard_rows.append({"idx": item["idx"], "name": item.get("name"), "blocked": bool(g.blocked),
                                   "attack_type": g.attack_type, "scan_failed": getattr(g, "scan_failed", None),
                                   "zone": getattr(g, "zone", None),
                                   "coverage_status": getattr(g, "coverage_status", None),
                                   "action": getattr(g, "action", None), "policy": getattr(g, "policy", None),
                                   "reasons": list(getattr(g, "reasons", ()) or ())})
        except Exception as exc:
            guard_rows.append({"exception": type(exc).__name__})

    _phase = "done"
    warnings.showwarning = _show
    timings.sort()
    flagged = sum(1 for r in records if r["flagged"])
    by_ds: dict = {}
    for r in records:
        d = by_ds.setdefault(r["dataset"], {"n": 0, "flagged": 0})
        d["n"] += 1
        d["flagged"] += bool(r["flagged"])
    summary = {
        **header,
        "records": len(records), "records_sha256": hashlib.sha256(rec_path.read_bytes()).hexdigest(),
        "flagged": flagged, "errors": sum(1 for r in records if r["status"] != "ok"),
        "by_dataset": by_ds,
        "zones": {z: sum(1 for r in records if r["zone"] == z) for z in sorted({r["zone"] for r in records})},
        "degraded_values": sorted({json.dumps(r.get("degraded")) for r in records}),
        "coverage_status_values": sorted({(r.get("coverage") or {}).get("status") or "absent" for r in records}),
        "classifier_layer_states": sorted({((r.get("coverage") or {}).get("layers") or {}).get("pair_classifier")
                                           or "absent" for r in records}),
        "results_reporting_full_coverage": sum(1 for r in records if r.get("degraded") == []
                                               and (r.get("coverage") or {}).get("status") in (None, "full")),
        "models_opened": _opened_models,
        "unverified_model_opened": any(not m["verified"] for m in _opened_models),
        "process_violations": _violations,
        "egress_attempts_python_level": len(_egress),
        "egress_targets": sorted({(e["phase"], e["kind"], e["target"], e["caller"]) for e in _egress}),
        "warning_categories": {c: caught_warnings.count(c) for c in sorted(set(caught_warnings))},
        "timing_ms": {"mean": round(sum(timings) / len(timings), 3) if timings else None,
                      "p50": round(timings[len(timings) // 2], 3) if timings else None,
                      "max": round(timings[-1], 3) if timings else None},
    }
    _flush("summary.json", summary)
    _flush("egress_python.json", {**header, "note": "process-local observations only; not proof of isolation",
                                  "observations": _egress})
    if policy_rows:
        counts: dict = {}
        for row in policy_rows:
            for name in presets:
                counts.setdefault(name, {}).setdefault(row[name]["action"], 0)
                counts[name][row[name]["action"]] += 1
        _flush("policy.json", {**header, "action_counts": counts, "rows": policy_rows})
    if guard_rows:
        _flush("preflight.json", {**header, "rows": guard_rows})
    print(json.dumps({"mode": args.mode, "records": len(records), "flagged": flagged,
                      "errors": summary["errors"], "egress": len(_egress)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
