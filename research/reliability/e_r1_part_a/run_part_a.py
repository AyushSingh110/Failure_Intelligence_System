"""
E-R1 Part A runner: the 17 decision-contract fixtures under the evals/hermetic.py guard.

    python research/reliability/e_r1_part_a/run_part_a.py [--out DIR]

The parent process extracts the pre-D-060 route source (git object, B0), then
starts a child interpreter with a sanitized environment (no provider keys, dead
proxies, telemetry off, Hugging Face offline). The child arms the hermetic guard
BEFORE importing any FIE module, runs every case against the current route (B2)
and the B0 route, and writes one JSON document. The run fails if the guard
recorded any denied event, or if any B2 case shows a forbidden outcome.

No network, no model inference, no external service. The fixture cases are the
ones in tests/reliability/e_r1_part_a.py; pytest runs the same cases.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEGACY_COMMIT = "f0888bb270692433988a15e93e95262fc6488796"
SCHEMA = "e-r1-part-a/1"

CHILD = r'''
import json, os, sys
repo, legacy_path, out_path, log_path = sys.argv[1:5]
sys.path[:0] = [repo, os.path.join(repo, "tests")]
os.chdir(repo)
from evals import hermetic
hermetic.install(mode="worker", event_log=log_path)
selftest = hermetic.selftest()

import logging
logging.disable(logging.CRITICAL)
from reliability.e_r1_part_a import CASES, observe
from reliability.harness import load_route_from_source
legacy = load_route_from_source(open(legacy_path, encoding="utf-8").read(), "legacy_monitor_b0")

rows = []
for case in CASES:
    row = {"id": case.id, "fixture": case.fixture, "title": case.title,
           "unknown_input": case.unknown_input, "source_ref": case.source_ref,
           "b0_expected": sorted(case.b0_expected)}
    for label, route in (("B2", None), ("B0", legacy)):
        try:
            row[label] = observe(case) if case.function is not None else observe(case, route)
        except Exception as exc:
            row[label] = {"error": f"{type(exc).__name__}: {exc}", "violations": ["error"]}
    rows.append(row)

json.dump({"selftest": selftest, "rows": rows, "guard": hermetic.summary()},
          open(out_path, "w", encoding="utf-8"), indent=1, default=str)
'''


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "research/reliability/e_r1_part_a/results"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO))
    from evals.hermetic import sanitized_env

    with tempfile.TemporaryDirectory(prefix="e-r1-part-a-") as scratch:
        scratch = Path(scratch)
        legacy = scratch / "legacy_monitor.py"
        legacy.write_text(_git("show", f"{LEGACY_COMMIT}:app/routes/monitor.py"), encoding="utf-8")
        child_out, child_log = scratch / "child.json", scratch / "guard_events.jsonl"
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        env = sanitized_env(os.environ, {
            "FIE_NO_TELEMETRY": "1", "FIE_NO_AUTO_DOWNLOAD": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": hf_home,
            "GROQ_API_KEY": "", "SERPER_API_KEY": "", "SERPER_ENABLED": "false",
            "SENDGRID_API_KEY": "", "NOTIFICATION_EMAIL": "", "MONGODB_URI": "",
            "OLLAMA_ENABLED": "false", "DEBUG": "false",
        }, scratch)
        proc = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, str(REPO), str(legacy), str(child_out), str(child_log)],
            cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", timeout=1800,
        )
        if proc.returncode != 0 or not child_out.exists():
            sys.stderr.write(proc.stderr[-4000:])
            return 3
        result = json.loads(child_out.read_text(encoding="utf-8"))

    rows = result["rows"]
    b2_fail = [r["id"] for r in rows
               if (r["B2"].get("violations") if r["fixture"] != "A17"
                   else ([] if r["B2"].get("contract_state") == "NOT_CHECKED" else ["contract_state"]))]
    b0_mismatch = [r["id"] for r in rows if sorted(r["B0"].get("violations", [])) != r["b0_expected"]]
    doc = {
        "schema": SCHEMA,
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": _git("rev-parse", "HEAD").strip(),
        "working_tree_dirty": bool(_git("status", "--porcelain").strip()),
        "b0_route_source": f"{LEGACY_COMMIT}:app/routes/monitor.py",
        "python": sys.version.split()[0],
        "guard": result["guard"],
        "guard_selftest": result["selftest"],
        "cases": len(rows),
        "fixtures": len({r["fixture"] for r in rows}),
        "b2_cases_with_forbidden_outcome": b2_fail,
        "b0_cases_not_matching_documented_defect": b0_mismatch,
        "rows": rows,
    }
    stamp = doc["run_at_utc"].replace(":", "").replace("-", "")
    path = out_dir / f"{stamp}_part_a.json"
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")

    ok = result["guard"]["violations"] == 0 and not b2_fail
    print(f"{path}\nguard violations={result['guard']['violations']}  cases={len(rows)}  "
          f"B2 forbidden={b2_fail or 0}  B0 mismatches={b0_mismatch or 0}  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
