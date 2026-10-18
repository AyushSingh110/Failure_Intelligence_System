"""
E54 step 5 — read-only integrity audit, then SHA256SUMS.

  1. metrics.json equals a fresh recomputation from records/per_prompt.jsonl
  2. calibrated thresholds recompute from the calibration partition alone
  3. per-prompt V0 equals the whole-input score cache; every view carries a score
  4. scores are uncensored (values strictly between 0 and 0.50 exist)
  5. pinned digests unchanged; E53 checksums verify; git shows no change to protected paths
  6. no outbound network attempt was logged
Outputs (exclusive-create): results/integrity_check.json, SHA256SUMS.txt (with `checksums`)
"""
from __future__ import annotations

import json
import subprocess
import sys

import e54_common as C
import s03_metrics

PROTECTED = ["fie", "evals", "scripts", ".github", "docs/fie_rebuild_2026", "pyproject.toml", "hatch_build.py",
             "research/adversarial/e53_frozen_encoder_head_diagnostic"]


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=C.REPO, capture_output=True, text=True, check=True).stdout


def main() -> None:
    out = {}
    fresh = json.loads(json.dumps(s03_metrics.compute(), sort_keys=True, default=C._json_default))
    out["metrics_recomputed_equal"] = fresh == json.loads((C.RESULTS / "metrics.json").read_text(encoding="utf-8"))
    rows = s03_metrics.load()
    out["thresholds_from_calibration_only"] = fresh["thresholds"] == s03_metrics.thresholds(
        [r for r in rows if r["partition"] == "cal_benign"])
    whole = {r["sha256"]: r["score"] for r in C.read_jsonl(C.CACHE / "scores_whole.jsonl")}
    items = {it["item"]: it for it in C.read_jsonl(C.MANIFESTS / "items.jsonl")}
    out["v0_equals_whole_cache"] = all(r["scores"]["V0"] == whole[items[r["item"]]["text_sha256"]] for r in rows)
    out["every_view_scored"] = all(isinstance(v.get("score"), float) for r in rows for v in r["views"])
    out["views_total"] = sum(len(r["views"]) for r in rows)
    below = {m: sum(0.0 < r["scores"][m] < 0.5 for r in rows) for m in s03_metrics.models()}
    out["scores_strictly_between_0_and_0.50"] = below
    out["uncensored"] = all(v > 0 for v in below.values())
    out["pinned"] = C.verify_pinned()
    C.verify_e53()
    out["e53_checksums_verified"] = True
    out["git_head"] = git("rev-parse", "HEAD").strip()
    out["git_branch"] = git("branch", "--show-current").strip()
    out["git_status_protected"] = git("status", "--porcelain", "--", *PROTECTED).strip()
    out["git_diff_head_protected"] = git("diff", "--stat", "HEAD", "--", *PROTECTED).strip()
    out["canonical_pointer"] = (C.REPO / "evals/baselines/CANONICAL").read_text(encoding="utf-8").strip()
    out["network_attempts_logged"] = (C.LOGS / "network_attempts.log").exists()
    out["passed"] = bool(out["metrics_recomputed_equal"] and out["thresholds_from_calibration_only"]
                         and out["v0_equals_whole_cache"] and out["every_view_scored"] and out["uncensored"]
                         and out["git_head"] == "700db5fc9849eea30bd2f45b2a30688ee76b5f35"
                         and not out["git_status_protected"] and not out["git_diff_head_protected"]
                         and out["canonical_pointer"] == "BL-0002_pair-v6.3b_fie-70de7053"
                         and not out["network_attempts_logged"])
    C.write_json_new(C.RESULTS / "integrity_check.json", out)
    C.log(f"s05 integrity passed={out['passed']}")


def checksums() -> None:
    files = sorted(p for p in C.HERE.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"
                   and "__pycache__" not in p.parts and p.relative_to(C.HERE).as_posix() != "logs/RUN_LOG.md")
    C.write_new(C.HERE / "SHA256SUMS.txt",
                "".join(f"{C.sha256_file(p)}  {p.relative_to(C.HERE).as_posix()}\n" for p in files))
    C.log(f"SHA256SUMS.txt written over {len(files)} files (append-only RUN_LOG.md excluded)")


if __name__ == "__main__":
    checksums() if sys.argv[1:] == ["checksums"] else main()
