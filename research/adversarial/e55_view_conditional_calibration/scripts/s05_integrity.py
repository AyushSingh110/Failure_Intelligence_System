"""
E55 step 5 — read-only integrity audit, then SHA256SUMS (`checksums`).

  1. metrics.json and thresholds.json equal a fresh recomputation from records/per_prompt.jsonl
  2. thresholds use calibration rows only (recomputed with evaluation rows removed)
  3. V0 per-prompt equals the whole-score cache; every view scored; scores uncensored
  4. vendored E54 scorer files identical to E54's; pinned digests; E53/E54 checksums verify
  5. git: HEAD unchanged, protected paths clean; no network attempt logged
"""
from __future__ import annotations

import json
import subprocess
import sys

import e54_common as C
import s03_metrics

E54 = C.REPO / "research" / "adversarial" / "e54_segment_aware_scoring"
PROTECTED = ["fie", "evals", "scripts", ".github", "docs", "pyproject.toml", "hatch_build.py", "data",
             "research/adversarial/e53_frozen_encoder_head_diagnostic", "research/adversarial/e54_segment_aware_scoring"]


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=C.REPO, capture_output=True, text=True, check=True).stdout


def main() -> None:
    out = {}
    fresh = json.loads(json.dumps(s03_metrics.compute(), sort_keys=True, default=C._json_default))
    out["metrics_recomputed_equal"] = fresh == json.loads((C.RESULTS / "metrics.json").read_text(encoding="utf-8"))
    out["thresholds_file_equal"] = fresh["thresholds"] == json.loads((C.RESULTS / "thresholds.json").read_text(encoding="utf-8"))
    rows = s03_metrics.load()
    cal_only = json.loads(json.dumps(s03_metrics.thresholds([r for r in rows if r["partition"] == "cal_benign"]),
                                     sort_keys=True, default=C._json_default))
    out["thresholds_from_calibration_only"] = cal_only == fresh["thresholds"]
    whole = {r["sha256"]: r["score"] for r in C.read_jsonl(C.CACHE / "scores_whole.jsonl")}
    items = {it["item"]: it for it in C.read_jsonl(C.MANIFESTS / "items.jsonl")}
    out["v0_equals_whole_cache"] = all(r["scores"]["V0"] == whole[items[r["item"]]["text_sha256"]] for r in rows)
    out["every_view_scored"] = all(isinstance(v.get("score"), float) for r in rows for v in r["views"])
    out["below_0.50_counts"] = {m: sum(0.0 < r["scores"][m] < 0.5 for r in rows) for m in ("V0", "V1_lse4")}
    out["uncensored"] = all(v > 0 for v in out["below_0.50_counts"].values())
    out["vendored_scorer_identical"] = all(C.sha256_file(C.HERE / "scripts" / f) == C.sha256_file(E54 / "scripts" / f)
                                           for f in ("e54_views.py", "e54_common.py"))
    out["pinned"] = C.verify_pinned()
    C.verify_e53()
    for line in (E54 / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        d, rel = line.split("  ", 1)
        if C.sha256_file(E54 / rel) != d:
            raise SystemExit(f"STOP: E54 artifact changed: {rel}")
    out["e53_e54_checksums_verified"] = True
    out["git_head"] = git("rev-parse", "HEAD").strip()
    out["git_branch"] = git("branch", "--show-current").strip()
    out["git_status_protected"] = git("status", "--porcelain", "--", *PROTECTED).strip()
    out["git_diff_head_protected"] = git("diff", "--stat", "HEAD", "--", *PROTECTED).strip()
    out["canonical_pointer"] = (C.REPO / "evals/baselines/CANONICAL").read_text(encoding="utf-8").strip()
    out["network_attempts_logged"] = (C.LOGS / "network_attempts.log").exists()
    out["passed"] = bool(out["metrics_recomputed_equal"] and out["thresholds_file_equal"]
                         and out["thresholds_from_calibration_only"] and out["v0_equals_whole_cache"]
                         and out["every_view_scored"] and out["uncensored"] and out["vendored_scorer_identical"]
                         and out["git_head"] == "29bd6e2b99344e00c342f0aa6eb2174934446db9"
                         and not out["git_status_protected"] and not out["git_diff_head_protected"]
                         and out["canonical_pointer"] == "BL-0002_pair-v6.3b_fie-70de7053"
                         and not out["network_attempts_logged"])
    C.write_json_new(C.RESULTS / "integrity_check.json", out)
    C.log(f"s05 integrity passed={out['passed']}")


def checksums() -> None:
    files = sorted(p for p in C.HERE.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"
                   and "__pycache__" not in p.parts and p.relative_to(C.HERE).as_posix() != "logs/RUN_LOG.md")
    C.write_new(C.HERE / "SHA256SUMS.txt", "".join(f"{C.sha256_file(p)}  {p.relative_to(C.HERE).as_posix()}\n" for p in files))
    C.log(f"SHA256SUMS.txt written over {len(files)} files (append-only RUN_LOG.md excluded)")


if __name__ == "__main__":
    checksums() if sys.argv[1:] == ["checksums"] else main()
