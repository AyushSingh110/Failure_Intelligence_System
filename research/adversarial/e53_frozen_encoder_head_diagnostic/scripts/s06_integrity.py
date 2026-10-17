"""
E53 step 6 — read-only integrity audit, then SHA256SUMS over every generated file.

Checks:
  1. metrics.json equals a fresh recomputation from the saved per-prompt files
  2. thresholds.json equals a recomputation from scores_calibration.csv alone
  3. scores are uncensored (values strictly between 0 and 0.50 exist for every model)
  4. calibration rows are disjoint (normalised text) from training and base evaluation prompts
  5. pinned model artifacts unchanged; git reports no change under fie/, evals/baselines/, CANONICAL
  6. no outbound network attempt was logged
Outputs (exclusive-create): results/integrity_check.json, SHA256SUMS.txt
"""
from __future__ import annotations

import csv
import json
import math
import subprocess

import e53_common as C
import s05_stats


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=C.REPO, capture_output=True, text=True, check=True).stdout


def main() -> None:
    out = {}

    fresh = json.loads(json.dumps(s05_stats.compute(), sort_keys=True, default=C._json_default))
    saved = json.loads((C.RESULTS / "metrics.json").read_text(encoding="utf-8"))
    out["metrics_recomputed_equal"] = fresh == saved

    with open(C.RESULTS / "scores_calibration.csv", encoding="utf-8") as f:
        cal = list(csv.DictReader(f))
    thr = json.loads((C.RESULTS / "thresholds.json").read_text(encoding="utf-8"))
    prim = [c for c in cal if c["pool_primary"] == "1"]
    r_star = sum(float(c["A0"]) >= 0.5 for c in prim) / len(prim)
    ok = abs(r_star - thr["r_star_primary"]) < 1e-15
    for m, t in thr["primary"].items():
        s = sorted((float(c[m]) for c in prim), reverse=True)
        ok &= s[max(1, math.ceil(r_star * len(s) - 1e-12)) - 1] == t
    out["thresholds_recomputed_from_calibration_only"] = bool(ok)

    with open(C.RESULTS / "scores_eval.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models = [k for k in rows[0] if k not in ("item", "suite", "dataset", "variant", "source_idx", "label", "input_sha256")]
    below = {m: sum(0.0 < float(r[m]) < 0.5 for r in rows) for m in models}
    out["uncensored_scores_strictly_between_0_and_0.50"] = below
    out["uncensored"] = all(v > 0 for v in below.values())

    texts = {r["sha256"]: r["text"] for r in C.read_jsonl(C.CACHE / "texts.jsonl")}
    ref = {C._hash(texts[r["text_sha256"]]) for r in C.read_jsonl(C.MANIFESTS / "train_corpus.jsonl")}
    ref |= {C._hash(texts[it["text_sha256"]]) for it in C.read_jsonl(C.MANIFESTS / "eval_items.jsonl")
            if it["variant"] == "base"}
    kept = C.read_jsonl(C.MANIFESTS / "calibration_kept.jsonl")
    out["calibration_exact_overlap_with_train_or_eval"] = sum(k["norm_sha1"] in ref for k in kept)
    out["calibration_max_cosine_to_train_or_eval"] = max(k["max_cos_to_train_or_eval"] for k in kept)

    out["pinned_artifacts_verified"] = C.verify_pinned()
    out["canonical_pointer"] = (C.REPO / "evals" / "baselines" / "CANONICAL").read_text(encoding="utf-8").strip()
    out["git_status_protected_paths"] = git("status", "--porcelain", "--", "fie", "evals/baselines",
                                            "pyproject.toml", "hatch_build.py", ".github", "scripts",
                                            "docs/fie_rebuild_2026").strip()
    out["git_diff_head_protected_paths"] = git("diff", "--stat", "HEAD", "--", "fie", "evals/baselines").strip()
    out["git_head"] = git("rev-parse", "HEAD").strip()
    out["git_branch"] = git("branch", "--show-current").strip()
    out["network_attempts_logged"] = (C.LOGS / "network_attempts.log").exists()

    out["passed"] = bool(out["metrics_recomputed_equal"] and out["thresholds_recomputed_from_calibration_only"]
                         and out["uncensored"] and out["calibration_exact_overlap_with_train_or_eval"] == 0
                         and out["canonical_pointer"] == "BL-0002_pair-v6.3b_fie-70de7053"
                         and not out["git_status_protected_paths"] and not out["git_diff_head_protected_paths"]
                         and not out["network_attempts_logged"])
    C.write_json_new(C.RESULTS / "integrity_check.json", out)
    C.log(f"s06 integrity passed={out['passed']}")


def checksums() -> None:
    files = sorted(p for p in C.HERE.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS.txt" and "__pycache__" not in p.parts
                   and p.relative_to(C.HERE).as_posix() != "logs/RUN_LOG.md")
    lines = [f"{C.sha256_file(p)}  {p.relative_to(C.HERE).as_posix()}" for p in files]
    C.write_new(C.HERE / "SHA256SUMS.txt", "\n".join(lines) + "\n")
    C.log(f"SHA256SUMS.txt written over {len(lines)} files (RUN_LOG.md excluded: append-only)")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["checksums"]:
        checksums()
    else:
        main()
