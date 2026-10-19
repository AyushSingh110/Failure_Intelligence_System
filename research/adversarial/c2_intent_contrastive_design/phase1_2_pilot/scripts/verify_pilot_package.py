"""
C2 Phase 1.2 — read-only integrity verification of the pilot package (spec section 9).
Output (exclusive-create): logs/integrity_check.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PILOT = Path(__file__).resolve().parent.parent
C2 = PILOT.parent
REPO = C2.parents[2]
sys.dont_write_bytecode = True
ANNOT = ["reviewer_id", "M_manipulation", "H_harmful_intent", "C_context", "Q_confidence", "not_a_user_input",
         "toxic_only", "same_topic_pair_note", "notes", "timestamp_utc"]
FORBIDDEN_COLUMNS = {"label", "original_label", "source", "category", "stratum", "score", "pair_prob"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    out = {}
    a = read_csv(PILOT / "reviewer_packets/reviewer_a.csv")
    b = read_csv(PILOT / "reviewer_packets/reviewer_b.csv")
    m = json.loads((PILOT / "owner_only/pilot_mapping.json").read_text(encoding="utf-8"))
    man = json.loads((PILOT / "sampling_manifest.json").read_text(encoding="utf-8"))
    items = m["items"]
    out["packet_rows"] = {"a": len(a), "b": len(b)}
    out["same_pilot_ids"] = {r["pilot_id"] for r in a} == {r["pilot_id"] for r in b} == set(items)
    ua = {items[r["pilot_id"]]["text_sha256"] for r in a}; ub = {items[r["pilot_id"]]["text_sha256"] for r in b}
    out["unique_underlying_examples"] = {"a": len(ua), "b": len(ub)}
    out["same_200_examples"] = ua == ub and len(ua) == 200
    out["orders_differ"] = [r["pilot_id"] for r in a] != [r["pilot_id"] for r in b]
    out["repeat_gap_min"] = {k: min(abs(int(items[p][f"position_{k}"]) - int(items[v["repeat_of"]][f"position_{k}"]))
                                    for p, v in items.items() if v["repeat_of"]) for k in ("a", "b")}
    cols = set(a[0]) | set(b[0])
    out["packet_columns"] = sorted(cols)
    out["no_label_score_or_source_columns"] = not (cols & FORBIDDEN_COLUMNS)
    out["all_annotations_blank"] = all(r[c] == "" for r in a + b for c in ANNOT)
    out["packets_text_matches_mapping_hash"] = all(
        hashlib.sha256(r["text"].encode("utf-8")).hexdigest() == items[r["pilot_id"]]["text_sha256"] for r in a + b)
    # mapping recovers the original row from the original file
    ok = True
    for v in items.values():
        line = (REPO / v["file"]).read_text(encoding="utf-8").splitlines()[v["line"]]
        ok &= hashlib.sha256(json.loads(line)["prompt"].strip().encode("utf-8")).hexdigest() == v["text_sha256"]
    out["mapping_recovers_original_rows"] = bool(ok)
    out["original_files_unchanged"] = all(sha(REPO / f) == h for f, h in man["original_file_sha256"].items())
    out["no_other_reviewer_answers_or_scores_in_packets"] = out["all_annotations_blank"] and out["no_label_score_or_source_columns"]
    # prior evidence intact
    sums = []
    for d in sorted((REPO / "research/adversarial").glob("e5*")) + [C2]:
        for name in ("SHA256SUMS.txt", "SHA256SUMS_PHASE1_1.txt"):
            f = d / name
            if f.exists():
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    h, rel = line.split(None, 1)          # "hash  path" or "hash *path" (binary mode)
                    sums.append(sha(d / rel.lstrip("*").strip()) == h)
    out["prior_checksums_verified"] = all(sums) and len(sums) > 0
    out["model_digests"] = {p: sha(REPO / p)[:16] for p in ("fie/models/pair_intent_classifier_v6_3b.pkl", "fie/models/minilm-onnx/model.onnx")}
    out["canonical_pointer"] = (REPO / "evals/baselines/CANONICAL").read_text(encoding="utf-8").strip()
    out["git_head"] = git("rev-parse", "HEAD")
    out["git_status_protected"] = git("status", "--porcelain", "--", "fie", "evals", "data", "scripts", ".github", "docs",
                                      "pyproject.toml", "hatch_build.py", "research/adversarial/e53_frozen_encoder_head_diagnostic",
                                      "research/adversarial/e54_segment_aware_scoring", "research/adversarial/e55_view_conditional_calibration")
    out["annotations_returned_empty"] = not any((PILOT / "annotations_returned").iterdir())
    out["passed"] = bool(out["same_pilot_ids"] and out["same_200_examples"] and out["orders_differ"]
                         and min(out["repeat_gap_min"].values()) >= 50 and out["no_label_score_or_source_columns"]
                         and out["all_annotations_blank"] and out["packets_text_matches_mapping_hash"]
                         and out["mapping_recovers_original_rows"] and out["original_files_unchanged"]
                         and out["prior_checksums_verified"] and out["canonical_pointer"] == "BL-0002_pair-v6.3b_fie-70de7053"
                         and out["model_digests"]["fie/models/pair_intent_classifier_v6_3b.pkl"] == "9c682b285a9fa205"
                         and not out["git_status_protected"] and out["annotations_returned_empty"])
    with open(PILOT / "logs/integrity_check.json", "x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
