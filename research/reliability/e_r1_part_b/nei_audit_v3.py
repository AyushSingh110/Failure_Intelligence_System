"""
E-R1 Part B — fresh v3 NEI audit draw (PROTOCOL_v3.md §3) and the before/after view.

    python research/reliability/e_r1_part_b/nei_audit_v3.py

Pool: v3 NEI items in smoke + calibration, minus the 50 ids audited under v2.
Writes the drawn ids (no text) to manifest/nei_audit_v3_sample.json and prints the
items for review. Never opens test files.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V2, V3 = REPO / "research/reliability/data/fever/derived", REPO / "research/reliability/data/fever/derived_v3"
MAN = REPO / "research/reliability/e_r1_part_b/manifest"


def load(d: Path) -> dict:
    out = {}
    for name in ("smoke.jsonl", "calibration.jsonl"):
        for l in (d / name).read_text(encoding="utf-8").splitlines():
            it = json.loads(l)
            if it["label"] == "NOT ENOUGH INFO":
                out[it["item_id"]] = it
    return out


v2, v3 = load(V2), load(V3)
prior = set(json.loads((MAN / "nei_audit_sample.json").read_text())["sample"])
pool = sorted(k for k in v3 if k not in prior)
draw = random.Random("20261027:nei-audit-v3").sample(pool, min(50, len(pool)))
(MAN / "nei_audit_v3_sample.json").write_text(json.dumps(
    {"pool": len(pool), "v3_nei_smoke_calibration": len(v3), "excluded_prior_audit_ids": len(prior),
     "denominator": len(draw), "sample": draw}, indent=1), encoding="utf-8")

print(f"POOL {len(pool)}  DRAW {len(draw)}\n")
for n, k in enumerate(draw, 1):
    it = v3[k]
    print(f"[{n:02d}] {k} | page={it['related_page']} | span={it['matched_span']} | score={it['selection_score']}"
          f" | skipped={it['artefact_lines_skipped']}\n  CLAIM: {it['claim']}\n  PASSAGE: {it['passage_text']}\n")

print("\n==== BEFORE/AFTER: items flagged under v2 ====\n")
for k in ["calibration-52245", "calibration-165264", "calibration-71613", "calibration-72072",
          "calibration-152285", "calibration-196737", "smoke-119799", "calibration-42725",
          "calibration-135648", "calibration-170955"]:
    a, b = v2.get(k), v3.get(k)
    print(f"{k}\n  CLAIM: {a['claim']}\n  v2: {a['related_page']} :: {a['passage_text'][:200]}\n  v3: "
          + (f"{b['related_page']} (score {b['selection_score']}) :: {b['passage_text'][:200]}" if b else "EXCLUDED (no defensible page)")
          + "\n")
