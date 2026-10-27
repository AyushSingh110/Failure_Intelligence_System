"""
E-R1 Part B — select the 50-item NEI noise-audit sample (PROTOCOL.md §4).

    python -I research/reliability/e_r1_part_b/nei_noise_audit_sample.py

Draws from smoke + calibration ONLY (never test), seed 20261027, sorted by item id,
and prints claim / related page / passage for review. Writes the selected item ids
(no text) to manifest/nei_audit_sample.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DERIVED = REPO / "research/reliability/data/fever/derived"
OUT = REPO / "research/reliability/e_r1_part_b/manifest/nei_audit_sample.json"

items = []
for name in ("smoke.jsonl", "calibration.jsonl"):
    items += [json.loads(l) for l in (DERIVED / name).read_text(encoding="utf-8").splitlines()]
nei = sorted((i for i in items if i["label"] == "NOT ENOUGH INFO"), key=lambda i: i["item_id"])
chosen = random.Random("20261027:nei-audit").sample(nei, 50)
OUT.write_text(json.dumps({"pool": len(nei), "sample": [i["item_id"] for i in chosen]}, indent=1), encoding="utf-8")
for n, i in enumerate(chosen, 1):
    print(f"[{n:02d}] {i['item_id']} | page={i['related_page']}\n  CLAIM: {i['claim']}\n  PASSAGE: {i['passage_text']}\n")
