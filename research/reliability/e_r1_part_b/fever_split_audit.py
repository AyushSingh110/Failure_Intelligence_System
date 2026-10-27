"""
E-R1 Part B, Phase 2 — integrity and split audit of the FEVER claim files.

    python research/reliability/e_r1_part_b/fever_split_audit.py

Reads research/reliability/data/fever/raw/{paper_dev,paper_test}.jsonl (git-ignored)
and writes AGGREGATES ONLY (no claim text) to
research/reliability/e_r1_part_b/manifest/split_audit.json. Standard library only.
No network.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "research/reliability/data/fever/raw"
OUT = REPO / "research/reliability/e_r1_part_b/manifest/split_audit.json"
FILES = ("paper_dev.jsonl", "paper_test.jsonl")
REQUIRED = ("id", "verifiable", "label", "claim", "evidence")
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def norm_claim(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def audit_file(path: Path) -> tuple[dict, list[dict]]:
    data = path.read_bytes()
    rows, errors = [], []
    for i, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            errors.append({"line": i, "error": "blank"})
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append({"line": i, "error": f"json: {exc.msg}"})
    missing = collections.Counter(k for r in rows for k in REQUIRED if k not in r)
    labels = collections.Counter(r.get("label") for r in rows)
    ids = collections.Counter(r.get("id") for r in rows)
    claims = collections.Counter(norm_claim(r.get("claim", "")) for r in rows)

    struct = collections.Counter()
    sets_per_claim, sents_per_set, pages_per_set = collections.Counter(), collections.Counter(), collections.Counter()
    label_verifiable = collections.Counter()
    for r in rows:
        label_verifiable[(r.get("label"), r.get("verifiable"))] += 1
        ev = r.get("evidence")
        if not isinstance(ev, list) or not ev:
            struct["evidence_not_nonempty_list"] += 1
            continue
        sets_per_claim[min(len(ev), 10)] += 1
        for ev_set in ev:
            if not isinstance(ev_set, list) or not ev_set:
                struct["evidence_set_not_nonempty_list"] += 1
                continue
            sents_per_set[min(len(ev_set), 10)] += 1
            pages_per_set[min(len({it[2] for it in ev_set if isinstance(it, list) and len(it) == 4}), 10)] += 1
            for it in ev_set:
                if not (isinstance(it, list) and len(it) == 4):
                    struct["item_not_4_list"] += 1
                elif r.get("label") == "NOT ENOUGH INFO":
                    struct["nei_item_null_page" if it[2] is None and it[3] is None else "nei_item_with_page"] += 1
                else:
                    ok = isinstance(it[2], str) and it[2] and isinstance(it[3], int) and it[3] >= 0
                    struct["gold_item_ok" if ok else "gold_item_bad"] += 1
    stats = {
        "file": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "md5": hashlib.md5(data).hexdigest(),
        "lines": data.count(b"\n"),
        "rows_parsed": len(rows),
        "parse_errors": errors[:20],
        "missing_required_fields": dict(missing),
        "label_counts": {k: labels.get(k, 0) for k in LABELS},
        "other_labels": {str(k): v for k, v in labels.items() if k not in LABELS},
        "label_x_verifiable": {f"{k[0]}|{k[1]}": v for k, v in sorted(label_verifiable.items(), key=str)},
        "duplicate_ids": sum(1 for v in ids.values() if v > 1),
        "duplicate_normalized_claims": {"distinct_texts_repeated": sum(1 for v in claims.values() if v > 1),
                                        "rows_in_repeats": sum(v for v in claims.values() if v > 1)},
        "evidence_structure": dict(struct),
        "evidence_sets_per_claim (capped 10)": dict(sorted(sets_per_claim.items())),
        "sentences_per_set (capped 10)": dict(sorted(sents_per_set.items())),
        "pages_per_set (capped 10)": dict(sorted(pages_per_set.items())),
    }
    return stats, rows


def pages_of(rows: list[dict]) -> set[str]:
    return {it[2] for r in rows if r.get("label") != "NOT ENOUGH INFO"
            for s in r.get("evidence", []) for it in s if isinstance(it, list) and len(it) == 4 and it[2]}


def main() -> int:
    per_file, rows = {}, {}
    for name in FILES:
        per_file[name], rows[name] = audit_file(RAW / name)
    dev, test = rows["paper_dev.jsonl"], rows["paper_test.jsonl"]

    dev_ids, test_ids = {r["id"] for r in dev}, {r["id"] for r in test}
    dev_claims, test_claims = {norm_claim(r["claim"]) for r in dev}, {norm_claim(r["claim"]) for r in test}
    dev_pages, test_pages = pages_of(dev), pages_of(test)
    shared_pages = dev_pages & test_pages
    test_gold_rows_touching_dev_pages = sum(
        1 for r in test if r["label"] != "NOT ENOUGH INFO"
        and any(it[2] in dev_pages for s in r["evidence"] for it in s if len(it) == 4 and it[2]))
    test_gold_rows = sum(1 for r in test if r["label"] != "NOT ENOUGH INFO")

    all_ids = sorted(dev_ids | test_ids)
    report = {
        "schema": "e-r1-partB-split-audit/1",
        "files": per_file,
        "concatenation_check": {
            "sum_bytes": per_file["paper_dev.jsonl"]["bytes"] + per_file["paper_test.jsonl"]["bytes"],
            "zenodo_shared_task_dev_bytes": 4349935,
            "sum_rows": len(dev) + len(test),
            "documented_shared_task_dev_rows": 19998,
        },
        "overlap": {
            "claim_ids": len(dev_ids & test_ids),
            "normalized_claim_texts": len(dev_claims & test_claims),
            "gold_evidence_pages_dev": len(dev_pages),
            "gold_evidence_pages_test": len(test_pages),
            "gold_evidence_pages_shared": len(shared_pages),
            "test_gold_rows": test_gold_rows,
            "test_gold_rows_citing_a_page_also_cited_in_dev": test_gold_rows_touching_dev_pages,
        },
        "id_ranges": {
            "paper_dev": [min(dev_ids), max(dev_ids)], "paper_test": [min(test_ids), max(test_ids)],
            "interleaved": not (max(dev_ids) < min(test_ids) or max(test_ids) < min(dev_ids)),
            "union_size": len(all_ids),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("concatenation_check", "overlap", "id_ranges")}, indent=1))
    for name, s in per_file.items():
        print(name, s["rows_parsed"], s["label_counts"], "dupIDs", s["duplicate_ids"],
              "dupClaims", s["duplicate_normalized_claims"], s["evidence_structure"], s["parse_errors"][:2],
              s["missing_required_fields"], s["other_labels"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
