"""
E54 step 2 — score with the frozen PAIR v6.3b.

  --phase whole   score every item's whole input (V0); run the fidelity gate against
                  BL-0002 records and E53's SHIPPED scores. Exit 2 on failure.
  --phase views   only if the gate passed: score every remaining unique view text, then
                  write records/per_prompt.jsonl with every view score and every aggregate.

Outputs (exclusive-create): cache/scores_whole.jsonl, results/fidelity_gate.json,
cache/scores_views.jsonl, records/per_prompt.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
import time

import e54_common as C
import e54_views as V


def score_all(shas, texts, label):
    sc = C.Scorer()
    out, t0 = {}, time.perf_counter()
    for k, h in enumerate(shas):
        out[h] = sc.score(texts[h])
        if (k + 1) % 2000 == 0:
            print(f"{label} {k + 1}/{len(shas)} {time.perf_counter() - t0:.0f}s", flush=True)
    return out, time.perf_counter() - t0


def phase_whole() -> None:
    C.verify_e53()
    items = C.read_jsonl(C.MANIFESTS / "items.jsonl")
    texts = {r["sha256"]: r["text"] for r in C.read_jsonl(C.CACHE / "texts.jsonl")}
    shas = sorted({it["text_sha256"] for it in items})
    scores, el = score_all(shas, texts, "whole")
    C.write_jsonl_new(C.CACHE / "scores_whole.jsonl", [{"sha256": h, "score": scores[h]} for h in shas])

    # gate 1: BL-0002 PAIR layer records on the standard base items
    rec = {}
    for f in ("std.xstest", "std.orbench_hard", "std.jailbreakbench", "std.harmbench", "std.strongreject", "std.sorrybench"):
        for line in open(C.BL0002 / "records" / f"{f}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            rec[(f, r["dataset"], r["source_idx"])] = (r["input_sha256"], r["layer_scores"]["pair_classifier"])
    std = [it for it in items if it.get("suite", "").startswith("std.")]
    dec_mm = val_mm = 0
    max_d = 0.0
    for it in std:
        sha, r = rec[(it["suite"], it["domain"], it["source_idx"])]
        assert sha == it["input_sha256"]
        s = scores[it["text_sha256"]]
        dec_mm += (s >= 0.5) != (r > 0)
        if r > 0:
            max_d = max(max_d, abs(s - r))
            val_mm += abs(s - r) > 5e-5 + 1e-12
    # gate 2: E53 SHIPPED scores (same encoder path) on every shared input
    with open(C.E53 / "results" / "scores_eval.csv", encoding="utf-8") as f:
        e53 = {r["input_sha256"]: float(r["SHIPPED"]) for r in csv.DictReader(f)}
    shared = [it for it in items if it.get("input_sha256") in e53]
    d53 = max(abs(scores[it["text_sha256"]] - e53[it["input_sha256"]]) for it in shared)
    gate = {"bl0002": {"items": len(std), "decision_mismatches": int(dec_mm), "value_mismatches_gt_5e-5": int(val_mm),
                       "max_abs_diff_4dp": max_d},
            "e53_shipped": {"items": len(shared), "max_abs_diff": d53},
            "whole_texts_scored": len(shas), "elapsed_s": round(el, 1)}
    gate["passed"] = dec_mm == 0 and val_mm == 0 and d53 <= 1e-9
    C.write_json_new(C.RESULTS / "fidelity_gate.json", gate)
    C.log(f"s02 whole: {len(shas)} texts in {el:.0f}s; gate BL-0002 dec mm {dec_mm}, val mm {val_mm}, max {max_d:.1e}; "
          f"E53 max |d| {d53:.1e} over {len(shared)} items; passed={gate['passed']}")
    if not gate["passed"]:
        C.log("STOP: V0 fidelity gate failed; no candidate view scored")
        sys.exit(2)


def phase_views() -> None:
    gate = json.loads((C.RESULTS / "fidelity_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise SystemExit("STOP: fidelity gate did not pass")
    items = C.read_jsonl(C.MANIFESTS / "items.jsonl")
    views = {r["item"]: r for r in C.read_jsonl(C.MANIFESTS / "views.jsonl")}
    texts = {r["sha256"]: r["text"] for r in C.read_jsonl(C.CACHE / "texts.jsonl")}
    whole = {r["sha256"]: r["score"] for r in C.read_jsonl(C.CACHE / "scores_whole.jsonl")}
    todo = sorted({v["text_sha256"] for r in views.values() for v in r["views"]} - set(whole))
    extra, el = score_all(todo, texts, "views")
    C.write_jsonl_new(C.CACHE / "scores_views.jsonl", [{"sha256": h, "score": extra[h]} for h in todo])
    allsc = {**whole, **extra}

    rows = []
    for it in items:
        vr = views[it["item"]]
        vlist = [{**v, "score": allsc[v["text_sha256"]]} for v in vr["views"]]
        triples = [(v["kind"], v["text_sha256"], v["score"]) for v in vlist]
        agg = {"V0": allsc[it["text_sha256"]]}
        for var, kinds in V.VARIANTS.items():
            for rule in V.RULES:
                agg[f"{var}_{rule}"] = V.aggregate(triples, kinds, rule)
        rows.append({**{k: it[k] for k in ("item", "partition", "dataset", "domain", "variant", "unit", "label")},
                     "n_tokens": vr["n_tokens"], "raw_views": vr["raw_views"], "unique_views": vr["unique_views"],
                     "views": vlist, "scores": agg})
    C.write_jsonl_new(C.RECORDS / "per_prompt.jsonl", rows)
    C.log(f"s02 views: {len(todo)} extra view texts in {el:.0f}s; per-prompt records for {len(rows)} items")


if __name__ == "__main__":
    {"whole": phase_whole, "views": phase_views}[sys.argv[sys.argv.index("--phase") + 1]]()
