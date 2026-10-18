"""
E55 step 2 — score with the frozen PAIR v6.3b (E54 scorer, unchanged).

  --phase whole   every item's whole input (V0); fidelity gate vs BL-0002 records and
                  E54 V0 scores (stop rule 1). Exit 2 on failure.
  --phase views   only if the gate passed: score remaining V1 segment texts, then write
                  records/per_prompt.jsonl (every view score, V0 and V1_lse4).
"""
from __future__ import annotations

import json
import sys
import time

import e54_common as C
import e54_views as V

E54 = C.REPO / "research" / "adversarial" / "e54_segment_aware_scoring"


def score_all(shas, texts, label):
    sc = C.Scorer(); out = {}; t0 = time.perf_counter()
    for k, h in enumerate(shas):
        out[h] = sc.score(texts[h])
        if (k + 1) % 5000 == 0:
            print(f"{label} {k + 1}/{len(shas)} {time.perf_counter() - t0:.0f}s", flush=True)
    return out, time.perf_counter() - t0


def phase_whole() -> None:
    items = C.read_jsonl(C.MANIFESTS / "items.jsonl")
    texts = {r["sha256"]: r["text"] for r in C.read_jsonl(C.CACHE / "texts.jsonl")}
    shas = sorted({it["text_sha256"] for it in items})
    scores, el = score_all(shas, texts, "whole")
    C.write_jsonl_new(C.CACHE / "scores_whole.jsonl", [{"sha256": h, "score": scores[h]} for h in shas])
    rec = {}
    for f in ("std.xstest", "std.orbench_hard", "std.jailbreakbench", "std.harmbench", "std.strongreject", "std.sorrybench"):
        for line in open(C.BL0002 / "records" / f"{f}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            rec[(f, r["dataset"], r["source_idx"])] = (r["input_sha256"], r["layer_scores"]["pair_classifier"])
    std = [it for it in items if it.get("suite", "").startswith("std.")]
    dec = val = 0; mx = 0.0
    for it in std:
        sha, r = rec[(it["suite"], it["dataset"], it["source_idx"])]
        assert sha == it["input_sha256"]
        s = scores[it["text_sha256"]]
        dec += (s >= 0.5) != (r > 0)
        if r > 0:
            mx = max(mx, abs(s - r)); val += abs(s - r) > 5e-5 + 1e-12
    e54_items = {it["item"]: it for it in C.read_jsonl(E54 / "manifests" / "items.jsonl")}
    e54_v0 = {}
    for r in C.read_jsonl(E54 / "records" / "per_prompt.jsonl"):
        e54_v0[e54_items[r["item"]]["text_sha256"]] = r["scores"]["V0"]
    shared = [h for h in shas if h in e54_v0]
    d54 = max(abs(scores[h] - e54_v0[h]) for h in shared)
    gate = {"bl0002": {"items": len(std), "decision_mismatches": int(dec), "value_mismatches": int(val), "max_abs": mx},
            "e54_v0": {"shared_texts": len(shared), "max_abs_diff": d54}, "elapsed_s": round(el, 1)}
    gate["passed"] = dec == 0 and val == 0 and d54 <= 1e-9
    C.write_json_new(C.RESULTS / "fidelity_gate.json", gate)
    C.log(f"s02 whole: {len(shas)} texts {el:.0f}s; BL-0002 dec {dec} val {val} max {mx:.1e}; E54 max {d54:.1e} "
          f"over {len(shared)}; passed={gate['passed']}")
    if not gate["passed"]:
        C.log("STOP: fidelity gate failed"); sys.exit(2)


def phase_views() -> None:
    if not json.loads((C.RESULTS / "fidelity_gate.json").read_text(encoding="utf-8"))["passed"]:
        raise SystemExit("STOP: fidelity gate did not pass")
    items = C.read_jsonl(C.MANIFESTS / "items.jsonl")
    views = {r["item"]: r for r in C.read_jsonl(C.MANIFESTS / "views.jsonl")}
    texts = {r["sha256"]: r["text"] for r in C.read_jsonl(C.CACHE / "texts.jsonl")}
    whole = {r["sha256"]: r["score"] for r in C.read_jsonl(C.CACHE / "scores_whole.jsonl")}
    todo = sorted({v["text_sha256"] for r in views.values() for v in r["views"]} - set(whole))
    extra, el = score_all(todo, texts, "views")
    C.write_jsonl_new(C.CACHE / "scores_views.jsonl", [{"sha256": h, "score": extra[h]} for h in todo])
    sc = {**whole, **extra}
    rows = []
    for it in items:
        vr = views[it["item"]]
        vl = [{**v, "score": sc[v["text_sha256"]]} for v in vr["views"]]
        trip = [(v["kind"], v["text_sha256"], v["score"]) for v in vl]
        rows.append({**{k: it[k] for k in ("item", "partition", "dataset", "domain", "variant", "unit", "label")},
                     "family": it.get("family"), "category": it.get("category"),
                     "n_tokens": vr["n_tokens"], "unique_views_V1": vr["unique_views_V1"],
                     "raw_views_V1": vr["raw_views_V1"], "views": vl,
                     "scores": {"V0": sc[it["text_sha256"]],
                                "V1_lse4": V.aggregate(trip, ("whole", "seg"), "lse4")}})
    C.write_jsonl_new(C.RECORDS / "per_prompt.jsonl", rows)
    C.log(f"s02 views: {len(todo)} extra texts {el:.0f}s; records {len(rows)}")


if __name__ == "__main__":
    {"whole": phase_whole, "views": phase_views}[sys.argv[sys.argv.index("--phase") + 1]]()
