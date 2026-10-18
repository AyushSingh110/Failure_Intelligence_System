"""
E55 step 4 — latency and views for V0 and V1 (the conditional rule adds no scoring cost:
it only chooses which threshold to compare against). No cache; 30 evaluation items per
token bucket, evenly spaced by position; one warm-up; one timed pass per variant.
Output (exclusive-create): results/latency.json
"""
from __future__ import annotations

import time

import e54_common as C
import e54_views as V

BUCKETS = [("<=64", 0, 64), ("65-256", 65, 256), ("257-512", 257, 512), (">512", 513, 10**9)]


def main() -> None:
    import numpy as np
    items = [it for it in C.read_jsonl(C.MANIFESTS / "items.jsonl") if it["partition"] == "eval"]
    views = {r["item"]: r for r in C.read_jsonl(C.MANIFESTS / "views.jsonl")}
    texts = {r["sha256"]: r["text"] for r in C.read_jsonl(C.CACHE / "texts.jsonl")}
    sc = C.Scorer(); sc.score("warm-up text for the encoder")
    out = {}
    for name, lo, hi in BUCKETS:
        pool = [it for it in items if lo <= views[it["item"]]["n_tokens"] <= hi]
        if not pool:
            out[name] = {"n": 0}; continue
        pick = [pool[int(i)] for i in np.unique(np.linspace(0, len(pool) - 1, min(30, len(pool))).round().astype(int))]
        res = {}
        for var, kinds in (("V0", ("whole",)), ("V1", ("whole", "seg"))):
            times, calls = [], []
            for it in pick:
                t0 = time.perf_counter()
                vs, _ = V.build_views(texts[it["text_sha256"]])
                seen = {}
                for v in vs:
                    if v["kind"] in kinds and v["text"] not in seen:
                        seen[v["text"]] = sc.score(v["text"])
                if var == "V1":
                    V.aggregate([(v["kind"], v["text"], seen[v["text"]]) for v in vs if v["kind"] in kinds], kinds, "lse4")
                times.append((time.perf_counter() - t0) * 1000); calls.append(len(seen))
            res[var] = {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                        "mean_encoder_calls": float(np.mean(calls)), "max_encoder_calls": int(max(calls))}
        out[name] = {"n": len(pick), "datasets": sorted({it["dataset"] for it in pick}), "variants": res}
    C.write_json_new(C.RESULTS / "latency.json", {"protocol": __doc__.strip(), "buckets": out})
    C.log("s04 latency: " + "; ".join(f"{b}: V0 {out[b]['variants']['V0']['p50_ms']:.0f}/{out[b]['variants']['V0']['p95_ms']:.0f} "
                                      f"V1 {out[b]['variants']['V1']['p50_ms']:.0f}/{out[b]['variants']['V1']['p95_ms']:.0f} ms"
                                      for b, _, _ in BUCKETS if out[b].get("n")))


if __name__ == "__main__":
    main()
