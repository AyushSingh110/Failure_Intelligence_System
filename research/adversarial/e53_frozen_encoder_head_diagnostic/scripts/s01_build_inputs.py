"""
E53 step 1 — build every input deterministically and write manifests.

No model is loaded and nothing is scored here.

Outputs (exclusive-create):
  manifests/train_corpus.jsonl     one row per training example (text hash, label, weight, source, file, split)
  manifests/calibration_raw.jsonl  calibration candidates before near-duplicate filtering
  manifests/eval_items.jsonl       one row per evaluation item (harness-built, joinable to BL-0002 by input_sha256)
  cache/texts.jsonl                unique texts to embed: {"sha256", "text"}
  manifests/inputs_summary.json    counts, file hashes, dataset content hashes
"""
from __future__ import annotations

import collections
import json

import e53_common as C


def main() -> None:
    import numpy as np
    from sklearn.model_selection import train_test_split

    from evals import datasets as D
    from evals import transforms as T

    file_hashes = {rel: C.sha256_file(C.REPO / rel)
                   for rel, _ in C.TRAIN_FILES}
    file_hashes.update({rel: C.sha256_file(C.REPO / rel) for rel in C.CAL_FILES + C.EXTRA_EVAL_FILES})

    # ── training corpus + the recipe's 90/10 split ───────────────────────────
    corpus = C.build_training_corpus()
    labels = np.array([r["label"] for r in corpus])
    idx = np.arange(len(corpus))
    tr, va = train_test_split(idx, test_size=0.10, random_state=C.SEED, stratify=labels)
    tr_set = set(int(i) for i in tr)
    texts: dict[str, str] = {}
    lines = []
    for i, r in enumerate(corpus):
        h = C.sha256_text(r["text"])
        texts[h] = r["text"]
        lines.append(json.dumps({
            "i": i, "text_sha256": h, "label": r["label"], "weight": r["weight"],
            "source": r["source"], "category": r["category"], "file": r["file"],
            "mode": r["mode"], "split": "tr" if i in tr_set else "va",
        }, sort_keys=True))
    C.write_new(C.MANIFESTS / "train_corpus.jsonl", "\n".join(lines) + "\n")

    # ── calibration candidates (filtered later, after embedding) ─────────────
    cal = []
    for rel in C.CAL_FILES:
        for j, r in enumerate(C.read_jsonl(C.REPO / rel)):
            p = (r.get("prompt") or "").strip()
            h = C.sha256_text(p)
            texts[h] = p
            cal.append({"file": rel, "row": j, "text_sha256": h, "label": C._lab(r),
                        "source": r.get("source"), "category": r.get("category"),
                        "norm_sha1": C._hash(p)})
    C.write_new(C.MANIFESTS / "calibration_raw.jsonl",
                "\n".join(json.dumps(c, sort_keys=True) for c in cal) + "\n")

    # ── evaluation items through the harness's own builders ──────────────────
    reg = D.load_registry(C.REPO)
    suites = T.load_suites(C.REPO)
    fx = T.load_fixtures(C.REPO, suites)
    by_id = {s["id"]: s for s in suites["suites"]}
    needed = sorted({p["dataset"] for sid in ("std.xstest", "std.orbench_hard", "std.jailbreakbench",
                                               "std.harmbench", "std.strongreject", "std.sorrybench",
                                               "risk.framing", "risk.long_input")
                     for p in by_id[sid]["parts"]})
    dsets = {d: D.load_dataset(d, C.REPO, reg, verify=True) for d in needed}

    items = []
    for sid in ("std.xstest", "std.orbench_hard", "std.jailbreakbench", "std.harmbench",
                "std.strongreject", "std.sorrybench", "risk.framing", "risk.long_input"):
        for it in T.build_items(by_id[sid], dsets, fx):
            row = dsets[it["dataset"]].rows[it["source_idx"]]
            h = C.sha256_text(it["text"])
            assert h == it["input_sha256"]
            texts[h] = it["text"]
            items.append({
                "suite": sid, "dataset": it["dataset"], "variant": it["variant"],
                "source_idx": it["source_idx"], "suite_idx": it["idx"],
                "expected": it["expected"], "label": 1 if it["expected"] == "attack" else 0,
                "text_sha256": h, "input_sha256": it["input_sha256"],
                "category": row.get("category"),
            })
    for rel in C.EXTRA_EVAL_FILES:
        name = "test_jsonl" if rel.endswith("test.jsonl") and "pair_training/" in rel else "aug_benign_test"
        for j, r in enumerate(C.read_jsonl(C.REPO / rel)):
            p = (r.get("prompt") or "").strip()
            h = C.sha256_text(p)
            texts[h] = p
            items.append({"suite": "extra", "dataset": name, "variant": "base", "source_idx": j,
                          "suite_idx": j, "expected": "attack" if C._lab(r) else "benign",
                          "label": C._lab(r), "text_sha256": h, "input_sha256": h,
                          "category": r.get("category"), "source": r.get("source")})
    for k, it in enumerate(items):
        it["item"] = k
    C.write_new(C.MANIFESTS / "eval_items.jsonl",
                "\n".join(json.dumps(it, sort_keys=True) for it in items) + "\n")

    C.write_new(C.CACHE / "texts.jsonl",
                "\n".join(json.dumps({"sha256": h, "text": t}, ensure_ascii=False)
                          for h, t in sorted(texts.items())) + "\n")

    summary = {
        "train_corpus_rows": len(corpus),
        "train_label_counts": dict(collections.Counter(int(x) for x in labels)),
        "tr_rows": int(len(tr)), "va_rows": int(len(va)),
        "train_source_label_counts": {f"{s}|{l}": n for (s, l), n in sorted(
            collections.Counter((r["source"], r["label"]) for r in corpus).items())},
        "calibration_candidates": len(cal),
        "eval_items": len(items),
        "eval_items_by_suite": dict(collections.Counter(it["suite"] for it in items)),
        "unique_texts_to_embed": len(texts),
        "input_file_sha256": file_hashes,
        "dataset_content_sha256": {d: ds.content_sha256 for d, ds in dsets.items()},
        "fixtures": fx.hashes,
        "pinned_artifacts": C.verify_pinned(),
    }
    C.write_json_new(C.MANIFESTS / "inputs_summary.json", summary)
    C.log(f"s01 built inputs: corpus {len(corpus)} (tr {len(tr)}, va {len(va)}), "
          f"calibration candidates {len(cal)}, eval items {len(items)}, unique texts {len(texts)}")


if __name__ == "__main__":
    main()
