"""
E54 step 1 — build every partition and item, the long-benign sets and all views.
Nothing is scored here.

Outputs (exclusive-create):
  manifests/items.jsonl          one row per item: partition, dataset, domain, variant, unit, label, text_sha256
  manifests/views.jsonl          per item: views (kind, char span, token range, text_sha256), raw/unique counts, n_tokens
  cache/texts.jsonl              unique texts (items and views)
  manifests/build_summary.json   counts, input hashes, MMLU split, docs chunking, overlap audit
"""
from __future__ import annotations

import collections
import json
import re

import e54_common as C
import e54_views as V

MMLU_SUBJECTS = ["professional_law", "high_school_world_history", "high_school_us_history",
                 "high_school_european_history", "professional_medicine"]
DOC_FILES = ["README.md", "SECURITY.md", "Technical_report.md", "docs/OVERREFUSAL_FINDINGS.md",
             "docs/RESEARCH_REVIEW_2026-09.md", "docs/fie_rebuild_2026/BASELINE_AUDIT.md", "docs/ARCHITECTURE.md"]


def doc_chunks(text: str) -> list[str]:
    text = re.sub(r"```.*?```", "\n", text, flags=re.S)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(("|", "#"))]
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", "\n".join(lines))]
    paras = [p for p in paras if len(p.split()) >= 15]
    chunks, cur = [], []
    for p in paras:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= 120:
            chunks.append("\n\n".join(cur))
            cur = []
        if len(chunks) == 10:
            break
    return chunks


def main() -> None:
    import numpy as np
    from evals import datasets as D
    from evals import transforms as T

    C.verify_e53()
    pinned = C.verify_pinned()
    texts: dict[str, str] = {}
    items: list[dict] = []

    def add(partition, dataset, domain, variant, unit, label, text, **extra):
        h = C.sha256_text(text)
        texts[h] = text
        items.append({"item": len(items), "partition": partition, "dataset": dataset, "domain": domain,
                      "variant": variant, "unit": unit, "label": label, "text_sha256": h, **extra})

    # ── evaluation attacks / hard benign, through the harness's own builders ──
    reg = D.load_registry(C.REPO)
    suites = T.load_suites(C.REPO)
    fx = T.load_fixtures(C.REPO, suites)
    by_id = {s["id"]: s for s in suites["suites"]}
    sids = ("std.xstest", "std.orbench_hard", "std.jailbreakbench", "std.harmbench", "std.strongreject",
            "std.sorrybench", "risk.framing", "risk.long_input")
    need = sorted({p["dataset"] for s in sids for p in by_id[s]["parts"]})
    dsets = {d: D.load_dataset(d, C.REPO, reg, verify=True) for d in need}
    for sid in sids:
        for it in T.build_items(by_id[sid], dsets, fx):
            assert C.sha256_text(it["text"]) == it["input_sha256"]
            ds = it["dataset"]
            if sid == "risk.long_input":
                dataset = "hb120" if ds == "harmbench" else "xs_pad"
            elif sid == "risk.framing":
                dataset = f"frame_{ds}"
            else:
                dataset = ds
            add("eval", dataset, ds, it["variant"], f"{ds}:{it['source_idx']}",
                1 if it["expected"] == "attack" else 0, it["text"],
                suite=sid, source_idx=it["source_idx"], input_sha256=it["input_sha256"],
                category=dsets[ds].rows[it["source_idx"]].get("category"))

    # ── short benign (evaluation): test.jsonl benign rows ─────────────────────
    for j, r in enumerate(C.read_jsonl(C.REPO / "data/pair_training/test.jsonl")):
        if r["label"] in (0, "0", "benign", False):
            add("eval", "test_benign", r.get("source"), "base", f"test:{j}", 0, r["prompt"].strip())

    # ── calibration: E53 calibration pool (benign = thresholds; attack = report only) ──
    e53_texts = {r["sha256"]: r["text"] for r in C.read_jsonl(C.E53 / "cache" / "texts.jsonl")}
    for c in C.read_jsonl(C.E53 / "manifests" / "calibration_kept.jsonl"):
        if c["label"] == 0:
            add("cal_benign", "cal_short_benign", c["source"], "base", f"cal:{c['file']}:{c['row']}", 0,
                e53_texts[c["text_sha256"]])
        elif c["pool_primary"]:
            add("cal_attack", "cal_attack_primary", c["source"], "base", f"cal:{c['file']}:{c['row']}", 1,
                e53_texts[c["text_sha256"]])

    # ── MMLU-long ─────────────────────────────────────────────────────────────
    mmlu_path = C.REPO / "data/datasets/mmlu.json"
    mmlu = json.loads(mmlu_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(54)
    mmlu_split = {}
    for subj in MMLU_SUBJECTS:
        rows = [(i, r) for i, r in enumerate(mmlu) if r.get("category") == subj and len(r["question"].split()) >= 100][:60]
        groups = collections.OrderedDict()
        for i, r in rows:
            key = " ".join(r["question"].lower().split())[:250]
            groups.setdefault(key, []).append((i, r))
        keys = list(groups)
        order = rng.permutation(len(keys))
        for rank, gi in enumerate(order):
            part = "cal_benign" if rank % 2 == 0 else "eval"
            for i, r in groups[keys[gi]]:
                add(part, "long_mmlu_cal" if part == "cal_benign" else "long_mmlu", subj, "base",
                    f"mmlu_group:{subj}:{gi}", 0, r["question"].strip(), mmlu_index=i)
        mmlu_split[subj] = {"items": len(rows), "groups": len(keys)}

    # ── authored fixtures ─────────────────────────────────────────────────────
    auth_path = C.HERE / "fixtures" / "long_benign_authored_v1.jsonl"
    for r in C.read_jsonl(auth_path):
        cal = r["id"].endswith(("_01", "_03"))
        add("cal_benign" if cal else "eval", "long_auth_cal" if cal else "long_auth", r["category"], "base",
            f"auth:{r['id']}", 0, r["text"])

    # ── security-research docs (evaluation only, secondary) ───────────────────
    doc_hashes, doc_counts = {}, {}
    for rel in DOC_FILES:
        p = C.REPO / rel
        doc_hashes[rel] = C.sha256_file(p)
        ch = doc_chunks(p.read_text(encoding="utf-8"))
        doc_counts[rel] = len(ch)
        for k, t in enumerate(ch):
            add("eval", "docs", rel, "base", f"docs:{rel}", 0, t, chunk=k)

    # ── overlap audit: long-benign / test benign vs PAIR training corpus ──────
    train_norm = {C.norm_hash(e53_texts[r["text_sha256"]]) for r in C.read_jsonl(C.E53 / "manifests" / "train_corpus.jsonl")}
    overlap = collections.Counter(it["dataset"] for it in items
                                  if it["dataset"].startswith(("long_", "docs", "test_benign", "cal_short"))
                                  and C.norm_hash(texts[it["text_sha256"]]) in train_norm)

    # ── views ─────────────────────────────────────────────────────────────────
    view_rows = []
    for it in items:
        t = texts[it["text_sha256"]]
        views, n_tok = V.build_views(t)
        vs = []
        for v in views:
            h = C.sha256_text(v["text"])
            texts[h] = v["text"]
            vs.append({k: v[k] for k in v if k != "text"} | {"text_sha256": h})
        uniq = {k: len({v["text_sha256"] for v in vs if v["kind"] in kinds}) for k, kinds in
                {"V0": ("whole",), **V.VARIANTS}.items()}
        view_rows.append({"item": it["item"], "n_tokens": n_tok, "views": vs,
                          "raw_views": {k: sum(v["kind"] in kinds for v in vs) for k, kinds in
                                        {"V0": ("whole",), **V.VARIANTS}.items()},
                          "unique_views": uniq})

    C.write_jsonl_new(C.MANIFESTS / "items.jsonl", items)
    C.write_jsonl_new(C.MANIFESTS / "views.jsonl", view_rows)
    C.write_jsonl_new(C.CACHE / "texts.jsonl", [{"sha256": h, "text": t} for h, t in sorted(texts.items())])
    whole = {it["text_sha256"] for it in items}
    summary = {
        "items": len(items),
        "items_by_partition_dataset": {f"{k[0]}|{k[1]}": v for k, v in sorted(collections.Counter(
            (it["partition"], it["dataset"]) for it in items).items())},
        "unique_texts_total": len(texts), "unique_whole_texts": len(whole),
        "unique_view_texts_beyond_whole": len(set(texts) - whole),
        "unique_views_sum_by_variant": {k: sum(r["unique_views"][k] for r in view_rows) for k in ("V0", "V1", "V2", "V3")},
        "mmlu": {"path": "data/datasets/mmlu.json", "sha256": C.sha256_file(mmlu_path), "per_subject": mmlu_split},
        "authored_fixtures_sha256": C.sha256_file(auth_path),
        "docs": {"files_sha256": doc_hashes, "chunks": doc_counts},
        "e53_inputs_used": {"cache/texts.jsonl": C.sha256_file(C.E53 / "cache" / "texts.jsonl"),
                            "manifests/calibration_kept.jsonl": C.sha256_file(C.E53 / "manifests" / "calibration_kept.jsonl"),
                            "manifests/train_corpus.jsonl": C.sha256_file(C.E53 / "manifests" / "train_corpus.jsonl")},
        "test_jsonl_sha256": C.sha256_file(C.REPO / "data/pair_training/test.jsonl"),
        "dataset_content_sha256": {d: ds.content_sha256 for d, ds in dsets.items()},
        "fixtures": fx.hashes, "pinned": pinned,
        "exact_overlap_with_pair_training_corpus": dict(overlap),
    }
    C.write_json_new(C.MANIFESTS / "build_summary.json", summary)
    C.log(f"s01 built {len(items)} items; unique texts {len(texts)} "
          f"(whole {len(whole)}, extra views {len(set(texts) - whole)}); overlap with training {dict(overlap)}")


if __name__ == "__main__":
    main()
