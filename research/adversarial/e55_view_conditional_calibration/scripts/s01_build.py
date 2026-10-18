"""
E55 step 1 — verify the E54 exploratory reproduction (stop rule 2), then build every
item and its V1 views. Nothing new is scored here.

Outputs (exclusive-create):
  results/e54_reproduction.json  E54 rule recomputed from E54 records
  manifests/items.jsonl          partition, dataset, domain, variant, unit, label, text_sha256
  manifests/views.jsonl          V1 views (whole + seg) with spans, unique count, n_tokens
  cache/texts.jsonl              unique texts
  manifests/build_summary.json   counts, hashes, overlap audit, calibration bin sizes and adequacy
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys

import e54_common as C
import e54_views as V

E54 = C.REPO / "research" / "adversarial" / "e54_segment_aware_scoring"
NEUTRAL = ["astronomy", "high_school_geography", "high_school_macroeconomics", "high_school_microeconomics",
           "marketing", "management", "conceptual_physics", "high_school_physics", "college_physics",
           "electrical_engineering", "econometrics", "college_chemistry", "high_school_chemistry", "nutrition",
           "global_facts"]
LONG_SUBJ = ["professional_law", "high_school_world_history", "high_school_us_history",
             "high_school_european_history", "professional_medicine"]
TECH_DOCS = ["docs/DEPLOYMENT.md", "docs/OPERATIONS.md", "docs/PRODUCTION_ENGINEERING.md", "docs/CODEBASE.md",
             "TESTING.md", "CONTRIBUTING.md", "RELEASE_CHECKLIST.md", "docs/HALLUCINATION_PIPELINE.md"]
SEC_DOCS = ["docs/RESEARCH_LOG.md", "docs/FACT_SHEET.md", "docs/fie_rebuild_2026/ROADMAP.md", "CHANGELOG.md"]
PAD_LEVELS = [21, 42, 84, 168, 336]


def verify_e54_dir() -> None:
    for line in (E54 / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        d, rel = line.split("  ", 1)
        if C.sha256_file(E54 / rel) != d:
            raise SystemExit(f"STOP: E54 artifact changed: {rel}")


def e54_reproduction() -> dict:
    import numpy as np
    rows = C.read_jsonl(E54 / "records" / "per_prompt.jsonl")
    thr = json.loads((E54 / "results" / "metrics.json").read_text(encoding="utf-8"))["thresholds"]
    t1 = thr["calibrated"]["V0"]
    mc = [r for r in rows if r["partition"] == "cal_benign" and r["unique_views"]["V1"] > 1]
    m = sum(r["scores"]["V0"] >= 0.5 for r in mc)
    s = sorted((r["scores"]["V1_lse4"] for r in mc), reverse=True)
    tm = math.nextafter(s[m], math.inf)
    ev = [r for r in rows if r["partition"] == "eval"]
    f = lambda r: r["scores"]["V1_lse4"] >= (t1 if r["unique_views"]["V1"] <= 1 else tm)
    g = lambda r: r["scores"]["V0"] >= t1
    sel = lambda p: [r for r in ev if p(r)]
    pad = sel(lambda r: r["dataset"] == "hb120" and r["variant"] == "pad_after:84")
    lng = sel(lambda r: r["dataset"] in ("long_mmlu", "long_auth"))
    att = ["jailbreakbench", "harmbench", "strongreject", "sorrybench"]
    mac = np.mean([np.mean([f(r) for r in sel(lambda r, d=d: r["dataset"] == d)]) for d in att])
    mac0 = np.mean([np.mean([g(r) for r in sel(lambda r, d=d: r["dataset"] == d)]) for d in att])
    out = {"t_single": t1, "t_multi": tm, "n_multi_cal": len(mc), "fp_multi_cal": m,
           "pad_after_84": {"recall": float(np.mean([f(r) for r in pad])), "n": len(pad)},
           "long_benign_fpr": {"fpr": float(np.mean([f(r) for r in lng])), "n": len(lng)},
           "macro_cond": float(mac), "macro_v0": float(mac0), "macro_delta_pts": float(100 * (mac - mac0))}
    out["matches_report"] = (round(out["pad_after_84"]["recall"], 4) == 0.5583 and round(out["long_benign_fpr"]["fpr"], 4) == 0.2236
                             and round(out["macro_delta_pts"], 2) == -5.47)
    return out


def prose_chunks(text: str) -> list[str]:
    text = re.sub(r"```.*?```", "\n", text, flags=re.S)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(("|", "#"))]
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", "\n".join(lines))]
    paras = [p for p in paras if len(p.split()) >= 15]
    chunks, cur = [], []
    for p in paras:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= 120:
            chunks.append("\n\n".join(cur)); cur = []
        if len(chunks) == 10:
            break
    return chunks


def code_chunks(text: str) -> list[str]:
    chunks, cur = [], []
    for ln in text.splitlines():
        cur.append(ln)
        if sum(len(x.split()) for x in cur) >= 150:
            chunks.append("\n".join(cur).strip()); cur = []
        if len(chunks) == 5:
            break
    return [c for c in chunks if c]


def main() -> None:
    import numpy as np
    from evals import datasets as D
    from evals import transforms as T

    C.verify_e53(); verify_e54_dir()
    pinned = C.verify_pinned()
    repro = e54_reproduction()
    C.write_json_new(C.RESULTS / "e54_reproduction.json", repro)
    if not repro["matches_report"]:
        C.log("STOP: E54 exploratory numbers do not reproduce from E54 records")
        sys.exit(2)
    C.log(f"s01 E54 reproduction OK: pad84 {repro['pad_after_84']}, long {repro['long_benign_fpr']}, macro {repro['macro_delta_pts']:.2f}")

    variants = json.loads((C.HERE / "fixtures" / "e55_variants_v1.json").read_text(encoding="utf-8"))
    texts, items = {}, []

    def add(partition, dataset, domain, variant, unit, label, text, **extra):
        h = C.sha256_text(text); texts[h] = text
        items.append({"item": len(items), "partition": partition, "dataset": dataset, "domain": domain,
                      "variant": variant, "unit": unit, "label": label, "text_sha256": h, **extra})

    # ── standard sets (fidelity, macro recall, development references) ───────
    reg = D.load_registry(C.REPO); suites = T.load_suites(C.REPO); fx = T.load_fixtures(C.REPO, suites)
    by_id = {s["id"]: s for s in suites["suites"]}
    std_ids = ("std.xstest", "std.orbench_hard", "std.jailbreakbench", "std.harmbench", "std.strongreject", "std.sorrybench")
    dsets = {d: D.load_dataset(d, C.REPO, reg, verify=True) for d in
             ["xstest_safe", "xstest_unsafe", "orbench_hard", "jailbreakbench", "harmbench", "strongreject", "sorrybench"]}
    for sid in std_ids:
        for it in T.build_items(by_id[sid], dsets, fx):
            add("eval", it["dataset"], it["dataset"], "base", f"{it['dataset']}:{it['source_idx']}",
                1 if it["expected"] == "attack" else 0, it["text"], suite=sid, source_idx=it["source_idx"],
                input_sha256=it["input_sha256"], category=dsets[it["dataset"]].rows[it["source_idx"]].get("category"))

    # ── base attack set B ────────────────────────────────────────────────────
    e54_hb = [r["prompt"] for _, r in T.select_rows(dsets["harmbench"], {"first_unique": 120})]
    excl, seen, B = set(e54_hb), set(), []
    for i, r in enumerate(dsets["harmbench"].rows):
        if r["prompt"] not in excl and r["prompt"] not in seen:
            seen.add(r["prompt"]); B.append(("harmbench", i, r["prompt"], r.get("category")))
    for i, r in enumerate(dsets["sorrybench"].rows):
        B.append(("sorrybench", i, r["prompt"], r.get("category")))

    mmlu = json.loads((C.REPO / "data/datasets/mmlu.json").read_text(encoding="utf-8"))
    pool = [r["question"].strip() for r in mmlu if r.get("category") in NEUTRAL]
    templates = variants["framing_templates"]; manip = variants["manipulation_sentences"]

    def filler(j, L):
        order = np.random.default_rng(55_000 + j).permutation(len(pool))
        words = []
        for k in order:
            words.extend(pool[k].split())
            if len(words) >= L:
                break
        return " ".join(words[:L])

    for j, (fam, idx, p, cat) in enumerate(B):
        u = f"B:{j}"
        add("eval", "B", fam, "base", u, 1, p, family=fam, source_idx=idx, category=cat, j=j)
        for L in PAD_LEVELS:
            fl = filler(j, L)
            add("eval", "B", fam, f"pad_after:{L}", u, 1, p + "\n\n" + fl, family=fam, j=j)
            add("eval", "B", fam, f"pad_before:{L}", u, 1, fl + "\n\n" + p, family=fam, j=j)
            add("eval", "filler_only", "mmlu_neutral_filler", f"filler:{L}", u, 0, fl, j=j)
        for name, tpl in templates.items():
            add("eval", "B", fam, f"frame:{name}", u, 1, tpl["template"].replace("{p}", p), family=fam, j=j)
        s = manip[j % len(manip)]
        add("eval", "B", fam, "manip_append", u, 1, p + "\n" + s, family=fam, j=j)
        add("eval", "B", fam, "manip_prepend", u, 1, s + "\n" + p, family=fam, j=j)

    # ── short benign: TruthfulQA + HaluEval halves ───────────────────────────
    tqa = [r["question"].strip() for r in json.loads((C.REPO / "data/datasets/truthfulqa.json").read_text(encoding="utf-8"))]
    halu = [r["question"].strip() for r in json.loads((C.REPO / "data/datasets/halueval.json").read_text(encoding="utf-8"))][:600]
    rng = np.random.default_rng(55)
    tqa_eval_first = []
    for name, qs in (("tqa", tqa), ("halu", halu)):
        perm = rng.permutation(len(qs)); cal_set = set(int(x) for x in perm[: len(qs) // 2])
        for i, q in enumerate(qs):
            cal = i in cal_set
            add("cal_benign" if cal else "eval", f"{name}_cal" if cal else f"{name}_eval", name, "base", f"{name}:{i}", 0, q)
            if name == "tqa" and not cal and len(tqa_eval_first) < 100:
                tqa_eval_first.append((i, q))
    for i, q in tqa_eval_first:
        for tn, tpl in templates.items():
            add("eval", "benign_frame", "tqa", f"frame:{tn}", f"tqa:{i}", 0, tpl["template"].replace("{p}", q))
        s = manip[i % len(manip)]
        add("eval", "benign_manip", "tqa", "manip_append", f"tqa:{i}", 0, q + "\n" + s)
        add("eval", "benign_manip", "tqa", "manip_prepend", f"tqa:{i}", 0, s + "\n" + q)

    # ── long benign: MMLU-long-new ───────────────────────────────────────────
    e54_items = C.read_jsonl(E54 / "manifests" / "items.jsonl")
    e54_texts = {r["sha256"]: r["text"] for r in C.read_jsonl(E54 / "cache" / "texts.jsonl")}
    e54_mmlu_keys = {" ".join(e54_texts[it["text_sha256"]].lower().split())[:250] for it in e54_items
                     if it["dataset"].startswith("long_mmlu")}
    rng = np.random.default_rng(55)
    mmlu_counts = {}
    for subj in LONG_SUBJ:
        rows = [(i, r) for i, r in enumerate(mmlu) if r.get("category") == subj and len(r["question"].split()) >= 100][60:120]
        groups = collections.OrderedDict(); dropped = 0
        for i, r in rows:
            key = " ".join(r["question"].lower().split())[:250]
            if key in e54_mmlu_keys:
                dropped += 1; continue
            groups.setdefault(key, []).append((i, r))
        keys = list(groups)
        for rank, gi in enumerate(rng.permutation(len(keys))):
            cal = rank % 2 == 0
            for i, r in groups[keys[gi]]:
                add("cal_benign" if cal else "eval", "mmlu_new_cal" if cal else "mmlu_new", subj, "base",
                    f"mmlu:{subj}:{gi}", 0, r["question"].strip(), mmlu_index=i)
        mmlu_counts[subj] = {"rows": len(rows), "dropped_shared_with_e54": dropped, "groups": len(keys)}

    # ── technical docs, code (split by file), security-research docs (eval only) ──
    file_hashes = {}
    for k, rel in enumerate(sorted(TECH_DOCS)):
        p = C.REPO / rel; file_hashes[rel] = C.sha256_file(p); cal = k % 2 == 0
        for c in prose_chunks(p.read_text(encoding="utf-8")):
            add("cal_benign" if cal else "eval", "techdocs_cal" if cal else "techdocs", rel, "base", f"doc:{rel}", 0, c)
    code_files = sorted(p.relative_to(C.REPO).as_posix() for p in (C.REPO / "evals").glob("*.py") if p.name != "__init__.py")
    for k, rel in enumerate(code_files):
        p = C.REPO / rel; file_hashes[rel] = C.sha256_file(p); cal = k % 2 == 0
        for c in code_chunks(p.read_text(encoding="utf-8")):
            add("cal_benign" if cal else "eval", "code_cal" if cal else "code", rel, "base", f"code:{rel}", 0, c)
    for rel in SEC_DOCS:
        p = C.REPO / rel; file_hashes[rel] = C.sha256_file(p)
        for c in prose_chunks(p.read_text(encoding="utf-8")):
            add("eval", "secdocs", rel, "base", f"sec:{rel}", 0, c)

    # ── E54 calibration-benign pool (texts from E54 cache, checksum-verified) ──
    for it in e54_items:
        if it["partition"] == "cal_benign":
            add("cal_benign", f"e54cal_{it['dataset']}", it["domain"], "base", f"e54:{it['unit']}", 0, e54_texts[it["text_sha256"]])

    # ── views (V1 = whole + seg only) ─────────────────────────────────────────
    view_rows = []
    for it in items:
        t = texts[it["text_sha256"]]
        vs, n_tok = V.build_views(t)
        keep = []
        for v in vs:
            if v["kind"] in ("whole", "seg"):
                h = C.sha256_text(v["text"]); texts[h] = v["text"]
                keep.append({k: v[k] for k in ("kind", "start", "end")} | {"text_sha256": h})
        uniq = len({v["text_sha256"] for v in keep})
        view_rows.append({"item": it["item"], "n_tokens": n_tok, "views": keep, "unique_views_V1": uniq,
                          "raw_views_V1": len(keep)})

    # ── overlap audit (exact normalised) ──────────────────────────────────────
    e53_texts = {r["sha256"]: r["text"] for r in C.read_jsonl(C.E53 / "cache" / "texts.jsonl")}
    train_norm = {C.norm_hash(e53_texts[r["text_sha256"]]) for r in C.read_jsonl(C.E53 / "manifests" / "train_corpus.jsonl")}
    e54_norm = {C.norm_hash(e54_texts[it["text_sha256"]]) for it in e54_items}
    ov_train = collections.Counter(); ov_e54 = collections.Counter()
    for it in items:
        if it["label"] == 0 and not it["dataset"].startswith(("e54cal_", "xstest", "orbench")):
            n = C.norm_hash(texts[it["text_sha256"]])
            ov_train[it["dataset"]] += n in train_norm
            ov_e54[it["dataset"]] += n in e54_norm
    uv = {it["item"]: r["unique_views_V1"] for it, r in zip(items, view_rows)}
    cal = [it for it in items if it["partition"] == "cal_benign"]
    n_single = sum(uv[it["item"]] <= 1 for it in cal); n_multi = len(cal) - n_single
    bins = {"single": n_single, "multi": n_multi, "pool": len(cal),
            "adequate": {f"alpha_{a}": {"single": n_single >= 10 / a, "multi": n_multi >= 10 / a, "pool": len(cal) >= 10 / a}
                         for a in (0.10, 0.05)}}

    C.write_jsonl_new(C.MANIFESTS / "items.jsonl", items)
    C.write_jsonl_new(C.MANIFESTS / "views.jsonl", view_rows)
    C.write_jsonl_new(C.CACHE / "texts.jsonl", [{"sha256": h, "text": t} for h, t in sorted(texts.items())])
    C.write_json_new(C.MANIFESTS / "build_summary.json", {
        "items": len(items),
        "by_partition_dataset": {f"{a}|{b}": n for (a, b), n in sorted(collections.Counter((it["partition"], it["dataset"]) for it in items).items())},
        "base_set_B": {"n": len(B), "by_family": dict(collections.Counter(b[0] for b in B))},
        "filler_pool_questions": len(pool), "mmlu_long_new": mmlu_counts,
        "unique_texts": len(texts), "unique_views_V1_total": sum(uv.values()),
        "calibration_bins": bins,
        "overlap_exact_with_pair_training": dict(ov_train), "overlap_exact_with_e54_items": dict(ov_e54),
        "input_sha256": {"mmlu.json": C.sha256_file(C.REPO / "data/datasets/mmlu.json"),
                         "truthfulqa.json": C.sha256_file(C.REPO / "data/datasets/truthfulqa.json"),
                         "halueval.json": C.sha256_file(C.REPO / "data/datasets/halueval.json"),
                         "fixtures/e55_variants_v1.json": C.sha256_file(C.HERE / "fixtures" / "e55_variants_v1.json"),
                         **file_hashes},
        "dataset_content_sha256": {d: ds.content_sha256 for d, ds in dsets.items()},
        "pinned": pinned,
    })
    C.log(f"s01 built {len(items)} items, {len(texts)} unique texts; calibration bins single {n_single}, multi {n_multi}; "
          f"overlap train {dict(ov_train)}; overlap e54 {dict(ov_e54)}")


if __name__ == "__main__":
    main()
