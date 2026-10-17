"""
E53 step 4 — train every arm, score every item (uncensored float64), fix thresholds
from the calibration pool only, and run the A5 leave-one-source-out diagnostic.

Refuses to run unless results/a0_reproducibility.json says the sanity gate passed.

Outputs (exclusive-create):
  results/arm_configs.json        per model: rows, labels, source composition, weight sums
  results/models/<name>.joblib    fitted research heads (never under fie/models)
  results/scores_eval.csv         one row per evaluation item, one column per model
  results/scores_calibration.csv  one row per kept calibration row, one column per model
  results/thresholds.json         r*, per-model thresholds (primary and sensitivity pools)
  results/scores_a5.csv, results/a5_loso.json
"""
from __future__ import annotations

import collections
import csv
import io
import json
import math

import e53_common as C
from e53_models import fit_recipe, load_embeddings, predict, tr_rows_in_recipe_order

REMOVE = {
    "A1": {"harmfulqa"},
    "A2": {"anthropic_rlhf_redteam"},
    "A3": {"harmfulqa", "anthropic_rlhf_redteam"},
}
CONTROL_SEEDS = [1, 2, 3, 4, 5]


def composition(rows, weights):
    by = collections.defaultdict(lambda: {"rows": 0, "weight": 0.0})
    for r, w in zip(rows, weights):
        k = f"{r['source']}|{r['label']}"
        by[k]["rows"] += 1
        by[k]["weight"] += float(w)
    lab = collections.Counter(r["label"] for r in rows)
    wsum = collections.defaultdict(float)
    for r, w in zip(rows, weights):
        wsum[r["label"]] += float(w)
    return {"rows": len(rows), "label_counts": {str(k): v for k, v in sorted(lab.items())},
            "weight_sum_by_label": {str(k): round(v, 6) for k, v in sorted(wsum.items())},
            "by_source_label": {k: {"rows": v["rows"], "weight": round(v["weight"], 6)}
                                for k, v in sorted(by.items())}}


def source_balanced_weights(rows):
    """A4: within each label class every source gets equal total weight; class totals kept."""
    n_cs = collections.Counter((r["label"], r["source"]) for r in rows)
    sources_c = collections.defaultdict(set)
    W = collections.defaultdict(float)
    for r in rows:
        sources_c[r["label"]].add(r["source"])
        W[r["label"]] += r["weight"]
    return [W[r["label"]] / (len(sources_c[r["label"]]) * n_cs[(r["label"], r["source"])]) for r in rows]


def matched_threshold(scores, r_star):
    """Largest threshold with recall >= r_star: the ceil(r*·n)-th largest score."""
    s = sorted(scores, reverse=True)
    m = max(1, math.ceil(r_star * len(s) - 1e-12))
    return float(s[m - 1])


def save_model(model, name):
    import joblib
    path = C.RESULTS / "models" / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        joblib.dump(model, f)


def write_csv_new(path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    C.write_new(path, buf.getvalue())


def main() -> None:
    import joblib
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    # The first gate run (a0_reproducibility.json) failed because of a row-order defect in
    # this experiment's A0 (deviation D9). Only the corrected re-run may authorise arms.
    gate = json.loads((C.RESULTS / "a0_reproducibility_rerun1.json").read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise SystemExit("STOP: A0 sanity gate did not pass")

    E, row_of = load_embeddings()
    corpus = C.read_jsonl(C.MANIFESTS / "train_corpus.jsonl")
    tr = tr_rows_in_recipe_order()   # D9: recipe row order; subsets keep it
    items = C.read_jsonl(C.MANIFESTS / "eval_items.jsonl")
    cal = C.read_jsonl(C.MANIFESTS / "calibration_kept.jsonl")

    # ── define every training set before fitting anything ────────────────────
    specs = {"A0": (tr, [r["weight"] for r in tr])}
    for arm, rm in REMOVE.items():
        rows = [r for r in tr if r["source"] not in rm]
        specs[arm] = (rows, [r["weight"] for r in rows])
    specs["A4"] = (tr, source_balanced_weights(tr))
    for k, arm in enumerate(("A1", "A2", "A3"), start=1):
        rm = REMOVE[arm]
        n_remove = sum(1 for r in tr if r["source"] in rm)
        pool = [j for j, r in enumerate(tr) if r["label"] == 1 and r["source"] not in rm]
        for seed in CONTROL_SEEDS:
            rng = np.random.default_rng(seed)
            drop = set(int(x) for x in rng.choice(pool, size=n_remove, replace=False))
            rows = [r for j, r in enumerate(tr) if j not in drop]
            specs[f"R{k}_s{seed}"] = (rows, [r["weight"] for r in rows])

    configs = {name: {**composition(rows, w),
                      "removed_sources": sorted(REMOVE.get(name, set())),
                      "weights": "source_balanced" if name == "A4" else "recipe"}
               for name, (rows, w) in specs.items()}
    C.write_json_new(C.RESULTS / "arm_configs.json", configs)

    # ── fit and score ────────────────────────────────────────────────────────
    shipped = joblib.load(C.REPO / "fie" / "models" / "pair_intent_classifier_v6_3b.pkl")
    Xe = E[[row_of[it["text_sha256"]] for it in items]]
    Xc = E[[row_of[c["text_sha256"]] for c in cal]]
    names = ["SHIPPED"] + list(specs)
    ev = {"SHIPPED": predict(shipped, Xe)}
    ca = {"SHIPPED": predict(shipped, Xc)}
    for name, (rows, w) in specs.items():
        model = fit_recipe(E, row_of, rows, weights=w)
        save_model(model, name)
        ev[name], ca[name] = predict(model, Xe), predict(model, Xc)
        C.log(f"s04 fitted {name}: rows {len(rows)}")

    fmt = lambda x: format(float(x), ".17g")
    write_csv_new(C.RESULTS / "scores_eval.csv",
                  ["item", "suite", "dataset", "variant", "source_idx", "label", "input_sha256", *names],
                  [[it["item"], it["suite"], it["dataset"], it["variant"], it["source_idx"], it["label"],
                    it["input_sha256"], *[fmt(ev[n][k]) for n in names]] for k, it in enumerate(items)])
    write_csv_new(C.RESULTS / "scores_calibration.csv",
                  ["file", "row", "source", "label", "pool_primary", "pool_secondary", "text_sha256", *names],
                  [[c["file"], c["row"], c["source"], c["label"], int(c["pool_primary"]), int(c["pool_secondary"]),
                    c["text_sha256"], *[fmt(ca[n][k]) for n in names]] for k, c in enumerate(cal)])

    # ── thresholds: calibration only ─────────────────────────────────────────
    prim = np.array([c["pool_primary"] for c in cal])
    sec = np.array([c["pool_secondary"] for c in cal])
    r_star = float((ca["A0"][prim] >= 0.50).mean())
    r_star_sec = float((ca["A0"][sec] >= 0.50).mean())
    thr = {"rule": "largest t with calibration recall >= r*, i.e. the ceil(r*·n)-th largest pool score",
           "r_star_primary": r_star, "n_primary": int(prim.sum()),
           "r_star_secondary": r_star_sec, "n_secondary": int(sec.sum()),
           "primary": {}, "secondary": {}, "realised_calibration_recall_primary": {}}
    for n in names:
        thr["primary"][n] = matched_threshold(ca[n][prim].tolist(), r_star)
        thr["secondary"][n] = matched_threshold(ca[n][sec].tolist(), r_star_sec)
        thr["realised_calibration_recall_primary"][n] = float((ca[n][prim] >= thr["primary"][n]).mean())
    C.write_json_new(C.RESULTS / "thresholds.json", thr)
    C.log(f"s04 thresholds fixed from calibration: r*={r_star:.4f} (n={int(prim.sum())}); "
          + ", ".join(f"{n}={thr['primary'][n]:.4f}" for n in ("A0", "A1", "A2", "A3", "A4")))

    # ── A5: leave-one-source-out vs row-random out-of-fold ───────────────────
    Xtr = E[[row_of[r["text_sha256"]] for r in tr]]
    ytr = np.array([r["label"] for r in tr])
    oof = np.zeros(len(tr))
    for k, (a, b) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED).split(Xtr, ytr)):
        m = fit_recipe(E, row_of, [tr[j] for j in a])
        oof[b] = predict(m, Xtr[b])
    loso = np.full(len(tr), np.nan)
    counts = collections.Counter(r["source"] for r in tr)
    table = {}
    for src, n in sorted(counts.items()):
        if n < 20:
            continue
        keep = [r for r in tr if r["source"] != src]
        idx = [j for j, r in enumerate(tr) if r["source"] == src]
        m = fit_recipe(E, row_of, keep)
        loso[idx] = predict(m, Xtr[idx])
        lab = int(ytr[idx[0]])
        rc = float((oof[idx] >= 0.5).mean())
        lo = float((loso[idx] >= 0.5).mean())
        table[src] = {"label": lab, "n": len(idx), "rowcv_positive_rate": rc, "loso_positive_rate": lo,
                      "optimism_pts": round(100 * ((rc - lo) if lab == 1 else (lo - rc)), 2),
                      "rowcv_mean_score": float(oof[idx].mean()), "loso_mean_score": float(loso[idx].mean())}
    write_csv_new(C.RESULTS / "scores_a5.csv", ["i", "source", "label", "rowcv_oof_score", "loso_score"],
                  [[r["i"], r["source"], r["label"], fmt(oof[j]), "" if math.isnan(loso[j]) else fmt(loso[j])]
                   for j, r in enumerate(tr)])
    C.write_json_new(C.RESULTS / "a5_loso.json", {
        "definition": "per source with >=20 tr rows: LOSO model trained on tr minus source; row-random = "
                      "StratifiedKFold(5, shuffle, 42) out-of-fold. Positive = score >= 0.50. Optimism: attack "
                      "sources rowCV-LOSO recall; benign sources LOSO-rowCV positive rate (points).",
        "sources_skipped_lt20": {s: n for s, n in counts.items() if n < 20},
        "per_source": table})
    C.log(f"s04 A5 LOSO done for {len(table)} sources")


if __name__ == "__main__":
    main()
