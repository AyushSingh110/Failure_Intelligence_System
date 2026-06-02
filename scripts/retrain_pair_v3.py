"""
PAIR classifier v3 retraining — adds UnknownBench-v1 hard positives.

Problem:
  v2 was retrained on JBB near-boundary false positives.
  Result: FPR improved but TPR on novel unknown-category attacks remains low.
  Exp5 results show 169 missed attacks across 4 unknown categories (FNR ~85%).
  These missed prompts are genuinely hard — they deliberately avoid all known
  detection vocabulary. Adding them as hard positives trains the semantic model
  to generalise beyond keyword matching.

Fix:
  Load all FN records from exp5_unknown_categories batch JSON files.
  These become hard positives (label=1, weight=5x).
  Also load formal prose FPs as hard negatives to prevent over-correction.
  Retrain on original training data + v2 hard examples + new hard examples.
  Evaluate on held-out validation set and find optimal threshold.

Important:
  UnknownBench-v1 is now training data for v3.
  UnknownBench-v2 remains the HELD-OUT evaluation set.
  Never train on v2.

Usage:
  python scripts/retrain_pair_v3.py
  python scripts/retrain_pair_v3.py --dry-run   # show what would be loaded
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT     = Path(__file__).parent.parent
_MODELS   = _ROOT / "fie" / "models"
_V2_PKL   = _MODELS / "pair_intent_classifier_v2.pkl"
_V2_META  = _MODELS / "pair_intent_meta_v2.json"
_V3_PKL   = _MODELS / "pair_intent_classifier_v3.pkl"
_V3_META  = _MODELS / "pair_intent_meta_v3.json"

_EXP5_RESULTS = _ROOT / "evaluation" / "phase2" / "exp5_unknown_categories" / "results"
_EXP3_RESULTS = _ROOT / "evaluation" / "phase2" / "exp3_formal_prose" / "results"


def load_exp5_hard_positives() -> tuple[list[str], list[str]]:
    """
    Load all FN records (label=1 but predicted=False) from exp5 batch files.
    These are the unknown attacks that the current model missed.
    Returns: (prompts, categories)
    """
    prompts: list[str] = []
    categories: list[str] = []

    for cat_dir in sorted(_EXP5_RESULTS.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        for batch_file in sorted(cat_dir.glob("batch_*.json")):
            records = json.loads(batch_file.read_text(encoding="utf-8"))
            for r in records:
                if int(r["label"]) == 1 and not bool(r["predicted"]):
                    prompts.append(str(r["prompt"]))
                    categories.append(cat_name)

    counts: dict[str, int] = {}
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    print(f"\nExp5 hard positives (missed attacks) loaded: {len(prompts)}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<20} {n} FN prompts")

    return prompts, categories


def load_exp3_hard_negatives() -> list[str]:
    """
    Load FP records (label=0 but predicted=True) from exp3 formal prose results.
    These are benign academic prompts the model currently flags — hard negatives.
    """
    prompts: list[str] = []

    for batch_file in sorted(_EXP3_RESULTS.glob("batch_*_full.json")):
        records = json.loads(batch_file.read_text(encoding="utf-8"))
        for r in records:
            if int(r["label"]) == 0 and bool(r["predicted"]):
                prompts.append(str(r["prompt"]))

    print(f"\nExp3 hard negatives (formal prose FPs) loaded: {len(prompts)}")
    return prompts


def load_original_training_data() -> tuple[list[str], list[int]]:
    """
    Reconstruct balanced sample from original public training distribution.
    Same sources as v2 for consistency.
    """
    prompts: list[str] = []
    labels:  list[int] = []

    print("\nLoading original training data...")

    try:
        from datasets import load_dataset
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        sample = alpaca.shuffle(seed=42).select(range(min(500, len(alpaca))))
        for row in sample:
            text = (row.get("instruction", "") + " " + row.get("input", "")).strip()
            if text and len(text) > 10:
                prompts.append(text[:512]); labels.append(0)
        print(f"  alpaca benign       : {labels.count(0)}")
    except Exception as e:
        print(f"  alpaca: skipped ({e})")

    try:
        from datasets import load_dataset
        jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        for row in jbb["harmful"]:
            prompts.append(str(row["Goal"])[:512]); labels.append(1)
        print(f"  jailbreakbench atk  : {labels.count(1)}")
    except Exception as e:
        print(f"  jailbreakbench: skipped ({e})")

    try:
        from datasets import load_dataset
        jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        for row in jbb["benign"]:
            prompts.append(str(row["Goal"])[:512]); labels.append(0)
        print(f"  jbb benign          : added")
    except Exception as e:
        print(f"  jbb benign: skipped ({e})")

    print(f"\n  Total original data : {len(prompts)} "
          f"(attack={labels.count(1)}, benign={labels.count(0)})")
    return prompts, labels


def encode_prompts(prompts: list[str], embed_model: str) -> "np.ndarray":
    from sentence_transformers import SentenceTransformer
    print(f"\nEncoding {len(prompts)} prompts with {embed_model}...")
    t0 = time.time()
    model = SentenceTransformer(embed_model)
    vecs  = model.encode(
        prompts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"  Done in {round(time.time()-t0, 1)}s  shape={vecs.shape}")
    return vecs


def find_best_threshold(model, X_val: "np.ndarray", y_val: list[int]) -> tuple[float, dict]:
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    proba  = model.predict_proba(X_val)
    scores = proba[:, 1]

    prec, rec, threshs = precision_recall_curve(y_val, scores)
    f1s  = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1s[:-1]))
    t    = float(threshs[best])

    preds = (scores >= t).astype(int)
    y_arr = np.array(y_val)
    tp = int(((preds == 1) & (y_arr == 1)).sum())
    fp = int(((preds == 1) & (y_arr == 0)).sum())
    fn = int(((preds == 0) & (y_arr == 1)).sum())
    tn = int(((preds == 0) & (y_arr == 0)).sum())

    return t, {
        "threshold": round(t, 4),
        "recall":    round(tp / (tp + fn + 1e-9), 4),
        "fpr":       round(fp / (fp + tn + 1e-9), 4),
        "precision": round(tp / (tp + fp + 1e-9), 4),
        "f1":        round(f1s[best], 4),
        "auc_roc":   round(float(roc_auc_score(y_val, scores)), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def retrain(dry_run: bool = False) -> None:
    import numpy as np
    import joblib
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split

    if not _V2_META.exists():
        print(f"ERROR: v2 metadata not found at {_V2_META}")
        sys.exit(1)
    if not _EXP5_RESULTS.exists():
        print(f"ERROR: exp5 results not found at {_EXP5_RESULTS}")
        print("Run exp5 first: python -m evaluation.phase2.exp5_unknown_categories.run_eval")
        sys.exit(1)

    with open(_V2_META) as f:
        v2_meta = json.load(f)
    embed_model = v2_meta.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    # ── Load all data sources ─────────────────────────────────────────────────
    hard_pos_prompts, hard_pos_cats = load_exp5_hard_positives()
    hard_neg_prompts               = load_exp3_hard_negatives()
    orig_prompts, orig_labels       = load_original_training_data()

    if not hard_pos_prompts:
        print("ERROR: No exp5 FN prompts found. Run exp5 first.")
        sys.exit(1)

    # Build combined dataset
    all_prompts: list[str] = []
    all_labels:  list[int] = []
    weights:     list[float] = []

    # Original data — weight 1x
    all_prompts.extend(orig_prompts)
    all_labels.extend(orig_labels)
    weights.extend([1.0] * len(orig_prompts))

    # Exp5 hard positives — weight 5x (these are the genuinely hard unknown attacks)
    all_prompts.extend(hard_pos_prompts)
    all_labels.extend([1] * len(hard_pos_prompts))
    weights.extend([5.0] * len(hard_pos_prompts))

    # Exp3 formal prose hard negatives — weight 3x
    all_prompts.extend(hard_neg_prompts)
    all_labels.extend([0] * len(hard_neg_prompts))
    weights.extend([3.0] * len(hard_neg_prompts))

    print(f"\nFinal training set: {len(all_prompts)} examples")
    print(f"  Original      : {len(orig_prompts)}  (attack={orig_labels.count(1)}, benign={orig_labels.count(0)})")
    print(f"  Hard positives: {len(hard_pos_prompts)}  (exp5 missed attacks, weight=5x)")
    print(f"  Hard negatives: {len(hard_neg_prompts)}  (formal prose FPs, weight=3x)")

    if dry_run:
        print("\n[DRY RUN] Would train on the above data. Exiting.")
        return

    # ── Encode ────────────────────────────────────────────────────────────────
    X_all = encode_prompts(all_prompts, embed_model)
    y_all = np.array(all_labels)
    w_all = np.array(weights)

    # ── Stratified train/val split ────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_all, y_all, w_all,
        test_size=0.15, random_state=42, stratify=y_all,
    )

    print(f"\nTraining LinearSVC v3 on {len(X_tr)} examples...")
    base  = LinearSVC(C=0.8, max_iter=5000, class_weight="balanced")
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(X_tr, y_tr, **{"sample_weight": w_tr} if hasattr(base, "fit") else {})

    # ── Threshold search ──────────────────────────────────────────────────────
    print("\nFinding optimal threshold on validation set...")
    threshold, val_metrics = find_best_threshold(model, X_val, y_val.tolist())
    print(f"  Best threshold : {threshold:.4f}")
    print(f"  Val Recall     : {val_metrics['recall']:.4f}")
    print(f"  Val FPR        : {val_metrics['fpr']:.4f}")
    print(f"  Val F1         : {val_metrics['f1']:.4f}")
    print(f"  Val AUC        : {val_metrics['auc_roc']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    _MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _V3_PKL)

    meta = {
        "trained_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_type":       "LinearSVC + CalibratedClassifierCV (sigmoid)",
        "embed_model":      embed_model,
        "embed_dim":        384,
        "threshold":        threshold,
        "train_size":       len(X_tr),
        "val_size":         len(X_val),
        "hard_positives":   len(hard_pos_prompts),
        "hard_negatives":   len(hard_neg_prompts),
        "hard_pos_weight":  5.0,
        "hard_neg_weight":  3.0,
        "hard_pos_source":  "exp5_unknown_categories FN records",
        "hard_neg_source":  "exp3_formal_prose FP records",
        "val_metrics":      val_metrics,
        "v2_val_metrics":   v2_meta.get("val_metrics", {}),
        "improvement_vs_v2": {
            "fpr_delta":    round(val_metrics["fpr"] - v2_meta.get("val_metrics", {}).get("fpr", 0), 4),
            "recall_delta": round(val_metrics["recall"] - v2_meta.get("val_metrics", {}).get("recall", 0), 4),
        },
    }

    with open(_V3_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:")
    print(f"  Model    : {_V3_PKL}")
    print(f"  Metadata : {_V3_META}")
    print(f"\nv2 -> v3 FPR delta   : {meta['improvement_vs_v2']['fpr_delta']:+.4f}")
    print(f"v2 -> v3 Recall delta: {meta['improvement_vs_v2']['recall_delta']:+.4f}")
    print("\nv3 is now active — FIE auto-loads v3 > v2 > v1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain PAIR v3 on exp5 unknown hard positives")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be loaded without training")
    args = parser.parse_args()
    retrain(dry_run=args.dry_run)
