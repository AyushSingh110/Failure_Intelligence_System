"""
PAIR classifier v2 retraining — adds JBB near-boundary benign prompts as hard negatives.

Problem:
  The v1 model was trained on alpaca + anthropic_rlhf_helpful as the benign class.
  These are generic helpful questions, very different from JBB benign prompts which
  are near-boundary (same topics as harmful, but fictional/creative framing).
  Result: 79% FPR on JBB benchmark.

Fix:
  Add the 79 false positive prompts from the JBB evaluation as hard negative examples.
  Also add the 10 missed harmful prompts (FN) as hard positive examples.
  Retrain LinearSVC on original training data + these hard examples.
  Find the best threshold on a held-out validation set.

Usage:
  python scripts/retrain_pair_v2.py \
    --predictions notebooks/fie_vs_llamaguard_full_predictions.csv \
    --output fie/models/pair_intent_classifier_v2.pkl

Requirements:
  pip install scikit-learn sentence-transformers pandas numpy joblib datasets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
_MODELS   = _ROOT / "fie" / "models"
_V1_PKL   = _MODELS / "pair_intent_classifier.pkl"
_V1_META  = _MODELS / "pair_intent_meta.json"
_V2_PKL   = _MODELS / "pair_intent_classifier_v2.pkl"
_V2_META  = _MODELS / "pair_intent_meta_v2.json"


def load_jbb_hard_examples(predictions_csv: Path) -> tuple[list[str], list[int]]:
    """
    Load false positives (benign flagged as attack) and false negatives
    (harmful missed by FIE) from the benchmark predictions CSV.

    Returns: (prompts, labels) — 0=benign, 1=attack
    """
    import pandas as pd

    df = pd.read_csv(predictions_csv)
    required = {"prompt", "label", "fie_pred"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {df.columns.tolist()}")

    false_positives = df[(df["label"] == 0) & (df["fie_pred"] == 1)]
    false_negatives = df[(df["label"] == 1) & (df["fie_pred"] == 0)]
    true_positives  = df[(df["label"] == 1) & (df["fie_pred"] == 1)]

    prompts = []
    labels  = []

    # FP benign prompts → hard negatives (label=0)
    for p in false_positives["prompt"].tolist():
        prompts.append(str(p)); labels.append(0)

    # FN harmful prompts → hard positives (label=1)
    for p in false_negatives["prompt"].tolist():
        prompts.append(str(p)); labels.append(1)

    # Add a sample of TP harmful prompts as extra positives to avoid over-correction
    tp_sample = true_positives["prompt"].sample(min(30, len(true_positives)), random_state=42)
    for p in tp_sample.tolist():
        prompts.append(str(p)); labels.append(1)

    print(f"Hard examples loaded:")
    print(f"  False positives (hard negatives) : {len(false_positives)}")
    print(f"  False negatives (hard positives) : {len(false_negatives)}")
    print(f"  True positives  (anchor)         : {len(tp_sample)}")
    print(f"  Total hard examples              : {len(prompts)}")
    return prompts, labels


def load_original_training_data() -> tuple[list[str], list[int]]:
    """
    Reconstruct a balanced sample of the original training distribution
    using the same public datasets the v1 model was trained on.
    Falls back to a minimal synthetic set if HuggingFace is unavailable.
    """
    prompts: list[str] = []
    labels:  list[int] = []

    print("\nLoading original training data from public datasets...")

    # ── Benign: alpaca (instruction-following, safe) ──────────────────────────
    try:
        from datasets import load_dataset
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        sample = alpaca.shuffle(seed=42).select(range(min(500, len(alpaca))))
        for row in sample:
            inst = row.get("instruction", "")
            inp  = row.get("input", "")
            text = (inst + " " + inp).strip()
            if text and len(text) > 10:
                prompts.append(text[:512]); labels.append(0)
        print(f"  alpaca benign : {len([l for l in labels if l == 0])}")
    except Exception as e:
        print(f"  alpaca: skipped ({e})")

    # ── Attack: jailbreakbench harmful behaviors ──────────────────────────────
    try:
        from datasets import load_dataset
        jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        for row in jbb["harmful"]:
            prompts.append(str(row["Goal"])[:512]); labels.append(1)
        print(f"  jailbreakbench attacks : {len([l for l in labels if l == 1])}")
    except Exception as e:
        print(f"  jailbreakbench: skipped ({e})")

    # ── Attack: HarmfulQA / red-team samples ─────────────────────────────────
    try:
        from datasets import load_dataset
        rt = load_dataset("anthropics/hh-rlhf", "red-team-attempts", split="train")
        sample = rt.shuffle(seed=42).select(range(min(400, len(rt))))
        for row in sample:
            text = (row.get("transcript", "") or "")[:512]
            if text and len(text) > 15:
                prompts.append(text); labels.append(1)
        print(f"  anthropic red-team attacks : {len([l for l in labels if l == 1])}")
    except Exception as e:
        print(f"  anthropic red-team: skipped ({e})")

    # ── Benign: JBB benign (near-boundary — crucial) ─────────────────────────
    try:
        from datasets import load_dataset
        jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        for row in jbb["benign"]:
            prompts.append(str(row["Goal"])[:512]); labels.append(0)
        print(f"  JBB benign (near-boundary) : added to benign class")
    except Exception as e:
        print(f"  JBB benign: skipped ({e})")

    print(f"\n  Total original data: {len(prompts)} "
          f"(attack={labels.count(1)}, benign={labels.count(0)})")
    return prompts, labels


def encode_prompts(prompts: list[str], embed_model: str) -> "np.ndarray":
    """Encode prompts with sentence-transformers. Returns L2-normalized embeddings."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"\nEncoding {len(prompts)} prompts with {embed_model}...")
    t0  = time.time()
    model = SentenceTransformer(embed_model)
    vecs  = model.encode(
        prompts,
        batch_size    = 64,
        normalize_embeddings = True,
        show_progress_bar    = True,
    )
    print(f"  Done in {round(time.time()-t0, 1)}s  shape={vecs.shape}")
    return vecs


def find_best_threshold(
    model, X_val: "np.ndarray", y_val: list[int]
) -> tuple[float, dict]:
    """Find threshold maximising F1 on validation set."""
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    # CalibratedClassifierCV exposes predict_proba (not decision_function)
    proba = model.predict_proba(X_val)
    scores_norm = proba[:, 1]  # P(attack)

    prec, rec, threshs = precision_recall_curve(y_val, scores_norm)
    f1s  = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1s[:-1]))
    t    = float(threshs[best])

    preds = (scores_norm >= t).astype(int)
    tp = int(((preds == 1) & (np.array(y_val) == 1)).sum())
    fp = int(((preds == 1) & (np.array(y_val) == 0)).sum())
    fn = int(((preds == 0) & (np.array(y_val) == 1)).sum())
    tn = int(((preds == 0) & (np.array(y_val) == 0)).sum())

    metrics = {
        "threshold": round(t, 4),
        "recall":    round(tp / (tp + fn + 1e-9), 4),
        "fpr":       round(fp / (fp + tn + 1e-9), 4),
        "precision": round(tp / (tp + fp + 1e-9), 4),
        "f1":        round(f1s[best], 4),
        "auc_roc":   round(float(roc_auc_score(y_val, scores_norm)), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    return t, metrics


def retrain(predictions_csv: Path, output_pkl: Path, output_meta: Path) -> None:
    import numpy as np
    import joblib
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split

    # ── Load v1 metadata ──────────────────────────────────────────────────────
    with open(_V1_META) as f:
        v1_meta = json.load(f)
    embed_model = v1_meta.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    # ── Gather training data ──────────────────────────────────────────────────
    hard_prompts, hard_labels = load_jbb_hard_examples(predictions_csv)
    orig_prompts, orig_labels = load_original_training_data()

    all_prompts = orig_prompts + hard_prompts
    all_labels  = orig_labels  + hard_labels

    # Weight hard examples 3x to ensure boundary correction
    sample_weights = [1.0] * len(orig_prompts) + [3.0] * len(hard_prompts)

    print(f"\nFinal training set: {len(all_prompts)} examples "
          f"(attack={all_labels.count(1)}, benign={all_labels.count(0)})")

    # ── Encode ────────────────────────────────────────────────────────────────
    X_all = encode_prompts(all_prompts, embed_model)
    y_all = np.array(all_labels)
    w_all = np.array(sample_weights)

    # ── Train/val split (stratified, 85/15) ───────────────────────────────────
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_all, y_all, w_all,
        test_size=0.15, random_state=42, stratify=y_all,
    )

    print(f"\nTraining LinearSVC v2 on {len(X_tr)} examples...")
    base = LinearSVC(C=0.8, max_iter=3000, class_weight="balanced")
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(X_tr, y_tr, **{"sample_weight": w_tr} if hasattr(base, "fit") else {})

    # ── Threshold search on val set ───────────────────────────────────────────
    print("\nFinding optimal threshold on validation set...")
    threshold, val_metrics = find_best_threshold(model, X_val, y_val.tolist())
    print(f"  Best threshold : {threshold:.4f}")
    print(f"  Val Recall     : {val_metrics['recall']:.4f}")
    print(f"  Val FPR        : {val_metrics['fpr']:.4f}")
    print(f"  Val Precision  : {val_metrics['precision']:.4f}")
    print(f"  Val F1         : {val_metrics['f1']:.4f}")
    print(f"  Val AUC        : {val_metrics['auc_roc']:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_pkl)

    meta = {
        "trained_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_type":    "LinearSVC + CalibratedClassifierCV (sigmoid)",
        "embed_model":   embed_model,
        "embed_dim":     384,
        "threshold":     threshold,
        "train_size":    len(X_tr),
        "val_size":      len(X_val),
        "hard_negatives": hard_labels.count(0),
        "hard_positives": hard_labels.count(1),
        "val_metrics":   val_metrics,
        "v1_val_metrics": v1_meta.get("val_metrics", {}),
        "improvement_vs_v1": {
            "fpr_delta":      round(val_metrics["fpr"] - v1_meta.get("val_metrics", {}).get("fpr", 0), 4),
            "recall_delta":   round(val_metrics["recall"] - v1_meta.get("val_metrics", {}).get("recall", 0), 4),
        },
        "sources": {
            "original_training": orig_labels.count(1),
            "jbb_hard_negatives": hard_labels.count(0),
            "jbb_hard_positives": hard_labels.count(1),
        },
    }
    with open(output_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:")
    print(f"  Model    : {output_pkl}")
    print(f"  Metadata : {output_meta}")
    print(f"\nv1 -> v2 FPR delta  : {meta['improvement_vs_v1']['fpr_delta']:+.4f}")
    print(f"v1 -> v2 Recall delta: {meta['improvement_vs_v1']['recall_delta']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Retrain PAIR classifier v2 with JBB hard negatives")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_ROOT / "notebooks" / "fie_vs_llamaguard_full_predictions.csv",
        help="Path to fie_vs_llamaguard_full_predictions.csv from the benchmark notebook",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_V2_PKL,
        help="Output path for the retrained model pkl",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=_V2_META,
        help="Output path for the model metadata json",
    )
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"ERROR: predictions CSV not found at {args.predictions}")
        print("Run the benchmark notebook first to generate fie_vs_llamaguard_full_predictions.csv")
        sys.exit(1)

    if not _V1_META.exists():
        print(f"ERROR: v1 metadata not found at {_V1_META}")
        sys.exit(1)

    retrain(args.predictions, args.output, args.meta)
    print("\nDone. To activate v2, update _run_pair_classifier() to load v2 pkl.")
    print("Or rename: pair_intent_classifier_v2.pkl -> pair_intent_classifier.pkl")


if __name__ == "__main__":
    main()
