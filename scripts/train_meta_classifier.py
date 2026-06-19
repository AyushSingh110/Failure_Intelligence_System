"""Train a meta-classifier on layer_scores from scan_prompt() outputs.

The meta-classifier takes the 11 layer confidence scores as features and
predicts is_attack. This replaces fixed-weight aggregation with a learned
mapping that weights layers by their actual precision on your dataset.

Usage (Windows CMD — Groq-rate-limit safe with checkpoint/resume):

    python scripts\train_meta_classifier.py --dataset data\pair_training\train.jsonl
    python scripts\train_meta_classifier.py --dataset data\pair_training\train.jsonl --resume

Outputs:
    fie/models/meta_clf.pkl    — trained XGBoost or LogisticRegression
    fie/models/meta_clf.json   — feature names + threshold metadata

Checkpoint:
    data/meta_clf_checkpoint.jsonl  — one scored row per line; append-only
    Safe to Ctrl+C and re-run with --resume — skips already-processed rows.

Dataset format (JSONL, one per line):
    {"prompt": "...", "label": 1}   # label 1 = attack, 0 = benign
    {"text": "...", "is_attack": true}  # alternate field names accepted
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT_PATH = ROOT / "data" / "meta_clf_checkpoint.jsonl"
MODEL_OUT_PKL   = ROOT / "fie" / "models" / "meta_clf.pkl"
MODEL_OUT_JSON  = ROOT / "fie" / "models" / "meta_clf.json"

LAYER_NAMES = [
    "regex",
    "semantic",
    "indirect",
    "gcg_suffix",
    "many_shot",
    "pair",
    "base64",
    "unicode",
    "fiction_harm",
    "payload_split",
    "multilingual",
]


def _load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("text") or ""
            label_raw = obj.get("label") if "label" in obj else obj.get("is_attack", 0)
            label = int(bool(label_raw))
            if prompt:
                rows.append({"prompt": prompt, "label": label})
    return rows


def _load_checkpoint() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()
    seen = set()
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seen.add(obj["prompt"])
    return seen


def _load_checkpoint_rows() -> list[dict]:
    if not CHECKPOINT_PATH.exists():
        return []
    rows = []
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _score_prompt(prompt: str) -> dict:
    from fie.adversarial import scan_prompt
    result = scan_prompt(prompt)
    return result.layer_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="JSONL dataset with prompt + label")
    parser.add_argument("--resume", action="store_true",
                        help="resume from checkpoint (skip already-scored prompts)")
    parser.add_argument("--model", choices=["xgboost", "logistic"], default="xgboost",
                        help="meta-classifier algorithm (default: xgboost)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="flush checkpoint every N prompts (default: 50)")
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} rows from {args.dataset}")

    # ── Scoring phase ─────────────────────────────────────────────────────────
    already_scored: set[str] = set()
    if args.resume:
        already_scored = _load_checkpoint()
        print(f"Resuming: {len(already_scored)} prompts already scored")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = open(CHECKPOINT_PATH, "a", encoding="utf-8")

    pending = [r for r in dataset if r["prompt"] not in already_scored]
    print(f"Scoring {len(pending)} prompts ...")

    batch_count = 0
    for i, row in enumerate(pending):
        prompt = row["prompt"]
        try:
            scores = _score_prompt(prompt)
        except Exception as exc:
            print(f"  [warn] row {i}: scan failed ({exc}) — skipping")
            continue

        record = {
            "prompt": prompt,
            "label": row["label"],
            **{f"layer_{k}": scores.get(k, 0.0) for k in LAYER_NAMES},
        }
        checkpoint_file.write(json.dumps(record) + "\n")
        batch_count += 1

        if batch_count % args.checkpoint_every == 0:
            checkpoint_file.flush()
            print(f"  [{batch_count}/{len(pending)}] checkpoint flushed")

        # Minimal rate-limit guard (no Groq calls in scan_prompt, but be safe)
        if batch_count % 200 == 0:
            time.sleep(0.5)

    checkpoint_file.flush()
    checkpoint_file.close()
    print(f"Scoring done. Total rows in checkpoint: {len(_load_checkpoint())}")

    # ── Training phase ────────────────────────────────────────────────────────
    rows = _load_checkpoint_rows()
    if len(rows) < 10:
        print("Not enough rows to train — need at least 10. Exiting.")
        sys.exit(1)

    import numpy as np
    feature_cols = [f"layer_{k}" for k in LAYER_NAMES]
    X = np.array([[r.get(c, 0.0) for c in feature_cols] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)

    pos = y.sum()
    neg = len(y) - pos
    print(f"Training on {len(y)} rows ({pos} attacks, {neg} benign)")

    if args.model == "xgboost":
        try:
            import xgboost as xgb
            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=neg / max(pos, 1),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
        except ImportError:
            print("[warn] xgboost not installed — falling back to logistic regression")
            args.model = "logistic"

    if args.model == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)

    clf.fit(X, y)

    # Calibrate threshold: maximize F1 on training set (rough proxy)
    from sklearn.metrics import f1_score
    probs = clf.predict_proba(X)[:, 1]
    best_t, best_f1 = 0.50, 0.0
    for t in [i / 100 for i in range(30, 80)]:
        preds = (probs >= t).astype(int)
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    print(f"Best threshold: {best_t:.2f} (F1={best_f1:.4f} on train set)")

    # Save model
    MODEL_OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf, MODEL_OUT_PKL)

    meta = {
        "feature_names": feature_cols,
        "layer_names":   LAYER_NAMES,
        "threshold":     best_t,
        "train_f1":      round(best_f1, 4),
        "train_rows":    len(y),
        "model_type":    args.model,
    }
    with open(MODEL_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {MODEL_OUT_PKL}")
    print(f"Saved: {MODEL_OUT_JSON}")
    print("Done. Add meta_clf.pkl and meta_clf.json to the GitHub Release assets.")


if __name__ == "__main__":
    main()
