"""
XGBoost v3 training script.
Adds question_type as a categorical feature to the existing labeled dataset.
For records missing question_type in fie_result, infers it from the question text
using the same rule-based classifier used in production.

Run from the project root:
    python data/train_v3.py

Saves:
    models/failure_classifier_v3.pkl
    models/feature_columns_v3.pkl
"""
from __future__ import annotations

import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED_DIR = os.path.join(ROOT, "data", "labeled")
MODELS_DIR  = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Add project root to path so we can import the classifier ──────────────────
sys.path.insert(0, ROOT)
from engine.question_classifier import classify as classify_question_type


# ── Load all labeled data ─────────────────────────────────────────────────────
jsonl_files = sorted(glob.glob(os.path.join(LABELED_DIR, "synthetic_*.jsonl")))
if not jsonl_files:
    raise FileNotFoundError(f"No synthetic_*.jsonl files found in {LABELED_DIR}")

print(f"Loading {len(jsonl_files)} JSONL files...")

seen    = set()
records = []
for fpath in jsonl_files:
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r   = json.loads(line)
            key = (r.get("question", ""), r.get("label_type", ""))
            if key not in seen:
                seen.add(key)
                records.append(r)

print(f"Unique records loaded : {len(records)}")


# ── Build feature rows ────────────────────────────────────────────────────────
rows = []
qt_from_result = 0
qt_inferred    = 0

for r in records:
    fr = r["fie_result"]

    # question_type: try fie_result first, fall back to rule-based classifier
    qt = (fr.get("question_type") or "").strip().upper()
    if qt in {"FACTUAL", "TEMPORAL", "REASONING", "CODE", "OPINION"}:
        qt_from_result += 1
    else:
        qt = classify_question_type(r.get("question", ""))
        qt_inferred += 1

    rows.append({
        # numeric
        "agreement_score"    : fr.get("agreement_score",    1.0),
        "entropy_score"      : fr.get("entropy_score",      0.0),
        "jury_confidence"    : fr.get("jury_confidence",    0.0),
        "fix_confidence"     : fr.get("fix_confidence",     0.0),
        "gt_confidence"      : fr.get("gt_confidence",      0.0),
        # binary
        "high_failure_risk"  : int(fr.get("high_failure_risk",    False)),
        "fix_applied"        : int(fr.get("fix_applied",          False)),
        "requires_escalation": int(fr.get("requires_escalation",  False)),
        "gt_override"        : int(fr.get("gt_override",          False)),
        # categorical (v2 features)
        "archetype"          : fr.get("archetype",     "NONE") or "NONE",
        "jury_verdict"       : fr.get("jury_verdict",  "NONE") or "NONE",
        "fix_strategy"       : fr.get("fix_strategy",  "NONE") or "NONE",
        "gt_source"          : fr.get("gt_source",     "none") or "none",
        # NEW in v3
        "question_type"      : qt,
        # label
        "label"              : int(r["fie_should_detect"]),
    })

print(f"question_type from fie_result : {qt_from_result}")
print(f"question_type inferred        : {qt_inferred}")

df = pd.DataFrame(rows)
print(f"\nDataFrame shape: {df.shape}")
print("\nquestion_type distribution:")
print(df["question_type"].value_counts())
print("\nClass balance:")
print(df["label"].value_counts().rename({1: "Failure", 0: "Correct"}))


# ── Feature engineering ───────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "agreement_score", "entropy_score", "jury_confidence",
    "fix_confidence", "gt_confidence",
]
BINARY_FEATURES = [
    "high_failure_risk", "fix_applied", "requires_escalation", "gt_override",
]
CATEGORICAL_FEATURES = [
    "archetype", "jury_verdict", "fix_strategy", "gt_source",
    "question_type",   # <-- v3 addition
]

df_enc = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
feature_cols = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + [c for c in df_enc.columns if any(c.startswith(p + "_") for p in CATEGORICAL_FEATURES)]
)

X = df_enc[feature_cols].astype(float)
y = df["label"]

print(f"\nFeature matrix: {X.shape} ({len(feature_cols)} features)")
new_feats = [c for c in feature_cols if c.startswith("question_type_")]
print(f"New question_type columns: {new_feats}")


# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y,
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")


# ── Train XGBoost v3 ──────────────────────────────────────────────────────────
print("\nTraining XGBoost v3...")
model = XGBClassifier(
    n_estimators          = 300,
    max_depth             = 4,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    scale_pos_weight      = 1,
    eval_metric           = "logloss",
    early_stopping_rounds = 20,
    random_state          = 42,
    verbosity             = 0,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"Best iteration: {model.best_iteration}")


# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

recall      = recall_score(y_test, y_pred)
specificity = recall_score(y_test, y_pred, pos_label=0)
fpr         = 1 - specificity
f1          = f1_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_pred_prob)

# 5-fold CV
cv_model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, verbosity=0,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc   = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc")
cv_recall = cross_val_score(cv_model, X, y, cv=cv, scoring="recall")

# v2 baseline for comparison
POET_RECALL = 0.564
POET_FPR    = 0.387
V2_AUC      = 0.728

print("\n" + "=" * 60)
print("  XGBoost v3 RESULTS")
print("=" * 60)
print(f"  Recall      : {recall*100:.1f}%")
print(f"  Specificity : {specificity*100:.1f}%")
print(f"  FPR         : {fpr*100:.1f}%")
print(f"  F1          : {f1*100:.1f}%")
print(f"  AUC-ROC     : {auc:.3f}  (v2 was {V2_AUC:.3f})")
print(f"  CV AUC      : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"  CV Recall   : {cv_recall.mean()*100:.1f}% ± {cv_recall.std()*100:.1f}%")
print("=" * 60)

delta_auc = auc - V2_AUC
print(f"\n  AUC change vs v2: {delta_auc:+.3f}")
if auc > V2_AUC:
    print("  question_type feature IMPROVED the model.")
else:
    print("  question_type feature did not improve AUC on this split.")
    print("  (CV AUC is the more reliable number — check that too)")

print("\nTop 15 feature importances:")
importances = pd.Series(model.feature_importances_, index=feature_cols)
for feat, imp in importances.nlargest(15).items():
    print(f"  {feat:<45} {imp:.4f}")


# ── Save v3 model ─────────────────────────────────────────────────────────────
v3_model_path = os.path.join(MODELS_DIR, "failure_classifier_v3.pkl")
v3_cols_path  = os.path.join(MODELS_DIR, "feature_columns_v3.pkl")

joblib.dump(model,            v3_model_path)
joblib.dump(list(X.columns),  v3_cols_path)

print(f"\nSaved: {v3_model_path}")
print(f"Saved: {v3_cols_path}")
print("\nNext step: update engine/fie_config.py → MODEL_VERSION = 'xgboost-v3'")
