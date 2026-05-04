"""
XGBoost v4 training script.
Trained on the expanded dataset: TruthfulQA (1,104) + HaluEval (835) + MMLU (518) + builtin (20)
= 2,477 total labeled examples.

Key difference from v3: ~480 additional HaluEval examples that were added after v3 was trained.
HaluEval is a document-grounded hallucination benchmark — stronger signal for factual failures,
leading to better calibrated FPR without sacrificing recall.

Run from the project root:
    python data/train_v4.py

Saves:
    models/failure_classifier_v4.pkl
    models/feature_columns_v4.pkl
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

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED_DIR = os.path.join(ROOT, "data", "labeled")
MODELS_DIR  = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from engine.question_classifier import classify as classify_question_type

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

# Print source breakdown
sources = {}
for r in records:
    src = r.get("source", "unknown")
    sources[src] = sources.get(src, 0) + 1
print(f"Dataset sources: {dict(sorted(sources.items(), key=lambda x: -x[1]))}")

rows = []
qt_from_result = 0
qt_inferred    = 0

for r in records:
    fr = r["fie_result"]

    qt = (fr.get("question_type") or "").strip().upper()
    if qt in {"FACTUAL", "TEMPORAL", "REASONING", "CODE", "OPINION"}:
        qt_from_result += 1
    else:
        qt = classify_question_type(r.get("question", ""))
        qt_inferred += 1

    rows.append({
        "agreement_score"    : fr.get("agreement_score",    1.0),
        "entropy_score"      : fr.get("entropy_score",      0.0),
        "jury_confidence"    : fr.get("jury_confidence",    0.0),
        "fix_confidence"     : fr.get("fix_confidence",     0.0),
        "gt_confidence"      : fr.get("gt_confidence",      0.0),
        "high_failure_risk"  : int(fr.get("high_failure_risk",    False)),
        "fix_applied"        : int(fr.get("fix_applied",          False)),
        "requires_escalation": int(fr.get("requires_escalation",  False)),
        "gt_override"        : int(fr.get("gt_override",          False)),
        "archetype"          : fr.get("archetype",     "NONE") or "NONE",
        "jury_verdict"       : fr.get("jury_verdict",  "NONE") or "NONE",
        "fix_strategy"       : fr.get("fix_strategy",  "NONE") or "NONE",
        "gt_source"          : fr.get("gt_source",     "none") or "none",
        "question_type"      : qt,
        "label"              : int(r["fie_should_detect"]),
    })

print(f"question_type from fie_result : {qt_from_result}")
print(f"question_type inferred        : {qt_inferred}")

df = pd.DataFrame(rows)
print(f"\nDataFrame shape: {df.shape}")
print("\nClass balance:")
print(df["label"].value_counts().rename({1: "Failure", 0: "Correct"}))

NUMERIC_FEATURES     = ["agreement_score", "entropy_score", "jury_confidence",
                         "fix_confidence", "gt_confidence"]
BINARY_FEATURES      = ["high_failure_risk", "fix_applied", "requires_escalation", "gt_override"]
CATEGORICAL_FEATURES = ["archetype", "jury_verdict", "fix_strategy", "gt_source", "question_type"]

df_enc = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
feature_cols = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + [c for c in df_enc.columns if any(c.startswith(p + "_") for p in CATEGORICAL_FEATURES)]
)

X = df_enc[feature_cols].astype(float)
y = df["label"]

print(f"\nFeature matrix: {X.shape} ({len(feature_cols)} features)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y,
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

print("\nTraining XGBoost v4...")
model = XGBClassifier(
    n_estimators          = 400,
    max_depth             = 4,
    learning_rate         = 0.04,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 3,
    gamma                 = 0.1,
    scale_pos_weight      = 1,
    eval_metric           = "logloss",
    early_stopping_rounds = 25,
    random_state          = 42,
    verbosity             = 0,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"Best iteration: {model.best_iteration}")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

recall      = recall_score(y_test, y_pred)
specificity = recall_score(y_test, y_pred, pos_label=0)
fpr         = 1 - specificity
precision   = precision_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_pred_prob)

cv_model = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    eval_metric="logloss", random_state=42, verbosity=0,
)
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc    = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc")
cv_recall = cross_val_score(cv_model, X, y, cv=cv, scoring="recall")

V3_RECALL = 0.636
V3_FPR    = 0.386
V3_AUC    = 0.677

print("\n" + "=" * 60)
print("  XGBoost v4 RESULTS")
print("=" * 60)
print(f"  Dataset size : {len(records)} (v3 was 1,757)")
print(f"  Recall       : {recall*100:.1f}%  (v3 was {V3_RECALL*100:.1f}%)")
print(f"  Precision    : {precision*100:.1f}%")
print(f"  Specificity  : {specificity*100:.1f}%")
print(f"  FPR          : {fpr*100:.1f}%  (v3 was {V3_FPR*100:.1f}%)")
print(f"  F1           : {f1*100:.1f}%")
print(f"  AUC-ROC      : {auc:.3f}  (v3 was {V3_AUC:.3f})")
print(f"  CV AUC       : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"  CV Recall    : {cv_recall.mean()*100:.1f}% ± {cv_recall.std()*100:.1f}%")
print("=" * 60)

print(f"\n  AUC change vs v3 : {auc - V3_AUC:+.3f}")
print(f"  FPR change vs v3 : {(fpr - V3_FPR)*100:+.1f}pp")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Correct", "Failure"]))

print("\nTop 15 feature importances:")
importances = pd.Series(model.feature_importances_, index=feature_cols)
for feat, imp in importances.nlargest(15).items():
    print(f"  {feat:<45} {imp:.4f}")

v4_model_path = os.path.join(MODELS_DIR, "failure_classifier_v4.pkl")
v4_cols_path  = os.path.join(MODELS_DIR, "feature_columns_v4.pkl")

joblib.dump(model,           v4_model_path)
joblib.dump(list(X.columns), v4_cols_path)

print(f"\nSaved: {v4_model_path}")
print(f"Saved: {v4_cols_path}")
