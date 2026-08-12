r"""
Phase 1 (E28) — train PAIR v6.4-harmaug: the v6.3b recipe + HarmAug positives.

THE SINGLE-VARIABLE DESIGN
--------------------------
This reproduces train_pair_v63b.py EXACTLY — same estimator, same
hyper-parameters, same seed, same split, same per-source weights — and changes
one thing: the BASE corpus file becomes train.harmaug.jsonl instead of
train.decontam.jsonl.

That matters, and it is easy to get wrong. The shipped v6.3b model is not
trained on train.decontam.jsonl alone; it is trained on that PLUS dataset_v6,
augmented_v6, the E24 soft-harm positives and the E25 benign negatives. Handing
the merged HarmAug file to a trainer as the whole corpus would silently drop
four of the five sources, and the resulting comparison would measure "v6.3b's
extra data" rather than "HarmAug". Swapping only the base keeps every other
source identical, so any delta is attributable to the augmentation.

Effect on the real corpus (after cross-source dedup):

    v6.3b            4236 rows, 53.8% attack
    v6.3b + HarmAug  6114 rows, 68.0% attack

WHY THE ONNX ENCODER
--------------------
v6.3b was fitted on sentence-transformers embeddings, but the runtime dropped
torch in favour of ONNX, so in production the v6.3b head is ALREADY being fed
ONNX vectors. Fitting this head on ONNX vectors therefore makes the comparison
more faithful, not less: both heads are evaluated under the embeddings they
actually receive when serving. (The two backends were gated at cosine > 0.999.)
It is also the only option — sentence-transformers is no longer installed in
the training environment.

WHAT TO EXPECT
--------------
Attack recall should rise. Over-refusal is expected to rise too, because a
positives-only augmentation moves the corpus from 53.8% to 68.0% attack and
FIE already flags 53.6% of safe XSTest prompts. Measuring both halves is the
experiment; the HarmAug paper reports only the first.

Writes to its OWN model file. v6.3b remains the shipped default and is never
overwritten — this model is A/B only until it passes a gate.

Usage (Windows CMD):

    conda activate failure-engine
    python scripts\train_pair_harmaug.py
    python scripts\train_pair_harmaug.py --base data\pair_training\train.harmaug_capped.jsonl ^
           --tag v64_harmaug_capped

Output:
    fie/models/pair_intent_classifier_v64_harmaug.pkl
    fie/models/pair_intent_meta_v64_harmaug.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "fie" / "models"

# The four sources v6.3b uses alongside the base. Unchanged here on purpose.
V6_DATA  = ROOT / "data" / "pair_training_v6" / "dataset_v6.decontam.jsonl"
V6_AUG   = ROOT / "data" / "pair_training_v6" / "augmented_v6.decontam.jsonl"
V63_POS  = ROOT / "data" / "pair_training_v6" / "augmented_v6_3_train.jsonl"
V63B_NEG = ROOT / "data" / "pair_training_v6" / "augmented_v6_3b_benign_train.jsonl"

DEFAULT_BASE = ROOT / "data" / "pair_training" / "train.harmaug.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PAIR_PREFIX = "Represent this text for security threat classification: "
SEED = 42


def _norm(t): return " ".join(t.lower().split())
def _hash(t): return hashlib.sha1(_norm(t).encode("utf-8")).hexdigest()


def _lab(r):
    """
    Strict label parsing, identical to train_pair_v63b.py.

    The benign augmentation stores label:"benign", which bool() reads as True.
    Anything not explicitly an attack marker is benign.
    """
    return 1 if r.get("label", 0) in (1, "1", "attack", True) else 0


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _load_all(base: Path, pos_w: float, neg_w: float):
    """Same source list and weights as v6.3b; only `base` differs."""
    prompts, labels, weights, sources, seen = [], [], [], [], set()

    def add(rows, weight, src):
        n = 0
        for r in rows:
            p = (r.get("prompt") or "").strip()
            if not p or _hash(p) in seen:
                continue
            seen.add(_hash(p))
            prompts.append(p[:512])
            labels.append(_lab(r))
            weights.append(weight)
            sources.append(r.get("source", src))
            n += 1
        print(f"  {src}: +{n}")
        return n

    print("Loading corpus:")
    for path, w in [(base, 1.0), (V6_DATA, 1.0), (V6_AUG, 2.0)]:
        if path.exists():
            add(_load_jsonl(path), w, str(path.relative_to(ROOT)))
        else:
            print(f"  WARNING missing: {path.relative_to(ROOT)}")
    for path, w, tag in [(V63_POS, pos_w, "soft-harm +"), (V63B_NEG, neg_w, "benign -")]:
        if not path.exists():
            print(f"ERROR missing: {path.relative_to(ROOT)}")
            raise SystemExit(1)
        add(_load_jsonl(path), w, f"{path.relative_to(ROOT)} ({tag}, {w}x)")

    n_atk = sum(labels)
    n_ha = sum(1 for s in sources if s == "harmaug")
    print(f"\nTotal: {len(prompts)}  (attack={n_atk} [{n_atk/len(prompts):.1%}], "
          f"benign={len(labels) - n_atk})")
    print(f"  of which HarmAug-generated: {n_ha}")
    return prompts, labels, weights, n_ha


def _encode(prompts: list[str]):
    """
    ONNX MiniLM with the PAIR prefix, length-bucketed.

    Bucketing is not cosmetic: the tokenizer pads to the longest text in each
    batch, and this corpus mixes 11-word prompts with 250-word suffixes, so an
    unsorted pass costs roughly an order of magnitude more. The permutation is
    inverted before returning, so row i of the output is still prompt i.
    """
    import numpy as np
    from fie.onnx_encoder import OnnxEncoder

    enc = OnnxEncoder()
    if not enc.available:
        print(f"ERROR: ONNX encoder unavailable ({enc.status()}).")
        print("Fetch it with: python scripts/download_models.py --strict")
        raise SystemExit(1)

    texts = [PAIR_PREFIX + p for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    t0 = time.time()
    vecs = np.asarray(
        enc.encode([texts[i] for i in order], batch_size=64,
                   normalize_embeddings=True),
        dtype="float32")
    out = np.empty_like(vecs)
    out[np.asarray(order)] = vecs
    print(f"  encoded {len(prompts)} in {time.time() - t0:.1f}s")
    return out


def _sweep(model, Xv, yv):
    """Threshold sweep on the held-out split, identical to v6.3b."""
    import numpy as np
    s = model.predict_proba(Xv)[:, 1]
    y = np.array(yv)
    rows = []
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        p = (s >= t).astype(int)
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        tpr = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        prec = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0
        rows.append(dict(threshold=t, tpr=tpr, fpr=fpr, precision=prec, f1=f1))
    print("\nVal sweep:\n  t      TPR      FPR      Prec     F1")
    for r in rows:
        print(f"  {r['threshold']:.2f}  {r['tpr']:.2%}  {r['fpr']:.2%}  "
              f"{r['precision']:.2%}  {r['f1']:.4f}")
    target = [r for r in rows if r["tpr"] >= 0.85 and r["fpr"] <= 0.05]
    return max(target or rows, key=lambda r: r["f1"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="base corpus (the ONLY thing that differs from v6.3b)")
    ap.add_argument("--tag", default="v64_harmaug",
                    help="model filename tag; also the version recorded in the meta")
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--neg-weight", type=float, default=2.0)
    args = ap.parse_args()

    base = Path(args.base)
    if not base.is_absolute():
        base = (ROOT / base).resolve()
    if not base.exists():
        print(f"ERROR: base corpus missing: {base}")
        print("Build it with: python scripts\\harmaug_build_trainset.py")
        return 1

    out_pkl  = MODELS / f"pair_intent_classifier_{args.tag}.pkl"
    out_meta = MODELS / f"pair_intent_meta_{args.tag}.json"
    # Refuse to clobber a shipped model. v6.3b is the production default and an
    # accidental --tag v6_3b would overwrite it with an experimental head.
    for protected in ("v6_3b", "v6_3", "v6", "v5", "v4"):
        if args.tag == protected:
            print(f"ERROR: --tag {args.tag} would overwrite a shipped/A-B model.")
            return 1

    import numpy as np
    import sklearn
    print(f"sklearn: {sklearn.__version__}")
    print(f"base corpus: {base.relative_to(ROOT) if base.is_relative_to(ROOT) else base}\n")

    prompts, labels, weights, n_ha = _load_all(base, args.pos_weight, args.neg_weight)

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(prompts))
    tr, va = train_test_split(idx, test_size=0.10, random_state=SEED,
                              stratify=np.array(labels))
    print(f"\nEncoding train ({len(tr)}) ...")
    Xtr = _encode([prompts[i] for i in tr])
    print(f"Encoding val ({len(va)}) ...")
    Xva = _encode([prompts[i] for i in va])
    ytr = np.array([labels[i] for i in tr])
    yva = np.array([labels[i] for i in va])
    wtr = np.array([weights[i] for i in tr])

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    import joblib

    print(f"\nTraining LinearSVC ({args.tag}) ...")
    t0 = time.time()
    model = CalibratedClassifierCV(
        LinearSVC(C=0.8, max_iter=5000, class_weight="balanced"),
        cv=3, method="sigmoid")
    model.fit(Xtr, ytr, sample_weight=wtr)
    print(f"  trained in {round(time.time() - t0, 1)}s")

    best = _sweep(model, Xva, yva)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_pkl)
    n_atk = int(sum(labels))
    out_meta.write_text(json.dumps({
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": args.tag,
        "experiment": "E28 — HarmAug reproduction (Phase 1)",
        "model_type": "LinearSVC + CalibratedClassifierCV (sigmoid)",
        "embed_model": EMBED_MODEL,
        "embed_backend": "onnx",
        "threshold": best["threshold"],
        "pos_weight": args.pos_weight,
        "neg_weight": args.neg_weight,
        "val_metrics": best,
        "base_corpus": str(base),
        "corpus_rows": len(prompts),
        "corpus_attack": n_atk,
        "corpus_attack_frac": round(n_atk / len(prompts), 4),
        "harmaug_rows": n_ha,
        "sklearn": sklearn.__version__,
        "note": "v6.3b recipe with the base corpus swapped for the HarmAug merge. "
                "Every other source, hyper-parameter and seed is identical, so the "
                "delta against v6.3b is attributable to the augmentation. A/B only "
                "until it passes the over-refusal gate; v6.3b remains shipped.",
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"PAIR {args.tag} trained.  threshold={best['threshold']:.2f}  "
          f"val TPR={best['tpr']:.2%}  FPR={best['fpr']:.2%}  F1={best['f1']:.4f}")
    print(f"  -> {out_pkl.relative_to(ROOT)}   (v6.3b untouched)")
    print("=" * 66)
    print("NEXT: python scripts\\eval_harmaug.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
