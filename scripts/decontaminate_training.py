"""Phase 0 cleanup — remove eval/benchmark contamination from training data.

E8 found 148 JailbreakBench prompts (and 13 HarmBench) sitting inside PAIR's
training set. This removes, from every training file, any row that is the same
item as a held-out evaluation prompt — so future numbers need no clean-subset
correction and the held-out splits are truly held out.

Reference set (rows that must NOT appear in training):
  - JailbreakBench eval artifacts (GCG / PAIR / JBC)
  - HarmBench behaviors (standard / contextual / copyright)
  - val.jsonl and test.jsonl (our own held-out splits)

Removal criterion (deliberately conservative to avoid deleting legitimate,
merely-topically-similar attacks):
  - EXACT  : identical after whitespace/case normalisation, OR
  - NEAR   : cosine similarity >= --cutoff (default 0.95 — near-identical text,
             not just same topic).

Non-destructive: writes `<name>.decontam.jsonl` next to each source file and a
report. Originals are untouched. retrain_pair_v6.py --decontam consumes these.

Usage:
    python scripts/decontaminate_training.py
    python scripts/decontaminate_training.py --cutoff 0.97   # stricter
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "data" / "benchmark_audit"

TRAIN_FILES = [
    ROOT / "data" / "pair_training" / "train.jsonl",
    ROOT / "data" / "pair_training_v6" / "dataset_v6.jsonl",
    ROOT / "data" / "pair_training_v6" / "augmented_v6.jsonl",
]
HELDOUT_FILES = [
    ROOT / "data" / "pair_training" / "val.jsonl",
    ROOT / "data" / "pair_training" / "test.jsonl",
]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _norm(t): return " ".join(t.lower().split())
def _hash(t): return hashlib.sha1(_norm(t).encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_jbb():
    spec = importlib.util.spec_from_file_location(
        "eval_jbb", ROOT / "data" / "eval_jailbreakbench.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return [r["prompt"].strip() for r in mod.load_attack_prompts() if r.get("prompt", "").strip()]


def _load_harmbench():
    spec = importlib.util.spec_from_file_location(
        "hb", ROOT / "evaluation" / "datasets" / "harmbench.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return [p.strip() for (p, _l) in mod.load(
        categories=["standard", "contextual", "copyright"]) if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.95)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Building reference set (must NOT be in training)...")
    ref = []
    ref += _load_jbb();        print(f"  JailbreakBench: {len(ref)}")
    n = len(ref); ref += _load_harmbench(); print(f"  HarmBench: {len(ref)-n}")
    for hp in HELDOUT_FILES:
        rows = _load_jsonl(hp)
        ref += [(r.get('prompt') or '').strip() for r in rows if (r.get('prompt') or '').strip()]
        print(f"  {hp.name}: {len(rows)}")
    ref = [r for r in ref if r]
    ref_hashes = {_hash(r) for r in ref}
    print(f"  reference total: {len(ref)} ({len(ref_hashes)} unique)\n")

    from sentence_transformers import SentenceTransformer
    try:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    import numpy as np
    enc = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"Encoding reference on {device}...")
    ref_vecs = enc.encode(ref, batch_size=128, normalize_embeddings=True,
                          show_progress_bar=True)

    report = {"cutoff": args.cutoff, "files": {}}
    for path in TRAIN_FILES:
        rows = _load_jsonl(path)
        if not rows:
            print(f"\n(skip missing) {path.relative_to(ROOT)}")
            continue
        prompts = [(r.get("prompt") or "").strip() for r in rows]
        vecs = enc.encode(prompts, batch_size=128, normalize_embeddings=True,
                          show_progress_bar=False)
        max_sim = np.zeros(len(prompts), dtype=np.float32)
        chunk = 4096
        for i in range(0, ref_vecs.shape[0], chunk):
            sims = vecs @ ref_vecs[i:i + chunk].T
            max_sim = np.maximum(max_sim, sims.max(axis=1))

        kept, removed_exact, removed_near = [], 0, 0
        for r, p, s in zip(rows, prompts, max_sim):
            is_exact = _hash(p) in ref_hashes
            is_near = (not is_exact) and (s >= args.cutoff)
            if is_exact:
                removed_exact += 1
            elif is_near:
                removed_near += 1
            else:
                kept.append(r)

        out_path = path.with_suffix(".decontam.jsonl")
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in kept),
            encoding="utf-8")
        n_atk = sum(1 for r in kept if int(r.get("label", 0)) == 1)
        report["files"][str(path.relative_to(ROOT))] = {
            "original": len(rows), "removed_exact": removed_exact,
            "removed_near": removed_near, "kept": len(kept),
            "kept_attacks": n_atk, "kept_benign": len(kept) - n_atk,
            "output": str(out_path.relative_to(ROOT)),
        }
        print(f"\n  {path.name}: {len(rows)} -> {len(kept)}  "
              f"(removed exact={removed_exact}, near={removed_near})")
        print(f"    -> {out_path.relative_to(ROOT)}")

    (OUT / "decontam_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport -> {(OUT / 'decontam_report.json').relative_to(ROOT)}")
    print("Next: python scripts/retrain_pair_v6.py --decontam")


if __name__ == "__main__":
    main()
