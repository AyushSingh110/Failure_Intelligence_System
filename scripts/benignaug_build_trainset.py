r"""
Merge BenignAug negatives into the HarmAug trainset (Phase 2, E29).

Produces the corpus for the symmetric experiment: v6.3b + HarmAug positives +
BenignAug negatives. The comparison chain is then

    v6.3b                     baseline
    v6.3b + HarmAug           E28 — recall up, OR-Bench over-refusal up
    v6.3b + HarmAug + Benign  E29 — does the benign half pay the cost back?

DECONTAMINATION MATTERS MORE HERE, NOT LESS
-------------------------------------------
BenignAug generates safe prompts in the same register as XSTest and
OR-Bench-hard, which are the benchmarks it will be evaluated on. A generated
prompt that duplicates an XSTest item would be trained on and then scored on —
the exact contamination E8 found, and it would manufacture the improvement this
experiment is trying to measure.

So the same filter as harmaug_build_trainset.py runs here: exact match plus
cosine >= 0.95 against AdvBench, HarmBench, JailbreakBench, XSTest (safe and
unsafe), OR-Bench-hard and our own val/test splits, plus dedup against the
existing corpus and against itself.

ONLY TEACHER-APPROVED ROWS ARE USED
-----------------------------------
Rows the judge marked HARMFUL, or whose verdict could not be parsed, are
excluded. Training a genuinely harmful prompt as a negative teaches the
classifier to allow it — a security regression, not a data-quality nit. The
rejected rows stay in the checkpoint so the filter is auditable.

Usage (Windows CMD):

    conda activate failure-engine
    python scripts\benignaug_build_trainset.py
    python scripts\benignaug_build_trainset.py --dry-run

Output:
    data/pair_training/train.harmaug_benignaug.jsonl
    data/benignaug/build_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENIGN_IN  = ROOT / "data" / "benignaug" / "benignaug_prompts.jsonl"
BASE_TRAIN = ROOT / "data" / "pair_training" / "train.harmaug.jsonl"
TRAIN_OUT  = ROOT / "data" / "pair_training" / "train.harmaug_benignaug.jsonl"
REPORT_OUT = ROOT / "data" / "benignaug" / "build_report.json"

HELDOUT = [
    ROOT / "data" / "pair_training" / "val.jsonl",
    ROOT / "data" / "pair_training" / "test.jsonl",
]
OVERREFUSAL = [
    ROOT / "data" / "overrefusal" / "xstest_safe_clean.jsonl",
    ROOT / "data" / "overrefusal" / "xstest_unsafe_clean.jsonl",
    ROOT / "data" / "overrefusal" / "orbench_hard_clean.jsonl",
]


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def _hash(t: str) -> str:
    return hashlib.sha1(_norm(t).encode("utf-8")).hexdigest()


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _prompts(rows: list[dict]) -> list[str]:
    return [(r.get("prompt") or r.get("text") or "").strip() for r in rows]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_reference() -> tuple[list[str], dict, list[str]]:
    ref: list[str] = []
    breakdown: dict[str, int] = {}
    failures: list[str] = []

    loaders = [
        ("advbench", lambda: [p.strip() for (p, _l) in _load_module(
            "advbench", ROOT / "evaluation" / "datasets" / "advbench.py").load() if p.strip()]),
        ("harmbench", lambda: [p.strip() for (p, _l) in _load_module(
            "harmbench", ROOT / "evaluation" / "datasets" / "harmbench.py").load(
            categories=["standard", "contextual", "copyright"]) if p.strip()]),
        ("jailbreakbench", lambda: [r["prompt"].strip() for r in _load_module(
            "eval_jbb", ROOT / "data" / "eval_jailbreakbench.py").load_attack_prompts()
            if r.get("prompt", "").strip()]),
    ]
    for label, fn in loaders:
        try:
            got = fn()
        except Exception as exc:
            msg = f"{label}: {type(exc).__name__}: {exc}"
            failures.append(msg)
            print(f"  [FAIL] {msg}")
            got = []
        breakdown[label] = len(got)
        ref += got
        print(f"  {label:<18} {len(got):>5}")

    for path in HELDOUT + OVERREFUSAL:
        got = [p for p in _prompts(_load_jsonl(path)) if p]
        if not got:
            failures.append(f"{path.name}: empty or missing")
            print(f"  [FAIL] {path.name}: empty or missing")
        breakdown[path.stem] = len(got)
        ref += got
        print(f"  {path.stem:<18} {len(got):>5}")

    return [r for r in ref if r], breakdown, failures


def _build_encoder():
    from fie.onnx_encoder import OnnxEncoder
    enc = OnnxEncoder()
    if enc.available:
        print("  encoder: ONNX MiniLM")
        return enc
    from sentence_transformers import SentenceTransformer
    print("  encoder: sentence-transformers (fallback)")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _encode(enc, texts: list[str], label: str, batch_size: int = 64):
    """Length-bucketed encode; the permutation is inverted before returning."""
    import numpy as np
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    t0 = time.time()
    v = np.asarray(enc.encode([texts[i] for i in order], batch_size=batch_size,
                              normalize_embeddings=True), dtype="float32")
    out = np.empty_like(v)
    out[np.asarray(order)] = v
    print(f"  {label} ({len(texts)}) ... {time.time() - t0:.1f}s")
    return out


def _max_similarity(vecs, ref_vecs, chunk: int = 4096):
    import numpy as np
    out = np.zeros(len(vecs), dtype="float32")
    for i in range(0, ref_vecs.shape[0], chunk):
        out = np.maximum(out, (vecs @ ref_vecs[i:i + chunk].T).max(axis=1))
    return out


def _dedup_within(n_rows: int, vecs, cutoff: float):
    import numpy as np
    kept = np.empty((n_rows, vecs.shape[1]), dtype="float32")
    n_kept = 0
    keep_idx, dropped = [], 0
    for i in range(n_rows):
        if n_kept and float((kept[:n_kept] @ vecs[i]).max()) >= cutoff:
            dropped += 1
            continue
        kept[n_kept] = vecs[i]
        n_kept += 1
        keep_idx.append(i)
    return keep_idx, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.95)
    ap.add_argument("--intra-cutoff", type=float, default=0.95)
    ap.add_argument("--base", default=str(BASE_TRAIN),
                    help="corpus to merge into (default: the HarmAug merge)")
    ap.add_argument("--out", default=str(TRAIN_OUT))
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import numpy as np

    raw = _load_jsonl(BENIGN_IN)
    if not raw:
        print(f"ERROR: no rows in {_rel(BENIGN_IN)}")
        print("Run: python scripts\\benignaug_generate.py --target 2000")
        return 1

    # Teacher gate first — everything downstream assumes these are safe.
    approved = [r for r in raw
                if r.get("teacher_safe", False) and r.get("teacher_parsed", True)]
    n_rej = len(raw) - len(approved)
    print(f"BenignAug generated rows : {len(raw)}")
    print(f"  approved by teacher    : {len(approved)}")
    print(f"  rejected / unparsed    : {n_rej}")
    if not approved:
        print("\nNothing approved by the teacher - nothing to build.")
        return 3

    base = _load_jsonl(Path(args.base))
    if not base:
        print(f"ERROR: base corpus missing or empty: {args.base}")
        print("Build it with: python scripts\\harmaug_build_trainset.py")
        return 1
    n_base_atk = sum(1 for r in base if int(r.get("label", 0)) == 1)
    print(f"Base corpus            : {len(base)} "
          f"({n_base_atk} attack / {len(base) - n_base_atk} benign)\n")

    print("Building contamination reference:")
    ref, ref_breakdown, ref_failures = _build_reference()
    ref_hashes = {_hash(r) for r in ref}
    print(f"  {'TOTAL':<18} {len(ref):>5} ({len(ref_hashes)} unique)\n")

    base_prompts = [p for p in _prompts(base) if p]
    base_hashes = {_hash(p) for p in base_prompts}

    print("Encoding:")
    enc = _build_encoder()
    ba_prompts = _prompts(approved)
    ref_vecs  = _encode(enc, ref, "reference")
    base_vecs = _encode(enc, base_prompts, "base")
    ba_vecs   = _encode(enc, ba_prompts, "benignaug")

    ref_sim  = _max_similarity(ba_vecs, ref_vecs)
    base_sim = _max_similarity(ba_vecs, base_vecs)

    kept, kept_vecs = [], []
    counts = Counter()
    examples: dict[str, list] = {"contaminated_exact": [], "contaminated_near": [],
                                 "dup_of_base": []}

    for row, prompt, rs, bs, vec in zip(approved, ba_prompts, ref_sim, base_sim, ba_vecs):
        if not prompt:
            counts["empty"] += 1
            continue
        h = _hash(prompt)
        if h in ref_hashes:
            counts["contaminated_exact"] += 1
            if len(examples["contaminated_exact"]) < 5:
                examples["contaminated_exact"].append(prompt[:160])
            continue
        if rs >= args.cutoff:
            counts["contaminated_near"] += 1
            if len(examples["contaminated_near"]) < 5:
                examples["contaminated_near"].append(
                    {"prompt": prompt[:160], "max_cosine": round(float(rs), 4)})
            continue
        if h in base_hashes or bs >= args.cutoff:
            counts["dup_of_base"] += 1
            if len(examples["dup_of_base"]) < 5:
                examples["dup_of_base"].append(
                    {"prompt": prompt[:160], "max_cosine": round(float(bs), 4)})
            continue
        kept.append(row)
        kept_vecs.append(vec)

    print("\nContamination filter:")
    print(f"  exact match to a benchmark/held-out prompt : {counts['contaminated_exact']}")
    print(f"  near match (cosine >= {args.cutoff})              : {counts['contaminated_near']}")
    print(f"  duplicate of an existing training row      : {counts['dup_of_base']}")
    print(f"  surviving                                  : {len(kept)}")

    if kept:
        print(f"\nDeduplicating within generated set (cosine >= {args.intra_cutoff}) ...")
        keep_idx, intra = _dedup_within(len(kept), np.vstack(kept_vecs), args.intra_cutoff)
        kept = [kept[i] for i in keep_idx]
        counts["near_duplicate_of_each_other"] = intra
        print(f"  removed near-clones : {intra}")
        print(f"  unique augmentation : {len(kept)}")

    if not kept:
        print("\nNothing survived filtering - nothing to build.")
        return 3

    merged = list(base) + [{
        "prompt":        (r.get("prompt") or "").strip(),
        "label":         0,
        "source":        "benignaug",
        "category":      r.get("category", "benign"),
        "harm_category": r.get("harm_category", ""),
        "style":         r.get("style", ""),
    } for r in kept]

    n_atk = sum(1 for r in merged if int(r.get("label", 0)) == 1)
    n_ben = len(merged) - n_atk
    frac = n_atk / len(merged)
    base_frac = n_base_atk / len(base)

    print("\nMerged training set:")
    print(f"  total   : {len(merged)}  (base {len(base)} + benignaug {len(kept)})")
    print(f"  attack  : {n_atk}  ({frac:.1%})")
    print(f"  benign  : {n_ben}  ({1 - frac:.1%})")
    print(f"  attack fraction moved {base_frac:.1%} -> {frac:.1%} "
          f"({(frac - base_frac) * 100:+.1f} pts)")

    report = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "cutoff":        args.cutoff,
        "intra_cutoff":  args.intra_cutoff,
        "benignaug_generated": len(raw),
        "teacher_rejected":    n_rej,
        "teacher_approved":    len(approved),
        "removed":       dict(counts),
        "kept_augmentation": len(kept),
        "base_rows":     len(base),
        "base_attack_frac": round(base_frac, 4),
        "merged_rows":   len(merged),
        "merged_attack": n_atk,
        "merged_benign": n_ben,
        "attack_fraction": round(frac, 4),
        "reference_breakdown": ref_breakdown,
        "reference_total": len(ref),
        "reference_failures": ref_failures,
        "harm_category_distribution": dict(Counter(
            r.get("harm_category", "?") for r in kept)),
        "style_distribution": dict(Counter(r.get("style", "?") for r in kept)),
        "examples_removed": examples,
        "note": "val.jsonl and test.jsonl untouched; no generated row enters any "
                "held-out split. Only teacher-approved benign rows are merged.",
    }

    if args.dry_run:
        print("\n(dry run - nothing written)")
        return 0

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report_path = Path(args.report).resolve() if args.report else (
        REPORT_OUT if out_path == TRAIN_OUT
        else REPORT_OUT.with_name(f"{out_path.stem}_report.json"))
    report["output_file"] = str(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nwrote {_rel(out_path)}   ({len(merged)} rows)")
    print(f"wrote {_rel(report_path)}")
    print("\nnext: python scripts\\train_pair_harmaug.py --base "
          f"{_rel(out_path)} --tag v65_benignaug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
