r"""
Turn generated HarmAug prompts into a trainable, uncontaminated PAIR dataset.

This is step 2 of the HarmAug reproduction. Step 1 (`harmaug_generate.py`)
produced raw positives; this script decides which of them may legitimately be
trained on, merges them with the existing decontaminated training set, and
writes a new file that `retrain_pair_*.py` can consume.

WHAT THIS REMOVES, AND WHY IT IS THE WHOLE POINT
------------------------------------------------
The generated prompts came from an LLM that has read the entire public
red-teaming literature. Asking it for "a harmful instruction prompt about
weapons" will, some fraction of the time, reproduce an AdvBench or HarmBench
behaviour close to verbatim. Training on those and then reporting recall on the
same benchmarks would be measuring memorisation and calling it generalisation —
exactly the failure E8 already found in this repo (148 JailbreakBench prompts
sitting inside PAIR's training set).

So every generated prompt is checked against:

  * val.jsonl / test.jsonl        — our own held-out splits
  * AdvBench, HarmBench, JailbreakBench   — attack benchmarks we report on
  * XSTest (safe + unsafe), OR-Bench-hard — over-refusal benchmarks we report on
  * the base training set          — duplicates add weight, not information
  * each other                     — temperature 1.0 produces near-clones

Removal criterion matches scripts/decontaminate_training.py: EXACT after
whitespace/case normalisation, or cosine >= --cutoff (0.95 = near-identical
text, not merely the same topic).

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not touch val.jsonl or test.jsonl. Not one generated row enters the
held-out splits. The whole comparison — does HarmAug augmentation help? — is
only meaningful if before and after are scored on the *same untouched* eval
data.

THE CLASS-BALANCE PROBLEM (read this before training)
-----------------------------------------------------
The base set is 1185 benign / 1201 attack — near-perfect balance. HarmAug
generates positives ONLY, so a faithful merge lands around 73% attack.

That is not a bug in this script; it is what the HarmAug recipe actually does.
It matters here more than it did in the original paper because FIE's largest
documented weakness is the opposite failure: 53.6% of safe XSTest prompts and
90.4% of OR-Bench-hard prompts are already flagged. Skewing the training
distribution further toward "attack" is expected to make that worse.

That expectation is the experiment. Train on this, measure BOTH attack recall
and over-refusal, and report the trade-off — the HarmAug paper reports only the
first half, and so does the 14-model benchmark that followed it. Measuring the
cost is the contribution; `--max-pos-frac` and Phase 2 (BenignAug) are the fix.

Usage (Windows CMD):

    conda activate failure-engine
    python scripts\harmaug_build_trainset.py
    python scripts\harmaug_build_trainset.py --max-pos-frac 0.5   REM capped variant
    python scripts\harmaug_build_trainset.py --dry-run            REM report only

Output:
    data/pair_training/train.harmaug.jsonl   merged training set
    data/harmaug/build_report.json           what was removed and why
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HARMAUG_IN   = ROOT / "data" / "harmaug" / "harmaug_prompts.jsonl"
BASE_TRAIN   = ROOT / "data" / "pair_training" / "train.decontam.jsonl"
TRAIN_OUT    = ROOT / "data" / "pair_training" / "train.harmaug.jsonl"
REPORT_OUT   = ROOT / "data" / "harmaug" / "build_report.json"

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
                continue          # tolerate a torn final line from a hard kill
    return rows


def _rel(p: Path) -> str:
    """Display path relative to the repo when possible, absolute otherwise."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _prompts(rows: list[dict]) -> list[str]:
    return [(r.get("prompt") or r.get("text") or "").strip() for r in rows]


# ── Benchmark loaders ────────────────────────────────────────────────────────
#
# Each is wrapped: a benchmark that fails to load must not silently shrink the
# reference set, because a smaller reference set means LESS decontamination and
# a quietly inflated recall number later. Failures are collected and reported,
# and --strict turns them into a hard stop.

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_advbench() -> list[str]:
    mod = _load_module("advbench", ROOT / "evaluation" / "datasets" / "advbench.py")
    return [p.strip() for (p, _l) in mod.load() if p.strip()]


def _load_harmbench() -> list[str]:
    mod = _load_module("harmbench", ROOT / "evaluation" / "datasets" / "harmbench.py")
    return [p.strip() for (p, _l) in
            mod.load(categories=["standard", "contextual", "copyright"]) if p.strip()]


def _load_jbb() -> list[str]:
    mod = _load_module("eval_jbb", ROOT / "data" / "eval_jailbreakbench.py")
    return [r["prompt"].strip() for r in mod.load_attack_prompts()
            if r.get("prompt", "").strip()]


def _build_reference(strict: bool) -> tuple[list[str], dict, list[str]]:
    """Every prompt a generated row must not resemble."""
    ref: list[str] = []
    breakdown: dict[str, int] = {}
    failures: list[str] = []

    for label, loader in (("advbench", _load_advbench),
                          ("harmbench", _load_harmbench),
                          ("jailbreakbench", _load_jbb)):
        try:
            got = loader()
        except Exception as exc:
            msg = f"{label}: {type(exc).__name__}: {exc}"
            failures.append(msg)
            print(f"  [FAIL] {msg}")
            if strict:
                continue
            got = []
        breakdown[label] = len(got)
        ref += got
        print(f"  {label:<18} {len(got):>5}")

    for path in HELDOUT + OVERREFUSAL:
        rows = _load_jsonl(path)
        got = [p for p in _prompts(rows) if p]
        if not got:
            msg = f"{path.name}: empty or missing"
            failures.append(msg)
            print(f"  [FAIL] {msg}")
        breakdown[path.stem] = len(got)
        ref += got
        print(f"  {path.stem:<18} {len(got):>5}")

    ref = [r for r in ref if r]
    return ref, breakdown, failures


def _build_encoder():
    """
    ONNX first, sentence-transformers second — same order as fie/layers/pair.py.

    The runtime dropped torch (4.5 GB) in favour of ONNX; this script should not
    quietly reintroduce the dependency just because a training box happens to
    have it installed.
    """
    try:
        from fie.onnx_encoder import OnnxEncoder
        enc = OnnxEncoder()
        if enc.available:
            print("  encoder: ONNX MiniLM")
            return enc
    except Exception as exc:
        print(f"  onnx unavailable ({type(exc).__name__}: {exc})")

    from sentence_transformers import SentenceTransformer
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"  encoder: sentence-transformers on {device}")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def _encode(enc, texts: list[str], label: str, batch_size: int = 64):
    """
    Encode with length bucketing, then restore the original order.

    The tokenizer pads to the longest text IN THE BATCH. This corpus has a
    median around 11 words but contains 250+ word GCG suffixes, so a randomly
    ordered batch pads almost everything out to the 256-token cap — every short
    prompt then costs a full-length forward pass. Sorting by length first makes
    each batch homogeneous, which is a large constant-factor win (the first
    version of this script spent over ten minutes here).

    The permutation is inverted before returning, so callers still get vectors
    in input order and every downstream index stays valid.
    """
    import numpy as np
    if not texts:
        return np.zeros((0, 384), dtype="float32")

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]

    t0 = time.time()
    vecs_sorted = enc.encode(sorted_texts, batch_size=batch_size,
                             normalize_embeddings=True)
    vecs_sorted = np.asarray(vecs_sorted, dtype="float32")

    out = np.empty_like(vecs_sorted)
    out[np.asarray(order)] = vecs_sorted          # invert the permutation
    print(f"  {label} ({len(texts)}) ... {time.time() - t0:.1f}s")
    return out


def _max_similarity(vecs, ref_vecs, chunk: int = 4096):
    """Per-row maximum cosine against the reference, chunked to bound memory."""
    import numpy as np
    out = np.zeros(len(vecs), dtype="float32")
    for i in range(0, ref_vecs.shape[0], chunk):
        sims = vecs @ ref_vecs[i:i + chunk].T
        out = np.maximum(out, sims.max(axis=1))
    return out


def _dedup_within(rows: list[dict], vecs, cutoff: float) -> tuple[list[int], int]:
    """
    Greedy near-duplicate removal inside the generated set.

    Temperature 1.0 with a fixed category/style grid produces clusters of
    near-clones. The generator already dropped exact repeats by fingerprint, but
    "Explain how to pick a lock" and "Explain how to pick a lock on a door" are
    distinct strings and near-identical training signal. Keeping both does not
    add information; it silently upweights whatever the model likes generating.
    """
    import numpy as np
    # Preallocated, not a list of vectors re-stacked each iteration: vstack in
    # the loop is O(n^2) in memory traffic and dominated the runtime for no
    # reason. A single buffer with a fill counter is the same algorithm.
    kept = np.empty((len(rows), vecs.shape[1]), dtype="float32")
    n_kept = 0
    keep_idx: list[int] = []
    dropped = 0
    for i in range(len(rows)):
        if n_kept and float((kept[:n_kept] @ vecs[i]).max()) >= cutoff:
            dropped += 1
            continue
        kept[n_kept] = vecs[i]
        n_kept += 1
        keep_idx.append(i)
    return keep_idx, dropped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", type=float, default=0.95,
                    help="cosine >= this against a benchmark = contaminated (default 0.95)")
    ap.add_argument("--intra-cutoff", type=float, default=0.95,
                    help="cosine >= this between two generated rows = duplicate")
    ap.add_argument("--max-pos-frac", type=float, default=None,
                    help="cap attacks at this fraction of the merged set by "
                         "subsampling HarmAug rows (e.g. 0.5). Default: no cap "
                         "— the faithful HarmAug recipe.")
    ap.add_argument("--base", default=str(BASE_TRAIN),
                    help="base training file to merge into")
    ap.add_argument("--out", default=str(TRAIN_OUT))
    ap.add_argument("--report", default=None,
                    help="where to write the provenance report "
                         "(default: derived from --out)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen; write nothing")
    ap.add_argument("--strict", action="store_true",
                    help="abort if any benchmark fails to load (recommended "
                         "before publishing numbers)")
    args = ap.parse_args()

    import numpy as np

    # ── Load generated positives ─────────────────────────────────────────────
    harmaug = _load_jsonl(HARMAUG_IN)
    if not harmaug:
        print(f"ERROR: no rows in {HARMAUG_IN}")
        print("Run: python scripts\\harmaug_generate.py --target 2000")
        return 1
    print(f"HarmAug generated rows : {len(harmaug)}")

    base = _load_jsonl(Path(args.base))
    if not base:
        print(f"ERROR: base training file is empty or missing: {args.base}")
        return 1
    n_base_atk = sum(1 for r in base if int(r.get("label", 0)) == 1)
    print(f"Base training rows     : {len(base)} "
          f"({n_base_atk} attack / {len(base) - n_base_atk} benign)\n")

    # ── Reference set ────────────────────────────────────────────────────────
    print("Building contamination reference (rows generated data must NOT resemble):")
    ref, ref_breakdown, ref_failures = _build_reference(args.strict)
    ref_hashes = {_hash(r) for r in ref}
    print(f"  {'TOTAL':<18} {len(ref):>5} ({len(ref_hashes)} unique)\n")

    if ref_failures and args.strict:
        print("ABORTING (--strict): reference set is incomplete, so decontamination")
        print("would be weaker than reported. Failures:")
        for f in ref_failures:
            print(f"  - {f}")
        return 2

    # Base training set is a second reference: duplicates of it are dropped from
    # the augmentation, but they are NOT contamination — different reason, so
    # counted separately.
    base_prompts = [p for p in _prompts(base) if p]
    base_hashes = {_hash(p) for p in base_prompts}

    # ── Encode ───────────────────────────────────────────────────────────────
    print("Encoding:")
    enc = _build_encoder()
    ha_prompts = _prompts(harmaug)
    ref_vecs  = _encode(enc, ref, "reference")
    base_vecs = _encode(enc, base_prompts, "base")
    ha_vecs   = _encode(enc, ha_prompts, "harmaug")

    ref_sim = _max_similarity(ha_vecs, ref_vecs)
    base_sim = _max_similarity(ha_vecs, base_vecs)

    # ── Filter ───────────────────────────────────────────────────────────────
    kept: list[dict] = []
    kept_vecs: list = []
    counts = Counter()
    examples: dict[str, list] = {"contaminated_exact": [], "contaminated_near": [],
                                 "dup_of_base": []}

    for row, prompt, rs, bs, vec in zip(harmaug, ha_prompts, ref_sim, base_sim, ha_vecs):
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

    print(f"\nContamination filter:")
    print(f"  exact match to a benchmark/held-out prompt : {counts['contaminated_exact']}")
    print(f"  near match (cosine >= {args.cutoff})              : {counts['contaminated_near']}")
    print(f"  duplicate of an existing training row      : {counts['dup_of_base']}")
    print(f"  surviving                                  : {len(kept)}")

    # ── Intra-set deduplication ──────────────────────────────────────────────
    if kept:
        print(f"\nDeduplicating within generated set (cosine >= {args.intra_cutoff}) ...")
        keep_idx, intra_dropped = _dedup_within(kept, np.vstack(kept_vecs), args.intra_cutoff)
        kept = [kept[i] for i in keep_idx]
        counts["near_duplicate_of_each_other"] = intra_dropped
        print(f"  removed near-clones : {intra_dropped}")
        print(f"  unique augmentation : {len(kept)}")

    if not kept:
        print("\nNothing survived filtering - nothing to build.")
        return 3

    # ── Optional class-balance cap ───────────────────────────────────────────
    capped = 0
    if args.max_pos_frac is not None:
        if not 0.0 < args.max_pos_frac < 1.0:
            print("ERROR: --max-pos-frac must be strictly between 0 and 1")
            return 1
        n_benign = len(base) - n_base_atk
        base_frac = n_base_atk / len(base)
        # Solve (n_base_atk + k) / (len(base) + k) <= f  for k.
        f = args.max_pos_frac
        allowed = max(0, int((f * len(base) - n_base_atk) / (1.0 - f)))

        # A positives-only augmentation can never LOWER the attack fraction, so
        # a target at or below what the base already sits at admits zero rows.
        # Writing a file called "...harmaug..." that contains no HarmAug data
        # would be a silent trap: it trains fine, scores like the baseline, and
        # the experiment looks like "augmentation had no effect".
        if allowed == 0:
            print(f"\nERROR: --max-pos-frac {f} admits 0 augmentation rows.")
            print(f"  The base set is ALREADY {base_frac:.1%} attack "
                  f"({n_base_atk} attack / {n_benign} benign).")
            print("  HarmAug only generates positives, so no number of added rows")
            print("  can bring the fraction DOWN to your target.")
            suggestions = [x for x in (0.60, 0.65, 0.70)
                           if int((x * len(base) - n_base_atk) / (1.0 - x)) > 0]
            if suggestions:
                print("\n  Workable targets (rows admitted):")
                for x in suggestions:
                    k = int((x * len(base) - n_base_atk) / (1.0 - x))
                    print(f"    --max-pos-frac {x:.2f}  ->  {min(k, len(kept))} rows")
            print("\n  To actually reach 50% you need benign augmentation "
                  "(Phase 2 / BenignAug),")
            print("  not a cap on this one. Nothing was written.")
            return 4

        if allowed < len(kept):
            rng = random.Random(args.seed)
            # Stratify by category so capping does not silently delete a whole
            # harm class — a uniform sample of 400 from 12 categories is fine in
            # expectation and lumpy in practice.
            by_cat: dict[str, list[dict]] = {}
            for r in kept:
                by_cat.setdefault(r.get("category", "?"), []).append(r)
            for rows in by_cat.values():
                rng.shuffle(rows)
            selected: list[dict] = []
            cats = sorted(by_cat)
            i = 0
            while len(selected) < allowed:
                pool = by_cat[cats[i % len(cats)]]
                if pool:
                    selected.append(pool.pop())
                i += 1
                if all(not by_cat[c] for c in cats):
                    break
            capped = len(kept) - len(selected)
            kept = selected
            print(f"\nClass-balance cap (--max-pos-frac {f}):")
            print(f"  benign in base      : {n_benign}")
            print(f"  attacks allowed     : {allowed}")
            print(f"  dropped by cap      : {capped}")

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = list(base) + [{
        "prompt":   (r.get("prompt") or "").strip(),
        "label":    1,
        "source":   "harmaug",
        "category": r.get("category", "harmful"),
        # Kept so a later analysis can ask which phrasings actually helped —
        # the styles are the only axis of variation this augmentation has.
        "style":    r.get("style", ""),
    } for r in kept]

    n_atk = sum(1 for r in merged if int(r.get("label", 0)) == 1)
    n_ben = len(merged) - n_atk
    pos_frac = n_atk / len(merged)

    print(f"\nMerged training set:")
    print(f"  total   : {len(merged)}  (base {len(base)} + harmaug {len(kept)})")
    print(f"  attack  : {n_atk}  ({pos_frac:.1%})")
    print(f"  benign  : {n_ben}  ({1 - pos_frac:.1%})")

    if pos_frac > 0.60:
        print(f"\n  ! {pos_frac:.0%} of this set is labelled 'attack'.")
        print("    FIE already over-refuses (53.6% XSTest / 90.4% OR-Bench-hard).")
        print("    Expect this to get WORSE, and measure it - that measurement is")
        print("    the point of the experiment, not a side effect. To build a")
        print("    less skewed variant instead:")
        print("      python scripts\\harmaug_build_trainset.py --max-pos-frac 0.60 ^")
        print("             --out data\\pair_training\\train.harmaug_capped.jsonl")
        print("    (0.50 is unreachable - the base is already 50.3% attack and")
        print("     HarmAug adds positives only. See Phase 2 / BenignAug.)")

    report = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "cutoff":         args.cutoff,
        "intra_cutoff":   args.intra_cutoff,
        "max_pos_frac":   args.max_pos_frac,
        "seed":           args.seed,
        "harmaug_input":  len(harmaug),
        "removed":        dict(counts),
        "removed_by_cap": capped,
        "kept_augmentation": len(kept),
        "base_rows":      len(base),
        "merged_rows":    len(merged),
        "merged_attack":  n_atk,
        "merged_benign":  n_ben,
        "positive_fraction": round(pos_frac, 4),
        "reference_breakdown": ref_breakdown,
        "reference_total": len(ref),
        "reference_failures": ref_failures,
        "category_distribution": dict(Counter(r.get("category", "?") for r in kept)),
        "style_distribution": dict(Counter(r.get("style", "?") for r in kept)),
        "examples_removed": examples,
        "note": "val.jsonl and test.jsonl are untouched; no generated row enters "
                "any held-out split.",
    }

    if args.dry_run:
        print("\n(dry run - nothing written)")
        return 0

    # Resolve before use: a relative --out (the natural thing to type) is not
    # under ROOT as written, and .relative_to() raises on it.
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report name follows the output name. A fixed report path meant a second
    # build with a different --out silently overwrote the first build's report,
    # leaving a dataset whose provenance record described a different dataset.
    report_path = Path(args.report).resolve() if args.report else (
        REPORT_OUT if out_path == TRAIN_OUT
        else REPORT_OUT.with_name(f"{out_path.stem}_report.json"))
    report["output_file"] = str(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nwrote {_rel(out_path)}   ({len(merged)} rows)")
    print(f"wrote {_rel(report_path)}")
    print("\nnext: retrain PAIR on this file, then measure BOTH")
    print("      attack recall and over-refusal against the v6.3 baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
