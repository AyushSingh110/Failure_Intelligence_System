"""
FIE Adversarial Detection Evaluation Harness — FIE-Eval-200

Evaluates FIE adversarial detection (all 8 layers) against a labeled JSONL dataset.
Uses the /diagnose endpoint — no Groq shadow-model calls, no rate-limit risk.

Usage:
    # Make sure FIE server is running first:
    #   uvicorn app.main:app --reload

    # Validate dataset without calling the API:
    python data/eval_adversarial.py --dry-run

    # Run all 200 prompts (single pass):
    python data/eval_adversarial.py

    # Seed-based chunked evaluation (40 prompts × 5 seeds, saves after each seed):
    python data/eval_adversarial.py --seeds 5 --chunk-size 40 --seed 1
    python data/eval_adversarial.py --seeds 5 --chunk-size 40 --seed 2
    # ... repeat for seeds 3-5, or omit --seed to run all seeds sequentially:
    python data/eval_adversarial.py --seeds 5 --chunk-size 40

    # Merge all saved seed result files into a final combined report:
    python data/eval_adversarial.py --merge

    # Per-layer ablation study:
    python data/eval_adversarial.py --layers-ablation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATTACK_LABEL = "attack"
BENIGN_LABEL = "benign"

ALL_LAYERS = [
    "regex",
    "prompt_guard",
    "faiss",
    "indirect_injection",
    "gcg_suffix",
    "perplexity_proxy",
    "exfiltration",
    "semantic_consistency",
]

ATTACK_CATEGORIES = [
    "direct_injection",
    "jailbreak_persona",
    "jailbreak_roleplay",
    "instruction_override",
    "token_smuggling",
    "indirect_injection",
    "obfuscated_attack",
]

BENIGN_CATEGORIES = [
    "benign_general",
    "benign_creative",
    "benign_technical",
]

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FIE Adversarial Detection Evaluation Harness — FIE-Eval-200",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="data/adversarial_eval_set.jsonl",
        help="Path to the labeled JSONL evaluation dataset (default: data/adversarial_eval_set.jsonl)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to save the JSON results report. Defaults to data/eval_results_<TIMESTAMP>.json",
    )
    p.add_argument(
        "--fie-url",
        default="http://localhost:8000/api/v1",
        help="Base URL of the FIE API (default: http://localhost:8000/api/v1)",
    )
    p.add_argument(
        "--api-key",
        default="",
        help="Optional API key / bearer token to pass in Authorization header",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds to wait between API calls (default: 0.3 — /diagnose is fast, no Groq calls)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per call in seconds (default: 30)",
    )
    p.add_argument(
        "--layers-ablation",
        action="store_true",
        help="Run per-layer ablation study after the main evaluation",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the dataset without making any API calls",
    )
    # ── Seed-based chunked evaluation ──────────────────────────────────────
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        help=(
            "Total number of seeds / chunks to split the dataset into (default: 1 = run all). "
            "Example: --seeds 5 splits 200 prompts into 5 chunks of 40."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Which seed (1-based) to run. "
            "If omitted and --seeds > 1, all seeds are run sequentially."
        ),
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=40,
        help="Number of prompts per seed/chunk (default: 40)",
    )
    p.add_argument(
        "--seed-dir",
        default="data/eval_seeds",
        help="Directory to save per-seed intermediate result files (default: data/eval_seeds)",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Merge all saved seed result files from --seed-dir into a final combined report. "
            "Does not call the API — reads previously saved JSONL files."
        ),
    )
    return p

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """Load and validate a JSONL evaluation dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {path}", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    required_fields = {"prompt", "label", "category"}
    parse_errors = 0

    with dataset_path.open("r", encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {i}: JSON parse error — {exc}", file=sys.stderr)
                parse_errors += 1
                continue

            missing = required_fields - set(obj.keys())
            if missing:
                print(f"[WARN] Line {i}: missing fields {missing} — skipping", file=sys.stderr)
                parse_errors += 1
                continue

            if obj["label"] not in (ATTACK_LABEL, BENIGN_LABEL):
                print(
                    f"[WARN] Line {i}: invalid label '{obj['label']}' — skipping",
                    file=sys.stderr,
                )
                parse_errors += 1
                continue

            records.append(obj)

    if parse_errors:
        print(f"[WARN] {parse_errors} lines skipped due to parse/validation errors.")

    return records


def print_dataset_stats(records: list[dict], dataset_path: str) -> None:
    attack_records = [r for r in records if r["label"] == ATTACK_LABEL]
    benign_records = [r for r in records if r["label"] == BENIGN_LABEL]

    cat_counts: dict[str, int] = {}
    for r in records:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    print("=" * 60)
    print("  FIE-Eval-200 — Dataset Validation (dry-run)")
    print(f"  File   : {dataset_path}")
    print(f"  Total  : {len(records)} prompts")
    print("=" * 60)
    print(f"  Attack prompts : {len(attack_records)}")
    print(f"  Benign prompts : {len(benign_records)}")
    print()
    print("  Per-Category Distribution:")
    for cat in sorted(cat_counts):
        label_type = "attack" if cat in ATTACK_CATEGORIES else "benign"
        print(f"    {cat:<28} : {cat_counts[cat]:>3}  ({label_type})")
    print("=" * 60)
    print("  Dataset looks valid. Ready for evaluation.")
    print("=" * 60)

# ---------------------------------------------------------------------------
# API call — /diagnose (no Groq calls, no rate-limit risk)
# ---------------------------------------------------------------------------

def call_diagnose(
    prompt: str,
    fie_base_url: str,
    api_key: str,
    timeout: float,
) -> dict | None:
    """
    POST to /diagnose endpoint with an empty primary output.
    All 8 adversarial detection layers run identically to /monitor
    but without triggering Groq shadow-model calls.
    Returns the JSON response dict or None on error.
    """
    url = fie_base_url.rstrip("/") + "/diagnose"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "prompt":        prompt,
        "model_outputs": [""],   # empty primary output — adversarial detection is prompt-driven
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(
            f"\n[WARN] Connection error — is the FIE server running at {fie_base_url}?",
            file=sys.stderr,
        )
        return None
    except requests.exceptions.Timeout:
        print(f"\n[WARN] Timeout after {timeout}s calling {url}", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as exc:
        print(f"\n[WARN] HTTP error {exc.response.status_code}: {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"\n[WARN] Unexpected error calling {url}: {exc}", file=sys.stderr)
        return None


def extract_verdict(response: dict | None) -> dict:
    """
    Extract verdict fields from a /diagnose (or /monitor) response.

    Returns a dict with keys:
        is_adversarial      : bool
        jury_confidence     : float
        root_cause          : str | None
        detection_layers    : list[str]
        api_error           : bool
    """
    if response is None:
        return {
            "is_adversarial":   False,
            "jury_confidence":  0.0,
            "root_cause":       None,
            "detection_layers": [],
            "api_error":        True,
        }

    jury = response.get("jury") or {}
    is_adversarial  = bool(jury.get("is_adversarial", False))
    jury_confidence = float(jury.get("jury_confidence", 0.0))

    primary_verdict  = jury.get("primary_verdict") or {}
    root_cause       = primary_verdict.get("root_cause")
    evidence         = primary_verdict.get("evidence") or {}
    detection_layers = evidence.get("detection_layers_fired") or []

    return {
        "is_adversarial":   is_adversarial,
        "jury_confidence":  jury_confidence,
        "root_cause":       root_cause,
        "detection_layers": detection_layers,
        "api_error":        False,
    }

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    error_count = sum(1 for r in results if r.get("api_error"))

    for r in results:
        if r.get("api_error"):
            continue
        truth = r["label"] == ATTACK_LABEL
        pred  = bool(r["is_adversarial"])
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1

    attack_total = tp + fn
    benign_total = fp + tn

    recall    = tp / attack_total if attack_total > 0 else 0.0
    fpr       = fp / benign_total if benign_total > 0 else 0.0
    precision = tp / (tp + fp)   if (tp + fp) > 0   else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "tp":           tp,
        "fp":           fp,
        "fn":           fn,
        "tn":           tn,
        "attack_total": attack_total,
        "benign_total": benign_total,
        "recall":       round(recall,    4),
        "fpr":          round(fpr,       4),
        "precision":    round(precision, 4),
        "f1":           round(f1,        4),
        "accuracy":     round(accuracy,  4),
        "error_count":  error_count,
    }


def compute_per_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    per_cat: dict[str, dict] = {}
    for cat, cat_results in by_cat.items():
        non_error = [r for r in cat_results if not r.get("api_error")]
        total     = len(non_error)
        if total == 0:
            per_cat[cat] = {"total": 0, "detected": 0, "rate": 0.0}
            continue

        if cat in ATTACK_CATEGORIES:
            detected = sum(1 for r in non_error if r["is_adversarial"])
            per_cat[cat] = {
                "total":    total,
                "detected": detected,
                "rate":     round(detected / total, 4),
                "type":     "recall",
            }
        else:
            flagged = sum(1 for r in non_error if r["is_adversarial"])
            per_cat[cat] = {
                "total":   total,
                "flagged": flagged,
                "rate":    round(flagged / total, 4),
                "type":    "fpr",
            }

    return per_cat


def compute_per_layer(results: list[dict]) -> dict[str, dict]:
    layer_total: dict[str, int] = {layer: 0 for layer in ALL_LAYERS}
    layer_incremental: dict[str, int] = {layer: 0 for layer in ALL_LAYERS}

    for r in results:
        if r.get("api_error") or r["label"] != ATTACK_LABEL or not r["is_adversarial"]:
            continue
        layers_fired = r.get("detection_layers", [])
        for layer in ALL_LAYERS:
            if layer in layers_fired:
                layer_total[layer] += 1

    for r in results:
        if r.get("api_error") or r["label"] != ATTACK_LABEL or not r["is_adversarial"]:
            continue
        layers_fired = r.get("detection_layers", [])
        for layer in ALL_LAYERS:
            if layer in layers_fired:
                layer_incremental[layer] += 1
                break

    return {
        "total":       layer_total,
        "incremental": layer_incremental,
    }

# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(
    dataset_path: str,
    records: list[dict],
    results: list[dict],
    metrics: dict,
    per_category: dict[str, dict],
    per_layer: dict[str, dict],
    seed_info: str = "",
) -> None:
    SEP = "=" * 60

    print()
    print(SEP)
    print("  FIE Adversarial Detection Evaluation")
    print(f"  Dataset : {dataset_path}  ({len(records)} prompts)")
    if seed_info:
        print(f"  Scope   : {seed_info}")
    print(SEP)
    print()

    m = metrics
    print("Overall Results:")
    print(f"  Recall    : {m['recall'] * 100:.1f}%  ({m['tp']}/{m['attack_total']} attacks detected)")
    print(f"  FPR       : {m['fpr'] * 100:.1f}%   ({m['fp']}/{m['benign_total']} benign prompts falsely flagged)")
    print(f"  Precision : {m['precision'] * 100:.1f}%")
    print(f"  F1        : {m['f1'] * 100:.1f}%")
    print(f"  Accuracy  : {m['accuracy'] * 100:.1f}%")
    print()

    print("Per-Category Recall:")
    for cat in ATTACK_CATEGORIES:
        info     = per_category.get(cat, {})
        total    = info.get("total", 0)
        detected = info.get("detected", 0)
        rate     = info.get("rate", 0.0)
        print(f"  {cat:<28} : {rate * 100:>6.1f}%  ({detected}/{total})")
    print()

    print("Per-Category False Positive Rate:")
    for cat in BENIGN_CATEGORIES:
        info    = per_category.get(cat, {})
        total   = info.get("total", 0)
        flagged = info.get("flagged", 0)
        rate    = info.get("rate", 0.0)
        print(f"  {cat:<28} : {rate * 100:>6.1f}%  ({flagged}/{total} flagged)")
    print()

    print("Per-Layer Contribution:")
    layer_total       = per_layer.get("total", {})
    layer_incremental = per_layer.get("incremental", {})
    for i, layer in enumerate(ALL_LAYERS):
        total_count = layer_total.get(layer, 0)
        incr_count  = layer_incremental.get(layer, 0)
        if i == 0:
            note = f"caught {total_count:>3} attacks"
        else:
            note = f"caught {total_count:>3} attacks  (+{incr_count} new)"
        print(f"  {layer:<22} : {note}")
    print()
    print(f"Errors: {m['error_count']} API call(s) failed")
    print(SEP)
    print()

# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(results: list[dict]) -> dict[str, dict]:
    baseline_metrics = compute_metrics(results)
    ablation_results: dict[str, dict] = {}

    for ablated_layer in ALL_LAYERS:
        modified_results = []
        for r in results:
            if r.get("api_error"):
                modified_results.append(r)
                continue
            layers_fired    = r.get("detection_layers", [])
            remaining       = [l for l in layers_fired if l != ablated_layer]
            new_verdict     = len(remaining) > 0
            modified_results.append({**r, "is_adversarial": new_verdict})

        abl_metrics = compute_metrics(modified_results)
        ablation_results[ablated_layer] = {
            "recall":       abl_metrics["recall"],
            "fpr":          abl_metrics["fpr"],
            "recall_drop":  round(baseline_metrics["recall"] - abl_metrics["recall"], 4),
            "fpr_drop":     round(baseline_metrics["fpr"]    - abl_metrics["fpr"],    4),
        }

    return ablation_results


def print_ablation_table(baseline: dict, ablation: dict[str, dict]) -> None:
    SEP = "=" * 70
    print()
    print(SEP)
    print("  Per-Layer Ablation Study")
    print("  (Impact of removing each detection layer)")
    print(SEP)
    print(f"  {'Layer':<26}  {'Recall':>8}  {'FPR':>8}  {'ΔRecall':>9}  {'ΔFPR':>8}")
    print("-" * 70)
    print(
        f"  {'BASELINE (all layers)':<26}  "
        f"{baseline['recall'] * 100:>7.1f}%  "
        f"{baseline['fpr'] * 100:>7.1f}%  "
        f"{'—':>9}  {'—':>8}"
    )
    print("-" * 70)
    for layer in ALL_LAYERS:
        info        = ablation.get(layer, {})
        recall      = info.get("recall", 0.0)
        fpr         = info.get("fpr", 0.0)
        recall_drop = info.get("recall_drop", 0.0)
        fpr_drop    = info.get("fpr_drop", 0.0)
        rd_str = f"-{recall_drop * 100:.1f}%" if recall_drop >= 0 else f"+{-recall_drop * 100:.1f}%"
        fd_str = f"-{fpr_drop * 100:.1f}%"    if fpr_drop    >= 0 else f"+{-fpr_drop    * 100:.1f}%"
        print(
            f"  w/o {layer:<22}  "
            f"{recall * 100:>7.1f}%  "
            f"{fpr * 100:>7.1f}%  "
            f"{rd_str:>9}  "
            f"{fd_str:>8}"
        )
    print(SEP)
    print()

# ---------------------------------------------------------------------------
# Seed-based intermediate file helpers
# ---------------------------------------------------------------------------

def seed_result_path(seed_dir: str, seed_num: int) -> Path:
    return Path(seed_dir) / f"seed_{seed_num:02d}.jsonl"


def save_seed_results(seed_dir: str, seed_num: int, results: list[dict]) -> Path:
    """Append per-prompt result dicts to a per-seed JSONL file."""
    path = seed_result_path(seed_dir, seed_num)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Seed {seed_num} results saved → {path}  ({len(results)} records)")
    return path


def load_seed_results(seed_dir: str) -> list[dict]:
    """Load and merge all seed_NN.jsonl files from seed_dir."""
    seed_dir_path = Path(seed_dir)
    if not seed_dir_path.exists():
        print(f"[ERROR] Seed directory not found: {seed_dir}", file=sys.stderr)
        sys.exit(1)

    seed_files = sorted(seed_dir_path.glob("seed_*.jsonl"))
    if not seed_files:
        print(f"[ERROR] No seed_*.jsonl files found in {seed_dir}", file=sys.stderr)
        sys.exit(1)

    all_results: list[dict] = []
    for f in seed_files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
        print(f"  Loaded {f.name}  ({sum(1 for _ in open(f))} records)")

    # Sort by original index to restore dataset order
    all_results.sort(key=lambda r: r.get("idx", 0))
    return all_results

# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def _classify_outcome(label: str, predicted_adversarial: bool, api_error: bool) -> str:
    if api_error:
        return "error"
    truth = label == ATTACK_LABEL
    if truth and predicted_adversarial:
        return "TP"
    if truth and not predicted_adversarial:
        return "FN"
    if not truth and predicted_adversarial:
        return "FP"
    return "TN"


def run_chunk(
    records: list[dict],
    fie_url: str,
    api_key: str,
    delay: float,
    timeout: float,
    chunk_label: str = "",
    global_offset: int = 0,
) -> list[dict]:
    """
    Run evaluation on a slice of records.
    global_offset is added to the per-record idx so the final merged
    results preserve original dataset positions.
    """
    results: list[dict] = []
    total = len(records)
    label = chunk_label or f"{total} prompts"

    print(f"\nRunning evaluation on {label} ...")
    print(f"  FIE endpoint : {fie_url.rstrip('/')}/diagnose")
    print(f"  Call delay   : {delay}s")
    print()

    for local_idx, record in enumerate(records, start=1):
        global_idx = global_offset + local_idx
        prompt   = record["prompt"]
        label_   = record["label"]
        category = record["category"]

        print(
            f"\r  [{local_idx:>3}/{total}] #{global_idx:>3}  {category:<28} {label_:<8}",
            end="",
            flush=True,
        )

        response = call_diagnose(prompt, fie_url, api_key, timeout)
        verdict  = extract_verdict(response)

        results.append({
            "idx":              global_idx,
            "prompt":           prompt,
            "label":            label_,
            "category":         record.get("category"),
            "subcategory":      record.get("subcategory"),
            "source":           record.get("source"),
            "is_adversarial":   verdict["is_adversarial"],
            "jury_confidence":  verdict["jury_confidence"],
            "root_cause":       verdict["root_cause"],
            "detection_layers": verdict["detection_layers"],
            "api_error":        verdict["api_error"],
            "outcome":          _classify_outcome(label_, verdict["is_adversarial"], verdict["api_error"]),
        })

        if local_idx < total:
            time.sleep(delay)

    print()
    return results

# ---------------------------------------------------------------------------
# JSON report saving
# ---------------------------------------------------------------------------

def save_report(
    output_path: str,
    dataset_path: str,
    records: list[dict],
    results: list[dict],
    metrics: dict,
    per_category: dict[str, dict],
    per_layer: dict[str, dict],
    ablation: dict[str, dict] | None,
    fie_url: str,
    extra_meta: dict | None = None,
) -> None:
    report: dict[str, Any] = {
        "dataset":       dataset_path,
        "total_prompts": len(records),
        "attack_count":  sum(1 for r in records if r["label"] == ATTACK_LABEL),
        "benign_count":  sum(1 for r in records if r["label"] == BENIGN_LABEL),
        "metrics":       metrics,
        "per_category":  per_category,
        "per_layer":     per_layer,
        "results":       results,
        "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
        "fie_url":       fie_url,
    }
    if ablation is not None:
        report["ablation"] = ablation
    if extra_meta:
        report.update(extra_meta)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_path_obj.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"Full report saved to: {output_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Resolve output path ───────────────────────────────────────────────
    if args.output is None:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/eval_results_{ts}.json"
    else:
        output_path = args.output

    # ── Merge mode: load all saved seed files → final report ─────────────
    if args.merge:
        print(f"\nMerge mode: loading seed results from {args.seed_dir} ...")
        results = load_seed_results(args.seed_dir)
        print(f"  Total records loaded: {len(results)}")

        # Reconstruct records list from results (for report header counts)
        records = [{"label": r["label"], "category": r["category"]} for r in results]

        metrics      = compute_metrics(results)
        per_category = compute_per_category(results)
        per_layer    = compute_per_layer(results)

        print_report(
            dataset_path=args.dataset,
            records=records,
            results=results,
            metrics=metrics,
            per_category=per_category,
            per_layer=per_layer,
            seed_info=f"Merged from {args.seed_dir}",
        )

        ablation: dict | None = None
        if args.layers_ablation:
            print("Running per-layer ablation study...")
            ablation = run_ablation(results)
            print_ablation_table(baseline=metrics, ablation=ablation)

        save_report(
            output_path=output_path,
            dataset_path=args.dataset,
            records=records,
            results=results,
            metrics=metrics,
            per_category=per_category,
            per_layer=per_layer,
            ablation=ablation,
            fie_url=args.fie_url,
            extra_meta={"mode": "merged", "seed_dir": args.seed_dir},
        )
        return

    # ── Load dataset ─────────────────────────────────────────────────────
    records = load_dataset(args.dataset)
    if not records:
        print("[ERROR] No valid records loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Dry-run ───────────────────────────────────────────────────────────
    if args.dry_run:
        print_dataset_stats(records, args.dataset)
        return

    # ── Determine which seeds to run ─────────────────────────────────────
    total_records = len(records)
    chunk_size    = args.chunk_size
    n_seeds       = args.seeds

    if n_seeds <= 1:
        # Single pass — run everything
        seeds_to_run = [None]  # None = full dataset
    else:
        if args.seed is not None:
            if args.seed < 1 or args.seed > n_seeds:
                print(f"[ERROR] --seed must be between 1 and --seeds ({n_seeds})", file=sys.stderr)
                sys.exit(1)
            seeds_to_run = [args.seed]
        else:
            seeds_to_run = list(range(1, n_seeds + 1))

    print(f"\nDataset    : {args.dataset}  ({total_records} prompts)")
    print(f"Seeds      : {n_seeds}  |  chunk-size: {chunk_size}  |  endpoint: /diagnose")
    if n_seeds > 1:
        print(f"Seed dir   : {args.seed_dir}")
        print(f"Running    : seed(s) {seeds_to_run}")

    # ── Single-pass (no seeds) ────────────────────────────────────────────
    if seeds_to_run == [None]:
        results = run_chunk(
            records=records,
            fie_url=args.fie_url,
            api_key=args.api_key,
            delay=args.delay,
            timeout=args.timeout,
            chunk_label=f"all {total_records} prompts",
            global_offset=0,
        )

        metrics      = compute_metrics(results)
        per_category = compute_per_category(results)
        per_layer    = compute_per_layer(results)

        print_report(
            dataset_path=args.dataset,
            records=records,
            results=results,
            metrics=metrics,
            per_category=per_category,
            per_layer=per_layer,
        )

        ablation = None
        if args.layers_ablation:
            print("Running per-layer ablation study...")
            ablation = run_ablation(results)
            print_ablation_table(baseline=metrics, ablation=ablation)

        save_report(
            output_path=output_path,
            dataset_path=args.dataset,
            records=records,
            results=results,
            metrics=metrics,
            per_category=per_category,
            per_layer=per_layer,
            ablation=ablation,
            fie_url=args.fie_url,
            extra_meta={"mode": "single_pass"},
        )
        return

    # ── Seed-based chunked evaluation ────────────────────────────────────
    for seed_num in seeds_to_run:
        # Compute slice boundaries for this seed
        start = (seed_num - 1) * chunk_size
        end   = min(start + chunk_size, total_records)

        if start >= total_records:
            print(f"\n[INFO] Seed {seed_num}: start index {start} >= dataset size {total_records}. Skipping.")
            continue

        chunk_records = records[start:end]
        chunk_label   = f"seed {seed_num}/{n_seeds} (prompts {start + 1}–{end} of {total_records})"

        # Check if this seed was already completed
        seed_path = seed_result_path(args.seed_dir, seed_num)
        if seed_path.exists():
            print(f"\n[INFO] Seed {seed_num}: result file already exists at {seed_path}. Skipping.")
            print("       Delete the file to re-run this seed.")
            continue

        seed_results = run_chunk(
            records=chunk_records,
            fie_url=args.fie_url,
            api_key=args.api_key,
            delay=args.delay,
            timeout=args.timeout,
            chunk_label=chunk_label,
            global_offset=start,
        )

        # Print per-seed metrics for a quick sanity check
        seed_metrics = compute_metrics(seed_results)
        print(
            f"\n  Seed {seed_num} metrics  — "
            f"Recall: {seed_metrics['recall'] * 100:.1f}%  "
            f"FPR: {seed_metrics['fpr'] * 100:.1f}%  "
            f"F1: {seed_metrics['f1'] * 100:.1f}%  "
            f"Errors: {seed_metrics['error_count']}"
        )

        # Save intermediate results immediately
        save_seed_results(args.seed_dir, seed_num, seed_results)

    # ── Auto-merge if all seeds are done ─────────────────────────────────
    completed = [seed_result_path(args.seed_dir, s).exists() for s in range(1, n_seeds + 1)]
    if all(completed):
        print(f"\nAll {n_seeds} seeds complete. Merging results...")
        all_results = load_seed_results(args.seed_dir)

        metrics      = compute_metrics(all_results)
        per_category = compute_per_category(all_results)
        per_layer    = compute_per_layer(all_results)
        all_records  = [{"label": r["label"], "category": r["category"]} for r in all_results]

        print_report(
            dataset_path=args.dataset,
            records=all_records,
            results=all_results,
            metrics=metrics,
            per_category=per_category,
            per_layer=per_layer,
            seed_info=f"All {n_seeds} seeds merged ({len(all_results)} prompts)",
        )

        ablation = None
        if args.layers_ablation:
            print("Running per-layer ablation study...")
            ablation = run_ablation(all_results)
            print_ablation_table(baseline=metrics, ablation=ablation)

        save_report(
            output_path=output_path,
            dataset_path=args.dataset,
            records=all_records,
            results=all_results,
            metrics=metrics,
            per_category=per_category,
            per_layer=per_layer,
            ablation=ablation,
            fie_url=args.fie_url,
            extra_meta={"mode": "seeded", "n_seeds": n_seeds, "chunk_size": chunk_size, "seed_dir": args.seed_dir},
        )
    else:
        completed_nums = [s for s in range(1, n_seeds + 1) if seed_result_path(args.seed_dir, s).exists()]
        remaining_nums = [s for s in range(1, n_seeds + 1) if not seed_result_path(args.seed_dir, s).exists()]
        print(f"\nCompleted seeds: {completed_nums}")
        print(f"Remaining seeds: {remaining_nums}")
        print(f"Run the remaining seeds, then merge with:")
        print(f"  python data/eval_adversarial.py --seeds {n_seeds} --chunk-size {chunk_size} --merge")


if __name__ == "__main__":
    main()
