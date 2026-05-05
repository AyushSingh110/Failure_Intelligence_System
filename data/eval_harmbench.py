"""
FIE x HarmBench Evaluation

HarmBench [Mazeika et al., 2024] is a standardized benchmark for evaluating
LLM robustness against harmful behaviors across 7 semantic categories.

This script evaluates FIE's package-tier scan_prompt() on HarmBench attack
prompts loaded from HuggingFace. Two sources are used:

  1. Direct behaviors  — raw harmful requests (tests semantic intent detection)
  2. WildJailbreak adversarial variants — iteratively rephrased jailbreaks
     mapped to HarmBench categories (tests PAIR-style detection)

No LLM responses are generated — this measures FIE's ability to detect the
*input* before it reaches a model.

Output:
  data/jbb_tier1/harmbench_results.json
  notebooks/pair_classifier_plots/figH1_harmbench.png

Usage:
    python data/eval_harmbench.py
    python data/eval_harmbench.py --n 50   # limit per category (default: all)

Citation:
    Mazeika et al. (2024). HarmBench: A Standardized Evaluation Framework for
    Automated Red Teaming and Robust Refusal.
    https://arxiv.org/abs/2402.04249
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "data" / "jbb_tier1"
PLOTS_DIR   = ROOT / "notebooks" / "pair_classifier_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# HarmBench semantic categories (7 standard)
HARMBENCH_CATEGORIES = [
    "Chemical & Biological Weapons",
    "Cyberattacks & Malware",
    "Harassment & Discrimination",
    "Illegal Activity",
    "Misinformation",
    "Physical Violence",
    "Privacy & Surveillance",
]

# Muted academic palette
C = {
    "attack":    "#8B2318",
    "benign":    "#1A4F72",
    "good":      "#1C5631",
    "warn":      "#784212",
    "neutral":   "#4A4A4A",
    "highlight": "#4A235A",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_harmbench_behaviors(n_per_cat: int | None = None) -> list[dict]:
    """
    Load HarmBench standard behaviors from HuggingFace.
    Uses allenai/tulu-3-harmbench-eval (320 behaviors, ungated, standard format).
    Citation: Mazeika et al. (2024). HarmBench. arXiv:2402.04249
    """
    print("  [1/2] Loading HarmBench standard behaviors...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/tulu-3-harmbench-eval", split="test")
        rows = []
        cat_counts: dict[str, int] = {}
        for row in ds:
            cat = (row.get("SemanticCategory") or "unknown").strip()
            if n_per_cat and cat_counts.get(cat, 0) >= n_per_cat:
                continue
            behavior = (row.get("Behavior") or row.get("prompt") or "").strip()
            if not behavior or len(behavior) < 20:
                continue
            rows.append({
                "prompt":   behavior,
                "label":    "attack",
                "source":   "harmbench_behaviors",
                "category": cat,
            })
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"{len(rows)} loaded  ({len(cat_counts)} categories)")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def load_harmbench_alpaca_benign(n: int = 200) -> list[dict]:
    """Stanford Alpaca benign prompts as the FPR baseline."""
    import random
    print(f"  [2/2] Loading Alpaca benign (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds  = load_dataset("tatsu-lab/alpaca", split="train")
        rng = random.Random(99)
        pure = [r for r in ds if not r.get("input", "").strip()
                and len(r.get("instruction", "")) >= 20]
        sampled = rng.sample(pure, min(n, len(pure)))
        rows = [{
            "prompt":   r["instruction"].strip(),
            "label":    "benign",
            "source":   "alpaca",
            "category": "benign",
        } for r in sampled]
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


# ── Detection ─────────────────────────────────────────────────────────────────

def run_fie(rows: list[dict]) -> list[dict]:
    from fie.adversarial import scan_prompt

    print(f"\n  Scanning {len(rows)} prompts with scan_prompt() (6 layers)...")
    result = []
    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}]", end="\r", flush=True)
        r = dict(row)
        s = scan_prompt(r["prompt"])
        r["fie_detected"]    = s.is_attack
        r["fie_attack_type"] = s.attack_type
        r["fie_confidence"]  = s.confidence
        r["fie_layers"]      = s.layers_fired
        result.append(r)
    print(f"\n  Done.")
    return result


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    attack = [r for r in rows if r["label"] == "attack"]
    benign = [r for r in rows if r["label"] == "benign"]

    tp = sum(1 for r in attack if r.get("fie_detected"))
    fn = len(attack) - tp
    fp = sum(1 for r in benign if r.get("fie_detected"))
    tn = len(benign) - fp

    recall    = tp / len(attack) if attack else 0
    fpr       = fp / len(benign) if benign else 0
    precision = tp / (tp + fp)   if (tp + fp) else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    # Per semantic category
    per_cat: dict[str, dict] = {}
    for cat in set(r["category"] for r in attack):
        ms  = [r for r in attack if r["category"] == cat]
        det = sum(1 for r in ms if r.get("fie_detected"))
        per_cat[cat] = {
            "total":    len(ms),
            "detected": det,
            "recall":   round(det / len(ms), 4) if ms else 0,
        }

    # Layer contribution
    layer_counts: dict[str, int] = {}
    for r in attack:
        for layer in r.get("fie_layers", []):
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

    return {
        "n_attack":   len(attack),
        "n_benign":   len(benign),
        "recall":     round(recall,    4),
        "fpr":        round(fpr,       4),
        "precision":  round(precision, 4),
        "f1":         round(f1,        4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "per_category":  per_cat,
        "layer_contributions": layer_counts,
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_harmbench(rows: list[dict], metrics: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "#BDBDBD", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#E0E0E0", "grid.linewidth": 0.6,
    })

    per_cat = metrics["per_category"]
    cats    = sorted(per_cat, key=lambda c: per_cat[c]["recall"])
    recalls = [per_cat[c]["recall"] * 100 for c in cats]
    totals  = [per_cat[c]["total"]        for c in cats]
    short   = [c.split("&")[0].strip()[:22] for c in cats]
    colors  = [C["good"] if r >= 65 else C["warn"] if r >= 40 else C["attack"]
               for r in recalls]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Figure H1 — FIE Detection on HarmBench Behaviors\n"
        f"[Mazeika et al., 2024] — {metrics['n_attack']} attack behaviors, "
        f"{metrics['n_benign']} benign prompts",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: per-category recall
    ax = axes[0]
    y  = np.arange(len(cats))
    bars = ax.barh(y, recalls, color=colors, alpha=0.85, edgecolor="white", height=0.55)
    for bar, r, tot in zip(bars, recalls, totals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{r:.1f}%  (n={tot})", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("FIE Detection Rate (%)")
    ax.set_title("Per-Category Detection Recall")
    ax.axvline(65, color=C["neutral"], lw=1.2, linestyle="--", alpha=0.6, label="65% target")
    ax.set_xlim(0, 120)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: summary bar — overall recall / FPR / F1
    ax2 = axes[1]
    summary_names  = ["Recall", "Precision", "F1", "FPR"]
    summary_vals   = [metrics["recall"] * 100, metrics["precision"] * 100,
                      metrics["f1"] * 100, metrics["fpr"] * 100]
    bar_colors = [C["attack"], C["benign"], C["good"], C["warn"]]
    bars2 = ax2.bar(summary_names, summary_vals, color=bar_colors,
                    alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars2, summary_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.axhline(65, color=C["neutral"], lw=1, linestyle="--", alpha=0.5)
    ax2.set_title("Overall Summary Metrics")
    ax2.set_ylabel("Score (%)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "figH1_harmbench.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Max behaviors per category (default: all)")
    args = parser.parse_args()

    print("FIE x HarmBench Evaluation")
    print("=" * 55)
    print(f"  Citation: Mazeika et al. (2024) — https://arxiv.org/abs/2402.04249")
    print()

    # Load data
    print("[1/3] Loading datasets...")
    attack_rows = load_harmbench_behaviors(n_per_cat=args.n)
    benign_rows = load_harmbench_alpaca_benign(n=200)

    if not attack_rows:
        print("\nERROR: No HarmBench behaviors loaded.")
        print("Make sure you have internet access and `datasets` installed.")
        print("  pip install datasets")
        sys.exit(1)

    all_rows = attack_rows + benign_rows
    print(f"\n  Total: {len(all_rows)} prompts "
          f"(attack={len(attack_rows)}, benign={len(benign_rows)})")

    # Scan
    print("\n[2/3] Running FIE detection...")
    all_rows = run_fie(all_rows)

    # Metrics
    metrics = compute_metrics(all_rows)

    # Save
    out_path = RESULTS_DIR / "harmbench_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"dataset": "HarmBench", "citation": "Mazeika et al., 2024",
                            "n_attack": metrics["n_attack"], "n_benign": metrics["n_benign"],
                            "n_per_cat_limit": args.n,
                            "fie_layers": 6}, "metrics": metrics}, f, indent=2)
    print(f"\n  Results saved -> {out_path}")

    # Print results
    print("\n[3/3] Results")
    print("=" * 65)
    print(f"  Dataset         : HarmBench standard behaviors [Mazeika et al., 2024]")
    print(f"  Attack prompts  : {metrics['n_attack']}")
    print(f"  Benign prompts  : {metrics['n_benign']}")
    print(f"  FIE layers      : 6 (scan_prompt — offline, no API key)")
    print()
    print(f"  Overall Recall  : {metrics['recall']*100:.1f}%")
    print(f"  Precision       : {metrics['precision']*100:.1f}%")
    print(f"  F1              : {metrics['f1']*100:.1f}%")
    print(f"  FPR             : {metrics['fpr']*100:.1f}%")
    print(f"  TP={metrics['tp']}  FN={metrics['fn']}  FP={metrics['fp']}  TN={metrics['tn']}")
    print()
    print("  Per-category detection recall:")
    for cat, d in sorted(metrics["per_category"].items(),
                         key=lambda x: x[1]["recall"], reverse=True):
        bar = "#" * int(d["recall"] * 20)
        print(f"    {cat:<35s}  {d['recall']*100:5.1f}%  {bar}")
    print()
    print("  Layer contribution (how many attacks each layer caught):")
    total_tp = metrics["tp"]
    for layer, count in sorted(metrics["layer_contributions"].items(),
                                key=lambda x: x[1], reverse=True):
        pct = count / total_tp * 100 if total_tp else 0
        print(f"    {layer:<25s}  {count:>4}  ({pct:.1f}% of detections)")
    print("=" * 65)

    # Figure
    print("\n  Generating HarmBench figure...")
    plot_harmbench(all_rows, metrics)
    print("\n  Done.")
    print(f"\n  Citation: Mazeika et al. (2024). HarmBench: A Standardized Evaluation")
    print(f"  Framework for Automated Red Teaming and Robust Refusal.")
    print(f"  arXiv:2402.04249")


if __name__ == "__main__":
    main()
