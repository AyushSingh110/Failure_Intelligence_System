"""
Layer ablation study for FIE package-tier scan_prompt().

For each ablation condition (one layer removed at a time, plus each layer alone),
re-runs detection on the JailbreakBench dataset and computes:
  - Overall recall, FPR, F1
  - Per-method recall (GCG / PAIR / JBC)

All processing is local — no API calls.

Output:
  data/jbb_tier1/ablation_results.json
  notebooks/pair_classifier_plots/fig_ablation_study.png

Usage:
    python data/eval_ablation_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fie.adversarial import (
    _run_pattern_detection,
    _run_guard_detection,
    _run_indirect_injection_detection,
    _run_gcg_detection,
    _run_perplexity_proxy,
    _run_pair_classifier,
)

RESULTS_DIR = ROOT / "data" / "jbb_tier1"
PLOTS_DIR   = ROOT / "notebooks" / "pair_classifier_plots"
PLOTS_DIR.mkdir(exist_ok=True)

ATTACK_METHODS = ["GCG", "PAIR", "JBC"]

# Muted academic palette
C = {
    "attack":    "#8B2318",
    "benign":    "#1A4F72",
    "good":      "#1C5631",
    "warn":      "#784212",
    "neutral":   "#4A4A4A",
    "highlight": "#4A235A",
    "bg":        "#F5F5F5",
}

ALL_LAYERS = ["regex", "prompt_guard", "indirect", "gcg", "perplexity", "pair"]

LAYER_DISPLAY = {
    "regex":        "L1: Regex Patterns",
    "prompt_guard": "L2: Prompt Guard",
    "indirect":     "L4: Indirect Inject",
    "gcg":          "L5: GCG Suffix",
    "perplexity":   "L6: Perplexity Proxy",
    "pair":         "L7: PAIR Classifier",
}


# ── Core: run detection with arbitrary subset of layers ───────────────────────

def scan_with_layers(prompt: str, active: set[str]) -> tuple[bool, list[str]]:
    """Run only the specified layers; return (is_attack, layers_fired)."""
    best_conf     = 0.0
    best_root     = None
    layers_fired  = []

    if "regex" in active:
        hit, _ = _run_pattern_detection(prompt)
        if hit is not None:
            layers_fired.append("regex")
            if hit.base_confidence > best_conf:
                best_conf = hit.base_confidence
                best_root = hit.root_cause

    if "prompt_guard" in active:
        root, conf, _ = _run_guard_detection(prompt)
        if root is not None:
            layers_fired.append("prompt_guard")
            if conf > best_conf:
                best_conf = conf
                best_root = root

    if "indirect" in active:
        root, conf, _ = _run_indirect_injection_detection(prompt)
        if root is not None:
            layers_fired.append("indirect")
            if conf > best_conf:
                best_conf = conf
                best_root = root

    if "gcg" in active:
        root, conf, _ = _run_gcg_detection(prompt)
        if root is not None:
            layers_fired.append("gcg")
            if conf > best_conf:
                best_conf = conf
                best_root = root

    if "perplexity" in active:
        root, conf, _ = _run_perplexity_proxy(prompt)
        if root is not None:
            layers_fired.append("perplexity")
            if conf > best_conf:
                best_conf = conf
                best_root = root

    if "pair" in active:
        root, conf, _ = _run_pair_classifier(prompt)
        if root is not None:
            layers_fired.append("pair")
            if conf > best_conf:
                best_conf = conf
                best_root = root

    is_attack = best_conf >= 0.50 and best_root is not None
    return is_attack, layers_fired


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute(rows: list[dict], detections: list[bool]) -> dict:
    attack_idx = [i for i, r in enumerate(rows) if r["label"] == "attack"]
    benign_idx = [i for i, r in enumerate(rows) if r["label"] == "benign"]

    tp = sum(1 for i in attack_idx if detections[i])
    fn = len(attack_idx) - tp
    fp = sum(1 for i in benign_idx if detections[i])
    tn = len(benign_idx) - fp

    recall    = tp / len(attack_idx) if attack_idx else 0
    fpr       = fp / len(benign_idx) if benign_idx else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    per_method = {}
    for m in ATTACK_METHODS:
        idxs = [i for i, r in enumerate(rows) if r["label"] == "attack" and r["method"] == m]
        det  = sum(1 for i in idxs if detections[i])
        per_method[m] = round(det / len(idxs), 4) if idxs else 0

    return {
        "recall": round(recall, 4),
        "fpr":    round(fpr,    4),
        "f1":     round(f1,     4),
        "precision": round(precision, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "per_method": per_method,
    }


# ── Ablation conditions ───────────────────────────────────────────────────────

def build_conditions() -> dict[str, set[str]]:
    full = set(ALL_LAYERS)
    conds = {"Full System": full}

    # Remove one layer at a time
    for layer in ALL_LAYERS:
        conds[f"Remove {LAYER_DISPLAY[layer]}"] = full - {layer}

    # Each layer alone
    for layer in ALL_LAYERS:
        conds[f"Only {LAYER_DISPLAY[layer]}"] = {layer}

    return conds


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ablation(results: dict) -> None:
    # Section 1: Remove-one-layer (recall delta from full)
    full_recall = results["Full System"]["recall"]
    full_pair   = results["Full System"]["per_method"]["PAIR"]
    full_fpr    = results["Full System"]["fpr"]

    remove_keys = [k for k in results if k.startswith("Remove")]
    only_keys   = [k for k in results if k.startswith("Only")]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        "Figure A1 — Layer Ablation Study (FIE Package Tier)\n"
        "JailbreakBench Dataset (282 attacks + 100 benign)",
        fontsize=13, fontweight="bold"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "#BDBDBD", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
    })

    # Panel 1: Recall drop when each layer is removed
    ax = axes[0]
    short_names = [k.replace("Remove ", "").replace("L1: ", "").replace("L2: ", "")
                    .replace("L4: ", "").replace("L5: ", "").replace("L6: ", "")
                    .replace("L7: ", "") for k in remove_keys]
    drops = [full_recall - results[k]["recall"] for k in remove_keys]
    colors = [C["attack"] if d > 0.05 else C["warn"] if d > 0 else C["good"] for d in drops]
    y = np.arange(len(remove_keys))
    bars = ax.barh(y, [d * 100 for d in drops], color=colors, alpha=0.85,
                   edgecolor="white", height=0.55)
    for bar, d in zip(bars, drops):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"-{d*100:.1f}pp", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Recall Drop (percentage points)")
    ax.set_title("Impact on Overall Recall\n(removing one layer)")
    ax.axvline(0, color=C["neutral"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Per-method recall when each layer is removed (PAIR is key)
    ax2 = axes[1]
    method_colors = {"GCG": C["neutral"], "PAIR": C["attack"], "JBC": C["benign"]}
    x = np.arange(len(remove_keys))
    width = 0.28

    for j, (method, col) in enumerate(method_colors.items()):
        vals = [results[k]["per_method"].get(method, 0) * 100 for k in remove_keys]
        offset = (j - 1) * width
        ax2.barh(x + offset, vals, width, color=col, alpha=0.80, edgecolor="white",
                 linewidth=0.6, label=method)

    ax2.set_yticks(x)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.set_xlabel("Recall (%)")
    ax2.set_title("Per-Method Recall\n(removing one layer)")
    ax2.axvline(full_recall * 100, color=C["neutral"], lw=1, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3: Each layer alone — recall
    ax3 = axes[2]
    solo_names = [k.replace("Only ", "").replace("L1: ", "").replace("L2: ", "")
                   .replace("L4: ", "").replace("L5: ", "").replace("L6: ", "")
                   .replace("L7: ", "") for k in only_keys]
    solo_recall = [results[k]["recall"] * 100 for k in only_keys]
    solo_pair   = [results[k]["per_method"].get("PAIR", 0) * 100 for k in only_keys]
    solo_fpr    = [results[k]["fpr"] * 100 for k in only_keys]
    y3 = np.arange(len(only_keys))

    ax3.barh(y3 - 0.22, solo_recall, 0.3, color=C["attack"],  alpha=0.82,
             edgecolor="white", label="Overall Recall")
    ax3.barh(y3 + 0.0,  solo_pair,   0.3, color=C["highlight"], alpha=0.82,
             edgecolor="white", label="PAIR Recall")
    ax3.barh(y3 + 0.22, solo_fpr,    0.3, color=C["warn"],    alpha=0.82,
             edgecolor="white", label="FPR")

    ax3.set_yticks(y3)
    ax3.set_yticklabels(solo_names, fontsize=9)
    ax3.set_xlabel("Score (%)")
    ax3.set_title("Each Layer Alone\n(recall + FPR)")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "figA1_ablation_study.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load data
    path = sorted(RESULTS_DIR.glob("baseline_comparison.json"))
    if not path:
        print("ERROR: run eval_baseline_comparison.py first")
        sys.exit(1)

    with open(path[-1], encoding="utf-8") as f:
        rows = json.load(f)

    attack = [r for r in rows if r["label"] == "attack"]
    benign = [r for r in rows if r["label"] == "benign"]
    print(f"Ablation Study — FIE Package Tier")
    print(f"  Prompts : {len(rows)} (attack={len(attack)}, benign={len(benign)})")

    # Warm up PAIR classifier (lazy load)
    print("  Warming up PAIR classifier...", end=" ", flush=True)
    scan_with_layers("warmup", {"pair"})
    print("ready")

    conditions = build_conditions()
    results    = {}

    print(f"\n  Running {len(conditions)} ablation conditions...\n")
    for name, active in conditions.items():
        detections = []
        for row in rows:
            det, _ = scan_with_layers(row["prompt"], active)
            detections.append(det)
        m = compute(rows, detections)
        results[name] = m
        pair_r = m["per_method"].get("PAIR", 0)
        gcg_r  = m["per_method"].get("GCG", 0)
        jbc_r  = m["per_method"].get("JBC", 0)
        print(f"  {name:<40s}  recall={m['recall']*100:5.1f}%  "
              f"PAIR={pair_r*100:5.1f}%  GCG={gcg_r*100:5.1f}%  "
              f"JBC={jbc_r*100:5.1f}%  FPR={m['fpr']*100:4.1f}%  F1={m['f1']*100:5.1f}%")

    # Save
    out = RESULTS_DIR / "ablation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out}")

    # Summary table
    print("\n" + "=" * 85)
    print("  ABLATION SUMMARY — Remove one layer at a time")
    print("=" * 85)
    full = results["Full System"]
    print(f"  {'Condition':<42} {'Recall':>8} {'PAIR':>8} {'FPR':>7} {'F1':>8}")
    print("  " + "-" * 75)
    for name, m in results.items():
        if name == "Full System" or name.startswith("Remove"):
            marker = " *" if name == "Full System" else "  "
            print(f"  {name:<42}{marker}"
                  f"{m['recall']*100:>7.1f}%"
                  f"{m['per_method'].get('PAIR',0)*100:>8.1f}%"
                  f"{m['fpr']*100:>7.1f}%"
                  f"{m['f1']*100:>8.1f}%")
    print("=" * 85)

    print("\n  SOLO LAYER PERFORMANCE")
    print("=" * 85)
    print(f"  {'Layer':<42} {'Recall':>8} {'PAIR':>8} {'FPR':>7} {'F1':>8}")
    print("  " + "-" * 75)
    for name, m in results.items():
        if name.startswith("Only"):
            print(f"  {name:<42}  "
                  f"{m['recall']*100:>7.1f}%"
                  f"{m['per_method'].get('PAIR',0)*100:>8.1f}%"
                  f"{m['fpr']*100:>7.1f}%"
                  f"{m['f1']*100:>8.1f}%")
    print("=" * 85)

    # Plot
    print("\n  Generating ablation figure...")
    plot_ablation(results)
    print("\n  Done.")


if __name__ == "__main__":
    main()
