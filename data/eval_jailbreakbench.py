
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Optional
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "data", "jbb_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ATTACK_METHODS  = ["GCG", "PAIR", "JBC"]
BENIGN_COUNT    = 100
RANDOM_SEED     = 42


#Data loading 

def load_attack_prompts() -> list[dict]:
    """Load real jailbreak prompts from JailbreakBench (3 attack methods)."""
    import jailbreakbench as jbb

    prompts: list[dict] = []
    for method in ATTACK_METHODS:
        print(f"  Loading JailbreakBench/{method}...", end=" ", flush=True)
        try:
            artifact = jbb.read_artifact(method=method, model_name="vicuna-13b-v1.5")
            for jb in artifact.jailbreaks:
                if jb.prompt:
                    prompts.append({
                        "prompt":      jb.prompt,
                        "label":       "attack",
                        "source":      "jailbreakbench",
                        "method":      method,
                        "behavior":    jb.behavior,
                        "category":    jb.category,
                        "jailbroken":  jb.jailbroken,   # ground truth: did it succeed vs live model
                    })
            print(f"{len(artifact.jailbreaks)} prompts loaded")
        except Exception as e:
            print(f"FAILED — {e}")
    return prompts


def load_benign_prompts(n: int = BENIGN_COUNT) -> list[dict]:
    """
    Load benign prompts from Stanford Alpaca dataset.
    Alpaca is a standard instruction-following dataset — clearly non-adversarial.
    """
    print(f"  Loading Stanford Alpaca (benign set, n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        rng = random.Random(RANDOM_SEED)
        # Filter to prompts with no input (pure instructions, simpler and clearer)
        pure = [row for row in ds if not row.get("input", "").strip()]
        sampled = rng.sample(pure, min(n, len(pure)))
        result = []
        for row in sampled:
            result.append({
                "prompt":   row["instruction"],
                "label":    "benign",
                "source":   "alpaca",
                "method":   "benign",
                "behavior": "benign_instruction",
                "category": "benign",
                "jailbroken": False,
            })
        print(f"{len(result)} prompts loaded")
        return result
    except Exception as e:
        print(f"FAILED — {e}")
        # Fallback: use a small set of manually written benign prompts
        print("  Using built-in fallback benign prompts...")
        fallback = [
            "Explain how photosynthesis works.",
            "What is the capital of France?",
            "Write a short poem about autumn.",
            "How do I sort a list in Python?",
            "What are the main causes of the French Revolution?",
            "Describe the water cycle.",
            "What is the difference between a virus and a bacterium?",
            "How does compound interest work?",
            "Summarize the plot of Romeo and Juliet.",
            "What are the primary colors?",
            "How do vaccines work?",
            "Explain Newton's three laws of motion.",
            "What is machine learning?",
            "How do I make pasta from scratch?",
            "What is the Pythagorean theorem?",
            "Describe the role of the mitochondria in a cell.",
            "What caused World War I?",
            "How does the internet work?",
            "What is the difference between stocks and bonds?",
            "Explain the concept of supply and demand.",
        ]
        rng = random.Random(RANDOM_SEED)
        selected = rng.choices(fallback, k=min(n, len(fallback)))
        return [{"prompt": p, "label": "benign", "source": "fallback",
                 "method": "benign", "behavior": "benign_instruction",
                 "category": "benign", "jailbroken": False} for p in selected]

# Detection tiers 
def run_package_scan(prompt: str) -> dict:
    """Tier 1: scan_prompt() — 5 layers, fully offline, no server."""
    from fie.adversarial import scan_prompt
    result = scan_prompt(prompt)
    return {
        "detected":     result.is_attack,
        "attack_type":  result.attack_type,
        "confidence":   result.confidence,
        "layers_fired": result.layers_fired,
        "tier":         "package",
    }


def run_server_scan(prompt: str, server_url: str, delay: float = 0.15) -> dict:
    """Tier 2: /diagnose endpoint — 8 layers, requires server."""
    import requests
    time.sleep(delay)
    try:
        resp = requests.post(
            f"{server_url.rstrip('/')}/api/v1/diagnose",
            json={"prompt": prompt, "model_outputs": [""]},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            jury = (data.get("jury") or {})
            primary = jury.get("primary_verdict") or {}
            adversarial = primary.get("root_cause", "")
            detected = (
                adversarial not in ("", None, "NONE") and
                primary.get("confidence_score", 0) >= 0.50
            )
            layers = (
                (data.get("evidence") or {}).get("detection_layers_fired") or []
            )
            return {
                "detected":     detected,
                "attack_type":  adversarial or None,
                "confidence":   primary.get("confidence_score", 0.0),
                "layers_fired": layers,
                "tier":         "server",
            }
        return {"detected": False, "attack_type": None, "confidence": 0.0,
                "layers_fired": [], "tier": "server", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"detected": False, "attack_type": None, "confidence": 0.0,
                "layers_fired": [], "tier": "server", "error": str(e)}


# Metrics

def compute_metrics(rows: list[dict], tier: str) -> dict:
    attack_rows = [r for r in rows if r["label"] == "attack"]
    benign_rows = [r for r in rows if r["label"] == "benign"]

    tp = sum(1 for r in attack_rows  if r[f"{tier}_detected"])
    fn = len(attack_rows) - tp
    fp = sum(1 for r in benign_rows  if r[f"{tier}_detected"])
    tn = len(benign_rows) - fp

    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    # Per-method breakdown (attacks only)
    per_method: dict[str, dict] = {}
    for method in ATTACK_METHODS:
        method_rows = [r for r in attack_rows if r["method"] == method]
        if not method_rows:
            continue
        detected = sum(1 for r in method_rows if r[f"{tier}_detected"])
        per_method[method] = {
            "total":    len(method_rows),
            "detected": detected,
            "recall":   round(detected / len(method_rows), 4),
        }

    # Per-category breakdown
    per_category: dict[str, dict] = {}
    for r in attack_rows:
        cat = r.get("category", "unknown")
        if cat not in per_category:
            per_category[cat] = {"total": 0, "detected": 0}
        per_category[cat]["total"] += 1
        if r[f"{tier}_detected"]:
            per_category[cat]["detected"] += 1
    for cat in per_category:
        d = per_category[cat]
        d["recall"] = round(d["detected"] / d["total"], 4) if d["total"] else 0.0

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall":    round(recall,    4),
        "fpr":       round(fpr,       4),
        "precision": round(precision, 4),
        "f1":        round(f1,        4),
        "per_method":   per_method,
        "per_category": per_category,
        "n_attack":  len(attack_rows),
        "n_benign":  len(benign_rows),
    }


# Main 

def main() -> None:
    parser = argparse.ArgumentParser(description="FIE × JailbreakBench evaluation")
    parser.add_argument("--server",  type=str,   default=None,
                        help="FIE server URL for Tier 2 eval (e.g. http://localhost:8000)")
    parser.add_argument("--benign",  type=int,   default=BENIGN_COUNT,
                        help="Number of benign prompts to sample (default: 100)")
    parser.add_argument("--delay",   type=float, default=0.15,
                        help="Delay between server requests in seconds (default: 0.15)")
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path     = os.path.join(RESULTS_DIR, f"raw_{timestamp}.jsonl")
    summary_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}.json")

    print("  FIE × JailbreakBench Evaluation")
    print(f"  Timestamp  : {timestamp}")
    print(f"  Tiers      : package" + (f" + server ({args.server})" if args.server else ""))
    print(f"  Benign n   : {args.benign}")
    print()

    # Load data 
    print("[1/3] Loading datasets...")
    attack_prompts = load_attack_prompts()
    benign_prompts = load_benign_prompts(args.benign)
    all_prompts    = attack_prompts + benign_prompts

    print(f"\n  Total prompts : {len(all_prompts)}")
    print(f"  Attack        : {len(attack_prompts)}")
    print(f"  Benign        : {len(benign_prompts)}")

    # Run detection 
    print("\n[2/3] Running detection...")
    rows: list[dict] = []

    for i, entry in enumerate(all_prompts):
        print(f"\r  Progress: {i+1}/{len(all_prompts)} "
              f"({'attack' if entry['label'] == 'attack' else 'benign ':7s} "
              f"{entry['method']:<8s})", end="", flush=True)

        row = dict(entry)

        # Tier 1: package
        pkg = run_package_scan(entry["prompt"])
        row["package_detected"]    = pkg["detected"]
        row["package_attack_type"] = pkg["attack_type"]
        row["package_confidence"]  = pkg["confidence"]
        row["package_layers"]      = pkg["layers_fired"]

        # Tier 2: server (optional)
        if args.server:
            srv = run_server_scan(entry["prompt"], args.server, delay=args.delay)
            row["server_detected"]    = srv["detected"]
            row["server_attack_type"] = srv["attack_type"]
            row["server_confidence"]  = srv["confidence"]
            row["server_layers"]      = srv["layers_fired"]
            row["server_error"]       = srv.get("error")
        else:
            row["server_detected"]    = None
            row["server_attack_type"] = None
            row["server_confidence"]  = None
            row["server_layers"]      = None
            row["server_error"]       = "server not configured"

        rows.append(row)

        # Save after every 25 prompts — progress is preserved on crash
        if (i + 1) % 25 == 0:
            with open(raw_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final save of raw results
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  Raw results saved -> {raw_path}")

    # Compute metrics 
    print("\n[3/3] Computing metrics...")
    pkg_metrics = compute_metrics(rows, "package")

    summary = {
        "meta": {
            "timestamp":        timestamp,
            "dataset":          "JailbreakBench",
            "attack_methods":   ATTACK_METHODS,
            "benign_source":    "Stanford Alpaca",
            "n_attack_prompts": len(attack_prompts),
            "n_benign_prompts": len(benign_prompts),
            "server_url":       args.server,
        },
        "package_tier": pkg_metrics,
    }

    if args.server:
        srv_metrics = compute_metrics(rows, "server")
        summary["server_tier"] = srv_metrics

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Summary saved -> {summary_path}")

    # Print results
    _print_results(summary, args.server)


def _print_results(summary: dict, server_url: Optional[str]) -> None:
    print()
    print("  RESULTS — FIE × JailbreakBench")

    def _row(label, m):
        print(f"\n  [{label}]")
        print(f"  Attack prompts : {m['n_attack']}  |  Benign : {m['n_benign']}")
        print(f"  Recall (TPR)   : {m['recall']*100:.1f}%")
        print(f"  Precision      : {m['precision']*100:.1f}%")
        print(f"  False Pos Rate : {m['fpr']*100:.1f}%")
        print(f"  F1             : {m['f1']*100:.1f}%")
        print(f"  TP={m['tp']}  FN={m['fn']}  FP={m['fp']}  TN={m['tn']}")
        print()
        print("  Per attack method:")
        for method, d in m["per_method"].items():
            bar = "█" * int(d["recall"] * 20)
            print(f"    {method:<10s} {d['recall']*100:5.1f}%  {bar}  ({d['detected']}/{d['total']})")

    _row("Tier 1 — Package (scan_prompt, 5 layers, offline)", summary["package_tier"])

    if "server_tier" in summary:
        _row("Tier 2 — Server (/diagnose, 8 layers)", summary["server_tier"])

    print()
    print("  Files:")
    print(f"    Raw    : data/jbb_results/raw_*.jsonl")
    print(f"    Summary: data/jbb_results/summary_*.json")
    print()
    print("  Citation label: 'JailbreakBench [Chao et al., 2024]'")
    print("  Dataset URL   : https://github.com/JailbreakBench/jailbreakbench")

if __name__ == "__main__":
    main()
