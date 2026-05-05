"""
Collect semantic jailbreak training data for the PAIR-detection package-tier classifier.

Sources (all public, HuggingFace):
  1. JailbreakBench PAIR prompts        — 82  real PAIR attacks (already have)
  2. BeaverTails                        — harmful Q subset (conversational, no structural anomalies)
  3. Anthropic HH-RLHF red-team subset  — human red-team attempts (many PAIR-like)
  4. WildJailbreak (adversarial subset)  — adversarial rephrasing attacks

Benign sources:
  1. Stanford Alpaca                    — clean instruction-following
  2. Anthropic HH-RLHF helpful subset   — legitimate user questions

Output:
  data/pair_training/train.jsonl  — 80% split
  data/pair_training/val.jsonl    — 10% split
  data/pair_training/test.jsonl   — 10% split (held out — used for final eval only)
  data/pair_training/stats.json   — dataset statistics

Each row:
  {"prompt": "...", "label": 1, "source": "...", "category": "..."}
  label: 1 = adversarial/jailbreak, 0 = benign
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR     = os.path.join(ROOT, "data", "pair_training")
RANDOM_SEED = 42

# Targets per source — adjust if a dataset is unavailable
TARGET_BEAVERTAILS   = 800
TARGET_RLHF_REDTEAM  = 600
TARGET_WILDJAILBREAK = 500
TARGET_JBB_PAIR      = 82    # use all

TARGET_ALPACA_BENIGN = 800
TARGET_RLHF_HELPFUL  = 700

MIN_PROMPT_LEN = 30   # characters — filter out one-liners
MAX_PROMPT_LEN = 2000 # characters — avoid very long documents


def _keep(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    return MIN_PROMPT_LEN <= len(t) <= MAX_PROMPT_LEN


def collect_jbb_pair() -> list[dict]:
    """Load PAIR prompts from JailbreakBench (already evaluated in Tier 1)."""
    print("  [1/6] JailbreakBench PAIR prompts...", end=" ", flush=True)
    try:
        import jailbreakbench as jbb
        artifact = jbb.read_artifact(method="PAIR", model_name="vicuna-13b-v1.5")
        rows = []
        for jb in artifact.jailbreaks:
            if jb.prompt and _keep(jb.prompt):
                rows.append({
                    "prompt":   jb.prompt.strip(),
                    "label":    1,
                    "source":   "jailbreakbench_pair",
                    "category": "semantic_jailbreak",
                })
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def collect_beavertails(n: int = TARGET_BEAVERTAILS) -> list[dict]:
    """
    BeaverTails: 330K Q&A pairs with is_safe labels.
    We take the unsafe questions — these are harm-seeking queries phrased as
    natural instructions (no adversarial suffixes), similar to PAIR attacks.
    """
    print(f"  [2/6] BeaverTails unsafe questions (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        rng = random.Random(RANDOM_SEED)

        unsafe = [
            row for row in ds
            if not row.get("is_safe", True)
            and _keep(row.get("prompt", ""))
        ]
        sampled = rng.sample(unsafe, min(n, len(unsafe)))
        rows = [{
            "prompt":   row["prompt"].strip(),
            "label":    1,
            "source":   "beavertails",
            "category": row.get("category", "harmful_query"),
        } for row in sampled]
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def collect_rlhf_redteam(n: int = TARGET_RLHF_REDTEAM) -> list[dict]:
    """
    Anthropic HH-RLHF red-team attempts.
    The red_team_attempts split contains human-written adversarial prompts
    used to elicit harmful behavior — very similar to PAIR-style attacks.
    """
    print(f"  [3/6] Anthropic HH-RLHF red-team (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
        rng = random.Random(RANDOM_SEED + 1)

        prompts = []
        for row in ds:
            # The transcript field starts with "\n\nHuman: " — extract the first human turn
            transcript = row.get("transcript", "")
            if "\n\nHuman:" in transcript:
                first_turn = transcript.split("\n\nHuman:")[1].split("\n\nAssistant:")[0].strip()
                if _keep(first_turn):
                    prompts.append(first_turn)

        sampled = rng.sample(prompts, min(n, len(prompts)))
        rows = [{
            "prompt":   p,
            "label":    1,
            "source":   "anthropic_rlhf_redteam",
            "category": "red_team_attempt",
        } for p in sampled]
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def collect_wildjailbreak(n: int = TARGET_WILDJAILBREAK) -> list[dict]:
    """
    WildJailbreak (Ding et al., 2024): 262K adversarial prompts.
    We use the adversarial (non-vanilla) subset — these are rephrased attacks.
    """
    print(f"  [4/6] WildJailbreak adversarial subset (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/wildjailbreak", "train", split="train")
        rng = random.Random(RANDOM_SEED + 2)

        # adversarial column distinguishes rephrased from vanilla harmful
        adversarial = [
            row for row in ds
            if row.get("adversarial") and _keep(row.get("adversarial", ""))
        ]
        sampled = rng.sample(adversarial, min(n, len(adversarial)))
        rows = [{
            "prompt":   row["adversarial"].strip(),
            "label":    1,
            "source":   "wildjailbreak",
            "category": "adversarial_rephrasing",
        } for row in sampled]
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def collect_alpaca_benign(n: int = TARGET_ALPACA_BENIGN) -> list[dict]:
    """Stanford Alpaca — clean instruction-following, clearly non-adversarial."""
    print(f"  [5/6] Stanford Alpaca benign (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        rng = random.Random(RANDOM_SEED + 3)

        pure = [row for row in ds if not row.get("input", "").strip() and _keep(row.get("instruction", ""))]
        sampled = rng.sample(pure, min(n, len(pure)))
        rows = [{
            "prompt":   row["instruction"].strip(),
            "label":    0,
            "source":   "alpaca",
            "category": "benign_instruction",
        } for row in sampled]
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def collect_rlhf_helpful(n: int = TARGET_RLHF_HELPFUL) -> list[dict]:
    """
    Anthropic HH-RLHF helpful subset — legitimate user questions.
    Extract the first human turn from chosen (helpful) responses.
    """
    print(f"  [6/6] Anthropic HH-RLHF helpful benign (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
        rng = random.Random(RANDOM_SEED + 4)

        prompts = []
        for row in ds:
            chosen = row.get("chosen", "")
            if "\n\nHuman:" in chosen:
                first_turn = chosen.split("\n\nHuman:")[1].split("\n\nAssistant:")[0].strip()
                if _keep(first_turn):
                    prompts.append(first_turn)

        sampled = rng.sample(prompts, min(n, len(prompts)))
        rows = [{
            "prompt":   p,
            "label":    0,
            "source":   "anthropic_rlhf_helpful",
            "category": "benign_question",
        } for p in sampled]
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def deduplicate(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        key = r["prompt"][:200].lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def split_dataset(rows: list[dict], seed: int = RANDOM_SEED) -> tuple[list, list, list]:
    """80 / 10 / 10 stratified split."""
    rng = random.Random(seed)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]

    def _split(items):
        rng.shuffle(items)
        n     = len(items)
        n_val = max(1, int(n * 0.10))
        n_test = max(1, int(n * 0.10))
        return items[n_val + n_test:], items[:n_val], items[n_val:n_val + n_test]

    tr_p, va_p, te_p = _split(pos)
    tr_n, va_n, te_n = _split(neg)

    train = tr_p + tr_n
    val   = va_p + va_n
    test  = te_p + te_n

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def save_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print("PAIR Training Data Collection")
    print(f"  Output dir : {OUT_DIR}")
    print(f"  Timestamp  : {timestamp}\n")

    print("[1/3] Collecting adversarial examples...")
    attack_rows: list[dict] = []
    attack_rows += collect_jbb_pair()
    attack_rows += collect_beavertails()
    attack_rows += collect_rlhf_redteam()
    attack_rows += collect_wildjailbreak()

    print("\n[2/3] Collecting benign examples...")
    benign_rows: list[dict] = []
    benign_rows += collect_alpaca_benign()
    benign_rows += collect_rlhf_helpful()

    print("\n[3/3] Processing and saving...")
    all_rows = deduplicate(attack_rows + benign_rows)

    n_attack = sum(1 for r in all_rows if r["label"] == 1)
    n_benign = sum(1 for r in all_rows if r["label"] == 0)
    print(f"  After dedup: {len(all_rows)} total | attack={n_attack} | benign={n_benign}")

    train, val, test = split_dataset(all_rows)

    train_path = os.path.join(OUT_DIR, "train.jsonl")
    val_path   = os.path.join(OUT_DIR, "val.jsonl")
    test_path  = os.path.join(OUT_DIR, "test.jsonl")

    save_jsonl(train, train_path)
    save_jsonl(val,   val_path)
    save_jsonl(test,  test_path)

    stats = {
        "timestamp":    timestamp,
        "total":        len(all_rows),
        "n_attack":     n_attack,
        "n_benign":     n_benign,
        "class_balance": round(n_attack / len(all_rows), 3) if all_rows else 0,
        "splits": {
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        },
        "sources": {},
    }
    for r in all_rows:
        src = r["source"]
        if src not in stats["sources"]:
            stats["sources"][src] = {"total": 0, "attack": 0, "benign": 0}
        stats["sources"][src]["total"] += 1
        if r["label"] == 1:
            stats["sources"][src]["attack"] += 1
        else:
            stats["sources"][src]["benign"] += 1

    stats_path = os.path.join(OUT_DIR, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n  train.jsonl : {len(train)} examples")
    print(f"  val.jsonl   : {len(val)} examples")
    print(f"  test.jsonl  : {len(test)} examples  (HELD OUT — do not peek until final eval)")
    print(f"  stats.json  : {stats_path}")
    print(f"\n  Class balance (attack ratio): {stats['class_balance']*100:.1f}%")
    print("\n  Source breakdown:")
    for src, d in stats["sources"].items():
        print(f"    {src:<35s} total={d['total']:4d}  attack={d['attack']:4d}  benign={d['benign']:4d}")
    print(f"\n  Done. Run data/train_pair_classifier.py next.")


if __name__ == "__main__":
    main()
