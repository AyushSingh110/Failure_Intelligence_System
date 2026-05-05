"""
Collect semantic jailbreak training data for the PAIR-detection package-tier classifier.

Attack sources:
  1. JailbreakBench (all methods)   — 282 real attacks from our own Tier 1 eval results
  2. Anthropic HH-RLHF red-team    — 600 human red-team attempts (PAIR-like natural language)
  3. lmsys/toxic-chat               — real user toxic interactions (ungated)
  4. declare-lab/HarmfulQA          — 1960 harmful questions across 10 harm categories

Benign sources:
  1. Stanford Alpaca                — clean instruction-following
  2. Anthropic HH-RLHF helpful     — legitimate user questions

Output:
  data/pair_training/train.jsonl  — 80% split
  data/pair_training/val.jsonl    — 10% split
  data/pair_training/test.jsonl   — 10% split (held out — do not use until final eval)
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

# Load .env and authenticate with HuggingFace for gated datasets
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

_HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
if _HF_TOKEN:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_HF_TOKEN, add_to_git_credential=False)
        print("  HuggingFace: authenticated via .env token")
    except Exception as _e:
        print(f"  HuggingFace: auth warning — {_e}")
else:
    print("  HuggingFace: no token found — gated datasets may fail")

OUT_DIR     = os.path.join(ROOT, "data", "pair_training")
RANDOM_SEED = 42

TARGET_RLHF_REDTEAM  = 600
TARGET_TOXICCHAT     = 400
TARGET_HARMFULQA     = 600
TARGET_WILDJAILBREAK = 500

TARGET_ALPACA_BENIGN = 800
TARGET_RLHF_HELPFUL  = 700

MIN_PROMPT_LEN = 30
MAX_PROMPT_LEN = 2000


def _keep(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    return MIN_PROMPT_LEN <= len(t) <= MAX_PROMPT_LEN


# ── Attack sources ─────────────────────────────────────────────────────────────

def collect_jbb_all() -> list[dict]:
    """
    Load ALL JailbreakBench attack prompts from our own saved Tier 1 eval JSONL.
    This bypasses the litellm dependency issue with the jailbreakbench package.
    GCG prompts have adversarial suffixes but are still valuable training signal.
    """
    print("  [1/6] JailbreakBench (from local Tier 1 results)...", end=" ", flush=True)
    jbb_dir = os.path.join(ROOT, "data", "jbb_tier1")
    jsonl_files = sorted([
        os.path.join(jbb_dir, f)
        for f in os.listdir(jbb_dir)
        if f.startswith("raw_") and f.endswith(".jsonl")
    ])
    if not jsonl_files:
        print("FAILED — no raw_*.jsonl found in data/jbb_tier1/")
        return []

    rows = []
    seen: set[str] = set()
    for path in jsonl_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("label") != "attack":
                    continue
                prompt = (r.get("prompt") or "").strip()
                if not _keep(prompt):
                    continue
                key = prompt[:150].lower()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "prompt":   prompt,
                    "label":    1,
                    "source":   "jailbreakbench",
                    "category": f"jbb_{r.get('method', 'unknown').lower()}",
                })
    print(f"{len(rows)} loaded")
    return rows


def collect_rlhf_redteam(n: int = TARGET_RLHF_REDTEAM) -> list[dict]:
    """Anthropic HH-RLHF red-team attempts — human-written adversarial prompts."""
    print(f"  [2/6] Anthropic HH-RLHF red-team (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
        rng = random.Random(RANDOM_SEED + 1)

        prompts = []
        for row in ds:
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


def collect_toxicchat(n: int = TARGET_TOXICCHAT) -> list[dict]:
    """
    lmsys/toxic-chat: 10K real ChatGPT user interactions labeled by human annotators.
    We take the toxic subset (jailbreaking=1 or toxicity=1) — real users trying
    to elicit harmful content, phrased as natural conversation.
    """
    print(f"  [3/6] lmsys/toxic-chat toxic subset (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
        rng = random.Random(RANDOM_SEED + 2)

        toxic = [
            row for row in ds
            if (row.get("jailbreaking") == 1 or row.get("toxicity") == 1)
            and _keep(row.get("user_input", ""))
        ]
        sampled = rng.sample(toxic, min(n, len(toxic)))
        rows = [{
            "prompt":   row["user_input"].strip(),
            "label":    1,
            "source":   "toxicchat",
            "category": "jailbreak" if row.get("jailbreaking") == 1 else "toxic_query",
        } for row in sampled]
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def collect_harmfulqa(n: int = TARGET_HARMFULQA) -> list[dict]:
    """
    declare-lab/HarmfulQA: 1960 harmful questions across 10 harm categories.
    Phrased as natural questions — very similar to PAIR attack style.
    """
    print(f"  [4/6] declare-lab/HarmfulQA (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("declare-lab/HarmfulQA", split="train")
        rng = random.Random(RANDOM_SEED + 3)

        prompts = []
        for row in ds:
            # question field contains the harmful question
            q = (row.get("question") or row.get("prompt") or "").strip()
            if _keep(q):
                prompts.append((q, row.get("category", "harmful")))

        sampled = rng.sample(prompts, min(n, len(prompts)))
        rows = [{
            "prompt":   p,
            "label":    1,
            "source":   "harmfulqa",
            "category": cat,
        } for p, cat in sampled]
        print(f"{len(rows)} loaded")
        return rows
    except Exception as e:
        print(f"FAILED — {e}")
        return []


def collect_wildjailbreak(n: int = TARGET_WILDJAILBREAK) -> list[dict]:
    """
    allenai/WildJailbreak: 262K adversarial prompts (gated — needs HF token).
    Uses the adversarial column: prompts iteratively rewritten to evade refusals.
    """
    print(f"  [4b/6] WildJailbreak adversarial subset (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/wildjailbreak", "train", split="train",
                          token=_HF_TOKEN)
        rng = random.Random(RANDOM_SEED + 6)

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


# ── Benign sources ─────────────────────────────────────────────────────────────

def collect_alpaca_benign(n: int = TARGET_ALPACA_BENIGN) -> list[dict]:
    """Stanford Alpaca — clean instruction-following, clearly non-adversarial."""
    print(f"  [5/6] Stanford Alpaca benign (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        rng = random.Random(RANDOM_SEED + 4)

        pure = [
            row for row in ds
            if not row.get("input", "").strip() and _keep(row.get("instruction", ""))
        ]
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
    """Anthropic HH-RLHF helpful subset — legitimate user questions."""
    print(f"  [6/6] Anthropic HH-RLHF helpful benign (n={n})...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
        rng = random.Random(RANDOM_SEED + 5)

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


# ── Utilities ──────────────────────────────────────────────────────────────────

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
    """80 / 10 / 10 stratified split by label."""
    rng = random.Random(seed)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]

    def _split(items):
        rng.shuffle(items)
        n      = len(items)
        n_val  = max(1, int(n * 0.10))
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print("PAIR Training Data Collection")
    print(f"  Output dir : {OUT_DIR}")
    print(f"  Timestamp  : {timestamp}\n")

    print("[1/3] Collecting adversarial examples...")
    attack_rows: list[dict] = []
    attack_rows += collect_jbb_all()
    attack_rows += collect_rlhf_redteam()
    attack_rows += collect_toxicchat()
    attack_rows += collect_harmfulqa()
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

    if n_attack < 200:
        print(f"\n  WARNING: only {n_attack} attack examples — too few to train reliably.")
        print("  Check which sources failed above and fix before training.")

    train, val, test = split_dataset(all_rows)

    save_jsonl(train, os.path.join(OUT_DIR, "train.jsonl"))
    save_jsonl(val,   os.path.join(OUT_DIR, "val.jsonl"))
    save_jsonl(test,  os.path.join(OUT_DIR, "test.jsonl"))

    stats = {
        "timestamp":     timestamp,
        "total":         len(all_rows),
        "n_attack":      n_attack,
        "n_benign":      n_benign,
        "class_balance": round(n_attack / len(all_rows), 3) if all_rows else 0,
        "splits":        {"train": len(train), "val": len(val), "test": len(test)},
        "sources":       {},
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

    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n  train.jsonl : {len(train)} examples")
    print(f"  val.jsonl   : {len(val)} examples")
    print(f"  test.jsonl  : {len(test)} examples  (HELD OUT — do not use until final eval)")
    print(f"\n  Class balance (attack ratio): {stats['class_balance']*100:.1f}%")
    print("\n  Source breakdown:")
    for src, d in stats["sources"].items():
        print(f"    {src:<35s} total={d['total']:4d}  attack={d['attack']:4d}  benign={d['benign']:4d}")
    print("\n  Done. Run data/train_pair_classifier.py next.")


if __name__ == "__main__":
    main()
