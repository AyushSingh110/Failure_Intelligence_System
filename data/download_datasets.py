from __future__ import annotations
import argparse
import json
import os
import random
import sys

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "datasets")


# TruthfulQA
def download_truthfulqa():
    out    = os.path.join(OUTPUT_DIR, "truthfulqa.json")
    sample = os.path.join(OUTPUT_DIR, "truthfulqa_sample.json")

    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
        print(f"TruthfulQA already exists: {len(existing)} examples at {out}")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Run  pip install datasets  first.")
        sys.exit(1)

    print("Downloading TruthfulQA...")
    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation")
    except Exception as exc:
        print(f"Download failed: {exc}")
        sys.exit(1)

    records = []
    for item in dataset:
        best = item.get("best_answer", "") or (item.get("correct_answers") or [""])[0]
        if not best:
            continue
        records.append({
            "question":          item["question"],
            "correct_answer":    best,
            "correct_answers":   item.get("correct_answers", []),
            "incorrect_answers": item.get("incorrect_answers", []),
            "category":          item.get("category", "unknown"),
            "source":            "truthfulqa",
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records):,} TruthfulQA examples → {out}")

    random.seed(42)
    samp = random.sample(records, min(100, len(records)))
    with open(sample, "w", encoding="utf-8") as f:
        json.dump(samp, f, indent=2, ensure_ascii=False)
    print(f"Sample (100 ex) saved → {sample}")


# MMLU 

def download_mmlu():
    out = os.path.join(OUTPUT_DIR, "mmlu.json")

    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
        print(f"MMLU already exists: {len(existing):,} examples at {out}")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Run  pip install datasets  first.")
        sys.exit(1)

    print("Downloading MMLU (all subjects, test split)...")
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)

    examples = []
    for row in ds:
        choices = row["choices"]
        idx     = row["answer"]
        examples.append({
            "question":       row["question"],
            "correct_answer": choices[idx],
            "wrong_answers":  [choices[i] for i in range(len(choices)) if i != idx],
            "category":       row["subject"],
            "domain":         "mmlu",
            "source":         "mmlu",
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(examples):,} MMLU examples → {out}")

    subjects = sorted(set(e["category"] for e in examples))
    print(f"Subjects ({len(subjects)}): {', '.join(subjects[:8])}{'...' if len(subjects) > 8 else ''}")


# HaluEval 

def download_halueval():
    out = os.path.join(OUTPUT_DIR, "halueval.json")

    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
        print(f"HaluEval already exists: {len(existing):,} examples at {out}")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Run  pip install datasets  first.")
        sys.exit(1)

    print("Downloading HaluEval QA (real LLM hallucinations)...")
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    # Auto-detect field names — HuggingFace versions differ
    cols = ds.column_names
    print(f"  Detected columns: {cols}")

    correct_field = next((c for c in ["right_answer", "answer", "correct_answer"] if c in cols), None)
    halluc_field  = next((c for c in ["hallucinated_answer", "hallucination", "wrong_answer"] if c in cols), None)

    if not correct_field or not halluc_field:
        print(f"ERROR: Could not find expected fields. Available: {cols}")
        print("Update download_halueval() with the correct field names above.")
        sys.exit(1)

    print(f"  Using correct_field='{correct_field}', halluc_field='{halluc_field}'")

    examples = []
    for row in ds:
        examples.append({
            "question":       row["question"],
            "correct_answer": row[correct_field],
            "wrong_answers":  [row[halluc_field]],  # real LLM hallucination
            "category":       "halueval_qa",
            "domain":         "halueval",
            "source":         "halueval",
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(examples):,} HaluEval examples → {out}")

# Entry point 

def main():
    parser = argparse.ArgumentParser(description="Download datasets for FIE data generation")
    parser.add_argument(
        "--source",
        choices=["all", "truthfulqa", "mmlu", "halueval"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  FIE Dataset Downloader")
    print("=" * 60)
    print()

    if args.source in ("all", "truthfulqa"):
        download_truthfulqa()
        print()
    if args.source in ("all", "mmlu"):
        download_mmlu()
        print()
    if args.source in ("all", "halueval"):
        download_halueval()
        print()
    print("Done. Generate training data:")

if __name__ == "__main__":
    main()
