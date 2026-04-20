import json
import os
import sys

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "datasets")
FULL_PATH    = os.path.join(OUTPUT_DIR, "truthfulqa.json")
SAMPLE_PATH  = os.path.join(OUTPUT_DIR, "truthfulqa_sample.json")


def download_truthfulqa():
    print("=" * 60)
    print("  FIE Dataset Downloader")
    print("=" * 60)
    print()

    # Check if already downloaded
    if os.path.exists(FULL_PATH):
        with open(FULL_PATH) as f:
            existing = json.load(f)
        print(f"TruthfulQA already downloaded: {len(existing)} examples at {FULL_PATH}")
        print("Delete the file and re-run if you want to refresh.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: HuggingFace datasets library not installed.")
        print("Run:  pip install datasets")
        sys.exit(1)

    print("Downloading TruthfulQA from HuggingFace...")
    print("(This requires internet — ~2MB download)")
    print()

    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation")
    except Exception as exc:
        print(f"Download failed: {exc}")
        print()
        print("This may be a network issue or HuggingFace may require authentication.")
        print("The synthetic_generator.py has 40 built-in examples and works without")
        print("this dataset, so you can skip this step for now.")
        sys.exit(1)

    print(f"Downloaded {len(dataset)} examples.")
    print()

    # Convert to clean list of dicts
    records = []
    for item in dataset:
        correct_answers   = item.get("correct_answers", [])
        incorrect_answers = item.get("incorrect_answers", [])
        best_answer       = item.get("best_answer", "")
        if not best_answer and correct_answers:
            best_answer = correct_answers[0]

        # Skip entries with no verifiable correct answer
        if not best_answer:
            continue

        record = {
            "question":           item["question"],
            "correct_answer":     best_answer,
            "correct_answers":    correct_answers,
            "incorrect_answers":  incorrect_answers,
            "category":           item.get("category", "unknown"),
            "source":             "truthfulqa",
        }
        records.append(record)

    print(f"Clean records extracted: {len(records)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FULL_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Full dataset saved: {FULL_PATH}")

    # Save a 100-example sample for quick testing
    import random
    random.seed(42)
    sample = random.sample(records, min(100, len(records)))
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Sample (100 examples) saved: {SAMPLE_PATH}")

    print()
    print("Done. Now run:")
    print("  python data/synthetic_generator.py --source truthfulqa --n 50")


def show_sample():
    """Prints 5 examples from the downloaded dataset so you can inspect the format."""
    if not os.path.exists(SAMPLE_PATH):
        print("Dataset not yet downloaded. Run this script first.")
        return
    with open(SAMPLE_PATH) as f:
        sample = json.load(f)
    print(f"\nShowing 5 random examples from {SAMPLE_PATH}:\n")
    import random
    for item in random.sample(sample, min(5, len(sample))):
        print(f"  Question:  {item['question']}")
        print(f"  Correct:   {item['correct_answer']}")
        wrong = item["incorrect_answers"][:2] if item["incorrect_answers"] else ["(none)"]
        print(f"  Wrong ex:  {wrong}")
        print(f"  Category:  {item['category']}")
        print()


if __name__ == "__main__":
    download_truthfulqa()
    show_sample()
