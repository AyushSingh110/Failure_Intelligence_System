from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Optional

import requests

# Configuration 
FIE_BASE_URL = "http://localhost:8000/api/v1"
FIE_API_KEY  = "fie-qvl80ejtjwscy5d8"   

LABELED_DIR  = os.path.join(os.path.dirname(__file__), "labeled")
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

# Delay between API calls to avoid overwhelming FIE (seconds)
REQUEST_DELAY = 2.0


# Built-in Q&A examples 
BUILTIN_EXAMPLES = [
    # HISTORY 
    {
        "question":       "Who invented the telephone?",
        "correct_answer": "Alexander Graham Bell",
        "wrong_answers":  ["Thomas Edison", "Nikola Tesla", "Samuel Morse"],
        "category":       "history",
        "domain":         "invention",
    },
    {
        "question":       "In which year did World War II end?",
        "correct_answer": "1945",
        "wrong_answers":  ["1944", "1946", "1943"],
        "category":       "history",
        "domain":         "dates",
    },
    {
        "question":       "Who was the first person to walk on the moon?",
        "correct_answer": "Neil Armstrong",
        "wrong_answers":  ["Buzz Aldrin", "Yuri Gagarin", "John Glenn"],
        "category":       "history",
        "domain":         "people",
    },
    {
        "question":       "Which country was the first to grant women the right to vote?",
        "correct_answer": "New Zealand",
        "wrong_answers":  ["Australia", "United States", "United Kingdom"],
        "category":       "history",
        "domain":         "facts",
    },
    {
        "question":       "Who wrote the play Romeo and Juliet?",
        "correct_answer": "William Shakespeare",
        "wrong_answers":  ["Christopher Marlowe", "John Milton", "Charles Dickens"],
        "category":       "history",
        "domain":         "literature",
    },
    {
        "question":       "What was the name of the first artificial satellite launched into space?",
        "correct_answer": "Sputnik 1",
        "wrong_answers":  ["Explorer 1", "Vostok 1", "Luna 1"],
        "category":       "history",
        "domain":         "science",
    },

    # SCIENCE 
    {
        "question":       "What is the chemical symbol for gold?",
        "correct_answer": "Au",
        "wrong_answers":  ["Go", "Gd", "Gl"],
        "category":       "science",
        "domain":         "chemistry",
    },
    {
        "question":       "How many bones are in the adult human body?",
        "correct_answer": "206",
        "wrong_answers":  ["205", "207", "210", "198"],
        "category":       "science",
        "domain":         "biology",
    },
    {
        "question":       "What is the speed of light in a vacuum?",
        "correct_answer": "approximately 299,792,458 metres per second",
        "wrong_answers":  ["300,000 km/s", "186,000 miles per second", "3 × 10^8 m/s"],
        "category":       "science",
        "domain":         "physics",
    },
    {
        "question":       "What is the most abundant gas in Earth's atmosphere?",
        "correct_answer": "Nitrogen",
        "wrong_answers":  ["Oxygen", "Carbon dioxide", "Argon"],
        "category":       "science",
        "domain":         "earth-science",
    },
    {
        "question":       "What is the powerhouse of the cell?",
        "correct_answer": "The mitochondria",
        "wrong_answers":  ["The nucleus", "The ribosome", "The chloroplast"],
        "category":       "science",
        "domain":         "biology",
    },
    {
        "question":       "What planet is known as the Red Planet?",
        "correct_answer": "Mars",
        "wrong_answers":  ["Venus", "Jupiter", "Saturn"],
        "category":       "science",
        "domain":         "astronomy",
    },
    {
        "question":       "What is the atomic number of carbon?",
        "correct_answer": "6",
        "wrong_answers":  ["4", "8", "12"],
        "category":       "science",
        "domain":         "chemistry",
    },
    {
        "question":       "Who developed the theory of general relativity?",
        "correct_answer": "Albert Einstein",
        "wrong_answers":  ["Isaac Newton", "Niels Bohr", "Max Planck"],
        "category":       "science",
        "domain":         "physics",
    },

    # GEOGRAPHY 
    {
        "question":       "What is the capital of Australia?",
        "correct_answer": "Canberra",
        "wrong_answers":  ["Sydney", "Melbourne", "Brisbane"],
        "category":       "geography",
        "domain":         "capitals",
    },
    {
        "question":       "What is the longest river in the world?",
        "correct_answer": "The Nile",
        "wrong_answers":  ["The Amazon", "The Yangtze", "The Mississippi"],
        "category":       "geography",
        "domain":         "physical-geography",
    },
    {
        "question":       "Which country has the largest land area in the world?",
        "correct_answer": "Russia",
        "wrong_answers":  ["Canada", "China", "United States"],
        "category":       "geography",
        "domain":         "facts",
    },
    {
        "question":       "What is the smallest country in the world?",
        "correct_answer": "Vatican City",
        "wrong_answers":  ["Monaco", "San Marino", "Liechtenstein"],
        "category":       "geography",
        "domain":         "facts",
    },
    {
        "question":       "Which ocean is the largest?",
        "correct_answer": "The Pacific Ocean",
        "wrong_answers":  ["The Atlantic Ocean", "The Indian Ocean", "The Arctic Ocean"],
        "category":       "geography",
        "domain":         "physical-geography",
    },
    {
        "question":       "What is the capital of Japan?",
        "correct_answer": "Tokyo",
        "wrong_answers":  ["Osaka", "Kyoto", "Hiroshima"],
        "category":       "geography",
        "domain":         "capitals",
    },

    # MATHEMATICS 
    {
        "question":       "What is the value of pi to two decimal places?",
        "correct_answer": "3.14",
        "wrong_answers":  ["3.12", "3.16", "3.41"],
        "category":       "mathematics",
        "domain":         "constants",
    },
    {
        "question":       "What is the square root of 144?",
        "correct_answer": "12",
        "wrong_answers":  ["11", "13", "14"],
        "category":       "mathematics",
        "domain":         "arithmetic",
    },
    {
        "question":       "How many sides does a hexagon have?",
        "correct_answer": "6",
        "wrong_answers":  ["5", "7", "8"],
        "category":       "mathematics",
        "domain":         "geometry",
    },
    {
        "question":       "What is the result of 15 multiplied by 15?",
        "correct_answer": "225",
        "wrong_answers":  ["215", "235", "220"],
        "category":       "mathematics",
        "domain":         "arithmetic",
    },

    # GENERAL KNOWLEDGE
    {
        "question":       "What is the hardest natural substance on Earth?",
        "correct_answer": "Diamond",
        "wrong_answers":  ["Corundum", "Quartz", "Graphite"],
        "category":       "general",
        "domain":         "science",
    },
    {
        "question":       "How many strings does a standard guitar have?",
        "correct_answer": "6",
        "wrong_answers":  ["5", "7", "8"],
        "category":       "general",
        "domain":         "music",
    },
    {
        "question":       "What is the tallest mountain in the world?",
        "correct_answer": "Mount Everest",
        "wrong_answers":  ["K2", "Kangchenjunga", "Lhotse"],
        "category":       "general",
        "domain":         "geography",
    },
    {
        "question":       "What language has the most native speakers in the world?",
        "correct_answer": "Mandarin Chinese",
        "wrong_answers":  ["English", "Spanish", "Hindi"],
        "category":       "general",
        "domain":         "linguistics",
    },
    {
        "question":       "How many players are on a standard soccer team?",
        "correct_answer": "11",
        "wrong_answers":  ["10", "12", "9"],
        "category":       "general",
        "domain":         "sports",
    },
    {
        "question":       "What is the chemical formula for water?",
        "correct_answer": "H2O",
        "wrong_answers":  ["H2O2", "HO", "H3O"],
        "category":       "science",
        "domain":         "chemistry",
    },
]


#Corruption functions

def corrupt_answer(correct: str, wrong_answers: list[str]) -> tuple[str, str]:
    """
    Returns (corrupted_answer, corruption_type).

    Priority:
    1. Use a pre-defined wrong answer from the dataset (most realistic)
    2. Fall back to a simple string mutation if no wrong answers available
    """
    if wrong_answers:
        chosen = random.choice(wrong_answers)
        return chosen, "predefined_wrong_answer"

    # Fallback mutations when no wrong answers are provided
    mutations = [
        (_swap_words(correct),      "word_swap"),
        (_add_wrong_year(correct),  "year_corruption"),
        (_negate_answer(correct),   "negation"),
    ]
    valid = [(m, t) for m, t in mutations if m != correct]
    if valid:
        return random.choice(valid)
    return "I don't know", "unknown_fallback"


def _swap_words(text: str) -> str:
    """Randomly swap two words — simple corruption."""
    words = text.split()
    if len(words) < 2:
        return text + " (incorrect)"
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def _add_wrong_year(text: str) -> str:
    """If text contains a 4-digit year, shift it by a random amount."""
    import re
    years = re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", text)
    if not years:
        return text
    year = years[0]
    shift = random.choice([-5, -3, -2, 2, 3, 5])
    return text.replace(year, str(int(year) + shift), 1)


def _negate_answer(text: str) -> str:
    """Adds 'not' to create a negation — a common LLM mistake pattern."""
    words = text.split()
    if len(words) >= 2:
        words.insert(1, "not")
        return " ".join(words)
    return "not " + text


# ── FIE API caller ────────────────────────────────────────────────────────────

def call_fie_monitor(
    question:       str,
    model_output:   str,
    primary_model:  str = "synthetic-test-model",
) -> Optional[dict]:
    """
    Calls the FIE /monitor endpoint with a (question, answer) pair.
    Returns the full JSON response, or None if the call failed.
    """
    url     = f"{FIE_BASE_URL}/monitor"
    headers = {"Content-Type": "application/json"}
    if FIE_API_KEY:
        headers["X-API-Key"] = FIE_API_KEY

    payload = {
        "prompt":              question,
        "primary_output":      model_output,
        "primary_model_name":  primary_model,
        "run_full_jury":       True,
        "latency_ms":          500,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"    FIE returned HTTP {resp.status_code}: {resp.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        print("    ERROR: Cannot connect to FIE backend.")
        print("    Make sure the server is running: uvicorn app.main:app --reload")
        return None
    except Exception as exc:
        print(f"    FIE call failed: {exc}")
        return None


def extract_fie_result(response: dict) -> dict:
    """
    Extracts the key signals from a FIE monitor response.
    Returns a flat dict with the most important values.
    """
    fsv    = response.get("failure_signal_vector", {})
    fix    = response.get("fix_result") or {}
    gt     = response.get("ground_truth") or {}
    jury   = response.get("jury") or {}
    pv     = jury.get("primary_verdict") or {} if jury else {}

    return {
        "high_failure_risk":    response.get("high_failure_risk", False),
        "entropy_score":        fsv.get("entropy_score", 0.0),
        "agreement_score":      fsv.get("agreement_score", 1.0),
        "archetype":            response.get("archetype", ""),
        "question_type":        fsv.get("question_type", "UNKNOWN"),
        "jury_verdict":         pv.get("root_cause", ""),
        "jury_confidence":      pv.get("confidence_score", 0.0),
        "fix_applied":          fix.get("fix_applied", False),
        "fix_strategy":         fix.get("fix_strategy", ""),
        "fix_confidence":       fix.get("fix_confidence", 0.0),
        "fix_output":           fix.get("fixed_output", ""),
        "gt_source":            gt.get("source", "none"),
        "gt_confidence":        gt.get("confidence", 0.0),
        "gt_override":          gt.get("verified_answer", "") != "",
        "requires_escalation":  gt.get("requires_escalation", False) or
                                response.get("requires_human_review", False),
        "classifier_probability": response.get("classifier_probability"),
        "model_version":        response.get("model_version", ""),
        "failure_summary":      response.get("failure_summary", ""),
    }


# ── Main generator ────────────────────────────────────────────────────────────

class SyntheticDataGenerator:

    def __init__(self, source: str = "builtin", failures_only: bool = False, subject: str = None):
        self.source        = source
        self.failures_only = failures_only
        self.subject       = subject
        self.examples      = self._load_examples()
        print(f"Loaded {len(self.examples)} source examples from '{source}'")

    def _load_examples(self) -> list[dict]:
        if self.source == "truthfulqa":
            return self._load_json_dataset(
                "truthfulqa.json",
                lambda d: {
                    "question":       d["question"],
                    "correct_answer": d["correct_answer"],
                    "wrong_answers":  d.get("incorrect_answers", []),
                    "category":       d.get("category", "unknown"),
                    "domain":         "truthfulqa",
                    "source":         "truthfulqa",
                },
                filter_fn=lambda d: bool(d.get("question") and d.get("correct_answer")),
            )
        if self.source == "mmlu":
            return self._load_json_dataset(
                "mmlu.json",
                lambda d: d,
                filter_fn=lambda d: bool(d.get("question") and d.get("correct_answer")),
                subject=getattr(self, "subject", None),
            )
        if self.source == "halueval":
            return self._load_json_dataset(
                "halueval.json",
                lambda d: d,
                filter_fn=lambda d: bool(d.get("question") and d.get("correct_answer")),
            )
        return BUILTIN_EXAMPLES

    def _load_json_dataset(
        self,
        filename:  str,
        transform,
        filter_fn=None,
        subject:   str = None,
    ) -> list[dict]:
        path = os.path.join(DATASETS_DIR, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found.")
            print(f"Run: python data/download_datasets.py --source {self.source}")
            print("Falling back to built-in examples.")
            return BUILTIN_EXAMPLES
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for d in data:
            if filter_fn and not filter_fn(d):
                continue
            if subject and d.get("category", "") != subject:
                continue
            examples.append(transform(d))
        print(f"Loaded {len(examples)} examples from {filename}"
              + (f" (subject={subject})" if subject else ""))
        return examples

    def run(self, n: int = 20) -> str:
        """
        Generates n labeled examples and saves them to data/labeled/.

        For each example we run TWO calls:
          1. Corrupted (wrong) answer   → FIE SHOULD detect (true positive)
          2. Correct answer             → FIE SHOULD NOT detect (true negative)
             (skipped if --failures-only)

        Returns the path of the saved JSONL file.
        """
        os.makedirs(LABELED_DIR, exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path   = os.path.join(LABELED_DIR, f"synthetic_{timestamp}.jsonl")
        summ_path  = os.path.join(LABELED_DIR, f"run_summary_{timestamp}.json")

        pool = random.sample(self.examples, min(n, len(self.examples)))

        stats = {
            "total_run":       0,
            "failure_cases":   0,
            "correct_cases":   0,
            "true_positives":  0,   # FIE caught a real failure
            "false_negatives": 0,   # FIE missed a real failure
            "true_negatives":  0,   # FIE correctly said "OK" for a correct answer
            "false_positives": 0,   # FIE flagged a correct answer
            "errors":          0,
        }

        print()
        print(f"Generating {len(pool)} example pairs ({len(pool) * (1 if self.failures_only else 2)} FIE calls)...")
        print(f"Output: {out_path}")
        print()

        with open(out_path, "w", encoding="utf-8") as out_f:

            for i, example in enumerate(pool, 1):
                q       = example["question"]
                correct = example["correct_answer"]
                wrong   = example.get("wrong_answers", [])
                cat     = example.get("category", "unknown")
                domain  = example.get("domain",   "unknown")

                print(f"[{i}/{len(pool)}] {q[:70]}")

                # ── Case A: Corrupted answer (FIE should detect this) ──────
                corrupted, corruption_type = corrupt_answer(correct, wrong)
                print(f"  FAILURE case: '{corrupted[:50]}' (corruption: {corruption_type})")

                resp_fail = call_fie_monitor(q, corrupted, "corrupt-model")
                time.sleep(REQUEST_DELAY)

                if resp_fail is None:
                    print("  SKIP (FIE unreachable)")
                    stats["errors"] += 1
                    continue

                fie_fail = extract_fie_result(resp_fail)
                fie_detected = fie_fail["high_failure_risk"] or fie_fail["fix_applied"]

                if fie_detected:
                    stats["true_positives"] += 1
                    outcome_a = "TRUE_POSITIVE ✓"
                else:
                    stats["false_negatives"] += 1
                    outcome_a = "FALSE_NEGATIVE ✗ (FIE missed this failure)"

                print(f"  → FIE: risk={fie_fail['high_failure_risk']} | verdict={fie_fail['jury_verdict']} | {outcome_a}")

                record_fail = {
                    "label_type":        "FAILURE",
                    "fie_should_detect": True,
                    "fie_detected":      fie_detected,
                    "outcome":           "true_positive" if fie_detected else "false_negative",
                    "question":          q,
                    "submitted_answer":  corrupted,
                    "correct_answer":    correct,
                    "corruption_type":   corruption_type,
                    "category":          cat,
                    "domain":            domain,
                    "question_type":     fie_fail.get("question_type", "UNKNOWN"),
                    "source":            example.get("source", self.source),
                    "fie_result":        fie_fail,
                    "timestamp":         datetime.utcnow().isoformat(),
                }
                out_f.write(json.dumps(record_fail, ensure_ascii=False) + "\n")
                stats["failure_cases"] += 1
                stats["total_run"]     += 1

                if self.failures_only:
                    continue

                # ── Case B: Correct answer (FIE should NOT flag this) ──────
                print(f"  CORRECT case: '{correct[:50]}'")

                resp_ok = call_fie_monitor(q, correct, "correct-model")
                time.sleep(REQUEST_DELAY)

                if resp_ok is None:
                    print("  SKIP (FIE unreachable)")
                    stats["errors"] += 1
                    continue

                fie_ok       = extract_fie_result(resp_ok)
                fie_flagged  = fie_ok["high_failure_risk"] or fie_ok["fix_applied"]

                if not fie_flagged:
                    stats["true_negatives"] += 1
                    outcome_b = "TRUE_NEGATIVE ✓"
                else:
                    stats["false_positives"] += 1
                    outcome_b = "FALSE_POSITIVE ✗ (FIE wrongly flagged correct answer)"

                print(f"  → FIE: risk={fie_ok['high_failure_risk']} | verdict={fie_ok['jury_verdict']} | {outcome_b}")
                print()

                record_ok = {
                    "label_type":        "CORRECT",
                    "fie_should_detect": False,
                    "fie_detected":      fie_flagged,
                    "outcome":           "true_negative" if not fie_flagged else "false_positive",
                    "question":          q,
                    "submitted_answer":  correct,
                    "correct_answer":    correct,
                    "corruption_type":   "none",
                    "category":          cat,
                    "domain":            domain,
                    "question_type":     fie_ok.get("question_type", "UNKNOWN"),
                    "source":            example.get("source", self.source),
                    "fie_result":        fie_ok,
                    "timestamp":         datetime.utcnow().isoformat(),
                }
                out_f.write(json.dumps(record_ok, ensure_ascii=False) + "\n")
                stats["correct_cases"] += 1
                stats["total_run"]     += 1

        # ── Summary ──────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  GENERATION COMPLETE")
        print("=" * 60)

        total_failure = stats["failure_cases"]
        total_correct = stats["correct_cases"]

        if total_failure > 0:
            recall    = round(stats["true_positives"]  / total_failure, 4)
            miss_rate = round(stats["false_negatives"] / total_failure, 4)
        else:
            recall = miss_rate = 0.0

        if total_correct > 0:
            tnr = round(stats["true_negatives"]  / total_correct, 4)
            fpr = round(stats["false_positives"] / total_correct, 4)
        else:
            tnr = fpr = 0.0

        print(f"  Total examples run:    {stats['total_run']}")
        print(f"  Errors (FIE offline):  {stats['errors']}")
        print()
        print("  FAILURE DETECTION (did FIE catch real failures?)")
        print(f"    True positives:   {stats['true_positives']} / {total_failure}")
        print(f"    False negatives:  {stats['false_negatives']} / {total_failure}")
        print(f"    Recall:           {recall:.1%}  (higher = better at catching failures)")
        print()
        if not self.failures_only:
            print("  FALSE ALARM RATE (did FIE wrongly flag correct answers?)")
            print(f"    True negatives:   {stats['true_negatives']} / {total_correct}")
            print(f"    False positives:  {stats['false_positives']} / {total_correct}")
            print(f"    Specificity:      {tnr:.1%}  (higher = fewer false alarms)")
            print()
        print(f"  Results saved to: {out_path}")

        summary = {
            **stats,
            "recall":      recall,
            "miss_rate":   miss_rate,
            "specificity": tnr,
            "fpr":         fpr,
            "timestamp":   datetime.now().isoformat(),
            "source":      self.source,
            "output_file": out_path,
            "interpretation": {
                "recall":      "% of real failures FIE caught. Target: > 80%",
                "miss_rate":   "% of real failures FIE missed. Target: < 20%",
                "specificity": "% of correct answers FIE did NOT flag. Target: > 70%",
                "fpr":         "% of correct answers FIE wrongly flagged. Target: < 30%",
            },
        }
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to:  {summ_path}")
        print()
        return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global FIE_BASE_URL, FIE_API_KEY, REQUEST_DELAY

    parser = argparse.ArgumentParser(
        description="FIE Synthetic Labeled Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/synthetic_generator.py
  python data/synthetic_generator.py --n 30
  python data/synthetic_generator.py --source truthfulqa --n 50
  python data/synthetic_generator.py --failures-only
  python data/synthetic_generator.py --fie-url http://localhost:8001/api/v1
        """,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of Q&A pairs to process (default: 10, max: dataset size)",
    )
    parser.add_argument(
        "--source",
        choices=["builtin", "truthfulqa", "mmlu", "halueval"],
        default="builtin",
        help="Which Q&A source to use (default: builtin)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="MMLU subject filter e.g. 'medicine', 'law', 'history' (mmlu only)",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Only run corrupted answers, skip correct-answer tests",
    )
    parser.add_argument(
        "--fie-url",
        default=FIE_BASE_URL,
        help=f"FIE backend URL (default: {FIE_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=FIE_API_KEY,
        help="FIE API key (leave empty if no auth)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Seconds between FIE calls (default: {REQUEST_DELAY})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible runs (default: 42)",
    )

    args = parser.parse_args()

    FIE_BASE_URL  = args.fie_url
    FIE_API_KEY   = args.api_key
    REQUEST_DELAY = args.delay

    random.seed(args.seed)

    print()
    print("=" * 60)
    print("  FIE Synthetic Data Generator")
    print("=" * 60)
    print(f"  FIE backend: {FIE_BASE_URL}")
    print(f"  Source:      {args.source}")
    print(f"  Examples:    {args.n}")
    print(f"  Mode:        {'failures only' if args.failures_only else 'failures + correct'}")
    print(f"  Seed:        {args.seed}")
    print()

    # Quick connectivity check
    try:
        r = requests.get(f"{FIE_BASE_URL.replace('/api/v1', '')}/docs", timeout=5)
        print(f"  FIE backend reachable (HTTP {r.status_code})")
    except Exception:
        print("  WARNING: Cannot reach FIE backend at", FIE_BASE_URL)
        print("  Start the server first: uvicorn app.main:app --reload")
        print()
        ans = input("  Continue anyway? (y/N): ").strip().lower()
        if ans != "y":
            sys.exit(0)

    gen = SyntheticDataGenerator(source=args.source, failures_only=args.failures_only, subject=args.subject)
    gen.run(n=args.n)


if __name__ == "__main__":
    main()
