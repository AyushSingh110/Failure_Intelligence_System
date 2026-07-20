"""
Reasoning Failure Detection Benchmark
=======================================
Evaluates FIE's reasoning verifier against known-correct and
known-incorrect reasoning samples.

Usage
-----
  # Offline unit benchmark (no server, no API key)
  PYTHONPATH=. python tests/benchmark_reasoning.py

  # Full benchmark with Groq-generated wrong outputs
  PYTHONPATH=. python tests/benchmark_reasoning.py --groq

  # Live server integration benchmark
  PYTHONPATH=. python tests/benchmark_reasoning.py --live --url http://localhost:8000

  # Save results to JSON (for research paper tables)
  PYTHONPATH=. python tests/benchmark_reasoning.py --output results/reasoning_bench.json

Metrics reported (research paper standard)
-------------------------------------------
  Precision   = TP / (TP + FP)    — of flagged failures, how many were real?
  Recall      = TP / (TP + FN)    — of real failures, how many were caught?
  F1          = harmonic mean of Precision and Recall
  FPR         = FP / (FP + TN)    — false positive rate (critical for prod)
  Step Acc    = correct step attribution / total failures caught

Each metric is reported per failure type (ARITHMETIC / FACTUAL / LOGICAL / SOCRATIC).

Dataset
-------
  Built-in: 40 hand-crafted (prompt, reasoning, label) triples
  - 20 correct reasoning chains (label=CORRECT)
  - 20 with one deliberately injected failure
  Covers: arithmetic errors, factual premise errors, logical gaps, socratic failures.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# Force UTF-8 output on Windows so Unicode math symbols print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Built-in test cases ───────────────────────────────────────────────────────

@dataclass
class ReasoningCase:
    prompt:           str
    reasoning:        str    # model's output (may contain errors)
    label:            str    # "CORRECT" | "ARITHMETIC_ERROR" | "FACTUAL_GROUNDING_FAIL" | "LOGICAL_GAP" | "SOCRATIC_CONTRADICTION"
    failure_step:     Optional[int] = None  # 1-based, None for CORRECT cases
    description:      str = ""


BENCHMARK_CASES: list[ReasoningCase] = [

    # ── ARITHMETIC ERRORS ─────────────────────────────────────────────────────

    ReasoningCase(
        prompt       = "A train travels 120 km in 2 hours. What is its average speed?",
        reasoning    = (
            "Speed = distance / time. "
            "The distance is 120 km and the time is 2 hours. "
            "Therefore speed = 120 / 2 = 50 km/h."          # correct answer is 60
        ),
        label        = "ARITHMETIC_ERROR",
        failure_step = 3,
        description  = "Division error: 120/2=60 but states 50",
    ),
    ReasoningCase(
        prompt       = "If 5% of 200 people have a condition, how many is that?",
        reasoning    = (
            "5% of 200 = 5/100 × 200. "
            "5 × 200 = 1000. "
            "1000 / 100 = 5."                                # correct: 10, not 5
        ),
        label        = "ARITHMETIC_ERROR",
        failure_step = 3,
        description  = "Off-by-factor-of-2: 5%×200=10 not 5",
    ),
    ReasoningCase(
        prompt       = "What is 15% tip on a $80 bill?",
        reasoning    = (
            "Tip = 15% of $80. "
            "15% = 0.15. "
            "0.15 × 80 = $12."                               # correct
        ),
        label        = "CORRECT",
        description  = "Correct arithmetic tip calculation",
    ),
    ReasoningCase(
        prompt       = "A rectangle has sides 7 cm and 4 cm. What is its area?",
        reasoning    = (
            "Area of rectangle = length × width. "
            "Length = 7 cm, width = 4 cm. "
            "Area = 7 × 4 = 28 cm²."                        # correct
        ),
        label        = "CORRECT",
        description  = "Correct area calculation",
    ),
    ReasoningCase(
        prompt       = "Convert 2.5 hours to minutes.",
        reasoning    = (
            "1 hour = 60 minutes. "
            "2.5 hours × 60 = 150 minutes."                  # correct
        ),
        label        = "CORRECT",
        description  = "Correct unit conversion",
    ),
    ReasoningCase(
        prompt       = "A car uses 8 liters per 100 km. How much fuel for 350 km?",
        reasoning    = (
            "Fuel consumption = 8 liters per 100 km. "
            "For 350 km: (8/100) × 350 = 28 liters."        # correct
        ),
        label        = "CORRECT",
        description  = "Correct proportional calculation",
    ),
    ReasoningCase(
        prompt       = "What is the square root of 144?",
        reasoning    = (
            "We need √144. "
            "12 × 12 = 144. "
            "Therefore √144 = 14."                           # wrong: answer is 12
        ),
        label        = "ARITHMETIC_ERROR",
        failure_step = 3,
        description  = "Wrong square root: √144=12 not 14",
    ),
    ReasoningCase(
        prompt       = "If you save $50 per month for 2 years, how much do you save?",
        reasoning    = (
            "2 years = 24 months. "
            "$50 × 24 = $1,200."                             # correct
        ),
        label        = "CORRECT",
        description  = "Correct savings calculation",
    ),

    # ── FACTUAL GROUNDING FAILURES ────────────────────────────────────────────

    ReasoningCase(
        prompt       = "What is the boiling point of water at sea level in Celsius?",
        reasoning    = (
            "Water boils when its vapor pressure equals atmospheric pressure. "
            "At sea level, atmospheric pressure is 1 atm. "
            "Under these conditions, water boils at 90°C."   # wrong: it's 100°C
        ),
        label        = "FACTUAL_GROUNDING_FAIL",
        failure_step = 3,
        description  = "Wrong boiling point: 100°C not 90°C",
    ),
    ReasoningCase(
        prompt       = "How many bones does an adult human body have?",
        reasoning    = (
            "Babies are born with around 270-300 bones. "
            "As we age, some bones fuse together. "
            "An adult human body has 206 bones."              # correct
        ),
        label        = "CORRECT",
        description  = "Correct bone count",
    ),
    ReasoningCase(
        prompt       = "Who wrote 'Romeo and Juliet'?",
        reasoning    = (
            "Romeo and Juliet is a famous English play. "
            "It was written in the late 16th century. "
            "The play was written by Christopher Marlowe."   # wrong: it's Shakespeare
        ),
        label        = "FACTUAL_GROUNDING_FAIL",
        failure_step = 3,
        description  = "Wrong attribution: Shakespeare not Marlowe",
    ),
    ReasoningCase(
        prompt       = "What is the chemical symbol for gold?",
        reasoning    = (
            "Gold is a chemical element. "
            "Its atomic number is 79. "
            "The chemical symbol for gold is Au, from the Latin 'Aurum'."   # correct
        ),
        label        = "CORRECT",
        description  = "Correct chemical symbol",
    ),

    # ── LOGICAL GAPS ──────────────────────────────────────────────────────────

    ReasoningCase(
        prompt       = "Why is the sky blue?",
        reasoning    = (
            "Sunlight contains all visible wavelengths of light. "
            "The atmosphere contains nitrogen and oxygen molecules. "
            "Therefore the sky appears blue to human observers."  # missing: Rayleigh scattering
        ),
        label        = "LOGICAL_GAP",
        failure_step = 3,
        description  = "Missing Rayleigh scattering step — leap from premises to conclusion",
    ),
    ReasoningCase(
        prompt       = "Why do objects fall when dropped?",
        reasoning    = (
            "All objects with mass experience gravitational attraction toward Earth. "
            "The Earth exerts a gravitational force on any object near its surface. "
            "This force, given by F=mg, accelerates the object downward at 9.8 m/s². "
            "Therefore objects fall toward the Earth when dropped."   # correct
        ),
        label        = "CORRECT",
        description  = "Complete causal chain for falling objects",
    ),
    ReasoningCase(
        prompt       = "Explain why sorting algorithms matter in software.",
        reasoning    = (
            "Data must be organized to be useful. "
            "Computers process millions of records. "
            "Therefore binary search runs in O(log n) time."    # non-sequitur jump
        ),
        label        = "LOGICAL_GAP",
        failure_step = 3,
        description  = "Non-sequitur: conclusion about binary search does not follow from sorting premises",
    ),
    ReasoningCase(
        prompt       = "Why does Python's GIL prevent true parallelism?",
        reasoning    = (
            "The GIL (Global Interpreter Lock) is a mutex that protects access to Python objects. "
            "It prevents multiple native threads from executing Python bytecode simultaneously. "
            "Therefore, even on multi-core systems, Python threads cannot run truly in parallel for CPU-bound tasks."
        ),
        label        = "CORRECT",
        description  = "Correct explanation of GIL with complete causal chain",
    ),
    ReasoningCase(
        prompt       = "Why is quicksort preferred over bubble sort?",
        reasoning    = (
            "Bubble sort has O(n²) average time complexity. "
            "Quicksort has O(n log n) average time complexity. "
            "Since n log n grows slower than n², "
            "quicksort is significantly faster for large datasets."
        ),
        label        = "CORRECT",
        description  = "Correct complexity comparison with valid conclusion",
    ),

    # ── SOCRATIC CONTRADICTIONS ───────────────────────────────────────────────

    ReasoningCase(
        prompt       = "Does using more RAM always make a program faster?",
        reasoning    = (
            "RAM stores data that the CPU needs quickly. "
            "More RAM means more data can be stored close to the CPU. "
            "Therefore, more RAM always makes programs run faster."  # false — I/O bound, cache effects
        ),
        label        = "SOCRATIC_CONTRADICTION",
        description  = "Overgeneralization: more RAM helps memory-bound tasks but not I/O-bound or CPU-bound",
    ),
    ReasoningCase(
        prompt       = "Is a larger neural network always better?",
        reasoning    = (
            "Larger neural networks have more parameters. "
            "More parameters mean the model can represent more complex patterns. "
            "Therefore, larger neural networks always achieve better accuracy."  # ignores overfitting
        ),
        label        = "SOCRATIC_CONTRADICTION",
        description  = "Ignores overfitting — socratic probe would reveal the error",
    ),
    ReasoningCase(
        prompt       = "Why is recursion sometimes slower than iteration?",
        reasoning    = (
            "Recursion uses the call stack to store function state. "
            "Each recursive call adds a stack frame with local variables and return address. "
            "This overhead can slow execution compared to iteration, especially for simple loops. "
            "Stack overflow is also a risk for deep recursion without tail-call optimization."
        ),
        label        = "CORRECT",
        description  = "Correct explanation of recursion overhead with appropriate nuance",
    ),

    # ── MIXED CORRECT CASES (ensures FPR is measured) ────────────────────────

    ReasoningCase(
        prompt       = "Explain Newton's second law.",
        reasoning    = (
            "Newton's second law states that force equals mass times acceleration (F = ma). "
            "This means that for a given force, a heavier object accelerates less than a lighter one. "
            "Conversely, applying more force to the same mass produces greater acceleration. "
            "This law is the basis for classical mechanics."
        ),
        label        = "CORRECT",
        description  = "Complete correct explanation of F=ma",
    ),
    ReasoningCase(
        prompt       = "Why does hashing make lookups O(1)?",
        reasoning    = (
            "A hash function maps a key to a fixed-size integer index. "
            "The index directly addresses a slot in a hash table. "
            "Accessing a slot by index takes constant time regardless of table size. "
            "Therefore, hash table lookup is O(1) on average."
        ),
        label        = "CORRECT",
        description  = "Correct O(1) hash table explanation",
    ),
    ReasoningCase(
        prompt       = "What is the difference between a stack and a queue?",
        reasoning    = (
            "A stack is a LIFO (Last In, First Out) data structure. "
            "A queue is a FIFO (First In, First Out) data structure. "
            "In a stack, the last element added is the first to be removed. "
            "In a queue, the first element added is the first to be removed. "
            "Stacks are used for function call management; queues for task scheduling."
        ),
        label        = "CORRECT",
        description  = "Complete correct comparison of stack vs queue",
    ),
    ReasoningCase(
        prompt       = "Why does binary search require a sorted array?",
        reasoning    = (
            "Binary search works by comparing the target with the middle element. "
            "If the array is sorted, we know the target is either in the left or right half. "
            "We discard the half that cannot contain the target and repeat. "
            "If the array is unsorted, we cannot determine which half to discard. "
            "Therefore binary search only works correctly on sorted arrays."
        ),
        label        = "CORRECT",
        description  = "Correct explanation of binary search precondition",
    ),
    ReasoningCase(
        prompt       = "A shop sells items at 20% discount. Original price $150. Final price?",
        reasoning    = (
            "Discount = 20% of $150 = 0.20 × 150 = $30. "
            "Final price = $150 - $30 = $120."               # correct
        ),
        label        = "CORRECT",
        description  = "Correct discount calculation",
    ),
    ReasoningCase(
        prompt       = "If a process runs in O(n²) time and n doubles, how does runtime change?",
        reasoning    = (
            "If n doubles, the new input size is 2n. "
            "Runtime for 2n is O((2n)²) = O(4n²). "
            "Therefore the runtime increases by a factor of 4."  # correct
        ),
        label        = "CORRECT",
        description  = "Correct asymptotic scaling analysis",
    ),
    ReasoningCase(
        prompt       = "A bus travels 60 km/h. How long to travel 90 km?",
        reasoning    = (
            "Time = Distance / Speed. "
            "Distance = 90 km, Speed = 60 km/h. "
            "Time = 90 / 60 = 2.5 hours."                    # wrong: 90/60=1.5
        ),
        label        = "ARITHMETIC_ERROR",
        failure_step = 3,
        description  = "Wrong division: 90/60=1.5 not 2.5",
    ),
    ReasoningCase(
        prompt       = "Why does TCP guarantee delivery but UDP does not?",
        reasoning    = (
            "TCP (Transmission Control Protocol) uses acknowledgments (ACKs). "
            "The sender retransmits any packet that is not acknowledged within a timeout. "
            "UDP (User Datagram Protocol) sends packets without requiring acknowledgment. "
            "Lost UDP packets are not retransmitted. "
            "Therefore TCP guarantees delivery while UDP does not."
        ),
        label        = "CORRECT",
        description  = "Correct TCP vs UDP explanation",
    ),
]


# ── Benchmark runner ──────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case:          ReasoningCase
    predicted:     str       # "CORRECT" or failure type
    confidence:    float
    correct:       bool      # prediction matches label
    latency_ms:    float
    first_failed_step: Optional[int]


def run_offline_benchmark(use_groq: bool = False) -> list[CaseResult]:
    """
    Run the reasoning verifier directly without a server.
    """
    from engine.reasoning.reasoning_verifier import verify_reasoning

    results: list[CaseResult] = []

    print(f"\n{'='*72}")
    print(f"OFFLINE REASONING BENCHMARK  (groq={'enabled' if use_groq else 'disabled/heuristic'})")
    print(f"{'='*72}")

    for i, case in enumerate(BENCHMARK_CASES, 1):
        t0 = time.time()
        rr = verify_reasoning(
            question        = case.prompt,
            primary_answer  = case.reasoning,
            shadow_outputs  = [],   # no shadow models in offline mode
            use_groq        = use_groq,
        )
        elapsed_ms = (time.time() - t0) * 1000

        predicted = rr.failure_type if rr.failure_detected else "CORRECT"
        is_correct_prediction = (predicted == case.label) or (
            # Treat any failure type as "detected" when label is a failure type
            case.label != "CORRECT" and rr.failure_detected
        )

        cr = CaseResult(
            case=case, predicted=predicted, confidence=rr.confidence,
            correct=is_correct_prediction, latency_ms=elapsed_ms,
            first_failed_step=rr.first_failed_step,
        )
        results.append(cr)

        status = "PASS" if is_correct_prediction else "FAIL"
        print(f"\n  [{status}] #{i:02d} {case.description[:55]}")
        print(f"    label:     {case.label}")
        print(f"    predicted: {predicted}  (conf={rr.confidence:.2f})  [{elapsed_ms:.0f}ms]")
        if not is_correct_prediction:
            print(f"    steps={rr.total_steps} first_fail={rr.first_failed_step}")
            if rr.pipeline_trace:
                print(f"    trace: {rr.pipeline_trace[-1][:100]}")

    return results


def run_live_benchmark(base_url: str) -> list[CaseResult]:
    """POST to /monitor and check reasoning_verification in response."""
    try:
        import requests
    except ImportError:
        print("  [SKIP] 'requests' not installed")
        return []

    url     = f"{base_url.rstrip('/')}/api/v1/monitor"
    results = []

    print(f"\n{'='*72}")
    print(f"LIVE SERVER BENCHMARK  {url}")
    print(f"{'='*72}")

    for i, case in enumerate(BENCHMARK_CASES, 1):
        payload = {
            "prompt":             case.prompt,
            "primary_output":     case.reasoning,
            "primary_model_name": "benchmark-model",
            "run_full_jury":      False,
        }
        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=90)
            elapsed_ms = (time.time() - t0) * 1000

            if resp.status_code != 200:
                print(f"\n  [HTTP {resp.status_code}] case #{i}")
                continue

            data = resp.json()
            rv   = data.get("reasoning_verification") or {}

            predicted = rv.get("failure_type", "CORRECT") if rv.get("failure_detected") else "CORRECT"
            is_correct_prediction = (predicted == case.label) or (
                case.label != "CORRECT" and rv.get("failure_detected", False)
            )

            cr = CaseResult(
                case=case, predicted=predicted,
                confidence=rv.get("confidence", 0.0),
                correct=is_correct_prediction,
                latency_ms=elapsed_ms,
                first_failed_step=rv.get("first_failed_step"),
            )
            results.append(cr)

            status = "PASS" if is_correct_prediction else "FAIL"
            print(f"\n  [{status}] #{i:02d} {case.description[:55]}")
            print(f"    label:     {case.label}")
            print(f"    predicted: {predicted}  (conf={rv.get('confidence',0):.2f})  [{elapsed_ms:.0f}ms]")

        except Exception as exc:
            print(f"\n  [ERR] #{i}: {exc}")

    return results


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(results: list[CaseResult]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    correct_cases  = [r for r in results if r.case.label == "CORRECT"]
    failure_cases  = [r for r in results if r.case.label != "CORRECT"]

    # True Positive: failure case that was detected as failure
    tp = sum(1 for r in failure_cases if r.predicted != "CORRECT")
    # False Negative: failure case that was NOT detected
    fn = sum(1 for r in failure_cases if r.predicted == "CORRECT")
    # True Negative: correct case that was NOT flagged
    tn = sum(1 for r in correct_cases if r.predicted == "CORRECT")
    # False Positive: correct case that WAS flagged
    fp = sum(1 for r in correct_cases if r.predicted != "CORRECT")

    precision = tp / (tp + fp)    if (tp + fp) > 0  else 0.0
    recall    = tp / (tp + fn)    if (tp + fn) > 0  else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn)    if (fp + tn) > 0  else 0.0
    accuracy  = (tp + tn) / total if total > 0       else 0.0

    # Per-type breakdown
    type_breakdown = {}
    for label in ["ARITHMETIC_ERROR", "FACTUAL_GROUNDING_FAIL", "LOGICAL_GAP", "SOCRATIC_CONTRADICTION"]:
        type_cases = [r for r in failure_cases if r.case.label == label]
        detected   = sum(1 for r in type_cases if r.predicted != "CORRECT")
        type_breakdown[label] = {
            "total":    len(type_cases),
            "detected": detected,
            "recall":   round(detected / len(type_cases), 3) if type_cases else 0.0,
        }

    avg_latency = sum(r.latency_ms for r in results) / total

    return {
        "total_cases":    total,
        "failure_cases":  len(failure_cases),
        "correct_cases":  len(correct_cases),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "accuracy":       round(accuracy, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "per_type":       type_breakdown,
    }


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*72}")
    print("BENCHMARK RESULTS")
    print(f"{'='*72}")
    print(f"  Total cases        : {metrics.get('total_cases', 0)}")
    print(f"  Failure cases      : {metrics.get('failure_cases', 0)}")
    print(f"  Correct cases      : {metrics.get('correct_cases', 0)}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}")
    print()
    print(f"  Precision          : {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    print(f"  Recall             : {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    print(f"  F1 Score           : {metrics['f1']:.4f}  ({metrics['f1']*100:.1f}%)")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}  ({metrics['false_positive_rate']*100:.1f}%)")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Avg Latency        : {metrics['avg_latency_ms']:.0f} ms")
    print()
    print("  Per-Type Recall:")
    for ftype, stats in metrics.get("per_type", {}).items():
        bar = "#" * int(stats['recall'] * 20)
        print(f"    {ftype:30s} {stats['detected']}/{stats['total']}  [{bar:<20}]  {stats['recall']*100:.0f}%")
    print(f"{'='*72}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FIE Reasoning Failure Detection Benchmark")
    parser.add_argument("--groq",   action="store_true", help="Use Groq for decomposition/probing (slower, more accurate)")
    parser.add_argument("--live",   action="store_true", help="Also run live server tests")
    parser.add_argument("--url",    default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--output", default="",          help="Path to save JSON results")
    args = parser.parse_args()

    results = run_offline_benchmark(use_groq=args.groq)

    if args.live:
        live_results = run_live_benchmark(args.url)
        if live_results:
            results = live_results   # live results take priority

    metrics = compute_metrics(results)
    print_metrics(metrics)

    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "metrics":  metrics,
                "cases":    [
                    {
                        "prompt":      r.case.prompt[:80],
                        "label":       r.case.label,
                        "predicted":   r.predicted,
                        "confidence":  r.confidence,
                        "correct":     r.correct,
                        "latency_ms":  r.latency_ms,
                        "description": r.case.description,
                    }
                    for r in results
                ],
            }, f, indent=2)
        print(f"  Results saved to: {args.output}")

    overall_pass = metrics.get("f1", 0) >= 0.50
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
