"""
FIE Reasoning Failure Detection — Dataset Evaluation Suite
============================================================
Downloads and evaluates FIE's reasoning verifier on four standard benchmarks:

  GSM8K       — 1,319 grade-school math word problems (arithmetic chains)
  MATH-500    — 500 competition math problems (multi-step reasoning, Hendrycks et al.)
  TruthfulQA  — 817 factual questions with known wrong answers (hallucination benchmark)
  GPQA        — 448 graduate-level STEM questions by domain experts (GATED — needs HF login)
                Visit https://huggingface.co/datasets/Idavidrein/gpqa to request free access

Usage
-----
  # Step 1: Install dependencies
  pip install datasets tqdm requests

  # Step 2: Download datasets (auto via HuggingFace hub)
  PYTHONPATH=. python tests/eval_datasets.py --download-only

  # Step 3: Run offline evaluation (no API key needed)
  PYTHONPATH=. python tests/eval_datasets.py --offline --dataset gsm8k

  # Step 4: Run with Groq (much higher recall for logical/socratic)
  PYTHONPATH=. python tests/eval_datasets.py --groq --dataset gsm8k

  # Step 5: Run all datasets + save results to JSON
  PYTHONPATH=. python tests/eval_datasets.py --groq --dataset all --output results/eval_all.json

  # Step 6: Live server benchmark (server must be running on :8000)
  PYTHONPATH=. python tests/eval_datasets.py --live --url http://localhost:8000 --dataset gsm8k --limit 50

Datasets downloaded to:  ~/.cache/huggingface/datasets/
Results JSON written to: ./results/<timestamp>.json   (unless --output specified)

Metrics reported (research paper standard)
------------------------------------------
  Precision   = TP / (TP + FP)    — of flagged failures, how many were real?
  Recall      = TP / (TP + FN)    — of real failures, how many were caught?
  F1          = harmonic mean of Precision and Recall
  FPR         = FP / (FP + TN)    — false positive rate (critical for production)
  Per-type breakdown across failure types
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Evaluation case dataclass ─────────────────────────────────────────────────

@dataclass
class EvalCase:
    case_id:       str
    dataset:       str
    prompt:        str
    reasoning:     str        # the model's reasoning chain to evaluate
    ground_truth:  str        # correct final answer
    model_answer:  str        # what the model actually answered (may be wrong)
    is_wrong:      bool       # True = this is a FAILURE case (answer is incorrect)
    failure_type:  str        # ARITHMETIC_ERROR | FACTUAL_GROUNDING_FAIL | LOGICAL_GAP | CORRECT
    difficulty:    str = "UNKNOWN"   # EASY | MEDIUM | HARD (from dataset metadata)


@dataclass
class EvalResult:
    case:          EvalCase
    predicted_failure: bool
    predicted_type:    str
    confidence:    float
    latency_ms:    float
    total_steps:   int
    first_failed_step: Optional[int]
    pipeline_trace: list[str] = field(default_factory=list)

    @property
    def is_correct_prediction(self) -> bool:
        if self.case.is_wrong:
            return self.predicted_failure
        else:
            return not self.predicted_failure


# ── GSM8K Loader ─────────────────────────────────────────────────────────────

def _inject_arithmetic_error(answer_text: str) -> tuple[str, str]:
    """
    Take a correct GSM8K reasoning chain and inject a subtle arithmetic error
    to create a FAILURE case. Returns (corrupted_answer, original_answer).
    """
    # Find any "= NUMBER" in the chain and corrupt the last one
    pattern = re.compile(r'(= )(\d+(?:\.\d+)?)', re.MULTILINE)
    matches = list(pattern.finditer(answer_text))
    if not matches:
        return answer_text, answer_text

    # Corrupt a middle equation (not the last one, which is the answer)
    target = matches[len(matches) // 2]
    original_val = float(target.group(2))
    # Add ~20% error — large enough to be clearly wrong
    corrupted_val = round(original_val * 1.22)
    corrupted = (
        answer_text[:target.start(2)]
        + str(corrupted_val)
        + answer_text[target.end(2):]
    )
    return corrupted, answer_text


def load_gsm8k(limit: int = 200, error_injection_ratio: float = 0.4) -> list[EvalCase]:
    """
    Load GSM8K test split. For each problem:
    - CORRECT cases: use the original correct reasoning chain.
    - FAILURE cases: inject an arithmetic error into a correct chain.

    GSM8K format: question | answer (reasoning chain + #### final_answer)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Install 'datasets': pip install datasets")
        return []

    print("  Loading GSM8K from HuggingFace (downloads ~50 MB on first run)...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"  [ERROR] Could not load GSM8K: {e}")
        return []

    cases: list[EvalCase] = []
    for i, row in enumerate(ds):
        if len(cases) >= limit:
            break

        question = row["question"].strip()
        answer   = row["answer"].strip()

        # Split reasoning from final answer (separated by ####)
        if "####" in answer:
            reasoning_part, final_answer = answer.split("####", 1)
            reasoning_part = reasoning_part.strip()
            final_answer   = final_answer.strip()
        else:
            reasoning_part = answer
            final_answer   = ""

        # Create CORRECT case
        cases.append(EvalCase(
            case_id      = f"gsm8k_{i:04d}_correct",
            dataset      = "gsm8k",
            prompt       = question,
            reasoning    = reasoning_part,
            ground_truth = final_answer,
            model_answer = final_answer,
            is_wrong     = False,
            failure_type = "CORRECT",
            difficulty   = "EASY",
        ))

        # Create FAILURE case by injecting an error (controlled rate)
        if len(cases) % int(1.0 / error_injection_ratio) == 0:
            corrupted, original = _inject_arithmetic_error(reasoning_part)
            if corrupted != original:
                cases.append(EvalCase(
                    case_id      = f"gsm8k_{i:04d}_error",
                    dataset      = "gsm8k",
                    prompt       = question,
                    reasoning    = corrupted,
                    ground_truth = final_answer,
                    model_answer = "WRONG",      # signal that the answer would be wrong
                    is_wrong     = True,
                    failure_type = "ARITHMETIC_ERROR",
                    difficulty   = "EASY",
                ))
                if len(cases) >= limit:
                    break

    print(f"  Loaded {len(cases)} GSM8K cases "
          f"({sum(1 for c in cases if c.is_wrong)} failure, "
          f"{sum(1 for c in cases if not c.is_wrong)} correct)")
    return cases


# ── MATH Dataset Loader ───────────────────────────────────────────────────────

def load_math(limit: int = 100, error_injection_ratio: float = 0.4) -> list[EvalCase]:
    """
    Load MATH-500 (curated 500-problem subset, freely accessible, no gating).
    Uses HuggingFaceH4/MATH-500 — same problems as Hendrycks MATH benchmark.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Install 'datasets': pip install datasets")
        return []

    print("  Loading MATH-500 from HuggingFace (downloads ~5 MB on first run)...")
    ds = None
    for repo in [
        ("HuggingFaceH4/MATH-500",    None,    "test"),
        ("EleutherAI/hendrycks_math", "algebra", "test"),
        ("EleutherAI/hendrycks_math", "counting_and_probability", "test"),
    ]:
        repo_id, config, split = repo
        try:
            ds = load_dataset(repo_id, config, split=split) if config else load_dataset(repo_id, split=split)
            print(f"    OK — {len(ds)} examples from {repo_id}")
            break
        except Exception as e:
            print(f"    {repo_id}: {str(e)[:80]}")

    if ds is None:
        print("  [ERROR] Could not load any MATH variant — skipping.")
        return []

    _difficulty_map = {"Level 1": "EASY", "Level 2": "EASY", "Level 3": "MEDIUM",
                       "Level 4": "HARD", "Level 5": "HARD",
                       "1": "EASY", "2": "EASY", "3": "MEDIUM", "4": "HARD", "5": "HARD"}

    cases: list[EvalCase] = []
    for i, row in enumerate(ds):
        if len(cases) >= limit:
            break

        question   = (row.get("problem") or row.get("question") or "").strip()
        solution   = (row.get("solution") or row.get("answer") or "").strip()
        # MATH-500 uses "Level N" strings; EleutherAI uses integer "level"
        raw_level  = str(row.get("level", "3"))
        difficulty = _difficulty_map.get(raw_level, "MEDIUM")

        if not question or not solution:
            continue

        # Extract final answer from \boxed{} if present
        boxed = re.search(r'\\boxed\{([^}]+)\}', solution)
        final_answer = boxed.group(1) if boxed else ""

        cases.append(EvalCase(
            case_id      = f"math_{i:04d}_correct",
            dataset      = "math",
            prompt       = question,
            reasoning    = solution,
            ground_truth = final_answer,
            model_answer = final_answer,
            is_wrong     = False,
            failure_type = "CORRECT",
            difficulty   = difficulty,
        ))

        if len(cases) % int(1.0 / error_injection_ratio) == 0:
            corrupted, original = _inject_arithmetic_error(solution)
            if corrupted != original:
                cases.append(EvalCase(
                    case_id      = f"math_{i:04d}_error",
                    dataset      = "math",
                    prompt       = question,
                    reasoning    = corrupted,
                    ground_truth = final_answer,
                    model_answer = "WRONG",
                    is_wrong     = True,
                    failure_type = "ARITHMETIC_ERROR",
                    difficulty   = difficulty,
                ))
                if len(cases) >= limit:
                    break

    print(f"  Loaded {len(cases)} MATH cases "
          f"({sum(1 for c in cases if c.is_wrong)} failure, "
          f"{sum(1 for c in cases if not c.is_wrong)} correct)")
    return cases


# ── GPQA Loader (gated — requires HuggingFace account approval) ───────────────

def load_gpqa(limit: int = 60, subset: str = "gpqa_main") -> list[EvalCase]:
    """
    Load GPQA (Graduate-Level Google-Proof Q&A) — 448 expert STEM questions.

    GATED DATASET — requires HuggingFace access:
      1. Visit https://huggingface.co/datasets/Idavidrein/gpqa and request access
      2. Run: huggingface-cli login   (paste your token from hf.co/settings/tokens)
         OR set env var: HF_TOKEN=hf_xxxxxxxxxxxx

    Subsets available:
      gpqa_main     — 448 questions  (recommended)
      gpqa_diamond  — 198 highest-quality questions (hardest, use for paper)
      gpqa_extended — 546 questions

    Why GPQA matters:
      Questions written by domain experts (PhDs) that require genuine multi-step
      reasoning + factual knowledge. Even GPT-4 scores ~39% without tools.
      Failure cases here test FACTUAL_GROUNDING_FAIL + SOCRATIC_CONTRADICTION.

    Each row has:
      Question          — the question text
      Correct Answer    — verified correct answer
      Incorrect Answer 1/2/3 — plausible wrong answers written by the same expert
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Install 'datasets': pip install datasets")
        return []

    print(f"  Loading GPQA ({subset}) — requires HuggingFace login...")
    print(f"  (If this fails: run 'huggingface-cli login' or set HF_TOKEN env var)")
    try:
        # token= pulls from HF_TOKEN env var automatically if set
        ds = load_dataset("Idavidrein/gpqa", subset, split="train")
        print(f"    OK — {len(ds)} examples")
    except Exception as e:
        err = str(e)
        if "gated" in err.lower() or "access" in err.lower() or "403" in err:
            print(f"    ACCESS DENIED — visit https://huggingface.co/datasets/Idavidrein/gpqa")
            print(f"    Click 'Agree and access repository', then run: huggingface-cli login")
        else:
            print(f"    FAILED: {err[:120]}")
        return []

    cases: list[EvalCase] = []
    for i, row in enumerate(ds):
        if len(cases) >= limit:
            break

        question    = (row.get("Question") or "").strip()
        correct_ans = (row.get("Correct Answer") or "").strip()
        subdomain   = (row.get("Subdomain") or row.get("High-level domain") or "science").strip()

        # Collect the three expert-written wrong answers
        wrong_answers = [
            (row.get("Incorrect Answer 1") or "").strip(),
            (row.get("Incorrect Answer 2") or "").strip(),
            (row.get("Incorrect Answer 3") or "").strip(),
        ]
        wrong_answers = [w for w in wrong_answers if w]

        if not question or not correct_ans:
            continue

        # ── CORRECT case: model correctly reasons to the verified answer ──────
        correct_reasoning = (
            f"This is a {subdomain} question that requires careful analysis. "
            f"Considering the established principles in {subdomain}: "
            f"The key insight is that {correct_ans[:200]}. "
            f"This can be verified against known {subdomain} literature. "
            f"Therefore, the answer is: {correct_ans}."
        )
        cases.append(EvalCase(
            case_id      = f"gpqa_{i:04d}_correct",
            dataset      = "gpqa",
            prompt       = question,
            reasoning    = correct_reasoning,
            ground_truth = correct_ans,
            model_answer = correct_ans,
            is_wrong     = False,
            failure_type = "CORRECT",
            difficulty   = "HARD",
        ))

        # ── FAILURE case: model states a plausible but wrong expert answer ────
        # Uses Incorrect Answer 1 — written by the same PhD who wrote the question,
        # so it's a maximally plausible wrong answer (not obviously wrong).
        if wrong_answers and len(cases) % 2 == 0:
            wrong_ans = wrong_answers[0]
            wrong_reasoning = (
                f"Analyzing this {subdomain} problem step by step: "
                f"Based on {subdomain} principles, {wrong_ans[:200]}. "
                f"This follows from the underlying theory in {subdomain}. "
                f"Therefore, the correct answer must be: {wrong_ans}."
            )
            cases.append(EvalCase(
                case_id      = f"gpqa_{i:04d}_wrong",
                dataset      = "gpqa",
                prompt       = question,
                reasoning    = wrong_reasoning,
                ground_truth = correct_ans,
                model_answer = wrong_ans,
                is_wrong     = True,
                failure_type = "FACTUAL_GROUNDING_FAIL",
                difficulty   = "HARD",
            ))
            if len(cases) >= limit:
                break

    print(f"  Loaded {len(cases)} GPQA cases "
          f"({sum(1 for c in cases if c.is_wrong)} failure, "
          f"{sum(1 for c in cases if not c.is_wrong)} correct)")
    return cases


# ── TruthfulQA Loader (replaces gated GPQA) ───────────────────────────────────

def load_truthfulqa(limit: int = 60) -> list[EvalCase]:
    """
    Load TruthfulQA — 817 questions testing factual accuracy (free, no gating).
    Each question has a best correct answer and a list of incorrect answers.

    We use this to test FACTUAL_GROUNDING_FAIL / SOCRATIC_CONTRADICTION detection:
    - CORRECT case: reasoning that states the verified correct answer
    - FAILURE case: reasoning that states one of the known wrong answers confidently

    TruthfulQA is the standard benchmark for LLM factual accuracy and is
    widely cited in the hallucination detection literature.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Install 'datasets': pip install datasets")
        return []

    print("  Loading TruthfulQA from HuggingFace (downloads ~1 MB on first run)...")
    try:
        ds = load_dataset("truthful_qa", "generation", split="validation")
        print(f"    OK — {len(ds)} examples")
    except Exception as e:
        print(f"  [ERROR] Could not load TruthfulQA: {e}")
        return []

    cases: list[EvalCase] = []
    for i, row in enumerate(ds):
        if len(cases) >= limit:
            break

        question     = (row.get("question") or "").strip()
        best_answer  = (row.get("best_answer") or "").strip()
        # incorrect_answers is a list; pick the first plausible wrong answer
        wrong_answers = row.get("incorrect_answers") or []
        wrong_answer  = wrong_answers[0].strip() if wrong_answers else ""

        if not question or not best_answer:
            continue

        # Build a correct reasoning chain around the verified answer
        correct_reasoning = (
            f"Let me think about this carefully. "
            f"Based on verified facts, {best_answer} "
            f"This is well-established and can be confirmed through reliable sources. "
            f"Therefore, the answer is: {best_answer}."
        )

        # CORRECT case
        cases.append(EvalCase(
            case_id      = f"truthfulqa_{i:04d}_correct",
            dataset      = "truthfulqa",
            prompt       = question,
            reasoning    = correct_reasoning,
            ground_truth = best_answer,
            model_answer = best_answer,
            is_wrong     = False,
            failure_type = "CORRECT",
            difficulty   = "MEDIUM",
        ))

        # FAILURE case — confidently states a known wrong answer
        if wrong_answer and len(cases) % 2 == 0:
            wrong_reasoning = (
                f"Based on common understanding, {wrong_answer} "
                f"This is the widely accepted explanation for this question. "
                f"Therefore, the answer is: {wrong_answer}."
            )
            cases.append(EvalCase(
                case_id      = f"truthfulqa_{i:04d}_wrong",
                dataset      = "truthfulqa",
                prompt       = question,
                reasoning    = wrong_reasoning,
                ground_truth = best_answer,
                model_answer = wrong_answer,
                is_wrong     = True,
                failure_type = "FACTUAL_GROUNDING_FAIL",
                difficulty   = "MEDIUM",
            ))
            if len(cases) >= limit:
                break

    print(f"  Loaded {len(cases)} TruthfulQA cases "
          f"({sum(1 for c in cases if c.is_wrong)} failure, "
          f"{sum(1 for c in cases if not c.is_wrong)} correct)")
    return cases


# ── Offline evaluation engine ─────────────────────────────────────────────────

def _stream_save(results: list["EvalResult"], output_path: str, dataset: str,
                  mode: str, limit: int) -> None:
    """Overwrite output file with all results collected so far (called after each case)."""
    try:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": mode, "dataset": dataset, "limit": limit,
            "cases": [
                {
                    "case_id"           : r.case.case_id,
                    "dataset"           : r.case.dataset,
                    "is_wrong"          : r.case.is_wrong,
                    "failure_type"      : r.case.failure_type,
                    "difficulty"        : r.case.difficulty,
                    "predicted_failure" : r.predicted_failure,
                    "predicted_type"    : r.predicted_type,
                    "confidence"        : r.confidence,
                    "latency_ms"        : r.latency_ms,
                    "total_steps"       : r.total_steps,
                    "first_failed_step" : r.first_failed_step,
                    "correct_prediction": r.is_correct_prediction,
                }
                for r in results
            ],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def run_offline_evaluation(
    cases: list[EvalCase],
    use_groq: bool = False,
    batch_size: int = 5,        # cases per Groq batch (5 ≈ 15-25 API calls)
    batch_pause: float = 60.0,  # seconds between batches
    output_path: Optional[str] = None,   # stream-save results here after every case
    resume: bool = True,         # skip case_ids already in output_path
) -> list[EvalResult]:
    """
    Run FIE reasoning verifier directly (no server needed).

    Groq mode:
      • Runs in batches with a pause between them to respect free-tier rate limits.
      • Saves results after EVERY case — interrupt at any time, resume safely.
      • Resume: pass the same output_path and resume=True; already-done cases are skipped.

    Multi-key strategy (recommended for free accounts):
      Run once per dataset, swap GROQ_API_KEY in .env between runs:
        python -m tests.eval_datasets --dataset gsm8k     --groq --output results/groq_gsm8k.json
        python -m tests.eval_datasets --dataset math      --groq --output results/groq_math.json
        python -m tests.eval_datasets --dataset truthfulqa --groq --output results/groq_truth.json
        python -m tests.eval_datasets --dataset gpqa      --groq --output results/groq_gpqa.json
        python -m tests.eval_datasets --combine
    """
    try:
        from engine.reasoning.reasoning_verifier import verify_reasoning
    except ImportError as e:
        print(f"  [ERROR] Cannot import FIE engine: {e}")
        print("  Run with: python -m tests.eval_datasets  (from project root)")
        return []

    mode = "groq" if use_groq else "offline"

    # ── Resume: load already-completed case_ids from output_path ─────────────
    done_ids: set[str] = set()
    results:  list[EvalResult] = []

    if resume and output_path and Path(output_path).exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                saved = json.load(f)
            done_ids = {c["case_id"] for c in saved.get("cases", [])}
            print(f"  Resume: {len(done_ids)} cases already done in {output_path}")
        except Exception:
            done_ids = set()

    remaining = [c for c in cases if c.case_id not in done_ids]

    if not remaining:
        print("  All cases already complete — nothing to run.")
        return results

    if use_groq:
        n_batches = (len(remaining) + batch_size - 1) // batch_size
        est_min   = (len(remaining) * 4 + n_batches * batch_pause) / 60
        print(f"  Groq batch mode  : {len(remaining)} remaining / {len(cases)} total")
        print(f"  Batch size       : {batch_size}  |  Pause: {batch_pause:.0f}s between batches")
        print(f"  Estimated time   : ~{est_min:.0f} min  ({n_batches} batches)")
        print(f"  Tip: swap GROQ_API_KEY in .env and rerun to use a fresh key's quota")
        if output_path:
            print(f"  Streaming saves  : {output_path}  (resume-safe)")

    # ── Fast offline path ─────────────────────────────────────────────────────
    if not use_groq:
        try:
            from tqdm import tqdm
            iterator = tqdm(remaining, total=len(remaining), desc="  Evaluating (offline)")
        except ImportError:
            iterator = remaining

        for case in iterator:
            t0 = time.time()
            try:
                rr = verify_reasoning(
                    question=case.prompt, primary_answer=case.reasoning,
                    shadow_outputs=[], use_groq=False,
                )
                elapsed_ms = (time.time() - t0) * 1000
                results.append(EvalResult(
                    case=case, predicted_failure=rr.failure_detected,
                    predicted_type=rr.failure_type if rr.failure_detected else "CORRECT",
                    confidence=rr.confidence, latency_ms=elapsed_ms,
                    total_steps=rr.total_steps, first_failed_step=rr.first_failed_step,
                    pipeline_trace=rr.pipeline_trace,
                ))
            except Exception as exc:
                results.append(EvalResult(
                    case=case, predicted_failure=False, predicted_type="ERROR",
                    confidence=0.0, latency_ms=(time.time()-t0)*1000,
                    total_steps=0, first_failed_step=None, pipeline_trace=[f"ERROR: {exc}"],
                ))
        if output_path:
            _stream_save(results, output_path, cases[0].dataset if cases else "", mode, len(cases))
        return results

    # ── Groq batch path ───────────────────────────────────────────────────────
    for batch_num, batch_start in enumerate(range(0, len(remaining), batch_size), 1):
        batch = remaining[batch_start: batch_start + batch_size]

        print(f"\n  Batch {batch_num}/{n_batches}  "
              f"(cases {batch_start+1}–{batch_start+len(batch)} of {len(remaining)})...")

        for j, case in enumerate(batch):
            if j > 0:
                time.sleep(2)  # brief gap within batch
            t0 = time.time()
            try:
                rr = verify_reasoning(
                    question=case.prompt, primary_answer=case.reasoning,
                    shadow_outputs=[], use_groq=True,
                )
                elapsed_ms = (time.time() - t0) * 1000
                results.append(EvalResult(
                    case=case, predicted_failure=rr.failure_detected,
                    predicted_type=rr.failure_type if rr.failure_detected else "CORRECT",
                    confidence=rr.confidence, latency_ms=elapsed_ms,
                    total_steps=rr.total_steps, first_failed_step=rr.first_failed_step,
                    pipeline_trace=rr.pipeline_trace,
                ))
                status = "FAIL" if rr.failure_detected else "OK  "
                print(f"    [{batch_start+j+1:3d}] {case.dataset.upper():10s} {status} "
                      f"conf={rr.confidence:.2f} steps={rr.total_steps} ({elapsed_ms:.0f}ms)")
            except Exception as exc:
                results.append(EvalResult(
                    case=case, predicted_failure=False, predicted_type="ERROR",
                    confidence=0.0, latency_ms=(time.time()-t0)*1000,
                    total_steps=0, first_failed_step=None, pipeline_trace=[f"ERROR: {exc}"],
                ))
                print(f"    [{batch_start+j+1:3d}] ERROR: {str(exc)[:80]}")

            # Stream-save after EVERY case so interrupts lose nothing
            if output_path:
                _stream_save(results, output_path,
                             cases[0].dataset if cases else "", mode, len(cases))

        # Pause between batches (skip after last)
        if batch_start + batch_size < len(remaining):
            print(f"  Batch {batch_num} done — pausing {batch_pause:.0f}s "
                  f"(rate-limit reset)...")
            time.sleep(batch_pause)

    print(f"\n  Done: {len(results)} cases processed.")
    return results


def combine_result_files(
    input_paths: list[str],
    output_path: str = "results/groq_combined.json",
) -> dict:
    """
    Merge multiple per-dataset result JSON files into one combined file.
    Deduplicates by case_id. Recomputes metrics from merged cases.

    Usage:
        python -m tests.eval_datasets --combine
    """
    all_cases_raw: list[dict] = []
    seen_ids: set[str] = set()

    for path in input_paths:
        if not Path(path).exists():
            print(f"  [SKIP] not found: {path}")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            cases = data.get("cases", [])
            before = len(all_cases_raw)
            for c in cases:
                if c["case_id"] not in seen_ids:
                    all_cases_raw.append(c)
                    seen_ids.add(c["case_id"])
            added = len(all_cases_raw) - before
            print(f"  {path}: {added} cases added ({len(cases)-added} duplicates skipped)")
        except Exception as e:
            print(f"  [ERROR] {path}: {e}")

    if not all_cases_raw:
        print("  No cases found — nothing combined.")
        return {}

    # Recompute metrics from raw dicts
    tp = sum(1 for c in all_cases_raw if c["is_wrong"]     and c["predicted_failure"])
    fp = sum(1 for c in all_cases_raw if not c["is_wrong"] and c["predicted_failure"])
    tn = sum(1 for c in all_cases_raw if not c["is_wrong"] and not c["predicted_failure"])
    fn = sum(1 for c in all_cases_raw if c["is_wrong"]     and not c["predicted_failure"])
    n  = len(all_cases_raw)

    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    fpr  = fp/(fp+tn) if (fp+tn) > 0 else 0.0
    acc  = (tp+tn)/n if n > 0 else 0.0

    # Per-dataset breakdown
    ds_names = sorted(set(c["dataset"] for c in all_cases_raw))
    per_dataset = {}
    for ds in ds_names:
        dc = [c for c in all_cases_raw if c["dataset"] == ds]
        d_tp = sum(1 for c in dc if c["is_wrong"] and c["predicted_failure"])
        d_fp = sum(1 for c in dc if not c["is_wrong"] and c["predicted_failure"])
        d_tn = sum(1 for c in dc if not c["is_wrong"] and not c["predicted_failure"])
        d_fn = sum(1 for c in dc if c["is_wrong"] and not c["predicted_failure"])
        d_p  = d_tp/(d_tp+d_fp) if (d_tp+d_fp) > 0 else 0.0
        d_r  = d_tp/(d_tp+d_fn) if (d_tp+d_fn) > 0 else 0.0
        d_f1 = 2*d_p*d_r/(d_p+d_r) if (d_p+d_r) > 0 else 0.0
        d_fpr= d_fp/(d_fp+d_tn) if (d_fp+d_tn) > 0 else 0.0
        per_dataset[ds] = {
            "total": len(dc), "tp": d_tp, "fp": d_fp, "tn": d_tn, "fn": d_fn,
            "precision": round(d_p, 4), "recall": round(d_r, 4),
            "f1": round(d_f1, 4), "fpr": round(d_fpr, 4),
        }

    combined = {
        "timestamp"  : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode"       : "groq_combined",
        "total_cases": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision"  : round(prec, 4),
        "recall"     : round(rec, 4),
        "f1"         : round(f1, 4),
        "fpr"        : round(fpr, 4),
        "accuracy"   : round(acc, 4),
        "per_dataset": per_dataset,
        "cases"      : all_cases_raw,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Combined: {n} cases from {len(ds_names)} datasets")
    print(f"  Precision={prec*100:.1f}%  Recall={rec*100:.1f}%  F1={f1*100:.1f}%  FPR={fpr*100:.1f}%")
    print(f"  Saved: {output_path}")
    return combined


# ── Live server evaluation ────────────────────────────────────────────────────

def run_live_evaluation(cases: list[EvalCase], base_url: str) -> list[EvalResult]:
    """POST each case to /api/v1/monitor and read reasoning_verification."""
    try:
        import requests
    except ImportError:
        print("  [ERROR] Install 'requests': pip install requests")
        return []

    url     = f"{base_url.rstrip('/')}/api/v1/monitor"
    results = []

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(cases, 1), total=len(cases), desc="  Live evaluation")
    except ImportError:
        iterator = enumerate(cases, 1)

    for i, case in iterator:
        payload = {
            "prompt":             case.prompt,
            "primary_output":     case.reasoning,
            "primary_model_name": f"eval-{case.dataset}",
            "run_full_jury":      False,   # faster — skip full jury for bulk eval
        }
        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=60)
            elapsed_ms = (time.time() - t0) * 1000

            if resp.status_code != 200:
                results.append(EvalResult(
                    case=case, predicted_failure=False, predicted_type="HTTP_ERROR",
                    confidence=0.0, latency_ms=elapsed_ms, total_steps=0, first_failed_step=None,
                    pipeline_trace=[f"HTTP {resp.status_code}: {resp.text[:100]}"],
                ))
                continue

            data = resp.json()
            rv   = data.get("reasoning_verification") or {}
            results.append(EvalResult(
                case               = case,
                predicted_failure  = rv.get("failure_detected", False),
                predicted_type     = rv.get("failure_type", "CORRECT") if rv.get("failure_detected") else "CORRECT",
                confidence         = rv.get("confidence", 0.0),
                latency_ms         = elapsed_ms,
                total_steps        = rv.get("total_steps", 0),
                first_failed_step  = rv.get("first_failed_step"),
                pipeline_trace     = rv.get("pipeline_trace", []),
            ))
        except Exception as exc:
            elapsed_ms = (time.time() - t0) * 1000
            results.append(EvalResult(
                case=case, predicted_failure=False, predicted_type="EXCEPTION",
                confidence=0.0, latency_ms=elapsed_ms, total_steps=0, first_failed_step=None,
                pipeline_trace=[f"EXCEPTION: {exc}"],
            ))

    return results


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(results: list[EvalResult], dataset_name: str = "") -> dict:
    """Compute Precision / Recall / F1 / FPR overall and per failure type."""
    tp = sum(1 for r in results if r.case.is_wrong     and r.predicted_failure)
    fp = sum(1 for r in results if not r.case.is_wrong and r.predicted_failure)
    tn = sum(1 for r in results if not r.case.is_wrong and not r.predicted_failure)
    fn = sum(1 for r in results if r.case.is_wrong     and not r.predicted_failure)

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy   = (tp + tn) / len(results) if results else 0.0
    avg_lat    = sum(r.latency_ms for r in results) / len(results) if results else 0.0

    # Per-type breakdown
    failure_types = sorted(set(c.case.failure_type for c in results if c.case.is_wrong))
    per_type: dict[str, dict] = {}
    for ft in failure_types:
        ft_cases = [r for r in results if r.case.failure_type == ft]
        ft_tp    = sum(1 for r in ft_cases if r.predicted_failure)
        ft_total = len(ft_cases)
        per_type[ft] = {
            "recall"   : ft_tp / ft_total if ft_total else 0.0,
            "caught"   : ft_tp,
            "total"    : ft_total,
        }

    # Difficulty breakdown (for MATH)
    difficulty_break: dict[str, dict] = {}
    for diff in ("EASY", "MEDIUM", "HARD"):
        diff_cases = [r for r in results if r.case.difficulty == diff and r.case.is_wrong]
        if diff_cases:
            d_tp = sum(1 for r in diff_cases if r.predicted_failure)
            difficulty_break[diff] = {"recall": d_tp / len(diff_cases), "caught": d_tp, "total": len(diff_cases)}

    return {
        "dataset"           : dataset_name,
        "total_cases"       : len(results),
        "failure_cases"     : tp + fn,
        "correct_cases"     : tn + fp,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision"         : round(precision, 4),
        "recall"            : round(recall, 4),
        "f1"                : round(f1, 4),
        "fpr"               : round(fpr, 4),
        "accuracy"          : round(accuracy, 4),
        "avg_latency_ms"    : round(avg_lat, 1),
        "per_type_recall"   : per_type,
        "difficulty_recall" : difficulty_break,
    }


def print_metrics(m: dict) -> None:
    ds = m["dataset"] or "Combined"
    print(f"\n{'='*72}")
    print(f"EVALUATION RESULTS — {ds.upper()}")
    print(f"{'='*72}")
    print(f"  Total cases          : {m['total_cases']}")
    print(f"  Failure cases        : {m['failure_cases']}")
    print(f"  Correct cases        : {m['correct_cases']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print()
    print(f"  Precision            : {m['precision']:.4f}  ({m['precision']*100:.1f}%)")
    print(f"  Recall               : {m['recall']:.4f}  ({m['recall']*100:.1f}%)")
    print(f"  F1 Score             : {m['f1']:.4f}  ({m['f1']*100:.1f}%)")
    print(f"  False Positive Rate  : {m['fpr']:.4f}  ({m['fpr']*100:.1f}%)")
    print(f"  Accuracy             : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Avg Latency          : {m['avg_latency_ms']:.0f} ms")
    print()

    if m["per_type_recall"]:
        print(f"  Per-Type Recall:")
        for ft, stats in m["per_type_recall"].items():
            bar = "#" * int(stats["recall"] * 20)
            pad = " " * (20 - len(bar))
            print(f"    {ft:<35} {stats['caught']}/{stats['total']}  [{bar}{pad}]  {stats['recall']*100:.0f}%")

    if m["difficulty_recall"]:
        print(f"\n  By Difficulty (failure recall):")
        for diff, stats in m["difficulty_recall"].items():
            bar = "#" * int(stats["recall"] * 20)
            pad = " " * (20 - len(bar))
            print(f"    {diff:<10} {stats['caught']}/{stats['total']}  [{bar}{pad}]  {stats['recall']*100:.0f}%")

    print(f"{'='*72}")


# ── Download-only mode ────────────────────────────────────────────────────────

def download_all_datasets() -> None:
    """Pre-download all datasets to HuggingFace cache."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install 'datasets' first: pip install datasets")
        sys.exit(1)

    # (repo_id, config_or_None, split)
    configs = [
        ("openai/gsm8k",             "main",       "test"),
        ("HuggingFaceH4/MATH-500",   None,         "test"),
        ("EleutherAI/hendrycks_math", "algebra",   "test"),   # fallback for MATH
        ("truthful_qa",              "generation", "validation"),
        ("Idavidrein/gpqa",          "gpqa_main",  "train"),  # gated — needs HF login
        ("Idavidrein/gpqa",          "gpqa_diamond","train"), # gated — 198 hardest questions
    ]

    for repo_id, config, split in configs:
        label = f"{repo_id}" + (f" ({config})" if config else "")
        print(f"  Downloading {label}...")
        try:
            ds = (
                load_dataset(repo_id, config, split=split)
                if config else
                load_dataset(repo_id, split=split)
            )
            print(f"    OK — {len(ds)} examples")
        except Exception as e:
            print(f"    FAILED: {str(e)[:120]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FIE Reasoning Dataset Evaluation")
    parser.add_argument("--dataset",        choices=["gsm8k", "math", "truthfulqa", "gpqa", "all"], default="gsm8k")
    parser.add_argument("--limit",          type=int,   default=50,   help="Max cases per dataset")
    parser.add_argument("--offline",        action="store_true", help="Offline heuristic mode (no Groq)")
    parser.add_argument("--groq",           action="store_true", help="Use Groq API for reasoning decomposition")
    parser.add_argument("--live",           action="store_true", help="POST to live server instead of direct call")
    parser.add_argument("--url",            default="http://localhost:8000", help="Live server URL")
    parser.add_argument("--download-only",  action="store_true", help="Just download datasets and exit")
    parser.add_argument("--output",         help="Stream-save results to this file (per-case, resumable)")
    parser.add_argument("--resume",         action="store_true", default=True,
                        help="Skip case_ids already in --output file (default: True)")
    parser.add_argument("--no-resume",      action="store_true", help="Disable resume — re-run all cases")
    parser.add_argument("--batch-size",     type=int,   default=5,
                        help="Cases per Groq batch before pausing (default: 5)")
    parser.add_argument("--batch-pause",    type=float, default=62.0,
                        help="Seconds to pause between batches (default: 62)")
    parser.add_argument("--combine",        action="store_true",
                        help="Merge all results/groq_*.json files into a combined report and exit")
    parser.add_argument("--combine-input",  nargs="+",
                        help="Explicit list of JSON files to combine (default: results/groq_*.json)")
    parser.add_argument("--combine-output", default="results/groq_combined.json",
                        help="Output path for combined results (default: results/groq_combined.json)")
    parser.add_argument("--verbose",        action="store_true", help="Print per-case results")
    args = parser.parse_args()

    # ── Combine mode ─────────────────────────────────────────────────────────────
    if args.combine:
        import glob as _glob
        if args.combine_input:
            input_files = args.combine_input
        else:
            input_files = sorted(_glob.glob("results/groq_*.json"))
            # Exclude the combined output itself
            combine_out = str(Path(args.combine_output).resolve())
            input_files = [f for f in input_files if str(Path(f).resolve()) != combine_out]

        if not input_files:
            print("No groq result files found in results/. Run per-dataset evaluations first.")
            sys.exit(1)

        print(f"Combining {len(input_files)} result files:")
        for f in input_files:
            print(f"  {f}")

        combined = combine_result_files(input_files, output_path=args.combine_output)
        print(f"\nCombined results saved to: {args.combine_output}")

        # Print per-dataset breakdown
        print(f"\n{'='*72}")
        print("PER-DATASET BREAKDOWN")
        print(f"{'='*72}")
        for ds, dm in combined.get("per_dataset", {}).items():
            print(f"  {ds.upper():<12}  "
                  f"P={dm['precision']*100:.1f}%  R={dm['recall']*100:.1f}%  "
                  f"F1={dm['f1']*100:.1f}%  FPR={dm['fpr']*100:.1f}%  "
                  f"({dm['tp']+dm['fn']} failures / {dm['total']} total)")
        print(f"{'='*72}")
        sys.exit(0)

    # ── Download-only mode ────────────────────────────────────────────────────────
    if args.download_only:
        print("Downloading all benchmark datasets...")
        download_all_datasets()
        print("Done. Datasets cached to ~/.cache/huggingface/datasets/")
        return

    use_groq = args.groq and not args.offline
    do_resume = args.resume and not args.no_resume

    # ── Determine output path ─────────────────────────────────────────────────────
    output_path = args.output
    if not output_path and use_groq:
        Path("results").mkdir(exist_ok=True)
        output_path = f"results/groq_{args.dataset}.json"

    # ── Load requested datasets ───────────────────────────────────────────────────
    # When running all datasets together, divide limit so total stays manageable.
    # When running a single dataset, use the full limit.
    is_all = args.dataset == "all"
    all_cases: list[EvalCase] = []
    if args.dataset in ("gsm8k", "all"):
        all_cases += load_gsm8k(limit=args.limit)
    if args.dataset in ("math", "all"):
        all_cases += load_math(limit=args.limit // 2 if is_all else args.limit)
    if args.dataset in ("truthfulqa", "all"):
        all_cases += load_truthfulqa(limit=args.limit // 3 if is_all else args.limit)
    if args.dataset in ("gpqa", "all"):
        all_cases += load_gpqa(limit=args.limit // 4 if is_all else args.limit)

    if not all_cases:
        print("No cases loaded — check dataset availability.")
        sys.exit(1)

    print(f"\nTotal : {len(all_cases)} cases loaded")
    print(f"Mode  : {'LIVE SERVER @ ' + args.url if args.live else ('GROQ + heuristic' if use_groq else 'offline heuristic')}")
    if output_path:
        print(f"Output: {output_path}  (resume={'ON' if do_resume else 'OFF'})")
    if use_groq:
        print(f"Batch : {args.batch_size} cases / {args.batch_pause}s pause")

    # ── Run evaluation ────────────────────────────────────────────────────────────
    if args.live:
        results = run_live_evaluation(all_cases, args.url)
    else:
        results = run_offline_evaluation(
            all_cases,
            use_groq    = use_groq,
            batch_size  = args.batch_size,
            batch_pause = args.batch_pause,
            output_path = output_path,
            resume      = do_resume,
        )

    # ── Verbose per-case log ──────────────────────────────────────────────────────
    if args.verbose:
        print(f"\n{'='*72}")
        print("PER-CASE RESULTS")
        print(f"{'='*72}")
        for r in results:
            status = "PASS" if r.is_correct_prediction else "FAIL"
            label  = r.case.failure_type
            pred   = r.predicted_type
            print(f"  [{status}] {r.case.case_id[:50]}")
            print(f"    label={label} predicted={pred} conf={r.confidence:.2f} steps={r.total_steps} [{r.latency_ms:.0f}ms]")
            if not r.is_correct_prediction and r.pipeline_trace:
                print(f"    trace: {r.pipeline_trace[-1][:100]}")

    # ── Metrics per dataset + combined ───────────────────────────────────────────
    dataset_names = sorted(set(r.case.dataset for r in results))
    all_metrics = []

    for ds_name in dataset_names:
        ds_results = [r for r in results if r.case.dataset == ds_name]
        m = compute_metrics(ds_results, ds_name)
        print_metrics(m)
        all_metrics.append(m)

    if len(dataset_names) > 1:
        combined_m = compute_metrics(results, "COMBINED")
        print_metrics(combined_m)
        all_metrics.append(combined_m)

    # ── Save final JSON (non-Groq runs, or when no output_path was set) ───────────
    if not output_path:
        Path("results").mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        mode_tag = "live" if args.live else ("groq" if use_groq else "offline")
        output_path = f"results/eval_{args.dataset}_{mode_tag}_{ts}.json"

    if not use_groq:
        # Groq runs already stream-saved per-case; only write final for offline/live
        try:
            payload = {
                "timestamp"  : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode"       : "live" if args.live else "offline",
                "dataset"    : args.dataset,
                "limit"      : args.limit,
                "metrics"    : all_metrics,
                "cases"      : [
                    {
                        "case_id"           : r.case.case_id,
                        "dataset"           : r.case.dataset,
                        "is_wrong"          : r.case.is_wrong,
                        "failure_type"      : r.case.failure_type,
                        "difficulty"        : r.case.difficulty,
                        "predicted_failure" : r.predicted_failure,
                        "predicted_type"    : r.predicted_type,
                        "confidence"        : r.confidence,
                        "latency_ms"        : r.latency_ms,
                        "total_steps"       : r.total_steps,
                        "first_failed_step" : r.first_failed_step,
                        "correct_prediction": r.is_correct_prediction,
                    }
                    for r in results
                ],
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        except Exception as e:
            print(f"\nCould not save results: {e}")
    else:
        print(f"\nStream-saved results in: {output_path}")

    # ── Exit code: 0 if recall >= 30% ────────────────────────────────────────────
    final_metrics = all_metrics[-1] if all_metrics else {}
    recall = final_metrics.get("recall", 0.0)
    sys.exit(0 if recall >= 0.30 else 1)


if __name__ == "__main__":
    main()
