# EXECUTION_001 — Evaluation Harness and Pinned Baseline (WP-001)

| | |
| --- | --- |
| Work package | WP-001 |
| Status | **IN PROGRESS** |
| Branch | `rebuild/wp-001-eval-harness` (local only, not pushed) |
| Started | 2026-10-08 |
| Contract | [PLAN_001_EVAL_HARNESS.md](../IMPLEMENTATION_PLANS/PLAN_001_EVAL_HARNESS.md) |

# Objective

Build a measurement layer that answers, for every future change to FIE: what happened
compared with the exact previous version? Pin a first baseline for the shipped model.
The harness measures. It serves no request and changes no production file.

# Approved Plan

[PLAN_001_EVAL_HARNESS.md](../IMPLEMENTATION_PLANS/PLAN_001_EVAL_HARNESS.md), approved by the
owner on 2026-10-08 with these decisions.

| # | Approved decision |
| --- | --- |
| OD-1 | Canonical baseline is PAIR v6.3b. PAIR v6.2 is a reference reproduction |
| OD-2 | Profile deviations V1–V10 approved. `sdk-offline-failsecure` is described everywhere as the **canonical reproducibility/evaluation profile**, never as the production runtime. Reports distinguish (A) shipped/default behaviour, (B) the canonical reproducibility profile, (C) the lite profile, (D) stability/unseeded behaviour |
| OD-3 | Risk fixtures and live-service security findings stay on the local, un-pushed branch until WP-002 closes the gaps |
| OD-4 | Full per-prompt deterministic baseline records are committed |
| OD-5 | Differences in evidence files are reported, not fatal, in WP-001 |
| OD-6 | Observed pickle/joblib classes are recorded. No allowlist is enforced |
| OD-7 | No packaging change |
| OD-8 | A fresh environment is built from the declared pins and tested. It does not replace the canonical environment unless counts reproduce and it is demonstrably comparable |
| OD-9 | No CI workflow change |
| OD-10 | HarmBench stays at its frozen 387 rows; 380 unique is reported beside it |
| OD-11 | AdvBench is a labelled case study, never in the headline macro |
| OD-12 | Rebuild documentation is tracked on the local branch. The branch is not pushed |
| OD-13 | Tests live in `tests/evals/` |

Allowed paths: `evals/`, `tests/evals/`, `.gitignore`, approved files under `docs/fie_rebuild_2026/`.

# Pre-State

Recorded 2026-10-08, before any change.

| Item | Value |
| --- | --- |
| Branch before | `main` |
| HEAD | `24cb2e9d76e977364f728f47790935283ef110b7` |
| `git status` | Clean |
| Work branch | `rebuild/wp-001-eval-harness`, created from `24cb2e9` |
| Remote | `origin` = GitHub. Nothing pushed |
| OS | Windows 11 (`Windows-10-10.0.26200-SP0`) |
| Python | 3.10.19, conda env `failure-engine` |
| Packages | scikit-learn 1.7.2, onnxruntime 1.23.2, tokenizers 0.22.2, numpy 2.2.6, xgboost 3.2.0, joblib 1.5.3, pandas 2.3.3, requests 2.32.5, langdetect 1.0.9, deep-translator 1.11.4, pytest 9.0.2 |
| Drift from `requirements.txt` | numpy (pin 2.1.3), xgboost (2.1.4), joblib (1.4.2), pandas (2.2.3), requests (2.32.3) |

Model files, all matching `scripts/model_manifest.json`:

| Role | File | SHA-256 |
| --- | --- | --- |
| PAIR classifier (shipped) | `fie/models/pair_intent_classifier_v6_3b.pkl` | `9c682b285a9fa20519da0c764c34b5ed49702c663f64ba8e39c8c1f728702514` |
| PAIR metadata | `fie/models/pair_intent_meta_v6_3b.json` | `5babb7ec71c8a6e809379a3290c0c16fb0d35d696af0f17d2dd27ccc0b35c27f` |
| Meta-classifier | `fie/models/meta_clf.pkl` | `be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6` |
| Meta-classifier metadata | `fie/models/meta_clf.json` | `989091d0b2296c837d1a3bc3ac6857600fe15439e68a7ecd96de1d6ca8ec76ce` |
| Encoder | `fie/models/minilm-onnx/model.onnx` | `57eb46cc82cd048d1986b0a4c30d50e4a87fc06d15773075b74441748f8ed2ea` |
| Tokenizer | `fie/models/minilm-onnx/tokenizer.json` | `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0` |
| PAIR classifier (reference v6.2) | `fie/models/pair_intent_classifier_v6.pkl` | `25c1a421b03ff493b72452cc83a67092951140cec1ccc17d5aae64b3dd00619e` |
| PAIR metadata (reference v6.2) | `fie/models/pair_intent_meta_v6.json` | `b83c0e7c56772146eae539714e0288d965f5182f17574e0e13f9c47e8e9e2d9d` |

Test baseline: `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA`,
run with every `.env` key overridden. **87 passed, 0 failed, 0 skipped.** Per-test outcomes
were saved for a test-by-test comparison at the end.

# Step-by-Step Implementation

_Filled in as each step completes._

# Files Created

# Files Modified

# Tests

# Acceptance Checks

# Baseline Counts

# Reference Counts

# Determinism Results

# Hermeticity Results

# Model Integrity Results

# Dataset Integrity Results

# Latency

# Fresh Environment Results

# Errors Encountered

# Root Causes

# Fixes

# Security Findings

# Scope Verification

# Known Limitations

# Deviations

# Final Status

# Next Recommended Work
