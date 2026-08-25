# FIE — Verified Fact Sheet

**Generated 2026-08-13.** Every number below was read from a source file or the
running pipeline on that date. The source is named on each line so any claim can
be re-checked in one command.

**Rule: if a number is not in this file, do not put it in a CV, email, paper or
interview answer.** Regenerate this sheet after any retrain or re-measurement.

---

## 1. Identity

| Fact | Value | Source |
| --- | --- | --- |
| Package | `pip install fie-sdk` | `pyproject.toml` → `name` |
| Parameters | **~23M** (22.6M) | `fie/models/minilm-onnx/model.onnx`, 86.1 MB fp32 |
| Runtime | Offline, CPU, no API call, no GPU | `fie/onnx_encoder.py` |
| Embedder | MiniLM-L6-v2, exported to ONNX | `scripts/model_manifest.json` |

## 2. Layer count — **12**, not 11

Verified live, not from documentation:

```bash
python -c "import sys;sys.path.insert(0,'.');import logging;logging.disable(logging.CRITICAL);\
from fie.adversarial import scan_prompt,warmup;warmup();\
print(len(scan_prompt('hi',use_llama_guard=False).layer_scores))"
# -> 12
```

The twelve: `regex`, `prompt_guard`, `pair_classifier`, `perplexity_proxy`,
`gcg_suffix`, `many_shot`, `indirect_injection`, `copyright`, `direct_harm`,
`fiction_harm`, `virtualization`, `multilingual`.

**Where "11" is still correct:** the phrase *"the other 11 layers"* means
12 minus PAIR. That is right and should not be changed
(`docs/RESEARCH_LOG.md`, `scripts/eval_harmaug.py`, `paper/`).

**Where "11" means the old architecture:** `docs/arxiv_paper/main.tex`
(June 2026) genuinely ran on 11 layers and draws 11 boxes. It carries a
historical note and must **not** be renumbered.

Crescendo / multi-turn escalation is handled by `fie/session_tracker.py` at
**session level** — it is *not* one of the 12 single-prompt layers.

## 3. Latency — **32.5 ms mean, 41.8 ms p95**

Source: `data/ablation/latency_study.json`.

| | mean | p95 |
| --- | --- | --- |
| Full pipeline (12 layers) | **32.5 ms** | **41.8 ms** |
| Lean (6 layers pruned) | 27.0 ms | 35.5 ms |

Pruning 6 layers changed neither recall nor FPR (both 0.8571 / 0.0784) for a
1.2× speed-up — evidence those layers contribute little.

> Superseded claims now corrected in-repo: README said ~22–25 ms, the landing
> page said `<15 ms`. Both were wrong. Use **~33 ms**.

## 4. Detection

Source: `data/benchmark_audit/combined_recall_report.json` (FIE, PAIR v6.2).

| Benchmark | Recall | n |
| --- | --- | --- |
| JailbreakBench | 96.3% | 134 |
| HarmBench | 81.9% | 387 |
| StrongREJECT | 89.7% | 242 |
| SORRY-Bench | 75.2% | 387 |
| **Macro (4 benchmarks)** | **85.8%** [83.7–87.6] | 4 |
| Micro (pooled) | 83.0% [80.9–85.0] | 1150 |
| AdvBench (case study, reported separately) | 95.2% | 168 |

Calibration: **ECE 0.06**.

## 5. Benchmark leakage — the headline finding

- **148 of JailbreakBench's 282 prompts (52.5%)** were already in training data.
- Also 13 HarmBench prompts.
- Method (`scripts/decontaminate_training.py`): (1) lowercase, collapse
  whitespace, SHA-1 hash, compare for exact matches; (2) MiniLM embeddings,
  cosine ≥ **0.95** counts as a duplicate.
- 0.95 chosen because lower cutoffs deleted genuine training data that was only
  topically similar.
- val/test split out *before* training, matches removed from the training side,
  zero overlap verified afterwards.

## 6. Out-of-domain false positives

Source: `data/fpr_baseline.json` (v5) vs `data/fpr_v6.json` (v6), n=340.

| | v5 | v6 |
| --- | --- | --- |
| Overall OOD FPR | 41.5% | **9.7%** |
| Medical | 71.3% | **0%** |

Cause: benign training data came almost entirely from Alpaca (short, casual
instructions), so the model had never seen a normal medical or legal question.
Fixed by rebuilding the benign side to be domain-balanced. **No architecture
change.**

## 7. Over-refusal — the published weakness

Source: `data/overrefusal/overrefusal_report.json`.

| Benchmark | Rate | n |
| --- | --- | --- |
| XSTest safe | **53.6%** | 250 |
| OR-Bench-hard | **90.4%** | 250 |
| Pooled | **72.0%** (360/500) | 500 |

That pooled 72% is **~7×** the 9.7% in-domain FPR previously reported
(72 ÷ 9.7 = 7.4). Do not say "8×".

## 8. Comparison: `gpt-oss-safeguard-20b`

Source: `data/baselines/guard_baselines.json`.

| | FIE (23M, offline) | 20B guard (API) |
| --- | --- | --- |
| Macro recall | 85.8% | **94.3%** |
| XSTest over-refusal | 53.6% | **11%** |
| OR-Bench-hard over-refusal | 90.4% | **80%** |

**State both honestly.** The 20B beats FIE on detection *and* on XSTest — by
roughly 5×. The real finding is that **even a 20B guard fails OR-Bench-hard at
80%**.

⚠️ **Caveat:** the 20B was measured on **100-prompt subsets**; FIE on full sets
(up to n=387). Not strictly like-for-like.

## 9. Augmentation experiments (E28 / E29)

Paired bootstrap, 10,000 resamples, seed 42, fixed threshold 0.50,
PAIR-isolated. Source: `data/benchmark_audit/benignaug_*.json`.

**E28 — HarmAug reproduction (vs baseline):**

| Set | Δ | p |
| --- | --- | --- |
| HarmBench | +4.9 | <0.001 |
| StrongREJECT | +5.4 | <0.001 |
| XSTest | **−4.0** (better) | 0.041 |
| OR-Bench-hard | **+5.2** (worse) | <0.001 |

The two over-refusal benchmarks moved in **opposite directions**.

**E29 — BenignAug (vs HarmAug):**

| Set | Δ | p |
| --- | --- | --- |
| OR-Bench-hard | **−7.2** (better) | <0.001 |
| StrongREJECT | −6.6 (worse) | <0.001 |
| XSTest | +1.2 (no change) | 0.640 |

Conclusion: the two augmentations are opposite directions on one trade-off
curve, not gains that stack.

⚠️ Under-powered: 647 benign rows vs 1,878 harmful. Direction established,
balance point not.

## 10. Architecture self-criticism

- Ablation: most detection comes from the PAIR semantic classifier; several
  layers contribute little. Single point of failure.
- The meta-classifier fired on **83%** of prompts and changed **zero** verdicts
  (E27, p=1.00 on recall/precision/F1/over-refusal).

## 11. Known gaps

Multilingual attacks (training data is mostly English) · encoded/obfuscated
payloads such as base64 · attack families that post-date training.

## 12. Defending the detector itself

User text never enters an instruction channel. The detector **classifies** text
as data; it does not read and follow it. So "ignore your instructions" is just
more text to score.

---

## Regenerate

```bash
python -c "import sys;sys.path.insert(0,'.');import logging;logging.disable(logging.CRITICAL);\
from fie.adversarial import scan_prompt,warmup;warmup();\
print(sorted(scan_prompt('hi',use_llama_guard=False).layer_scores))"   # layer list
python scripts/latency_study.py            # latency
python scripts/measure_combined_recall.py  # macro recall
python scripts/measure_overrefusal.py      # over-refusal
python scripts/measure_guard_baselines.py  # 20B comparison
```
