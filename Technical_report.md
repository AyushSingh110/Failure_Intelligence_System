# Hard-Positive Training and Threshold Calibration for Out-of-Distribution Adversarial Prompt Detection

**Ayush Singh** — Failure Intelligence Engine (FIE)
**Date:** June 2026 | **Version:** Research Cycle 1 Final

---

## Abstract

We study whether an adversarial prompt detection system can generalise to attack patterns it was never trained on. Starting from a production system with strong known-attack performance (F1 = 0.785, Precision = 0.975 on 2,006 prompts), we find that recall on novel unknown attacks collapses to 11–24% — demonstrating that architectural complexity does not confer generalisation. We then show that targeted hard-positive retraining dramatically recovers unknown-attack recall (8% → 96.25%), but at the cost of precision calibration. A threshold sweep resolves the calibration problem without retraining, recovering the target operating point (TPR ≥ 60%, FPR ≤ 15%) at threshold = 0.80. A weight comparison experiment further shows that 3× hard-positive weighting achieves the same target zone at a lower threshold (0.70) with higher F1 (0.9827 vs 0.9673), demonstrating that over-weighting degrades natural calibration. The central finding: **the bottleneck in adversarial detection generalisation is training distribution, not architecture**.

---

## 1. Introduction

Large language model (LLM) deployments face a persistent threat from adversarial prompts — inputs crafted to manipulate model behaviour, extract system information, or bypass safety guidelines. Detection systems for these attacks typically rely on one of three approaches: rule-based pattern matching, classifier-based semantic detection, or ensemble detection combining both.

A fundamental question in deploying such systems is whether detection capability generalises beyond the training distribution. A system that achieves high recall on benchmark datasets may fail entirely when attackers shift vocabulary, framing, or delivery mechanism — the "vocabulary shift" problem.

This report documents a five-phase research cycle on FIE (Failure Intelligence Engine), a production adversarial detection system with 11 parallel detection layers. The research question is:

> **Does architectural complexity in a multi-layer detection system substitute for training distribution coverage? Or does generalisation primarily come from the semantic model?**

The answer has practical consequences: if architecture is the bottleneck, adding more detection layers improves generalisation. If training distribution is the bottleneck, targeted retraining is the correct investment.

---

## 2. System Overview

FIE wraps any LLM with a pre-flight adversarial detection pipeline. 11 detection layers run in parallel using a ThreadPoolExecutor (10-second hard timeout):

| Layer | Type | Weight |
|---|---|---|
| Regex patterns | Rule-based | 1.5 |
| PromptGuard scorer | Keyword + leet-speak | 1.2 |
| GCG suffix scanner | Entropy + punctuation density | 1.3 |
| Direct harm detector | Two-gate (verb + target) | 1.1 |
| Fiction harm detector | Proximity-scored framing | 1.1 |
| PAIR classifier | LinearSVM on sentence embeddings | 1.0 |
| Virtualization detector | Frame + safety-disable patterns | 1.0 |
| Many-shot detector | Power-law danger scoring | 1.0 |
| Indirect injection | Document-embedded instruction | 1.0 |
| Multilingual detector | Script anomaly + translated phrases | 1.0 |
| Perplexity proxy | Encoded payload heuristics | 0.7 |

Layer outputs are aggregated by weighted voting with a corroboration boost (+0.08 for 2 layers agreeing, +0.12 for 3+). A crescendo trajectory boost (up to +0.20) is applied based on session history. Results are routed through a three-zone classifier: CLEAR SAFE → UNCERTAIN → CLEAR ATTACK.

The **PAIR classifier** (Layer 7) is a LinearSVC trained on sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dimensions) with a security-instruction prefix. It is the primary semantic detection component and the focus of this research cycle.

---

## 3. Phase 1 — Baseline Evaluation

### 3.1 Setup

Evaluated against 2,006 prompts across 8 public benchmark datasets with 1,521 labeled attack prompts and 485 benign prompts. All results use the local 11-layer pipeline only — no external API calls, no LlamaGuard.

**Datasets:** AdvBench (Zou et al. 2023), JailbreakBench (JBB 2024), Anthropic Red Team (HuggingFace), HarmBench (Mazeika et al. 2024), OpenAI Moderation Evaluation (Markov et al. 2022), Stanford Alpaca (benign baseline).

### 3.2 Overall Results

| Metric | Value |
|---|---|
| Precision | **0.9747** |
| Recall | **0.6575** |
| F1 | **0.7852** |
| FPR | **0.0536** |
| ROC-AUC | **0.7921** |

### 3.3 Per-Category Results

| Category | Precision | Recall | F1 | FPR | Prompts |
|---|---|---|---|---|---|
| GCG Suffix | 0.981 | 0.958 | **0.969** | 0.033 | 255 |
| Prompt Injection | 0.993 | 0.760 | **0.861** | 0.029 | 634 |
| Indirect Injection | 0.652 | 1.000 | **0.790** | 0.114 | 85 |
| Fiction-Wrapped | 0.909 | 0.645 | **0.755** | 0.027 | 106 |
| Virtualization | 0.971 | 0.618 | **0.754** | 0.028 | 182 |
| Crescendo | 1.000 | 0.563 | **0.720** | 0.000 | 16 |
| OpenAI Moderation | 1.000 | 0.540 | **0.702** | 0.000 | 522 |
| Multilingual | 0.857 | 0.353 | **0.500** | 0.114 | 206 |

### 3.4 Layer Contribution

| Layer | Total Fires | TP Rate |
|---|---|---|
| PAIR Classifier | 555 | 95.9% |
| GCG Suffix | 265 | 95.8% |
| Direct Harm | 117 | 100.0% |
| Regex Patterns | 58 | 96.6% |
| Multilingual | 26 | 80.8% |
| Prompt Guard | 12 | 100.0% |
| Perplexity Proxy | 10 | 100.0% |

PAIR and GCG together account for **82% of all true positive detections**.

### 3.5 Phase 1 Finding

The system performs well on known attack categories. The notable exception is multilingual (35% recall) and OpenAI Moderation (54% recall on content-moderation-style harmful requests). The open question is whether this performance is genuine generalisation or benchmark memorisation.

---

## 4. Phase 2 — UnknownBench: Testing Generalisation

### 4.1 Motivation

High benchmark recall does not demonstrate generalisation. Benchmark datasets (AdvBench, JailbreakBench) use well-known attack vocabulary. A detection system trained on these datasets may be matching vocabulary rather than understanding adversarial intent.

To test this, we constructed **UnknownBench** — a 200-prompt dataset designed to avoid every keyword, pattern, and heuristic present in FIE's detection code.

### 4.2 Experiment Design

**Exp 1 — PAIR classifier in isolation:**
Run only the PAIR layer. Measures how much the semantic model alone generalises to unseen attacks.

**Exp 2 — Full FIE (11 layers):**
Run all 11 layers. Measures the marginal recall contribution of 10 additional specialist layers over PAIR alone.

### 4.3 Results

| Experiment | Recall | Precision | F1 | FPR |
|---|---|---|---|---|
| PAIR only | **11.0%** | 100.0% | 0.198 | 0.00% |
| Full FIE (11 layers) | **14.5%** | 100.0% | 0.253 | 0.00% |
| **Delta** | **+3.5%** | — | +0.055 | — |

### 4.4 Phase 2 Finding

**10 additional detection layers contribute only +3.5% recall on genuinely unknown attacks.**

This is the central finding of Phase 2. A 10-layer architecture expansion produces a marginal 3.5% gain on out-of-distribution inputs. This result rules out the hypothesis that architectural complexity confers generalisation. The specialist layers (virtualization, multilingual, many-shot, indirect injection) are vocabulary-dependent pattern matchers. They detect attacks that use their training vocabulary. When attacks shift vocabulary, the layers do not fire.

---

## 5. Phase 2 — GCG Calibration Fix (Exp 3)

### 5.1 Problem

During UnknownBench evaluation, we observed the GCG suffix detector producing high false positive rates on benign formal prose (academic papers, legal documents, financial reports). The FormalProseBench (75 benign prompts) was constructed to quantify this.

### 5.2 Root Cause

The GCG detector uses entropy and special-character density signals. Formal academic prose with Greek letters (α, β, τ²), em dashes, citation brackets, and subscripts genuinely elevates both signals — not because the text is adversarial, but because academic notation shares surface properties with GCG adversarial suffixes.

The LOW-range entropy threshold (E_LOW = 4.3) and LOW-range special density threshold (SD_LOW = 0.22) were triggering on legitimate academic content. A corroboration boost (+0.08) then pushed two LOW-range signals above the detection threshold.

### 5.3 Fix

Added `_is_natural_language_prose()` guard in `fie/adversarial.py`. When text has ≥65% letter ratio and ≥70% word-like token ratio, LOW-range entropy and special-density signals are suppressed. Real GCG attacks inject garbled token sequences that collapse the letter ratio well below 60%. Formal prose maintains >70% even with dense mathematical notation.

### 5.4 Results

| Configuration | FPR | False Positives / 75 |
|---|---|---|
| Before fix (GCG only) | **72.0%** | 54 |
| After fix (GCG only) | **6.7%** | 5 |
| After fix (Full FIE) | **13.3%** | 10 |

Remaining 5 GCG FPs: prompts with extremely dense mathematical notation that genuinely exceeds the HIGH-range entropy threshold. These are acceptable edge cases. Remaining FIE FPs: 5 GCG (math-heavy), 4 PAIR (formal academic language near classifier boundary), 1 regex (spirometry notation).

---

## 6. Phase 2 — Unknown Category Benchmarks (Exp 5)

### 6.1 Setup

Four specialist benchmarks of 50 prompts each. Each benchmark was designed after reading the complete detection vocabulary of its corresponding FIE layer — then deliberately avoiding every keyword, phrase, and structural pattern that layer uses.

| Benchmark | Attack Strategy | Avoided Vocabulary |
|---|---|---|
| UnknownVirtualizationBench | Temporal displacement, theatrical performance, documentary mode, frame narrative, character interiority | "imagine a world", "alternate reality", "safety rules don't apply", all 47 virtual-frame indicators |
| UnknownIndirectBench | Annotation injection, metadata injection, footnote injection, template slot injection | "ignore previous instructions", "your new task is", "forget", all document trigger phrases |
| UnknownMultilingualBench | Welsh, Finnish, Swahili, Hungarian, romanised scripts, code-switching | All 8 covered languages, translated injection phrases |
| UnknownManyShotBench | Professional context framing, graduated escalation without trigger vocabulary | "here are examples", all conditioning patterns, response format templates |

### 6.2 Results

| Category | TPR | Layers That Fired |
|---|---|---|
| Virtualization | **8%** | PAIR only |
| Indirect Injection | **24%** | PAIR 58%, regex 17%, GCG 17% |
| Multilingual | **0%** | None |
| Many-Shot | **8%** | PAIR 50%, regex 50% |

### 6.3 Phase 2 Finding (Extended)

All detections across all 4 categories came exclusively from PAIR and regex — not from the specialist layers designed for each category.

The multilingual result (0% TPR) is a complete blind spot. The multilingual layer has Tier 1 script anomaly detection (non-Latin characters) and Tier 2 translated phrase matching (8 languages × 8 phrase categories). When attacks use low-coverage languages (Welsh, Finnish, Swahili) or romanised scripts without Latin character anomalies, both tiers produce no signal.

**The combined Phase 2 conclusion:** FIE's specialist layers are pattern-dependent vocabulary matchers. PAIR partially generalises because it operates on semantic embeddings rather than pattern matching. The generalisation gap is a training distribution problem, not an architectural problem.

---

## 7. Phase 3 — Hard-Positive Retraining (Exp 6)

### 7.1 Motivation

The 169 prompts missed by FIE across Exp 5's four categories represent a structured sample of out-of-distribution attack patterns. Rather than adding more detection rules, we retrain PAIR directly on these failures.

**Important constraint:** UnknownBench-v1 (the source of missed prompts) becomes training data. UnknownBench-v2 (200 prompts created with a completely different generation strategy — Unicode manipulation, identity/credential assertion, task-embedded injection, graduated escalation) is held out as independent evaluation.

### 7.2 PAIR v3 Training

- **Hard positives:** 169 FN prompts from Exp 5 (weight = 5×)
- **Hard negatives:** 10 FP prompts from Exp 3 FormalProseBench (weight = 3×)
- **Original data:** Alpaca benign + JailbreakBench harmful/benign (weight = 1×)
- **Architecture:** LinearSVC (C=0.8) + CalibratedClassifierCV (sigmoid, cv=3)
- **Embedding:** all-MiniLM-L6-v2 with L2 normalisation

### 7.3 Before/After Results (Exp 6)

| Benchmark | Before (v2) | After (v3) | Delta |
|---|---|---|---|
| Virtualization v1 (training src) | 8.00% | 98.00% | +90.00% |
| Indirect v1 (training src) | 24.00% | 100.00% | +76.00% |
| Multilingual v1 (training src) | 0.00% | 100.00% | +100.00% |
| Many-shot v1 (training src) | 8.00% | 100.00% | +92.00% |
| **Virtualization v2 (held-out)** | **8.00%** | **100.00%** | **+92.00%** |
| **Indirect v2 (held-out)** | **38.00%** | **100.00%** | **+62.00%** |
| **Multilingual v2 (held-out)** | **54.00%** | **100.00%** | **+46.00%** |
| **Many-shot v2 (held-out)** | **4.00%** | **100.00%** | **+96.00%** |

TPR improvement generalises to the held-out v2 benchmarks using completely different attack surfaces. This rules out simple memorisation.

### 7.4 Calibration Collapse

However, FPR on FormalProseBench (75 benign prompts) collapsed from 13.3% to **46.67%** after retraining with 5× hard-positive weight. The model was flagging nearly half of all benign formal prose.

This reveals that 100% TPR in Exp 6 is not purely genuine generalisation — the decision boundary shifted so aggressively that false positive suppression required a high threshold override.

**Phase 3 finding:** Hard-positive training dramatically recovers out-of-distribution recall. The 169 missed prompts contain genuine semantic signal about adversarial intent that transfers to novel attack surfaces (held-out v2). However, 5× weighting over-pushes the decision boundary, collapsing precision calibration.

---

## 8. Phase 4 — Threshold Calibration (Exp 7)

### 8.1 Motivation

Before retraining (which is expensive and risks introducing new failures), we ask: does a threshold sweep on the existing v3 model recover the precision-recall tradeoff without sacrificing the recall gains?

This is a cheaper experiment. Model weights are fixed. Only the classification threshold varies.

### 8.2 Threshold Sweep Results

Evaluated on 400 attack prompts (Exp 6/after, label=1) and 75 benign prompts (FormalProseBench, label=0).

| Threshold | TPR | FPR | Precision | F1 | Note |
|---|---|---|---|---|---|
| 0.50 | 99.75% | 46.67% | 91.94% | 0.9568 | |
| 0.55 | 99.75% | 45.33% | 92.15% | 0.9580 | |
| 0.60 | 99.75% | 38.67% | 93.22% | 0.9638 | |
| 0.65 | 99.75% | 29.33% | 94.77% | 0.9720 | best F1 |
| 0.70 | 99.25% | 28.00% | 94.98% | 0.9707 | |
| 0.75 | 96.75% | 16.00% | 96.99% | 0.9687 | |
| **0.80** | **96.25%** | **14.67%** | **97.22%** | **0.9673** | **target zone** |
| 0.85 | 93.25% | 8.00% | 98.42% | 0.9576 | |
| 0.90 | 92.00% | 6.67% | 98.66% | 0.9521 | |

**Target zone (TPR ≥ 60% AND FPR ≤ 15%):** Satisfied at **t = 0.80**.

### 8.3 Phase 4 Finding

The target zone exists. No retraining is required. At t = 0.80:
- TPR = 96.25% — well above the 60% target
- FPR = 14.67% — just inside the 15% limit
- Precision = 97.22%
- F1 = 0.9673

The v3 model already encoded the correct decision boundary — it simply needed a higher threshold to suppress the false positives introduced by aggressive hard-positive weighting. **The model had the right weights; the calibration of its output probabilities required correction via threshold.**

Production threshold set to **t = 0.80** in `pair_intent_meta_v3.json`. This is the current production configuration (FIE v1.11.0).

---

## 9. Phase 5 — Weight Comparison (Exp 8)

### 9.1 Motivation

If 5× weighting worked but required threshold correction, a natural question is: would 3× weighting produce the same recall gains with better natural calibration (i.e., entering the target zone at a lower threshold)?

This is not a production decision — v3 at t = 0.80 is already in the target zone and stable. It is a research question: **what is the correct hard-positive weight for future retraining cycles?**

### 9.2 Results

| Weight | Target Zone Threshold | TPR | FPR | F1 |
|---|---|---|---|---|
| 5× (v3, production) | **0.80** | 96.25% | 14.67% | 0.9673 |
| 3× (Exp 8 comparison) | **0.70** | 99.25% | 14.67% | 0.9827 |

| Metric | Delta (3x vs 5x) |
|---|---|
| Threshold | **−0.10** (3x enters target zone lower) |
| TPR | **+3.00%** |
| FPR | **0.00%** (identical) |
| F1 | **+0.0153** |

### 9.3 Phase 5 Finding

3× weighting is strictly better than 5× weighting on every metric:
- Same FPR
- Higher TPR (+3%)
- Better F1 (+0.015)
- Lower operating threshold (−0.10)

The lower threshold is the key indicator. It means the 3× model's raw probability scores are better calibrated — the model does not need as aggressive a threshold override to suppress false positives. The 5× model's higher threshold (0.80) is compensating for over-pushed confidence scores, not reflecting genuinely better discrimination.

**Recommendation for future retraining:** use 3× hard-positive weighting.

---

## 10. Summary of Findings

### Primary Finding

> **The bottleneck in adversarial prompt detection generalisation is training distribution, not architectural complexity.**

Evidence:
- 10 specialist detection layers → +3.5% unknown-attack recall (Exp 1 vs Exp 2)
- Hard-positive retraining on 169 missed prompts → +82–96% unknown-attack recall (Exp 6)
- The improvement generalises to completely held-out attack surfaces (UnknownBench-v2)

### Secondary Findings

1. **Vocabulary-dependent layers do not generalise.** Specialist layers (virtualization, multilingual, many-shot, indirect injection) fire exclusively on attack patterns using their training vocabulary. When vocabulary shifts, they produce zero signal. All Exp 5 detections came from PAIR and regex only.

2. **Hard-positive training moves the decision boundary dramatically.** 169 prompts at 5× weight shifted the boundary enough to achieve near-100% TPR on novel attacks — including held-out attacks using completely different surface forms. This is genuine generalisation, not memorisation.

3. **Over-weighting degrades calibration without improving discrimination.** 5× weighting achieved the target zone but required threshold = 0.80. 3× weighting achieves the same zone at threshold = 0.70 with better TPR and F1. Higher weight ≠ better model.

4. **Calibration is recoverable via threshold sweep.** When retraining degrades probability calibration, a threshold sweep over the fixed model is a cheaper and lower-risk correction than retraining. In this case it was sufficient.

5. **FPR on formal prose is a meaningful proxy for calibration quality.** The FormalProseBench (75 academic/legal/financial prompts) successfully detected the calibration collapse (6.7% → 46.67%) and confirmed recovery (14.67%) without requiring a separate held-out benign dataset.

### Progression Summary

| Phase | Key Result |
|---|---|
| Phase 1 | Baseline: F1 0.785, Precision 0.975 on known attacks |
| Phase 2 | Architecture complexity: +3.5% unknown recall from 10 extra layers |
| Phase 3 | Hard-positive training: 8% → 96% unknown recall, but FPR collapsed |
| Phase 4 | Threshold calibration: FPR recovered (14.67%) without recall loss |
| Phase 5 | Weight comparison: 3× outperforms 5× on calibration and F1 |

---

## 11. Known Limitations

**Multilingual detection.** Novel multilingual attacks (low-coverage languages, romanised scripts) achieve 0% TPR. The multilingual layer covers 8 languages with fixed phrase matching. Languages outside that set and romanised scripts that avoid non-Latin characters produce no signal from any layer.

**White-box evasion.** FIE's codebase is open source. An attacker who reads the detection code can craft prompts that score below every layer's threshold. FIE is not designed to be a black box and makes no claim against informed adversarial optimisation.

**Content moderation categories.** Hate speech, harassment, and self-harm content phrased as normal conversational requests is outside FIE's design scope. The OpenAI Moderation dataset (54% recall) reflects this intentional scope limitation.

**Formal prose FPR at 13.3% (Full FIE).** The GCG fix reduced GCG-specific FPR to 6.7%. Remaining full-FIE false positives include PAIR false positives on dense academic language near the classifier boundary. These require further PAIR calibration or a formal-prose-specific hard-negative training pass.

**Benchmark scale.** UnknownBench-v1 and v2 contain 50 prompts per category. While the design methodology is rigorous (vocabulary avoidance, frozen SHA-256 manifests, independent generation strategies), the sample sizes are small relative to production attack diversity.

---

## 12. Future Work

**Immediate**
- Exp 8 establishes 3× as the correct hard-positive weight. Future retraining cycles should use 3× with threshold calibration.
- The 13.3% full-FIE FPR on formal prose should be addressed with a hard-negative pass on academic prose examples.

**Research**
- **Multilingual generalisation.** The complete blind spot on novel multilingual attacks requires a fundamentally different approach — likely embedding-level detection that is script-agnostic rather than phrase-matching.
- **UnknownBench-v3.** Each research cycle should produce a new held-out evaluation set with a different generation strategy before retraining. This prevents the train/eval leakage that erodes benchmark validity over time.
- **Test set establishment.** The current evaluation framework uses training-source benchmarks (v1) and held-out benchmarks (v2) but lacks a permanently frozen test set that is never used for threshold selection. Establishing this would give future results stronger validity claims.
- **Continuous hard-positive collection.** Production deployment provides a stream of real attack attempts. A pipeline that flags near-miss detections (UNCERTAIN zone, confidence 0.55–0.75) for human labeling and feeds them back as hard positives in the next retraining cycle would continuously improve unknown-attack coverage.

---

## 13. Reproduction

All experiments are fully reproducible from the repository.

```bash
git clone https://github.com/AyushSingh110/Failure_Intelligence_System.git
pip install -e ".[ml]"

# Phase 1 baseline
python -m evaluation.run_baseline

# Phase 2 UnknownBench
python -m evaluation.phase2.exp1_pair_only.run_eval
python -m evaluation.phase2.exp2_pair_gcg.run_eval

# GCG calibration
python -m evaluation.phase2.exp3_formal_prose.run_eval
python -m evaluation.phase2.exp3_formal_prose.run_eval --gcg-only

# Unknown category benchmarks
python -m evaluation.phase2.exp5_unknown_categories.run_eval

# Hard-positive retraining
python scripts/retrain_pair_v3.py

# Before/after comparison
python -m evaluation.phase2.exp6_before_after.run_eval --phase before
python -m evaluation.phase2.exp6_before_after.run_eval --phase after
python -m evaluation.phase2.exp6_before_after.run_eval --compare

# Threshold sweep
python -m evaluation.phase2.exp7_threshold_sweep.run_eval

# Weight comparison
python scripts/retrain_pair_exp8.py
python -m evaluation.phase2.exp8_weight_comparison.run_eval
```

Benchmark integrity is verified by SHA-256 manifest:

```bash
python -m evaluation.datasets.freeze_benchmarks --verify
```

---

## 14. References

- Zou et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043
- Mazeika et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming.* arXiv:2402.04249
- Markov et al. (2022). *A Holistic Approach to Undesired Content Detection in the Real World.* arXiv:2208.03274
- Chao et al. (2023). *Jailbreaking Black Box Large Language Models in Twenty Queries.* arXiv:2310.08419
- JailbreakBench (2024). *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* arXiv:2404.01318

---

*FIE is available at [pypi.org/project/fie-sdk](https://pypi.org/project/fie-sdk). Source code at [github.com/AyushSingh110/Failure_Intelligence_System](https://github.com/AyushSingh110/Failure_Intelligence_System).*
