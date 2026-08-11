# Over-Refusal, Contamination-Audited Recall, and a Data-Fixable Defense

*A self-contained summary of FIE experiments E18–E26 (June 2026). Numbers and methods
are traceable to `docs/RESEARCH_LOG.md`; figures in `notebooks/pair_classifier_plots/`.*

## Thesis

Contamination and in-distribution evaluation systematically inflate apparent guardrail
quality. Under a clean, out-of-distribution, over-refusal-aware protocol: (a) reported
recall is honest only after a leakage audit; (b) a large over-blocking blind spot
appears; (c) detection capability concentrates in one semantic classifier, not the
layered architecture; (d) recall gaps are *data-fixable* while the over-refusal blind
spot is not easily fixed. We release the protocol, decontaminated splits, and an
offline reference system (FIE / PAIR).

FIE = an offline 12-layer prompt-attack detector whose detection is carried by **PAIR**,
a LinearSVC + sigmoid-calibrated classifier on 384-d all-MiniLM-L6-v2 embeddings
(~23M params, ~48 ms, fully local).

---

## 1. The over-refusal blind spot (E18–E20)

Prior work measured FIE's false-positive rate on *naturally-occurring* benign prompts
(≈9% OOD FPR). We re-measured on the field's **standardized over-refusal benchmarks** —
XSTest and OR-Bench-hard (prompts engineered to look harmful but be safe).

| Over-refusal rate (% safe prompts flagged) | FIE | ProtectAI | jackhhao |
| --- | --- | --- | --- |
| XSTest-safe | 53.6% | 0.0% | 0.0% |
| OR-Bench-hard | 90.4% | 6.0% | 2.0% |
| Pooled | 72.0% | 3.0% | 1.0% |
| *paired* unsafe-contrast recall | 88.9% | **0.0%** | **0.0%** |

**The comparison is a 2-D tradeoff, not a one-sided win.** The baselines' ~0%
over-refusal is *not* discrimination — their paired unsafe-contrast recall is also 0%;
they are inert on this register (injection detectors miss plain harmful *content*). FIE
is the only system that engages it (88.9% unsafe recall) — and pays with 54–90%
over-blocking. **Our own prior 9% OOD-FPR understated the problem ~8×** vs
adversarially-benign prompts (E18).

**Mechanism (E19):** 47% of FIE's over-refusals are *high-confidence* (mean 0.76), not
borderline. A threshold tweak recovers ≤28% — it is a representation/training-data
property, not an operating-point one. Worst category: figurative language (88%).

**No acceptable operating point (E20, fig `figA_overrefusal_tradeoff.png`):** sweeping
FIE's threshold never reaches low-over-refusal + high-recall. Even at t=0.95 (recall
collapsed to ~37%), over-refusal (7.2%) cannot reach the baselines' 0%. The "good
corner" is empty.

---

## 2. Contamination-audited recall suite (E21–E22)

We expanded recall evaluation from 2 to **5 benchmarks** (JailbreakBench, HarmBench,
AdvBench, StrongREJECT, SORRY-Bench), each leakage-audited against PAIR's training
(exact + cosine ≥ 0.92) and cross-deduplicated, with SHA-pinned frozen splits.

**Contamination is real and would have inflated numbers:** AdvBench is **67.5%**
training-leaked (its mirror is in PAIR training); JailbreakBench 52.5%. We quarantine
AdvBench as a *contamination case study* (its "clean" remainder still scores 95% —
residual memorization) and pool the headline over the four clean-enough benchmarks.

**Clean recall (macro across 4 benchmarks, 95% CI):**

| System | Macro recall | Note |
| --- | --- | --- |
| **FIE (PAIR v6.2)** | **85.8% [83.7, 87.6]** | offline, ~48 ms |
| ProtectAI DeBERTa | 21.8% | injection detector — register-bound |
| jackhhao | 25.6% | jailbreak detector — register-bound |

Baselines score ~87% on JailbreakBench (their register) but **0–11%** on harmful-content
benchmarks — a register mismatch, not a like-for-like ranking (the proper guard-model
comparison is §3).

**Per-category gaps, named (SORRY-Bench, official taxonomy):** FIE's recall degrades
along an explicit→soft-harm gradient (Hate Speech 91% → Unqualified Advice 55%). Two
distinct causes, kept separate: (i) *definitional* — medical/legal/financial *advice*
is liability-restricted, which FIE deliberately treats as benign; (ii) *genuine* —
environmental crimes (11%), fake news, false advertising, PII, IP infringement are
adversarial soft-harm FIE under-detects (the euphemism gap).

---

## 3. The strong-baseline frontier (E23, fig `figA2_frontier_overlay.png`)

We placed a strong policy-following guard, **gpt-oss-safeguard-20b** (via Groq, online),
on both axes against FIE and the weak offline baselines.

| System (class) | macro recall | XSTest over-refusal | OR-Bench over-refusal |
| --- | --- | --- | --- |
| gpt-oss-safeguard-20b (20B, online) | 94.2% | **11.0%** | 80.0% |
| FIE PAIR (≈23M, 48 ms, offline) | 85.8% | 53.6% | 90.4% |
| ProtectAI / jackhhao (offline) | ~22–26% | ~0% | ~2–6% |

**Two findings:** (1) The 20B online guard out-detects FIE *and* reaches the XSTest
good corner FIE cannot — so **FIE's value is deployability (offline/tiny/fast), not
detection supremacy.** (2) **OR-Bench-hard defeats even the 20B guard (80%)** — the
adversarial-benign over-blocking blind spot is *universal*, not FIE-specific. The good
corner is empty for every system on OR-Bench. (Offline LlamaGuard-3/ShieldGemma were
not loadable on the 4 GB-GPU / limited-RAM target hardware — itself a deployability
data point.)

---

## 4. A data-fixable defense — and an asymmetry (E24–E26)

We tested whether FIE can fix its measured *recall* gaps by targeted augmentation,
under strict rigor (leakage-audit vs all eval splits; seed-split held-out test;
two-sided no-regression gate; cross-benchmark generalization check).

- **E24 — diagnostic:** targeted soft-harm/euphemism augmentation (hand-authored +
  NLLB back-translation) lifts held-out aug-test +25.5 and **transfers to SORRY
  soft-harm +20.8 at equal over-refusal** → the gap is a **data problem, not a
  representation ceiling**. But the recall-only fix *fails the over-refusal gate*
  (XSTest +8) — closing one axis cost the other.
- **E25 — counter-augmentation:** adding safe-but-scary benign negatives yields
  **v6.3b**, which banks the recall gain at **flat over-refusal** (XSTest −0.8,
  OR-Bench +0.0). No register-overfitting (both benchmarks held). But the benign
  negatives only *neutralize* the positives' side-effect — they do **not** reduce the
  baseline over-refusal blind spot (held-out benign FPR unchanged at n=46).
- **E26 — ship-gate:** the Pareto win survives the **full 12-layer pipeline**
  (soft-harm recall +32.1, over-refusal flat, no clean regression). **v6.3b shipped as
  the new default.**

**The asymmetry (the spine):** recall gaps are **data-fixable** (E24/E26 ship a +32-pt
soft-harm improvement); the over-refusal blind spot **resisted** benign augmentation at
n=46 (XSTest 53%, OR-Bench 90% unchanged). Whether over-refusal is data-starved or
representation-limited is left to future work.

---

## Honest limitations

- gpt-oss-safeguard ran online (n=100/set); offline guards were hardware-infeasible.
- SORRY soft-harm per-category n is small (~50); domain-level and held-out aug-test
  conclusions are better-powered.
- The over-refusal "not fixable" claim is hedged: benign augmentation was small (n=46);
  the data-starved vs representation-limited question is open (time-boxed future work:
  ~200 register-diverse benign, one retrain, one gate).
- All FIE numbers are offline; the strong-guard comparison is capability-vs-cost, not
  apples-to-apples deployment.

## Reproducibility

Frozen leakage-audited splits: `data/benchmark_audit/*_clean.jsonl`,
`data/overrefusal/*_clean.jsonl` (+ manifests with HF revisions and SHA-256). Scripts:
`audit_overrefusal.py`, `overrefusal_by_category.py`, `overrefusal_tradeoff.py`,
`audit_attack_benchmarks.py`, `measure_combined_recall.py`, `measure_guard_baselines.py`,
`frontier_overlay.py`, `build_augmentation_v63.py`, `build_benign_augmentation_v63b.py`,
`train_pair_v63{,b}.py`, `eval_v63{_vs_v62,b}.py`, `gate_v63b_fullpipeline.py`. Full
method/number trail: `docs/RESEARCH_LOG.md` (E18–E26).
