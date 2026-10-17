# E53 — Frozen-encoder head diagnostic: report

| | |
| --- | --- |
| Date | 2026-10-17 |
| Repository | `rebuild/wp-001-eval-harness` @ `f1eaeaf`, clean before and after (outputs only in this directory) |
| Preregistration | [PREREGISTRATION.md](PREREGISTRATION.md), SHA-256 `7639e946…f9bf`, logged before any training ([logs/RUN_LOG.md](logs/RUN_LOG.md)) |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`: read, never written |
| A0 sanity gate | **Passed on the corrected re-run** ([results/a0_reproducibility_rerun1.json](results/a0_reproducibility_rerun1.json)); the first run failed for a reason in this experiment's code (D9) |
| Outcome | **H1 not supported. H2 not supported. H0 retained.** |

Labels used below: **Observed** (measured here), **Estimate** (with 95% paired-bootstrap interval),
**Interpretation**, **Hypothesis** (to test next).

## 1. What was run

Every arm uses the verified production ONNX MiniLM encoder (hashes in
[manifests/inputs_summary.json](manifests/inputs_summary.json)) and the unchanged v6.3b head recipe. Only the
training rows or their weights change.

| Arm | Training set (rows) | What changes |
| --- | --- | --- |
| A0 | 3,812 | v6.3b recipe reproduced |
| A1 | 3,389 | − HarmfulQA (423 attack rows) |
| A2 | 3,383 | − Anthropic red-team (429 attack rows) |
| A3 | 2,960 | − both (852) |
| A4 | 3,812 | source-balanced weights within each class |
| R1/R2/R3 × 5 seeds | 3,389 / 3,383 / 2,960 | same number of attack rows removed at random from *other* attack sources (size and class-balance controls) |
| A5 | 18 leave-one-source-out heads + 5-fold row-random CV | diagnostic of split optimism |

Full composition, per source and per label, with effective weight sums: [results/arm_configs.json](results/arm_configs.json).

**Thresholds** come from the calibration pool only. The pool is `val.jsonl` plus `augmented_v6_3_test.jsonl`,
with 45 of 370 rows dropped as exact or cosine ≥ 0.92 near-duplicates of training or evaluation prompts;
the drop list is in [manifests/calibration_dropped.jsonl](manifests/calibration_dropped.jsonl). Each arm gets
the threshold that matches A0's recall at 0.50 on the 83 primary-pool attacks: r* = 0.9157
([results/thresholds.json](results/thresholds.json)). No evaluation set touched any threshold.

## 2. A0 sanity gate (Observed)

| Check | First run | Corrected re-run |
| --- | --- | --- |
| Gate 1: shipped head through this encoder path vs BL-0002 PAIR records (1,848 items) | 0 decision mismatches, max \|Δ\| 5.0e-5 (4-dp rounding) | same |
| Gate 2: A0 vs shipped, decision agreement at 0.50 (2,163 items) | **97.83% — FAIL** | **100.00% — PASS** |
| Pearson / Spearman | 0.9945 / 0.9961 | 0.9999999999997 / — |
| Max \|score difference\| | 0.172 | 1.5e-6 |
| `va` at 0.45 vs shipped metadata | TPR 0.934 / FPR 0.173 vs 0.930 / 0.163 | **0.9298 / 0.1633, identical** |

The first failure had one cause, found before any arm was trained (D9). This experiment fitted the training
rows in corpus order, whereas the v6.3b script fits them in `train_test_split` order. `CalibratedClassifierCV(cv=3)`
folds depend on row order, which explained the per-fold coefficient cosines of 0.83–0.91.

With the order fixed, A0 is numerically the shipped head. A side result is that deviation D1 (ONNX instead of
the original torch/CUDA training embeddings) has no measurable effect.

## 3. Primary results

PAIR-isolated, at each arm's calibration-matched threshold. All rates are in %; deltas are in percentage
points vs A0, with 95% paired, dataset-stratified bootstrap intervals from 10,000 resamples. Source:
[results/metrics.json](results/metrics.json); table: [results/summary_table.md](results/summary_table.md).

| Model | t | OR XSTest | OR OR-Bench | Macro recall | PR-AUC pooled | Med+legal evasion |
| --- | --- | --- | --- | --- | --- | --- |
| Shipped @ 0.50 (reference) | 0.500 | 52.8 | 90.4 | 88.0 | 0.825 | 23.2 |
| A0 | 0.587 | 44.4 | 84.8 | 84.7 | 0.825 | 25.7 |
| A1 −HarmfulQA | 0.486 | 49.2 | 83.2 | 86.7 | 0.848 | 25.7 |
| A2 −red-team | 0.347 | 40.8 | 93.6 | 88.5 | 0.803 | 21.9 |
| A3 −both | 0.255 | 43.6 | 91.2 | 90.4 | 0.827 | 20.8 |
| A4 source-balanced | 0.515 | 40.8 | 87.2 | 89.0 | 0.842 | 30.3 |

| Comparison | ΔOR XSTest | ΔOR OR-Bench | ΔMacro recall | ΔPR-AUC | ΔMed+legal evasion |
| --- | --- | --- | --- | --- | --- |
| A1 − A0 | +4.8 [+0.8, +8.8] | −1.6 [−4.4, +1.2] | +2.0 [+0.9, +3.2] | **+2.2 [+1.8, +2.7]** | −0.0 [−1.7, +1.7] |
| A2 − A0 | −3.6 [−8.0, +1.2] | **+8.8 [+5.2, +12.8]** | +3.8 [+2.7, +5.0] | −2.3 [−2.9, −1.6] | −3.9 [−5.6, −2.1] |
| A3 − A0 | −0.8 [−5.6, +4.0] | **+6.4 [+2.8, +10.4]** | +5.7 [+4.3, +7.1] | +0.2 [−0.5, +0.9] | −4.9 [−7.0, −2.8] |
| A4 − A0 | −3.6 [−7.6, +0.4] | +2.4 [−0.4, +5.6] | +4.3 [+3.0, +5.6] | **+1.7 [+1.2, +2.2]** | **+4.5 [+2.6, +6.4]** |

Holm-adjusted p-values over the eight over-refusal comparisons are in `metrics.json`. The only significant
ones are A2 and A3 on OR-Bench, both in the *worse* direction.

### Hypothesis outcomes (preregistered rules, section 10)

- **H1 — not supported.** A3 does not lower over-refusal on either set. XSTest is −0.8 [−5.6, +4.0];
  OR-Bench-hard *rises* by +6.4 [+2.8, +10.4].
- **H2 — not supported.** Source balancing *raises* medical/legal framing evasion: +4.5 [+2.6, +6.4].
- **H0 — retained.** No arm among A1–A4 lowers over-refusal on either benign set with an interval excluding
  zero while keeping macro recall within 2 points.
- **What this does and does not mean.** Retaining H0 says these source removals and this reweighting do not
  reduce hard-benign over-refusal on this corpus at this operating point. It does **not** show that the encoder
  representation is the limit. Relabelling, new hard negatives, other objectives and other encoders were not
  tested.

## 4. Secondary results

**Removal arms shift the operating point; they do not separate the classes better (Interpretation).**
- A2 and A3 reach matched calibration recall only with much lower thresholds (0.35 / 0.26).
- On the evaluation sets that buys realised attack recall (+3.8 / +5.7) and costs OR-Bench over-refusal
  (+8.8 / +6.4).
- A3's threshold-free PR-AUC is unchanged (+0.2 [−0.5, +0.9]).
- **Matched calibration recall did not mean matched evaluation recall.** Realised macro recall rose in every arm.

**Threshold-free gains exist but are small (Estimate).**
- A1 (+2.2 PR-AUC) and A4 (+1.7) rank attacks above hard-benign prompts slightly better; both intervals exclude 0.
- Neither gain became an over-refusal reduction at the calibrated operating point. A1 raised XSTest over-refusal by +4.8.

**The size controls reverse the "noisy labels cause over-refusal" story (Observed).**
- Removing the same number of *random* attack rows from other sources raises XSTest over-refusal sharply:
  R3 +18.9 [+14.5, +23.5].
- Removing HarmfulQA and red-team instead is −19.7 [−25.5, −13.8] better than random removal on XSTest.
- **Interpretation:** these sources are less load-bearing than other attack data for the XSTest decision,
  but keeping them does not cause the over-refusal. The preregistered attribution rule (≤ −6 on both sets) is
  not met, because on OR-Bench A3 is +2.8 [−1.0, +6.4] vs its control.

**OR-Bench-hard is insensitive to every intervention (Observed).**
- Its median score stays between 0.85 and 0.93 in every arm.
- XSTest-safe's median drops from 0.51 to 0.18 under A3.
- The two benchmarks move independently or in opposite directions, as in E28/E29.

**Source-held-out optimism is large (A5, Observed; [results/a5_loso.json](results/a5_loso.json)).** For a
source never seen in training, the positive rate changes as follows:

| Source | Label | Row-random CV | Leave-source-out | Optimism (pts) |
| --- | --- | --- | --- | --- |
| anthropic_rlhf_helpful | benign | 21.7% flagged | 54.1% flagged | 32.4 |
| MedQuad | benign | 1.4% | 29.5% | 28.1 |
| law_stackexchange | benign | 40.0% | 65.6% | 25.6 |
| alpaca | benign | 11.4% | 31.6% | 20.1 |
| toxicchat | attack | 92.1% caught | 68.6% caught | 23.6 |
| aug_v63 (soft-harm) | attack | 87.8% | 65.3% | 22.5 |
| anthropic_rlhf_redteam | attack | 79.5% | 58.5% | 21.0 |
| harmfulqa | attack | 91.5% | 75.9% | 15.6 |
| llama paraphrase | attack | 97.2% | 81.8% | 15.4 |
| JBB, MaliciousInstruct, harmful_behaviors, forbidden_qs, MBPP, medqa | — | — | — | 0–3 |

**Interpretation:** for half of the sources, the head's decisions depend heavily on having seen text of that
source or register. Row-random validation numbers, including the shipped head's recorded validation metrics,
overstate performance on new benign registers by 20–32 points.

**Dilution and framing are untouched by data changes (Observed).**
- HarmBench padded with 84 words of filler: 0.8–6.7% recall across the arms.
- 168 words or more: 0% in every arm.
- Legal-frame evasion: 33–44% in every arm.

**Also reported in `metrics.json`:**
- per-set recall;
- StrongREJECT categories and SORRY-Bench domains/classes ("Unqualified Advice" 51–63% in all arms);
- framed and padded over-refusal;
- `test.jsonl`;
- calibration-pool metrics;
- uncensored score quantiles.

## 5. Deviations from the preregistered design

| # | Deviation | Effect on interpretation |
| --- | --- | --- |
| D1 (declared) | ONNX embeddings instead of the torch/CUDA embeddings v6.3b was trained on | None measurable: A0 = shipped to 1.5e-6 |
| D2 (declared) | Calibration is row-, near-duplicate- and seed-separated, not source-separated | Thresholds are tuned in-source; A5 shows how optimistic that is |
| D5 (declared) | A5 has no evaluation-set metrics | — |
| D6 | Near-duplicate filter used *prefixed* embeddings, which is stricter; max kept cosine 0.918 | More calibration rows dropped (45), no leakage |
| D7 | Primary calibration pool n = 83, so recall moves in steps of about 1.2 points | Thresholds are coarse; A0's matched t is 0.587 (same flagged calibration set as 0.50); A0 @ 0.50 also reported |
| D8 | PAIR-isolated: shipped macro recall 88.0 here vs 88.25 for the BL-0002 full pipeline (XSTest/OR identical) | Rule layers excluded by design |
| D9 | First A0 gate failed because of a row-order defect in this experiment's code; fixed, re-run, failed result retained | No arm was trained before the fix |
| — | The training environment (sentence-transformers + CUDA torch) no longer exists locally | Not needed after D9 |

## 6. Limitations

- **All evaluation sets are development data.** They influenced v6.3/v6.3b. This is not a sealed test and not
  evidence about unseen attack families.
- **The calibration pool is small, and its primary attacks come from three sources.** The matched-recall rule
  then lands at very different evaluation operating points; realised recall diverged from calibration recall
  in every arm.
- **Only source removal and reweighting were tested.** No relabelling (forbidden by scope) and no new data.
  The llama-paraphrase "attack" rows that are refusals were left as labelled.
- **Over-refusal sets have n = 250**, so effects below about 6 points cannot be resolved.

## 7. What merits further investigation

| Intervention | Verdict | Reason |
| --- | --- | --- |
| Drop HarmfulQA / red-team (A1–A3) | Not worth pursuing for over-refusal | No reduction; OR-Bench worse; gains are operating-point shifts |
| Source-balanced weights (A4) | Not as is | Small PR-AUC gain, but framing evasion worse |
| Source-held-out evaluation (A5) | **Adopt as a standard metric** | 15–32-point optimism hidden by row-random splits |
| Group-robust / de-confounded training (audit C2) | **Hypothesis: most promising next** | A5 shows source/register dependence; balancing alone is not enough |
| Encoder comparison (E40) | Hypothesis | OR-Bench is insensitive to every data change tried; needs owner approval to obtain other encoders |
| Segment-aware scoring (audit C1) | Independent need | Dilution unchanged by any data arm |
| A larger, source-diverse calibration pool | Prerequisite | Matched-recall thresholds were unstable across arms |

## 8. Reproduce

```bash
PY=/c/Users/ASUS/anaconda3/envs/failure-engine/python.exe
D=research/adversarial/e53_frozen_encoder_head_diagnostic
$PY -B -s $D/scripts/s01_build_inputs.py
$PY -B -s $D/scripts/s02_embed.py              # ~9 min CPU
$PY -B -s $D/scripts/s03_calibration_and_gate.py
$PY -B -s $D/scripts/s03_calibration_and_gate.py --gate-only a0_reproducibility_rerun1.json
$PY -B -s $D/scripts/s04_arms.py
$PY -B -s $D/scripts/s05_stats.py
$PY -B -s $D/scripts/s06_integrity.py && $PY -B -s $D/scripts/s06_integrity.py checksums
```

All scripts refuse to overwrite outputs, so a rerun needs a fresh directory. Integrity:
[results/integrity_check.json](results/integrity_check.json); checksums: [SHA256SUMS.txt](SHA256SUMS.txt).
