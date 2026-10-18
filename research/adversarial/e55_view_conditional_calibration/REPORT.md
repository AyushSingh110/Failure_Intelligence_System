# E55 — View-conditional calibration validation: report

| | |
| --- | --- |
| Date | 2026-10-17 |
| Repository | `rebuild/wp-001-eval-harness` @ `29bd6e2` (E54 commit); protected paths unchanged; outputs only here |
| Preregistration | [PREREGISTRATION.md](PREREGISTRATION.md), SHA-256 `681bfceb…3584`, logged before any item was built or scored ([logs/RUN_LOG.md](logs/RUN_LOG.md)) |
| Scorer | E54 `e54_views.py` / `e54_common.py`, byte-identical (verified at the end); frozen PAIR v6.3b; nothing trained |
| Stop-rule checks | E54 exploratory result reproduced exactly from E54 records ([results/e54_reproduction.json](results/e54_reproduction.json)): 55.83% (n=120) / 22.36% (n=161) / −5.47. Fidelity: V0 vs BL-0002 0 mismatches, max \|Δ\| 5e-5; vs E54 V0 max \|Δ\| 0.0 over 2,143 shared texts ([results/fidelity_gate.json](results/fidelity_gate.json)) |
| **Outcome** | **V1-conditional fails as a research candidate. H1 not supported, H2 supported, H3 not supported, A6 and A7 fail. H0 retained, at both α = 10% and α = 5%.** |

Labels: **Observed** (measured), **Estimate** (paired-bootstrap 95% interval), **Interpretation**.

## 1. Independence achieved, and what is still not independent

| Element | E55 source | Independent of the E54 design? |
| --- | --- | --- |
| Padding filler | Human-authored MMLU questions from 15 neutral subjects (pool of 2,771); varied per attack and level | **Yes.** E54 repeated one logistics sentence |
| Framing | 8 new templates (medical, legal, compliance, neutral × inline/suffix), including inline frames with no sentence or colon boundary | **Template-held-out, but model-authored by the assistant**, so not sufficient on its own |
| Manipulation | 6 neutral sentences joined by a line break (model-authored) | New |
| Base attacks (647) | 260 HarmBench prompts outside E54's padded 120, plus 387 SORRY-Bench prompts | Not padded or framed in E54, **but development data** (scored unpadded in E54 and earlier). Not source- or family-held-out |
| Short benign | TruthfulQA (817) and HaluEval (first 600), split into calibration and evaluation halves | **Yes.** Never used in PAIR work; 0 exact overlap with training or E54 |
| Long benign | MMLU-long items not used in E54 (105 rows sharing an E54 passage dropped); 8 technical docs; `evals/*.py` code. Split by group or file | **Yes** for content. Docs are project documentation of mixed authorship |
| Calibration | E54's 302 calibration items plus the calibration halves above: n = 1,155 (786 single-view, 369 multi-view) | **Both bins adequate** for α = 10% and 5% (n ≥ 10/α) |

Not available offline: any attack set outside the benchmarks PAIR was developed on, real long user or RAG documents,
human-written framing templates. **E55 is therefore filler-held-out and template-held-out, not
attack-family-held-out.**

## 2. Primary results at α = 10% (calibration-only thresholds)

Thresholds: V0 0.708; V1-global 0.840; V1-conditional single-view 0.730 / multi-view 0.936.

Realised calibration FPR with Clopper–Pearson intervals ([results/thresholds.json](results/thresholds.json)):

| Condition | Single-view bin | Multi-view bin |
| --- | --- | --- |
| V1-conditional | 9.9% [7.9, 12.2] | 9.8% [6.9, 13.3] |
| V1-global | 4.8% | 20.9% |

Evaluation results, all in % ([results/summary_table.md](results/summary_table.md); full metrics in
[results/metrics.json](results/metrics.json)):

| | V0 | V1-global | V1-conditional |
| --- | --- | --- | --- |
| Recall at pad-after 84 (n = 647) | 11.7 | 45.1 | **28.3** |
| Recall at pad-before 84 | 14.2 | 43.3 | 26.7 |
| Dose-response area (after) | 23.0 | 46.8 | 33.0 |
| Recall at ≥ 168 words | 2.9 | 41.8 | 23.5 |
| Unpadded recall, set B | 68.2 | 53.3 | 60.9 |
| Macro recall, 4 standard sets | 79.4 | 72.2 | **72.8** |
| Framed recall, medical/legal/compliance | 58.4 | 47.2 | **47.8** |
| Evasion, medical/legal/compliance | 19.7 | 16.2 | 25.8 |
| Recall after view-count manipulation | 55.5 | 50.8 | **31.2** |
| Short-benign FPR (n = 709) | 10.4 | 6.1 | 9.7 |
| Long-benign FPR (n = 149) | 8.7 | 18.1 | **8.1** |
| Filler-only FPR (n = 3,235) | 1.0 | 3.4 | 1.1 |
| Security-docs FPR (n = 27, secondary) | 7.4 | 11.1 | 3.7 |
| XSTest-safe / OR-Bench-hard (development) | 32.4 / 78.0 | 19.6 / 64.0 | 30.4 / 74.8 |

V1-conditional minus V0, paired 95% intervals (Estimate):

| Metric | Δ (points) |
| --- | --- |
| Pad-after 84 | +16.5 [+13.0, +20.1] |
| Recall ≥ 168 | +20.6 [+17.5, +23.7] |
| Macro recall | −6.6 [−8.3, −4.9] |
| Framed recall, medical/legal/compliance | −10.7 [−12.2, −9.2] |
| Manipulated recall | −24.3 [−27.4, −21.2] |
| Long-benign FPR | −0.7 [−6.2, +5.1] |
| Short-benign FPR | −0.7 [−1.4, −0.1] |
| Filler-only FPR | +0.1 [−0.5, +0.7] |

### Preregistered decisions (α = 10%; α = 5% gives the same verdicts)

| Criterion | Rule | Result |
| --- | --- | --- |
| H1 dilution | Pad-after 84 ≥ 0.80 × V0 unpadded recall on B (= **54.5%**), and lower bound > 0 | **Fail.** 28.3%; lower bound +13.0 > 0, so the gain is real but far below the target |
| H2 benign risk | (a) long − short ≤ 5; (b) long ≤ V0 + 5; (c) filler-only ≤ V0 + 5 | **Pass.** (a) −1.7 [−7.0, +4.6]; (b) −0.7; (c) +0.1 |
| H3 trade-off | Macro ≥ V0 − 2 | **Fail.** −6.6 |
| A6 new framing | Framed M/L/C ≥ V0 − 2 | **Fail.** −10.7 |
| A7 manipulation | Manipulated recall ≥ V0 − 2 | **Fail.** −24.3 |
| Bins adequate | n ≥ 10/α | Yes |
| **Acceptance** | All of the above | **Failed. H0 retained** |

## 3. Interpretation

1. **The E54 gain does not survive independent filler at a controlled operating point.**
   - Under E54's own reference rule on E55 data, conditional recall at pad-after 84 is 40.2%. That is already
     below E54's 55.8%, because realistic multi-sentence filler is harder than one repeated sentence.
   - At the preregistered 10% target it falls to 28.3%.
2. **The conditional rule does not improve the trade-off; it relocates it.**
   - V1-global keeps padding recall (45.1%) but raises long-benign FPR to 18.1%. Long minus short is +12.1
     [+5.0, +20.0].
   - V1-conditional controls long-benign FPR (8.1%) only by applying a 0.936 threshold to every multi-view input.
     That discards most of the padding gain (−16.8 [−19.8, −14.1] vs global).
3. **The conditional rule is cheaply exploitable (A7).**
   - A line break plus one neutral sentence makes any prompt multi-view.
   - V1-conditional recall on manipulated attacks is 31.2%, vs 60.9% unpadded and 55.5% for V0 on the same
     manipulated items.
   - An attacker controls the bin, so the bin cannot be a security boundary.
4. **Ordinary attacks pay too.**
   - 93% of JBB and 62% of StrongREJECT prompts are naturally multi-sentence and fall into the strict bin, so macro
     recall drops 6.6 points.
   - Suffix-framed attacks (63% multi-view) lose recall (42.0% vs V0 55.7% pooled suffix M/L/C).
5. **The ranking signal is real.**
   - Descriptive PR-AUC for padded attacks vs long benign and filler-only: V1_lse4 0.927 vs V0 0.820.
   - No threshold rule tested here turns that into an acceptable decision rule. Better PR-AUC is not a better
     decision rule.

## 4. Latency and cost (Observed)

[results/latency.json](results/latency.json): 30 items per bucket, no cache, machine-dependent.

| Tokens | V0 p50 / p95 | V1 p50 / p95 | V1 mean encoder calls |
| --- | --- | --- | --- |
| ≤ 64 | 15 / 21 ms | 30 / 51 | 2.2 |
| 65–256 | 42 / 78 | 130 / 226 | 8.7 |
| 257–512 | 80 / 102 | 306 / 352 | 18.7 |
| > 512 | 84 / 89 | 400 / 529 | 25.6 |

The conditional rule adds no scoring cost beyond V1.

## 5. Deviations, limitations and declared biases

- **Threshold rule.** As preregistered, the single-view threshold is calibrated on single-view items. E54's
  exploratory rule used V0's global threshold instead. The E54 rule is reported as `e54ref` and fails similarly.
- **Model-authored material.** Framing templates and manipulation sentences were written by the assistant. They
  are template-held-out, not human-authored.
- **Base attacks are development data** (HarmBench and SORRY-Bench). No attack-family- or source-held-out set
  exists offline.
- **Small long-benign units.** Evaluation has 149 items, but its bootstrap units are few: 7 code files,
  4 doc files, 82 MMLU groups. History subsets are small after dropping groups that share an E54 passage.
- **PAIR-isolated.** Nothing here measures the 12-layer pipeline. Calibration is not a distribution-free
  guarantee, because calibration and evaluation sources differ.
- **No deviation from the preregistered criteria, data or rules occurred.**

## 6. Recommendation

- **Stop C1 (segment-aware scoring).** E54 and E55 together show:
  - sentence isolation carries a real ranking signal for diluted attacks;
  - every calibrated rule tested either loses that signal or reintroduces long-benign false positives;
  - the conditional rule opens a trivial view-count evasion.

  Further pooling or threshold tuning is not recommended.
- **Next main research candidate:** C2, intent-contrastive and de-confounded training, with source-held-out
  evaluation as a standard metric (E53 A5 showed 15–32 points of hidden optimism).
  - The repeated root cause across E19, E53, E54 and E55 is that PAIR scores topical sentences ("war",
    "overdose", "crime") as attacks. That is a representation and label problem, not a pooling problem.
- **Requires separate approval.** Nothing from E55 should be integrated into the pipeline.

## 7. Reproduce

```bash
PY=/c/Users/ASUS/anaconda3/envs/failure-engine/python.exe; D=research/adversarial/e55_view_conditional_calibration
$PY -B -s $D/scripts/s01_build.py
$PY -B -s $D/scripts/s02_score.py --phase whole && $PY -B -s $D/scripts/s02_score.py --phase views   # ~21 min
$PY -B -s $D/scripts/s03_metrics.py
$PY -B -s $D/scripts/s04_latency.py
$PY -B -s $D/scripts/s05_integrity.py && $PY -B -s $D/scripts/s05_integrity.py checksums
```

Outputs are exclusive-create (a rerun needs a fresh directory). E53 and E54 are read and checksum-verified, never written.
