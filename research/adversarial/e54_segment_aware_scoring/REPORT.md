# E54 — Segment-aware multi-view scoring diagnostic: report

| | |
| --- | --- |
| Date | 2026-10-17 |
| Repository | `rebuild/wp-001-eval-harness` @ `700db5f` (E53 commit); protected paths unchanged; outputs only in this directory |
| Preregistration | [PREREGISTRATION.md](PREREGISTRATION.md), SHA-256 `5aa0d370…2024`, logged before any scoring ([logs/RUN_LOG.md](logs/RUN_LOG.md)). Authored fixtures hashed and logged before it |
| Frozen model | PAIR v6.3b head `9c682b28…` + ONNX MiniLM `57eb46cc…`; nothing trained |
| Fidelity gate | **Passed.** V0 vs BL-0002: 0 decision mismatches on 1,848 items, max \|Δ\| 5.0e-5 (record rounding). V0 vs E53 `SHIPPED`: max \|Δ\| 5.0e-16 over 7,720 items ([results/fidelity_gate.json](results/fidelity_gate.json)) |
| Outcome | **H1, H2, H3 each not supported by any candidate. No candidate achieves the useful trade-off. H0 retained.** |

Labels: **Observed** = measured; **Estimate** = with paired-bootstrap 95% interval; **Interpretation**;
**Exploratory** = after the decisions, not preregistered, not a decision input.

## 1. What was measured

- **Scope.** Frozen PAIR v6.3b only (not the 12-layer pipeline), evaluated as 10 scorers:
  - V0 = whole input (production behaviour);
  - V1 = whole + sentence/clause segments;
  - V2 = whole + 64- and 128-token windows with stride 32;
  - V3 = all views;
  - each of V1–V3 pooled by max, top-3 mean and log-sum-exp (β = 4).
- **Volume.** 8,467 items, 85,031 views, 20,147 unique texts scored. Per-prompt records keep every view's kind,
  character span, token range and score: [records/per_prompt.jsonl](records/per_prompt.jsonl).
- **Two operating points:**
  - **fixed 0.50** (diagnostic, exposes pooling inflation);
  - **calibrated**: each scorer's threshold matches V0's calibration-benign FPR at 0.50, τ = 19.9% on 302
    calibration prompts (139 short, 151 MMLU-long, 12 authored). No evaluation outcome was used.
- **Data.** All evaluation sets except MMLU, the authored fixtures and the repository docs are development data
  that shaped v6.3b. MMLU-long, the authored fixtures, the repository docs and `test.jsonl` benign have no
  exact overlap with the PAIR training corpus (checked).

## 2. Primary results (calibrated operating point, Observed)

All values in %. Source: [results/metrics.json](results/metrics.json); full table:
[results/summary_table.md](results/summary_table.md).

| Scorer | t | Pad-after 84 | DRA after | Med+legal framed recall | Med+legal evasion | OR XSTest | OR OR-Bench | Long-benign FPR | Short-benign FPR | Macro recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | 0.493 | 2.5 | 25.3 | 68.2 | 23.0 | 53.6 | 90.8 | 21.7 | 11.6 | 88.4 |
| V1_max | 0.895 | 60.8 | 61.0 | 63.3 | 0.7 | 13.2 | 52.4 | 28.0 | 0.0 | 66.6 |
| V1_top3 | 0.759 | 0.0 | 13.1 | 27.8 | 61.8 | 27.6 | 72.8 | 29.8 | 1.4 | 72.9 |
| **V1_lse4** | 0.833 | **65.0** | **66.4** | 67.4 | 4.8 | 19.6 | 64.4 | 27.3 | 0.0 | 72.8 |
| V2_max | 0.788 | 1.7 | 16.0 | 48.0 | 34.6 | 24.0 | 68.8 | 36.0 | 1.4 | 73.8 |
| V2_top3 | 0.693 | 1.7 | 17.9 | 55.8 | 30.0 | 33.2 | 78.4 | 34.2 | 2.7 | 79.9 |
| V2_lse4 | 0.682 | 1.7 | 19.9 | 57.0 | 29.1 | 34.0 | 78.8 | 35.4 | 3.4 | 80.7 |
| V3_max | 0.901 | 60.8 | 61.0 | 62.3 | 0.8 | 12.8 | 51.6 | 32.9 | 0.0 | 65.7 |
| V3_top3 | 0.812 | 1.7 | 13.9 | 26.5 | 61.1 | 20.4 | 65.6 | 36.0 | 0.0 | 67.9 |
| V3_lse4 | 0.841 | 60.8 | 61.9 | 66.0 | 5.7 | 19.6 | 64.0 | 31.7 | 0.0 | 72.2 |

Paired differences against V0 (Estimate) for the strongest candidate, V1_lse4:

| Metric | Δ (points) |
| --- | --- |
| Pad-after 84 recall | +62.5 [+53.3, +70.8] |
| Pad-before 84 recall | +54.2 [+45.0, +62.5] |
| Med+legal evasion | −18.1 [−20.8, −15.5] |
| Med+legal framed recall | −0.9 [−2.8, +1.0] |
| Long-benign FPR | +5.6 [−3.1, +14.5] |
| Macro recall | −15.6 [−17.6, −13.6] |

The other candidates are in `comparisons`.

### Hypothesis outcomes (preregistered §8)

| | Rule | Result |
| --- | --- | --- |
| H1 | Pad-after-84 recall ≥ 0.80 × V0 unpadded recall = **71.3%** | **Not supported.** Best is 65.0% (V1_lse4) |
| H2 | Med+legal framed recall ≥ V0 + 5 with lower bound > 0 | **Not supported.** Every candidate is lower than or equal to V0 |
| H3 | (a) fixed-0.50 long − short ≤ 5; (b) long FPR ≤ V0 + 5; (c) hard-benign ≤ V0 + 2 | **Not supported.** (a) fails for every candidate (+37 to +65 points), and **V0 itself fails (a): +10.2 [+1.0, +19.6]** |
| Useful trade-off | H1 ∧ H3 ∧ macro recall ≥ V0 − 2 | None |
| **H0** | — | **Retained** |
| Criterion 4 | Is recall zero at ≥ 168 words? | V0: yes, on both sides. V1/V3 max and lse: no (56–65% at the calibrated point, 81–88% at 0.50). V2: about 1–9% |

## 3. Why it failed (Interpretation, supported by the Observed numbers)

1. **One global threshold cannot serve single-view and multi-view inputs.**
   - Pooling over many views inflates scores on long benign text: median V1_max − V0 = +0.46 on MMLU-long and
     +0.32 on docs.
   - Matching the pooled calibration FPR therefore pushes the threshold to 0.83–0.90.
   - Short prompts have a single view, so their scores are identical to V0's. At that threshold they lose recall
     (macro −16 to −22 points), and they "gain" the large XSTest and OR-Bench reductions (−34 to −40 points).
   - **The hard-benign "improvements" in the table are therefore V0 at a higher threshold, not an effect of
     segmentation.**
2. **Union inflation on long benign text is severe.** At the fixed 0.50 threshold:
   - V1_max flags 78.9% of long-benign primary documents (V0 21.7%) and 79.6% of the security-research docs
     (V0 37.0%);
   - MMLU history passages are flagged at 78–96%, because one sentence about a massacre or a war is enough.

   Long-benign FPR rises with the number of views (V1_max at 0.50: 75% with 6–15 views, 96% with 16–40).
3. **Segmentation does carry separating signal on long inputs** (descriptive, threshold-free curves):
   - padded HarmBench vs long benign: PR-AUC rises from 0.822 (V0) to 0.947 (V1_lse4);
   - recall at 10% long-benign FPR rises from 3.1% to 41.7%.

   So the method is not empty; the failure is mainly in the decision layer and in benign topical sentences.
4. **Token windows do not fix dilution.** A short request inside a 64-token window is still diluted by
   surrounding filler: V2 recall at pad 84 is 1.7%. Sentence isolation is what works, and that is the
   component helped by the declared template bias (§5).
5. **Framing:** segmentation removes medical/legal evasion (V1_max 0.7% vs 23.0%). At the calibrated
   threshold, however, framed recall does not improve, because the threshold rose.

## 4. Exploratory follow-up (not preregistered, not a decision input)

[results/exploratory_view_conditional_thresholds.json](results/exploratory_view_conditional_thresholds.json)
uses two thresholds, both taken from calibration data only:

- single-view inputs use V0's threshold;
- multi-view inputs use a threshold matching V0's FPR on the multi-view calibration benign set.

Results for V1_lse4 under that scheme:

| Metric | V1_lse4 | V0 |
| --- | --- | --- |
| Pad-after 84 recall | 55.8% | 2.5% |
| Long-benign FPR | 22.4% | 21.7% |
| Docs FPR | 18.5% | 37.0% |
| XSTest over-refusal | 53.6% | 53.6% (unchanged) |
| OR-Bench over-refusal | 90.4% | 90.8% (unchanged) |
| Macro recall | 82.9% | 88.4% |
| Med+legal framed recall | 58.5% | 68.2% |

- This is the only configuration seen that keeps long-benign FPR flat while recovering most padded attacks.
- It still misses the H1 bar.
- It loses recall on multi-sentence attacks and framed prompts, which fall in the multi-view regime.
- It was chosen after seeing results, so it is a hypothesis for the next experiment, not a finding.

## 5. Deviations and declared biases

| # | Item | Effect |
| --- | --- | --- |
| B1 (declared before scoring) | The segmenter splits at `:` and sentence ends, and I had seen the framing templates and the filler | V1/V3 framing and padding results are optimistic. Real attacks need not be cleanly separable sentences |
| D1 | Exploratory view-conditional analysis added after the decisions | Labelled; not used for H1–H3/H0 |
| D2 | The latency bucket >512 tokens has only 8 items (not 30) | Wide latency estimate for that bucket |
| D3 | Docs bootstrap units are files (7) | Wide docs intervals |
| D4 | Authored long-benign set is a 12/12 pilot written by the assistant | Per-category FPRs are counts, not rates |
| D5 | The exploratory script first crashed on V0 (no multi-view rows); fixed, no output had been written | None |

## 6. Latency and cost (Observed)

From [results/latency.json](results/latency.json): one process, ONNX one text per call, no cache; times are
machine-dependent.

| Tokens | V0 p50 / p95 | V1 p50 / p95 | V2 p50 / p95 | V3 p50 / p95 | V3 mean encoder calls |
| --- | --- | --- | --- | --- | --- |
| ≤ 64 | 12 / 21 ms | 31 / 53 | 13 / 23 | 31 / 57 | 2.5 |
| 65–256 | 42 / 79 | 67 / 166 | 165 / 423 | 201 / 462 | 9.1 |
| 257–512 | 80 / 84 | 224 / 338 | 770 / 958 | 976 / 1012 | 36.0 |
| > 512 (n=8) | 81 / 84 | 201 / 483 | 1105 / 1802 | 1232 / 2170 | 50.6 |

Total encoder calls over all items: V0 8,467; V1 35,347; V2 44,767; V3 71,646. V1 is the only variant with
modest cost (about 2–3×).

## 7. Limitations

- **PAIR-isolated.** Nothing here says how the 12-layer pipeline would behave.
- **Most evaluation data is development data.** The padding and framing constructions are fixed harness
  fixtures (one filler sentence, four templates), not adaptive or natural attacks.
- **Long-benign coverage is narrow.** MMLU exam passages, 12 assistant-authored documents and 54 repository
  chunks (whose "benign" label is contestable). No real user documents, RAG passages or multilingual long text.
- **Calibration is modest (n = 302).** The FPR target carries about ±4 points of sampling error. There is no
  risk-control guarantee, and none would hold under shift.
- **Nine candidates were compared without multiplicity correction.** Any future "pass" would need
  confirmation on sealed data.
- **Scores are not calibrated probabilities on these distributions**, and a high segment score does not show
  malicious intent. MMLU war passages score above 0.9.

## 8. Decision and recommendation

- **Do not integrate any E54 scorer.** None meets the preregistered trade-off.
- **Retain as evidence:**
  - sentence-level isolation (V1, log-sum-exp) is the only cheap view type that resists dilution;
  - token windows (V2/V3) add cost without benefit;
  - top-3 pooling is harmful.
- **Hypothesis for the next experiment (owner approval required): E55, view-conditional calibration of V1_lse4.**
  - Separate operating points for single-view and multi-view inputs.
  - A larger, source-diverse long-benign calibration set built before scoring.
  - Preregistered on fresh long-benign and long-attack constructions that do not share E54's filler or
    templates, to remove bias B1.
  - The open problem it must address: benign long text containing topical violence (history, law, medicine)
    scores high at the sentence level. That is the PAIR topic-sensitivity problem again (E19, E53), so a
    lasting fix probably needs the representation work (audit C2 / E40), which is not authorized here.

## 9. Reproduce

```bash
PY=/c/Users/ASUS/anaconda3/envs/failure-engine/python.exe; D=research/adversarial/e54_segment_aware_scoring
$PY -B -s $D/scripts/s01_build.py
$PY -B -s $D/scripts/s02_score.py --phase whole && $PY -B -s $D/scripts/s02_score.py --phase views   # ~8 min
$PY -B -s $D/scripts/s03_metrics.py
$PY -B -s $D/scripts/s06_exploratory.py      # exploratory
$PY -B -s $D/scripts/s04_latency.py
$PY -B -s $D/scripts/s05_integrity.py && $PY -B -s $D/scripts/s05_integrity.py checksums
```

Outputs are exclusive-create, so a rerun needs a fresh directory. E53 artifacts are read and verified by
checksum, never written.
