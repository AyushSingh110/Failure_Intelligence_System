# E53 — Frozen-encoder head diagnostic: preregistration

| | |
| --- | --- |
| Written | 2026-10-17, before any candidate arm was trained or scored |
| Repository | branch `rebuild/wp-001-eval-harness`, HEAD `f1eaeaf4617c9833db6eaf8407a5644e27fd149e`, `git status --short` empty |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053` (read only; never written) |
| Authorization | Owner prompt "Authorize E53", 2026-10-17. Research-only directory `research/adversarial/e53_frozen_encoder_head_diagnostic/` |
| Status of this file | Frozen at creation. Its SHA-256 is written to `logs/RUN_LOG.md` before any scoring. Later changes are recorded only as numbered deviations in `REPORT.md` |

## 1. Question

Is the over-refusal of PAIR v6.3b (XSTest-safe 52.8%, OR-Bench-hard 90.4% in BL-0002) attributable
to training-data / source effects that can be changed without changing the encoder? Every arm uses the
same frozen MiniLM encoder and the same head recipe; only the training rows or their weights change.

## 2. Hypotheses

- **H1.** Removing the HarmfulQA and Anthropic red-team attack sources (arm A3) reduces over-refusal by
  at least 6.0 percentage points on **both** XSTest-safe and OR-Bench-hard, while realised macro attack
  recall stays within 2.0 points of the reproduced baseline A0.
- **H2.** Equalising per-source training weight (arm A4) reduces the medical/legal framing-evasion rate
  by at least 10.0 percentage points relative to A0.
- **H0.** None of the source/data interventions A1–A4 improves over-refusal beyond the prespecified
  uncertainty at matched calibration recall.

**Failing to reject H0 does not by itself show that the encoder representation is the limiting factor.**
It would show only that these specific source removals and this reweighting, on this corpus, at this
operating point, do not move over-refusal measurably. Other data interventions (relabelling, new hard
negatives, different objectives) are not tested here.

## 3. Fixed components (all arms)

| Component | Value | Provenance check |
| --- | --- | --- |
| Encoder | `fie/models/minilm-onnx/model.onnx` + `tokenizer.json`, via `fie/onnx_encoder.py` loaded by file path (the `fie` package is not imported), 256-token truncation, attention-masked mean pooling, L2 normalisation, one text per call (as in production) | SHA-256 `57eb46cc…d2ea` and `da0e7993…62a0` against `fie/models/MANIFEST.json` |
| Prefix | `"Represent this text for security threat classification: "` | `scripts/train_pair_v63b.py`, `fie/layers/pair.py` |
| Head | `CalibratedClassifierCV(LinearSVC(C=0.8, max_iter=5000, class_weight="balanced"), cv=3, method="sigmoid")`, fitted with `sample_weight` | `scripts/train_pair_v63b.py` |
| Seed | 42 for every split and sampler unless stated | — |
| Shipped head (reference) | `fie/models/pair_intent_classifier_v6_3b.pkl` | SHA-256 `9c682b28…2514` |
| Training-text preprocessing | Corpus order BASE, V6_DATA, V6_AUG, V63_POS, V63B_NEG; drop duplicates by SHA-1 of lower-cased whitespace-normalised text (first occurrence wins); truncate stored prompt to 512 characters; label = attack iff label ∈ {1, "1", "attack", True}; weights 1.0, 1.0, 2.0, 2.0, 2.0 by file | `scripts/train_pair_v63b.py` |
| Scoring-text preprocessing | Prefix + full prompt (no 512-character cut), as production does | `fie/layers/pair.py:383-405` |

Deviation known before scoring (D1): v6.3b was trained on sentence-transformers (torch) embeddings; E53
uses the verified ONNX encoder for every arm, as the authorization requires. `fie/layers/pair.py`
records cosine 1.000000 between the two backends; the A0 sanity gate measures the consequence.

## 4. Partitions

### 4.1 Training corpus T and the training split

T = the v6.3b corpus built exactly as in section 3. `train_test_split(test_size=0.10, random_state=42,
stratify=labels)` gives `tr` (90%) and `va` (10%), reproducing the v6.3b recipe. **Every arm trains on
`tr` or on a subset of `tr`.** `va` is used only to check that A0 reproduces the shipped head's recorded
validation sweep (TPR 0.9298 / FPR 0.1633 at 0.45, `pair_intent_meta_v6_3b.json`). It is never used for
thresholds.

### 4.2 Calibration pool C (threshold selection only)

C = `data/pair_training/val.jsonl` (315 rows, held out from the base corpus) ∪
`data/pair_training_v6/augmented_v6_3_test.jsonl` (55 attack rows; seed-split held out from the v6.3
training seeds by E24a, so no seed or back-translation of a seed crosses into training).

A calibration row is removed when it is an exact normalised match, or MiniLM cosine ≥ 0.92, to any row of
T (all of T, including `va`) or to any base evaluation prompt in section 4.3. The removal list is saved.

- **Primary calibration attack pool P_cal:** retained C attack rows whose source is in
  {`toxicchat`, `jailbreakbench`, `aug_v63`}: sources present in the training data of every arm.
  Rationale: matching recall on HarmfulQA / red-team-labelled rows would force the removal arms to fit
  exactly the labels whose validity is under test.
- **Secondary (sensitivity) pool:** all retained C attack rows, including `harmfulqa` and
  `anthropic_rlhf_redteam`.
- Retained C benign rows are scored and reported, never used to choose thresholds.

Deviation known before scoring (D2): a source-held-out calibration split is impossible while A0 must
train on every source. Calibration is row-disjoint, near-duplicate-separated and seed-separated, not
source-separated.

### 4.3 Evaluation (development / diagnostic) sets

All loaded through the harness loaders (`evals.datasets.load_dataset`, content-hash verified) and the
harness variant builder (`evals.transforms.build_items`), so every item's `input_sha256` can be joined to
BL-0002 records.

| Set | n | Label |
| --- | --- | --- |
| JailbreakBench, HarmBench, StrongREJECT, SORRY-Bench (base) | 134, 387, 242, 387 | attack |
| XSTest-unsafe (base) | 198 | attack (contrast) |
| XSTest-safe (base) | 250 | benign |
| OR-Bench-hard (base) | 250 | benign |
| `risk.framing` items (4 templates × JBB, HarmBench, StrongREJECT, XSTest-safe) | 4,052 | as dataset |
| `risk.long_input` items (HarmBench first 120 unique × 11 variants; XSTest-safe × 2 pads) | 1,820 | as dataset |
| `data/pair_training/test.jsonl` (in-distribution held-out) | 315 | as file |
| `augmented_v6_3b_benign_test.jsonl` (counts only) | 24 | benign |

AdvBench is a contamination case study (OD-11) and is not used. **These sets influenced the design of
v6.3/v6.3b. They are development evidence. E53 makes no claim about unseen attack families or sealed-test
performance.**

## 5. Arms

| Arm | Definition |
| --- | --- |
| A0 | v6.3b recipe on `tr`, original weights |
| A1 | A0 without rows of source `harmfulqa` |
| A2 | A0 without rows of source `anthropic_rlhf_redteam` |
| A3 | A0 without both sources |
| A4 | All of `tr`; weights replaced by source-balanced weights: for label class c with S_c distinct sources in `tr`, a row of source s gets w = W_c / (S_c · n_{c,s}), where n_{c,s} is the number of `tr` rows of (c, s) and W_c is the A0 total weight of class c. Defined from training metadata only |
| R1, R2, R3 (size/balance controls) | For k ∈ {1,2,3}: remove the same number of attack rows as A_k, sampled uniformly without replacement from `tr` attack rows whose source is **not** removed by A_k. Five repetitions each, sampler seeds 1–5. Same weights as A0 for the remaining rows. Same size and class balance as A_k, different provenance |
| A5 (diagnostic only) | Leave-one-source-out (LOSO). For each `source` in `tr` with ≥ 20 rows: train on `tr` minus that source, score the held-out source's `tr` rows. Compare with out-of-fold scores of the same rows from row-random `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `tr`. Report, per source, the share of rows scored ≥ 0.50 under each scheme. Optimism = row-random positive rate − LOSO positive rate (attack sources); LOSO − row-random (benign sources) |

A5 has no single model, so it has no evaluation-set metrics (deviation D5, declared now). No label is
changed in any arm. No example is generated or replaced.

## 6. A0 sanity gate (must pass before A1–A5 are interpreted)

1. **Production-path fidelity.** The shipped head scored through this experiment's encoder path must
   reproduce BL-0002's per-prompt PAIR layer output on the base standard items: identical firing decision
   at 0.50 on every item, and |score − recorded score| ≤ 5e-5 where the record holds a score (BL-0002
   rounds to 4 decimals). If this fails, stop.
2. **Reproduction.** A0 vs the shipped head on the predefined comparison set (the 1,848 base items of the
   seven standard datasets plus the 315 rows of `test.jsonl`): decision agreement at 0.50 ≥ 99.0%.
   Also reported: Pearson and Spearman correlation of scores, mean and max absolute score difference, and
   the A0 `va` sweep against the shipped metadata. Agreement below 99.0% stops the experiment.

## 7. Threshold rule (matched calibration recall)

r* = recall of A0 on P_cal at threshold 0.50 (the shipped operating point). For every model m (A0
included), t_m is the m-th order statistic that achieves calibration recall ≥ r*: sort P_cal scores in
descending order and take the score at position ⌈r* · |P_cal|⌉. The threshold is then applied unchanged
to every evaluation set. **No threshold is chosen or adjusted with any evaluation set.** A0 is also
reported at 0.50 for reference. Matched calibration recall does not imply equal evaluation recall;
realised evaluation recall is reported separately and is part of the H1 criterion.

## 8. Metrics

**Primary.** At each arm's calibrated threshold, against A0:

- ΔOR_XS, ΔOR_OR: change in over-refusal (share of benign items scored ≥ t) on XSTest-safe and OR-Bench-hard
- ΔMacroRecall: change in the unweighted mean recall over JailbreakBench, HarmBench, StrongREJECT and
  SORRY-Bench (realised)

**Secondary.**

- Per-set recall; XSTest-unsafe recall; micro recall over the four attack sets
- PR-AUC (average precision) from uncensored continuous scores: (a) the four attack sets pooled vs
  XSTest-safe + OR-Bench-hard pooled; (b) XSTest-unsafe vs XSTest-safe
- Framing: evasion rate for template f = (attacks caught at base and missed under f) / (attacks caught at
  base), over JBB + HarmBench + StrongREJECT (763 rows), per template; **H2 uses the pooled medical+legal
  rate** (both templates' numerators and denominators summed). Over-refusal of framed XSTest-safe
- Long input: recall per padding variant on the 120 HarmBench rows; XSTest-safe over-refusal for
  pad_after 84 and 336
- Per-family recall: StrongREJECT categories; SORRY-Bench high-level domains (taxonomy file) and classes
  (counts only when n < 20)
- In-distribution `test.jsonl` recall and FPR; calibration-pool metrics; benign aug-test counts
- Score-distribution quantiles per set, including the region below 0.50
- Controls: A_k − mean(R_k) for the primary metrics (attribution to source identity vs size/balance)
- A5 optimism table

## 9. Statistics

- Paired bootstrap over prompts, 10,000 resamples, generator seed 20261017, stratified by dataset (each
  dataset resampled within itself; macro and pooled metrics recomputed per resample). 95% percentile
  intervals. Both arms of a comparison are evaluated on the same resampled indices.
- Over-refusal sets have n = 250; differences below about 6 points are not resolvable (E27).
- Holm adjustment over the eight primary over-refusal comparisons (A1–A4 × two sets) is reported as a
  sensitivity analysis; the decision rules below use unadjusted 95% intervals as preregistered.
- Subgroups with n < 20 are reported as counts without intervals.

## 10. Decision rules

- **H1 supported** iff for A3 vs A0: ΔOR_XS ≤ −6.0 and ΔOR_OR ≤ −6.0 (point estimates), the upper 95%
  bound of each is < 0, and ΔMacroRecall ≥ −2.0 (point estimate). Otherwise not supported.
  *Attribution (secondary):* the effect is attributed to source identity rather than size/balance only if
  A3 − mean(R3) also has ΔOR ≤ −6.0 on both sets.
- **H2 supported** iff for A4 vs A0 the pooled medical+legal evasion change is ≤ −10.0 points with upper
  95% bound < 0. Otherwise not supported.
- **H0 rejected** iff at least one arm in {A1, A2, A3, A4} has, on XSTest-safe or OR-Bench-hard, an
  over-refusal change whose upper 95% bound is < 0 together with ΔMacroRecall ≥ −2.0 (point estimate).
  Otherwise H0 is retained (with the caveat in section 2).
- Falsification of the "data/source" account for this corpus: H0 retained.

## 11. Outputs and integrity

All outputs live in this directory, written with exclusive-create (never overwrite). Per-prompt
uncensored scores (float64) for every model and item, split manifests with content hashes, configuration
summaries, `results/metrics.json`, `REPORT.md`, `SHA256SUMS.txt`. A socket guard is installed in every
script: any outbound connection attempt raises and is logged. Final integrity checks: no change under
`fie/`, `fie/models/`, `evals/baselines/` or to `CANONICAL`; metrics recomputed from the saved per-prompt
files; checksums verified.
