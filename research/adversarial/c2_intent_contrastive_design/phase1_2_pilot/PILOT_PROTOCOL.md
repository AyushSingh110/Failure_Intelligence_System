# C2 Phase 1.2 — 200-row taxonomy pilot: protocol

| | |
| --- | --- |
| Date prepared | 2026-10-18 |
| Repository | `rebuild/wp-001-eval-harness` @ `fa2fffc` (Phase 1.1 commit). Protected paths unchanged |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053` (unchanged; shipped head `9c682b28…`, encoder `57eb46cc…`) |
| Status | **Package ready for human annotation. The pilot has NOT started. No human judgment exists. No acceptance criterion has been evaluated.** |
| Inputs | Taxonomy: `C2_PHASE1_DESIGN.md` §3. Rubric: `C2_PHASE1_1_HUMAN_REVIEW_PROTOCOL.md` §1–4. Decisions: `C2_PHASE1_1_OWNER_DECISION_GATE.md` (sections OD-C2-1…8, mirroring §10 of the design; the gate file itself has no "section 10") |

## 1. Owner decisions

Nothing below is approved. Recommendations stay recommendations.

**Blocking the pilot:**

| Decision | What the pilot needs |
| --- | --- |
| **OD-C2-4 (reviewer part)** | Name two independent reviewers and one adjudicator. Without them the pilot cannot start |
| **Approval to run the pilot itself** | This package has been prepared, not authorised for use |

**Not blocking the pilot** (they shape what happens after it):

- **OD-C2-2 (refusals):** the pilot only *confirms* whether the 10 sampled refusal-cue rows are user inputs.
  No row receives a training label from the pilot.
- **OD-C2-3 (dual-use):** reviewers mark `dual_use` explicitly. It is kept separate and never mapped to
  attack or benign in the pilot.
- **Later stages:** OD-C2-1 (results retention), OD-C2-4 drafting part and OD-C2-4b (no harmful-side
  generation, which remains in force), OD-C2-5 (final test, frozen before any training), OD-C2-6 (dedup
  encoder), OD-C2-7 (heads), OD-C2-8 (no prompt-injection claims).

**Standing constraints carried into the pilot:**

- the 82 refusal rows are **not** given attack labels;
- dual-use stays separately identifiable;
- no example generation of any kind;
- no model output is ever used as an annotation;
- no performance claim, prompt-injection or otherwise.

## 2. Assignments (to be filled by the owner)

| Role | Pseudonymous ID | Person (held by the owner, not written here) | Confirmed independent of the other reviewer |
| --- | --- | --- | --- |
| Reviewer A | `R1` | ____________ | ☐ |
| Reviewer B | `R2` | ____________ | ☐ |
| Adjudicator | `ADJ` | ____________ (must not be R1 or R2) | ☐ |

## 3. Sample (deterministic; no model prediction used)

Script: [scripts/build_pilot_sample.py](scripts/build_pilot_sample.py), seed `20261018`. Full manifest:
[sampling_manifest.json](sampling_manifest.json).

**Construction:**

1. Rebuild the v6.3b corpus (4,236 rows) from the five original files, in recipe order and with recipe
   dedup.
2. Verify each row against E53's `train_corpus.jsonl` (512-character text hash, source, label).
3. Show reviewers the **full original text**, not the 512-character training cut. 12 sampled rows were
   truncated in training.

**Strata.** 200 unique rows, 18 sources, no shortages:

| Stratum | Rows | Purpose |
| --- | --- | --- |
| S1 cross-label conflict pairs | 10 (5 pairs, cosine 0.92–0.95) | Both rows kept **intentionally**: near-identical texts with opposite labels |
| S2 red-team first turns | 40 (20 with harm vocabulary, 20 without) | Labels may not match content |
| S3 HarmfulQA | 40 (15 with harm vocabulary, 25 without) | Suspected mixed benign questions |
| S4 ToxicChat | 30 (15 `jailbreak`, 15 `toxic_query`) | Manipulation candidates; toxic-only cases |
| S5 Llama paraphrase | 20 (10 refusal-cue, 10 other) | Refusals labelled attack |
| S6 retained attack sources | 25 (AdvBench mirror 6, forbidden_qs 6, MaliciousInstruct 4, JBB behaviours 3, aug_v63 seeds 6) | Rubric check on "clean" positives; 10% audit pilot |
| S7 retained benign sources | 35 (alpaca 4 + 2, hh-helpful 7, law_stackexchange 6, MedQuad 3, medqa 3, mbpp 2, hand-written 4, aug_v63b 4); harm-vocabulary quotas where possible | Hard and benign registers; likely dual-use |

**Facts about the sample:**

- **Original labels are 160 attack / 40 benign.** This skew is a deliberate consequence of oversampling
  review-required sources, and reviewers never see it.
- 66 rows carry a harm-vocabulary cue; 10 carry a refusal cue.
- **Near-duplicates:** no selected row has cosine ≥ 0.95 to another selected row, except inside S1 pairs
  (intentional). No source row appears twice except the preregistered repeats (§4).

**Limitations:**

- Not representative of the corpus.
- Cues are regex heuristics.
- Dual-use rows cannot be identified before review.
- Only 15 manipulation candidates exist in the sample (71 in the whole corpus).
- hh-rlhf rows are first human turns whose later context is not available locally.

## 4. Packets and blinding

**Packets:**

- [reviewer_packets/reviewer_a.csv](reviewer_packets/reviewer_a.csv) and
  [reviewer_packets/reviewer_b.csv](reviewer_packets/reviewer_b.csv).
- Each has 220 items: the 200 originals plus 20 **preregistered hidden repeats**. Each repeat has a separate
  pilot id, and in **both** packets every repeat sits at least 50 positions from its original.
- Orders are independent seeded permutations.
- Columns: `position`, `pilot_id`, `text`, then blank annotation fields.
- No label, source, category, score or other reviewer's answer appears.
- UTF-8 with BOM, CRLF line ends (opens directly in Excel / LibreOffice).

**Owner only:** [owner_only/pilot_mapping.json](owner_only/pilot_mapping.json) maps each pilot id to its
original file and line, text SHA-256, training-text hash, original label, source, stratum, repeat link and
positions. **Never send this file to reviewers.**

**Reviewers receive** their own packet, [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) and
[annotation_schema.json](annotation_schema.json). **They do not receive** the other packet, this protocol,
the manifests or anything under `owner_only/`.

## 5. Procedure and preregistered analysis

1. **Briefing.** The owner briefs R1 and R2 together on the guide only (≈ 30 minutes). No items are
   discussed.
2. **Independent annotation.** Each reviewer fills every row of their own packet, without contact, and
   returns it as `annotations_returned/reviewer_a_returned.csv` / `reviewer_b_returned.csv`. Locked columns
   are unchanged. The returned files are never edited afterwards; corrections are new files with a suffix.
3. **Validation and metrics:** `python scripts/compute_pilot_agreement.py`. It validates the schema, the
   locked text and the consistency rules, then writes `results/pilot_agreement.json` and
   `results/adjudication_queue.csv` (exclusive-create):
   - **Agreement:** Cohen's κ on the 200 originals (repeat copies excluded) for H (4 values including
     `unclear`) and M (3 values).
   - **Uncertainty:** seeded bootstrap 95% CIs (2,000 row resamples), confusion matrices, percent agreement,
     PABAK and prevalence, all reported.
   - **Imbalance rule (fixed now):** if for either reviewer the least frequent non-`unclear` value of a
     dimension has fewer than 10 of 200 rows, κ is reported but marked *not evaluable*. That dimension's
     criterion becomes percent agreement ≥ 90%, with the confusion matrix and raw disagreement counts.
     M is expected to fall under this rule (about 15 manipulation candidates).
   - **Self-consistency:** per reviewer, the share of the 20 repeat pairs with identical (M, H).
   - **Unclear rate:** per reviewer, per dimension, over the 200 originals.
4. **Adjudication.** `ADJ` fills the adjudication queue: `final_*` fields plus a one-line rationale, without
   changing reviewer files. A row enters the queue for any of these reasons:
   - M or H differs between reviewers;
   - any `unclear`, `dual_use` or `Q = low`;
   - M = yes together with H = harmful;
   - every S1 conflict row (10 known now);
   - every red-team row marked H = `none` (count unknown until review);
   - every refusal-cue row (10 known now).
5. **Revision.** Only after the outcomes are reviewed, any guide change is written as a new version
   (`REVIEWER_GUIDE_v2.md`) with a change log, **before** any further annotation.

**Acceptance criteria** (all must hold; none evaluated yet):

| Criterion | Threshold |
| --- | --- |
| H agreement | κ ≥ 0.60, or percent agreement ≥ 90% if κ is not evaluable |
| M agreement | κ ≥ 0.60, or percent agreement ≥ 90% if κ is not evaluable |
| Self-consistency | ≥ 90% for each reviewer (≥ 18 of 20 repeat pairs) |
| Unclear rate | ≤ 15% for H and for M, for each reviewer |

**If the pilot fails:** revise the guide and run a **separately identified** pilot (`phase1_2_pilot_r2`) on a
new seeded sample that excludes these 200 rows. **Do not proceed to full review or training automatically.**

**If the pilot passes:** report to the owner. Full review (S2) needs its own approval.

## 6. Workload (Estimates; nothing measured)

- **Packet size:** 220 items per reviewer, 5,608 words in total; median 12 words, 90th percentile 38, max
  280.
- **Annotation:** at ≈ 1–1.5 minutes per item, ≈ 3.7–5.5 hours per reviewer, so ≈ 7.5–11 reviewer-hours in
  total.
- **Adjudication:** the queue is expected to hold at least the 20 known escalations (10 conflict + 10
  refusal-cue rows) plus all disagreements and unclear rows. If 20–35% of the 200 rows need adjudication at
  ≈ 2 minutes each, that is ≈ 1.5–2.5 hours.
- **Briefing and setup:** ≈ 1 hour per person.
- **Total:** ≈ 10–15 person-hours. Each reviewer records actual time per batch of 50 so that S2 can be
  re-estimated from measurements.

## 7. Task checklists

**Owner:**

- ☐ Approve the pilot.
- ☐ Name R1, R2 and ADJ.
- ☐ Send each reviewer only their own packet, the guide and the schema.
- ☐ Keep `owner_only/` private.
- ☐ Receive returned files into `annotations_returned/`.
- ☐ Run the agreement script.
- ☐ Hand the queue to ADJ.
- ☐ Record decisions.

**Reviewers R1 and R2:**

- ☐ Read the guide.
- ☐ Annotate every row in your own packet.
- ☐ Do not edit locked columns.
- ☐ Do not consult the other reviewer, the web or any AI tool.
- ☐ Record time per 50 rows.
- ☐ Return the file under the agreed name.

**Adjudicator:**

- ☐ Read the guide.
- ☐ Complete `final_*` and `rationale` for every queued row.
- ☐ Mark unresolved rows `final_Q = low`.
- ☐ Do not edit reviewer files.
