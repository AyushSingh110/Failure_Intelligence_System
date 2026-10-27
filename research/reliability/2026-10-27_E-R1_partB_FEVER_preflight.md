# E-R1 Part B — FEVER Licence, Artifact and Evidence-Materialisation Preflight

**Date:** 2026-10-27 · **Status:** preflight only, for owner review.
**Nothing has been downloaded or fetched, and no network lookup was made.** The stage forbids network access, so every external fact below comes from prior knowledge and is marked **[VERIFY]**. Each must be confirmed against the official source as the *first* step of the authorised download, before any data is used.
**Decision this informs:** OD-R4 (FEVER as the primary dataset). See the [E-R1 plan](2026-10-27_E-R1_verifier_benchmark_plan.md) §3.

## 1. Artifact

| Item | Expected value | Confidence |
|---|---|---|
| Dataset | FEVER 1.0, *Fact Extraction and VERification*. Thorne, Vlachos, Christodoulopoulos and Mittal, NAACL 2018 | high |
| Official source | fever.ai dataset page, linking the claim files and the Wikipedia dump [VERIFY exact URLs and whether they still resolve] | medium |
| Claim files | `train.jsonl` (≈145k claims); `shared_task_dev.jsonl` (≈20k, labelled); `shared_task_test.jsonl` (≈20k, **unlabelled**); `paper_dev.jsonl` and `paper_test.jsonl` (≈10k each, labelled; together they equal shared_task_dev) [VERIFY counts] | medium–high |
| Claim record | `id`, `verifiable`, `label` ∈ {SUPPORTS, REFUTES, NOT ENOUGH INFO}, `claim`, `evidence` | high |
| Evidence field | A list of alternative **evidence sets**. Each item is `[annotation_id, evidence_id, wikipedia_page_title, sentence_index]`. **Identifiers only, no text.** For NOT ENOUGH INFO the page and sentence are `null` | high |
| Wikipedia snapshot | `wiki-pages.zip`: the June 2017 dump, introductory sections only, about 5.4M pages. Stored as ~109 JSONL shards with `id`, `text`, and `lines` (one tab-separated, indexed sentence per line) [VERIFY] | medium–high |
| Revision pinning | No version tag beyond "1.0"; pin by the **SHA-256 of every downloaded file**, recorded at download time | — |

## 2. Licence and attribution [VERIFY all]

- **Annotations:** to my knowledge released under **CC BY-SA 3.0**, because claims and evidence derive from Wikipedia.
- **Wikipedia dump:** CC BY-SA 3.0 (and GFDL), like Wikipedia text of that period.
- **Obligations if FIE keeps or publishes derived text:**
  - attribute FEVER and Wikipedia;
  - cite Thorne et al. (2018);
  - release any redistributed derived text under the same licence.
- **Effect on OD-R8:**
  - Raw and derived text stays local and is git-ignored.
  - Only metadata is tracked: the licence record, manifest, hashes, counts and aggregate results.
  - Short quoted examples in a report need attribution.
- **Stop rule.** The first download step saves the licence text shown on the official page into `research/reliability/data/fever/LICENSE_AS_DOWNLOADED.txt`, and its hash goes into the manifest. If it differs from the above, stop and report.

## 3. Does FEVER contain evidence text?

**No.** Gold evidence is a set of page-and-sentence pointers.
- **Materialising the text needs the 2017 Wikipedia dump:** the same snapshot the annotators used. A current Wikipedia page would silently change the evidence.
- **Live Wikipedia API:** ruled out. It is a different revision, adds network calls per item, and is not reproducible.
- **Third-party "gold-evidence" re-packagings** (for example community datasets on the Hugging Face Hub): not recommended as primary. Their faithfulness to the official dump cannot be checked without downloading the dump anyway, and their licence and provenance differ.

## 4. Size and disk [VERIFY]

| File | Expected size (from memory) |
|---|---|
| `paper_dev.jsonl` + `paper_test.jsonl` | a few MB in total |
| `train.jsonl` | ≈ 30–40 MB; **not needed**, because nothing is trained |
| `wiki-pages.zip` | ≈ 1.5–2 GB compressed; several GB if fully extracted |

**Planned peak disk:**
- ≈ 2 GB for the zip;
- under 50 MB of extracted sentences for the sample;
- no full extraction.

The zip can be deleted after extraction once its hash is recorded. A re-run then re-downloads and re-verifies it against that hash. Free disk is checked before download.

## 5. Reproducible 1,200-item sample without a large corpus

Only the two small claim files and one pass over the zip are needed.

1. **Claims.** Download `paper_dev.jsonl` and `paper_test.jsonl`, and record their hashes.
2. **Split.** The official split is reused: **calibration = paper_dev** and **sealed test = paper_test**.
   - This is a change from the plan's 30/70 random split. Approval needed (OD-B1).
   - It removes split-seed freedom.
   - Page overlap between the halves is measured and reported. Claims whose pages appear in both halves are kept, but reported as a slice.
3. **Deterministic sampling.**
   - Seed `20261027`. Items are sorted by claim `id`, then sampled per label.
   - 400 SUPPORTS / 400 REFUTES / 400 NEI, split 30/70 between calibration and test: about 360 calibration items and 840 test items.
   - Duplicate claim texts are dropped first.
4. **Evidence choice.** One evidence set per SUPPORTS/REFUTES claim, deterministically the first set with the fewest sentences. Its size is recorded so multi-sentence and multi-page items form a slice.
5. **Streaming extraction.** One streaming pass reads the zip member by member and keeps only the referenced pages' `lines`. Nothing else is extracted.
6. **Integrity.** Every referenced `(page, sentence_index)` must resolve to a non-empty sentence.
   - Unresolvable items are **excluded and counted**, never re-sampled, and the exclusion list is published.
   - Wikipedia markup tokens in the dump (`-LRB-`, `-RRB-` and so on) are normalised by a fixed, published mapping.
7. **Freeze.** Row hashes go into a manifest via `evals/canonical.py` before any model sees the data.

## 6. NOT ENOUGH INFO without false support

- **The trap.** NEI claims have **no gold evidence**. An empty passage would let a judge "support" from parametric knowledge. Worse, a pipeline that treats missing evidence as neutral would score a vacuous pass.
- **Rule 1: NEI items get a real, related passage.**
  - The passage is the introductory sentences of the Wikipedia page whose title best matches the claim's main entity, chosen by a deterministic, published rule: exact title match on the claim's leading noun phrase, with a documented fallback.
  - The passage is marked `evidence_kind = "related_not_gold"`. This mirrors the FEVER baseline's NEI sampling.
- **Rule 2: an item without a passage is never scored.** If no passage can be built, the item is excluded and counted.
- **Rule 3: scoring.**
  - Gold NEI with any SUPPORTED prediction counts as **false SUPPORTED**; with a CONTRADICTED prediction, it counts as false CONTRADICTED.
  - Missing or malformed output is NEI (abstention). It is reported, never counted as correct support.
- **Rule 4: label noise.** Some sampled related passages may actually contain support. A blinded 50-item spot check estimates this, and the rate is reported as label noise. Labels are not changed.

## 7. Additional controls

- **Claim-only arm.** FEVER has a known claim-only artifact: negation words predict REFUTES.
  - A "claim-only" arm gives the judge no evidence.
  - Precision from this arm marks how much a judge can score without reading.
  - Pre-registered: any arm whose SUPPORTED precision falls within 0.02 of its claim-only score is reported as **not evidence-grounded**.
- **Snapshot match.** Evidence is the 2017 snapshot, so time-sensitive claims are judged against 2017 facts. That is correct for the benchmark, but it says nothing about FIE's live-question behaviour (E-R2 territory).

## 8. Compute and staging (OD-R5, OD-R6)

- **No NLI model** (OD-R5 deferred).
- **Core arms:**
  - V0 / V1: `llama3.1:8b` with the production prompt; V1 rescores V0's generations at no extra cost;
  - V2: heuristics, no model;
  - V3: the quote-grounded prompt on `llama3.1:8b`;
  - claim-only control;
  - C+ / C−.
- **Stage 1, smoke run.** 30 calibration items. Measure s/item on this machine (4 GB VRAM) and report before Stage 2.
- **Stage 2.** Calibration split → thresholds fixed and recorded.
- **Stage 3.** Sealed test, run once.
- **Model panel V5** (`qwen3:8b`, `granite3.3:8b`, `mistral:7b`): deferred if Stage 1 projects more than about 8 h for the core arms.
- **Ordering rule.** No run starts while code changes are pending. The S0 changes must pass the targeted tests first, then be committed by the owner.

## 9. Thresholds (OD-R7, retained) and reporting

- **SUPPORTED:** precision Wilson 95% lower bound ≥ 0.95, with **recall on gold SUPPORTS ≥ 0.60**.
- **CONTRADICTED:** precision Wilson 95% lower bound ≥ 0.90 before any override is even considered.

**Reported separately:**
- supported-claim recall;
- overall coverage, meaning non-NEI predictions over all items;
- risk–coverage / AURC.

**Discipline:**
- All thresholds are chosen on calibration (paper_dev) only and recorded in the run directory before test is unsealed.
- No change after test results are seen. Any later run is labelled exploratory.

## 10. Approvals needed before download

| ID | Decision |
|---|---|
| OD-B1 | Use the official paper_dev / paper_test halves as calibration / test, instead of a random 30/70 split |
| OD-B2 | Download `paper_dev.jsonl`, `paper_test.jsonl` and `wiki-pages.zip` from the official source (≈ 2 GB, one time). The licence is recorded first, under the §2 stop rule |
| OD-B3 | The NEI related-passage rule (§6) and the 50-item noise spot check |
| OD-B4 | The claim-only control and its 0.02 rule (§7) |
| OD-B5 | Delete `wiki-pages.zip` after extraction, keeping only its hash |
