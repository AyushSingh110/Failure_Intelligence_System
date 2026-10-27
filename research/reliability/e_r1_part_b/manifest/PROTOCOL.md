# E-R1 Part B — Frozen Protocol (preregistration addendum)

**Written:** 2026-10-27, before any evidence passage was read and before any model was run.
**Parent documents:** [E-R1 plan](../../2026-10-27_E-R1_verifier_benchmark_plan.md) §3 · [FEVER preflight](../../2026-10-27_E-R1_partB_FEVER_preflight.md).
**Deviations from those documents:** each is listed in §9, with the reason. Nothing else is changed.

## 1. Data

- **Source:** Zenodo record `10.5281/zenodo.4925954` (revision 2), by Thorne, Vlachos, Christodoulopoulos and Mittal, licence `cc-by-sa-3.0`.
- **Files:** `paper_dev.jsonl`, `paper_test.jsonl`, `wiki-pages.zip`. The md5 of each is checked against Zenodo; the SHA-256 is recorded in `sample_manifest.json`.
- **Splits:**
  - `paper_dev`: smoke and calibration.
  - `paper_test`: the sealed test.
  - The two halves concatenate exactly into `shared_task_dev`; see `split_audit.json`.
  - `shared_task_test` (unlabelled) and `train` are not used.

## 2. Sampling (`fever_materialize.py`, `fever_materialize/1`)

- **Seed:** 20261027.
- **Order:** rows sorted by FEVER id, then `random.Random(f"{seed}:{split}:{label}").sample`.
- **Deduplication:** every row whose normalised claim (lower-cased, whitespace-collapsed, final period stripped) repeats *within* its split is dropped, not only the later copies.
- **Quotas per label:**
  - smoke: 10 (from paper_dev);
  - calibration: 120 (paper_dev, disjoint from smoke);
  - test: 280 (paper_test).
  - Total: 30 + 360 + 840 = 1,230 items.
- **Exclusions.** An item whose primary evidence cannot be resolved, or an NEI item without a related page, is **excluded and counted, never re-sampled**.
- **Cross-split pages.** Test items citing a page also cited in paper_dev are flagged `page_shared_across_splits` and reported as a slice. They are not removed.

## 3. Evidence

- **SUPPORTS / REFUTES (gold).**
  - All evidence sets are kept as identifiers.
  - The **primary set** shown to models is the first set with the fewest sentences.
  - Sentence text is taken exactly from the dump's `lines` field.
  - The shown passage replaces the dump's escapes using the fixed mapping in `sample_manifest.json`. Raw text is kept too.
- **NOT ENOUGH INFO (`related_not_gold`).** The related passage is the first two non-empty sentences of the page whose title equals the longest claim span (1–6 tokens) starting with a capitalised token. Ties go to the leftmost span.
  - It is never called gold evidence.
  - It is not evidence that the claim is false.
  - A SUPPORTED prediction on it is **false support**, and a CONTRADICTED prediction is **false contradiction**.
- **No live Wikipedia, ever.**

## 4. NEI noise audit (stop rule fixed now)

- **Sample:** 50 NEI items, drawn from **smoke + calibration only** (seed 20261027, sorted by id). Test passages are not inspected.
- **Reviewer:** the assistant. This is an LLM-performed review, not a human one, and is reported as such. Owner or human confirmation is recommended.
- **Each item is classified as:**
  - **OK:** related, does not settle the claim;
  - **ACCIDENTAL_SUPPORT** or **ACCIDENTAL_REFUTE:** the passage settles the claim;
  - **CONSTRUCTION_FAIL:** wrong entity sense, empty or garbled.
- **STOP before scoring** if ACCIDENTAL_SUPPORT + ACCIDENTAL_REFUTE ≥ 3/50, or CONSTRUCTION_FAIL ≥ 5/50.

## 5. Arms (smoke: core arms only)

| Arm | Model | Input | Decision |
|---|---|---|---|
| V0 | `llama3.1:8b` | The **production Serper prompt**, copied verbatim and checked against `serper_verifier.py` at run time. Slots: question = `Is the following claim correct? <claim>`, model output = claim, search result = passage. The production truncations are kept (`[:200]`, `[:300]`, `[:400]`) | Production `_parse_groq_verdict`: matches → SUPPORTED, else CONTRADICTED. It cannot say NEI |
| V1 | V0's generations | — | `parse_verdict_strict`: CONSISTENT → SUPPORTED, INCONSISTENT → CONTRADICTED, anything else → NEI. Confidence threshold τ = 0 in smoke; τ is chosen on calibration only |
| V2 | none | claim, passage[:400] | Production `_heuristic_match`: True → SUPPORTED, False → CONTRADICTED |
| V3 | `llama3.1:8b` | quote-grounded prompt (§6), passage up to 2,000 chars | One explicit `LABEL` line. SUPPORTED/REFUTED counts only if `QUOTE` is a ≥3-word substring of the passage (case- and whitespace-normalised); otherwise NEI |
| CO | `llama3.1:8b` | claim only (§6) | One explicit `LABEL` line, else NEI. **Reported separately** |
| C+ / C− | — | — | always SUPPORTED / always NEI |

**Inference settings:** Ollama `/api/generate` on 127.0.0.1 only. `temperature 0`, `seed 0`, `num_ctx 2048`. `num_predict` is 100 for V0 (as in production), 160 for V3 and 20 for CO. Prompt SHA-256 hashes are recorded per arm.

## 6. Prompts (V3 and CO; V0 is production's)

The V3 and CO texts below are hashed into the run summary. Any change is a new protocol version.

V3:
```
You are verifying a claim against an evidence passage. Use ONLY the passage; ignore anything else you know.

Claim: {claim}

Passage: {passage}

Choose one label:
SUPPORTED - the passage states information that makes the claim true.
REFUTED - the passage states information that makes the claim false.
NOT_ENOUGH_INFO - the passage does not settle the claim.

Answer with EXACTLY these two lines:
LABEL: SUPPORTED or REFUTED or NOT_ENOUGH_INFO
QUOTE: <the exact sentence or phrase from the passage that decides the label, or NONE>
```

CO:
```
Decide whether the following claim is true, using only your own knowledge.

Claim: {claim}

Answer with EXACTLY one line:
LABEL: SUPPORTED or REFUTED or NOT_ENOUGH_INFO
```

## 7. Metrics and acceptance criteria (OD-R7, unchanged)

- **SUPPORTED precision:** Wilson 95% lower bound ≥ 0.95, with **recall on gold SUPPORTS ≥ 0.60**.
- **CONTRADICTED precision:** Wilson 95% lower bound ≥ 0.90 before any override may even be considered.
- **Reported separately:**
  - supported-claim recall;
  - overall coverage (non-NEI predictions / items);
  - false-SUPPORTED rate on REFUTES and on NEI;
  - parse-failure rate;
  - risk–coverage curves;
  - the claim-only arm.
- **Not-grounded rule.** An evidence arm whose SUPPORTED precision is within 0.02 of CO's is reported as **not evidence-grounded**.
- **Threshold discipline.** Thresholds are selected on calibration only and written to the run directory before the test split is opened. Smoke results are **feasibility only**: no threshold or prompt is changed because of them without a new protocol version.

## 8. Sealed test

- **Storage.** `test_sealed.jsonl` holds no labels; labels are kept in `test_sealed_labels.jsonl`. Both are git-ignored.
- **Who opens it.** No stage before the authorised sealed run opens the labels file. The materialiser writes it; nothing else reads it.
- **Hashes.** Both files' hashes are in `sample_manifest.json`.

## 9. Deviations from earlier documents

| Item | Earlier | Now | Why |
|---|---|---|---|
| Calibration / test | random 30/70, grouped | official paper_dev / paper_test halves | Removes split-seed freedom. The halves are verified disjoint by id and by claim text. Page overlap is reported (4.8% of test gold rows) |
| Smoke subset | unspecified | 30 extra paper_dev items, disjoint from calibration | The owner required the smoke subset to be excluded from measurement |
| Sample size | 1,200 | 1,200 measured + 30 smoke | — |
| V3 truncation | unspecified | 2,000 chars | Primary sets are short; the limit is recorded |
| Noise-audit stop rule | "reported as label noise" | thresholds in §4 | Owner instruction to stop on construction failure |

## 10. Addendum, 2026-10-27, added after extraction (sections 1–9 unchanged)

1. **Extractor fix before freeze (`fever_materialize/2`).**
   - The claim files spell page titles in NFD (decomposed accents); the dump spells them in NFC.
   - Version 1 therefore failed to resolve 9 pages and wrongly excluded 17 gold items.
   - Version 2 matches on the NFC form of both sides and keeps the original identifiers.
   - The sampled items are unchanged; this is a matching bug fix, not re-sampling.
   - After the fix: 0 unresolved gold items. 9 NEI items have no related page and are excluded and counted.
   - Archive members outside `wiki-pages/*.jsonl` are skipped: `license.html` and 113 `__MACOSX` resource forks.
2. **NEI noise audit: STOP.**
   - See `nei_noise_audit_result.json`: 6/50 construction failures (threshold 5) and 4/50 accidental support or refute (threshold 3).
   - **No model was scored.** The §3 NEI rule needs revision and a fresh audit before any smoke run; that is an owner decision.
