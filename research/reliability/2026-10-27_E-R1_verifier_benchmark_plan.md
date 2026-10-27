# E-R1 — Verifier Decision Contract and Gold-Evidence Judge Benchmark (Preregistration Draft)

**Prepared:** 2026-10-27 · **Status:** draft for owner review. Nothing has been run, downloaded or created.
**Parent:** [2026-10-27_output_reliability_audit_v2.md](2026-10-27_output_reliability_audit_v2.md) (§7–§9)
**Supersedes, if OD-R9 is approved:** [2026-10-20_verifier_replay_experiment_plan.md](2026-10-20_verifier_replay_experiment_plan.md), folded in as Part A. That file is left unchanged.
**Workstream:** B only. No C2 artifact, canonical baseline, threshold, model or production file is touched.

## 1. Research questions

- **RQ1 (contract).** Do FIE's current verifier parsers and action policies turn unknown, failed or degenerate inputs into approval or replacement? Does the proposed contract eliminate this?
- **RQ2 (judge capability).** Given a claim and its **gold** evidence passage, can any candidate verifier label SUPPORTS / REFUTES / NOT-ENOUGH-INFO with SUPPORTED precision whose 95% Wilson lower bound is ≥ 0.95, at ≥ 60% coverage of truly-supported claims?
- **RQ3 (attribution).** How much of the current false-SUPPORTED rate comes from parser defaults rather than judge errors (V0 vs V1 on identical generations)?
- **RQ4 (cheap safeguards).** Does a deterministic verbatim-quote check (V3), or a local NLI model (V4), beat the free-form LLM judge on SUPPORTED precision?
- **RQ5 (correlation).** Are judge errors correlated across model families? Does unanimity buy the precision that independence would predict?

**Not in scope:** retrieval quality (E-R2), claim decomposition (E-R3), any correction, any change to production behaviour.

## 2. Part A — Decision-contract fixtures (zero cost, no data approval needed)

**Method.** Pure-function replays, with no network. Each fixture feeds the *same* synthetic inputs into:

- **B0** — current parsers and policies, called directly or as faithful re-implementations, with the source location cited per case;
- **B1** — strict tri-state;
- **B2** — B1 plus an evidence-identity requirement.

Run under the `evals/hermetic.py` guard so that any network attempt fails the run.

| # | Fixture | B0 expected (from code) | B1/B2 required |
|---|---|---|---|
| A1 | Serper verdict `UNCERTAIN` | consistent → `FULLY_PROVENANCED` | `INSUFFICIENT_EVIDENCE` |
| A2 | Verdict line missing or malformed | consistent | `INSUFFICIENT_EVIDENCE` |
| A3 | `INCONSISTENT`, confidence 0.0 | override with the generated answer | `CONTRADICTED` candidate, no override below the calibrated threshold |
| A4 | Groq unavailable, short correct answer vs long snippet (overlap ≤ 0.25) | snippet replaces the answer | `VERIFIER_UNAVAILABLE` |
| A5 | Groq unavailable, answer "42" (zero 4-letter tokens) | "confirmed" at 0.0 | `VERIFIER_UNAVAILABLE` |
| A6 | Wikidata entity without description, or Groq failure | (True, 0.0) → falls through to shadow consensus | `VERIFIER_UNAVAILABLE`; no consensus correction |
| A7 | Wikidata `INCONSISTENT` ≥ 0.75 with an 8B `CORRECT_VALUE` absent from the description | override, `FULLY_PROVENANCED` | not `SUPPORTED`/override: the value is not evidence-extracted |
| A8 | 1 surviving shadow | consensus 1.0 → replacement | `ensemble_unavailable`; no replacement |
| A9 | 2 shadows, equal weight, different text | 0.50 → first shadow replaces | no replacement |
| A10 | 3 shadows, weights [3, 2, 1], all different | 0.50 → HIGH shadow replaces | no replacement |
| A11 | 0 shadows, Groq down | not high-risk → no verification, "stable" | `ensemble_unavailable` and answer state `PASS_QUALIFIED`, never verified |
| A12 | Wikidata confirms the primary, then the fix engine or Wikipedia runs | fix or Wikipedia may replace a confirmed answer | the confirmation terminates the action chain |
| A13 | GT escalates, then the `/monitor` Wikipedia fallback | escalation cleared, Wikipedia answer returned | escalation preserved |
| A14 | Wikipedia model says "context insufficient" | raw summary returned at 0.72, `success=True` | `INSUFFICIENT_EVIDENCE` |
| A15 | Cache hit for a temporal prompt stored N days ago | `FULLY_PROVENANCED` | stale → re-verify or `INSUFFICIENT_EVIDENCE` (as-of policy) |
| A16 | Cache hit whose stored source is shadow-derived or a tenant correction | `FULLY_PROVENANCED` | label reflects the stored source class |
| A17 | Reasoning factual step with any GT result | `is_correct=True`, confidence 0 (dead branch) | `NOT_CHECKED`, never "correct" |

**Pass criterion (B1/B2):** 0 of 17 fixtures end in `SUPPORTED`, `PASS_VERIFIED` or replacement from an unknown, failed or degenerate input. B0's failures are reported as a table with source line references. This part demonstrates contract compliance only; it is not evidence of better reliability, which is Part B's job.

## 3. Part B — Gold-evidence judge benchmark

### 3.1 Data (requires OD-R4)

- **Choose one primary dataset** (licence to be verified before download):
  - **FEVER**: claims, Wikipedia sentence evidence, 3-way labels.
  - **VitaminC**: contrastive evidence revisions, which stress freshness and minimal edits.
  - **HaluEval-QA with `knowledge`**: question-answer framing that matches FIE's input. Gives paired right and hallucinated answers *for the same question*; a clean re-download also replaces the documented corrupted local file.
- **Sample.** Target 1,200 items stratified 400/400/400 by label after deduplication by source document or entity.
- **FIE-shaped transform** (HaluEval-QA only). Render both answers through one template (`"Q: … A: …"` → claim sentence) so the 2-vs-9.5-word style confound in the current frozen set cannot drive results. Report the raw-format results separately.
- **Split.** 30% calibration, 70% sealed test, grouped by source document or entity. Freeze row hashes with `evals/canonical.py` before any model sees the data.
- **Storage.** Raw text under `research/reliability/data/` (gitignored, OD-R8). Commit only the manifest: source URL, revision or date, licence, hashes, counts, split seed.
- **Power.** With about 280 items predicted SUPPORTED in test at an observed precision of 0.98, the Wilson 95% LB is about 0.957. If fewer than 250 SUPPORTED predictions occur, report "insufficient power to establish ≥ 0.95" rather than pass.

### 3.2 Arms

All arms run locally. No Groq or other external API.

| Arm | Judge | Parser / decision | Purpose |
|---|---|---|---|
| **V0** | `llama3.1:8b` (Ollama; same base model as production `llama-3.1-8b-instant`, quantised) with the **production Serper prompt** | production `_parse_groq_verdict` (default True) | Current semantics, replayed |
| **V1** | **same generations as V0** | strict tri-state: only explicit CONSISTENT plus a valid confidence ≥ calibrated τ is SUPPORTED; explicit INCONSISTENT ≥ τ′ is CONTRADICTED; everything else is NEI | Isolates parser effect (RQ3) at zero extra compute |
| **V2** | none | production `_heuristic_match` and `_heuristic_verification` | Current fallbacks |
| **V3** | `llama3.1:8b` with a new prompt: 3-way label plus the **verbatim supporting or refuting span** | SUPPORTED only if the span is a substring of the passage (normalised whitespace) and the label is explicit; otherwise NEI | Quote grounding (RQ4) |
| **V4** *(optional, OD-R5)* | local NLI or fact-check cross-encoder | entailment ≥ τ → SUPPORTED; contradiction ≥ τ′ → CONTRADICTED | Cheap verifier baseline (RQ4) |
| **V5** | V3 prompt on `qwen3:8b`, `granite3.3:8b`, `mistral:7b` | per-judge results; unanimity and majority rules | Correlation (RQ5) |
| **C+** | — | always SUPPORTED | Fail-open extreme (≈ current) |
| **C−** | — | always NEI | Zero-coverage safe extreme |

Fixed generation settings: temperature 0, max tokens fixed per prompt, Ollama model digests recorded, prompts stored verbatim in the run directory.

### 3.3 Metrics

**Primary:**

- SUPPORTED precision, with a Wilson 95% LB.
- Coverage, defined as recall on gold SUPPORTS.

**Secondary:**

- CONTRADICTED precision (LB) and recall.
- False-SUPPORTED rate, split into REFUTES items and NEI items.
- Risk–coverage curve and AURC over the calibrated score.
- 3×3 confusion matrix.
- Parse-failure rate.
- For V5: per-judge results, pairwise Cohen's κ on *error indicators*, and observed vs independence-predicted unanimity precision.
- Latency per item, p50/p95 on this machine.

**Slices** (where the dataset supports them): numeric, date and temporal claims; negation; multi-entity claims; passage length; and, for VitaminC, contrastive pairs.

**Statistics:**

- Paired V0 vs V1 and V1 vs V3: McNemar on false-SUPPORTED, and a bootstrap clustered by source document (10,000 resamples, fixed seed).
- No threshold may be tuned on test.

### 3.4 Preregistered decision rules

| Outcome | Interpretation | Next step |
|---|---|---|
| Some arm achieves SUPPORTED LB ≥ 0.95 at coverage ≥ 0.60 **and** CONTRADICTED LB ≥ 0.90 | A usable judge exists, given evidence | E-R2 (source recall), using that arm as the judge |
| SUPPORTED LB ≥ 0.95, but coverage < 0.60 | Safe but conservative | E-R2, accepting more qualified answers; owner reviews the coverage budget |
| No arm reaches SUPPORTED LB ≥ 0.95 | Even gold evidence is not enough; RAG would rest on an unreliable judge | Pivot to structured-fact verification (audit §14 #2) and keep warn-only; do not start RAG |
| CONTRADICTED LB < 0.90 for all arms | Overrides cannot be justified | No correction stage; FLAG only |
| V1 ≫ V0 on false-SUPPORTED | Parser defaults are a dominant harm | Supports contract adoption (OD-R2) independently of the other results |
| V5 error κ high, unanimity gains small | Agreement among judges is not independent evidence | Single best judge plus deterministic checks; no judge panels |

Thresholds are taken from OD-R7. If the owner changes them, the change is recorded *before* the test split is unsealed.

## 4. Reproducibility and integrity

- Run IDs, dataset manifest hashes, Ollama model digests, prompt hashes, harness commit and seeds are written to `research/reliability/runs/<UTC>_E-R1/` (gitignored raw; committed `SUMMARY.json`).
- Network is blocked during scoring (hermetic guard). Ollama is reached only on localhost; record its version.
- Test-set sealing: hashes are committed before the first scoring run, and any re-run after unsealing is labelled exploratory.
- Every excluded item and every parse failure is counted and reported.

## 5. Cost, time and privacy

- **External cost:** zero. **Data egress:** none beyond the approved one-time dataset download.
- **Compute (estimate, to be measured):** on a 4 GB VRAM GPU with CPU offload, about 2–8 s per item per LLM arm. 1,200 items × (V0 + V3 + 3 × V5) ≈ 6,000 generations ≈ 3–13 h. V1, V2, C+ and C− are free; V4 is likely under 1 h on CPU.
- **Privacy:** public data only, and no user prompts. Raw text stays local.

## 6. Deliverables after an approved run

1. Part A fixture table (B0/B1/B2) with source references.
2. Part B report covering, per arm: metrics with confidence intervals, the confusion matrix, risk–coverage plots, slice tables and the V5 correlation analysis.
3. An error taxonomy from a random sample of 100 false-SUPPORTED and 100 false-CONTRADICTED items: negation, number or unit, date, entity confusion, partial support, judge hallucination, parse failure.
4. A go/no-go recommendation against §3.4, with no integration implied.

## 7. Approvals needed before any step

| Approval | Covers |
|---|---|
| OD-R4 | Dataset choice and download |
| OD-R5 | Optional NLI model download |
| OD-R6 | Local compute |
| OD-R7 | Thresholds |
| OD-R8 | Data location |
| OD-R9 | Status of the 2026-10-20 plan |

Part A needs only OD-R6 (negligible compute) and permission to add a research-only script under `research/reliability/`.
