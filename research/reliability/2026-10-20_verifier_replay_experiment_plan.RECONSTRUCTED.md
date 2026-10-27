<!--
RECONSTRUCTION, NOT THE ORIGINAL FILE.  Provenance: 2026-10-20_RECOVERY_PROVENANCE.md
-->

> **⚠ Reconstructed document (OD-S0-5, restored 2026-10-27).**
> The original `2026-10-20_verifier_replay_experiment_plan.md` was deleted from the working tree on
> 2026-10-27 at about 11:51 IST. It had never been committed. Everything below the horizontal rule was
> rebuilt from a raw `cat` of the original captured in the audit session's transcript at about 11:50 IST.
> The rebuilt text is character-identical to that capture; the final newline and the line-ending style
> could not be observed and were inferred (LF; final newline, consistent with the original's
> `wc -l` count of 102). SHA-256 of the body below the rule: `63b9d9a95ee1b9287321465af47c872c6bd058779a52cd3c6fdc980fdbceae1b`.
> The original's own hash was never recorded, so byte-identity with the original cannot be proven.
> Nothing below the rule has been edited; its links refer to the original filenames.

---

# FIE Reliability Experiment Plan: Verifier Decision Contract

**Owner review required before implementation or dataset construction**  
**Prepared:** 2026-10-20  
**Parent audit:** [Independent audit and research recommendation](2026-10-20_output_reliability_audit.md)

## Objective

Measure whether replacing permissive binary verifier handling with explicit evidence states reduces unsupported answer approvals, without unacceptable loss of useful answer coverage. This experiment targets the confirmed interpretation risk in the Serper/Wikipedia/Wikidata paths. It is output-reliability work and remains separate from adversarial-input C2 research.

## Hypothesis

If a verifier response is missing, malformed, uncertain, unavailable, or based on insufficient evidence, representing it as `INSUFFICIENT_EVIDENCE` or `VERIFIER_UNAVAILABLE` instead of “consistent” will reduce false approvals. A stricter evidence-presence requirement may reduce approvals further, at some cost in coverage.

## Scope and constraints

- First run is local, deterministic replay only; no model or search calls.
- Do not change production code, thresholds, dependencies, canonical baselines, or existing data.
- Do not edit C2 artifacts.
- Any new evaluation items or annotations are append-only and require owner approval.
- Store prompt and answer fixtures only if approved data policy permits; otherwise use synthetic contract cases that contain no user data.

## Variants

| Variant | Decision behavior |
|---|---|
| B0 — Current | Replay the current parser and caller semantics exactly as source defines them. |
| B1 — Explicit tri-state | `INCONSISTENT` → `CONTRADICTED`; well-formed `CONSISTENT` with a valid confidence → candidate support; everything else → `INSUFFICIENT_EVIDENCE` or `VERIFIER_UNAVAILABLE`. |
| B2 — Evidence contract | B1 plus require retained evidence text and source identity; no source identity means no `FULLY_PROVENANCED` state. |

No thresholds should be selected from the held-out evaluation set. Calibrate only on a separately designated calibration partition if a confidence threshold is needed.

## Contract matrix

Cover each source/verifier outcome at least once:

| Case | Expected state under B1/B2 |
|---|---|
| Explicit, valid `CONSISTENT`, evidence actually supports the claim | `SUPPORTED` candidate, subject to source policy and threshold |
| Explicit `INCONSISTENT`, evidence contradicts the claim | `CONTRADICTED` |
| Explicit `UNCERTAIN` | `INSUFFICIENT_EVIDENCE` |
| Missing verdict / empty response / malformed response | `INSUFFICIENT_EVIDENCE` |
| Confidence missing, nonnumeric, out of range | `INSUFFICIENT_EVIDENCE` |
| Search no results / no usable snippet | `INSUFFICIENT_EVIDENCE` |
| API timeout, exception, provider disabled | `VERIFIER_UNAVAILABLE` (or no-tool-required if classified so before invocation) |
| Word overlap high but passage does not entail claim | Not `SUPPORTED`; verifier evidence is insufficient/contradictory per label |
| Word overlap low but passage paraphrases claim | Do not infer contradiction from lexical mismatch |
| Relevant and authoritative passages disagree | `CONFLICTING_EVIDENCE` |
| Correct subject but wrong entity/page | `INSUFFICIENT_EVIDENCE`; source relevance fails |
| Retrieved passage is stale for a time-sensitive question | `INSUFFICIENT_EVIDENCE` for current-state claim |
| One of several answer claims unsupported | Claim-level unresolved; whole answer cannot be fully supported |

## Data and annotation phase (after approval)

1. Inventory the current frozen TruthfulQA pair set, local HaluEval material and any candidate RAG set; verify schemas, provenance, licensing and known limitations before use.
2. Keep existing files immutable. Create a new versioned manifest under a reliability-specific path.
3. Add a small source-grounded set spanning static fact, temporal fact, answerable-without-evidence, ambiguous entity, contradictory sources, and retrieval failure cases. Freeze source snapshots and timestamps.
4. Annotate atomic claim spans, normalized propositions, evidence spans, stance, source identity/type, authority decision, freshness and final action. Double-label a stratified subset and adjudicate.
5. Split by question/entity/source/time cluster. Keep near-duplicates and source variants in a single split. Seal the test split before choosing thresholds.

No data is to be downloaded or uploaded as part of this plan.

## Metrics

Primary:

- False approval rate among contradicted, unsupported, insufficient, conflicting and unavailable cases.
- Claim-level support precision/recall and contradiction precision/recall.
- Correct-answer false flag rate.
- Selective risk versus coverage and abstention/escalation fraction.
- Contract matrix failures, particularly any unknown → approved transition.

Operational:

- p50/p95 latency, model/API calls, input/output tokens, external requests and provider failures.
- Correction net effect: supported after correction, still wrong, or newly unsupported.

Use paired comparisons with question-clustered bootstrap confidence intervals. Report slices by source class, domain, time-sensitivity, answer length and verifier availability. Do not drop adversarial-guard-blocked items silently; count separately and preserve Workstream A results.

## Proposed go/no-go criteria

These are proposed review gates, not established performance:

1. Contract matrix: zero unknown, malformed, empty, failed, timed-out or evidence-free verifier outcomes marked `SUPPORTED` or automatically corrected.
2. Labeled set: unsupported/contradicted auto-approval falls at least 30% relative to B0, with paired 95% CI excluding zero improvement.
3. Correct-answer false flags increase no more than 5 percentage points unless the owner explicitly chooses a more conservative product trade-off.
4. Any `FULLY_PROVENANCED` decision carries a source identifier and claim-linked passage.
5. A `SUPPORTED` auto-approval threshold must demonstrate at least 0.95 claim-level precision with 95% confidence; otherwise the outcome remains qualified/unverified or is reviewed by a human.
6. Target at least 70% coverage on answerable in-scope cases, reported with confidence intervals. Falling below this is a prompt to diagnose evidence coverage, not permission to lower the support bar without review.
7. Correction is not enabled unless post-correction checks do not reduce support precision relative to the original answer on paired examples.

## Deliverables after an approved run

- Frozen data and source manifest with hashes and license notes.
- Annotation guide and disagreement/adjudication summary.
- B0/B1/B2 decision matrix and paired metric report with confidence intervals.
- Error analysis: shared model errors, retrieval miss, source-quality issue, parser failure, claim-extraction omission, NLI/verifier error, and correction-introduced error.
- Recommendation to proceed, revise, or stop; no integration or rollout implied by a passing experiment.

## Approval points

Before work beyond this plan, the owner should approve: (1) the initial domain and source policy, (2) creating a new labeled evaluation set and annotation effort, (3) any provider calls or external costs in a later phase, and (4) the proposed precision/coverage trade-off. The current audit did not make these calls or create evaluation data.
