# Workstream B — Stage S0 Containment and E-R1 Part A: Execution Report

**Date:** 2026-10-27 · **Status:** implemented and validated locally, **awaiting owner review. Not committed.**
**Branch / base:** `rebuild/wp-001-eval-harness` at `f0888bb` (the owner committed the audit and plan as `f0888bb` during this session; see §1).
**Decisions:** D-060 to D-063 ([decision log §15](../../docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md)).
**Parent documents** (unchanged): [audit v2](2026-10-27_output_reliability_audit_v2.md) · [E-R1 plan](2026-10-27_E-R1_verifier_benchmark_plan.md).
**Scope:** Workstream B only. No C2 artifact, canonical baseline, adversarial model, detector threshold, `fie/` file or `engine/agents/adversarial/patterns.py` was modified.

**Evidence labels:** **[RUN]** measured in this session · **[CODE]** read from source · **[DOC]** stated in a document · **[GAP]** known limitation.

---

## 0. Summary

- **What changed.** `/api/v1/monitor` is now warn-only.
  - The primary answer is always the answer delivered.
  - An engine correction is reported as a *proposal* (`fix_result.proposed_output`), with `fix_applied=False` and `fixed_output` equal to the original.
  - A fallback no longer clears a human-review escalation.
  - The provenance label claims no more than the evidence shows.
  - Each response carries an explicit `reliability` decision: evidence state, action, reasons. This is separate from Workstream A's ALLOW / REVIEW / BLOCK.
- **What did not change.**
  - The classifier's inputs, its probability and `high_failure_risk`. Verified identical before and after on all 19 route cases [RUN].
  - Every legacy field of the verifiers and the ground-truth pipeline.
  - The fix engine.
  - The signal log's values.
  - The regex defect.
- **Part A, the contract fixtures:**
  - The current route (B2) delivers no replacement, no `SUPPORTED`/`PASS_VERIFIED` from unknown evidence and no cleared escalation in any case.
  - The pre-containment route (B0) shows its documented defect in every case.
  - Coverage: 17 fixtures as 20 cases. Each passes under pytest's socket guard and again in a child process under `evals/hermetic.py`, with 0 recorded network attempts [RUN].
- **Not yet established.**
  - Whether any verifier is accurate. That is Part B.
  - The contract proves only that unknown evidence can no longer become approval or replacement. It is no evidence that FIE's answers are more reliable.

## 1. Initial state and preservation (Step 1)

| Item | Recorded value [RUN] |
|---|---|
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD at session start | `7e0e852eef93a1e42abdc0e0f2a1f3913137896e`, with `?? research/reliability/` as the only change |
| HEAD from 13:54 IST | `f0888bb270692433988a15e93e95262fc6488796` "reliability docs" (owner commit; merged to `origin/main` as PR #30, `dd5b08f`) |
| Working tree before any edit | clean at `f0888bb` (ignored files only) |
| `2026-10-27_output_reliability_audit_v2.md` | sha256 `9818b8d3…ad8dd30d`; identical to the blob in `f0888bb`; **not modified** |
| `2026-10-27_E-R1_verifier_benchmark_plan.md` | sha256 `aba72e7c…74aca6c`; identical to the blob in `f0888bb`; **not modified** |
| Test baseline (offline, CI invocation) | 1,494 passed, 24 skipped, exit 0 (see §5 for a caveat on timing) |

HEAD moved during Step 1 because the owner committed and pushed the two reports. Nothing in this session staged, committed, pushed, reset, cleaned or switched anything.

### 1.1 The two 2026-10-20 reports: actual status

| File | In the working tree? | In any git ref? | Recovered copy |
|---|---|---|---|
| `2026-10-20_output_reliability_audit.md` | **No** | **No** (`git log --all -- research/reliability/*` shows only `f0888bb`, which adds the two 2026-10-27 files) | scratchpad, sha256 `5b11f8ac…0b702d200c`, 43,878 B: a byte-exact copy of the content read before it vanished |
| `2026-10-20_verifier_replay_experiment_plan.md` | **No** | **No** | scratchpad, sha256 `63b9d9a9…980fdbceae1b`, 7,372 B: **rebuilt** from the session transcript, so not guaranteed byte-exact |

Recovered copies are at `C:\Users\ASUS\AppData\Local\Temp\claude\c--Users-ASUS-Desktop-Failure-Intelligence-System\87822716-f1cf-4113-8c0c-b7957107d1a9\scratchpad\recovered_2026-10-20\`.

**The discrepancy, resolved.**
- Audit v2 line 7 says the 10-20 files "are preserved unchanged". That was true when the line was drafted. It became false at 11:51 that day, when the files disappeared from disk.
- The earlier summary, that the files disappeared, is the accurate one.
- The two links in audit v2 are therefore broken in the working tree and in `f0888bb` on `main`.
- Audit v2 is left exactly as committed, as instructed. **Correcting line 7 is an owner decision.** An erratum, or restoring the files, would make it true again.
- I compared the recovered copies with the available files. No file is available to compare them with, so nothing was overwritten or restored.
- Restoring them is a separate approval. Recommended: restore the audit (byte-exact); restore the plan only if it is labelled "reconstructed".

## 2. Code changes and rationale (Step 2)

Totals: 8 files modified (+235 / −34), plus 1 new production module and the new test and research files.

| File | Change | Why |
|---|---|---|
| `engine/reliability_contract.py` (new, pure: no I/O, network or model) | `EvidenceState` (SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE, VERIFIER_UNAVAILABLE, NOT_CHECKABLE, NOT_CHECKED). `AnswerAction` (PASS_VERIFIED, PASS_QUALIFIED, FLAG, ESCALATE; no CORRECTED). `WARN_ONLY = True`. `VALIDATED_VERIFIERS = frozenset()`. Strict tri-state `parse_verdict_strict`. `assess_ground_truth`, `decide_action`, `provenance_label`, `contain_fix`, `combine_states`, `step_evidence_state`, `ensemble_available` | One auditable place for every rule. PASS_VERIFIED cannot be reached until a verifier is listed, which needs Part B and OD-R7 |
| `engine/verifier/serper_verifier.py` | `SerperResult.evidence_state` added. `_verify_via_groq_detailed` returns the legacy triple plus the strict state, and the heuristic fallback reports `VERIFIER_UNAVAILABLE`. `_verify_via_groq` keeps its 3-tuple | Requirement 2: a failed verifier must be distinguishable from a confirmation. Legacy values are byte-identical (tested) |
| `engine/verifier/wikidata_verifier.py` | Same pattern. A missing description is `INSUFFICIENT_EVIDENCE`; no judge is `VERIFIER_UNAVAILABLE` | Same |
| `engine/verifier/ground_truth_pipeline.py` | Additive evidence record on the result: `verifier`, `evidence_state`, `evidence_text`, `evidence_source`, `cache_source_class`, `cache_verified_at`. Set at each return point; **no legacy field or branch changed** | The contract can see what was actually established |
| `engine/ground_truth_cache.py` | `CacheHit.source_class` (default `"unknown"`); the similarity-search projection now reads it | A16: the label must reflect the stored source class |
| `engine/rag_grounder.py` | `GroundingResult.insufficient_context`, set when the raw summary is returned in place of an answer | A14 |
| `app/schemas.py` | `FixResult.proposed_output` and `ReliabilityAssessment`, plus `MonitorResponse.reliability`. All additive with defaults | Report the proposal without delivering it |
| `app/routes/monitor.py` | (a) The Wikipedia fallback no longer clears `requires_human_review`. (b) A new Step 8b applies the contract **after** the classifier: `contain_fix`, `decide_action`, the `reliability` object, `AUTO-FIXED:` → `CORRECTION PROPOSED, NOT APPLIED (warn-only):`. (c) Contract provenance label. (d) Fewer than two shadow answers → "Shadow ensemble unavailable — stability not assessed" instead of "Model outputs are stable". (e) The signal log keeps recording the pre-containment proposal | Requirements 1–7. The classifier and the signal log still see exactly what they saw before (§3.2) |
| `tests/test_monitor_rag_fix.py` | One test's assertions changed to the approved contract (§4.3) | OD-R2 |

**Deliberately not changed:**
- `engine/fix_engine.py`: it still proposes, and its tests still pin its proposals.
- `engine/agents/adversarial/patterns.py`, per OD-R3.
- `engine/reasoning/step_verifier.py`: its dead branch is classified, not fixed (§5, gap G2).
- `engine/pipeline/langgraph_pipeline.py`: not served; it keeps its legacy correction semantics (gap G3).
- `fie/` (gap G1).
- Thresholds, models, baselines, C2 artifacts and `data/`.

No retrieval system, NLI model, dependency or correction method was added.

## 3. Before and after on the vulnerable paths

### 3.1 Delivered behaviour

All rows were measured by running the real handler with only I/O leaves replaced [RUN]. "Delivered" is what the SDK's `correct` mode actually returns: the real `correct_wrapper` is run on the response.

| Path | B0: before (route at `f0888bb`) | B2: after |
|---|---|---|
| Serper verdict UNCERTAIN, missing or malformed | confirmation, `FULLY_PROVENANCED`; then CONTEXT_INJECTION replaced the answer with a canned "no live data" text | original delivered; `INSUFFICIENT_EVIDENCE`; `REQUIRES_TOOL_VERIFICATION`; canned text kept as a proposal |
| Serper INCONSISTENT at confidence 0.0 | the search answer replaced the original, `FULLY_PROVENANCED` | original delivered; `CONTRADICTED` → FLAG; proposal recorded; reason `proposed_value_not_in_evidence` |
| Judge unavailable → word-overlap heuristic | snippet replaced a correct short answer; "42" confirmed at 0.0 | `VERIFIER_UNAVAILABLE`; original delivered |
| Wikidata: no description, or judge down | fell through to shadow consensus → replaced | `INSUFFICIENT_EVIDENCE` / `VERIFIER_UNAVAILABLE`; original delivered; `UNVERIFIED_MODEL_INFERENCE` |
| Wikidata override with an 8B-written value | replaced, `FULLY_PROVENANCED` | `CONTRADICTED` → FLAG; `PARTIALLY_PROVENANCED`; `proposed_value_not_in_evidence` |
| Shadow consensus (1 shadow; 0.50 ties) | replaced by the single shadow or by the first / self-rated HIGH shadow | original delivered; reason `shadow_agreement_is_not_evidence` and `ensemble_unavailable` |
| Confirmation followed by the fix engine or Wikipedia | the **confirmed** answer was replaced, still labelled `FULLY_PROVENANCED` | original delivered; `SUPPORTED`, FLAG (the classifier still says high risk); proposal visible |
| Escalation followed by Wikipedia | escalation **cleared**, Wikipedia answer delivered | escalation kept (`ESCALATE`, `requires_human_review=True`); proposal visible |
| Wikipedia "context insufficient" | the raw article summary was delivered at 0.72 | not delivered; reason `wikipedia_context_insufficient` |
| Cache hit: stale live answer, tenant correction, shadow-derived source | every hit `FULLY_PROVENANCED`; a differing hit replaced the answer | `INSUFFICIENT_EVIDENCE`; label by source class (`REQUIRES_TOOL_VERIFICATION` / `PARTIALLY_PROVENANCED` / `UNVERIFIED_MODEL_INFERENCE`); reason `cache_stale_for_live_question` |
| No shadow responses | "Model outputs are stable" | "Shadow ensemble unavailable — stability not assessed"; `NOT_CHECKED`, `PASS_QUALIFIED` |
| Jury says TOKEN_SMUGGLING (the regex defect) | canned refusal replaced the answer | refusal kept as a proposal; original delivered |

### 3.2 Invariance checks [RUN]

| Quantity | Result |
|---|---|
| XGBoost inputs (`fix_applied`, `requires_escalation`, `gt_override`, `fix_strategy`, `gt_source`, `provenance_label`), probability and `high_failure_risk` | identical B0 vs B2 on all 19 route cases (`test_classifier_inputs_and_risk_are_unchanged_by_containment`) |
| Verifier legacy triples | `_verify_via_groq(...) == _verify_via_groq_detailed(...)[:3]` for Serper (5 replies) and Wikidata (3 cases) |
| `GroundTruthPipelineResult` legacy fields | unchanged: only new fields were added; B0 route runs reproduce the documented legacy outcomes on top of the modified engine |
| `fix_engine` | untouched; its existing tests pass |

## 4. E-R1 Part A results (Step 3)

### 4.1 Method

- **Cases:** `tests/reliability/e_r1_part_a.py`, which defines 17 fixtures as 20 cases. A6, A12 and A16 have two variants each.
- **How each case runs:**
  - Each case calls the **real** `/monitor` handler, ground-truth pipeline, verifiers, fix engine, classifier and SDK `correct_wrapper`.
  - Only the I/O leaves are replaced: shadow fan-out, the judge's `complete()`, Serper `_search_google`, Wikidata `_fetch_search_results`, Wikipedia retrieval, the cache collection, notifications and persistence. See `tests/reliability/harness.py`.
- **A17** is function-level: `_verify_factual`.
- **B0** is the route source at `f0888bb`, executed beside the current engine.
  - **What this establishes:** pre-containment *route* behaviour. That is valid because the engine changes are additive and their legacy outputs are tested unchanged.
  - **What it cannot establish:** the behaviour of the deployed server, the real jury (its root cause is fixed per fixture), or real judge and search content.
- **B1 / B2:** B1 is the strict parser alone (unit tests); B2 is the full contract in the route.
- **Two network guards, both with 0 attempts:**
  - **pytest:** a socket guard (connect, connect_ex and getaddrinfo patched; any attempt fails the test).
  - **Child process:** `research/reliability/e_r1_part_a/run_part_a.py` runs the same cases under `evals/hermetic.py` in a child interpreter. The environment is sanitized: no keys, dead proxies, Hugging Face offline, telemetry off.
  - **Guard self-test** passed on all five canaries: 42,931 audit events seen, 0 violations. Result file: `results/20261027T085419Z_part_a.json`.

### 4.2 Fixture table [RUN]

Forbidden outcomes are: `replaced` · `fully_provenanced` (on unknown input or on a replacement) · `supported` (unknown input) · `pass_verified` · `escalation_cleared` · `claims_stable`.

| # | Case | B0 violations | B2 violations | B2 state → action · label |
|---|---|---|---|---|
| A1 | Serper UNCERTAIN | replaced, fully_provenanced | — | INSUFFICIENT_EVIDENCE → PASS_QUALIFIED · REQUIRES_TOOL_VERIFICATION |
| A2 | no VERDICT line | replaced, fully_provenanced | — | INSUFFICIENT_EVIDENCE → PASS_QUALIFIED · REQUIRES_TOOL_VERIFICATION |
| A3 | INCONSISTENT @ 0.0 | replaced, fully_provenanced | — | CONTRADICTED → FLAG · REQUIRES_TOOL_VERIFICATION |
| A4 | judge down; short vs long | replaced, fully_provenanced | — | VERIFIER_UNAVAILABLE → PASS_QUALIFIED · REQUIRES_TOOL_VERIFICATION |
| A5 | judge down; "42" | replaced, fully_provenanced | — | VERIFIER_UNAVAILABLE → PASS_QUALIFIED · REQUIRES_TOOL_VERIFICATION |
| A6a | Wikidata no description | replaced | — | INSUFFICIENT_EVIDENCE → FLAG · UNVERIFIED |
| A6b | Wikidata judge down | replaced | — | VERIFIER_UNAVAILABLE → FLAG · UNVERIFIED |
| A7 | 8B CORRECT_VALUE not in description | replaced, fully_provenanced | — | CONTRADICTED → FLAG · PARTIALLY_PROVENANCED |
| A8 | 1 surviving shadow | replaced | — | INSUFFICIENT_EVIDENCE → FLAG · UNVERIFIED |
| A9 | 2 equal shadows (after confirmation) | replaced, fully_provenanced | — | SUPPORTED → FLAG · FULLY_PROVENANCED |
| A10 | 3/2/1 shadows (after confirmation) | replaced, fully_provenanced | — | SUPPORTED → FLAG · FULLY_PROVENANCED |
| A11 | 0 shadows | claims_stable | — | NOT_CHECKED → PASS_QUALIFIED · UNVERIFIED |
| A12a | confirm → shadow consensus | replaced, fully_provenanced | — | SUPPORTED → FLAG · FULLY_PROVENANCED |
| A12b | confirm → Wikipedia | replaced, fully_provenanced | — | SUPPORTED → FLAG · FULLY_PROVENANCED |
| A13 | escalation → Wikipedia | replaced, escalation_cleared | — | INSUFFICIENT_EVIDENCE → ESCALATE · UNVERIFIED |
| A14 | Wikipedia context insufficient | replaced, escalation_cleared | — | INSUFFICIENT_EVIDENCE → ESCALATE · UNVERIFIED |
| A15 | stale live cache hit (400 d) | replaced, fully_provenanced | — | INSUFFICIENT_EVIDENCE → FLAG · REQUIRES_TOOL_VERIFICATION |
| A16a | cache: tenant correction | replaced, fully_provenanced | — | INSUFFICIENT_EVIDENCE → FLAG · PARTIALLY_PROVENANCED |
| A16b | cache: shadow-derived source | replaced, fully_provenanced | — | INSUFFICIENT_EVIDENCE → FLAG · UNVERIFIED |
| A17 | reasoning factual step | unchecked_reported_correct | contract: NOT_CHECKED; **production field unchanged** | see G2 |

**Pass criterion (plan §2):** none of the 17 fixtures may end in SUPPORTED, PASS_VERIFIED or replacement from an unknown, failed or degenerate input. **Met**: 0 of 19 route cases, plus A17 classified as NOT_CHECKED.

**Baseline failures:** all 19 B0 route cases show exactly their documented defect; 0 mismatches.

**How to read A9, A10 and A12.**
- In these cases SUPPORTED with FULLY_PROVENANCED is a legitimate B2 outcome. The Wikidata confirmation was well formed, with evidence and an entity id.
- The degenerate part is the follow-on replacement, and that is now blocked.
- That judge's accuracy is unmeasured. This is why PASS_VERIFIED stays unreachable.

**Refinements to the plan's "B1/B2 required" column.** Neither one lets anything through.
- **A6, missing description:** INSUFFICIENT_EVIDENCE, not VERIFIER_UNAVAILABLE. Nothing was there to judge.
- **A15:** the contract does not set a cache freshness window, because none is approved. *No* cache hit can be SUPPORTED, since the cache retains no evidence text, so staleness only adds a reason.

**Beyond the 17.** A regex-defect path test (TOKEN_SMUGGLING → SANITIZE_AND_RERUN) shows the refusal replacing the answer in B0 and contained in B2.

### 4.3 The one changed existing test

- **Test:** `tests/test_monitor_rag_fix.py::test_monitor_uses_rag_for_externally_verified_factual_mismatch`
- **Classification:** **conflict with an established public API contract, resolved by an owner decision** (OD-R2 → D-060).
- **Before:** it asserted `fix_applied is True` and the Wikipedia answer in `fixed_output`.
- **After:** it asserts the approved contract:
  - `fix_applied is False`;
  - `fixed_output` is the original;
  - the Wikipedia answer is in `proposed_output`;
  - `reliability.correction_applied is False`;
  - the action is not PASS_VERIFIED;
  - the summary is not `AUTO-FIXED`.
- The scenario is unchanged and the test asserts more than before.

## 5. Tests (Step 4)

| Suite | Result [RUN] |
|---|---|
| `tests/reliability/` (new) | 148 passed in the full run below; 149 after the regex-defect test was added (§4.2) |
| Targeted existing (`test_monitor_rag_fix`, `test_explainability`, `test_integration`, `test_core`) | all pass |
| Full offline suite (`pytest tests/ -m "not network"`, as in CI) | **1,640 passed, 2 failed, 24 skipped** (baseline: 1,494 passed, 24 skipped; the 146 extra passes are the new suite) |
| Hermetic Part A run | PASS: 20 cases, 0 guard violations, 0 B2 forbidden outcomes, 0 B0 mismatches |

**Failure classification.** Neither failure is a behaviour regression, and neither test was edited.

1. **`tests/security/test_answer_cache_isolation.py::test_a_tenant_gets_its_own_correction_back`**
   - **What fails.** The test asserts `fix_result.fixed_output` equals the tenant's own cached correction. Under D-060 the cache hit is contained: `fixed_output` is the original answer and the correction is in `proposed_output`.
   - **What still passes.** The test's tenant-isolation assertions on `ground_truth.from_cache` and `ground_truth.verified_answer` pass. The isolation property itself is intact.
   - **Classification:** **conflict with an established API contract, and a discovery requiring a separate decision.**
   - **Why the test was not edited.** OD-R2 forbids substituting `fixed_output`, but it does not name tenant-feedback corrections. Unlike the Wikipedia, Serper and Wikidata paths, these are human-supplied. So whether they should be exempt is the owner's call; see **OD-S0-1** in §8.
2. **`tests/test_runtime_policy.py::test_detector_source_files_are_untouched`**
   - **What fails.** This WP-004 tripwire asserts that `git status --porcelain` is empty for `fie/adversarial.py`, `fie/layers`, `fie/_lite.py`, `fie/onnx_encoder.py`, `fie/multilingual.py`, `fie/monitor.py`, `fie/integrations`, `evals`, `app`, `engine` and `storage`. It reports the uncommitted `app/` and `engine/` changes.
   - **Classification:** **an existing process guard, triggered as designed by uncommitted changes.**
   - **No detector file changed:** no `fie/` path appears in the output.
   - It will pass once the owner commits. If the owner reads it as "nothing under `app/` or `engine/` may change", this package needs that guard's scope agreed (OD-S0-2).

**Pre-existing finding (not caused by this patch).** `tests/test_monitor_rag_fix.py` attempts one DNS lookup for `failure-intelligence-system.onrender.com` when run on its own. The cause is the `fie` import-time telemetry ping, because that file does not set `FIE_NO_TELEMETRY`. CI does not catch this; tests in the repository root have no network guard. The new suite sets the variable and enforces a guard.

**Timing caveat on the baseline.** The baseline run ended at 14:09:23. My first two edits (14:07:50 and 14:08:12) fell inside its last 90 seconds. Both modules are imported lazily inside functions, so the baseline count is very likely clean, but that is not proven. The post-change full run is the authoritative result.

**Not run, by instruction:** Ollama inference, the E-R1 Part B benchmark, any external API.

## 6. API compatibility (requirement 7)

**Schema:** additive only.
- New fields: `FixResult.proposed_output` (`""`), `MonitorResponse.reliability` (`null` on early-return paths), `ReliabilityAssessment`.
- No field was removed, renamed or retyped.

**Behaviour changes, deliberate and owner-approved under OD-R2:**
1. `fix_result.fix_applied` is never `True` on `/monitor`; `fixed_output` equals the input answer.
   - SDK `correct` mode against an updated server therefore returns the original answer (tested).
   - The `correct` mode **contract** ("user gets the fixed answer") is effectively suspended.
   - Dashboards counting `fix_applied` (`Frontend/src/hooks/useData.js:78`) will show 0 corrections.
2. `requires_human_review` stays `True` where the Wikipedia fallback used to clear it.
   - More human-review e-mail notifications will be sent (`notify_human_review`, tested in A13 and A14).
3. `failure_summary` text:
   - `AUTO-FIXED: …` becomes `CORRECTION PROPOSED, NOT APPLIED (warn-only): …`;
   - "Model outputs are stable" becomes "Shadow ensemble unavailable — stability not assessed" when fewer than two shadows answered;
   - a cleared-escalation summary is now prefixed `HUMAN REVIEW REQUIRED:`.
4. `failure_signal_vector.provenance_label` is downgraded wherever FULLY_PROVENANCED is not justified.
5. **Adversarial rewrites are contained too.** A `SANITIZE_AND_RERUN` canned refusal is no longer delivered by the monitor. Blocking remains the job of the pre-flight guard, which runs first and is unchanged. Under the regex defect this rewrite fired on every jury run, so containing it removes harm.
6. Explanations:
   - Read `fix_applied=False`, so the external explanation now says a fix "was considered but not confidently applied".
   - Explanation confidence moves slightly. This is the `_compute_explanation_confidence` penalty.

**Unchanged by design:**
- `high_failure_risk`, `classifier_probability`, the jury and the GT `ground_truth` block.
- The signal log's values, which still record the proposal. In that log `fix_applied` now means "proposed"; see G5.

## 7. No external calls, downloads or additional models (Step 6.6)

- **pytest:** every reliability test runs under a socket guard; zero attempts were recorded.
- **Child process:** the hermetic child run recorded 0 violations, and its self-test proved the guard was armed.
- **No live services:** no Groq, Serper, Wikipedia, Wikidata, SendGrid or MongoDB call was made. No Ollama model was started.
- **No downloads:** no dataset or model weight was downloaded; Hugging Face was forced offline in the child.
- **Dependencies:** none added.
- **Local inference:** the only model executed was the existing local XGBoost classifier (and the pre-flight guard's local models), the same as in the existing suite.
- **Git:** no commit, push, publish or deploy.

## 8. Remaining gaps

| ID | Gap | Proposed handling |
|---|---|---|
| G1 | **SDK still substitutes on a legacy response.** `fie/monitor.py` `correct` mode returns `fixed_output` whenever a server says `fix_applied=True` (pinned by `test_b0_sdk_correct_mode_still_substitutes_a_legacy_response`). An updated server never says that; a not-yet-redeployed or self-hosted older server still can. `fie/` is the shipped package, so this was not changed | Owner decision: SDK hardening in a packaging release through the Workstream A process, or redeploy the server first |
| G2 | **A17: the reasoning dead branch.** `_verify_factual` still returns `is_correct=True, confidence=0` for an unchecked step. The API exposes that as `ReasoningStepResult.is_correct=True`, and the branch still spends GT calls. The contract classifies it NOT_CHECKED, but the field is unchanged, because fixing the branch would *activate* an unvalidated verdict path | Separate decision: remove the call, or gate it behind Part B |
| G3 | **The research pipeline keeps legacy correction semantics.** `engine/pipeline/langgraph_pipeline.py` (research only, not served) still applies corrections | Apply the contract before any future measurement that uses it |
| G4 | **The deployed server is unchanged until redeployed** | Owner deployment |
| G5 | **Signal-log semantics.** `fix_applied` in the log now means "the engine proposed a correction" | Documented in code. Revisit if the log is used for retraining |
| G6 | **The question classifier labels "current CEO … right now" FACTUAL.** The contract compensates with an explicit `time_sensitive` flag (jury temporal, TEMPORAL type or live category) | Separate classifier issue |
| G7 | **CONFLICTING_EVIDENCE is unreachable.** The pipeline consults one source per request. The rule is fixed in `combine_states` and unit-tested | Activates with multi-source checks |
| G8 | **Parts A and B measure different things.** Part A proves contract compliance on synthetic inputs, not accuracy. B0 runs the old route on the new engine; it does not reproduce the deployed system | Part B |

**Owner decisions raised by this stage:**

- **OD-S0-1 — Tenant corrections under warn-only.** Should a tenant's own feedback correction (cache `source_class = tenant_feedback`) still replace the answer? Today it does not.
  - Option (a), recommended: keep warn-only, and update `test_a_tenant_gets_its_own_correction_back` to assert `proposed_output`. The isolation assertion stays as strong as before.
  - Option (b): exempt exact-match tenant corrections. Semantic matches at 0.92 and the missing TTL make a broader exemption unsafe.
- **OD-S0-2 — Scope of the WP-004 tripwire.** Does it permit Workstream B changes under `app/` and `engine/` once committed? It passes after commit as written.
- **OD-S0-3 — G1.** Harden the SDK, or deploy the server first.
- **OD-S0-4 — G2.** Remove the dead factual-step GT call, or keep it until Part B.
- **OD-S0-5 — Erratum for audit v2 line 7, and restoring the 2026-10-20 files (§1.1).**

## 9. Workstream A boundary (Step 5)

- **The regex.** `engine/agents/adversarial/patterns.py:94` is still `r"\x00||"`, untouched. Its empty alternative matches every input, so the jury diagnoses TOKEN_SMUGGLING on every run.
- **Dependency.** Fixing the regex re-routes real traffic into the Serper, Wikidata, shadow-consensus and Wikipedia correction paths. Before this containment, every one of those paths could replace answers (A1–A16).
- **Order.** The containment must therefore be merged, deployed and verified first. Only then should Workstream A fix the regex under its own process, with recall and over-refusal reported against `BL-0002_pair-v6.3b_fie-70de7053`.
- **Workstream A artifacts.** The canonical baseline, C2 annotation artifacts, adversarial data, models and thresholds were not touched.
- **No shared score.** The reliability states are disjoint from ALLOW / REVIEW / BLOCK (tested), and no combined score was introduced.

## 10. Next recommended step: E-R1 Part B

Part B is unchanged from the plan, and nothing has been downloaded. It needs these owner decisions, in order:
- **OD-R4:** the dataset. Recommended: **FEVER** as primary, for its gold Wikipedia evidence and three-way labels. HaluEval-QA with `knowledge` should be considered second, because it matches FIE's question-answer framing.
- **OD-R7:** thresholds. SUPPORTED needs a Wilson lower bound ≥ 0.95 at ≥ 0.60 coverage; CONTRADICTED needs a lower bound ≥ 0.90.
- **OD-R8:** the data location.
- **OD-R6:** local compute (Ollama, about 3–13 h).
- **OD-R5:** optionally, an NLI model.

Part A's assets carry over directly:
- `parse_verdict_strict` is arm V1.
- The verifiers' `_heuristic_match` / `_parse_groq_verdict` are arms V2 and V0.
- A judge that passes Part B is the only route to a non-empty `VALIDATED_VERIFIERS`, and so to PASS_VERIFIED.

## 11. Files

**New:**
- `engine/reliability_contract.py`
- `tests/reliability/{__init__,conftest,harness,e_r1_part_a,test_e_r1_part_a,test_reliability_contract}.py`
- `research/reliability/e_r1_part_a/run_part_a.py`
- `research/reliability/e_r1_part_a/results/20261027T085419Z_part_a.json`
- this report

**Modified:** the 8 files in §2.

**Decision log:** §15 added.

Everything is uncommitted.
