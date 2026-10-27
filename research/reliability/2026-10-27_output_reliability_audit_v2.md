# FIE Output Reliability — Independent Audit v2, Candidate Comparison and Research Strategy

**Date:** 2026-10-27
**Scope:** Workstream B (output reliability) only. Workstream A (adversarial-input detection, C2) is out of scope. No C2 artifact, canonical baseline, threshold or model was touched.
**Status:** Read-only analysis and planning. No production code, configuration, dataset or prior report was changed. No external API, Serper, Groq or web call was made, and nothing was downloaded. The only computation was local, read-only analysis of existing JSONL logs under `data/hallucination_eval/`; the commands are in Appendix A.
**Companion plan:** [2026-10-27_E-R1_verifier_benchmark_plan.md](2026-10-27_E-R1_verifier_benchmark_plan.md)
**Prior work:** [2026-10-20_output_reliability_audit.md](2026-10-20_output_reliability_audit.md) and [2026-10-20_verifier_replay_experiment_plan.md](2026-10-20_verifier_replay_experiment_plan.md). Both are preserved unchanged. §2 lists what this audit confirms, corrects and adds.

### Evidence labels used throughout

| Label | Meaning |
|---|---|
| **[CODE]** | Confirmed by reading current source at the cited location |
| **[DATA]** | Derived in this audit from an existing local artifact (reproducible from Appendix A) |
| **[REPORT]** | Stated in an existing repository report; not independently reproduced |
| **[LIT]** | Published research, cited from memory; check each citation before external use |
| **[HYP]** | Hypothesis: plausible but unmeasured, with the experiment that would resolve it |
| **[REC]** | Recommendation |

---

## 1. Executive summary

The prior audit's central claim is correct: FIE's verifiers turn *unknown* into *consistent*. This audit finds that the more serious problems sit one level up, in **when FIE checks evidence at all** and **what it does with the result**. The local evaluation logs show that these problems matter more than parser semantics.

**The five most important verified facts:**

1. **Evidence verification is gated on shadow disagreement, so the failure mode ensembles cannot see is never checked.** [CODE] [langgraph_pipeline.py:632-640](../../engine/pipeline/langgraph_pipeline.py#L632-L640) and [monitor.py:395](../../app/routes/monitor.py#L395) only reach Wikidata/Serper when `high_failure_risk` is set. That flag comes from primary-vs-shadow disagreement or entropy. When shadows agree with a wrong primary answer (the shared-misconception case), or when fewer than two shadows respond, no external check runs. [DATA] In the frozen TruthfulQA run, **59/140 wrong answers (42.1%) reached verification versus 56/140 correct answers (40.0%)**. Selection into verification is no better than chance, and 57.9% of wrong answers were never checked. All 112 zero-shadow requests across two runs skipped verification and were returned as if stable.

2. **FIE's corrections do not target wrong answers better than chance, and they reach end users.** [DATA] Across the two signal-level runs, automatic corrections replaced **31 correct answers and 26 hallucinated ones**. Wikidata "GT_VERIFIED" overrides, at self-reported confidence 0.8–1.0, replaced **16 correct and 14 hallucinated** answers; on HaluEval, 13 of 20 overrides hit correct answers. The existing ablation reports `gt_contradiction` precision **0.393** (11 TP / 17 FP) [REPORT `signal_ablation_halueval.json`]. [CODE] SDK `correct` mode returns `fixed_output` to the caller ([fie/monitor.py:290-299](../../fie/monitor.py#L290-L299)).

3. **Every July baseline ran with a broken jury.** [CODE] [engine/agents/adversarial/patterns.py:94](../../engine/agents/adversarial/patterns.py#L94) contains the regex alternative `r"\x00||"`. Its empty alternative matches every string; locally, `'What is the capital of France?'` produces a `TOKEN_SMUGGLING` match. The regression came in `ccee211` (2026-05-22, "updated project structure"), which mangled `\x00|\u0000|%00`. [DATA] `root_cause = TOKEN_SMUGGLING` for **345/345** non-blocked jury runs. As a result, 26 benign factual answers were replaced by a canned "I can help you with legitimate questions" response (`SANITIZE_AND_RERUN`). The temporal Serper branch and the Wikipedia-correction branch **never executed**: zero `serper.dev` calls were recorded. The published AUCs (0.497 TruthfulQA, 0.396 HaluEval) therefore measure a configuration that is not the intended design. [CODE] Neither the shipped `fie/` guard nor `evals/` imports this module, so canonical Workstream-A guard numbers are very likely unaffected.

4. **Fixing that regex alone would switch on latent harmful paths.** [CODE] With real root causes restored, `/monitor` would start running a Wikipedia fallback that **overrides human escalation**: it clears `requires_human_review` after a "could not verify" outcome ([monitor.py:539-565](../../app/routes/monitor.py#L539-L565)). It also runs after a Wikidata *confirmation*. Shadow-consensus correction would also become active, and with exact-text voting it degenerates into "the most self-confident single shadow wins" (§4.2). The Workstream-A fix and Workstream-B containment must therefore be sequenced together.

5. **No local dataset can measure evidence support.** [DATA] The frozen TruthfulQA set has no evidence passages, and the guard blocks 29.5% of it [REPORT]. The frozen HaluEval set has no passages and a total style confound: correct candidates have a median of 2 words (1% end with a period), hallucinated ones 9.5 words (95% do), and no question appears with both labels. The local `data/datasets/halueval.json` has had its `knowledge` field stripped and is documented as corrupted [REPORT, optimization log §3b]. Nothing in the repository can currently answer "does verifier X correctly decide supported versus unsupported given evidence?"

**What this means for the hypotheses:**

- **Shadow ensembles.** Your concerns are confirmed in code and data. Agreement is never checked against evidence, disagreement cannot pick a winner, and shadow failure is fail-open. Adding models would not help. [DATA] On HaluEval, `ensemble_dissim` AUC is 0.209 and `embedding_distance` 0.192. Those are strongly *inverted*: hallucinated candidates resemble shadow outputs more than correct ones do. That is a style signal, not a truth signal.
- **Wikipedia/Serper fallback.** Your concern is confirmed and is worse than stated. The fallback is not verification. It generates a replacement answer from one summary using an 8B model told to "answer directly and confidently", and on `/monitor` that output *overrides* escalation. Serper keeps one snippet, uses no authority, freshness or provenance, and overrides on an `INCONSISTENT` verdict at any confidence.
- **RAG.** FIE already has crude retrieve-then-generate. The gap is not missing RAG; it is missing **evidence-to-claim verification with a measured-precision judge** and a **policy that does not act without it**. FIE monitors an answer it did not generate (§6.1), so the relevant form is **post-hoc retrieval for verification**, not retrieval before generation.

**Recommendation.**

- **[REC] Improvement 1 — evidence-gated decision and action contract.** Use explicit per-claim evidence states. Treat shadows as a routing signal only. Never approve or replace an answer without a source-identified, claim-linked passage judged by a verifier whose precision has been measured. Treat failures as `VERIFIER_UNAVAILABLE`. Make the monitor warn-only until then.
- **[REC] First experiment — E-R1, gold-evidence verifier benchmark.** Run offline on local Ollama (no Groq). It asks whether FIE's current verifier semantics, a strict tri-state parser, the current lexical fallbacks, a quote-grounded judge and optionally a local NLI model can tell SUPPORTS from REFUTES from NOT-ENOUGH-INFO *when handed the correct evidence*. Every candidate approach (RAG, claim-level checking, correction, routing) assumes a judge with known precision; this establishes whether one exists before anything is built on it. It needs one approval: a public evidence-labelled dataset.
- **[REC] Second-best — structured-fact verification.** Use Wikidata *property values*, unit-normalised numbers and dates instead of an LLM reading a one-line description. Coverage is narrower, but precision is likely high and deterministic. Pursue this if no judge passes E-R1.
- **[REC] Not first:** a vector database, more shadow models, general web RAG, or any correction or regeneration work.

---

## 2. Relationship to the 2026-10-20 audit

| Prior claim | Status in this audit |
|---|---|
| Serper `_parse_groq_verdict` turns UNCERTAIN, absent or malformed verdicts into `matches=True`; the confirmation path has no confidence gate | **Confirmed** [CODE] [serper_verifier.py:192-216](../../engine/verifier/serper_verifier.py#L192-L216), [ground_truth_pipeline.py:224-233](../../engine/verifier/ground_truth_pipeline.py#L224-L233) |
| The Wikipedia correction bypasses verification and uses fixed 0.92/0.72 confidence | **Confirmed, and more severe:** on `/monitor` it also overrides escalation and runs after a confirmation (§4.4) |
| Wikidata compares against an entity *description*, not property values | **Confirmed, and more severe:** the "correct value" comes from an 8B model, which is parametric knowledge labelled `FULLY_PROVENANCED` (§4.3) |
| A temporal question with Serper disabled escalates | **Confirmed in code, never exercised:** temporal routing needs `root_cause == TEMPORAL_KNOWLEDGE_CUTOFF`, which never occurred (§4.6) |
| A failed `gt_verify` routes to finalize, "avoiding treating an error as success" | **Corrected:** it also avoids escalation; the request finalizes silently with no "verification unavailable" state |
| Shadow models: Llama-3.3-70B, GPT-OSS-120B, Qwen3-32B | **Confirmed** through `_MODEL_ALIASES` ([groq_service.py:109-117](../../engine/groq_service.py#L109-L117)) |
| AUC 0.497 shows agreement is not reliability | **Confirmed, with a new caveat:** measured under the jury regex defect (§4.6) |
| First experiment: tri-state replay on authored contract cases | **Partly disagreed.** It is necessary hygiene, but a B1 parser passes an authored matrix by construction, and the labelled-set gate depends on data that does not exist. Folded into E-R1 as Part A; E-R1 Part B supplies the informative measurement (§7) |
| *Not in the prior audit* | Disagreement-gated verification; shadow-failure fail-open; degenerate exact-text consensus; correction not targeting errors; jury regex defect; cache without TTL serving `FULLY_PROVENANCED`; Serper override at any confidence and heuristic override; dead factual-step verifier; unreachable self-consistency branch; HaluEval style confound in the frozen set; 8B judge over larger models |

---

## 3. Verified map of the current implementation

### 3.1 Two production surfaces, one measured

- **Served endpoint:** `/api/v1/monitor` → [app/routes/monitor.py](../../app/routes/monitor.py), an inline implementation.
- **Measured pipeline:** LangGraph [engine/pipeline/langgraph_pipeline.py](../../engine/pipeline/langgraph_pipeline.py). Its only callers are [scripts/measure_pipeline_baseline.py](../../scripts/measure_pipeline_baseline.py) and the uncalled `engine/pipeline/realtime_pipeline.py`. [CODE] This was already recorded in the rebuild audit (§5.10, D-001).
- **The two paths differ** in ways that matter. `/monitor` passes a tenant `scope` to the GT pipeline, so the answer cache is live there; LangGraph does not, so its cache is inert. `/monitor` runs the Wikipedia fallback after escalation; LangGraph does not. Their verification gates also differ (§3.3).

### 3.2 Data flow from answer to verdict

```
caller ── prompt + primary_output (FIE never generates the primary answer) ──►
  ① shadow fan-out: 3 Groq models, prompt + canary system msg + "CONFIDENCE: HIGH/MEDIUM/LOW" suffix
       └ failed/timeout shadow → dropped silently (successful ones kept)
  ② signals: SBERT clustering → agreement, entropy, primary_outlier (needs ≥2 shadows)
       high_failure_risk = primary_outlier OR entropy ≥ threshold
  ③ reasoning_verify (FACTUAL/REASONING): step decomposition; factual steps call the GT pipeline (results never used, §4.7)
  ④ jury (if run_full_jury): adversarial specialist → root_cause (always TOKEN_SMUGGLING today, §4.6)
       domain critic: Wikipedia summary → 8B answer → compare (jury feature only)
  ⑤ gate: high_failure_risk AND (jury_conf ≥ 0.45 | no jury) AND type has external sources
  ⑥ GT pipeline: cache → [temporal? Serper] → claim extraction (1 claim, 8B) → Wikidata → shadow consensus
  ⑦ action: GT answer ≠ primary → replace | else fix_engine (strategy by root_cause) | else Wikipedia
  ⑧ XGBoost over the pipeline's own action features → final high_failure_risk flag
```

### 3.3 The ten required determinations

| # | Question | Finding | Evidence |
|---|---|---|---|
| 1 | Mechanisms implemented | Shadow ensemble with semantic clustering and entropy; step-level reasoning verifier; 3-agent jury; GT pipeline (cache, Serper, Wikidata, shadow consensus, self-consistency); fix engine (5 strategies); Wikipedia RAG correction; XGBoost post-hoc classifier; provenance labels | [CODE] files in §3.1–3.2 |
| 2 | Shadow models, selection, failure | Fixed list of 3 Groq models (`config.groq_models`, aliased); no per-query selection; 30 s timeout; retries only on 429; a failure is dropped silently. Verifier, claim extractor and RAG model are `llama-3.1-8b-instant`, smaller than the models it judges and the same family as one shadow | [CODE] [config.py:83-92](../../config.py#L83-L92), [groq_service.py:173-320](../../engine/groq_service.py#L173-L320) |
| 3 | Wikipedia/Serper triggers | **Serper:** only when `root_cause == TEMPORAL_KNOWLEDGE_CUTOFF` and the prompt is not in a 9-pattern "permanent fact" list; disabled by default in `.env.example`. **Wikidata:** non-temporal path with an extracted claim. **Wikipedia:** (a) jury domain-critic layer on every jury run (280/280 requests); (b) correction when root cause ∈ {KNOWLEDGE_BOUNDARY_FAILURE, FACTUAL_HALLUCINATION}, no fix applied and no live-data pattern | [CODE] [ground_truth_pipeline.py:168](../../engine/verifier/ground_truth_pipeline.py#L168), [domain_critic.py:211-245](../../engine/agents/domain_critic.py#L211-L245), [monitor.py:539-543](../../app/routes/monitor.py#L539-L543); [DATA] host counts |
| 4 | Evidence passed between stages | Strings only: one Serper snippet (≤400 chars into the judge), one Wikidata description, one Wikipedia summary (≤1000 chars). No URL, revision, timestamp, rank, passage offsets or claim links survive; the user sees a free-text `source` label | [CODE] `SerperResult`, `WikidataResult`, `GroundingResult` dataclasses |
| 5 | How results are combined | **Ordered short-circuit**, not fusion: cache > Serper > Wikidata > shadow consensus. Consensus = exact lowercased-text vote weighted by self-reported confidence. The final flag is XGBoost over action features (`fix_applied`, `gt_source`, `fix_strategy`, `jury_verdict_str`), so it scores the pipeline's own decisions | [CODE] [ground_truth_pipeline.py:407-430](../../engine/verifier/ground_truth_pipeline.py#L407-L430), [langgraph_pipeline.py:909-932](../../engine/pipeline/langgraph_pipeline.py#L909-L932) |
| 6 | Claim-level or answer-level? | Answer-level. One "MAIN" claim (subject/property/value) from the first 500 characters; the Serper judge sees the first 300 characters of the answer; multi-claim answers are not decomposed | [CODE] [claim_extractor.py:27-45, 81-87](../../engine/claim_extractor.py#L27-L87), [serper_verifier.py:159](../../engine/verifier/serper_verifier.py#L159) |
| 7 | Contradictions, outages, empty results, stale pages, timeouts, malformed output | No multi-source comparison, so conflicting evidence is unrepresentable. Serper: none/timeout/exception → escalate; malformed or UNCERTAIN → "confirmed". Wikidata: any failure → (True, 0.0) → inconclusive → falls through to shadow consensus. No freshness logic anywhere; the cache has no TTL | [CODE] §4.1–4.5 |
| 8 | Is missing/failed verification read as reliable? | **Yes, in several places:** Serper parse default; Serper zero-token heuristic; Wikidata or verifier unavailable → shadow consensus; `<2` shadows → not high-risk → no verification; failed `gt_verify` → silent finalize; step verifier defaults to `is_correct=True` | [CODE] §4.1, §4.2, §4.7 |
| 9 | Can it explain itself and abstain safely? | Partly. `pipeline_trace` and `escalation_reason` give a narrative, and escalation keeps the original answer. But provenance labels overstate (cache → `FULLY_PROVENANCED`; Wikidata override = 8B output) and `/monitor` cancels escalation via Wikipedia. There is no state for "insufficient", "conflicting" or "unavailable" in the response | [CODE] §4.3–4.5 |
| 10 | Tests and evaluations supporting the claims | `tests/test_monitor_rag_fix.py` mocks Wikipedia and asserts the correction is applied (wiring only); numeric-unit checks; fix-engine strategy forcing; jury priority. `test_ground_truth.py` and `test_provenance_phase3.py` were deleted in `1cc4036` (only `.pyc` remain). No test covers malformed or uncertain verdicts, outages leading to correction, cache freshness or consensus degeneracy. Outcome evaluations: AUC 0.497 / 0.396, both at or below chance, under the regex defect | [CODE] `tests/`; `git log --diff-filter=D`; [REPORT] baseline reports |

---

## 4. Confirmed defects (with failure scenarios)

Severity reflects output-reliability impact assuming the jury regex is fixed: several defects are latent today only because of it.

### 4.1 Verification is conditioned on disagreement and on shadow availability — **Critical**

- **[CODE]** `_route_to_verify` returns `gt_verify` only if `high_failure_risk` ([langgraph_pipeline.py:632-640](../../engine/pipeline/langgraph_pipeline.py#L632-L640)). `/monitor` raises "Auto-fix skipped: primary output matches shadow consensus" when it is false ([monitor.py:395-396](../../app/routes/monitor.py#L395-L396)).
- **[CODE]** `is_primary_outlier` returns False with fewer than 2 shadows ([consistency.py:175-176](../../engine/detector/consistency.py#L175-L176)), and a single output has entropy 0. A Groq outage, rate limit or missing key therefore yields "stable".
- **[DATA]** Verification reach in the frozen TruthfulQA run: wrong 42.1% vs correct 40.0%. All 112 zero-shadow requests (85 TQA-sig, 27 HaluEval) were finalized without verification.
- **Failure scenario.** The primary model answers a TruthfulQA misconception and two shadows repeat it. The outlier test is false, entropy is low, and FIE returns the answer with no evidence check and no "unverified" state.
- **Why it matters.** It contradicts the purpose of evidence checking. [LIT] TruthfulQA found larger models *more* likely to reproduce imitative falsehoods (Lin et al., 2022), and work on correlated errors across LLM families finds high agreement on wrong answers (Kim et al., 2025, "Correlated Errors in Large Language Models"). The regime where agreement is least informative is exactly the one excluded from verification.

### 4.2 Consensus correction degenerates to a single self-confident shadow — **High (latent)**

[CODE] `_apply_shadow_consensus` groups by `output.strip().lower().rstrip(".,!?")[:200]` ([fix_engine.py:141-152](../../engine/fix_engine.py#L141-L152)). Free-text answers almost never match exactly, so each shadow forms its own group. The gate is `strength < 0.50 → escalate`:

| Surviving shadows and self-reported weights | Strength | Outcome |
|---|---|---|
| 3 shadows, [2, 2, 2] | 0.33 | escalate |
| 3 shadows, [3, 2, 1] | **0.50** | the HIGH shadow replaces the primary |
| 3 shadows, [3, 1, 1] | **0.60** | the HIGH shadow replaces the primary |
| 2 shadows, [2, 2] | **0.50** | the *first in sort order* (Llama-3.3-70B) replaces the primary |
| 1 shadow | **1.00** | that shadow replaces the primary |

- **Shadow failure raises correction confidence.** [DATA] Single-shadow requests produced `shadow_consensus` strength 1.0 (e.g. `truthfulqa-0021-correct`, `truthfulqa-0191-correct`).
- **The detector and the corrector use different notions of "same answer"** (SBERT clustering vs exact text). [CODE] `_apply_prompt_decomposition` with `model_fn=None`, as both production paths pass it, returns `shadow_outputs[0]` as the fix ([fix_engine.py:493-495](../../engine/fix_engine.py#L493-L495)).
- [CODE] Self-reported HIGH/MEDIUM/LOW is used as a vote weight without calibration ([groq_service.py:121-129](../../engine/groq_service.py#L121-L129)).

### 4.3 Provenance labels overstate the evidence — **High**

- **Wikidata override.** [CODE] `CORRECT_VALUE` is produced by `llama-3.1-8b-instant` from a short entity description ("telecommunications device", "American inventor") that usually does not contain the property value. When the description lacks the answer, the 8B model either answers from parametric memory or abstains. Either way the result is labelled `FULLY_PROVENANCED` with source "Wikidata (Qxxx)" ([wikidata_verifier.py:204-251](../../engine/verifier/wikidata_verifier.py#L204-L251), [ground_truth_pipeline.py:285-295](../../engine/verifier/ground_truth_pipeline.py#L285-L295)). [DATA] These overrides fired at 0.8–1.0 confidence on correct answers.
- **Wikidata entity choice.** [CODE] The first non-"creative work" hit wins, with a fallback to the first candidate. There is no disambiguation against the question.
- **Cache.** [CODE] Any cache hit is labelled `FULLY_PROVENANCED`, whatever its stored source ([ground_truth_pipeline.py:149-160](../../engine/verifier/ground_truth_pipeline.py#L149-L160)). Writes come from Serper or Wikidata answers at confidence ≥ 0.90, and those overrides reached 1.0 on correct answers.

### 4.4 The Wikipedia fallback replaces answers and overrides escalation — **High (latent on `/monitor`)**

- [CODE] On `/monitor` the fallback runs whenever no fix has been applied: after a GT escalation, after a fix-engine escalation, and **after a Wikidata or Serper confirmation**. It then sets `requires_human_review = False` ([monitor.py:539-565](../../app/routes/monitor.py#L539-L565)). The code comment documents this as a deliberate behaviour change.
- [CODE] `ground_with_wikipedia` returns `success=True` with the raw summary (≤1000 chars, confidence 0.72) when the 8B model says the context is insufficient or when Groq fails ([rag_grounder.py:40-55](../../engine/rag_grounder.py#L40-L55)). Retrieval is the direct REST summary of the de-questioned prompt, or else the top-1 search hit. There is no disambiguation-page check, revision, relevance score or authority check ([retriever.py](../../engine/rag/retriever.py)).
- [CODE] The RAG prompt instructs: "answer directly and confidently… Do not say the context is insufficient unless the answer truly is missing" ([rag_pipeline.py:22-30](../../engine/rag/rag_pipeline.py#L22-L30)). That biases the model against abstention.
- **Failure scenario.** A question about a less-known entity: shadows disagree, Wikidata finds the wrong same-name entity, and the GT pipeline escalates. Wikipedia then fetches the top search hit, whose summary describes a different entity, and the 8B model writes a confident answer from it. FIE returns that answer at 0.92, the escalation is cancelled, and `correct` mode delivers it to the user.
- [CODE] On the LangGraph path, Wikipedia is strategy 3 inside `auto_correct` and is not reached after escalation, which is another divergence between the two paths.

### 4.5 Serper and cache semantics fail open toward approval *and* override — **High (latent)**

- **Approval side** (prior audit, confirmed). UNCERTAIN, missing or malformed verdicts mean "confirmed" and produce `FULLY_PROVENANCED` with no confidence gate. The zero-token heuristic returns (True, 0.0): an answer like "42" is "confirmed" at confidence 0.
- **Override side (new).** [CODE] An `INCONSISTENT` verdict overrides the primary with the 8B `GROUNDED_ANSWER` at any confidence, including 0.0 ([ground_truth_pipeline.py:212-223](../../engine/verifier/ground_truth_pipeline.py#L212-L223)). When Groq is unavailable, `_heuristic_match` divides overlap by the snippet length. A correct short answer ("Paris") against a long snippet therefore scores ≤ 0.25 → "inconsistent" → **the raw snippet replaces the answer**, labelled `FULLY_PROVENANCED`.
- **Freshness (new).** [CODE] The GT cache has no TTL or as-of check ([ground_truth_cache.py](../../engine/ground_truth_cache.py)). Lookup happens *before* temporal routing, and semantic matching at cosine ≥ 0.92 (MiniLM) can link "CEO of X in 2023" to "CEO of X now" [HYP: measure the near-duplicate collision rate]. A Serper answer to a temporal question is replayed indefinitely as `FULLY_PROVENANCED`.

### 4.6 The jury regex defect contaminates routing and all baselines — **Critical for measurement**

[CODE] [patterns.py:94](../../engine/agents/adversarial/patterns.py#L94) contains `r"\x00||"`, whose empty alternative matches everything. It carries `base_confidence = 0.90`, which suppresses the LLM intent layer and flows into `jury_confidence`. [DATA] `root_cause = TOKEN_SMUGGLING` in 345/345 jury runs. Consequences:

- **`SANITIZE_AND_RERUN`** replaced 26 benign answers (14 correct, 12 hallucinated) with a canned refusal.
- **Temporal and Wikipedia-correction branches never ran.** `jury_confidence` (median 0.82 TQA, 0.71 HaluEval) reflects the adversarial specialist, not reliability.
- **XGBoost** saw a constant `jury_verdict_str`.

Implications:

1. The published baselines cannot attribute performance to components.
2. Fixing the regex activates §4.2 and §4.4 in the same release.
3. This sits in Workstream-A code. It is reported here, not changed. [CODE] The `fie/` guard and `evals/` do not import it.

### 4.7 Dead or unreachable verification code still costs latency — **Medium**

- [CODE] `_verify_factual` in the reasoning verifier calls the full GT pipeline (Groq claim extraction plus Wikidata) per factual step. It then accepts the result only if `gt.source in ("wikidata", "serper")`, but the pipeline never emits those exact strings, so it always returns `is_correct=True, confidence=0.0` ([step_verifier.py:213-263](../../engine/reasoning/step_verifier.py#L213-L263)). [REPORT] `reasoning_confidence` AUC was exactly 0.5 [0.5, 0.5]. `reasoning_verify` costs p50 3.0 s and p95 12.7 s per request [REPORT baseline_report.json].
- [CODE] The GT pipeline's REASONING/CODE self-consistency branch is unreachable from both production paths: both skip GT when Wikidata and Serper are disabled for the type, which `pipeline_gates` does for REASONING and CODE.

### 4.8 Measurement assets cannot test evidence-grounded reliability — **High for research**

[DATA/REPORT] Neither frozen set carries evidence, claim spans or source labels.

- **TruthfulQA:** 29.5% guard-blocked; "correct" candidates include non-answers ("I have no comment").
- **HaluEval:** style-separable (2 vs 9.5 median words); unpaired by question; its local raw file is documented as corrupted. [DATA] The first raw row's `correct_answer`, "First for Women was started first.", looks like a hallucinated answer: Arthur's Magazine (1844) predates it. That is a single observation, not audited.
- **Signal-level data:** the TQA-sig run had 178/250 zero-shadow requests, so its AUC of 0.54 rests on n = 40.

---

## 5. Hypotheses that need experiments

| ID | Hypothesis | Why plausible | Resolving experiment |
|---|---|---|---|
| H1 | Given gold evidence, an 8B judge with a strict tri-state parser reaches SUPPORTED precision ≥ 0.95 with a 95% lower bound, at ≥ 60% coverage | [LIT] Small fact-checkers do well on grounding documents (MiniCheck, Tang et al., 2024; AlignScore, Zha et al., 2023) | **E-R1** |
| H2 | The current parser labels most NOT-ENOUGH-INFO items as consistent | `matches` defaults to True; UNCERTAIN maps to True | E-R1, V0 vs V1 on the same generations |
| H3 | Requiring a verbatim quote of the supporting span (checked deterministically) cuts false SUPPORTED more than prompt changes alone | It removes the judge's ability to assert unseen support | E-R1, V3 |
| H4 | A local NLI cross-encoder matches or beats the 8B judge on SUPPORTED precision at a fraction of latency, but fails on numbers, dates and negation | [LIT] SummaC and AlignScore findings; known NLI weaknesses | E-R1, V4 (needs model approval) |
| H5 | Judge errors are correlated across families, so judge-panel unanimity adds less precision than independence would predict | [LIT] Correlated-error and self-preference findings (Kim et al., 2025; Panickssery et al., 2024) | E-R1, V5 (error κ across judges) |
| H6 | Wikipedia summaries and Wikidata descriptions contain the evidence needed for most single-hop factual claims less than half the time | Summaries are lead paragraphs; descriptions are a few words | E-R2: evidence recall of current sources against gold evidence |
| H7 | Disagreement-gated verification misses most verifiable wrong answers | [DATA] 57.9% of wrong answers unverified | E-R3: verify-all vs gated on the same items, after a judge passes E-R1 |
| H8 | Warn-only (no correction) has lower net harm than current correction | [DATA] Corrections hit correct answers at ≥ the base rate | E-R3 with replacement-correctness labels |
| H9 | Semantic cache matches at 0.92 serve wrong answers for temporal near-duplicates | Embedding similarity ignores dates and qualifiers | Offline: embed paired temporal variants, count collisions |
| H10 | The XGBoost flag is near chance mainly because its inputs are the pipeline's own (broken) actions, not because detection is hopeless | Constant jury feature; action-derived features | Re-run the signal ablation after the regex triage, on raw signals only |

---

## 6. RAG for FIE: benefits and limits

### 6.1 Position in the pipeline

FIE receives `primary_output` from the caller ([fie/monitor.py:111, 179](../../fie/monitor.py#L111)); it does not generate the answer. "Retrieve before generation" therefore applies only if FIE becomes a generation gateway, or within a correction step. **[REC]** Use retrieval *after* generation, for claim verification. Pre-generation RAG is a separate product decision (a gateway mode), not a reliability fix.

### 6.2 Where retrieval helps

**Benefit.** Retrieval supplies evidence the judges lack. It is the only route to checking agree-but-wrong answers (§4.1), live facts, and tenant-specific facts. [LIT] FActScore-style decomposition with retrieval (Min et al., 2023) and search-augmented evaluators (SAFE, Wei et al., 2024) show that per-claim evidence checking tracks human factuality judgments better than answer-level scoring.

### 6.3 Where retrieval does not help

Retrieval does not make a judge accurate, a source authoritative, or a passage relevant. CRAG (Yan et al., 2024) exists because retrieved documents are often wrong or irrelevant and need an evaluator. Retrieval cannot resolve conflicting sources or answer questions with no authoritative answer, and it adds latency and egress. A RAG stack built before H1 is known would rest on an unmeasured judge.

### 6.4 Retrieval technology

Choose it by evidence type, not fashion:

| Evidence need | Appropriate source and retrieval | Vector DB needed? |
|---|---|---|
| Entity–property–value facts | Structured KB lookup (Wikidata property values with qualifiers and time) | No |
| Encyclopedic single-hop facts | Full-article passages (not summaries) with BM25 or entity-linked section retrieval, then rerank | Only if BM25 recall fails |
| Versioned technical facts | Official docs or release notes, pinned by version | No (small corpus) |
| Live or temporal facts | Dated search with source allow-lists, multi-source, as-of timestamp | No |
| Tenant facts | Tenant-provided corpus, authorised only | Possibly, at scale |

### 6.5 The verifier (the actual bottleneck)

Order of trust:

1. **Deterministic checks:** numbers, units, dates, string identity against structured values.
2. **NLI or alignment models**, for passage-level entailment.
3. **An LLM judge**, for nuanced relations. It must give a quote-grounded rationale and parse into three states.

Source authority is a *separate* policy decision, never inferred from entailment.

---

## 7. Comparison of improvement candidates

Effort and latency are relative engineering estimates [HYP]; none are measurements.

| Candidate | Failure addressed (§4) | Expected benefit; strength of evidence | Main failure modes | Complexity | New requirements | Latency/cost | External/privacy | Eval difficulty | Depends on | Initial? |
|---|---|---|---|---|---|---|---|---|---|---|
| **P. Evidence-gated decision and action contract** (explicit states; shadows route only; no approve/override without a measured-precision verdict; failures → UNAVAILABLE; warn-only until measured) | 4.1, 4.2, 4.3, 4.4, 4.5 | Removes fail-open approval and override. Strong: code plus [DATA] showing corrections at or below chance | More "unverified" outcomes (lower coverage); a contract is only as good as the judge | Low–Med | None | ≈0, saves calls | Reduces egress if correction calls drop | Easy for the contract; needs E-R1 to set thresholds | E-R1 thresholds | **Yes (with E-R1)** |
| **A. Better shadow ensemble** (more or different models, smarter aggregation) | Partly 4.1 and 4.2 | Low. [DATA] Agreement signals inverted or at chance; [LIT] correlated errors | Shared errors; style artefacts; cost | Med | More API models | +1 call per model; worse tail | More third-party exposure | Hard (needs correlation study) | — | **No**; keep as a routing signal and measure later |
| **B. Claim decomposition and claim-level verification** | 4.3, §3.3 #6 | High for long answers. [LIT] FActScore, SAFE, FactSelfCheck | Extractor omission and distortion; dependent claims; costly labels | Med | Decomposer (local LLM); claim labels | +1 call plus a per-claim judge | Local possible | Med–hard | E-R1 judge | Second stage (E-R3) |
| **C. Evidence-grounded verification / post-hoc RAG** | 4.1 (agree-but-wrong), 4.4 | High *if* the judge and source recall suffice. [LIT] supportive; FIE-specific unknown | Irrelevant or stale retrieval; false authority; latency | Med–High | Corpus or source adapters; retrieval index | +0.5–5 s per claim [HYP] | Web egress unless local | Hard | E-R1, E-R2 | After E-R1/E-R2, bounded domain |
| **D. Source-selection policy** (per query class; no generic fallback) | 4.4, 4.5 | Med–High; prevents wrong-source answers. Reasoning-backed | Policy gaps; maintenance | Low–Med | Source registry | ≈0 | Controls egress | Med | C | Design now, implement with C |
| **E. Calibration and selective decisions** | All confidences are constants or self-reports | High for safe operating points. [LIT] selective prediction; conformal factuality (Mohri & Hashimoto, 2024) | Drift; small calibration sets | Med | Calibration split | ≈0 | — | Med | Labelled data (E-R1) | Built into E-R1 analysis |
| **F. Correction and regeneration** | Ill-founded `fixed_output` | Uncertain; can add errors. [LIT] RARR shows attribution-preserving revision needs re-verification | New unsupported claims; lost qualifiers | High | Editor plus re-verifier | +1–2 calls | Same as the judge | Hard | B, C, E | **No**; last |
| **G1. Structured-fact verification** (Wikidata property values, number, unit and date normalisation) | 4.3 | High precision on a narrow slice. Reasoning-backed | Low coverage; entity linking | Low–Med | None new (Wikidata already used) | Fast lookups | Wikimedia egress (already present) | Easy–Med | Entity linking | **Second-best first step** |
| **G2. Remove dead work** (factual-step GT calls; always-on Wikipedia jury fetch) | 4.7 | Latency: p50 3.0 s / p95 12.7 s in `reasoning_verify` [REPORT] | None if dead | Low | — | Saves seconds and calls | Reduces egress | Easy | — | Housekeeping with P |
| **G3. Quote-grounded judging** (require a verbatim evidence span, check by substring) | 4.3, judge hallucination | Med–High, cheap. [HYP] H3 | Paraphrase-only support; quote gaming | Low | — | +~20 output tokens | — | Easy within E-R1 | — | Inside E-R1 |

### 7.1 Ranking

| Rank | Candidate | Rationale |
|---|---|---|
| 1 | **P + E-R1** | Highest expected value per unit of risk. P removes harm that is measurable today; E-R1 measures the judge everything else needs. Lowest cost and highest measurement confidence |
| 2 | **G1 structured-fact verification** | Precise, cheap and auditable. The fallback if E-R1 fails, and complementary if it passes |
| 3 | **B + C** (claim-level post-hoc verification over a bounded, source-policied corpus) | Highest ceiling, but gated on E-R1 (judge) and E-R2 (source recall) |
| 4 | **D** | Designed together with C |
| 5 | **E** | Calibration is woven into every experiment rather than run as its own step |
| 6 | **A** | Ensemble changes only as a measured routing optimisation |
| 7 | **F** | Correction last, and only with re-verification |

### 7.2 Why P + E-R1 first, ahead of the alternatives

- **P versus the prior tri-state-only plan.** Tri-state parsing covers about one-third of the confirmed harm. The larger harms are disagreement gating, consensus degeneracy, Wikipedia overriding escalation, and corrections hitting correct answers. All of them need an *action* policy, not just a parser.
- **E-R1 versus building RAG, claim decomposition or routing first.** Each of those assumes a judge with known SUPPORTED and CONTRADICTED precision. If an 8B judge given the *right* passage cannot exceed, say, 0.95 precision, then retrieval only adds noise to an unreliable decision. E-R1 isolates that question cheaply: no retrieval, no Groq, no production change.
- **E-R1 versus a log-only replay.** The existing logs lack replacement texts and evidence, so they can show harm (done in §1), but they cannot validate a fix.

---

## 8. Recommended first experiment (summary)

The full preregistration is in the [companion plan](2026-10-27_E-R1_verifier_benchmark_plan.md).

- **Part A — contract fixtures (zero cost).** Authored cases for every outcome in §3.3 #7–8 (empty, malformed, UNCERTAIN, timeout, zero tokens, long-snippet heuristic, single shadow, two equal shadows, confirmation followed by fix, escalation followed by Wikipedia). They are run against pure-function replays of the current parsers and policies (B0) and the proposed contract (B1/B2). Expected result: B0 fails many; B1 must fail none.
- **Part B — gold-evidence judge benchmark.** Run on a public evidence-labelled dataset (needs approval), with arms V0–V5 on local Ollama models. V0/V1 share generations; V4 (NLI) is optional and needs a model approval.
- **Controls.** V0 (current semantics replayed locally), V2 (current lexical fallbacks, zero-cost), "always NEI" (zero coverage, zero risk) and "always SUPPORTED" (current fail-open extreme).
- **Primary metric.** SUPPORTED precision with a Wilson 95% lower bound, at coverage of truly-supported claims, on a sealed test split, with thresholds chosen only on the calibration split.

---

## 9. Evaluation framework (all stages)

### 9.1 Outcome taxonomy

Separate, and never collapsed into one score:

- **Per claim:** `SUPPORTED` · `CONTRADICTED` · `INSUFFICIENT_EVIDENCE` · `CONFLICTING_EVIDENCE` · `VERIFIER_UNAVAILABLE` · `NOT_CHECKABLE` (opinion, identity, no factual content).
- **Per answer (action):** `PASS_VERIFIED` (all critical claims supported) · `PASS_QUALIFIED` (unverified claims disclosed) · `FLAG` (a claim contradicted, original kept) · `ABSTAIN/ESCALATE` · `CORRECTED` (later stage only, re-verified).
- **Separate channels:** `guard_blocked` (Workstream A) and `ensemble_available` (shadow count). A guard-blocked item is never counted as reliable or unreliable.

### 9.2 Case coverage

Every case type the brief lists maps to a source:

| Case | Source |
|---|---|
| Correct + supported; plausible-wrong; unsupported; contradicted | E-R1 dataset (SUPPORTS / REFUTES / NEI) |
| Insufficient evidence | E-R1 NEI class, plus E-R2 retrieval misses |
| Conflicting or outdated sources | Contrastive-evidence data (e.g. VitaminC revisions) for E-R1; dated source snapshots for E-R2 |
| Shadows agree but wrong; disagree but primary right | TruthfulQA frozen items (existing logs) once claim labels exist (E-R3) |
| Correct flagged (FP); incorrect approved (FN) | All stages: confusion matrix over §9.1 |
| Retrieval failures; verifier failures and timeouts | Contract fixtures (Part A) plus fault injection in E-R2/E-R3 |
| Correction introduces a new error | Stage F only: paired claim-level labels before and after correction |

### 9.3 Metrics

**Essential now (E-R1):**

- SUPPORTED precision (Wilson LB) and coverage.
- CONTRADICTED precision and recall.
- False-SUPPORTED rate on REFUTES and NEI items.
- NEI handling rate.
- Risk–coverage curve, with AURC.
- Contract-matrix failures.
- Per-item latency on local hardware.

**Later:**

- Claim-extraction recall and precision against human atomic claims (E-R3).
- Evidence recall@k and citation precision (E-R2).
- ECE and Brier on calibrated scores.
- Answer-level FPR and FNR at operating points.
- p50/p95 end-to-end latency; calls, tokens and egress per request.
- Correction net effect.
- Robustness to injected evidence conflicts, stale passages and outages.

### 9.4 Data provenance and protocol

- Pin dataset revision and licence, hash rows with the existing `evals/canonical.py` serializer, and record a manifest.
- Split by source document or entity (not row) into calibration (30%) and sealed test (70%). Seal the test hashes before any run.
- Thresholds come only from calibration. Each arm is evaluated on test once.
- Report paired comparisons (McNemar, or a clustered bootstrap by source document).
- Keep all raw text local and gitignored, per the confidential-material policy. Commit only manifests, hashes and aggregates.

---

## 10. Proposed architecture and staged roadmap

### 10.1 Target design (after evidence supports it)

```
primary_output ─► claim decomposition ─► per-claim router ──► source policy (by claim type)
                                              │                   ├ structured KB  (entity/property/value)
 shadow signals ──► routing priority only ────┘                   ├ curated corpus (encyclopedic/docs)
 (never support)                                                  └ dated search   (temporal; allow-list)
                                                                        │ evidence: passage + URL + revision/as-of + rank
                                        verifier: deterministic ▸ NLI ▸ quote-grounded LLM judge
                                                                        │ per-claim state (§9.1)
                                        aggregation by claim criticality ─► action (§9.1) + explanation
                                        (correction only: evidence-extracted value, re-verified)
```

### 10.2 Stages

| Stage | Content | Exit criterion |
|---|---|---|
| **S0 Containment** (owner decision; no research) | Confirm D-001 applies: monitor warn-only, no `fixed_output` substitution, no escalation override; sequence the regex triage with this | Owner decision recorded |
| **S1 E-R1** | Contract plus gold-evidence judge benchmark | §8 criteria in the companion plan |
| **S2 E-R2 Source recall** | For E-R1 claims, does Wikidata (values), a Wikipedia full article (not summary), or dated search contain the gold evidence? | Choose the source policy per claim type from measured recall |
| **S3 E-R3 Claim-level, end to end** | Decomposition plus verification on FIE-shaped QA, including agree-but-wrong items; gated vs verify-all; warn-only vs current correction | Reduced false approvals; correct-answer flags within the agreed budget |
| **S4 Selective routing** | Use shadow disagreement, claim criticality and cost to choose which claims to verify; calibrate | Same risk at lower cost |
| **S5 Constrained correction** | Evidence-extracted replacements only; re-verify; RARR-style minimal edits | Net claim-level gain with paired CI > 0 |

---

## 11. Compute, latency, cost and privacy

| Item | Current (measured [REPORT], 2026-07-14, n=400) | E-R1 | Target design [HYP] |
|---|---|---|---|
| End-to-end latency | p50 10.5 s, p95 27.6 s | Offline; not on the request path | Verification only on routed claims; NLI ≈ tens–hundreds of ms per pair on CPU; LLM judge seconds per claim on 4 GB VRAM, so it must be selective |
| Main contributors | Jury p50 7.2 s; reasoning_verify p50 3.0 s / p95 12.7 s; gt_verify p50 2.2 s / p95 6.6 s; shadows p50 2.5 s | — | Removing dead factual-step calls (§4.7) is a likely saving |
| Calls per request | 6.45 Groq, 2.7 HTTP, 2,325 tokens | 0 external; local Ollama ≈ 2–8 s per item (estimate) → roughly 3–10 GPU-hours for all arms | Fewer Groq calls if correction and dead steps go |
| Money | Groq + Serper usage | **0** | Depends on the source policy |
| Privacy | Prompt and answer to Groq (shadows, extractor, judge, RAG); prompt to Serper; queries to Wikimedia; raw text stored in MongoDB [REPORT audit §5.10] | Fully local; public data only | Local-first judges; per-source egress policy; tenant corpora never leave the tenant |

---

## 12. Risks and rollback

- **Coverage loss.** Strict contracts raise "unverified" rates. Mitigation: report risk–coverage; qualified answers instead of silent passes; the owner sets the budget.
- **Benchmark transfer.** Gold-evidence results (FEVER-style) overstate real performance, where evidence must be retrieved. E-R2 and E-R3 exist for that reason, and E-R1 is a necessary condition, not a sufficient one.
- **Cross-workstream sequencing.** The regex fix changes jury routing. If shipped before S0 containment, it activates §4.2 and §4.4. Rollback: revert as a single unit; keep the contract behind a flag; shadow-log old and new decisions.
- **Local-versus-Groq judge mismatch.** Ollama `llama3.1:8b` is quantised, while Groq serves its own build. Mitigation: label V0 as a "semantics replay" and do not claim exact production fidelity.
- **Licence and redistribution.** Keep raw data local and commit only manifests.
- **Over-reliance on any single judge.** Report per-family results; never treat unanimity as truth.

---

## 13. Decisions requiring your approval

| # | Decision | Options / default proposed |
|---|---|---|
| OD-R1 | Accept these findings and record an entry in `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Accept / amend |
| OD-R2 | Containment: monitor warn-only (no `fixed_output` substitution; no Wikipedia override of escalation) until S3 passes | Proposed: **yes**, consistent with D-001 |
| OD-R3 | Triage of [patterns.py:94](../../engine/agents/adversarial/patterns.py#L94) by the Workstream-A process, sequenced *with* OD-R2; annotate July baselines as "measured under jury regex defect" | Proposed: yes; no edit by this workstream |
| OD-R4 | Approve one public evidence-labelled dataset download for E-R1 (licence to be verified before use) | FEVER (3-way, Wikipedia evidence) **or** VitaminC (contrastive revisions) **or** HaluEval-QA with `knowledge` (a clean re-download also fixes the documented corruption) |
| OD-R5 | Optional: approve one local NLI or fact-check model download for arm V4 | Yes / skip V4 |
| OD-R6 | Approve local Ollama compute (≈3–10 GPU-hours) with existing models: llama3.1:8b, qwen3:8b, granite3.3:8b, mistral:7b | Yes / limit arms |
| OD-R7 | Approve E-R1 thresholds: SUPPORTED precision LB ≥ 0.95 at coverage ≥ 0.60; CONTRADICTED precision LB ≥ 0.90 for any override | Accept / adjust |
| OD-R8 | Data location and policy: `research/reliability/data/` gitignored for raw text; manifests and aggregates committed | Accept / alternative path |
| OD-R9 | Whether the prior 2026-10-20 plan is superseded by E-R1 Part A or kept as a separate deliverable | Proposed: folded in |

---

## 14. Three most valuable improvements

| Rank | Improvement | What it does | Evidence |
|---|---|---|---|
| **1 (pursue first)** | **Evidence-gated decision and action contract, validated by E-R1** | Explicit per-claim states; failures are never support; shadows route but never validate; no approval or replacement without a measured-precision, source-linked verdict; warn-only until measured | §4.1–4.5 [CODE]; corrections at or below chance (31 correct vs 26 wrong replaced; Wikidata overrides 16 vs 14; `gt_contradiction` precision 0.393) [DATA/REPORT]; 57.9% of wrong answers never verified [DATA]; zero API cost to test |
| **2** | **Structured-fact verification** | Wikidata property values with qualifiers and dates; deterministic number, unit and date checks; replaces "8B reads a description" | §4.3 [CODE]: current Wikidata path is parametric 8B output labelled `FULLY_PROVENANCED` |
| **3** | **Claim-level, post-hoc evidence verification over a source-policied bounded corpus** | Decompose, retrieve per claim with provenance and as-of, verify, aggregate by criticality | §3.3 #6 (single claim, truncated); §4.4 (top-1 summary); [LIT] FActScore, SAFE. Gated on E-R1 and E-R2 |

**Why the first one.** It is the only option that is cheap, reversible, measurable with existing local compute, and targeted at harms the repository's own logs already show. Every other option, RAG included, inherits its judge and its policy from it.

**Stopping here.** Nothing is implemented. Waiting for review of OD-R1 to OD-R9.

---

## Appendix A — Reproducing the [DATA] figures

Run from the repository root (read-only, no network):

```bash
# Routing, correction and root-cause cross-tabs for the signal-level runs
python -I - <<'EOF'
import json; from collections import Counter
for p in ['data/hallucination_eval/baseline_raw_tqa_sig.jsonl','data/hallucination_eval/baseline_raw_halueval.jsonl']:
    d={}
    for l in open(p,encoding='utf-8'):
        if l.strip(): r=json.loads(l); d[r['id']]=r
    live=[r for r in d.values() if not r.get('guard_blocked')]
    print(p, len(live),
          Counter(r['signals']['root_cause'] for r in live),
          Counter((r['signals']['gt_corrected'], r['label']) for r in live),
          Counter((r['signals']['fix_strategy'][:11], r['label']) for r in live),
          Counter((r['n_shadow_ok'], 'gt_verify' in r['metrics']['node_ms']) for r in live))
EOF
# Verification reach by label in the frozen full-shadow run
python -I -c "import json;from collections import Counter;d={json.loads(l)['id']:json.loads(l) for l in open('data/hallucination_eval/baseline_raw.jsonl',encoding='utf-8') if l.strip()};print(Counter((r['label'],'gt_verify' in r['metrics']['node_ms']) for r in d.values() if not r.get('guard_blocked')))"
# Regex defect
python -c "from engine.agents.adversarial import patterns as P;[print(ap.root_cause, bool(ap.pattern.search('What is the capital of France?'))) for v in vars(P).values() if isinstance(v,list) for ap in v if getattr(ap,'root_cause','')=='TOKEN_SMUGGLING']"
git show ccee211 | grep -n 'x00|'
```

Counts are per unique `id` (last record kept). `label 1` = hallucinated or wrong candidate.

## Appendix B — References [LIT]

These are cited from memory and were not re-fetched in this session; verify before external publication.

- Lin, Hilton & Evans (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. arXiv:2109.07958.
- Manakul, Liusie & Gales (2023). SelfCheckGPT. arXiv:2303.08896.
- Min et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision. arXiv:2305.14251.
- Wei et al. (2024). Long-form factuality in large language models (SAFE / LongFact). arXiv:2403.18802.
- Thorne et al. (2018). FEVER: a large-scale dataset for Fact Extraction and VERification. arXiv:1803.05355.
- Schuster, Fisch & Barzilay (2021). Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence. arXiv:2103.08541.
- Laban et al. (2022). SummaC: Re-Visiting NLI-based Models for Inconsistency Detection. TACL.
- Zha et al. (2023). AlignScore. arXiv:2305.16739.
- Tang, Laban & Durrett (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. arXiv:2404.10774.
- Gao et al. (2023). RARR: Researching and Revising What Language Models Say. arXiv:2210.08726.
- Yan et al. (2024). Corrective Retrieval Augmented Generation (CRAG). arXiv:2401.15884.
- Mohri & Hashimoto (2024). Language Models with Conformal Factuality Guarantees. arXiv:2402.10978.
- Farquhar et al. (2024). Detecting hallucinations in large language models using semantic entropy. Nature.
- Panickssery, Bowman & Feng (2024). LLM Evaluators Recognize and Favor Their Own Generations. arXiv:2404.13076.
- Kim, Garg, Peng & Garg (2025). Correlated Errors in Large Language Models. (ICML 2025; check identifier.)
- Geifman & El-Yaniv (2017). Selective Classification for Deep Neural Networks. NeurIPS.
