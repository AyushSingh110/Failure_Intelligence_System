# FIE Output Reliability: Independent Audit and Research Recommendation

**Date:** 2026-10-20  
**Scope:** Output reliability only: answer assessment, external evidence, correction and abstention. The adversarial-input C2 program and its annotation artifacts are out of scope and unchanged.  
**Status:** Read-only audit and proposed experiments. No production code, configuration, dataset, model, threshold, or prior report was changed. No API was called and no new data or dependency was obtained.

## Executive summary

FIE has a substantial, real output-monitoring implementation: three Groq shadow generations; ensemble and embedding-derived signals; a failure classifier and diagnostic jury; reasoning checks; a ground-truth pipeline; Wikidata, Serper and Wikipedia retrieval; cache and provenance labels; correction and human-escalation branches. Its functionality is broader than a simple disagreement detector. However, the main correctness gap is at the **decision boundary between evidence and approval**: several paths turn *unknown, malformed, weakly relevant, or merely similar text* into an affirmative “consistent” result, while some answers are promoted to “fully provenanced” without machine-checkable claim-to-passage support. Separately, a single extracted claim and exact-text shadow voting are too coarse to justify whole-answer correction.

The strongest existing outcome evidence argues against treating ensemble agreement as reliability. The frozen, decontaminated July 2026 TruthfulQA run has AUC 0.497 (95% CI 0.424–0.562), and ECE 0.267; its report says 120/400 pairs were blocked by the adversarial guard and excluded. A separate report records 29.5% blocked on benign TruthfulQA questions. These are findings from the repository’s reports, not newly reproduced results. The optimization log also cautions that its feature-level HaluEval result does not equal a clean end-to-end result. Its headline “HaluEval 0.38” is described as an end-to-end result, but the artifact and exact protocol should be reconciled before it is combined with the TruthfulQA number.

**Recommendation:** the first reliability experiment should be a frozen, labeled, local replay of the verifier decision contract, comparing current behavior with a *tri-state* policy (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT/UNAVAILABLE`) that never equates failure-to-refute with support. It should focus on Serper/Wikipedia/Wikidata verifier responses and the exact correction/abstention action that follows. This is a smaller, more testable first step than introducing a RAG stack, and it directly probes the most serious confirmed design weakness without spending on external services. In the next implementation phase—only after review—preserve uncertain/failed as “unverified,” avoid auto-correction from consensus alone, and use source/evidence-specific claim verification where supported. Claim-level post-answer evidence checking is the second-best research direction. A vector database is not presently justified.

The evidence supports three priority improvements:

1. **Make verification tri-state and fail safely:** unknown, malformed, empty, timeout, and low-confidence verifier responses must not approve an answer. This is the first experiment and smallest likely decision-quality improvement.
2. **Verify atomic claims against cited passages and source metadata:** report support, contradiction, unresolved, or conflicting per claim; aggregate by claim importance instead of a single answer-level judgment.
3. **Route selectively and calibrate abstention:** use the shadow ensemble as an uncertainty/disagreement signal; use external evidence for factual claims, and measure risk versus coverage before correction or approval.

## 1. Audit method and evidence boundaries

I inspected the live modules under `engine/`, `app/routes/`, relevant tests, reliability reports, `.env.example`, and existing evaluation outputs. I did not run the test suite or evaluation harness: the task is read-only, and tests could trigger network/model behavior. No credentials were read. Repository state was clean at inspection (apart from Git’s environment warning that it could not read a user-level ignore file); no tracked or untracked project changes were present.

“Confirmed” below means directly visible in current source or an existing local artifact. Existing reports are historical claims from the repository and were not independently reproduced. “Research evidence” refers to the linked papers, whose results may not transfer to FIE’s models, domains, or deployment.

## 2. Verified map of the current implementation

### Request to verdict

The primary server path is in [`engine/pipeline/langgraph_pipeline.py`](../../engine/pipeline/langgraph_pipeline.py): `load_session` → `shadow_inference` → `adversarial_guard` → signal extraction/provenance/reasoning/jury/security checks → conditional ground-truth verification → escalation or auto-correction → finalization. The initial shadow calls currently precede the adversarial guard. `_route_to_verify` requires a failure signal, high risk, and (when full jury is on) jury confidence at least 0.45. A failed `gt_verify` node returns `gt_result=None`; `_route_after_gt` then routes to `finalize`, which avoids treating a thrown pipeline error as verification success.

The separate direct `/monitor` implementation in [`app/routes/monitor.py`](../../app/routes/monitor.py) also fans out shadows, builds ensemble signals, and conditionally runs correction / ground-truth logic. Its decision gates are not identical to the LangGraph path. Treat both call paths as production surfaces in any follow-up; do not assume a fix to one automatically fixes the other.

### Shadow models and aggregation

[`engine/groq_service.py`](../../engine/groq_service.py) fans out concurrently to three configured model identifiers: Llama 3.3 70B, GPT-OSS 120B and Qwen3 32B. Failed calls are returned as unsuccessful; successful outputs are kept. Retries are for HTTP 429 only, up to two retries with backoff; a timeout or other error becomes a failed response. The requests carry the prompt and canary system message. The optional confidence suffix asks each model to label its own confidence HIGH/MEDIUM/LOW and maps that label to weights 3/2/1. These are self-reports, not calibrated correctness probabilities. The response cache is scoped to tenant scope where supplied.

The actual “diversity” is a chosen set of different model families/providers, but the repository contains no measured error-correlation study for this set. Shared pretraining, common web sources, similar instruction tuning, and the same question/prompt leave correlated error plausible; three outputs are not three independent evidence sources.

`engine/failure_signal.py` and the signal builder derive answer frequency, disagreement/entropy, semantic similarity and related features. `engine/failure_classifier.py` predicts failure risk from a small feature vector; it is a detector/decision signal, not direct evidence of truth. The classifier’s probability is not a source citation. The exact-text correction voter in [`engine/fix_engine.py`](../../engine/fix_engine.py) groups lowercased outputs after trimming punctuation and truncating to 200 characters, then weights votes by the model’s self-rating. The confidence-weighted winner may be used as the replacement when its share is at least 0.50. This grouping is not semantic equivalence and does not test truth. The separate ground-truth pipeline’s consensus calculation uses normalized output text and weights; it labels a consensus-only path “PARTIALLY_PROVENANCED,” appropriately admitting that it lacks external support.

For REASONING/CODE, [`engine/verifier/self_consistency_verifier.py`](../../engine/verifier/self_consistency_verifier.py) computes mean pairwise SBERT cosine similarity, selects the medoid, and accepts it as pseudo-ground-truth when the configured threshold passes. If the encoder is unavailable, `SequenceMatcher` overlap is used with a 0.90 adjustment. High semantic similarity indicates agreement; it cannot establish correctness. A single non-empty output is marked consistent with score 1.0 by this helper, though the ground-truth pipeline requires at least two outputs before calling it.

### Routing, claims, and external sources

[`engine/question_classifier.py`](../../engine/question_classifier.py) maps question type to Wikidata/Serper/RAG gates. Factual/unknown questions can use Wikidata, Serper and Wikipedia RAG; temporal questions use Serper; reasoning/code use self-consistency; opinion/identity skip external grounding. Temporal search in the ground-truth pipeline is additionally selected from `root_cause == TEMPORAL_KNOWLEDGE_CUTOFF` except for a small regular-expression set of permanent-fact questions. A question classifier and a root-cause classifier therefore both affect whether external evidence is used.

[`engine/claim_extractor.py`](../../engine/claim_extractor.py) asks Groq to extract the **main** factual claim and returns one structured claim. This is not full answer decomposition. Multi-sentence answers, qualifiers, temporal boundaries, numerical conditions, and dependent claims can remain unchecked. Claim extraction itself is model-generated and may omit or distort the target proposition.

[`engine/verifier/wikidata_verifier.py`](../../engine/verifier/wikidata_verifier.py) searches Wikidata entity labels/descriptions, chooses the first acceptable candidate, and asks Groq to compare the claimed property/value with the entity description. It does not query a structured Wikidata property value; it relies on a short description and a Groq judgment. Candidate selection contains a special-case creative-work filter, but after trying filtered candidates it falls back to the first candidate. The source code contains no multi-source contradiction resolution or freshness policy for the response. Wiki verifier exceptions and missing entities return `found=False` and the main pipeline can continue to shadow consensus.

[`engine/verifier/serper_verifier.py`](../../engine/verifier/serper_verifier.py) sends the whole question to Serper, asks for five results, then extracts only the first non-empty answer box/knowledge-graph description/organic snippet. It does not preserve a list of passages, inspect multiple independent sources, rank by authority, store page timestamps, or establish source-to-claim links. A Groq verifier compares the answer with that single string; if Groq is unavailable, it uses word overlap. A live Google search result is therefore a discovery signal, not a demonstrated authoritative source.

The separate Wikipedia retrieval path in [`engine/rag/retriever.py`](../../engine/rag/retriever.py) strips a few leading question patterns and tries a direct English Wikipedia summary; on a miss it searches Wikipedia and retrieves the top result’s summary. It returns extracted text without a recorded page revision, canonical page identity, retrieval score, or disambiguation check. [`engine/rag/rag_pipeline.py`](../../engine/rag/rag_pipeline.py) instructs Groq to answer using that context. [`engine/rag_grounder.py`](../../engine/rag_grounder.py) then labels a generated grounded answer with a fixed confidence of 0.92 and the summary fallback 0.72. Those are constants, not empirically calibrated probabilities. A later comparison helper uses Groq or token-overlap heuristics; it is not called on the `auto_correct` Wikipedia replacement branch in the LangGraph path.

### Trigger and outcome behavior

The central ground-truth pipeline is [`engine/verifier/ground_truth_pipeline.py`](../../engine/verifier/ground_truth_pipeline.py). It checks a tenant-scoped answer cache where a scope exists, extracts one claim for non-temporal cases, routes temporal cases to Serper, factual claims to Wikidata, and then uses shadow consensus if external verification does not resolve the item.

Confirmed positive behavior:

- A temporal question with Serper disabled, no results, or an exception is marked `requires_escalation=True`.
- Wikidata overrides require a mismatch confidence of at least 0.75 and a minimally meaningful replacement; confirmation requires at least 0.60.
- Weak shadow consensus in the GT pipeline escalates instead of inventing a verified answer.
- The LangGraph `escalate` path retains the primary answer and marks human review rather than silently replacing it.
- The Wikipedia correction stage excludes prompts matching `prompt_requires_live_data` and only runs for specified factual root causes.

Confirmed shortcomings / defects in the current decision semantics:

1. **Serper’s unknown verdict becomes “consistent.”** In `_parse_groq_verdict`, `matches` starts `True`; it is set false only when the parsed label contains `INCONSISTENT`. Thus `UNCERTAIN`, an absent verdict, or arbitrary malformed text is represented as `matches_output=True`. In `run_ground_truth_pipeline`, any `serper.found` result with `matches_output=True` is reported as confirming the primary output and tagged `FULLY_PROVENANCED`, without checking confidence against a threshold. This can treat one search snippet plus an uncertain/invalid judge response as approval. The fallback word-overlap helper also returns `matches=True` when there are no usable tokens; a low overlap is ambiguous, not evidence of consistency.
2. **The Wikipedia branch bypasses verification.** `auto_correct` can return a Groq-generated answer from a single Wikipedia summary, or the summary itself, with a fixed 0.92/0.72 confidence and `fix_applied=True`; it does not require claim-level support, source credibility checks, or re-verification before the answer is returned as corrected. This happens after shadow consensus does not produce a fix. The test `tests/test_monitor_rag_fix.py` asserts that the Wikipedia+Groq answer is applied, but it tests the wiring with a mocked source and does not measure factuality or citation support.
3. **Verifier parse defaults are permissive.** The Groq parsers in Serper, Wikidata and `rag_grounder` default toward `matches=True`; in some parsers an unknown label is therefore indistinguishable from consistency. Confidence may be zero or defaulted, but the temporal Serper branch does not gate acceptance on confidence.
4. **The evidence object is lost.** `SerperResult` retains only one extracted string and a generic source label; `WikidataResult` retains one entity description, and the RAG result retains neither URL nor revision metadata. The final user-facing answer cannot be audited to the passage that supports each specific claim. There is no evidence-level contradiction state.
5. **Answer-level correction has no universal recheck.** Ground-truth overrides, shadow-voted replacements, and the Wikipedia branch are not consistently re-verified as a final answer. The source label can be more definite than the actual evidence.
6. **Shadow consensus can replace unsupported answers.** The fix engine can apply a winning shadow answer even when it disagrees with the primary output; model agreement is used as correction evidence. This is specifically vulnerable to shared misconceptions, as the repository’s own TruthfulQA result illustrates.

These are source-level confirmed risks; the rate at which they occur in real traffic is unknown. No run was made to quantify them.

### Existing tests and reports

`tests/test_monitor_rag_fix.py` covers numeric unit comparisons and the mocked Wikipedia correction route. `tests/security/test_answer_cache_isolation.py` checks tenant cache isolation. These are useful behavior/security contracts, not a labeled accuracy suite for support/contradiction/abstention. I found no dedicated unit/evaluation suite that exhaustively tests malformed/uncertain Serper and Wikidata responses, source contradiction, stale timestamps, no-result semantics, or all correction stages’ post-check behavior.

Important existing evidence:

- [`data/hallucination_eval/baseline_report.json`](../../data/hallucination_eval/baseline_report.json), run dated 2026-07-14: n=400 pairs, 2.93 mean successful shadows/request, 93.25% full-shadow; reported p50/p95 10.5s/27.6s; 6.45 Groq API calls/request; 2.7 HTTP calls/request; 2,325 total tokens/request. AUC 0.4967 (95% CI 0.4236–0.5619), ECE 0.2668 (95% CI 0.2175–0.332). It scores only 280 pairs after excluding 120 guard-blocked pairs.
- [`data/hallucination_eval/guard_overrefusal_report.json`](../../data/hallucination_eval/guard_overrefusal_report.json): 29.5% of 400 TruthfulQA answer pairs / 200 questions were blocked as `JAILBREAK_ATTEMPT` by the PAIR classifier. This is a Workstream A interaction; it is an evaluation-sampling confound and should be reported as such, not retuned as part of this output-reliability work.
- [`docs/HALLUCINATION_OPTIMIZATION_LOG.md`](../../docs/HALLUCINATION_OPTIMIZATION_LOG.md): documents decontamination and the frozen baseline, plus a historical HaluEval end-to-end claim of AUC 0.38 over ten repeated runs. The same log says the offline question-disjoint HaluEval result on an older feature snapshot was 0.956 and identifies the difference as unresolved. Keep those protocols separate; the log’s reported figure is not proof that evidence retrieval currently works.
- [`docs/HALLUCINATION_PIPELINE.md`](../../docs/HALLUCINATION_PIPELINE.md) and [`docs/HALLUCINATION_DETECTION_EXPLAINED.md`](../../docs/HALLUCINATION_DETECTION_EXPLAINED.md) describe broader design, but source code is authoritative where they diverge.

## 3. Evaluation of the working hypotheses

### Shadow ensembles

The current approach does reduce dependence on one sampled response and uses different model families. It has some useful failure behavior: failed shadows are omitted, and the pipeline records when they are unavailable. But it does not establish independent error distributions, and self-reported confidence weights are not calibrated. Similarity and majority answer do not tell the system which answer is true. For common false beliefs, model agreement can increase confidence in the wrong answer. Conversely, correct answers can be phrased differently and score as disagreement.

The current frozen result (AUC approximately 0.50 on decontaminated TruthfulQA) is direct evidence that the present fused detector does not discriminate this important shared-misconception regime. It does not prove ensembles never help: they may be useful for identifying idiosyncratic generations on other domains. Existing source has no per-model or pairwise error-correlation study. Do not add shadows until an ablation establishes marginal benefit versus one-model resampling, selective 1/2/3-model routing, and evidence verification, under the same frozen items.

### Wikipedia and Serper fallback

The premise that retrieval guarantees truth is false, and the present implementation illustrates why. Retrieval gives a potentially related snippet/article; neither the ranking nor source is checked as authoritative for the claim. The code uses one top result or one Wikipedia summary and does not retain citation-ready provenance. A verifier LLM is asked whether the texts are “consistent,” which is weaker than checking that each answer proposition is entailed by the evidence. Unknown/unparsable often collapses to “consistent.”

Wikipedia is reasonable as a discovery source for stable general facts and can be helpful when the matching page is unambiguous. It is not a universal authority, may be incomplete or stale, and its summary is not a universal ground-truth oracle. Serper can surface useful primary sources, but the current code extracts a single snippet, which can omit qualification/context or represent search ranking rather than authority. It has no explicit source policy, freshness threshold, canonical URL, multi-source corroboration or contradiction comparison. If provider configuration is missing, temporal Serper correctly escalates; but factual Wikidata failures can fall through to consensus, and non-temporal correction can fall through to Wikipedia.

### RAG: where it helps and where it does not

RAG addresses a real gap when the answer requires evidence absent from the model’s parametric knowledge, especially live, domain-specific, tenant-provided or auditable facts. But FIE already has rudimentary retrieve-then-answer behavior (Wikipedia summary → Groq answer) in the correction path and search-plus-judge behavior (Serper/Wikidata) in the GT path. The gap is therefore not “no RAG”; it is weak retrieval/source selection, no evidence provenance/claim alignment, permissive unknown handling, and no systematic output verification after correction.

For this system, the initial design should be **post-generation claim verification**. It directly tests whether current statements are supported or contradicted without changing the answer-generation distribution. Retrieve **before generation** only for question classes where external knowledge is necessary (live status, selected domain corpora, user-approved local material); then still verify generated claims against those retrieved passages. “RAG” should describe a staged evidence pipeline, not imply a vector index or correctness guarantee.

Start with a small, transparent source adapter and lexical/entity lookup if the corpus is narrow. For a trusted, versioned local corpus, BM25/full-text is an appropriate first baseline; add hybrid retrieval/reranking only if recall measurements show misses. A vector DB or knowledge graph is justified only if the evidence type and measured retrieval needs require it. Structured databases can be better for finite facts; deterministic checks are better for arithmetic, schema and dates. Lightweight NLI/cross-encoder evidence matching can be a cheaper verifier baseline, but NLI’s entailment label is not source authority and domain transfer needs testing. A verifier LLM may handle nuanced relations but remains an fallible, potentially correlated judge and adds latency/API exposure.

Research supports these boundaries, not automatic transfer: SelfCheckGPT evaluates sampled-consistency and found NLI a promising cost/quality tradeoff on WikiBio, while explicitly noting sentence-level factuality and dataset scope limits ([SelfCheckGPT](https://arxiv.org/abs/2303.08896)). Semantic entropy research proposes uncertainty signals for confabulations but is not a truth oracle ([Nature, 2024](https://www.nature.com/articles/s41586-024-07421-0)). FactSelfCheck argues for fact-level granularity, reports competitive detection and a correction experiment, while noting its WikiBio data limitations and extra model-based computation ([FactSelfCheck](https://arxiv.org/abs/2503.17229)). Provenance presents a compact NLI-based factuality checker for outputs grounded in retrieved context, tracing issues to context chunks ([Provenance](https://arxiv.org/abs/2411.01022)). These motivate baselines; they do not establish FIE-specific gains.

## 4. Candidate comparison

Estimates are relative engineering judgments. Latency/cost are incremental to the current call path and must be measured; external services are not priced here.

| Candidate | Failure addressed / expected benefit | Main failure modes | Effort, dependencies and runtime | Privacy / evaluation | Initial suitability |
|---|---|---|---|---|---|
| A. Tri-state verdict + fail-safe policy | Unknown, malformed, low-confidence, missing source must not become approved; high expected precision of decision semantics; strongest source-code justification | Can raise abstention/false alarms; tri-state alone cannot make retrieval relevant or judge accurate | Low effort; no new model/data; negligible runtime | Preserve data local; replay labeled verifier cases and measure approval errors + abstention | **First** |
| B. Claim decomposition + claim-level evidence links | Missed errors inside long answers; identifies which statement is unsupported/contradicted | Extractor omission, granularity/dependency mistakes, false NLI entailment; increases annotation work | Medium; one decomposition step plus claim-level verifier; moderate call/token latency if LLM used | Human labels required; evaluate claim boundary and support metrics | **Second** |
| C. Curated/local evidence retrieval, initially post-answer | Unsupported answers where suitable external truth exists; may catch shared misconceptions | Missing or biased corpus, stale/irrelevant passages, false authority; no guarantee of answer faithfulness | Medium for source adapters, high for ingestion/index; local search low marginal inference, NLI/LLM adds time | Strong privacy if local; license/provenance and curation obligations; retrieval recall and source adjudication are hard | Pilot after A, on a bounded domain |
| D. General web/Serper source policy | Temporal facts and broad coverage | Ranking/snippet incompleteness, SEO/manipulation, stale pages, source conflict; API dependence | Low-to-medium; extra external request and judge latency/cost | Sends prompt to provider; must record URLs/timestamps and evaluate source classes | Later, as source-discovery adapter, never universal fallback |
| E. More/better shadow ensemble or calibrated selective routing | Disagreement-specific errors and cost management | Shared error, self-confidence miscalibration, language/style disagreement | Model routing can reduce average cost; extra models increase tail latency/API calls; routing requires calibration data | Prompts sent to model provider(s); needs model-wise/domain-wise held-out calibration | Measure by ablation, not default next change |
| F. Lightweight NLI/cross-encoder or deterministic verifier | Evidence relation checks; deterministic domains | Entailment model domain mismatch, negation/numbers/temporal scope; NLI does not assess source authority | Medium model integration; local inference may add modest latency and deployment footprint | Local model preserves prompt privacy; benchmark against human claim labels | Strong verifier baseline alongside A/B |
| G. Correction + mandatory re-verification | Prevent unverified “fixes” from becoming trusted output | Correction can add claims, lose qualifiers, or remain wrong; repeated generation can compound error | Medium/high; at least one correction and verification pass, added latency/cost | Same provider/privacy considerations; evaluate net factuality including introduced errors | Only after decision/evidence baseline; always compare to warning/abstention |
| H. Calibrated risk/coverage policy | Uncalibrated scores and poor approval thresholds | Calibration drift, small-sample/domain shift, threshold gaming | Medium; no runtime cost except decision logic | Requires representative holdout and confidence intervals | After valid labels and tri-state outputs exist |
| I. Deterministic checks, cache, selective invocation | Arithmetic/date/schema errors and needless model calls | Coverage narrow; cache staleness or semantic key collision | Low/medium; deterministic checks very cheap; cache freshness/invalidation needed | Local; easiest to audit | Add by specific demonstrated error category, not broad factual QA |

The primary comparison baseline must not be “current whole system vs full RAG.” That would confound retrieval, judge, correction and routing changes. Compare policy changes on identical recorded/mocked evidence objects first, then compare components incrementally.

## 5. Evaluation plan and acceptance criteria

### Dataset and labels

Build a **new append-only reliability evaluation set** under `research/reliability/` or a dedicated `data/reliability_eval/` location only after approval. Keep the frozen canonical detector sets and all C2 annotations read-only. Use licensed/public benchmark items where redistribution permits, plus carefully sourced dated evidence snapshots; record per-item source, license, retrieval time/revision, question, generated answer, atomic claim, supporting/contradicting passage and annotator adjudication. Do not download anything in this phase.

Include at least: supported correct claims; plausible but false claims; unsupported claims; contradicted claims; no evidence/insufficient-evidence cases; conflicting sources; stale/current facts; evidence with misleading snippets; query/entity ambiguity; numerical units/qualifiers; correlated-shadow agreement on a known wrong answer; shadow disagreement where the primary answer is correct; extraction omissions; retrieval empty/error/timeout; malformed/uncertain verifier output; and correction-induced new errors. Include both short and long answers, multiple claims, dependent/contextual claims, domain labels, and source provenance.

Labels: claim span and normalized proposition; evidence stance (`SUPPORTS`, `CONTRADICTS`, `MIXED`, `NOT_ENOUGH_INFO`); source relevance; source authority tier defined per domain; evidence freshness as-of question time; verifier availability; final action. Double annotate a stratified subset and adjudicate disagreements. Freeze train/calibration/test partitions by question/entity/source/time to avoid near-duplicate and source leakage. Public benchmark labels are not sufficient for all evidence-authority judgments; separately mark human-authored versus model-proposed annotations.

Potential existing starting points: the local `eval_set_frozen.jsonl` (TruthfulQA correct/incorrect answer pairs; useful for shared misconception/consensus stress, but not evidence-sufficiency labels), HaluEval files (audit their schema/known local corruption as documented), and RAGTruth for RAG-specific hallucinations if license and access are approved. Do not assume these datasets are clean or interchangeable. Evaluate detector risk and evidence-grounded verification as distinct tasks.

### Baselines and protocol

1. **B0: current code semantics** replayed against deterministic mocked source/verifier responses (no external calls).
2. **B1: tri-state semantics only** on the same replay cases; unknown/malformed/failed => unverified/escalate, never supported.
3. **B2: current shadow detector only**, frozen outputs and no new sampling; report agreement, detector risk, and cost.
4. **B3: claim + retrieved passage NLI baseline** on a bounded labeled subset; no correction.
5. **B4: verifier LLM** only if approved, against the exact same evidence; not in first local experiment.
6. **B5: correction + re-verification** only after detection is measured; compare with retaining the original plus warning/abstention.

Keep question items identical for paired comparisons; cluster bootstrap by question (and source/entity where repeated) with confidence intervals. Separate retrieval failure from verifier failure and model decision failure. Report results by question type/domain/source, not only a pooled score. Do not treat a run blocked by the adversarial guard as “reliable”; keep it in a separate flow-interaction count, preserving Workstream A labels and results.

### Essential metrics (first experiment)

- **False approval rate:** unsupported/contradicted/unknown cases incorrectly marked supported or returned as corrected.
- **Support decision precision/recall** at claim level, plus contradiction precision/recall and the unresolved rate.
- **Unsupported-answer approval FNR** and correct-answer false flag rate / FPR.
- **Selective risk vs coverage:** error rate among auto-approved claims/answers at each coverage; report abstention/escalation coverage.
- **Confusion matrix** across supported, contradicted, insufficient evidence, conflicting evidence, verifier unavailable.
- **Decision contract accuracy** for empty, exception, timeout, malformed, `UNCERTAIN`, low-confidence, low-overlap and contradictory-source cases.
- **End-to-end p50/p95 latency; model/API calls and token counts** per request; report external HTTP calls separately.
- **Correction net change:** fraction corrected to supported, unchanged wrong, and newly introduced unsupported claims.

Later add Brier score, log loss, ECE with binning details and CIs for any probabilistic output; retrieval recall@k/precision@k and source coverage; citation precision; domain calibration; robustness to source poisoning and temporal drift. Calibration is not meaningful until labels represent the target deployment and scores are stable.

### Proposed gates before integration (not measured results)

These are recommended preregistered decision criteria; tune them only with a separate calibration split, then evaluate once on a sealed test set.

- On deterministic contract tests, **0 unknown/malformed/failed verifier cases may produce `SUPPORTED` or auto-correction**.
- On the adjudicated test set, the new policy must lower unsupported/contradicted auto-approval by at least **30% relative** versus B0, with paired 95% CI excluding zero improvement; do not accept if correct-answer false flags rise by more than **5 percentage points** without explicit product approval.
- Report claim-level supported precision and contradiction recall with 95% CIs; require supported precision of at least **0.95** for any automatic “verified” label (or retain “partially supported / unverified” and human review if sample size cannot establish it). This is a product risk threshold proposal, not a result.
- Preserve at least **70% coverage** on in-scope, answerable cases; otherwise investigate source coverage and verifier calibration before rollout. Always permit a lower coverage option if safety priorities demand it.
- No source without recorded identifier/provenance may produce `FULLY_PROVENANCED`; unresolved/conflicting evidence must remain unresolved/conflicting.
- Any correction path must be rechecked; its corrected claim support precision must not be worse than the original answer on paired cases. If no independent check is available, return the original with a qualification or abstain.
- Cost/latency are judged against the frozen baseline: for the first tri-state replay, zero added inference/API cost; for later live evidence experiments, report p50/p95 and calls/tokens. Require explicit approval for new services or paid calls.

## 6. First experiment: verifier decision-contract replay

**Question:** Does a fail-safe tri-state interpretation eliminate false “verified” approvals caused by missing/ambiguous verifier judgments while retaining useful coverage?

**Setup:** collect local fixtures from existing tests and authored edge cases representing each Serper/Wikidata/RAG parse outcome. Feed the same `found`, evidence text, parser output and confidence to a pure decision-policy comparison. Label cases by the evidence state; no network, external model, retraining, model downloads, or threshold changes. Include source mismatch and one-snippet/no-snippet behavior. Extend only after approval to an evidence-labeled set.

**Variants:** B0 current behavior; B1 strict tri-state (explicit `INCONSISTENT` only => contradicted; explicit, well-formed `CONSISTENT` plus valid supporting evidence and policy threshold => supported; everything else => insufficient/unavailable); B2 same plus source/evidence presence requirement. This will reveal whether the main gain comes from parser semantics alone or from the evidence-presence contract.

**Acceptance:** zero fail-open cases in the contract matrix; no false source “fully provenanced” state without retained source/evidence identity; publish the matrix, outcome counts, diffs and unanswered questions before any behavior change. For labeled data, apply the gates in §5. If B1 creates excessive abstention, inspect which sources/answer classes have recoverable evidence and decide whether better retrieval is warranted.

**Why first:** direct code inspection confirms that unknown can become consistent and that the Serper acceptance path skips the confidence threshold. This can be evaluated without a model, data purchase, API spend, or architecture rewrite. It establishes a trustworthy baseline contract that later retrieval, NLI and LLM verification can all share.

## 7. Proposed staged architecture and roadmap

### Stage 0 — Decision-contract and evidence schema (first experiment)

Define a versioned result object containing: verdict (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, `VERIFIER_UNAVAILABLE`), atomic claim ID/text, evidence span, canonical source identifier/URL, publisher/authority policy class, timestamp/revision, retrieval method, verifier method/version, confidence/calibration version, and reason. Never infer support solely from no contradiction. Record outputs without changing runtime behavior until review.

### Stage 1 — Bounded evidence verification

For a narrow class with appropriate primary sources (for example, versioned product documentation or a trusted internal corpus), retrieve candidate passages after answer generation, connect each atomic claim to passages, and run deterministic checks plus a compact NLI model as the low-cost verifier baseline. Model authority/source selection separately from entailment. Include source links and as-of timestamps. No answer regeneration yet.

For data that changes over time, source policy should require date-aware evidence and state the “as of” time. For contested claims, preserve conflicting evidence and provide qualified answer/review rather than select one snippet. For tenant-specific data, only use tenant-authorized stores; no external web request by default.

### Stage 2 — Selective shadow routing and calibrated actions

Use ensemble disagreement, answer type, claim criticality and verifier availability to decide when to spend on additional checks. Agreement is a low-cost consistency signal, never source-backed truth. Calibrate any acceptance score on separate data and choose an operating point by false approval cost versus coverage. Maintain the separate status labels rather than one composite “reliability” score. Model failure should yield `VERIFIER_UNAVAILABLE`, not approval.

### Stage 3 — Constrained correction and re-verification

Only correct claims with evidence that supports the replacement. Preserve supported parts and qualifiers; do not rewrite the entire answer if only one claim is wrong. Require a second check against the stored passages, and ensure all newly introduced factual claims are supported. Otherwise return a warning, a qualified answer with citations, ask for missing context, or abstain. Use human review for high-impact, disputed, or source-conflict cases.

### Stage 4 — Expand retrieval only if evidence shows need

Add source-specific web discovery, search reranking, hybrid BM25+dense retrieval, local vector indexing, or a knowledge graph only when Stage 1 shows the measured failure is retrieval recall, entity resolution, relation structure or corpus scale. RAG before generation is valuable for clearly time-sensitive or domain-grounded queries; post-generation checking remains necessary. Avoid broad general-web retrieval as a default for every answer.

## 8. Risks, trade-offs, rollback, and approvals

**Compute/latency/cost:** Current historical baseline is already expensive (p50 10.5s, p95 27.6s, 6.45 Groq API calls and ~2.3k tokens/request, with 2.7 HTTP calls). A local contract replay costs no inference; local lexical search/NLI can avoid external provider round trips but brings model footprint and CPU latency. External retrieval plus LLM verification adds latency, token/API costs and tail risk. Selective routing and caching should be measured, including cache freshness.

**Privacy:** Shadow, Groq verifier, Serper and Wikipedia paths transmit user prompt/answer data to third parties. Before expanding them, classify permitted data and document provider retention/processing terms. A local curated corpus and local NLI offer better data locality; public web retrieval still discloses the query to the search service.

**False positives and abstention:** Fail-safe behavior increases “unverified” outcomes. That may reduce unsupported approvals but can lower coverage; measure the risk/coverage curve and provide qualified responses when evidence is incomplete.

**Source risk:** Primary source is not automatically correct for every question; sources can conflict, be compromised, or state a claim only within a scope. Authority must be policy-specific, with source/version metadata and contradictions exposed.

**Correction risk:** Any generated correction can drop context, change units/time scope, or add new errors. Do not regard correction success as established until re-verification and paired human evaluation show net gain.

**Rollback:** If later approved, gate the new verifier behind a feature flag; preserve current behavior and baseline artifacts; log both candidate and current decisions in shadow mode; compare sealed-set results and operational metrics; disable immediately if false approvals, privacy exposure, latency, or error rate breach agreed limits. No threshold or model changes should share a release with the adversarial C2 work.

**Decisions for owner review before implementation:**

1. Approve creation of a new reliability-specific labeled evaluation artifact and human annotation effort, leaving all existing canonical/C2 data untouched.
2. Choose initial evaluation domain and acceptable source authority policy; if none is chosen, use frozen public fact QA plus a small source-annotated local fixture set.
3. Approve whether a later bounded live-source pilot may send prompts to Serper/Groq and incur usage costs; no such calls were made in this audit.
4. Approve proposed auto-verification precision and coverage gates, or set product-specific risk tolerances.
5. Approve any new local NLI model, external service, dependency, or retraining only after the replay report and data/license review.

## 9. Final recommendation

The most serious verified weakness is not lack of a vector database: it is **permissive verdict semantics combined with weak evidence provenance**. The source currently permits uncertain/malformed search judgments and word overlap to look consistent; the temporal acceptance path can then label the original answer fully provenanced without a confidence gate. The Wikipedia path can turn a generated answer over one summary directly into a correction with hard-coded confidence and no final verification.

Run the verifier decision-contract replay first. It is the smallest experiment with a directly observable expected win and zero external cost. If strict semantics reduce false approval but hurt coverage, the second-best option is claim-level post-answer evidence matching against a bounded, curated source set, initially comparing lexical/deterministic and lightweight NLI baselines. Do not begin with all-purpose RAG or more shadow calls. Use shadow disagreement to route checks and agreement as a weak uncertainty feature only; let evidence quality, claim support, calibrated risk and explicit abstention determine trust.

## References

- Manakul, Liusie & Gales (2023), [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection](https://arxiv.org/abs/2303.08896).
- Farquhar et al. (2024), [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0).
- Sawczyn et al. (2025), [FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs](https://arxiv.org/abs/2503.17229).
- Bohnet et al. (2024), [Provenance: A Light-weight Fact-checker for Retrieval Augmented LLM Generation Output](https://arxiv.org/abs/2411.01022).
- Wang et al. (NAACL 2025), [Multi-Perspective Preference Alignment for Retrieval-Augmented Generation](https://aclanthology.org/2025.naacl-long.459/), including discussion of evidence relevance, robustness to noisy context, citation quality and pipeline costs.
