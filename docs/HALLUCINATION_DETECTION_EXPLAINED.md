# FIE — Hallucination Detection: Complete Technical Explainer

> A single, self-contained reference for the hallucination-detection half of the Failure
> Intelligence Engine (FIE): the tech stack, the end-to-end architecture, how each detection
> signal works, **why** each piece was designed the way it is, how it evolved, and an honest,
> measured evaluation of what it actually does. Written to be read start-to-finish before a
> systems / ML interview.

Last updated: 2026-07-14. Companion docs: [HALLUCINATION_PIPELINE.md](HALLUCINATION_PIPELINE.md)
(current design, kept in sync with code) and
[HALLUCINATION_OPTIMIZATION_LOG.md](HALLUCINATION_OPTIMIZATION_LOG.md) (measured numbers, newest first).

---

## Table of contents

1. [What problem this solves](#1-what-problem-this-solves)
2. [The core idea in one paragraph](#2-the-core-idea-in-one-paragraph)
3. [Tech stack and why each piece](#3-tech-stack-and-why-each-piece)
4. [System architecture (the 12-node pipeline)](#4-system-architecture-the-12-node-pipeline)
5. [The request lifecycle, node by node](#5-the-request-lifecycle-node-by-node)
6. [The detection signals in depth (the math)](#6-the-detection-signals-in-depth-the-math)
7. [The XGBoost meta-classifier (signal fusion)](#7-the-xgboost-meta-classifier-signal-fusion)
8. [Ground-truth verification and auto-correction](#8-ground-truth-verification-and-auto-correction)
9. [Design decisions and why (the interesting part)](#9-design-decisions-and-why-the-interesting-part)
10. [How it evolved (v2 → slim)](#10-how-it-evolved-v2--slim)
11. [Honest evaluation — what it actually does](#11-honest-evaluation--what-it-actually-does)
12. [Known limitations and what I'd do next](#12-known-limitations-and-what-id-do-next)
13. [Interview Q&A crib sheet](#13-interview-qa-crib-sheet)
14. [Glossary](#14-glossary)

---

## 1. What problem this solves

Large language models produce fluent, confident text that is sometimes **factually wrong**
(a "hallucination") or contains a **reasoning error** (an arithmetic slip, a logical gap, a
conclusion built on a false premise). In production you usually only have the model's output —
you don't have a labelled "is this true?" signal. FIE's hallucination path is a **monitoring
and correction layer** that sits *around* a primary model and estimates, per response, the
probability that the response is a failure — then optionally verifies it against external
ground truth and auto-corrects it.

Two halves of FIE, kept deliberately separate:

- **Adversarial guard** (`fie/adversarial.py`) — an offline, ~48 ms classifier that detects
  *malicious prompts* (jailbreaks, prompt injection). This is the product's fast "wedge."
- **Hallucination pipeline** (`engine/pipeline/langgraph_pipeline.py`) — the heavier,
  online path this document is about: it detects *factual/reasoning failures* in a model's
  answer. They share some infrastructure (the SBERT encoder, the guard runs inside the
  pipeline too), but they answer different questions: "is the *prompt* an attack?" vs. "is the
  *answer* a hallucination?"

---

## 2. The core idea in one paragraph

You cannot know if a single answer is true just by looking at it. But you **can** get a strong
signal by asking the same question to several independent models and measuring **how much they
disagree** — genuine facts produce agreement, hallucinations produce scatter (this is
*self-consistency / semantic entropy*). FIE turns that intuition into a pipeline: fan the
prompt out to a **shadow ensemble** of LLMs, quantify their disagreement with information-
theoretic and embedding-based measures, separately **verify the reasoning steps** and **check
factual claims against external ground truth** (Wikidata / web search), and finally **fuse all
those signals with a gradient-boosted classifier (XGBoost)** into one calibrated failure
probability. High-risk answers can then be auto-corrected from the verified source or the
shadow consensus. The whole flow is orchestrated as a **stateful graph (LangGraph)** so each
step is explicit, conditionally skippable, and auditable.

---

## 3. Tech stack and why each piece

| Layer | Technology | Role in hallucination detection | Why this choice |
|---|---|---|---|
| API | **FastAPI** | Serves the `/monitor` endpoint that runs the pipeline | Async, type-checked (Pydantic), fast; standard for Python ML services |
| Orchestration | **LangGraph** (`StateGraph`) | Runs the 12 nodes as a stateful DAG with conditional routing | Explicit shared state, conditional edges (skip expensive work), built-in traceability — see §9 |
| Shadow LLMs | **Groq API** (`llama-3.3-70b-versatile`, `openai/gpt-oss-120b`, `qwen/qwen3-32b`) | The "second opinions" whose disagreement is the primary signal | Groq is very low-latency inference; three *different model families* reduce correlated errors |
| Fast LLM | **Groq `llama-3.1-8b-instant`** | Cheap helper for claim extraction, step decomposition, probe generation, GT verification | 8B model is cheap/fast for structured sub-tasks that don't need a 70B |
| Embeddings | **sentence-transformers `all-MiniLM-L6-v2`** (384-dim, L2-normalised) | Semantic similarity for consistency, self-consistency medoid, socratic contradiction, logical-gap detection | Runs **locally** on CPU/small GPU, ~90 MB, cosine = dot product after normalisation; no API cost |
| Signal fusion | **XGBoost** (`failure_classifier_slim.pkl`, 10 features) | Combines all signals into one calibrated failure probability | Gradient boosting excels at tabular signal fusion with nonlinear interactions; fast inference; easy to calibrate |
| Factual GT | **Wikidata** (`wbsearchentities` API) | Structured verification of factual claims (entity/property/value) | Free, structured, high-precision for encyclopedic facts |
| Live GT | **Serper.dev** (Google Search API) | Real-time verification for temporal/"live world state" questions | Facts that change over time can't be in a static KB; needs live search |
| RAG fallback | **Wikipedia REST API** + Groq | Retrieve-and-ground correction when Wikidata has no structured value | Cheap grounding source for factual auto-correction |
| Reasoning eval | **Python `ast`** (safe arithmetic sandbox) | Deterministically checks arithmetic inside reasoning | Arithmetic can be *proven* wrong offline — no LLM needed, zero cost, 100% precision |
| Persistence | **MongoDB (Atlas)** | Stores inference records, signal logs, hot-reloadable thresholds | Flexible schema for heterogeneous signal logs; config without redeploy |
| Vector index | **FAISS** | Nearest-neighbour registry of confirmed adversarial archetypes | Fast similarity search for the guard's growing attack memory |

**One-line rationale:** local + deterministic where possible (SBERT, `ast`, XGBoost), online
LLMs only where a genuine second opinion is needed (shadows), external APIs only when ground
truth is required (Wikidata/Serper).

---

## 4. System architecture (the 12-node pipeline)

The pipeline is a **linear LangGraph with conditional skips** (`engine/pipeline/langgraph_pipeline.py`).
State is a single `MonitorState` TypedDict that every node reads from and returns partial
updates to; LangGraph merges them.

```
load_session → shadow_inference → adversarial_guard
   → (attack? → finalize) → signal_extract → provenance_gate
   → reasoning_verify → jury_deliberate → security_checks
   → (high-risk? → gt_verify) → (escalate | auto_correct) → finalize → END
```

Three conditional gates control cost:
- **After `adversarial_guard`** — if the prompt is an attack, skip everything and go straight
  to `finalize` (persist the block).
- **After `security_checks`** — only run the expensive external ground-truth check if the
  answer is *high failure risk* (and, in full-jury mode, the jury agrees).
- **After `gt_verify`** — route to `escalate` (human review) or `auto_correct` depending on
  whether a reliable correction exists.

Why a graph and not a function? See §9 — short version: explicit state, conditional routing,
and a built-in audit trail (`pipeline_trace`).

---

## 5. The request lifecycle, node by node

A request arrives with at minimum `{prompt, primary_output, run_full_jury}` —
the user's question and the model answer we're judging.

1. **`load_session`** — pulls prior conversation turns from the session store (for multi-turn
   context). Cheap Mongo read.

2. **`shadow_inference`** — the heart of the detector. Fans the *prompt* out to the 3 Groq
   shadow models **in parallel** (`ThreadPoolExecutor`, so wall-clock = slowest model). Each
   shadow is also asked to self-rate confidence (`HIGH/MEDIUM/LOW` → weight 3/2/1) and a
   **canary token** is injected into their system prompt to detect prompt-exfiltration. Output:
   `model_outputs = [primary_answer, shadow1, shadow2, shadow3]`. Responses are cached by
   `(model, prompt)` for 1 h to avoid duplicate cost.

3. **`adversarial_guard`** — runs the offline attack scan on the prompt (~48 ms, no LLM). If it
   flags an attack, the pipeline short-circuits to `finalize`. *(Measured caveat: this layer
   over-blocks ~30% of benign factual questions — see §11.)*

4. **`signal_extract`** — computes the failure-signal vector from `model_outputs`:
   consistency/agreement, **semantic entropy**, ensemble disagreement, and the primary-vs-
   shadow embedding distance. Classifies the *question type* (FACTUAL / TEMPORAL / REASONING /
   CODE / OPINION / IDENTITY) which gates later stages. Assigns a failure *archetype* and
   updates an EMA drift tracker. All local (SBERT + math), no LLM.

5. **`provenance_gate`** — labels where the answer's truth *should* come from
   (GENERAL_KNOWLEDGE / LIVE_WORLD_STATE / USER_SPECIFIC_STATE / MIXED). If live data is needed
   but the risk gate won't run GT, it marks the answer `NULL_REQUIRED_BUT_MISSING` — i.e. "we
   couldn't verify what this needed." This is an *honesty* mechanism: the system says "unknown"
   instead of pretending.

6. **`reasoning_verify`** (FACTUAL/REASONING only) — decomposes the answer into typed steps,
   verifies each, and runs adversarial "socratic" probes (§6). Catches arithmetic and logical
   errors the ensemble signal alone would miss.

7. **`jury_deliberate`** (full-jury mode only) — a "DiagnosticJury" of three specialist agents
   (adversarial specialist, linguistic auditor, domain critic) that produce a root-cause
   verdict + confidence. *(It also currently makes a Groq call to write a human-readable
   explanation — see §11, this is the slowest node.)*

8. **`security_checks`** — multi-turn "crescendo" escalation tracking + model-extraction rate
   heuristics. Independent of the hallucination score.

9. **`gt_verify`** (only if high-risk) — the ground-truth pipeline: routes by question type to
   Wikidata (facts), Serper (live/temporal), or shadow self-consistency (reasoning/code). §8.

10. **`escalate` / `auto_correct`** — if GT found a reliable correct answer, replace the output;
    if consensus is strong, use the shadow medoid; else escalate to human review.

11. **`finalize`** — runs the **XGBoost meta-classifier** over all collected signals to produce
    the final calibrated `xgb_probability` and `xgb_is_failure` decision, persists the record,
    logs the signal vector for future calibration, and fires notifications.

---

## 6. The detection signals in depth (the math)

### 6.1 Self-consistency / semantic entropy (primary signal)

**Intuition:** ask N models; cluster their answers by *meaning*; if they concentrate in one
meaning-cluster the answer is probably right; if they scatter across clusters it's probably a
hallucination.

- **Consistency / agreement** (`engine/detector/consistency.py`): group the N outputs into
  semantic clusters and compute an agreement score (fraction in the majority meaning) and the
  answer-cluster counts.
- **Semantic entropy** (`engine/detector/entropy.py`): Shannon entropy over the *meaning-cluster*
  distribution, `H = -Σ pᵢ log pᵢ`, computed from the same cluster counts so entropy and
  agreement can never disagree. Low H = confident consensus; high H = scattered = risk. This is
  the well-known *semantic entropy* idea (Kuhn et al.): entropy over meanings, not raw strings,
  so "Paris" and "The capital is Paris" count as the same answer.
- **Ensemble disagreement** (`engine/detector/ensemble.py`): mean pairwise cosine similarity of
  SBERT embeddings across all outputs; `disagreement = 1 − similarity`.
- **Primary outlier check**: is the *primary* answer the odd one out vs. the shadows?

A guard rail (`failure_agent._build_signal`): ensemble disagreement is only allowed to raise
risk when entropy > 0 — otherwise trivial paraphrase differences ("Paris" vs "The capital of
France is Paris") would false-positive.

**Why 3 *different* model families?** Sampling one model many times gives you its *precision*
but not its *blind spots* — a model confidently wrong stays wrong across samples. Different
families (Llama, GPT-OSS, Qwen) have *decorrelated* errors, so genuine disagreement surfaces
model-specific hallucinations. (Trade-off and a cheaper alternative are discussed in §12.)

### 6.2 Reasoning verification (`engine/reasoning/`)

Three phases, each with an offline fallback so it degrades gracefully without Groq:

1. **Decompose** (`chain_decomposer.py`): split the answer into typed atomic steps
   (ARITHMETIC / FACTUAL_PREMISE / LOGICAL_INFERENCE / DEFINITION / CONCLUSION). Uses the fast
   Groq model when available, else a regex heuristic splitter.
2. **Verify steps** (`step_verifier.py`):
   - **Arithmetic** — parsed and evaluated in a **safe `ast` sandbox** (only arithmetic
     operators whitelisted). If `120/2 = 50` appears, it's *proven* wrong offline with 100%
     precision, zero cost. This is the single highest-precision signal in the system.
   - **Factual premise** — routed through the GT pipeline (or a hedge-language heuristic offline).
   - **Logical inference** — SBERT similarity between a step and its prior context; a large
     semantic jump = a logical gap.
   - **Conclusion** — flagged if it's built on an already-failed prior step (error propagation).
3. **Socratic probe** (`socratic_probe.py`): generate ~2 adversarial challenge questions
   targeting the weakest assumption, then check whether the shadow outputs *contradict* the
   answer (contradiction markers + embedding divergence). Catches hidden-assumption errors that
   look internally consistent.

**Why include FACTUAL, not just REASONING?** Most production hallucinations are factual errors
dressed up as answers; restricting reasoning checks to "REASONING" questions would miss the
majority.

### 6.3 The DiagnosticJury (`engine/agents/`)

Three specialist agents vote and a primary verdict + confidence is aggregated:
- **Adversarial specialist** — is this a disguised attack / injection?
- **Linguistic auditor** — pure-regex checks for double negation, ambiguous reference, nested
  clauses, contradictions, multi-hop complexity (proxies for prompts that induce failure).
- **Domain critic** — SBERT self-contradiction check (primary vs. secondary output divergence).

The jury exists to turn raw signals into an interpretable *root cause*
(FACTUAL_HALLUCINATION / KNOWLEDGE_BOUNDARY_FAILURE / TEMPORAL_KNOWLEDGE_CUTOFF / …) that both
the classifier and the human-facing explanation consume.

---

## 7. The XGBoost meta-classifier (signal fusion)

All the above produce heterogeneous signals. Rather than hand-tuning thresholds, FIE learns
the mapping from signals → failure probability with **XGBoost** (`engine/failure_classifier.py`).

- **Slim model, 10 features** (`failure_classifier_slim.pkl`): agreement_score,
  entropy_score, jury_confidence, high_failure_risk, fix_confidence, requires_escalation, plus
  a few one-hot flags (`jury_verdict==FACTUAL_HALLUCINATION`, `==KNOWLEDGE_BOUNDARY_FAILURE`,
  `fix_strategy==NONE`, `question_type==FACTUAL`). A SHAP ablation (June 2026) showed these 10
  features reproduce the full 560-one-hot-feature model's AUC, so the slim path drops pandas /
  get_dummies entirely — a pure 10-float vector → `predict_proba`. Faster and simpler.
- **Temperature calibration**: raw XGBoost probabilities are over-confident, so a Platt-style
  temperature scaling `σ(logit(p)/T)` (T≈1.15) is applied so the probability is trustworthy for
  thresholding.
- **Per-question-type thresholds** (hot-reloadable from Mongo via `fie_config`): CODE /
  REASONING / OPINION are known blind spots and get their own thresholds; the default is ≈0.52.
- **Graceful fallback**: if the model file is missing, it falls back to the rule-based
  `high_failure_risk` flag (POET) so the system never hard-fails.

**Why XGBoost and not a neural net or fixed rules?** The inputs are ~10 tabular signals with
nonlinear interactions and limited training data (~2 k labelled rows). Gradient-boosted trees
are the state of the art for exactly this regime: strong with little data, fast inference
(microseconds), no GPU, and the SHAP tooling makes the feature contributions auditable — which
matters for a system whose whole selling point is explainability.

---

## 8. Ground-truth verification and auto-correction

`engine/verifier/ground_truth_pipeline.py` routes by question type:

- **FACTUAL** → extract an atomic claim (`subject / property / value`) and check it against
  **Wikidata**; if Wikidata contradicts with high confidence, the verified value becomes the
  correction.
- **TEMPORAL / live** → **Serper** Google search, then a Groq fact-check comparing the answer
  to the live snippet.
- **REASONING / CODE** → no external source exists, so use **shadow self-consistency**
  (`self_consistency_verifier.py`): embed all shadow outputs, take the **medoid** (the most
  representative answer) as pseudo-ground-truth if consistency clears a threshold.
- **OPINION / IDENTITY** → no ground truth is meaningful; the jury verdict is authoritative.

Verified answers ≥ 0.90 confidence are **write-through cached** so repeat questions skip the
external call.

**Auto-correction** (`auto_correct` node) tries, in order: (1) the GT verified answer,
(2) weighted shadow consensus via the fix engine, (3) Wikipedia RAG grounding. If none is
reliable, it **escalates to human review** rather than guessing — again, the honesty principle.

---

## 9. Design decisions and why (the interesting part)

**Why a shadow ensemble at all?** Because there is no free ground-truth signal on a single
production answer. Cross-model disagreement is the cheapest available proxy for "this is
uncertain / likely wrong," and it needs no labels.

**Why LangGraph over a plain function or a linear script?**
- *Explicit shared state* — 40+ fields flow between stages; a single `MonitorState` beats
  threading dozens of return values.
- *Conditional routing* — attacks skip the whole pipeline; low-risk answers skip the expensive
  external GT. The graph expresses this cleanly with conditional edges.
- *Auditability* — every node appends to `pipeline_trace`, giving a per-request narrative
  ("why did FIE decide this?"), which is essential for an explainability product.
- *Composability* — nodes are pure `state → partial update` functions, trivially testable and
  reorderable (which is exactly what the current optimization work exploits).

**Why an offline SBERT encoder instead of an embedding API?** Embeddings are called *many*
times per request (consistency, self-consistency, socratic, logical-gap). An API call per
embedding would dominate latency and cost. `all-MiniLM-L6-v2` runs locally in milliseconds and
is "good enough" for semantic clustering.

**Why deterministic checks (arithmetic `ast`, regex) alongside LLMs?** Whenever a failure can
be *proven* (arithmetic) or matched with a cheap pattern (hedging language), doing so is
higher-precision and zero-cost compared to asking an LLM. LLMs are reserved for the parts that
genuinely need judgement.

**Why temperature-calibrate and per-type thresholds?** A probability is only useful if it's
*calibrated* — 0.7 should mean 70%. Raw boosted-tree scores aren't, and different question
types have different base rates and blind spots, so one global threshold is wrong. Thresholds
live in Mongo so they can be tuned without a redeploy.

**Why the provenance/escalation "honesty" machinery?** A monitor that confidently corrects with
a wrong answer is worse than one that says "I can't verify this." So when no reliable source
exists, FIE labels the answer unverified and escalates rather than fabricating a fix.

**Why keep the adversarial guard and the hallucination path separate but co-located?** They
answer different questions (malicious prompt vs. untrue answer) and have wildly different
latency budgets (48 ms vs. seconds). Keeping the guard as a fast offline gate *inside* the
pipeline lets attacks short-circuit before paying for shadow inference — though the current
node ordering doesn't yet exploit that (a known optimization, see the optimization log).

---

## 10. How it evolved (v2 → slim)

- **v2 → v3 → v4 classifier**: successive retrains as more labelled signal logs accumulated
  and features were added (question type, provenance category, reasoning failure type). v4 used
  a ~560-column one-hot feature space.
- **slim-v1 (June 2026)**: a SHAP feature-importance ablation showed **10 features reproduce the
  full model's AUC**. The slim model removed the pandas/one-hot machinery entirely for a pure
  10-float inference path — a deliberate *simplification* backed by measurement, not a guess.
- **Ollama → Groq**: shadow inference originally used local Ollama models (mistral / llama3.2 /
  phi3) but moved to the Groq API because a 4 GB GPU can't run three 7B+ models concurrently at
  acceptable latency; Groq gives 70B-class quality with low latency and no local VRAM cost.
  (The Ollama path still exists behind a flag for offline mode.)
- **Measurement discipline added (July 2026)**: an instrumentation layer + a decontaminated,
  frozen benchmark harness were built to measure latency, cost, and detection quality honestly
  (§11). This is when the leakage in the historical accuracy numbers was caught.

---

## 11. Honest evaluation — what it actually does

*(All numbers measured July 2026 on a frozen, decontaminated benchmark; see the optimization
log for commands and CIs. Being able to discuss these honestly is the point.)*

### Performance (per request, full pipeline, cold cache)

| Metric | Value |
|---|---|
| End-to-end latency (p50 / p95) | **10.5 s / 27.6 s** |
| Groq API calls / request | **6.45** |
| Tokens / request | **~2 325** |
| Slowest nodes (p50) | jury 7.2 s · reasoning 3.0 s · shadow 2.5 s · GT 2.2 s |

So it is **heavy, online, and multi-call** — the reason the current optimization project
exists. A notable finding: the single most expensive node (`jury_deliberate`, 7.2 s) is slow
because it makes a *cosmetic* Groq call to write a human-readable explanation — not because the
diagnosis itself is expensive.

### Detection quality — the important, nuanced truth

The headline lesson mirrors what we already learned on the adversarial side: **historical
accuracy numbers were inflated by train/test contamination.** The XGBoost classifier was
trained on TruthfulQA + HaluEval + MMLU questions, and the previously-reported AUC 0.65–0.95
was measured on *the same questions it trained on*.

Measured honestly (decontaminated, held-out):

| Setting | ROC-AUC | Meaning |
|---|---|---|
| TruthfulQA, decontaminated, end-to-end | **0.497** | at chance on *adversarial* misconceptions |
| Classifier features, question-disjoint CV, **HaluEval** | **0.956** | near-perfect on *non-adversarial* hallucinations |
| Classifier features, question-disjoint CV, TruthfulQA | 0.581 | near chance |
| Classifier features, question-disjoint CV, MMLU | 0.536 | near chance |

**Interpretation (this is the key insight to be able to explain):** consistency-based detection
works when a model's *wrong* answer **diverges** from the ensemble (HaluEval-style HotpotQA
hallucinations → the shadows disagree → detected). It fails when the wrong answer is a
**shared misconception** the strong shadow models *also* believe (TruthfulQA is literally
designed to elicit these → the shadows agree on the wrong answer → nothing to detect). So the
detector is **not "at chance everywhere"** — it's at chance specifically on the class of
hallucination that consensus fundamentally cannot catch. This is a property of the method, not
a bug, and it tells us exactly where a *different* signal (ground-truth verification) is
required. *(A clean end-to-end HaluEval re-measurement is in progress to confirm the offline
0.956 holds through the current pipeline.)*

### Over-refusal finding

The adversarial guard, running inside the pipeline, **blocks ~30% of benign factual questions**
on TruthfulQA (13.5% on HaluEval) — 100% of it from the **PAIR intent-classifier layer**
mislabelling ordinary questions as `JAILBREAK_ATTEMPT`. This is the same benign-FPR / out-of-
distribution problem documented in the adversarial research log, surfacing on the hallucination
path.

---

## 12. Known limitations and what I'd do next

- **Shared-misconception hallucinations are undetectable by consensus.** Fix: lean on
  ground-truth verification (Wikidata/Serper) and NLI-based factuality for FACTUAL questions,
  not just ensemble agreement.
- **Latency (p50 10.5 s) and cost (6.45 Groq calls) are high.** Concrete wins already
  identified: defer the cosmetic explanation LLM call, move the guard before shadow inference,
  drop redundant per-step GT, early-exit on stable answers.
- **3 separate shadow models are expensive.** A single model with k temperature samples +
  local-NLI semantic entropy would give most of the signal at a fraction of the cost — with the
  honest caveat that it loses the *cross-model* decorrelation that catches model-specific
  hallucinations.
- **Fully offline mode** (Ollama small model + local Wikipedia index + local NLI) is feasible
  on a 4 GB GPU but trades detection quality (weaker shadows, no live search) for zero API cost
  and privacy.
- **The shipped slim classifier looks under-fit** (in-sample AUC 0.586 vs. a fresh retrain's
  0.743 CV) — worth retraining on clean, current-pipeline features.

---

## 13. Interview Q&A crib sheet

**Q: How do you detect a hallucination without ground-truth labels?**
Cross-model self-consistency: fan the prompt to several independent LLMs and measure semantic
entropy / disagreement across their answers. Agreement ≈ likely true; scatter ≈ likely
hallucinated. Then verify factual claims against Wikidata/web and fuse everything with XGBoost.

**Q: Why semantic entropy and not just string matching?**
Because "Paris" and "The capital of France is Paris" are the same answer. We cluster by
*meaning* (SBERT embeddings) and take entropy over meaning-clusters, so paraphrases don't
inflate disagreement.

**Q: Why XGBoost for fusion?**
~10 tabular signals, nonlinear interactions, ~2 k labelled rows, need fast CPU inference and
auditable feature importances (SHAP). Gradient-boosted trees dominate that regime; a neural net
would overfit and a fixed rule set can't capture the interactions.

**Q: How do you keep it honest / explainable?**
Every node logs to a `pipeline_trace`; provenance labels say where truth should come from; when
no reliable source exists it escalates to a human instead of fabricating a correction; and the
calibrated probability is thresholded per question type.

**Q: What went wrong / what did you learn?**
The historical accuracy (AUC 0.65–0.95) was train/test contaminated. After building a
decontaminated benchmark, the real picture is that the detector is near-perfect on
non-adversarial hallucinations (HaluEval ~0.95) but at chance on shared-misconception
benchmarks (TruthfulQA ~0.50) — because consensus can't catch errors the ensemble shares. That
diagnosis (measure before optimizing, and decontaminate first) is the most important thing I
did.

**Q: How would you make it production-viable (fast/cheap/offline)?**
Defer the cosmetic explanation call, guard-before-shadow early-exit, collapse the 3-model
fan-out to single-model self-consistency + local NLI, and add a local factual-verification path
so the FACTUAL class (which consensus can't do) is covered offline.

---

## 14. Glossary

- **Shadow model** — an auxiliary LLM the prompt is also sent to, to get an independent answer.
- **Semantic entropy** — Shannon entropy over *meaning-clusters* of the ensemble's answers; high
  = uncertain/likely hallucination.
- **Medoid** — the ensemble answer most similar to all the others; used as pseudo-ground-truth.
- **Provenance** — where an answer's truth *should* come from (general knowledge / live world /
  user-specific); used to decide what verification is required and to say "unverified" honestly.
- **Archetype** — a learned cluster/label of failure type (e.g. STABLE, FACTUAL_HALLUCINATION).
- **Decontamination** — removing benchmark questions the model trained on before measuring, so
  accuracy isn't inflated by memorisation.
- **PAIR classifier** — the adversarial guard's learned prompt-intent layer (the source of the
  benign over-refusal noted in §11).
- **POET** — the rule-based fallback used when the XGBoost model is unavailable.
```
