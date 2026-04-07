# Failure Intelligence Engine (FIE)
### The First AI Reliability Platform That Does Not Just Detect Failures — It Fixes Them

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [The Solution — What FIE Is](#3-the-solution--what-fie-is)
4. [Why FIE Is Unique — The Core Differentiator](#4-why-fie-is-unique--the-core-differentiator)
5. [Competitive Comparison](#5-competitive-comparison)
6. [System Architecture — How It Works](#6-system-architecture--how-it-works)
7. [The 4-Phase Intelligence Pipeline](#7-the-4-phase-intelligence-pipeline)
8. [All Algorithms and Logic Used](#8-all-algorithms-and-logic-used)
9. [Explainability Layer](#9-explainability-layer)
10. [Security and Multi-Tenancy](#10-security-and-multi-tenancy)
11. [Dashboard and User Interface](#11-dashboard-and-user-interface)
12. [Python SDK — Integration in Minutes](#12-python-sdk--integration-in-minutes)
13. [API Reference](#13-api-reference)
14. [Technology Stack](#14-technology-stack)
15. [Deployment Architecture](#15-deployment-architecture)
16. [How Any Company Can Use FIE](#16-how-any-company-can-use-fie)
17. [Roadmap — What We Are Building Next](#17-roadmap--what-we-are-building-next)
18. [Full Updated Pipeline After Roadmap](#18-full-updated-pipeline-after-roadmap)
19. [Local Development Setup](#19-local-development-setup)
20. [Project Structure](#20-project-structure)

---

## 1. Executive Summary

Large language models (LLMs) like GPT-4, Claude, and Llama are being embedded into critical products — customer services, legal research, healthcare support, financial advice, and education. These models fail silently. They produce different kind of wrong answers like  confident wrong answers, hallucinate facts, get hijacked by adversarial prompts, and degrade over time without any visible signal.

**No existing tool fixes this.** The entire market — Arize, WhyLabs, Langfuse, Galileo, Patronus, Confident AI — detects failures and sends alerts. A human then has to investigate and act.

**Failure Intelligence Engine (FIE)** is the only platform that closes the loop. It detects the failure, diagnoses the root cause, and automatically applies the correct fix — returning a safer, grounded answer to your application in real time.

FIE is a 4-phase AI reliability engine:
- **Phase 1** — Signal extraction: measures output entropy, agreement, ensemble disagreement, semantic distance
- **Phase 2** — Archetype labeling and adaptive clustering with trend tracking
- **Phase 3** — Diagnostic jury: three specialist agents deliberate in parallel on root cause
- **Phase 4** — Auto-fix engine: selects and applies the right mitigation strategy

The system is production-deployed on Google Cloud Run with a React dashboard, a Python SDK published on PyPI, MongoDB-backed multi-tenant storage, and Google OAuth authentication.

---

## 2. Problem Statement

### The LLM Reliability Crisis

Every company shipping AI-powered products faces the same invisible problem: LLMs fail unpredictably and silently.

#### What these failures look like in production

| Failure Type | Example | Business Impact |
|---|---|---|
| **Hallucination** | Model confidently states a wrong fact | User acts on bad information |
| **Prompt Injection** | Attacker overrides system prompt | Security breach, data leakage |
| **Model Inconsistency** | Same question gives different answers on refresh | Trust erosion, user confusion |
| **Knowledge Cutoff** | Model answers "current" questions with stale data | Wrong decisions based on outdated info |
| **Prompt Complexity Failure** | Multi-part question gets partial answer | Incomplete output, degraded UX |
| **Overconfident Errors** | Model presents wrong answer with high certainty | Most dangerous failure — user doesn't question it |
| **Model Drift** | Accuracy degrades after model version update | Silent product degradation, no alert |

#### Why existing monitoring tools are not enough

Every current tool in the market follows the same pattern:

```
LLM output → Detect problem → Send alert → Human investigates → Human fixes
```

This means:
- A hallucination reaches the user before anyone knows about it
- A prompt injection attack succeeds before the alert is even read
- Engineers spend hours triaging alerts instead of building
- There is no automatic recovery — every fix is manual

This is the problem FIE solves.

---

## 3. The Solution — What FIE Is

FIE sits between your application and your LLM. It intercepts outputs, runs them through a multi-stage intelligence pipeline, and returns either the original output (if safe) or a corrected one (if a fix is possible and confident).

```
Your App  →  LLM  →  FIE  →  Corrected Output  →  Your App
```

### What FIE does in one request cycle

1. Your application calls an LLM and gets a response
2. FIE receives the prompt and primary output
3. FIE fans the same prompt out to three shadow models in parallel
4. FIE computes a Failure Signal Vector from all four outputs
5. FIE labels the failure archetype and clusters it against historical patterns
6. FIE runs three specialist agents in parallel to diagnose the root cause
7. FIE selects and applies the correct auto-fix strategy
8. FIE returns the fixed response with full explainability artifacts
9. Everything is stored in MongoDB, tenant-isolated, and visible in the dashboard

The entire cycle completes in 1–8 seconds. Your application gets back either the original answer or a corrected one — with no manual intervention required.

---

## 4. Why FIE Is Unique — The Core Differentiator

### The fundamental gap in the market

Every single competitor in the LLM observability space — from well-funded startups to enterprise APM tools — does the same thing: **passive monitoring**. They observe. They score. They alert.

**FIE is the only active reliability system.**

### Four capabilities that exist nowhere else

#### 1. Active Auto-Correction at Runtime

FIE does not send you an alert when your model hallucinates. FIE replaces the hallucination with a grounded, verified answer — in the same API response cycle.

No competitor does this. Arize alerts you. Galileo guards against it. FIE fixes it.

#### 2. Diagnostic Jury Architecture

FIE runs three specialist agents in parallel, each approaching the failure from a different angle — adversarial, linguistic, and domain. Their verdicts are aggregated with confidence weighting and priority rules. This produces expert-level root cause diagnosis, not just a "failure detected" flag.

No competitor uses a multi-agent jury. They use single LLM-as-judge evaluators.

#### 3. Shadow Model Ensemble for Consensus

FIE runs the same prompt through three Groq shadow models simultaneously. The agreement across these models is both a failure signal (high disagreement = unstable output) and a correction mechanism (majority vote = consensus answer). The shadow models pay for themselves by providing both detection and correction in a single parallel call.

No competitor maintains active shadow models running in parallel with your primary model.

#### 4. RAG + Verification + Substitution Loop

When the DomainCritic agent detects a factual hallucination and no shadow consensus is available, FIE retrieves relevant Wikipedia context, verifies the primary output against it using Groq, and substitutes the grounded answer. This is an end-to-end factual correction loop — not just detection.

No competitor corrects hallucinations. They detect them.

---

## 5. Competitive Comparison

### Market Landscape (2026)

The LLM observability market is crowded with passive monitoring tools. Here is where every major player sits and what they cannot do.

| Platform | Detect Failures | Root Cause Diagnosis | Auto-Correct | Shadow Models | Adversarial Defense | RAG Correction |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FIE (ours)** | Yes | Yes (3-agent jury) | **Yes** | **Yes (3 models)** | **Yes (3-layer)** | **Yes** |
| Arize AI | Yes | Partial (LLM judge) | No | No | No | No |
| WhyLabs | Yes | No | No | No | Detect only | No |
| Langfuse | Tracing only | No | No | No | No | No |
| Galileo AI | Yes | Partial | Guards only | No | Guards only | No |
| Patronus AI | Yes | No | No | No | No | No |
| Confident AI | Yes (50+ metrics) | No | No | No | No | No |
| Datadog LLM | Yes | No | No | No | Detect only | No |
| LangSmith | Tracing only | No | No | No | No | No |
| TruLens | Eval only | No | No | No | No | No |
| Evidently AI | Drift only | No | No | No | No | No |
| Maxim AI | Yes | No | Simulation only | No | No | No |
| Giskard | Test-time only | No | No | No | Test-time only | No |
| Helicone | Routing only | No | No | No | Basic | No |

### Feature-by-feature breakdown

#### Failure Detection

| Feature | FIE | Arize | WhyLabs | Galileo | Patronus |
|---|---|---|---|---|---|
| Shannon Entropy scoring | Yes | No | Statistical | No | No |
| Semantic clustering + agreement | Yes | No | No | No | No |
| All-pairwise ensemble disagreement | Yes | No | No | No | No |
| Sentence-transformer embedding distance | Yes | Partial | No | No | No |
| Adaptive clustering (novel anomaly detection) | Yes | No | No | No | No |
| EMA-based model degradation trends | Yes | No | Drift only | No | No |

#### Diagnosis

| Feature | FIE | Arize | WhyLabs | Galileo | Patronus |
|---|---|---|---|---|---|
| Multi-agent specialist jury | Yes | No | No | No | No |
| Adversarial detection (3-layer: regex + guard + FAISS) | Yes | No | Detect | No | No |
| Linguistic complexity + OOD detection | Yes | No | No | No | No |
| Factual verification via RAG | Yes | No | No | No | Partial |
| Temporal knowledge cutoff detection | Yes | No | No | No | No |
| Confidence-weighted verdict aggregation | Yes | No | No | No | No |

#### Correction

| Feature | FIE | Arize | WhyLabs | Galileo | Patronus |
|---|---|---|---|---|---|
| Shadow consensus auto-fix | Yes | No | No | No | No |
| Prompt sanitization and rerun | Yes | No | No | No | No |
| Context injection for temporal failures | Yes | No | No | No | No |
| Prompt decomposition for complexity | Yes | No | No | No | No |
| RAG grounding substitution | Yes | No | No | No | No |
| Self-consistency fallback | Yes | No | No | No | No |

#### Explainability

| Feature | FIE | Arize | Confident AI | Patronus | Galileo |
|---|---|---|---|---|---|
| Structured XAI bundles (decision trace, signals, evidence) | Yes | No | Partial | No | No |
| Admin vs user-safe redaction | Yes | No | No | No | No |
| Plain-language human summaries | Yes | Partial | No | No | No |
| Alternatives considered + uncertainty notes | Yes | No | No | No | No |

### Pricing comparison

| Platform | Free Tier | Paid Start | Enterprise |
|---|---|---|---|
| **FIE** | Open + self-host | TBD | TBD |
| Arize Phoenix | Free (unlimited, self-host) | $50/month | $50K–$100K/year |
| Langfuse | 50K units/month | $8 per 100K units | Custom |
| Patronus AI | 20 pages, 5 runs | $25/month | Custom |
| WhyLabs | 1 project, 10M preds | $125/month | Custom |
| Galileo | Usage-based | Usage-based | Custom |
| Confident AI | Limited | Usage-based | Custom |

### The one-line positioning

> Every competitor monitors your LLM. FIE heals it.

---

## 6. System Architecture — How It Works

### High-level architecture

```
┌──────────────────────────────────────────────┐
│              USER APPLICATION                 │
│   Python SDK / LangChain / Direct API call   │
└──────────────────┬───────────────────────────┘
                   │  prompt + primary_output
                   ▼
┌──────────────────────────────────────────────┐
│           FIE BACKEND (FastAPI)               │
│   Google Cloud Run / Docker container         │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │           GROQ SHADOW MODELS            │ │
│  │  llama-3.1-8b + llama-3.3-70b +        │ │
│  │  llama-3.2-3b (parallel fan-out)        │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ PHASE 1  │  │ PHASE 2  │  │  PHASE 3  │  │
│  │ Signal   │→ │Archetype │→ │ Diagnostic│  │
│  │Extraction│  │Clustering│  │   Jury    │  │
│  └──────────┘  └──────────┘  └───────────┘  │
│                                    │         │
│  ┌──────────┐  ┌──────────┐        ▼         │
│  │ PHASE 4  │  │   XAI    │  ┌───────────┐  │
│  │ Auto-Fix │→ │ Explain  │  │    RAG    │  │
│  │  Engine  │  │  Layer   │  │ Grounder  │  │
│  └──────────┘  └──────────┘  └───────────┘  │
└──────────────────────┬───────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌─────────┐ ┌─────────────┐
   │  MongoDB   │ │ FAISS   │ │React/Vercel │
   │   Atlas    │ │  Index  │ │  Dashboard  │
   └────────────┘ └─────────┘ └─────────────┘
```

---

## 7. The 4-Phase Intelligence Pipeline

### Phase 1 — Failure Signal Extraction

FIE converts raw LLM outputs into a structured Failure Signal Vector (FSV). This is the numeric fingerprint of a potential failure.

**Input**: prompt + primary output + shadow model outputs
**Output**: Failure Signal Vector (7 fields)

```
agreement_score      → How many outputs semantically agree (0–1)
fsd_score            → First-second dominance gap (0–1)
answer_counts        → Semantic cluster sizes
entropy_score        → Normalized Shannon entropy across clusters (0–1)
ensemble_disagreement→ Whether any model pair disagrees below threshold
ensemble_similarity  → Mean pairwise cosine similarity across all models
high_failure_risk    → Composite boolean risk flag
```

**Risk flag logic**:
```
high_failure_risk = (
    entropy_score >= 0.75  OR
    agreement_score <= 0.80  OR
    (ensemble_disagreement AND entropy_score > 0.0)
)
```

### Phase 2 — Archetype Labeling and Adaptive Clustering

**Archetype labeling** (rule-based, 7 archetypes):

| Archetype | Trigger Conditions |
|---|---|
| HALLUCINATION_RISK | ensemble_disagreement=True AND entropy≥0.75 |
| OVERCONFIDENT_FAILURE | high_risk=True AND entropy<0.25 |
| MODEL_BLIND_SPOT | ensemble_disagreement=True AND (entropy>0 OR agreement≤0.80) |
| RESOURCE_CONSTRAINT | latency≥3000ms AND entropy≥0.75 |
| UNSTABLE_OUTPUT | entropy≥0.75 (alone) |
| LOW_CONFIDENCE | agreement≤0.80 (alone) |
| STABLE | No conditions met |

**Adaptive clustering** — each new signal is scored against existing cluster centroids. The similarity threshold grows logarithmically as the cluster count increases:

```
threshold = 0.80 + log(num_clusters + 1) × 0.03    (capped at 0.92)
```

Cluster types:
- `KNOWN_FAILURE` — score ≥ threshold (seen before)
- `NOVEL_ANOMALY` — score < 0.45 (first time seen)
- `AMBIGUOUS` — between the two bounds

**EMA trend tracking** — Exponential Moving Average with α=0.94 tracks:
- `ema_entropy` — average model divergence
- `ema_agreement` — average model consistency
- `ema_disagreement_rate` — frequency of ensemble failures
- `ema_high_risk_rate` — frequency of high-risk signals

Degradation is detected when velocity (first-half vs second-half risk rate) exceeds 0.05, or when the risk rate exceeds 0.40.

### Phase 3 — Diagnostic Jury

Three specialist agents run in parallel. Each analyzes the failure from a different domain of expertise.

**Agent 1 — AdversarialSpecialist**

Detects prompt injection, jailbreak attempts, instruction overrides, and token smuggling.

Three detection layers:
1. Regex pattern matching across 4 attack categories
2. Guard scoring via `score_prompt_attack()`
3. FAISS semantic search against a pre-indexed vector database of known attack patterns

Root causes it can assign: `PROMPT_INJECTION`, `JAILBREAK_ATTEMPT`, `TOKEN_SMUGGLING`, `INSTRUCTION_OVERRIDE`, `INTENTIONAL_PROMPT_ATTACK`

Confidence range: 0.75–0.91

**Agent 2 — LinguisticAuditor**

Detects prompt complexity, ambiguity, and out-of-distribution (OOD) prompts.

Metrics computed:
- Complexity score: token count + sentence depth + vocabulary diversity
- Ambiguity detection: multiple valid interpretation paths
- OOD detection: prompt embedding vs known training distribution

Root causes it can assign: `PROMPT_COMPLEXITY_OOD`, `LINGUISTIC_AMBIGUITY`, `CONFLICTING_INSTRUCTIONS`

Confidence range: 0.60–0.80

**Agent 3 — DomainCritic**

Detects factual mismatches, knowledge boundary failures, and temporal drift.

Verification methods:
- Wikipedia retrieval + Groq-based fact checking
- Self-contradiction detection within the output
- Temporal signal detection (questions requiring real-time knowledge)

Root causes it can assign: `FACTUAL_HALLUCINATION`, `KNOWLEDGE_BOUNDARY_FAILURE`, `TEMPORAL_KNOWLEDGE_CUTOFF`

Confidence range: 0.06–0.92

**Jury aggregation rules**:
1. Run all three agents in parallel
2. Filter any skipped agents
3. `jury_confidence = mean(active_confidences)`
4. Priority: adversarial verdict > temporal verdict > highest-confidence verdict
5. Return `JuryVerdict` with primary diagnosis and all supporting verdicts

### Phase 4 — Auto-Fix Engine

FIE selects the correct mitigation strategy based on the jury's root cause diagnosis.

**Confidence gates**:
- ≥ 0.70 → Full fix applied
- 0.40–0.70 → Conservative fix applied
- < 0.40 → No fix, warning added

| Strategy | Trigger | Mechanism |
|---|---|---|
| `SHADOW_CONSENSUS` | Factual hallucination, knowledge boundary | Majority vote from shadow model outputs already collected |
| `SANITIZE_AND_RERUN` | Prompt injection, jailbreak, token smuggling | Regex-strip adversarial patterns, re-run cleaned prompt |
| `CONTEXT_INJECTION` | Temporal knowledge cutoff | Prepend current date + context, re-run |
| `PROMPT_DECOMPOSITION` | Prompt complexity OOD | Add chain-of-thought guidance, simplify prompt structure |
| `SELF_CONSISTENCY` | Fallback | Run same prompt 3× more, take majority vote |
| `RAG_GROUNDING` | Factual failure fallback | Wikipedia retrieval → Groq verification → substitute grounded answer |
| `NO_FIX` | Confidence < 0.40 | Return original output with warning flag |

---

## 8. All Algorithms and Logic Used

This section documents every algorithmic technique implemented in FIE.

### 8.1 Shannon Entropy (Normalized)

**File**: `engine/detector/entropy.py`

Measures divergence across grouped LLM outputs. Uses the same cluster counts as the consistency module to guarantee alignment.

```
H = -Σ(p_i × log₂(p_i))
H_normalized = H / log₂(total_samples)
```

A score near 1.0 means every output is in its own cluster — complete disagreement. A score near 0.0 means all outputs are in the same cluster — complete agreement.

### 8.2 Semantic Clustering and Agreement Scoring

**File**: `engine/detector/consistency.py`

1. Normalize outputs: strip LLM prefixes, lowercase
2. Cluster by semantic equivalence using two rules:
   - Rule 1: Keyword substring matching (handles "Paris" vs "The capital of France is Paris")
   - Rule 2: Cosine similarity on sentence-transformer embeddings (handles paraphrases)
3. `agreement_score = top_cluster_count / total_samples`
4. `fsd_score = (top_count - second_count) / total_samples`

Fallback: exact string matching when transformer is unavailable.

### 8.3 First-Second Dominance (FSD) Score

Measures how much the dominant cluster leads over the runner-up. A high FSD with low agreement means one model is consistently wrong. A low FSD means outputs are scattered equally — deeper uncertainty.

### 8.4 All-Pairwise Ensemble Disagreement

**File**: `engine/detector/ensemble.py`

1. Extract content tokens by removing stop words
2. Build term-frequency vectors per output
3. Compute cosine similarity for ALL pairs (not just primary vs reference)
4. `ensemble_disagreement = ANY(pair_similarity < 0.65)`
5. `ensemble_similarity = mean(all_pair_similarities)`

Guard: only fires when `entropy > 0.0` to prevent false positives on very short outputs.

### 8.5 Sentence-Transformer Embedding Distance

**File**: `engine/detector/embedding.py`

- Primary: `sentence-transformers/all-MiniLM-L6-v2` producing 384-dimensional L2-normalized vectors
- Cosine similarity = dot product of normalized vectors
- `embedding_distance = 1.0 - similarity`
- Fallback: character 3-gram similarity when transformer unavailable

### 8.6 Exponential Moving Average (EMA) Tracker

**File**: `engine/evolution/tracker.py`

```
EMA_new = α × new_value + (1 - α) × EMA_old    where α = 0.94
```

Tracks entropy, agreement, disagreement rate, and high-risk rate over a window of 100 inferences.

Degradation detection:
- `degradation_velocity = risk_rate(second_half) - risk_rate(first_half)`
- `is_degrading = velocity > 0.05  OR  risk_rate > 0.40`

### 8.7 Adaptive Threshold Clustering

**File**: `engine/archetypes/clustering.py`

```
threshold = base(0.80) + log(num_clusters + 1) × 0.03    capped at 0.92
```

Signal similarity metric is a weighted combination of:
- Agreement score difference between new signal and centroid
- Entropy difference
- Ensemble disagreement match

The threshold grows logarithmically to prevent over-splitting as the cluster count increases, while still allowing genuinely novel patterns to form their own clusters.

### 8.8 FAISS Semantic Vector Search

**File**: `engine/agents/adversarial_specialist.py`, `storage/faiss_adversarial.index`

A FAISS flat index pre-populated with sentence-transformer embeddings of known adversarial attack patterns. At inference time, the input prompt is embedded and its nearest neighbors are retrieved. If the nearest neighbor distance is below 0.82, the prompt is flagged as likely adversarial.

### 8.9 Regex Pattern Matching for Attack Detection

**File**: `engine/prompt_guard.py`

Pattern categories:
- Prompt injection (ignore previous instructions, system override, etc.)
- Jailbreak attempts (DAN, hypothetical framing, roleplay bypass)
- Token smuggling (hidden Unicode, zero-width characters, base64 instructions)
- Instruction override (new task, forget context, act as)

### 8.10 RAG Retrieval Pipeline

**File**: `engine/rag_grounder.py`, `engine/rag/retriever.py`

1. Query Wikipedia API for the most relevant article
2. Extract the top N sentences as context
3. Send to Groq with: context + original prompt + primary output
4. Groq determines: is the primary output factually consistent with the retrieved context?
5. If not: return the grounded Wikipedia-sourced answer with confidence 0.92
6. If yes but uncertain: return Wikipedia answer with confidence 0.72

### 8.11 N-Gram Character Similarity (Fallback)

**File**: `engine/detector/embedding.py`

When the sentence-transformer model is unavailable, FIE falls back to character-level 3-gram Jaccard similarity:
```
similarity = |ngrams(a) ∩ ngrams(b)| / |ngrams(a) ∪ ngrams(b)|
```

This ensures the system degrades gracefully without crashing when heavy ML dependencies fail to load.

---

## 9. Explainability Layer

FIE is built on the principle that every decision must be explainable. Every analysis produces a structured XAI artifact.

### Explanation Bundle Structure

```json
{
  "explanation_version": "v1",
  "mode": "monitor | diagnose",
  "request_id": "string",
  "final_label": "HALLUCINATION_RISK",
  "final_fix_strategy": "SHADOW_CONSENSUS",
  "explanation_confidence": 0.84,
  "summary": "Three shadow models disagreed on the answer to this factual question. High entropy (0.82) and ensemble disagreement indicate hallucination risk.",
  "decision_trace": [
    { "phase": "signal_extraction", "decision": "high_failure_risk=True", "confidence": 0.91 },
    { "phase": "archetype_labeling", "decision": "HALLUCINATION_RISK", "confidence": 0.85 },
    { "phase": "jury_verdict", "decision": "FACTUAL_HALLUCINATION", "confidence": 0.78 },
    { "phase": "fix_engine", "decision": "SHADOW_CONSENSUS applied", "confidence": 0.84 }
  ],
  "signals": [
    { "name": "entropy_score", "value": 0.82, "weight": 0.35, "interpretation": "High divergence" },
    { "name": "agreement_score", "value": 0.25, "weight": 0.30, "interpretation": "Only 1 of 4 models agreed" },
    { "name": "ensemble_disagreement", "value": true, "weight": 0.25, "interpretation": "Pairwise similarity below threshold" }
  ],
  "evidence": [
    { "source": "FAISS adversarial index", "finding": "No adversarial pattern detected", "confidence": 0.95 },
    { "source": "Wikipedia RAG", "finding": "Primary output contradicts retrieved context", "confidence": 0.88 }
  ],
  "attributions": [
    { "factor": "Shannon entropy", "contribution": 0.35 },
    { "factor": "Model disagreement", "contribution": 0.30 },
    { "factor": "RAG verification failure", "contribution": 0.25 }
  ],
  "alternatives_considered": ["OVERCONFIDENT_FAILURE", "MODEL_BLIND_SPOT"],
  "uncertainty_notes": ["DomainCritic confidence was borderline (0.62). Consider manual review."]
}
```

### Admin vs User Redaction

**File**: `engine/explainability/redaction.py`

- **User view** (`explanation_external`): decision trace, summary, signals, public attributions, fix strategy
- **Admin view** (`explanation_internal`): everything above plus agent verdicts, jury reasoning, low-confidence assessments, internal evidence

This ensures sensitive diagnostic internals are not exposed to end users, while admins retain full visibility.

### Human-Readable Summaries

**File**: `engine/explainability/humanizer.py`

The humanizer converts structured signals into natural language for the dashboard:

> "Your model produced unstable outputs across 4 shadow models (entropy=0.82). Three different answers were generated for the same question. Shadow consensus was applied and the corrected answer has been returned. Confidence in the fix: 84%."

---

## 10. Security and Multi-Tenancy

### Authentication Flow

1. React frontend initiates Google OAuth
2. User approves → Google returns authorization code
3. Frontend sends code to `POST /api/v1/auth/google-callback`
4. Backend exchanges code with Google, creates or loads user in MongoDB
5. Backend returns JWT session token + API key
6. All inferences are tagged with the user's `tenant_id`

### User Model

```json
{
  "email": "user@company.com",
  "name": "Jane Smith",
  "api_key": "fie-a3b9f2c1d8e4",
  "tenant_id": "janesmith-a3f9c2",
  "is_admin": false,
  "plan": "free",
  "calls_used": 142,
  "calls_limit": 1000
}
```

### Tenant Isolation

- Every inference record includes `tenant_id`
- Non-admin users query with an automatic `tenant_id` filter — they cannot see other users' data
- Admin users can view all data across all tenants
- API key is the SDK authentication credential — `X-API-Key` header

### API Key Format

API keys are generated as `fie-<16 cryptographically random characters>`. They can be regenerated at any time from the Settings page.

---

## 11. Dashboard and User Interface

The React/Vite dashboard is deployed on Vercel and provides six main views.

### Dashboard Page

KPI overview with:
- Current risk rate (rolling percentage of high-risk inferences)
- Average entropy score
- Average agreement score
- Recent activity feed
- Degradation status indicator (EMA-based)

### Analyze Page

Phase 1 and Phase 2 explorer:
- Submit a prompt and outputs directly
- View the Failure Signal Vector in detail
- See archetype label and cluster assignment
- Compare against historical signals

### Diagnose Page

Full Phase 3 and Phase 4 view:
- Enter prompt, primary output, and optionally shadow outputs
- View each agent's verdict with confidence score
- See the jury's final diagnosis
- View fix strategy selected and corrected output
- Render the full ExplanationPanel with decision trace, signals, evidence, attributions

### Alerts Page

- High-risk event log
- Degradation trend chart (EMA over time)
- Novel anomaly first-sighting log
- Filter by archetype, model name, date range

### Vault Page

- Full inference history with pagination
- Filter by model, archetype, risk level, date
- Per-record detail view with explanation bundle

### Settings Page

- Display current API key
- Regenerate API key
- SDK integration code snippet (pre-filled with the user's API key)
- Account information (plan, calls used/limit)

---

## 12. Python SDK — Integration in Minutes

The `fie-sdk` package is published on PyPI and provides the `@monitor` decorator for LLM functions.

### Installation

```bash
pip install fie-sdk
```

### The `@monitor` decorator

```python
from fie import monitor

@monitor(
    fie_url="https://your-fie-backend.com",
    api_key="fie-your-api-key",
    mode="correct",         # "monitor" for background, "correct" for real-time fix
)
def call_llm(prompt: str) -> str:
    return your_llm_client.generate(prompt)

# Your application code
answer = call_llm("Who invented the telephone?")
# If FIE detected and fixed a hallucination, answer contains the corrected response.
# If the output was clean, answer is the original response.
```

### Mode: `"monitor"` (zero-latency background mode)

```python
@monitor(
    fie_url="https://your-fie-backend.com",
    api_key="fie-your-api-key",
    mode="monitor",   # Returns immediately, sends to FIE in background thread
)
def call_llm(prompt: str) -> str:
    return your_llm_client.generate(prompt)
```

Use this for: production apps where latency is critical. FIE analyzes in background and alerts you via dashboard. The corrected answer is not returned synchronously.

### Mode: `"correct"` (real-time correction mode)

```python
@monitor(
    fie_url="https://your-fie-backend.com",
    api_key="fie-your-api-key",
    mode="correct",   # Waits for FIE analysis, returns corrected answer if available
)
def call_llm(prompt: str) -> str:
    return your_llm_client.generate(prompt)
```

Use this for: applications where output quality matters more than latency (legal research, medical Q&A, financial advice). Adds ~2–8 seconds.

### Direct API integration (no SDK)

```python
import requests

response = requests.post(
    "https://your-fie-backend.com/api/v1/monitor",
    headers={"X-API-Key": "fie-your-api-key"},
    json={
        "prompt": "Who invented the telephone?",
        "primary_output": "Thomas Edison invented the telephone.",
        "primary_model_name": "gpt-4o",
        "run_full_jury": True,
        "latency_ms": 842.1
    }
)

result = response.json()
print(result["fix_result"]["corrected_output"])  # "Alexander Graham Bell invented the telephone."
print(result["failure_signal_vector"]["entropy_score"])  # 0.91
print(result["jury"]["primary_verdict"]["root_cause"])  # "FACTUAL_HALLUCINATION"
```

---

## 13. API Reference

All endpoints live under `POST /api/v1` or `GET /api/v1`.

### Authentication

All protected endpoints accept either:
- `Authorization: Bearer <jwt_token>` (dashboard sessions)
- `X-API-Key: fie-<your-key>` (SDK and direct API access)

### Core Endpoints

#### `POST /api/v1/monitor`

The main production endpoint. Runs all four phases.

**Request body**:
```json
{
  "prompt": "string",
  "primary_output": "string",
  "primary_model_name": "string",
  "model_version": "string (optional)",
  "temperature": 0.7,
  "latency_ms": 842.1,
  "run_full_jury": true
}
```

**Response shape**:
```json
{
  "request_id": "string",
  "failure_signal_vector": {
    "agreement_score": 0.25,
    "fsd_score": 0.0,
    "entropy_score": 0.91,
    "ensemble_disagreement": true,
    "ensemble_similarity": 0.33,
    "high_failure_risk": true
  },
  "archetype": "HALLUCINATION_RISK",
  "embedding_distance": 0.42,
  "jury": {
    "primary_verdict": {
      "root_cause": "FACTUAL_HALLUCINATION",
      "confidence": 0.84,
      "summary": "Model output contradicts Wikipedia-verified facts"
    },
    "jury_confidence": 0.79,
    "is_adversarial": false
  },
  "fix_result": {
    "strategy": "SHADOW_CONSENSUS",
    "corrected_output": "Alexander Graham Bell invented the telephone.",
    "fix_confidence": 0.84,
    "fix_applied": true
  },
  "human_explanation": {
    "summary": "string",
    "why_risky": "string",
    "recommended_action": "string",
    "severity": "HIGH"
  },
  "explanation_external": { "...XAI bundle..." }
}
```

#### `POST /api/v1/analyze`

Phase 1 only — fast signal extraction.

#### `POST /api/v1/analyze/v2`

Phase 1 + Phase 2 — signal extraction plus clustering and trend summary.

#### `POST /api/v1/diagnose`

Phase 3 only — run the diagnostic jury on outputs you provide.

#### `GET /api/v1/inferences`

List stored inferences for the authenticated tenant. Supports pagination.

Query params: `page`, `limit`, `model_name`, `archetype`, `start_date`, `end_date`

#### `GET /api/v1/inferences/{request_id}`

Get a single stored inference by ID.

#### `GET /api/v1/trend`

Get EMA-based trend summary for the authenticated tenant.

```json
{
  "ema_entropy": 0.62,
  "ema_agreement": 0.71,
  "ema_disagreement_rate": 0.38,
  "ema_high_risk_rate": 0.29,
  "is_degrading": true,
  "degradation_velocity": 0.08
}
```

#### `GET /api/v1/clusters`

Summarize active failure archetype clusters.

### Authentication Endpoints

- `POST /api/v1/auth/google-callback` — OAuth callback, returns JWT + API key
- `GET /api/v1/auth/me` — Current user info
- `POST /api/v1/auth/regenerate-key` — Generate new API key

### Health Endpoints

- `GET /` — Service identity
- `GET /health` — Health check

---

## 14. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend framework | FastAPI + Uvicorn | High-performance async REST API |
| Database | MongoDB Atlas | Persistent storage, tenant isolation |
| Shadow models | Groq (llama-3.1-8b, llama-3.3-70b, llama-3.2-3b) | Parallel fan-out for ensemble comparison |
| Local model fallback | Ollama | Optional self-hosted shadow models |
| Semantic search | FAISS | Adversarial attack vector similarity search |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 384-dim semantic similarity |
| RAG retrieval | Wikipedia API | Factual grounding context |
| Authentication | Google OAuth 2.0 + JWT | Secure user sessions |
| Frontend | React + Vite | Dashboard UI |
| Frontend deployment | Vercel | Global CDN hosting |
| Backend deployment | Google Cloud Run | Serverless containerized API |
| Container build | Docker + Google Cloud Build | CI/CD pipeline |
| SDK | PyPI `fie-sdk` | Python integration package |

---

## 15. Deployment Architecture

### Backend — Google Cloud Run

```
Developer pushes code
        ↓
Cloud Build runs cloudbuild.yaml
        ↓
Docker image built (Python 3.11, FastAPI, all dependencies)
        ↓
Image pushed to Google Artifact Registry
        ↓
Cloud Run pulls new image, deploys with zero downtime
        ↓
Auto-scales from 0 to N instances based on traffic
        ↓
Environment variables: MongoDB URI, Groq key, OAuth credentials, JWT secret
```

**Port**: 8080 (Cloud Run standard)
**Health check**: `GET /health`
**Auto-scaling**: Managed by Cloud Run on request volume

### Frontend — Vercel

```
React/Vite app in fie-dashboard/
        ↓
npm run build → dist/
        ↓
Vercel deployment on push to main branch
        ↓
Global CDN with automatic HTTPS
        ↓
Environment: VITE_API_URL, VITE_GOOGLE_CLIENT_ID, VITE_REDIRECT_URI
```

### Database — MongoDB Atlas

- Collections: `inferences`, `users`
- Indices: `request_id` (unique), `timestamp`, `model_name`, `tenant_id`
- Free tier: 512 MB
- Accessed via MongoDB driver with Server API v1

---

## 16. How Any Company Can Use FIE

### Who benefits from FIE

| Company Type | Use Case | Failure FIE Prevents |
|---|---|---|
| **Legal tech** | Contract analysis, legal Q&A | Hallucinated case citations, wrong statutes |
| **Healthcare** | Medical Q&A, drug interaction lookup | Factual errors on dosage or contraindications |
| **Financial services** | Market data queries, financial advice | Stale data, wrong figures, outdated regulations |
| **Customer support** | AI-powered helpdesk | Wrong product info, policy hallucinations |
| **Education** | Tutoring, exam prep | Confidently wrong answers on factual content |
| **E-commerce** | Product recommendation, search | Hallucinated specs, wrong compatibility claims |
| **Cybersecurity** | AI-assisted threat analysis | Prompt injection attacks on internal LLM tools |
| **Any company** | Any customer-facing LLM feature | All of the above |

### Integration path

**Step 1 — Sign up and get your API key**
- Log in at the FIE dashboard with your Google account
- Copy your API key from the Settings page

**Step 2 — Install the SDK**
```bash
pip install fie-sdk
```

**Step 3 — Wrap your LLM function**
```python
from fie import monitor

@monitor(
    fie_url="https://fie-backend.your-domain.com",
    api_key="fie-your-api-key",
    mode="monitor",    # Start with "monitor" for zero-latency deployment
)
def ask_llm(question: str) -> str:
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    ).choices[0].message.content
```

**Step 4 — Monitor from the dashboard**

Open the dashboard, go to the Alerts page, and watch for high-risk events. Switch to `mode="correct"` when you are ready for automatic correction.

**Step 5 — Review and act**

- Use the Vault to browse all monitored inferences
- Use the Analyze page to investigate specific failures
- Use the Diagnose page for deep root cause analysis
- Check the Trend page to detect model degradation over time

---

## 17. Roadmap — What We Are Building Next

This roadmap is organized by priority: production blockers first, then competitive moat features, then future vision.

---

### Priority 1 — Production Blockers (Next Sprint)

#### P1.1 — Per-Tenant Rate Limiting and Quota Management

**What**: Token bucket / sliding window rate limiter per tenant. Per-user API budget controls. Hard limits to prevent Groq free tier exhaustion.

**Why**: Groq free tier is 14,400 requests/day — a single active customer can exhaust it. Without rate limiting, the service cannot scale to multiple tenants reliably.

**Files to add/modify**: `engine/groq_service.py`, `app/auth.py`, new `app/rate_limiter.py`

---

#### P1.2 — Async Job Queue for Phase 3 and Phase 4

**What**: Move the DiagnosticJury and Auto-Fix engine to a Celery + Redis async job queue. `/monitor` returns `request_id` immediately. Client polls `GET /inferences/{request_id}` or subscribes via webhook for the result.

**Why**: Phase 3 + Phase 4 currently add 1–8 seconds to the synchronous response. Most use cases cannot tolerate this latency inline. Async processing allows the system to handle 10× more concurrent requests.

**Files to add**: `worker/tasks.py`, `worker/celery_config.py`

---

#### P1.3 — Webhook Event Push System

**What**: Users register a callback URL via `POST /api/v1/webhooks`. When `high_failure_risk=True`, FIE posts a structured event payload to the registered URL.

**Why**: Dashboard polling is not real-time. Production systems need push notifications when a failure is detected — not a dashboard to check manually.

**Files to add**: `app/webhooks.py`, extend `app/routes.py`

---

#### P1.4 — Slack, PagerDuty, and Email Alerts

**What**: Configurable alert channels per user. Triggers on: degradation detected, novel cluster first seen, adversarial attack detected, confidence above threshold.

**Why**: Engineers and on-call teams are not watching the FIE dashboard. Alerts need to reach them where they already are.

**Files to add**: `app/alerting.py`, extend `app/auth.py` (user settings schema)

---

#### P1.5 — MongoDB Compound Indexing and TTL

**What**: Add compound index on `(tenant_id, timestamp)` and `(tenant_id, model_name)`. Add TTL index with configurable expiry per plan tier.

**Why**: As inference counts grow into millions, unindexed queries will slow the dashboard and Vault to a crawl. TTL prevents storage cost from growing unboundedly on free tiers.

**Files to modify**: `storage/database.py`

---

### Priority 2 — Core Feature Expansion

#### P2.1 — Ground Truth Feedback Loop

**What**: `POST /api/v1/feedback/{request_id}` endpoint. Users submit the correct answer for a given inference. FIE uses labeled data to: measure jury precision/recall, measure fix effectiveness, and eventually retrain archetype thresholds.

**Why**: This closes the evaluation loop. Right now FIE detects and fixes blindly. With ground truth, FIE can prove its correction accuracy — and this dataset becomes a defensible asset no competitor can replicate.

---

#### P2.2 — Fine-Tuned Specialist Agents

**What**: Replace rule-based heuristics in jury agents with fine-tuned lightweight models (e.g., Llama-3.2-3B). Start with AdversarialSpecialist (clearest labeled training data available from security datasets).

**Why**: Static confidence scores (0.75–0.91) are not calibrated. Fine-tuned agents produce dynamic, calibrated confidence scores that actually reflect the probability of a correct diagnosis.

**Files to add**: `engine/agents/training/`

---

#### P2.3 — Streaming Monitor Endpoint (Server-Sent Events)

**What**: `POST /api/v1/monitor/stream` returns SSE. Dashboard shows live progress: Phase 1 complete → Phase 2 complete → Phase 3 complete → Fix applied.

**Why**: Users waiting 5–8 seconds for a result see nothing. Streaming gives immediate feedback and makes the pipeline feel fast even when it is not.

---

#### P2.4 — Extended RAG Sources

**What**: Pluggable `RAGSource` interface with implementations for: Wikipedia (current), arXiv (research papers), Bing/Serper (real-time web), user-provided document stores (PDF, URL scraping).

**Why**: Wikipedia cannot answer domain-specific or recent questions. Enterprise customers need RAG grounding against their own knowledge bases.

**Files to modify**: `engine/rag_grounder.py`, `engine/rag/retriever.py`

---

#### P2.5 — Model Degradation Alerts

**What**: When `is_degrading=True` persists for N consecutive inferences (configurable), trigger an automatic alert. Add degradation timeline view to dashboard. Support comparison of metrics before and after model version changes.

**Why**: EMA degradation is already computed but only visible via the `/trend` endpoint. Proactive alerting turns passive monitoring into active reliability management.

**Files to modify**: `engine/evolution/tracker.py`, `app/alerting.py`

---

#### P2.6 — Multi-Model Comparison Dashboard

**What**: Track all metrics per `model_name` + `model_version` combination. New `/compare` endpoint. Dashboard view to compare GPT-4o vs Claude vs Llama on the same prompts side-by-side.

**Why**: The most common enterprise question is "which model should I use?" FIE already collects the data to answer this empirically. Exposing it as a feature makes FIE a model selection tool in addition to a reliability tool.

---

#### P2.7 — OpenTelemetry Integration

**What**: Add OTel trace exporter. FIE emits spans compatible with Datadog, Grafana, Jaeger, and any OTel-compatible backend.

**Why**: Enterprise buyers already have observability stacks. "Another tool to integrate" is an objection. Making FIE emit OTel spans removes that objection entirely.

**Files to add**: `app/otel_exporter.py`

---

### Priority 3 — Competitive Moat

#### P3.1 — Failure Pattern Knowledge Base (Public Dataset)

**What**: Anonymize and publish aggregated cluster statistics as a monthly public report — "The FIE Failure Index." What percentage of LLM inferences are hallucinations? What percentage are adversarial? Which models fail most on factual questions?

**Why**: No competitor has this. It creates SEO, thought leadership, academic citations, and a network effect — more users generate richer failure pattern data which makes the platform more valuable.

---

#### P3.2 — Gaussian Mixture Model Clustering

**What**: Replace the current centroid-based adaptive clustering with GMM or hierarchical agglomerative clustering.

**Why**: GMM provides soft cluster membership (a signal can belong to multiple clusters with different probabilities), better novelty detection, and explainable cluster drift — all critical for production reliability analysis.

**Files to modify**: `engine/archetypes/clustering.py`

---

#### P3.3 — User-Defined Custom Archetypes

**What**: Enterprise users define custom failure archetypes via dashboard. Example: a fintech user defines "COMPLIANCE_VIOLATION" triggered by specific keywords + confidence thresholds. Stored per tenant in MongoDB.

**Why**: The 7 built-in archetypes cover general LLM failures. Enterprise customers in regulated industries need domain-specific failure taxonomy. This is a major enterprise differentiator.

**Files to modify**: `engine/archetypes/labeling.py`

---

#### P3.4 — Fix Effectiveness Scoring

**What**: After applying a fix, measure: did corrected output have lower entropy? Higher shadow agreement? Store `fix_effectiveness_score` per inference. Aggregate into `fix_engine_accuracy` metric per strategy type.

**Why**: FIE claims to fix failures — this is the proof. Fix effectiveness data becomes the primary selling point for enterprise buyers and the foundation for calibrating the auto-fix engine.

---

#### P3.5 — Native LangChain, LlamaIndex, and OpenAI SDK Integrations

**What**:
- LangChain callback handler: `FIECallbackHandler()`
- LlamaIndex observer: `FIEObserver()`
- OpenAI proxy: transparent HTTP proxy that intercepts OpenAI API calls

**Why**: The `@monitor` decorator is great for custom Python code. But 80% of enterprise LLM apps use LangChain or LlamaIndex. Native integrations reduce time-to-integration from 30 minutes to 5 lines.

---

#### P3.6 — SOC2 Type II Compliance

**What**: Audit logging (who accessed what, when), data residency options (EU deployment), documented data retention policy, encryption at rest verification, penetration test of adversarial detection components.

**Why**: Any enterprise contract above $50K/year requires SOC2. Without it, FIE cannot be seriously evaluated by large companies.

---

### Priority 4 — Future Vision

#### P4.1 — Predictive Failure Scoring (Pre-Flight)

**What**: Before calling the LLM, FIE scores the prompt alone for predicted failure probability using a lightweight binary classifier trained on prompt embeddings + complexity scores.

**Why**: Instead of detecting failures after they happen, FIE warns you that a prompt is likely to fail — before you spend API tokens. This is pre-emptive correction, an industry first.

---

#### P4.2 — Federated Failure Learning

**What**: Multiple enterprise customers contribute to shared failure pattern knowledge without sharing raw data. Federated averaging over cluster centroids.

**Why**: The cluster registry is already the right data structure for this. Federated learning makes the platform more valuable as it grows (more customers = better shared failure patterns) while maintaining strict data privacy — a critical concern for enterprise.

---

#### P4.3 — Automated Threshold Calibration

**What**: Given user-labeled ground truth (from P2.1), a calibration engine finds optimal per-tenant thresholds using Bayesian optimization. A fintech customer needs different entropy thresholds than a creative writing app.

**Why**: Current thresholds are universal defaults. Per-tenant calibration dramatically reduces false positives and improves fix precision for domain-specific use cases.

---

#### P4.4 — Mobile SDK (iOS and Android)

**What**: Swift and Kotlin SDKs wrapping the existing `/monitor` API. Simple integration for mobile AI applications.

**Why**: Mobile LLM apps are a rapidly growing market. The backend API is already mobile-compatible. SDKs remove the integration barrier.

---

## 18. Full Updated Pipeline After Roadmap

After all Priority 1 and Priority 2 features are implemented, the complete FIE pipeline will look like this:

```
┌────────────────────────────────────────────────────────────────────┐
│                        USER APPLICATION                             │
│  Python @monitor / LangChain handler / LlamaIndex / OpenAI proxy  │
│  iOS SDK / Android SDK / Direct HTTP                               │
└────────────────────────────┬───────────────────────────────────────┘
                             │ prompt + primary_output
                             ▼
              ┌──────────────────────────────┐
              │   PRE-FLIGHT RISK SCORING    │  (Roadmap P4.1)
              │   Predict failure from       │
              │   prompt alone — before LLM  │
              │   Returns: predicted_risk     │
              └──────────────┬───────────────┘
                             │
              ┌──────────────────────────────┐
              │   RATE LIMITER / QUOTA       │  (Roadmap P1.1)
              │   Per-tenant token bucket    │
              │   Groq budget enforcement    │
              └──────────────┬───────────────┘
                             │
       ┌─────────────────────┼─────────────────────────┐
       │                     │                         │
       ▼                     ▼                         ▼
┌─────────────┐     ┌─────────────────┐      ┌─────────────────┐
│ PRIMARY LLM │     │  SHADOW MODELS  │      │  EXTENDED RAG   │  (P2.4)
│ (user's     │     │  Groq parallel  │      │  Wikipedia      │
│  model)     │     │  llama-3.1-8b   │      │  arXiv          │
│             │     │  llama-3.3-70b  │      │  Bing/Serper    │
│             │     │  llama-3.2-3b   │      │  User Docs      │
└──────┬──────┘     └────────┬────────┘      └───────┬─────────┘
       │                     │                       │
       └─────────────────────┼───────────────────────┘
                             │
            ┌────────────────▼──────────────────────┐
            │         PHASE 1: FSV BUILDER           │
            │  Shannon Entropy                       │
            │  Semantic Clustering + Agreement Score │
            │  FSD Score                             │
            │  All-Pairwise Ensemble Disagreement    │
            │  Sentence-Transformer Embedding Dist.  │
            │  → high_failure_risk composite flag    │
            └────────────────┬──────────────────────┘
                             │ Returns request_id immediately (P1.2)
            ┌────────────────▼──────────────────────┐
            │         PHASE 2: ARCHETYPE ENGINE      │
            │  Rule-Based Labeling (7 archetypes)    │
            │  + Custom Tenant Archetypes   (P3.3)   │
            │  GMM / Hierarchical Clustering (P3.2)  │
            │  EMA Degradation Tracker               │
            │  → Degradation Alert trigger  (P2.5)   │
            └────────────────┬──────────────────────┘
                             │
            ┌──── ASYNC JOB QUEUE (Celery) ─────────┐  (P1.2)
            │                                        │
            ▼              ▼                         ▼
 ┌───────────────┐ ┌───────────────┐       ┌──────────────────┐
 │ ADVERSARIAL   │ │  LINGUISTIC   │       │  DOMAIN CRITIC   │
 │ SPECIALIST    │ │  AUDITOR      │       │  Fine-tuned (P2.2)│
 │ 3-layer FAISS │ │  Complexity   │       │  RAG-verified    │
 │ + regex +     │ │  Ambiguity    │       │  Factual check   │
 │ guard score   │ │  OOD detect   │       │  Temporal detect │
 │ Fine-tuned    │ │  Fine-tuned   │       │                  │
 │ (P2.2)       │ │  (P2.2)      │       │                  │
 └───────┬───────┘ └───────┬───────┘       └──────────┬───────┘
         │                 │                          │
         └─────────────────┼──────────────────────────┘
                           │
         ┌─────────────────▼────────────────────────────┐
         │           DIAGNOSTIC JURY (Phase 3)           │
         │  Priority: adversarial > temporal > highest   │
         │  jury_confidence = mean(active_confidences)   │
         │  Calibrated thresholds per tenant   (P4.3)    │
         └─────────────────┬────────────────────────────┘
                           │
         ┌─────────────────▼────────────────────────────┐
         │           PHASE 4: AUTO-FIX ENGINE            │
         │  SHADOW_CONSENSUS     → factual               │
         │  SANITIZE_AND_RERUN   → adversarial           │
         │  CONTEXT_INJECTION    → temporal              │
         │  PROMPT_DECOMPOSITION → complexity            │
         │  SELF_CONSISTENCY     → fallback              │
         │  RAG_GROUNDING        → factual fallback      │
         │                                              │
         │  Fix Effectiveness Scoring         (P3.4)    │
         └─────────────────┬────────────────────────────┘
                           │
         ┌─────────────────▼────────────────────────────┐
         │           EXPLAINABILITY LAYER                │
         │  decision_trace, signals, evidence           │
         │  attributions, alternatives, uncertainty     │
         │  Admin vs User redaction                     │
         │  OTel span export              (P2.7)        │
         └─────────────────┬────────────────────────────┘
                           │
         ┌─────────────────▼────────────────────────────┐
         │           PERSISTENCE (MongoDB)               │
         │  Tenant-isolated inference record            │
         │  Compound index: (tenant_id, timestamp)      │
         │  TTL auto-expiry by plan tier    (P1.5)      │
         │  Ground truth feedback stored    (P2.1)      │
         └─────────────────┬────────────────────────────┘
                           │
          ┌────────────────┼────────────────────┐
          ▼                ▼                    ▼
 ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐
 │   WEBHOOKS     │ │    ALERTS      │ │   REACT DASHBOARD  │
 │   (P1.3)      │ │  Slack / PD /  │ │  SSE streaming     │
 │  Push on       │ │  Email         │ │  (P2.3)            │
 │  high_risk     │ │  (P1.4)       │ │  Model comparison  │
 │  to user URL   │ │  Degradation   │ │  (P2.6)            │
 └────────────────┘ │  Novel cluster │ │  Ground truth UI   │
                    │  Adversarial   │ │  (P2.1)            │
                    └────────────────┘ └────────────────────┘
                           │
         ┌─────────────────▼────────────────────────────┐
         │       CALIBRATION ENGINE        (P4.3)        │
         │  Bayesian optimization of thresholds         │
         │  Per-tenant threshold profiles               │
         │  Trained on ground truth labels              │
         └──────────────────────────────────────────────┘
```

**Return to application**: `corrected_output` (or original if confidence < 0.40) + `request_id` for async result polling.

---

## 19. Local Development Setup

### Requirements

- Python 3.11+
- Node.js 18+
- MongoDB Atlas connection string (free tier works)
- Groq API key (free at console.groq.com)
- Google OAuth credentials (Google Cloud Console)

### 1. Clone and install backend

```bash
git clone https://github.com/AyushSingh110/Failure_Intelligence_System.git
cd Failure_Intelligence_System
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure backend environment

Create `.env` in the project root:

```env
MONGODB_URI=your_mongodb_atlas_uri
MONGODB_DB_NAME=fie_database

GROQ_API_KEY=your_groq_api_key
GROQ_ENABLED=true

OLLAMA_ENABLED=false
OLLAMA_BASE_URL=http://localhost:11434

GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:5173

JWT_SECRET_KEY=replace-with-a-long-random-secret
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24
ADMIN_EMAIL=your-admin-email@example.com
```

### 3. Start the backend

```bash
uvicorn app.main:app --reload
# Backend: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 4. Install and start the frontend

```bash
cd fie-dashboard
npm install
```

Create `fie-dashboard/.env`:

```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
VITE_REDIRECT_URI=http://localhost:5173
```

```bash
npm run dev
# Frontend: http://localhost:5173
```

### 5. Run tests

```bash
pytest                              # All tests
pytest tests/test_explainability.py # XAI tests
pytest tests/test_monitor_rag_fix.py # RAG correction tests
pytest tests/test_auth_visibility.py # Tenant isolation tests
```

---

## 20. Project Structure

```
Failure_Intelligence_System/
│
├── app/                          FastAPI application layer
│   ├── main.py                  Entry point, lifespan, CORS, startup
│   ├── routes.py                Core endpoints (all phases)
│   ├── auth.py                  Authentication, user model, JWT, API keys
│   ├── auth_routes.py           OAuth callback, user management
│   ├── auth_guard.py            Middleware: JWT and API key validation
│   └── schemas.py               Pydantic request/response models
│
├── engine/                       Core intelligence engine
│   ├── detector/                Phase 1 — Signal extraction
│   │   ├── consistency.py       Semantic clustering, agreement score, FSD
│   │   ├── entropy.py           Shannon entropy computation
│   │   ├── ensemble.py          All-pairwise disagreement detection
│   │   └── embedding.py         Sentence-transformer semantic distance
│   │
│   ├── archetypes/              Phase 2 — Clustering and labeling
│   │   ├── labeling.py          7-archetype rule-based labeling
│   │   ├── clustering.py        Adaptive cluster registry
│   │   ├── similarity.py        Signal similarity metrics
│   │   └── registry.py          Failure pattern registry
│   │
│   ├── agents/                  Phase 3 — DiagnosticJury
│   │   ├── failure_agent.py     Jury orchestrator and aggregation
│   │   ├── base_agent.py        Abstract agent interface
│   │   ├── adversarial_specialist.py  Injection/jailbreak/smuggling
│   │   ├── linguistic_auditor.py      Complexity/ambiguity/OOD
│   │   └── domain_critic.py           Factual/boundary/temporal
│   │
│   ├── evolution/               Trend monitoring
│   │   └── tracker.py           EMA-based degradation detection
│   │
│   ├── explainability/          XAI layer
│   │   ├── explanation_builder.py  Structured bundle assembly
│   │   ├── humanizer.py            Natural language summary
│   │   └── redaction.py            Admin vs user-safe filtering
│   │
│   ├── encoder.py               Lazy-loaded SentenceTransformer
│   ├── fix_engine.py            Phase 4 — Auto-fix strategy selection
│   ├── groq_service.py          Parallel Groq fan-out
│   ├── rag_grounder.py          Wikipedia retrieval + Groq verification
│   └── prompt_guard.py          Pattern-based attack detection
│
├── storage/                     Persistence layer
│   ├── database.py              MongoDB Atlas integration
│   ├── vault.json               Local fallback storage
│   └── faiss_adversarial.index  Pre-indexed adversarial attack vectors
│
├── fie/                         Python SDK (published as fie-sdk on PyPI)
│   ├── __init__.py              Exports @monitor decorator
│   ├── monitor.py               Decorator logic, both modes
│   ├── client.py                HTTP client to FIE backend
│   └── config.py                SDK configuration
│
├── fie-dashboard/               React frontend
│   └── src/
│       ├── pages/               LoginPage, DashboardPage, AnalyzePage,
│       │                        DiagnosePage, AlertsPage, VaultPage, SettingsPage
│       ├── components/          Layout, Sidebar, ExplanationPanel
│       ├── lib/                 api.js (HTTP client), auth.js (JWT)
│       └── hooks/               useData.js
│
├── tests/                       Test suite
├── config.py                    Global thresholds and settings
├── requirements.txt             Backend dependencies
├── pyproject.toml               SDK packaging metadata
├── Dockerfile                   Backend container image
├── cloudbuild.yaml              Google Cloud Build pipeline
└── README.md                    Quick-start documentation
```

---

## License

MIT — Free to use, modify, and distribute.

---

*Failure Intelligence Engine — Built to make LLMs reliable by default, not by accident.*
