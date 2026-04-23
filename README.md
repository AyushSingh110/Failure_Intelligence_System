# Failure Intelligence Engine (FIE)

**Real-time LLM failure detection, root cause diagnosis, and automatic correction.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)](https://mongodb.com/atlas)
[![Groq](https://img.shields.io/badge/Groq-Shadow%20Models-orange)](https://groq.com)
[![React](https://img.shields.io/badge/React-Dashboard-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![PyPI](https://img.shields.io/badge/PyPI-fie--sdk-blue?logo=pypi&logoColor=white)](https://pypi.org/project/fie-sdk)

FIE sits between your LLM and your users. When the model gives a wrong answer, FIE catches it, finds the correct answer from a trusted source, and returns the correction — before the user ever sees the mistake.

---

## What It Does

LLMs hallucinate. They say "Thomas Edison invented the telephone" with the same confidence as correct answers. There is no built-in signal. The wrong answer simply goes out to the user.

FIE solves this in real time:

1. **Detect** — runs the same prompt through 3 independent shadow models, computes an ensemble signal
2. **Diagnose** — a jury of 3 specialist agents votes on the root cause (hallucination, injection, temporal cutoff, etc.)
3. **Verify** — queries Wikidata or Google Search to find the correct answer
4. **Correct** — returns the verified answer to the user instead of the wrong one

The integration is one decorator:

```python
from fie import monitor

@monitor(fie_url="http://localhost:8000", api_key="fie-xxx", mode="correct")
def ask_ai(prompt: str) -> str:
    return your_llm(prompt)
```

---

## How It Works — Pipeline Overview

```
User Prompt
     │
     ▼
Your LLM  →  primary answer  →  FIE
                                 │
                     ┌───────────┴────────────┐
                     │                        │
              Phase 1: Shadow Ensemble    Phase 2: FSV
              3 models answer in         Agreement, entropy,
              parallel (Groq)            outlier detection
                     │
                     ▼
              Phase 3: Diagnostic Jury
              3 agents vote on root cause
              (AdversarialSpecialist, LinguisticAuditor, DomainCritic)
                     │
                     ▼
              Phase 4: Ground Truth Pipeline
              Cache → Wikidata → Serper → Shadow consensus
                     │
                     ▼
              Phase 5: Fix Engine
              Return corrected answer to user
```

---

## Shadow Model Ensemble

Three shadow models from different families run in parallel on every query:

| Model | Provider | Why |
|---|---|---|
| `llama-3.3-70b-versatile` | Meta | Strong general knowledge |
| `deepseek-r1-distill-llama-70b` | DeepSeek | Reasoning-focused, different RLHF |
| `qwen-qwq-32b` | Alibaba | Different pretraining corpus |

Different families reduce correlated failure — if one model is wrong, the others are unlikely to make the same mistake.

Each shadow model self-reports its certainty. FIE weights votes by confidence:

| Model reports | Vote weight |
|---|---|
| CONFIDENCE: HIGH | 3.0 |
| CONFIDENCE: MEDIUM | 2.0 |
| CONFIDENCE: LOW | 1.0 |

---

## Failure Signal Vector (FSV)

After collecting all 4 answers (1 primary + 3 shadows), FIE computes:

| Signal | What it measures |
|---|---|
| `agreement_score` | Fraction of models that gave the same answer |
| `fsd_score` | Gap between the top-2 answer clusters |
| `entropy_score` | Normalized Shannon entropy of the answer distribution |
| `ensemble_disagreement` | Embedding-based pairwise disagreement flag |
| `ensemble_similarity` | Cosine similarity between primary and secondary |
| `high_failure_risk` | Final risk flag |

### Shannon Entropy

```
H = -Σ p(x) × log₂(p(x))
H_normalized = H / log₂(total_outputs)
```

- All 4 models agree → entropy = 0.0 (no uncertainty)
- All 4 models differ → entropy = 1.0 (maximum uncertainty)
- 3 agree, 1 differs → entropy ≈ 0.41

Entropy is used alongside agreement because a 2-vs-2 split (entropy=1.0) is far more alarming than a 3-vs-1 split (entropy=0.41), even though both have low agreement.

### Primary-Outlier Detection — POET Algorithm

`high_failure_risk` is set by **POET (Primary Outlier Ensemble Test)** — the core novel algorithm in FIE. It does not check overall ensemble agreement. It specifically checks whether the **primary model** is the one disagreeing with the shadow majority:

```
shadow_agreement = agreement among shadows only (primary excluded)
if shadow_agreement < 0.60 → can't blame primary (shadows confused)
else:
    majority = most common shadow answer cluster
    if primary semantically matches majority → NOT an outlier
    if primary is far from majority (cosine sim < 0.72) → IS an outlier → high_failure_risk = True
```

This dropped the false positive rate from 80% to 20% compared to threshold-based ensemble agreement.

---

## Archetype Classification

7 failure archetypes based on the FSV:

| Archetype | When it fires |
|---|---|
| `HALLUCINATION_RISK` | Ensemble disagrees AND high entropy |
| `OVERCONFIDENT_FAILURE` | High risk but very low entropy (confident but wrong) |
| `MODEL_BLIND_SPOT` | Systematic knowledge gap in a domain |
| `UNSTABLE_OUTPUT` | High entropy alone (genuine ambiguity) |
| `LOW_CONFIDENCE` | Low agreement without high entropy |
| `RESOURCE_CONSTRAINT` | High latency AND high entropy |
| `STABLE` | All signals within normal range |

---

## Diagnostic Jury

Three agents independently analyze the failure and vote on the root cause:

### AdversarialSpecialist — `engine/agents/adversarial_specialist.py`

Detects intentional attacks using 3 detection layers:

- **Regex** — patterns for PROMPT_INJECTION, JAILBREAK_ATTEMPT, INSTRUCTION_OVERRIDE, TOKEN_SMUGGLING
- **Prompt Guard** — statistical heuristic scorer
- **FAISS semantic search** — finds novel attacks similar to known attack vectors

Priority: TOKEN_SMUGGLING > PROMPT_INJECTION > JAILBREAK > OVERRIDE

### LinguisticAuditor — `engine/agents/linguistic_auditor.py`

Detects structural response problems: excessive hedging, truncation, format inconsistency, length anomalies, repetition loops.

### DomainCritic — `engine/agents/domain_critic.py`

Detects factual and temporal failures using 5 weighted layers:

| Layer | Weight | What it checks |
|---|---|---|
| Contradiction signal | 0.40 | FSV entropy + agreement vs thresholds |
| Self-contradiction | 0.35 | Cosine similarity between primary and secondary |
| Hedge detection | 0.15 | Uncertainty phrases in model outputs |
| Temporal detection | 0.10 | Time-sensitive keywords in prompt |
| External verification | 0.45 | Wikipedia/RAG fact check |

**Permanent fact guard:** Chemical formulas, math identities, and physical constants are never routed to temporal (Serper) verification — they are verified via Wikidata only.

### Jury Aggregation

```
Priority 1: Adversarial verdict (if any agent detected an attack)
Priority 2: Temporal verdict (routes to live search)
Default:    Highest confidence verdict wins
```

---

## Ground Truth Pipeline

Runs only when both gates pass:
- **Gate 1:** `high_failure_risk = True`
- **Gate 2:** `jury_confidence >= 0.45`

Pipeline steps:

```
1. Cache lookup (MongoDB ground_truth_cache)
   → HIT: return verified answer immediately

2. Permanent fact check
   → chemical formula / math / physics constant → Wikidata only (no Serper)

3. Temporal routing
   → root_cause = TEMPORAL_KNOWLEDGE_CUTOFF → Serper (Google Search)
   → all other root causes → Wikidata

4. Wikidata (SPARQL)
   → Extract claim: subject / property / value
   → Search Wikidata with enriched query
   → contradiction + confidence ≥ 0.75 → OVERRIDE
   → confirmation + confidence ≥ 0.60 → CONFIRM

5. Serper (Google Search)
   → contradicts primary → OVERRIDE with search answer
   → confirms primary → CONFIRM

6. Shadow consensus fallback
   → weighted shadow agreement ≥ 0.60 → use majority shadow answer
   → below 0.60 → ESCALATE to human review

7. Write-through cache
   → verified answer with confidence ≥ 0.90 → saved to cache
```

---

## Fix Strategies

| Strategy | When used |
|---|---|
| `WIKIDATA_OVERRIDE` | Wikidata contradicts the primary answer |
| `SERPER_OVERRIDE` | Google Search contradicts the primary answer |
| `SHADOW_CONSENSUS` | External sources exhausted, shadows agree |
| `SANITIZE_AND_RERUN` | Adversarial attack detected |
| `CONTEXT_INJECTION` | Temporal failure, search result available |
| `PROMPT_DECOMPOSITION` | Question too complex |
| `HUMAN_ESCALATION` | No reliable ground truth, shadow consensus too weak |
| `NO_FIX` | Output is stable |

---

## Root Causes

| Root Cause | Meaning |
|---|---|
| `FACTUAL_HALLUCINATION` | Model stated a wrong fact |
| `TEMPORAL_KNOWLEDGE_CUTOFF` | Model's training data is outdated |
| `KNOWLEDGE_BOUNDARY_FAILURE` | Model uncertain at edge of training data |
| `PROMPT_INJECTION` | User attempting to override system prompt |
| `JAILBREAK_ATTEMPT` | User attempting to bypass safety guidelines |
| `INSTRUCTION_OVERRIDE` | User claiming fake authority |
| `TOKEN_SMUGGLING` | Special model tokens embedded in user input |
| `PROMPT_COMPLEXITY_OOD` | Question out-of-distribution / too complex |

---

## Signal Logging

Every inference is logged to MongoDB `signal_logs` with 30+ fields including agreement, entropy, archetype, root cause, jury confidence, GT source, fix applied, and latency. Human feedback can be submitted via `POST /api/v1/feedback/{request_id}` to label signal logs as correct or incorrect — building a labeled dataset for future classifier training.

---

## SDK Modes

```python
# mode="monitor" — async, no latency added
# FIE checks in background, original answer returned immediately
@monitor(fie_url="...", api_key="...", mode="monitor")
def ask_ai(prompt: str) -> str:
    return your_llm(prompt)

# mode="correct" — synchronous, real-time correction
# FIE verifies and returns corrected answer if wrong
@monitor(fie_url="...", api_key="...", mode="correct")
def ask_ai(prompt: str) -> str:
    return your_llm(prompt)
```

---

## API Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/api/v1/monitor` | Main production endpoint — full pipeline |
| POST | `/api/v1/diagnose` | Run diagnostic jury on provided outputs |
| POST | `/api/v1/analyze` | Phase 1 signal extraction only |
| POST | `/api/v1/feedback/{id}` | Submit human feedback on an inference |
| GET | `/api/v1/inferences` | List stored inferences |
| GET | `/api/v1/trend` | EMA-based degradation trend |
| GET | `/api/v1/clusters` | Archetype cluster summary |
| GET | `/monitor/signal-logs` | Raw signal logs |
| GET | `/monitor/calibration` | Per-confidence-bucket accuracy stats |
| GET | `/health` | Server health check |

---

## File Structure

```
Failure_Intelligence_System/
│
├── app/
│   ├── main.py                    FastAPI app entry point
│   ├── routes.py                  All API endpoints
│   ├── schemas.py                 Pydantic schemas (FSV, JuryVerdict, FixResult)
│   ├── auth.py / auth_guard.py    API key authentication + tenant isolation
│   └── auth_routes.py             Google OAuth routes
│
├── engine/
│   ├── groq_service.py            Shadow model fan-out + confidence weighting
│   ├── encoder.py                 SentenceTransformer singleton (all-MiniLM-L6-v2)
│   ├── fix_engine.py              Fix strategy selection and execution
│   ├── claim_extractor.py         Extract subject/property/value from model output
│   ├── prompt_guard.py            Statistical adversarial prompt scorer
│   ├── rag_grounder.py            Wikipedia RAG for external verification
│   │
│   ├── detector/
│   │   ├── consistency.py         compute_consistency(), is_primary_outlier()
│   │   ├── entropy.py             Shannon entropy computation
│   │   ├── ensemble.py            Pairwise embedding disagreement
│   │   └── embedding.py           compute_embedding_distance()
│   │
│   ├── archetypes/
│   │   ├── labeling.py            7-archetype classification rules
│   │   ├── clustering.py          Adaptive archetype cluster registry
│   │   └── registry.py            FAISS index for adversarial pattern search
│   │
│   ├── agents/
│   │   ├── base_agent.py          BaseJuryAgent, DiagnosticContext
│   │   ├── failure_agent.py       DiagnosticJury + FailureAgent singletons
│   │   ├── adversarial_specialist.py  3-layer adversarial attack detection
│   │   ├── domain_critic.py       5-layer factual/temporal failure detection
│   │   └── linguistic_auditor.py  Response structure and quality analysis
│   │
│   ├── verifier/
│   │   ├── ground_truth_pipeline.py  GT pipeline orchestrator
│   │   ├── wikidata_verifier.py      SPARQL queries against Wikidata
│   │   └── serper_verifier.py        Google Search via Serper.dev
│   │
│   ├── evolution/
│   │   └── tracker.py             EMA-based model degradation tracking
│   │
│   └── explainability/
│       └── explanation_builder.py Human-readable XAI explanation builder
│
├── fie/                           Python SDK (pip install fie-sdk)
│   ├── monitor.py                 @monitor decorator
│   ├── client.py                  HTTP client for FIE server
│   └── config.py                  FIEConfig
│
├── storage/
│   ├── database.py                MongoDB connection + inference CRUD
│   ├── signal_logger.py           30-field signal logging + feedback wiring
│   └── ground_truth_cache.py      Verified answer cache (write-through)
│
├── Frontend/                      React dashboard (Vite)
│
├── data/
│   ├── download_datasets.py       TruthfulQA download (817 examples)
│   └── synthetic_generator.py     Synthetic failure data generator
│
├── config.py                      Settings (thresholds, model names, flags)
├── test_local.py                  Group A/B recall + FPR benchmark test
├── test_ground_truth.py           Ground truth pipeline isolation test
├── demo.py                        Interactive demo (chatbot with FIE)
└── FIE_COMPLETE_TECHNICAL_STORY.md  Full technical documentation
```

---

## Local Setup

### Requirements

- Python 3.11+
- MongoDB Atlas URI
- Groq API key (free at console.groq.com)
- Node.js 18+ (for dashboard only)

### 1. Backend

```bash
git clone https://github.com/AyushSingh110/Failure_Intelligence_System.git
cd Failure_Intelligence_System
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Environment

Create `.env` in the project root:

```env
MONGODB_URI=your_mongodb_atlas_uri
MONGODB_DB_NAME=fie_database

GROQ_API_KEY=gsk_your_groq_key
GROQ_ENABLED=true

WIKIDATA_ENABLED=true
GROUND_TRUTH_CACHE_ENABLED=true

# Optional — needed for temporal question verification
SERPER_API_KEY=your_serper_key
SERPER_ENABLED=true

OLLAMA_ENABLED=false

GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:5173

JWT_SECRET_KEY=replace-with-a-long-random-secret
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24
ADMIN_EMAIL=your-admin-email@example.com
```

### 3. Start Server

```bash
uvicorn app.main:app --reload
# Server: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 4. Dashboard (optional)

```bash
cd Frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
```

### 5. Run Demo

```bash
python demo.py
```

### 6. Run Tests

```bash
# Full recall + FPR benchmark
python test_local.py

# Ground truth pipeline isolation
python test_ground_truth.py

# Backend unit tests
pytest
```

---

## Required APIs

| Service | Required | Purpose | Free tier |
|---|---|---|---|
| Groq | Yes | Shadow models | 14,400 req/day per model |
| MongoDB Atlas | Yes | Storage | 512MB free |
| Wikidata | Yes | Factual verification | No key needed |
| Serper.dev | Optional | Temporal verification | 2,500 searches/month |

---

## Example Request

```bash
curl -X POST http://localhost:8000/api/v1/monitor \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fie-your-key" \
  -d '{
    "prompt": "Who invented the telephone?",
    "primary_output": "Thomas Edison invented the telephone.",
    "primary_model_name": "gpt-4",
    "run_full_jury": true
  }'
```

Example response (trimmed):

```json
{
  "high_failure_risk": true,
  "archetype": "MODEL_BLIND_SPOT",
  "failure_signal_vector": {
    "agreement_score": 0.75,
    "entropy_score": 0.406,
    "high_failure_risk": true
  },
  "jury": {
    "primary_verdict": {
      "root_cause": "FACTUAL_HALLUCINATION",
      "confidence_score": 0.62
    }
  },
  "ground_truth": {
    "verified_answer": "Alexander Graham Bell",
    "confidence": 0.85,
    "source": "wikidata",
    "from_cache": false
  },
  "fix_result": {
    "fix_applied": true,
    "fix_strategy": "WIKIDATA_OVERRIDE",
    "fixed_output": "Alexander Graham Bell",
    "original_output": "Thomas Edison"
  }
}
```

---

## Key Thresholds

| Parameter | Value | File |
|---|---|---|
| High entropy threshold | 0.75 | `config.py` |
| Low agreement threshold | 0.80 | `config.py` |
| Primary-outlier cosine threshold | 0.72 | `engine/detector/consistency.py` |
| Shadow agreement minimum | 0.60 | `engine/detector/consistency.py` |
| GT Gate — jury confidence minimum | 0.45 | `app/routes.py` |
| Wikidata override confidence | 0.75 | `engine/verifier/ground_truth_pipeline.py` |
| Cache write confidence | 0.90 | `engine/verifier/ground_truth_pipeline.py` |
| Shadow consensus minimum | 0.60 | `engine/verifier/ground_truth_pipeline.py` |
| Embedding dimensions | 384 | `engine/encoder.py` |

---

## Technology Stack

- **Backend:** FastAPI, Pydantic, Python 3.11
- **Storage:** MongoDB Atlas
- **Shadow Models:** Groq API (Llama, DeepSeek, Qwen)
- **Semantic Encoder:** SentenceTransformers `all-MiniLM-L6-v2`
- **Vector Search:** FAISS
- **Fact Verification:** Wikidata SPARQL, Serper.dev
- **Frontend:** React, Vite
- **Auth:** Google OAuth, JWT
- **Deployment:** Docker, Google Cloud Run, Vercel

---

## Benchmark Results

Evaluated on **TruthfulQA** (817 adversarial questions designed to trigger LLM hallucinations). 869 labeled examples generated via the synthetic pipeline.

| Method | Recall | FPR | F1 | AUC-ROC |
|---|---|---|---|---|
| POET rule-based (baseline) | 56.4% | 38.7% | 58.7% | — |
| XGBoost classifier (equal FPR) | **65.5%** | 40.2% | 63.7% | 0.663 |
| XGBoost classifier (best F1) | 80.5% | 50.6% | **69.7%** | 0.663 |

**Cross-validation (5-fold):** Recall = 63.7% ± 4.0%

**Key finding:** The trained XGBoost classifier achieves +9.1% recall over POET at equal false positive rate. Feature importance analysis shows the Diagnostic Jury verdict is the strongest predictor — confirming that the 3-agent jury adds meaningful signal beyond ensemble disagreement alone.

---

## For Full Technical Documentation

See [README_files/FIE_COMPLETE_TECHNICAL_STORY.md](README_files/FIE_COMPLETE_TECHNICAL_STORY.md) — covers every algorithm, formula, pipeline decision, benchmark result, and file in detail.

---

## License

MIT
