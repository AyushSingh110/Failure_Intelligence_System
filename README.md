# Failure Intelligence Engine — v3.1.0

**AI reliability platform for detecting, diagnosing, explaining, and correcting LLM failures in real time.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Dashboard-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)](https://mongodb.com/atlas)
[![Groq](https://img.shields.io/badge/Groq-Shadow%20Models-orange)](https://groq.com)
[![PyPI](https://img.shields.io/badge/PyPI-fie--sdk-blue?logo=pypi&logoColor=white)](https://pypi.org/project/fie-sdk)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue)](https://github.com/facebookresearch/faiss)

Failure Intelligence Engine, or FIE, monitors LLM outputs after generation, estimates whether the answer is risky or unstable, diagnoses the likely root cause, generates explainable reasoning artifacts, and can automatically return a safer or more grounded answer when possible.

This repository contains:

- A FastAPI backend that runs the failure analysis pipeline
- A React dashboard for login, monitoring, diagnosis, alerts, and settings
- A lightweight Python SDK with an `@monitor` decorator
- Deployment assets for Google Cloud Run and Vercel

## What FIE does

FIE is built for failure modes that normal API monitoring cannot see:

- Hallucinations and factual drift
- Prompt injection and jailbreak attempts
- Contradictory outputs across models
- High-entropy unstable generations
- Complex prompt parsing failures
- Temporal or live-data questions that exceed model knowledge cutoff

Instead of only saying "something went wrong", FIE tries to answer:

1. Did the model likely fail?
2. What kind of failure is it?
3. Why did the system reach that diagnosis?
4. Can we automatically fix the response safely?

## Core features

- Real-time monitoring through `/api/v1/monitor`
- Multi-model shadow comparison using Groq models
- Failure Signal Vector with agreement, entropy, ensemble disagreement, and embedding distance
- Failure archetype labeling and adaptive clustering
- Diagnostic jury with specialist agents for adversarial, linguistic, and domain failures
- Auto-fix engine with multiple mitigation strategies
- Explainable AI payloads for both user-safe and internal views
- Human-readable summaries for the dashboard
- Google OAuth login with JWT sessions
- Per-user tenant isolation and API key generation
- MongoDB-backed inference history and user management
- React dashboard with Dashboard, Analyze, Diagnose, Alerts, Vault, and Settings pages
- Cloud deployment flow for Google Cloud Run backend and Vercel frontend
- Python SDK for easy app integration

## fie-sdk package

`fie-sdk` is an important part of this project, not just an internal helper. It is the Python package that lets other applications integrate FIE without needing to call the backend manually.

Package metadata from the repository:

- Package name: `fie-sdk`
- Current version: `0.2.0`
- Python requirement: `>=3.9`
- Source package: `fie/`
- PyPI long description source: root `README.md`

### What the package provides

- The `@monitor` decorator for LLM functions
- A small HTTP client for talking to the FIE backend
- Two runtime modes: background monitoring and real-time correction
- A simple integration path for external apps using only an API key and backend URL

### Install the package

From PyPI:

```bash
pip install fie-sdk
```

From this repository in editable mode:

```bash
pip install -e .
```

Build distribution artifacts locally:

```bash
python -m build
```

This produces distributable package files in `dist/`, including the wheel and source tarball.

Important: users only get the new package when version `0.2.0` is built and published. If `0.1.0` is the latest version on PyPI, `pip install fie-sdk` will still install `0.1.0`.

If you want the PyPI package page to show this new README, you must publish the new version after building it. PyPI renders the README from the uploaded release metadata, not directly from GitHub.

## How the system works

### End-to-end flow

1. Your application calls an LLM.
2. The Python SDK or frontend sends the prompt and primary output to FIE.
3. FIE fans the same prompt out to shadow models through Groq.
4. The backend builds a shared `model_outputs` list containing the primary answer and successful shadow responses.
5. Phase 1 computes a Failure Signal Vector.
6. Phase 2 assigns an archetype, updates clustering, and records degradation trends.
7. Phase 3 optionally runs the DiagnosticJury to estimate the most likely root cause.
8. The fix engine decides whether a safe automatic mitigation should be applied.
9. The explainability layer builds structured explanation bundles and a plain-language summary.
10. The result is stored in MongoDB and shown in the dashboard.

### Pipeline layers

#### Phase 1: Failure signal extraction

The first layer converts raw outputs into a structured signal:

- `agreement_score`: how many outputs semantically agree
- `fsd_score`: first-second dominance gap
- `entropy_score`: normalized divergence across grouped answers
- `ensemble_disagreement`: whether pairwise disagreement is significant
- `embedding_distance`: semantic distance between primary and reference outputs
- `high_failure_risk`: composite risk flag based on configured thresholds

Relevant files:

- `engine/detector/consistency.py`
- `engine/detector/entropy.py`
- `engine/detector/ensemble.py`
- `engine/detector/embedding.py`

#### Phase 2: Archetypes, clustering, and trend tracking

Once signals are extracted, FIE:

- Labels the likely failure archetype
- Assigns the event to an adaptive cluster
- Tracks evolution using EMA-based degradation monitoring

Relevant files:

- `engine/archetypes/labeling.py`
- `engine/archetypes/clustering.py`
- `engine/evolution/tracker.py`

#### Phase 3: DiagnosticJury

The jury is a set of specialized agents that analyze the prompt and outputs from different angles:

- `AdversarialSpecialist`: prompt injection, jailbreak, hidden instruction, token smuggling patterns
- `LinguisticAuditor`: complexity, ambiguity, and prompt structure issues
- `DomainCritic`: factual mismatch, knowledge boundary, and temporal failures

The jury aggregates verdicts, chooses a primary diagnosis, and produces a failure summary.

Relevant files:

- `engine/agents/failure_agent.py`
- `engine/agents/adversarial_specialist.py`
- `engine/agents/linguistic_auditor.py`
- `engine/agents/domain_critic.py`

#### Phase 4: Auto-fix engine

If the system has enough confidence, FIE can apply a mitigation strategy automatically:

- `SHADOW_CONSENSUS`
- `SANITIZE_AND_RERUN`
- `CONTEXT_INJECTION`
- `PROMPT_DECOMPOSITION`
- `SELF_CONSISTENCY`
- `NO_FIX`

There is also a grounded fallback path for factual issues using retrieval plus Groq-based correction.

Relevant files:

- `engine/fix_engine.py`
- `engine/rag_grounder.py`
- `engine/rag/retriever.py`

## Explainable AI layer

One of the biggest recent additions in this project is the explainability pipeline.

FIE now produces structured explanation artifacts instead of only returning a verdict. For monitored and diagnostic responses, the backend builds:

- `explanation_external`: user-safe explanation bundle
- `explanation_internal`: richer internal bundle for admins
- `human_explanation`: short natural-language summary for the UI

Each explanation bundle can include:

- Final label and selected fix strategy
- Confidence score for the explanation itself
- Decision trace across pipeline stages
- Signals that contributed to the diagnosis
- Evidence items and ranked attributions
- Alternative causes considered
- Uncertainty notes

The external explanation is redacted so sensitive internal reasoning is not exposed to normal users, while admins can still inspect deeper internal evidence.

Relevant files:

- `engine/explainability/explanation_builder.py`
- `engine/explainability/redaction.py`
- `engine/explainability/humanizer.py`
- `fie-dashboard/src/components/ExplanationPanel.jsx`

## Authentication and multi-tenant access

The project now includes a full authentication and tenancy layer.

### What happens on login

1. The React frontend starts Google OAuth.
2. Google returns an authorization code to the frontend.
3. The frontend sends that code to `/api/v1/auth/google-callback`.
4. The backend exchanges the code with Google, fetches user info, and creates or loads the user in MongoDB.
5. The backend returns a JWT session token plus the user's API key and tenant metadata.
6. Every stored inference is associated with that user's `tenant_id`.
7. Normal users only see their own data. Admin users can view internal explanations and global user data.

### Auth-related capabilities

- Google OAuth login
- JWT session verification
- Generated per-user API keys for SDK use
- Tenant isolation for inference history
- Admin-only visibility for internal explanations
- API key regeneration from the Settings page

Relevant files:

- `app/auth.py`
- `app/auth_routes.py`
- `app/auth_guard.py`
- `fie-dashboard/src/lib/auth.js`
- `fie-dashboard/src/pages/LoginPage.jsx`
- `fie-dashboard/src/pages/Settingspage.jsx`

## Dashboard overview

The current frontend is a React/Vite dashboard, not the older Streamlit UI described in the previous README.

Main pages:

- `Dashboard`: KPI overview, risk rate, entropy, agreement, recent activity
- `Analyze`: direct Phase 1 and Phase 2 exploration
- `Diagnose`: full diagnostic view with jury verdicts and explainability panel
- `Alerts`: high-risk events and degradation-oriented views
- `Vault`: stored inference history
- `Settings`: API key management and SDK integration instructions

Frontend location:

- `fie-dashboard/`

## Python SDK

The repository includes a lightweight SDK published as `fie-sdk`, and the package source lives in `fie/`.

### Main integration primitive

```python
from fie import monitor

@monitor(
    fie_url="http://localhost:8000",
    api_key="your-fie-api-key",
    mode="correct",
)
def call_your_llm(prompt: str) -> str:
    return your_llm(prompt)
```

### SDK modes

- `mode="monitor"`: returns the original answer immediately and sends the event to FIE in the background
- `mode="correct"`: waits for FIE analysis and returns the corrected output if a fix is applied

Relevant files:

- `fie/monitor.py`
- `fie/client.py`
- `pyproject.toml`

## API overview

Main backend routes live under `/api/v1`.

### Core endpoints

- `POST /api/v1/monitor`: main production endpoint
- `POST /api/v1/diagnose`: run diagnostic jury on provided outputs
- `POST /api/v1/analyze`: Phase 1 analysis
- `POST /api/v1/analyze/v2`: Phase 2 analysis with clustering and trend summary
- `GET /api/v1/inferences`: list stored inferences for the current tenant
- `GET /api/v1/trend`: get degradation metrics
- `GET /api/v1/clusters`: summarize archetype clusters
- `GET /api/v1/auth/me`: get current logged-in user

### Health endpoints

- `GET /`: service identity and status
- `GET /health`: health check

### Example monitor request

```json
{
  "prompt": "Who invented the telephone?",
  "primary_output": "Thomas Edison invented the telephone.",
  "primary_model_name": "gpt-4",
  "run_full_jury": true,
  "latency_ms": 842.1
}
```

### Example monitor response shape

```json
{
  "shadow_model_results": [],
  "all_model_outputs": [],
  "ollama_available": false,
  "failure_signal_vector": {
    "agreement_score": 0.25,
    "fsd_score": 0.0,
    "answer_counts": {},
    "entropy_score": 0.91,
    "ensemble_disagreement": true,
    "ensemble_similarity": 0.33,
    "high_failure_risk": true
  },
  "archetype": "FACTUAL_HALLUCINATION",
  "embedding_distance": 0.42,
  "jury": null,
  "high_failure_risk": true,
  "failure_summary": "High failure risk detected.",
  "fix_result": null,
  "explanation_internal": null,
  "explanation_external": null,
  "human_explanation": null
}
```

## Project structure

```text
.
|-- app/                  FastAPI app, routes, auth, schemas
|-- engine/               Detection, diagnosis, explainability, fixes, RAG
|-- storage/              Local storage artifacts and FAISS files
|-- fie/                  Python SDK
|-- fie-dashboard/        React dashboard
|-- tests/                Backend and feature tests
|-- Dockerfile            Backend container image
|-- cloudbuild.yaml       Google Cloud Build config
|-- requirements.txt      Backend dependencies
|-- pyproject.toml        SDK packaging metadata
```

## Local development

### Requirements

- Python 3.11+
- Node.js 18+
- MongoDB Atlas connection string
- Groq API key
- Google OAuth client credentials for frontend login

### 1. Clone and install backend

```bash
git clone https://github.com/AyushSingh110/Failure_Intelligence_System.git
cd Failure_Intelligence_System
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure backend environment

Create `.env` in the project root and add the values your deployment uses:

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

### 3. Run the backend

```bash
uvicorn app.main:app --reload
```

Backend default URL:

```text
http://localhost:8000
```

### 4. Run the React dashboard

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

Start the frontend:

```bash
npm run dev
```

Frontend default URL:

```text
http://localhost:5173
```

## Google Cloud and production deployment

This repo now includes a backend deployment path for Google Cloud Run and a frontend deployment path for Vercel.

### Backend: Google Cloud Run

The backend is containerized with the root `Dockerfile` and built through `cloudbuild.yaml`.

What the backend deployment does:

- Builds the FastAPI service into a container
- Runs the app on port `8080`
- Supports environment-driven configuration for MongoDB, Groq, JWT, and Google OAuth

#### Build image with Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml --substitutions=_IMAGE=YOUR_REGION-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/fie-backend
```

#### Deploy to Cloud Run

```bash
gcloud run deploy fie-backend \
  --image YOUR_REGION-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/fie-backend \
  --region YOUR_REGION \
  --platform managed \
  --allow-unauthenticated
```

After deployment, set the backend environment variables in Cloud Run:

- `MONGODB_URI`
- `MONGODB_DB_NAME`
- `GROQ_API_KEY`
- `GROQ_ENABLED`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- `JWT_SECRET_KEY`
- `JWT_ALGORITHM`
- `JWT_EXPIRE_HOURS`
- `ADMIN_EMAIL`

### Frontend: Vercel

The React dashboard in `fie-dashboard/` is prepared for Vercel deployment.

Recommended Vercel settings:

- Framework preset: `Vite`
- Root directory: `fie-dashboard`
- Build command: `npm run build`
- Output directory: `dist`

Required frontend environment variables:

```env
VITE_API_URL=https://your-cloud-run-url/api/v1
VITE_GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
VITE_REDIRECT_URI=https://your-vercel-app.vercel.app
```

### Google OAuth production setup

When deploying, make sure the same redirect URL is configured consistently in three places:

1. Google Cloud OAuth authorized origins
2. Google Cloud OAuth redirect URIs
3. Backend `GOOGLE_REDIRECT_URI`

For production, `GOOGLE_REDIRECT_URI` should match the public frontend URL used by the React app.

## Tests

The repository includes tests for the key reliability features, including:

- Explainability payload generation
- Admin vs non-admin explanation visibility
- RAG-based factual correction fallback
- Detection pipeline behavior
- Diagnostic jury behavior

Run all tests:

```bash
pytest
```

Run a focused test file:

```bash
pytest tests/test_explainability.py
pytest tests/test_monitor_rag_fix.py
pytest tests/test_auth_visibility.py
```

## Technology stack

- FastAPI and Pydantic for the backend API
- MongoDB Atlas for persistent storage
- Groq for shadow-model fan-out
- FAISS plus sentence-transformers for semantic retrieval and similarity
- React and Vite for the dashboard
- Google OAuth for authentication
- JWT for sessions
- Docker, Cloud Build, and Cloud Run for backend deployment
- Vercel for frontend deployment

## What is new in v3.1.0 — Ground Truth Verification Layer

This release implements a complete 10-step ground truth verification system that solves the most critical weakness in v3.0: shadow models from the same family hallucinating in the same direction.

### Step 1 — Diverse shadow model families

Shadow models are now from three different training lineages to minimize correlated failure:

- `llama-3.3-70b-versatile` — Meta (strong general model)
- `deepseek-r1-distill-llama-70b` — DeepSeek (reasoning-tuned, different RLHF)
- `qwen-qwq-32b` — Alibaba Qwen (different pretraining corpus)

Change location: `config.py` → `groq_models`

### Step 2 — Confidence signals from shadow models

Each shadow model is now asked to rate its own certainty (`HIGH`, `MEDIUM`, or `LOW`) alongside its answer. The rating is parsed and stripped from the output text automatically.

Change location: `engine/groq_service.py` → `fan_out_with_confidence()`, `_parse_confidence()`

### Step 3 — Confidence-weighted consensus

The auto-fix engine no longer uses simple majority vote. Each shadow model's vote is weighted by its self-reported confidence (`HIGH=3`, `MEDIUM=2`, `LOW=1`). If weighted consensus is below 50%, the fix escalates to human review instead of guessing.

Change location: `engine/fix_engine.py` → `_apply_shadow_consensus()`

### Step 4 — Atomic claim extraction

Before querying external sources, FIE extracts the main factual claim from the model output as `{subject, property, value}`. This structured form can be matched against Wikidata's knowledge graph.

New file: `engine/claim_extractor.py`

### Step 5 — Wikidata structured fact verification

Wikidata is a free, structured knowledge graph (no API key required). FIE searches for the entity in the claim and asks Groq to check whether the claimed value matches the Wikidata description. This breaks the circular model-verifying-model problem because Wikidata is a database, not an LLM.

New file: `engine/verifier/wikidata_verifier.py`

### Step 6 — Serper real-time search for temporal questions

When the DomainCritic detects `TEMPORAL_KNOWLEDGE_CUTOFF`, Wikidata is not sufficient. FIE routes to Serper.dev (Google Search API) for live results. Requires `SERPER_API_KEY` in `.env`.

New file: `engine/verifier/serper_verifier.py`

### Step 7 — Verified answer cache

Every human-submitted correction (Step 8) and every high-confidence external verification (Wikidata/Serper ≥ 90%) is saved to a MongoDB `ground_truth_cache` collection. The cache is checked first on every request using embedding similarity (threshold 0.92). A cache hit returns the verified answer with confidence 1.0 and skips all external API calls.

New file: `engine/ground_truth_cache.py`

### Step 8 — User feedback loop (new endpoint)

```text
POST /api/v1/feedback/{request_id}
```

Users submit whether the model's answer was correct. If `is_correct=false` and `correct_answer` is provided, the correct answer is saved to the ground truth cache immediately.

New schemas: `FeedbackRequest`, `FeedbackResponse` in `app/schemas.py`

### Step 9+10 — Escalation layer

When no external source can establish reliable ground truth and shadow consensus is too weak, FIE no longer auto-corrects. It returns the original answer with `requires_human_review=true` and an `escalation_reason` in the response. The inference appears in the dashboard escalation queue for manual review.

New strategy: `HUMAN_ESCALATION` in `engine/fix_engine.py`

---

## APIs required — how to get them

### Groq API (required — shadow models)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign in with Google
3. Dashboard → API Keys → Create API Key
4. Copy the key (starts with `gsk_`)
5. Add to `.env`: `GROQ_API_KEY=gsk_your_key_here`

Free tier: 14,400 requests/day per model. The three new models (llama-3.3-70b, deepseek-r1, qwen-qwq) are all available on the free tier.

### Wikidata (no key required)

Wikidata uses the public Wikimedia API. No registration needed. Just set in `.env`:

```env
WIKIDATA_ENABLED=true
```

### Serper.dev (optional — required for real-time temporal verification)

1. Go to [serper.dev](https://serper.dev)
2. Sign up with Google
3. Dashboard → API Key → copy the key
4. Add to `.env`:

```env
SERPER_API_KEY=your_key_here
SERPER_ENABLED=true
```

Free tier: 2,500 searches/month. Paid: $50/month for 50,000 searches.

Without Serper: temporal questions will escalate to human review instead of auto-correcting. This is intentional — better to escalate than return an unverified "current" answer.

---

## Updated `.env` file

```env
MONGODB_URI=your_mongodb_atlas_uri
MONGODB_DB_NAME=fie_database

GROQ_API_KEY=gsk_your_groq_key
GROQ_ENABLED=true

# Optional: real-time verification for temporal questions
SERPER_API_KEY=your_serper_key
SERPER_ENABLED=true

# Wikidata: no key needed, just enable
WIKIDATA_ENABLED=true
GROUND_TRUTH_CACHE_ENABLED=true

OLLAMA_ENABLED=false

GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:5173

JWT_SECRET_KEY=replace-with-a-long-random-secret
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24
ADMIN_EMAIL=your-admin-email@example.com
```

---

## How to test everything in real time

This section walks you through verifying each new feature using real API calls — no unit tests, just `curl` or a REST client like Postman.

### 1. Start the backend

```bash
cd Failure_Intelligence_System
.venv\Scripts\activate        # Windows
uvicorn app.main:app --reload
```

### 2. Get your API key

Login at `http://localhost:5173` → Settings page → copy your `fie-xxxx` API key.

Or call directly:

```bash
# After OAuth login, the token is in localStorage as "fie_token"
# Use it to fetch your API key:
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### 3. Test Step 1+2+3 — Diverse models + confidence-weighted consensus

Send a factual question where one answer is clearly wrong:

```bash
curl -X POST http://localhost:8000/api/v1/monitor \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fie-your-api-key" \
  -d '{
    "prompt": "Who invented the telephone?",
    "primary_output": "Thomas Edison invented the telephone in 1876.",
    "primary_model_name": "gpt-4o",
    "run_full_jury": true,
    "latency_ms": 843
  }'
```

What to look for in the response:

- `shadow_model_results`: you should see 3 different model families responding
- `fix_result.fix_strategy`: should be `"GT_VERIFIED"` or `"SHADOW_CONSENSUS"`
- `fix_result.fix_explanation`: should mention confidence weights
- `ground_truth.source`: should say `"Wikidata"` if Wikidata found the entity
- `fix_result.fixed_output`: should say Alexander Graham Bell, not Edison

### 4. Test Step 5 — Wikidata verification directly

Send an output you know is factually wrong:

```bash
curl -X POST http://localhost:8000/api/v1/monitor \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fie-your-api-key" \
  -d '{
    "prompt": "What is the capital of Australia?",
    "primary_output": "Sydney is the capital of Australia.",
    "primary_model_name": "gpt-4o",
    "run_full_jury": true
  }'
```

Expected: `ground_truth.source` = `"Wikidata"`, `fix_result.fixed_output` should say `"Canberra"`.

### 5. Test Step 6 — Serper real-time search (requires SERPER_API_KEY)

Send a temporal question:

```bash
curl -X POST http://localhost:8000/api/v1/monitor \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fie-your-api-key" \
  -d '{
    "prompt": "Who is the current CEO of OpenAI?",
    "primary_output": "Sam Altman is the current CEO of OpenAI.",
    "primary_model_name": "gpt-4o",
    "run_full_jury": true
  }'
```

If Serper is configured: `ground_truth.source` = `"Google Search via Serper.dev"`.
If Serper is NOT configured: `requires_human_review: true`, `escalation_reason` will explain why.

### 6. Test Step 8 — Feedback loop

First, get a `request_id` from a previous monitor call (it's in the stored inference):

```bash
# List your inferences to get a request_id
curl http://localhost:8000/api/v1/inferences \
  -H "X-API-Key: fie-your-api-key"
```

Then submit feedback:

```bash
curl -X POST http://localhost:8000/api/v1/feedback/YOUR_REQUEST_ID \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fie-your-api-key" \
  -d '{
    "is_correct": false,
    "correct_answer": "Alexander Graham Bell invented the telephone.",
    "notes": "Edison invented the phonograph, not the telephone."
  }'
```

Expected response: `"cache_updated": true`

### 7. Test Step 7 — Cache hit

Now send the same question again:

```bash
curl -X POST http://localhost:8000/api/v1/monitor \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fie-your-api-key" \
  -d '{
    "prompt": "Who invented the telephone?",
    "primary_output": "Thomas Edison invented the telephone.",
    "primary_model_name": "gpt-4o",
    "run_full_jury": true
  }'
```

Expected: `ground_truth.from_cache: true`, `ground_truth.confidence: 1.0`.
The pipeline skips Wikidata and Serper entirely — the answer came from the human-verified cache.

### 8. Test Step 10 — Escalation

Send a question where shadow models will disagree AND Wikidata/Serper cannot verify:

```bash
curl -X POST http://localhost:8000/api/v1/monitor \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fie-your-api-key" \
  -d '{
    "prompt": "What will the stock price of Apple be next month?",
    "primary_output": "Apple stock will be $210 next month.",
    "primary_model_name": "gpt-4o",
    "run_full_jury": true
  }'
```

Expected: `requires_human_review: true`, `fix_result.fix_strategy: "HUMAN_ESCALATION"`.

---

## Current state of the project

Implemented in the current codebase:

- Multi-stage failure detection pipeline
- Jury-based diagnosis with 3 specialist agents
- Auto-fix strategies (6 strategies)
- Explainable AI response bundles
- Human-readable explanations
- Google-authenticated dashboard access
- Tenant-aware storage and API key management
- React dashboard deployment flow
- Google Cloud backend deployment assets
- **NEW v3.1**: Diverse shadow model families (Meta + DeepSeek + Qwen)
- **NEW v3.1**: Confidence-weighted shadow model consensus
- **NEW v3.1**: Atomic claim extraction via Groq
- **NEW v3.1**: Wikidata structured fact verification (no API key needed)
- **NEW v3.1**: Serper real-time search for temporal questions
- **NEW v3.1**: Ground truth verified answer cache (MongoDB)
- **NEW v3.1**: User feedback loop (`POST /feedback/{request_id}`)
- **NEW v3.1**: Escalation layer for low-confidence cases

Areas that are present but optional or still evolving:

- Ollama as a local shadow-model provider
- Extended RAG workflows (arXiv, custom document stores)
- Per-tenant threshold calibration from feedback data
- Rate limiting and async job queue for high-traffic deployments

## License

MIT
