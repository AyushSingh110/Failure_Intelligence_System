# Failure Intelligence Engine

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

## Current state of the project

Implemented in the current codebase:

- Multi-stage failure detection pipeline
- Jury-based diagnosis
- Auto-fix strategies
- Explainable AI response bundles
- Human-readable explanations
- Google-authenticated dashboard access
- Tenant-aware storage and API key management
- React dashboard deployment flow
- Google Cloud backend deployment assets

Areas that are present but optional or still evolving:

- Ollama as a local shadow-model provider
- Extended RAG workflows
- More production hardening around quotas, rate limits, and observability

## License

MIT
