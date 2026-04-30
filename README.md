# Failure Intelligence Engine (FIE)

**Real-time LLM failure detection, diagnosis, and automatic correction.**

FIE sits between your LLM and your users. When the model gives a wrong answer, FIE catches it, finds the correct answer from a trusted source, and returns the correction — before the user ever sees the mistake.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)](https://mongodb.com/atlas)
[![PyPI](https://img.shields.io/badge/PyPI-fie--sdk-blue?logo=pypi&logoColor=white)](https://pypi.org/project/fie-sdk)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

---

## Quickstart — Use the SDK

```bash
pip install fie-sdk
```

```python
from fie import monitor

@monitor(
    fie_url="https://failure-intelligence-system-800748790940.asia-south1.run.app",
    api_key="your-api-key",
    mode="correct",   # or "monitor"
)
def ask_ai(prompt: str) -> str:
    return your_llm_call(prompt)

response = ask_ai("Who invented the telephone?")
# Returns corrected answer if LLM was wrong, original answer if correct
```

### SDK Modes

| Mode | Behavior |
| --- | --- |
| `monitor` | Non-blocking — FIE checks in background, original answer returned immediately |
| `correct` | Synchronous — FIE verifies and returns corrected answer if failure detected |

### Get an API Key

1. Sign in at [https://failure-intelligence-system.pages.dev](https://failure-intelligence-system.pages.dev)
2. Your API key is shown in the dashboard after login

---

## How It Works

```text
Your LLM answer → FIE
                   ├── Shadow ensemble (3 independent models cross-check)
                   ├── Failure Signal Vector (agreement, entropy, outlier detection)
                   ├── Diagnostic Jury (3 agents vote on root cause)
                   ├── Ground Truth Pipeline (Wikidata → Google Search → consensus)
                   └── Fix Engine (returns corrected answer or escalates)
```

**Classifier:** XGBoost v3 (AUC 0.728) backed by a 5-type question router. Factual questions go through full external verification; code/opinion questions skip it to avoid false positives.

---

## Self-Hosting

### Requirements

- Python 3.11+
- MongoDB Atlas (free tier works)
- Groq API key — free at [console.groq.com](https://console.groq.com)
- Node.js 18+ (dashboard only)

### 1. Clone & Install

```bash
git clone https://github.com/AyushSingh110/Failure_Intelligence_System.git
cd Failure_Intelligence_System
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` in the project root:

```env
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=fie_database

GROQ_API_KEY=gsk_your_groq_key
GROQ_ENABLED=true
GROQ_MODELS=["llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b","qwen-qwq-32b"]

SERPER_API_KEY=your_serper_key     # optional — needed for temporal questions
SERPER_ENABLED=true

OLLAMA_ENABLED=false

GOOGLE_CLIENT_ID=your-google-oauth-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret
GOOGLE_REDIRECT_URI=http://localhost:5173

JWT_SECRET_KEY=replace-with-a-long-random-secret-minimum-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24
ADMIN_EMAIL=your@email.com
```

### 3. Start Server

```bash
uvicorn app.main:app --reload
# Backend: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 4. Dashboard (optional)

```bash
cd Frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
```

---

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/monitor` | Main endpoint — full detection + correction pipeline |
| `POST` | `/api/v1/diagnose` | Run diagnostic jury only |
| `POST` | `/api/v1/analyze` | Signal extraction only (no jury, no GT) |
| `POST` | `/api/v1/feedback/{id}` | Submit human feedback on an inference |
| `GET` | `/api/v1/monitor/model-info` | Active model version, thresholds, AUC |
| `GET` | `/api/v1/analytics/usage` | Request volume, failure rate, daily breakdown |
| `GET` | `/api/v1/analytics/model-performance` | XGBoost accuracy, per-question-type stats |
| `GET` | `/api/v1/analytics/calibration` | Confidence calibration curves + ECE score |
| `GET` | `/api/v1/analytics/question-breakdown` | Failure/fix/escalation rate per question type |
| `GET` | `/api/v1/analytics/paper-metrics` | All benchmark metrics in one call |
| `GET` | `/api/v1/analytics/sdk-telemetry` | Usage data from opted-in SDK users |
| `GET` | `/health` | Health check |

### Example Request

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

---

## Opt-In Telemetry (SDK Users)

To share anonymized usage data (no prompts, no API keys):

```bash
FIE_TELEMETRY=true python your_app.py
```

This sends: SDK version, question type, failure detection rate, mode. Nothing else.

---

## Benchmark Results

Evaluated on TruthfulQA (817 adversarial questions).

| Method | Recall | FPR | F1 | AUC-ROC |
| --- | --- | --- | --- | --- |
| POET rule-based (baseline) | 56.4% | 38.7% | 58.7% | — |
| XGBoost v2 | 71.6% | 53.9% | 63.5% | 0.728 |

---

## Required Services

| Service | Required | Free Tier |
| --- | --- | --- |
| [Groq](https://console.groq.com) | Yes | 14,400 req/day |
| [MongoDB Atlas](https://mongodb.com/atlas) | Yes | 512 MB |
| [Wikidata](https://wikidata.org) | Yes | No key needed |
| [Serper.dev](https://serper.dev) | Optional | 2,500 searches/month |

---

## License

Apache-2.0
