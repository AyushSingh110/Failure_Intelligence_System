# FIE Codebase Guide

Detailed explanation of every file in the project — what it does, why it exists, and how it fits into the overall system. Use this as a map when navigating the code.

---

## `fie/` — The Python Package (`pip install fie-sdk`)

This is the installable package. Everything a user of the SDK touches lives here.

---

### `fie/__init__.py`

Public API surface of the package. Exports `scan_prompt`, `monitor`, `GuardedResponse`, `preflight_check`, `FIEClient`, `FIEConfig`, `get_config`, and `integrations`. Keeps the version string. Nothing else — no logic.

---

### `fie/adversarial.py`

The core detection engine. The most important file in the package.

**What it does:** Scans a prompt for adversarial attacks using 7 detection layers running in parallel. Returns a `ScanResult` with `is_attack`, `attack_type`, `confidence`, `layers_fired`, `matched_text`, `mitigation`, and per-layer `evidence`.

**Architecture (top to bottom):**

- **Normalization** — homoglyph mapping, leet-speak decoding, zero-width character stripping. All pattern layers run on the normalized prompt, not the raw one.
- **`_ATTACK_THRESHOLDS`** — per-attack-type confidence thresholds. Each attack type has a different bar because some (TOKEN_SMUGGLING) have near-zero FPR while others (JAILBREAK_ATTEMPT) are noisier.
- **`_LAYER_WEIGHTS`** — precision-calibrated weight for each layer used by the aggregator.
- **`LayerResult`** dataclass — the output contract for every layer wrapper: `layer_name`, `attack_type`, `confidence`, `evidence`, `latency_ms`.
- **`_ScanCache`** — LRU cache keyed by SHA-256 hash of the prompt. TTL 300s, 512 entries max. Raw prompts never stored.
- **Layer 1 — Regex** (`_run_pattern_detection`) — fast pattern library for direct injection, jailbreak personas, instruction overrides, token smuggling. Highest precision layer, weight 1.5.
- **Layer 2 — PromptGuard** (`_run_guard_detection`) — keyword-combination scoring with leet-speak normalization. Catches obfuscated versions of known patterns.
- **Layer 3 — Many-shot** (`_run_many_shot_detection`) — counts Human/Assistant turn pairs, scans for harmful Q turns, fires when scripted conditioning is detected. Based on Anil et al. 2024.
- **Layer 4 — Indirect injection** (`_run_indirect_injection_detection`) — splits prompt into task vs document portions, scans document section for embedded override instructions. Also checks model output for compliance signals when `primary_output` is provided.
- **Layer 5 — GCG** (`_run_gcg_detection`) — Shannon entropy on tail characters, spaced-punctuation density, non-word token density. Targets gradient-optimized adversarial suffix strings.
- **Layer 6 — Perplexity proxy** (`_run_perplexity_proxy`) — compression ratio (zlib), non-dictionary token density, character-type entropy, token-length variance. Catches Base64, Caesar/ROT, Unicode lookalike encoded payloads. Minimum 8 tokens and 60 characters required to prevent false positives on short technical vocabulary.
- **Layer 7 — PAIR classifier** (`_run_pair_classifier`) — LinearSVM on sentence-transformer embeddings. Bundled model (`pair_intent_classifier_v2.pkl`). Catches iteratively rephrased natural-language jailbreaks that evade all structural layers.
- **`_run_layer_safe`** — wraps a single layer with a `ThreadPoolExecutor(max_workers=1)` timeout and exception isolation. A slow or crashing layer never affects other layers.
- **`_run_all_layers_parallel`** — submits all 7 layers to a `ThreadPoolExecutor(max_workers=7)`. Collects results with `as_completed(timeout=10.0)`. A `TimeoutError` is caught and partial results are returned — scan never fails.
- **`_weighted_aggregate`** — combines fired layer results into a single verdict. Fast-path for regex/gcg_suffix at ≥90% of threshold (near-zero FPR, block immediately). Otherwise: group by attack_type, weighted average confidence, corroboration boost (+0.08 for 2 layers agreeing, +0.12 for 3+), winner = highest weighted score.
- **`_record_session`** — best-effort session tracking. Hashes the prompt, calls `get_tracker().record(...)`, logs escalations if any rule fires.
- **`scan_prompt`** — the public function. Cache → parallel layers → framing filter → aggregation → three-zone routing (CLEAR SAFE / UNCERTAIN / CLEAR ATTACK) → optional LlamaGuard in UNCERTAIN zone → session tracking → cache write → return.

**Why parallel layers?** Sequential execution meant each layer waited for the previous. A slow sentence-transformer load on first call blocked everything. Now total scan time = slowest single layer, not the sum.

**Why weighted voting?** Max-wins voting means a low-precision layer (perplexity proxy) with slightly higher confidence overrides a high-precision layer (regex). Weights respect the actual reliability of each layer.

**Why per-attack-type thresholds?** A single global threshold is wrong because different attack types have different false-positive rates. TOKEN_SMUGGLING patterns are extremely precise; JAILBREAK_ATTEMPT is noisier. Tuning them separately gives better precision/recall per type.

---

### `fie/llama_guard.py`

LlamaGuard 3 (8B) Tier-3 tiebreaker for the UNCERTAIN confidence zone.

**What it does:** Calls `meta-llama/llama-guard-3-8b` on Groq's free tier when `scan_prompt` reaches the UNCERTAIN zone (confidence between 60% and 100% of threshold). Returns `True` (unsafe) or `False` (safe).

**Key components:**

- **`_LGCache`** — SHA-256 keyed LRU cache, TTL 600s, 256 entries. Same prompt never hits Groq twice. Raw prompts never stored.
- **`_CircuitBreaker`** — opens after 3 consecutive Groq failures, half-opens after 60s recovery. Prevents the scan from hanging on a degraded Groq endpoint.
- **`query_llama_guard(prompt)`** — checks cache, checks circuit breaker, POSTs to Groq, parses `"safe"` / `"unsafe\n<category>"` response, updates cache and circuit state.
- **`reset_circuit()`** — forces circuit back to CLOSED. Used in tests.

**Why fail-open?** If LlamaGuard is unavailable and local confidence is below threshold, the prompt is allowed through. A false negative from an unavailable Tier-3 is less damaging than making the guard a single point of failure.

**Why only in UNCERTAIN zone?** CLEAR SAFE and CLEAR ATTACK are confident enough that a 300ms LlamaGuard call adds no value. Only the ambiguous middle ground benefits from a second opinion.

---

### `fie/session_tracker.py`

Multi-turn session escalation tracker. Detects attack patterns that span multiple conversation turns.

**What it does:** Records every turn (hash + metadata, never raw text). After each turn, checks 4 escalation rules against the session's recent history. Returns a `SessionEscalation` if a rule fires.

**Why this exists:** A single-turn scan can't detect crescendo attacks — where each individual turn looks borderline but the sequence of turns reveals clear adversarial intent. Spreading an attack across turns is a known evasion technique.

**Data structures:**

- **`TurnRecord`** — `prompt_hash`, `attack_type`, `confidence`, `is_attack`, `ts`. No raw prompts.
- **`_Session`** — holds a `deque(maxlen=20)` of `TurnRecord`s and a `last_seen` timestamp.
- **`SessionEscalation`** — `rule` (which rule fired), `severity` (LOW/MEDIUM/HIGH), `context` (rule-specific metadata, no raw content).

**Escalation rules:**

- `RAPID_FIRE` — ≥5 attack hits within 60 seconds. HIGH severity.
- `ESCALATING_CONF` — 3 consecutive attack turns with strictly increasing confidence. MEDIUM severity.
- `JAILBREAK_PIVOT` — same attack type repeated ≥3 times in last 10 turns. HIGH severity.
- `MULTI_VECTOR` — ≥3 distinct attack types in last 10 turns. MEDIUM severity.

**Privacy:** Only SHA-256 hashes stored. Sessions expire after 30 min idle. Hard cap of 10,000 sessions with LRU eviction to prevent unbounded memory growth.

**`get_tracker()`** — thread-safe singleton accessor.

---

### `fie/preflight.py`

SDK-side pre-flight guard. Called at the start of every `@monitor` wrapper before the primary LLM runs.

**What it does:** Runs `scan_prompt`, checks the result against block mode config, returns a `GuardResult`. If blocked, the caller returns a `GuardedResponse` and never calls the LLM.

**Key types:**

- **`GuardResult`** — internal result from `preflight_check`. Fields: `blocked`, `attack_type`, `confidence`, `layers_fired`, `refusal_message`.
- **`GuardedResponse`** — a `str` subclass returned to the user when a prompt is blocked. Inherits from `str` so it's transparent to callers that just forward the return value. Inspect `.blocked`, `.attack_type`, `.confidence` to detect block events explicitly.

**`preflight_check(prompt, session_id)`** — checks if prompt is empty, calls `_safe_scan(prompt, session_id)`, checks block mode (MongoDB → env var → default True), returns `GuardResult`.

**`_get_block_enabled()`** — reads from MongoDB `fie_config` first (hot-configurable, no restart needed), falls back to `PREFLIGHT_BLOCK_ENABLED` env var, then defaults to True.

**`make_guarded_response(guard)`** — convenience constructor that turns a `GuardResult` into a `GuardedResponse`.

**Why never raise?** `preflight_check` is wrapped in try/except everywhere. A bug in the guard must never take the primary model offline. Failures log and return `blocked=False`.

---

### `fie/framing_filter.py`

Dampens confidence scores for prompts with fictional, hypothetical, or academic framing.

**What it does:** Detects signals like "write a story where...", "hypothetically speaking...", "for a novel I'm writing..." and returns a dampening factor < 1.0 that gets multiplied against the aggregated confidence before the threshold check.

**Why it exists:** Without this, prompts like "write a story where a chemistry teacher explains a dangerous reaction" score high on semantic intent layers because they contain harmful topic signals, even though the framing indicates no real-world harm intent. This was causing false positives on legitimate creative writing requests.

**When dampening is suppressed:** If `regex`, `many_shot`, `prompt_guard`, or `indirect_injection` fired — these indicate a structural attack technique regardless of framing. A fictional wrapper on a real injection pattern is still an injection.

---

### `fie/monitor.py`

The `@monitor` decorator. The main entry point for SDK users.

**What it does:** Wraps any Python function that calls an LLM. Intercepts every prompt/response pair and routes it through one of three modes.

**Three modes:**

- **`local`** — fully offline. Pre-flight guard + heuristic `predict_local`. Never makes a network call. Returns the original answer unchanged (monitoring only, no correction in local mode).
- **`monitor`** — pre-flight guard blocks attacks. If safe, calls the primary LLM and returns the original answer immediately. Sends prompt+response to the FIE server in a background thread. Non-blocking — user never waits for FIE analysis.
- **`correct`** — pre-flight guard blocks attacks. If safe, calls the primary LLM. Sends to FIE server synchronously. If FIE detects a failure and has a verified fix, returns the fixed answer instead of the original. User waits for FIE verdict (timeout 300s).

**`_log_result`** — formats and logs the FIE server response to the Python logger. Shows archetype, risk level, fix status, ground truth verification result, and escalation reason.

**`_fire_slack_alert`** — posts a formatted Slack message to a webhook URL when `high_failure_risk` is True. Fire-and-forget, never blocks.

**`_send_local_telemetry`** — opt-in anonymized ping in local mode. Only sends when `FIE_TELEMETRY=true`. Runs in a daemon thread. Sends SDK version, question type, failure detection rate, attack type. No prompts, no outputs.

---

### `fie/local_predictor.py`

Zero-dependency offline hallucination heuristics. Used by `@monitor(mode="local")`.

**What it does:** Runs a set of regex-based detectors on the prompt+response pair and returns a `LocalPrediction` with `is_suspicious`, `confidence`, `question_type`, and `signals`.

**Detectors:**

- Hedging language — "I think", "probably", "I'm not sure", etc.
- Temporal cutoff signals — "as of my knowledge cutoff", "I cannot access the internet"
- Self-contradiction patterns — "however, this", "but actually", "on the other hand"
- Question-type routing — opinion/code questions return low risk by default because they don't have ground truth to be wrong about

**Limitations:** Cannot do shadow model cross-checking, Wikidata verification, or auto-calibrated thresholds. Those require a server. `local` mode is a lightweight safety net, not a replacement for the full pipeline.

---

### `fie/config.py`

Holds `FIEConfig` (dataclass with `fie_url` and `api_key`) and `get_config()` which resolves values from function arguments → `FIE_URL` env var → `FIE_API_KEY` env var. No logic beyond that.

---

### `fie/client.py`

HTTP client for the FIE server API. Used by `@monitor(mode="monitor")` and `@monitor(mode="correct")`.

**Key method: `client.monitor(prompt, primary_output, ...)`** — POSTs to `/api/v1/monitor`, returns the parsed JSON response or `None` on failure. Handles auth headers, timeout, and error logging.

**`client._send_telemetry(event, payload)`** — opt-in anonymized ping to the deployed server. Only fires when the user has opted in.

---

## `app/` — FastAPI Server

---

### `app/main.py`

ASGI entry point. Wires together middleware, routers, lifespan events, and rate limiting.

**Startup sequence:** `configure_logging()` → `initialize_vault()` (MongoDB + in-memory fallback) → `load_from_db()` (hot-configurable thresholds) → `_warm_encoder_in_background()` (sentence encoder warm-up, non-blocking).

**Middleware:** `security_and_logging` injects `X-Request-ID`, binds it into every log line via `bind_request_id()`, and adds security headers (HSTS, CSP, X-Frame-Options, etc.).

---

### `app/routes/inference.py`

Core inference endpoints: `/track`, `/analyze`, `/diagnose`. These are the main pipeline routes — accept a prompt+response pair, run the full LangGraph pipeline, return a `DiagnosticResponse`.

---

### `app/routes/monitor.py`

Real-time monitoring endpoint `POST /monitor`. Runs pre-flight guard server-side before shadow fan-out. Also hosts `POST /feedback/{request_id}` — the ground truth feedback loop that saves corrections to the GT cache, updates the signal log, and feeds the XGBoost retraining buffer.

---

### `app/routes/analytics.py`

Read-only analytics: `/trend` (EMA-based degradation), `/clusters` (archetype clustering), `/telemetry` (SDK usage data), `/analytics/usage`, `/analytics/model-performance`, `/analytics/calibration`.

---

### `app/routes/admin.py`

Admin-only endpoints: email notifications (`/notifications/digest`), guard config (`/admin/guard/config` — toggle block mode at runtime without restart), health checks.

---

### `app/routes/playground.py`

`POST /playground/run` — ephemeral side-by-side comparison. Runs the raw primary model call and the full FIE pipeline in parallel, returns both results. Results are NOT written to MongoDB — this is for demo and testing only.

---

### `app/schemas.py`

All Pydantic models used across routes: `MonitorRequest`, `MonitorResponse`, `FeedbackRequest`, `FeedbackResponse`, `PlaygroundRequest`, provenance types, and the `FailureSignalVector` model. Single source of truth for request/response shapes.

---

### `app/auth.py` / `app/auth_routes.py` / `app/auth_guard.py`

JWT-based auth with Google OAuth. `auth.py` generates and validates JWTs. `auth_routes.py` exposes `/auth/login`, `/auth/register`, `/auth/me`. `auth_guard.py` provides `require_user` and `require_admin` FastAPI dependencies used by protected routes.

---

## `engine/` — Detection Engine (Server-Side)

---

### `engine/pipeline/langgraph_pipeline.py`

12-node LangGraph state machine. Nodes: `ingest → classify_question → extract_claims → shadow_ensemble → build_fsv → run_xgboost → route_to_jury → [adversarial_specialist · linguistic_auditor · domain_critic] → jury_verdict → build_response`. LangGraph lets nodes be skipped conditionally (skip jury when XGBoost confidence is very high), retried on failure, and extended without touching the whole pipeline.

---

### `engine/failure_classifier.py`

XGBoost v4 classifier. Takes a 434-feature FSV, returns `failure_type`, `failure_prob`, and SHAP values for explainability. Runs in <10ms. Falls back: v4 → v3 → v2 → rule-based heuristics.

---

### `engine/groq_service.py`

Parallel Groq model calls. Sends the same prompt to 3 models (`llama-3.3-70b-versatile`, `deepseek-r1-distill-llama-70b`, `qwen-qwq-32b`) simultaneously using `ThreadPoolExecutor`. Computes `ensemble_disagreement` from the response set. Caches responses by prompt hash for 1 hour to reduce redundant Groq API calls.

---

### `engine/agents/`

**`adversarial_specialist.py`** — runs the same 7-layer adversarial pipeline server-side (with additional FAISS semantic search against 1,000+ known attack embeddings). Returns an `AgentVerdict`.

**`linguistic_auditor.py`** — checks grammar, fluency, confidence hedges, and refusal signals in the model's output.

**`domain_critic.py`** — checks factual claims against ground truth (Wikidata + Serper).

**`failure_agent.py`** — jury orchestration. Merges 3 agent verdicts with weights based on `failure_type`. If adversarial: upweight `adversarial_specialist` 3×. If factual: upweight `domain_critic` 3×.

---

### `engine/verifier/`

4-step verification chain for factual questions: Wikidata (free SPARQL) → Serper (Google Search) → self-consistency (3× model calls) → escalate to jury. `ground_truth_pipeline.py` orchestrates the chain. `model_router.py` routes calls to Groq or Ollama based on config.

---

### `engine/fie_config.py`

Hot-reloadable configuration from MongoDB `config` collection. `get_scan_threshold()`, `update_scan_threshold()`, `get_attack_thresholds()`, `get_preflight_config()`. Changes in MongoDB propagate without a server restart. Falls back to env vars if MongoDB is unavailable.

---

### `engine/logging_config.py`

Structured JSON logging setup. `configure_logging()` replaces all handlers with a JSON formatter that outputs `timestamp`, `level`, `logger`, `message`, and `rid` (correlation ID). `bind_request_id()` stores the request ID in a `contextvars.ContextVar` so every log line in a request automatically includes it.

---

## `storage/`

### `storage/database.py`

MongoDB client with in-memory dict fallback. `initialize_vault()` tries MongoDB on startup; if it fails, sets `_fallback_mode = True`. All read/write functions check this flag and route to the fallback when needed. Analytics degrade in fallback mode but detection and correction still work.

Collections: `events`, `feedback`, `archetypes`, `evolution`, `config`.

---

## `Frontend/`

React SPA (Vite + Recharts). Dashboard with 6 KPI cards, inference feed with risk/attack filters, analytics charts, Playground page (side-by-side raw vs FIE-protected), and model health panel. Connects to the FastAPI server via `VITE_API_URL` env var.

---

## `tests/`

- **`test_core.py`** — offline unit tests for question classifier, XGBoost fallback, SDK local predictor, entropy detector, SDK config. No server, no API key needed.
- **`benchmark_reasoning.py`** — reasoning quality benchmark runner. Measures FIE's correction accuracy on math, logic, factual, and hallucination test cases.
- **`eval_datasets.py`** — multi-dataset evaluation runner with batch mode (for Groq rate limits), per-case streaming saves, resume-by-case-id, and `--combine` to merge per-dataset result files.
- **`demo.py`** — interactive CLI demo using Groq's `llama-3.1-8b-instant` (weakest model, more failures to catch) with `@monitor(mode="correct")`.

---

## `config.py` (root)

`pydantic-settings` settings class. Reads all server env vars (`MONGODB_URI`, `GROQ_API_KEY`, `JWT_SECRET_KEY`, `CORS_ALLOWED_ORIGINS`, etc.) with type validation. Single source of truth for server configuration. Used via `get_settings()` singleton.
