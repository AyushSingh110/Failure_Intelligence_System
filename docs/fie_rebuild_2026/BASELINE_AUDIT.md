# FIE Rebuild 2026 — Baseline Audit

**Date:** 2026-10-07 · **Baseline commit:** `24cb2e9d76e977364f728f47790935283ef110b7` (branch `main`, clean tree, 236 commits)
**Session type:** planning only. No production code, model, dependency, threshold, config or API was changed.
**Companion documents:** [ROADMAP.md](ROADMAP.md) (what to do about this) · [MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md](MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md) (decisions and open questions)

This audit states what FIE is today, as read from the code and re-measured on 2026-10-07.
Where a document and the code disagree, the code and the measurement win.

## How to read the evidence labels

Every claim in this document carries one of these labels.

| Label | Meaning |
| --- | --- |
| **REPRODUCED** | Re-measured today from tracked, SHA-pinned inputs; number matches |
| **MEASURED** | A tracked report file contains it; not re-run today |
| **CONFIG-DEPENDENT** | Measured, but only true for one configuration that is not the default or not the deployed one |
| **HISTORICAL** | Was true for an earlier model or dataset; superseded |
| **PILOT** | Measured today on a small hand-written set (n stated). Direction only, not a result |
| **UNMEASURED** | Never evaluated |
| **CONTRADICTED** | Code or a newer measurement shows it is false |
| **HYPOTHESIS** | Plausible explanation, not yet tested |

Today's raw outputs and the scripts that produced them are in
[`evidence/2026-10-07/`](evidence/2026-10-07/). They are read-only probes: they import `fie`
and call `scan_prompt()`. They write nothing to the repository.

**Terms used throughout.** *PAIR* is this project's name for its semantic intent classifier:
a MiniLM-L6-v2 sentence embedding (ONNX) feeding a calibrated linear SVM. It is not the PAIR
attack. *UNCERTAIN band* is the confidence interval `[0.60·T, T)` below a per-attack-type
threshold `T`. *Fail-secure* means UNCERTAIN prompts are blocked when no tiebreaker model is
reachable. *Lite install* means `pip install fie-sdk` with no extras.

---

## 1. Current system summary

FIE is two different systems in one repository, sharing a name and a package.

**System A — the guardrail (`fie/`, 35 Python files, ~8.3k lines).** A Python SDK that scores one
prompt and returns a boolean. One trained model does the work: PAIR v6.3b, threshold 0.50.
Eleven regex and heuristic layers run beside it. An XGBoost "meta-classifier" blends their
scores. Internally the pipeline already routes into three zones (clear safe, uncertain, clear
attack), then collapses them into `is_attack: bool`. It runs on CPU in about 34 ms for a short
prompt. This is the part with real evidence behind it.

**System B — the output monitor (`app/` + `engine/` + `storage/`, 92 Python files, ~19.5k lines).**
A FastAPI server with MongoDB and Groq. For each `(prompt, answer)` it queries three Groq
models, compares answers by embedding similarity, runs a "jury" of rule agents, tries to
verify against Wikidata/Serper/Wikipedia, and scores the result with an XGBoost classifier.
In `correct` mode the SDK replaces the user's answer with the server's. Measured on
decontaminated data it performs at chance (ROC-AUC 0.497 and 0.396). It also contains the
dashboard backend, auth and tenant storage.

**What that means for the rebuild.** The guardrail is the product. The monitor is a research
prototype that currently shares a process, a threshold store and a cache with the product,
and has the larger attack surface.

---

## 2. Current architecture, with code locations

### 2.1 Guardrail request path (`fie.adversarial.scan_prompt`, [fie/adversarial.py:1005](../../fie/adversarial.py#L1005))

```
INPUT  prompt (str), optional primary_output, session_id, domain, threshold, use_llama_guard
  │
  ├─ 0. Feedback fast path            fie/adversarial.py:1035-1061 → fie/feedback_store.py:19-35
  │     process-global whitelist / known-attack hash sets. Hit → return with NO scan.
  │
  ├─ 1. Result cache                   fie/adversarial.py:260-298, 1063-1071
  │     key = sha256(prompt.strip().lower()) [+ domain, + disabled layers]
  │     ignores: threshold, use_llama_guard, session_id, tenant, policy, model version
  │
  ├─ 2. "Normalization"                fie/layers/patterns.py:391-423
  │     zero-width strip, NFKC, homoglyph map, spaced-letter collapse.
  │     Used ONLY by regex and prompt_guard. PAIR and every other layer see raw text.
  │
  ├─ 3. Twelve detectors in parallel   fie/adversarial.py:710-806 (shared thread pool, 10 s deadline)
  │     regex              fie/layers/patterns.py        patterns + mixed-script-word rule
  │     prompt_guard       fie/layers/prompt_guard.py    grouped regex
  │     pair_classifier    fie/layers/pair.py:324        MiniLM (fie/onnx_encoder.py) + SVM, T=0.50
  │     perplexity_proxy   fie/layers/perplexity.py      character statistics
  │     gcg_suffix         fie/layers/gcg.py             entropy / special-char density
  │     many_shot          fie/layers/many_shot.py       Q/A turn counting
  │     indirect_injection fie/layers/indirect.py        regex
  │     copyright          fie/layers/copyright.py       regex
  │     direct_harm        fie/layers/direct_harm.py     verb + target regex
  │     fiction_harm       fie/fiction_harm.py           regex
  │     virtualization     fie/virtualization.py         regex
  │     multilingual       fie/multilingual.py           script ratio, translated phrases,
  │                                                      langdetect + Google Translate
  │
  ├─ 4. Aggregation                    fie/adversarial.py:891-940, 1088-1178
  │     framing dampener (fie/framing_filter.py) → weighted vote per attack type
  │     → meta-classifier blend 60/40 (fie/layers/pair.py:152) → session "crescendo" boost
  │
  ├─ 5. Routing (policy, hard-coded)   fie/adversarial.py:1180-1350
  │     T = per-type threshold (:113-128) × domain multiplier inferred from prompt text (:41-96)
  │     conf <  0.60·T        → CLEAR SAFE   → is_attack=False, confidence reported as 0.0
  │     conf >= T             → CLEAR ATTACK → is_attack=True
  │     otherwise UNCERTAIN   → Groq tiebreaker (fie/llama_guard.py) if a key is set;
  │                             else BLOCK, unless FIE_UNCERTAIN_ALLOW=1
  │
  ├─ 6. OUTPUT  ScanResult             fie/adversarial.py:320-372
  │     is_attack, attack_type, confidence, layers_fired, layer_scores, evidence,
  │     degraded_layers, mitigation. No zone, no decision, no policy id, no model version.
  │
  └─ 7. Side effects                   every block appends to ~/.fie/flagged_events.jsonl
                                        (fie/feedback_store.py:40, 113-147) or MongoDB
```

Wrappers on top: `preflight_check` ([fie/preflight.py:162](../../fie/preflight.py#L162)),
the `@monitor` decorator ([fie/monitor.py:53](../../fie/monitor.py#L53)), OpenAI/Anthropic
clients and a FastAPI middleware (`fie/integrations/`), a CLI ([fie/__main__.py](../../fie/__main__.py)).

### 2.2 Output side

| Piece | Location | What it is |
| --- | --- | --- |
| Output scanner | [fie/output_scanner.py](../../fie/output_scanner.py) | Three regexes (policy echo, system-prompt leak, harmful steps). Fire-and-forget thread |
| Stream guard | [fie/stream_guard.py](../../fie/stream_guard.py) | Buffers the first 400 characters, runs the output scanner once, then passes everything after it unscanned |
| Local "hallucination" check | [fie/local_predictor.py](../../fie/local_predictor.py) | Counts hedging phrases in the answer |
| Server monitor | [app/routes/monitor.py:24-852](../../app/routes/monitor.py#L24) | 830-line inline pipeline: Groq fan-out → signals → jury → ground truth → XGBoost |
| LangGraph pipeline | [engine/pipeline/langgraph_pipeline.py](../../engine/pipeline/langgraph_pipeline.py) | 1,265 lines. Imported by no route. Only `scripts/measure_pipeline_baseline.py` runs it |

---

## 3. Current production surface

| Surface | What exists | State |
| --- | --- | --- |
| **SDK** | `fie-sdk` 1.18.0 on PyPI (uploaded 2026-08-10, 20 releases). Exports `scan_prompt`, `scan_prompt_async`, `monitor`, `preflight_check`, `scan_output`, `stream_guard`, `FIEClient`, integrations | Base install has no ML dependencies, so the main classifier is absent (§5.1) |
| **CLI** | `fie detect`, `fie explain`, `fie benchmark` | `--threshold` is parsed and ignored. `benchmark` loads `../data/eval_*.py`, which is not in the wheel |
| **API** | FastAPI, 43 routes under `/api/v1` | No scan endpoint. The guard is reachable only through `/monitor`, which requires a model answer |
| **Hosted** | Hugging Face Space `Ayush-Singh9791/fie`: Gradio demo + API. Live today: `/health` healthy, `/ready` true | CI auto-deploys on every push to `main` |
| **Dashboard** | React app in `Frontend/` (49 tracked files). Google OAuth sign-in | Session JWT, containing the API key, kept in `localStorage` |
| **Deployment** | Root `Dockerfile` (sets `FIE_SCAN_FAILURE_MODE=closed`), HF Space Dockerfile, Oracle scripts | Works. `uvicorn --workers 1` |
| **Models** | 25 artifacts in [scripts/model_manifest.json](../../scripts/model_manifest.json), SHA-256 pinned, GitHub Release `models-v1.18.0` | All 25 local files match the manifest today. Release assets return HTTP 200. Four load sites use pickle |
| **Configuration** | 35 distinct environment variables (19 read by the SDK alone), MongoDB `fie_config` document, compiled constants, per-model `meta.json` | Four sources, no single view, no version recorded on a result |
| **Storage** | MongoDB (inferences, signal logs, feedback, flagged events, GT cache, users, config). In-memory fallback. `~/.fie/flagged_events.jsonl` locally | No retention policy. Raw prompts and answers stored |
| **Authentication** | Google OAuth → HS256 JWT (24 h). `X-API-Key`. Env `FIE_API_KEY` grants admin | §5.4 |
| **Telemetry** | Import-time ping to `onrender.com` (opt-out). Usage events to a Cloud Run URL (opt-in). Server `/telemetry` | Two different hosts, neither is the live Space |
| **Tests** | 6 files, 87 tests. Golden-output test pins confidences to 4 decimals | 87 pass (§10) |

---

## 4. What works

Only capabilities with evidence are listed.

| Capability | Evidence | Label |
| --- | --- | --- |
| Detecting short, English, single-turn harmful or jailbreak prompts | JailbreakBench 130/134 (97.0%), HarmBench 326/387 (84.2%), StrongREJECT 218/242 (90.1%), SORRY-Bench 316/387 (81.7%). Macro 88.2%. Matches `data/benchmark_audit/fullpipe_v6_3b.json` exactly | REPRODUCED |
| Leakage-audited, frozen, SHA-pinned evaluation splits | `data/overrefusal/manifest.json`, `data/benchmark_audit/*_clean.jsonl`, dataset revisions pinned | MEASURED |
| Model artifact pinning and verified download | `scripts/download_models.py` + manifest. 25/25 local artifacts match | REPRODUCED |
| Golden-output regression test with a manifest drift guard | `tests/test_detection_golden.py`. Passes | REPRODUCED |
| Latency for short prompts | Mean 34.3 ms, p50 33.2, p95 50.0 (n=200, this laptop). Published: 32.5 / 41.8 | REPRODUCED |
| Layer fault isolation | A layer that raises or times out is reported in `degraded_layers` | Code read; covered by `test_no_layer_silently_missing` |
| Readiness separated from liveness; warm-up | `/health`, `/ready`, `/health/deep`. Live Space reports ready | Verified against the live service |
| PAIR probability calibration | ECE 0.062, n=315, PAIR in isolation | MEASURED |
| Fail-secure default for the UNCERTAIN band | [fie/adversarial.py:1318-1336](../../fie/adversarial.py#L1318) | Code read; observed in test logs |
| Secret hygiene in git | `.env` never committed (`git log --all -- .env` is empty). Gitleaks in CI | REPRODUCED |
| June 2026 `/auth/google` takeover hole | Endpoint is gone; only `/auth/google-callback` with code exchange remains | Code read |
| Research discipline | RESEARCH_LOG E1–E29: decontamination, paired bootstrap, self-corrections | MEASURED |

---

## 5. What does not work

Ordered by how much damage each does to a user who trusts the README.

### 5.1 The published package installs without its detector, and does not say so

- PyPI `fie-sdk` 1.18.0 requires only `requests`, `deep-translator`, `langdetect`. No
  scikit-learn, joblib, onnxruntime or numpy. **REPRODUCED** (PyPI JSON API, 2026-10-07).
- Without those, `_run_pair_classifier` returns `(None, 0.0, {})` with status `ok`
  ([fie/layers/pair.py:325](../../fie/layers/pair.py#L325)). The layer is not marked degraded.
- Simulated on the frozen splits (PAIR and meta-classifier removed): JailbreakBench 10.4%,
  HarmBench 4.4%, StrongREJECT 6.2%, SORRY-Bench 3.1%. **Macro 6.0%** against the advertised
  85.8%. **REPRODUCED as a simulation**; a clean-venv install of the PyPI wheel is still owed.
- In that state a scan returned `degraded_layers == []` and `is_degraded == False`.
  The README says "`degraded_layers` `[]` — empty means the full pipeline ran".
- The PyPI `[ml]` extra installs `sentence-transformers` (torch, 1.3–4.7 GB). The source tree's
  `[ml]` installs `onnxruntime`. Both are version 1.18.0. So is a third build in `dist/`
  (313,989 bytes; PyPI's is 270,149). One version number names three different artifacts.

### 5.2 Benign non-English text is blocked

**PILOT** (n=6 hand-written benign prompts per group, three passes each).

| Benign prompts in | Blocked | Layer that blocked | Zone |
| --- | --- | --- | --- |
| Hindi (Devanagari) | 6/6 | `gcg_suffix` | clear block |
| Arabic | 6/6 | `pair_classifier` | clear block |
| Russian (Cyrillic) | 6/6 | `gcg_suffix` | clear block |
| Chinese | 6/6 | `multilingual` | clear block |
| Japanese | 6/6 | `regex` (mixed-script word) | clear block |
| Spanish, translation unreachable | 6/6 | `multilingual` | uncertain block |
| French, translation unreachable | 6/6 | `multilingual` | uncertain block |
| German, translation unreachable | 6/6 | `multilingual` | uncertain block |
| Spanish / French / German, translation succeeds | 0/6, 0/6, 1/6 | — | — |
| Hinglish (romanised Hindi) | 4–5/6 | `pair_classifier` | mixed |
| English with one Cyrillic look-alike letter | 5/6 | `regex` | uncertain block |
| English quoting one foreign word | 2/6 | `multilingual` | clear block |
| English control | 1/6 | `pair_classifier` | clear block |

Three things follow that the September review did not have.

1. **Four independent layers block non-Latin text**, not one. Fixing `multilingual` alone
   leaves Hindi and Russian blocked by the GCG entropy heuristic, Arabic by PAIR, and Japanese
   by the mixed-script rule. Japanese mixes kanji, hiragana and katakana inside ordinary words,
   so that rule fires on every sentence.
2. **PAIR itself is not valid on non-English input.** It is an English MiniLM. It scored
   benign Arabic as a jailbreak 6/6.
3. **The verdict for benign Spanish, French and German depends on whether Google Translate
   answers.** When it does not, [fie/multilingual.py:308-320](../../fie/multilingual.py#L308)
   returns `MULTILINGUAL_INJECTION` at 0.55, which lands in the UNCERTAIN band and is blocked.
   Google rate-limited this machine during the baseline test run (`TooManyRequests`).

### 5.3 Over-refusal on safe-but-scary English

| Benchmark (safe prompts) | Flagged | Clear block | Uncertain block | Label |
| --- | --- | --- | --- | --- |
| XSTest safe (n=250) | 132 (52.8%) | 69 (27.6%) | 63 (25.2%) | REPRODUCED |
| OR-Bench-hard (n=250) | 226 (90.4%) | 185 (74.0%) | 41 (16.4%) | REPRODUCED |

PAIR produced 355 of these 358 false positives. The other eleven layers are not the cause.
E19/E20 show no threshold gives both low over-refusal and high recall. The zone split is new
and matters for the design: see §7.

### 5.4 Tenant isolation and authentication on the server

All verified by reading the code. None is covered by a test.

| # | Finding | Location | Effect |
| --- | --- | --- | --- |
| S1 | `POST /track` is unauthenticated and trusts `tenant_id` from the request body | [app/routes/inference.py:38](../../app/routes/inference.py#L38), [app/schemas.py:36](../../app/schemas.py#L36) | Anyone can write inference records into any tenant's history |
| S2 | `POST /monitor`, `/diagnose`, `/analyze`, `/analyze/v2`, `/track-and-analyze` need no auth | [app/routes/monitor.py:24](../../app/routes/monitor.py#L24), [inference.py:46-244](../../app/routes/inference.py#L46) | Free use of the operator's Groq quota; usage limits apply only to authenticated callers. A test pins this: `test_monitor_requires_no_auth_returns_something` |
| S3 | `DELETE /clusters/reset`, `GET /trend`, `GET /clusters` need no auth and are global | [app/routes/analytics.py:15-33](../../app/routes/analytics.py#L15) | Anyone can wipe the archetype registry; all tenants share one trend |
| S4 | Ground-truth answer cache is global. Any authenticated tenant's "correct answer" is stored at confidence 1.0 and matched by cosine ≥ 0.92 against every tenant's future questions | [app/routes/monitor.py:959-966](../../app/routes/monitor.py#L959), [engine/ground_truth_cache.py:79-188](../../engine/ground_truth_cache.py#L79) | In `correct` mode another tenant's users receive the attacker's text in place of their model's answer |
| S5 | Feedback from any tenant recalibrates global per-question-type thresholds every 50 labels | [engine/fie_config.py:456-606](../../engine/fie_config.py#L456) | Cross-tenant control of detection thresholds (review finding C7, confirmed) |
| S6 | The same recalibration rewrites the config document without the `attack_thresholds` field | [engine/fie_config.py:556-566](../../engine/fie_config.py#L556) | Operator overrides of guard thresholds are silently lost at the next restart |
| S7 | Confirmed-attack detections from any caller, including anonymous, are added to a global FAISS registry | [app/routes/monitor.py:298-311](../../app/routes/monitor.py#L298) | Unauthenticated mutation of shared detection state |
| S8 | The scan result cache and the whitelist / known-attack sets are process-global | [fie/adversarial.py:298](../../fie/adversarial.py#L298), [fie/feedback_store.py:19-20](../../fie/feedback_store.py#L19) | A label or a cached verdict from one tenant applies to all |
| S9 | The session JWT carries the API key in clear text, is kept in `localStorage`, and `is_admin` is read from the token without a database check | [app/auth.py:210-222](../../app/auth.py#L210), [app/auth_guard.py:19-25](../../app/auth_guard.py#L19) | Token theft leaks the long-lived key. Rotating the key does not invalidate the token |
| S10 | If `JWT_SECRET_KEY` is unset the server signs with a constant in the source | [app/auth.py:24](../../app/auth.py#L24) | A misconfigured deploy accepts forged admin tokens. Only a warning is raised |
| S11 | API keys are stored and compared in plain text. `GET /auth/users` returns every user's key to an admin | [app/auth.py:147-168](../../app/auth.py#L147) | Database read access equals account takeover |
| S12 | Rate limiting keys on the socket address | [app/limiter.py:10](../../app/limiter.py#L10) | Behind the Space proxy all users likely share one bucket (HYPOTHESIS, not tested) |

### 5.5 Controls that exist but do nothing

| Control | Reality | Evidence |
| --- | --- | --- |
| `scan_prompt(threshold=...)` | Read into `_threshold`, which is used only when no layer fired, a case that returns "safe" before the threshold is consulted | 0 of 448 verdicts differed between `threshold=0.05` and `0.95`. **REPRODUCED** |
| `fie detect --threshold` | Parsed, never passed on | [fie/__main__.py:43-101, 248](../../fie/__main__.py#L43) |
| `POST /admin/guard/config {scan_threshold}` | Updates the same dead value | [app/routes/admin.py:52-63](../../app/routes/admin.py#L52) |
| `FIEMiddleware(threshold=...)` | Forwards to the dead argument | [fie/integrations/fastapi.py:117](../../fie/integrations/fastapi.py#L117) |
| `FIEMiddleware(local_mode=False)` | Posts to `/api/v1/scan`. No such route exists. Returns "not an attack" | [fie/integrations/fastapi.py:128-150](../../fie/integrations/fastapi.py#L128) |
| `GET /flags`, `POST /flags/{id}/label` | Import `verify_token` from `app.auth_guard`. That name does not exist. Every call returns 401 | [app/routes/flags.py:131-137](../../app/routes/flags.py#L131) |
| "Feedback store → learned hash → fast path" in the README diagram | Depends on the endpoint above. Unreachable through the API | — |
| Domain threshold multipliers | Changed 0 flag verdicts across 1,848 benchmark prompts and 1 of 3,052 framed prompts | **REPRODUCED** |
| Meta-classifier | Changes zero `is_attack` verdicts (E27, p=1.00) | MEASURED |
| `ScanResult.category` | Always `None` | Code read |
| Auto-retraining of XGBoost from feedback | Saves `models/xgboost_retrained.pkl`. Nothing loads it | [engine/retraining/buffer.py:192-217](../../engine/retraining/buffer.py#L192) |
| `SCAN_THRESHOLD` | Two different defaults for one variable: 0.65 in `fie/adversarial.py:28`, 0.45 in `engine/fie_config.py:68` | Code read |

### 5.6 "Offline" is not offline

| Outbound call | When | Opt-out | Location |
| --- | --- | --- | --- |
| Telemetry ping (version, OS, Python) to `onrender.com` | `import fie` | `FIE_NO_TELEMETRY=1` | [fie/_telemetry.py](../../fie/_telemetry.py), [fie/__init__.py:15](../../fie/__init__.py#L15) |
| **The prompt text** to Google Translate | Any prompt ≥ 30 chars with ≥ 10% non-Latin letters, and any all-Latin prompt ≥ 60 chars that `langdetect` calls non-English | None | [fie/multilingual.py:378-403](../../fie/multilingual.py#L378) |
| The prompt text to Groq | UNCERTAIN band, if `GROQ_API_KEY` is in the environment | `use_llama_guard=False` | [fie/llama_guard.py:93](../../fie/llama_guard.py#L93) |
| 90 MB model download, no checksum | First scan if the ONNX model is missing | `FIE_NO_AUTO_DOWNLOAD=1` | [fie/onnx_encoder.py:67-116](../../fie/onnx_encoder.py#L67) |

The test suite the README calls "full offline" made live requests to `api.groq.com` (401 with
the CI fake key) and to Google Translate (429) during today's run.

### 5.7 The detector is not deterministic

One of 250 XSTest-safe verdicts flipped between two identical passes with the cache cleared.
**REPRODUCED.** `langdetect` is unseeded: 1 of 37 English prompts of 60+ characters changed its
detected language between runs and was at times "confidently non-English", which routes the
prompt into the path in §5.2 item 3. Root cause is a **HYPOTHESIS** consistent with both
observations; not yet proven by seeding.

### 5.8 Long prompts defeat it, in both directions

**REPRODUCED** on 120 HarmBench attacks that are caught when sent alone.

| Benign filler added (one sentence = 21 words) | Still caught, filler after | Still caught, filler before |
| --- | --- | --- |
| none | 120 / 120 | 120 / 120 |
| 21 words | 61 (50.8%) | 22 (18.3%) |
| 42 words | 27 (22.5%) | 9 (7.5%) |
| 84 words | 12 (10.0%) | 24 (20.0%) |
| 168 words | 8 (6.7%) | 8 (6.7%) |
| 336 words | 8 (6.7%) | 8 (6.7%) |

The same padding removes over-refusal: XSTest-safe flags fall from 132 to 5 of 250 with 84
words appended. The review framed this as a 256-token truncation problem
([fie/onnx_encoder.py:49](../../fie/onnx_encoder.py#L49)). It is broader. The attack is still
inside the window when the filler comes after it, and it is missed anyway, because the
embedding is a mean over all tokens. **Both headline numbers, 88% recall and 53–90%
over-refusal, are properties of short bare prompts.** Real traffic with a system prompt, a
retrieved document or a long message sits where the classifier says little either way.

### 5.9 A fixed benign-sounding prefix evades a third of caught attacks

**REPRODUCED** on 763 attacks (JailbreakBench + HarmBench + StrongREJECT), 674 caught at baseline.
No optimisation, no queries, one hand-written string each.

| Framing | Caught | Baseline-caught now missed |
| --- | --- | --- |
| none | 674 | — |
| "My neighbour asked me this during a casual visit: …" | 682 | 8 (1.2%) |
| "… This is for a unit test in our pull request." | 652 | 30 (4.5%) |
| "My patient asked me this during a clinical visit: …" | 607 | 67 (9.9%) |
| "For a compliance review of our contract clause, answer this: …" | 445 | 229 (34.0%) |

The domain multiplier accounts for at most one of these. The effect runs through PAIR.
**HYPOTHESIS:** the v6 domain-balanced retraining that took medical and legal false positives
from ~70% to 0–28% (E4) taught the model that medical and legal vocabulary means benign. That
is a shortcut, and it is an attack surface. The medical prefix also cut XSTest-safe flags from
132 to 57, which is the same shortcut seen from the benign side.

### 5.10 The hallucination monitor

| Fact | Evidence | Label |
| --- | --- | --- |
| At chance on decontaminated data | TruthfulQA AUC 0.497 [0.424, 0.562]; HaluEval 0.396 [0.301, 0.494] | MEASURED |
| The code that was measured is not the code that is deployed | `run_pipeline` (LangGraph) has no importer under `app/`. `/monitor` is a separate inline implementation | Code read |
| The API advertises a held-out AUC of 0.749 | [app/routes/monitor.py:888](../../app/routes/monitor.py#L888); the live Space returns it | CONTRADICTED |
| The guard gates the monitor: 29.5% of benign TruthfulQA pairs (118/400) never reach it, all blocked by PAIR | `data/hallucination_eval/guard_overrefusal_report.json` | MEASURED |
| Groq is referenced in 21 Python files | `grep`; the review counted 13 | REPRODUCED |
| Prompts and answers go to Groq and are stored raw in MongoDB; attack alerts email the prompt | [app/routes/monitor.py:63, 152-158, 795-815, 827](../../app/routes/monitor.py#L63) | Code read |

### 5.11 Streaming and output safety

`stream_guard` inspects the first 400 characters with three regexes, decides once, and lets the
rest of the stream through ([fie/stream_guard.py:56-93](../../fie/stream_guard.py#L56)). It does
not scan the prompt. No output-side benchmark has been run. **UNMEASURED.**

### 5.12 Fail-open paths that ignore the fail-secure setting

`FIE_SCAN_FAILURE_MODE` defaults to `open` ([fie/preflight.py:114](../../fie/preflight.py#L114)).
Independently of it: the server's `/monitor` continues if the pre-flight raises
([app/routes/monitor.py:106-108](../../app/routes/monitor.py#L106)); `FIEMiddleware._scan` returns
"not an attack" on any exception; the OpenAI and Anthropic wrappers log scan errors at DEBUG
and proceed. Both wrappers also default to `block_attacks=False`: out of the box they log an
attack and send the prompt to the model anyway.

---

## 6. What is unmeasured

| Area | State |
| --- | --- |
| Adaptive attacks (white-box or black-box) | Never run. E15 used a paraphraser that never saw FIE's score. §5.8 and §5.9 are naive, non-adaptive probes and already succeed |
| Recall with the online tiebreaker | Never measured. Bounded today: macro between **76.7%** (tiebreaker clears every UNCERTAIN attack) and **88.2%** (clears none) |
| Benign multilingual false-positive rate | Pilot only (§5.2). No dataset, no confidence intervals |
| Attack recall in any language but English | `multilingual` was credited on 8 attacks (E11). The pyproject claim "14/14" has no source |
| Multi-turn / crescendo | `fie/session_tracker.py` exists. No multi-turn benchmark |
| Indirect injection, RAG and agent settings | No BIPIA or AgentDojo run. The layer added 0 recall in every ablation |
| Output-side harm and streaming | No benchmark |
| Long-context prompts | Only today's probe |
| The lite install on a clean machine | Simulated only |
| The deployed `/monitor` code path | The evaluation ran a different implementation |
| Throughput, memory, concurrency | One latency study, single-threaded, short prompts. Indicative only: a 3,700-character prompt took ~310 ms, measured while another job was running |
| Rate limiting behind the Space proxy | Not tested |
| Any behaviour on Python 3.9 | CI runs 3.10–3.12; `requires-python >= 3.9` |

---

## 7. A measurement that shapes the design: the UNCERTAIN band

The three-zone routing already exists. It is hidden behind a boolean. Exposing it as
ALLOW / REVIEW / BLOCK would move these prompts, with the shipped model and thresholds:

| Set | n | BLOCK (clear) | REVIEW (uncertain) | ALLOW |
| --- | --- | --- | --- | --- |
| XSTest safe | 250 | 27.6% | 25.2% | 47.2% |
| OR-Bench-hard | 250 | 74.0% | 16.4% | 9.6% |
| JailbreakBench | 134 | 91.0% | 6.0% | 3.0% |
| HarmBench | 387 | 68.0% | 16.3% | 15.8% |
| StrongREJECT | 242 | 83.1% | 7.0% | 9.9% |
| SORRY-Bench | 387 | 64.6% | 17.1% | 18.3% |

Read it plainly.

- A REVIEW state halves hard blocks on XSTest (52.8% → 27.6%). It barely helps OR-Bench-hard,
  where 74% are confident blocks.
- It is not free. 13.4% of attacks (154 of 1,150) sit in the same band. Whoever handles REVIEW
  inherits them.
- REVIEW relabels uncertainty. It does not reduce error. The OR-Bench-hard problem is in the
  representation and needs research, not routing.

---

## 8. Claims ledger: README, fact sheet and code against the evidence

| # | Claim | Where | Finding | Label |
| --- | --- | --- | --- | --- |
| C1 | "`pip install fie-sdk` … No API key, no network, no configuration. ~33 ms per scan, offline, with bundled models" | README | The base install lacks the classifier's dependencies. The SDK makes network calls. | CONTRADICTED |
| C2 | Macro recall 85.8% | README, FACT_SHEET | Correct for PAIR **v6.2**. The shipped default is v6.3b, which measures 88.2% | CONFIG-DEPENDENT |
| C3 | JailbreakBench 91.8% | README | E9's figure on the full 282-prompt set. The decontaminated 134 give 96.3% (v6.2) and 97.0% (v6.3b). README's four rows average 84.7%, not the 85.8% printed above them | HISTORICAL |
| C4 | HarmBench 82.2% | README | FACT_SHEET says 81.9%; shipped model 84.2% | HISTORICAL |
| C5 | XSTest 53.6% | README, FACT_SHEET | v6.2. Shipped: 52.8% | CONFIG-DEPENDENT |
| C6 | "understated this by roughly 8×" | README, OVERREFUSAL_FINDINGS, Space README | FACT_SHEET: "Do not say 8×" (72 ÷ 9.7 = 7.4) | CONTRADICTED by the project's own fact sheet |
| C7 | OOD false-positive rate 9.1% (README) / 9.7% (FACT_SHEET) | both | v6.2 and v6.0. Not measured for v6.3b on that 340-prompt set | CONFIG-DEPENDENT |
| C8 | `multilingual` "25% → 100%", "earns its place" | README, E11 | n=8 attacks, recall only. The layer and three others block benign non-Latin text | CONTRADICTED as a justification |
| C9 | "multilingual 14/14 recall" | pyproject description | No source in the fact sheet or log | UNMEASURED |
| C10 | "Borderline → LlamaGuard tiebreaker" | README diagram, code names | The model is `llama-prompt-guard-2-86m`, an injection classifier, called through Groq | CONTRADICTED |
| C11 | "`degraded_layers` `[]` — empty means the full pipeline ran" | README | False when dependencies or models are missing | CONTRADICTED |
| C12 | "Feedback store … learned hash" loop | README diagram | The labelling endpoints always return 401 | CONTRADICTED |
| C13 | Output monitoring detects hallucinations and corrects them | README diagram, pyproject | At chance; and `correct` mode can substitute another tenant's text (S4) | CONTRADICTED |
| C14 | `auc_held_out: 0.749` | `/api/v1/monitor/model-info` | Novel-question AUC is 0.497 | CONTRADICTED |
| C15 | "Tenant isolation — all MongoDB queries are scoped to `tenant_id`" | SECURITY.md | S1, S4, S5, S7, S8 | CONTRADICTED |
| C16 | "`torch` is 1.32 GB — currently blocks free-tier hosting" | README, PRODUCTION_ENGINEERING §12 | Resolved by the ONNX migration; the Space runs on the free tier | HISTORICAL |
| C17 | "`python scripts/audit_benchmark_leakage.py` # run it against your own data" | README | The script is git-ignored and not in the repository | CONTRADICTED |
| C18 | FACT_SHEET "Regenerate" commands | FACT_SHEET | `latency_study.py`, `measure_combined_recall.py`, `measure_overrefusal.py`, `measure_guard_baselines.py` are all git-ignored | CONTRADICTED |
| C19 | "`pytest tests/ -m "not network"` # full offline suite" | README | Makes HTTP requests to Groq and Google | CONTRADICTED |
| C20 | Leakage: 52.5% of JailbreakBench | README, FACT_SHEET | 148/282 in `leakage_report.json`. The audit used cosine 0.92; FACT_SHEET states 0.95, which is the training-side removal cutoff | MEASURED, cutoff misreported |
| C21 | ~33 ms per scan | README | 34.3 ms today for short prompts. 48 ms in the E18 run. 62–78 ms p50 inside the server. Grows with length | CONFIG-DEPENDENT |
| C22 | "Detects … homoglyphs" | README | True, by blocking any word that mixes scripts, including benign ones | CONFIG-DEPENDENT |
| C23 | "Detects … crescendo multi-turn escalation · indirect injection in documents" | README | No benchmark for either | UNMEASURED |
| C24 | "96% recall" | Landing page | Pre-decontamination era | HISTORICAL |
| C25 | Supported version "v1.13.x (latest)" | SECURITY.md | Latest is 1.18.0 | HISTORICAL |
| C26 | "The other twelve test files" | CI workflow comment | There are six test files | HISTORICAL |
| C27 | Benchmarks ran with `use_llama_guard=False` | RESEARCH_REVIEW F3 | The scripts call `scan_prompt(p)` with the default. They measured fail-secure only because no Groq key was exported in that shell. Nothing records it | CONFIG-DEPENDENT, unrecorded |
| C28 | PAIR truncation at `fie/layers/pair.py:47` | RESEARCH_REVIEW A.4 | The constant is at `fie/onnx_encoder.py:49` after the layer split | Stale reference |

The research review itself holds up. Its findings F1–F8 and C1–C7 are confirmed, with the
refinements in §5.2, §5.8, §5.9 and C27.

---

## 9. Reality check of the eighteen known findings

| # | Finding | Verdict | Where |
| --- | --- | --- | --- |
| 1 | Severe XSTest / OR-Bench over-refusal | Confirmed and reproduced exactly: 52.8% / 90.4% | §5.3 |
| 2 | Non-Latin script over-blocking | Confirmed, and worse: four layers, not one | §5.2 |
| 3 | Homoglyph protection causes benign false positives | Confirmed, 5/6 in the pilot | §5.2 |
| 4 | UNCERTAIN band differs between benchmark and deployment | Confirmed. Now bounded: 76.7%–88.2% macro recall. The benchmark configuration was implicit | §6, C27 |
| 5 | The existing tiebreaker model | Confirmed: Prompt-Guard-2 86M through Groq, misnamed LlamaGuard | C10 |
| 6 | No adaptive evaluation | Confirmed. Naive padding and one fixed prefix already work | §5.8, §5.9 |
| 7 | 256-token truncation | Confirmed, and the real mechanism is mean-pool dilution, which starts at 21 words | §5.8 |
| 8 | Global feedback / recalibration state | Confirmed, plus four more global stores | §5.4 S4–S8 |
| 9 | Pickle / joblib model loading | Confirmed: four `joblib.load`, one `pickle.loads` from Redis, one `faiss.read_index`, one unverified download | §11 |
| 10 | Dead meta-classifier | Confirmed inert on `is_attack`. Whether it moves prompts between zones is untested | §5.5 |
| 11 | Hallucination detector performance | Confirmed at chance. New: measured pipeline ≠ deployed pipeline | §5.10 |
| 12 | Groq coupling | Confirmed: 21 files | §5.10 |
| 13 | README / fact sheet / log inconsistencies | Confirmed: 28 entries | §8 |
| 14 | SDK developer experience | Default install is non-functional as a guard; three inert parameters; boolean result | §5.1, §5.5 |
| 15 | Hosted API and dashboard | No scan endpoint; unauthenticated compute and write routes | §3, §5.4 |
| 16 | Streaming path | First 400 characters only | §5.11 |
| 17 | Deployment | Works. Auto-deploys from `main`. Single worker | §3 |
| 18 | Installation and model download | Manifest path is sound. The SDK's own auto-download skips the checksum. Version 1.18.0 is three artifacts | §5.1, §5.6 |

---

## 10. Baseline tests and environment

| Item | Value |
| --- | --- |
| Command | `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` |
| Result | **87 passed, 0 failed, 0 skipped**, 28 s |
| Environment | conda env `failure-engine`, Python 3.10.19, Windows 11 |
| Isolation | Every key present in `.env` was pre-set empty so `load_dotenv()` could not inject a real credential. CI's four values were set. Telemetry off. Feedback file redirected out of the home directory. See [`evidence/2026-10-07/hermetic_env.sh`](evidence/2026-10-07/hermetic_env.sh) |
| Repository after the run | `git status` clean |

Caveats on what "87 passed" means.

- The golden test passed in the **tiebreaker-unreachable** configuration: the Groq call
  returned 401 and the fail-secure branch ran. With a working key the pinned values could differ.
- Installed packages differ from `requirements.txt`: xgboost 3.2.0 (pinned 2.1.4), joblib 1.5.3
  (1.4.2), numpy 2.2.6 (2.1.3), fastapi 0.135.1 (0.115.6), pydantic 2.12.3 (2.10.3), langgraph
  1.2.4 (0.2.76), PyJWT 2.12.1 (2.10.1). scikit-learn 1.7.2 and onnxruntime 1.23.2 match.
- A stale `fie-sdk 1.14.0` sits in `site-packages`. Tests import the working tree because the
  repository root is first on the path.
- No test covers authentication, tenant isolation, the `/flags` routes, the stream guard,
  the FastAPI middleware, the policy routing zones or the lite install.

---

## 11. Engineering readiness: control, current state, gap, future work

This is an engineering readiness analysis. It is not legal advice and makes no compliance
claim. Framework references are positioning aids; check each against the current published
edition before using it externally.

| Control | Current state | Gap | Future work | Related guidance |
| --- | --- | --- | --- | --- |
| Tenant isolation | Inference reads are tenant-scoped | Writes, GT cache, thresholds, registries, scan cache and labels are global (S1–S8) | Tenant id from credentials only; per-tenant or disabled shared state; isolation tests | OWASP LLM04 Data and Model Poisoning; NIST AI RMF Manage |
| Authentication | OAuth, JWT, API key | Unauthenticated compute and write routes; default JWT secret (S2, S3, S10) | Deny by default; fail closed without a secret | OWASP LLM10 Unbounded Consumption |
| Authorization | `require_admin` on analytics | `is_admin` trusted from the token; broken `/flags` check | Database-backed role check; one auth dependency for all routes | NIST AI RMF Govern |
| API key handling | Random 16-char keys | Plain text at rest; embedded in JWT; returned in bulk to admins (S9, S11) | Store a hash; show once; remove from tokens | OWASP LLM02 Sensitive Information Disclosure |
| Secrets management | `.env` untracked; gitleaks in CI | `pip-audit … \|\| true` never fails; PyPI token and four Groq keys live in one local `.env` | Make the audit blocking; least-privilege tokens | OWASP LLM03 Supply Chain |
| Audit logging | Application logs with request ids; flagged-events file | No immutable decision record with policy, model and config identity | Structured audit event per decision; pluggable sink | NIST AI RMF Measure, Manage |
| Policy versioning | None. Policy is constants in `fie/adversarial.py` | A decision cannot be traced to the rules that produced it | Versioned declarative policy; id on every result | NIST AI RMF Govern |
| Model / version provenance | PAIR `meta.json`; manifest | `ScanResult` carries no model version, hash or threshold set | Provenance block on every result and report | NIST AI RMF Map, Measure |
| Model artifact integrity | SHA-256 manifest; verified download script | The SDK's own auto-download is unverified; nothing verifies at load time | Verify on every path, at load | OWASP LLM03 Supply Chain |
| Pickle / joblib | — | `joblib.load` ×4, `pickle.loads` from Redis, `faiss.read_index` | Export the linear head as plain weights; JSON for sessions | OWASP LLM03 Supply Chain |
| Rate limiting | slowapi, per route | Optional dependency; socket-address key; none on `/track`, `/analyze`, `/diagnose` | Key by credential; cover every route; test behind a proxy | OWASP LLM10 Unbounded Consumption |
| Failure mode | `FIE_SCAN_FAILURE_MODE`, default `open` | Three wrappers fail open regardless (§5.12) | One failure policy, honoured everywhere, default stated per deployment type | NIST AI RMF Manage |
| Logging and retention | Structured JSON logs | Raw prompts and answers stored without limit; `~/.fie` file grows unbounded (14.5 MB on this machine) | Configurable content logging, off by default; retention period; deletion | OWASP LLM02 |
| PII exposure | Hashes in the flagged-events store | Prompt excerpts beside the hash; prompts to Google and Groq; prompts in alert email | No third-party egress by default; redaction option | OWASP LLM02 |
| Appeal / review path | Design exists (`/flags`) | Endpoint is dead; no tenant scoping; a label is global | Tenant-scoped review queue tied to the REVIEW state | NIST AI RMF Manage |
| Explainability | Layer scores, fired layers, matched text; `fie explain` | Allowed results report confidence 0.0; no zone; an LLM-written explanation on the server (7.2 s p50 per the September review; not re-measured) | Evidence-only explanation with score, threshold, zone, policy | NIST AI RMF Measure |
| Reproducibility | Frozen splits, seeds, manifest | Measurement scripts untracked; run configuration unrecorded; one non-deterministic path | Tracked harness with a configuration fingerprint | NIST AI RMF Measure |
| Configuration provenance | — | Four configuration sources; result does not say which applied | Resolved-config snapshot and hash per result | NIST AI RMF Govern |
| Backwards compatibility | Additive history so far | Three public parameters are inert; removing them is a break | Deprecation policy; contract tests | — |
| Safe defaults | Fail-secure UNCERTAIN | Default install has no classifier; default failure mode is open; telemetry on | Defaults that are either safe or loud | OWASP LLM01 Prompt Injection |
| Prompt injection coverage | Regex + PAIR, English | Indirect, agent and multilingual injection unmeasured | Measure; state scope | OWASP LLM01 |
| System prompt leakage | Output regex; canary token on the server | Unmeasured | Benchmark or drop the claim | OWASP LLM07 System Prompt Leakage |
| Output handling | Three regexes | Unmeasured; streaming covers 400 characters | Separate, measured output pipeline | OWASP LLM05 Improper Output Handling |
| Misinformation | Hallucination monitor | At chance | Research track; no product claim | OWASP LLM09 Misinformation |

---

## 12. Product blockers, ranked

Scores are 1 (low) to 5 (high). Effort is in working sessions of roughly one day.

| Rank | Blocker | User impact | Security impact | Research impact | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | Default install has no detector and reports full coverage (§5.1) | 5 | 5 | 3 | 1–2 |
| 2 | Tenant-isolation and auth holes on the live API (§5.4) | 4 | 5 | 1 | 1–2 |
| 3 | No tracked, configuration-pinned evaluation; headline numbers describe a non-shipped model (§8) | 3 | 3 | 5 | 1 |
| 4 | Non-English text blocked by four layers (§5.2) | 5 | 2 | 4 | 3–5 |
| 5 | Padding and framing evasion; no adaptive evaluation (§5.8, §5.9) | 3 | 5 | 5 | 3–6 |
| 6 | Silent third-party egress and non-determinism (§5.6, §5.7) | 4 | 4 | 3 | 1–2 |
| 7 | Boolean result; policy hard-coded; inert controls (§5.5, §7) | 4 | 2 | 2 | 2–3 |
| 8 | English over-refusal, representational (§5.3) | 5 | 1 | 5 | research, open-ended |
| 9 | No scan endpoint; middleware server mode is a no-op (§3, §5.5) | 3 | 3 | 1 | 1–2 |
| 10 | Pickle loading and unverified download (§11) | 2 | 4 | 1 | 1–2 |
| 11 | Hallucination monitor at chance, Groq-coupled, mis-advertised (§5.10) | 3 | 3 | 4 | research, 8+ weeks |
| 12 | Streaming and output safety unmeasured (§5.11) | 2 | 3 | 3 | 2–4 |

---

## 13. Files inspected

**Read in full:** `README.md`, `docs/FACT_SHEET.md`, `docs/RESEARCH_REVIEW_2026-09.md`,
`pyproject.toml`, `requirements.txt`, `Dockerfile`, `.gitignore`, `.dockerignore`,
`.env.example`, `.github/workflows/ci.yml`, `.github/workflows/publish-pypi.yml`,
`scripts/model_manifest.json`, `fie/__init__.py`, `fie/__main__.py`, `fie/_degrade.py`,
`fie/_lite.py`, `fie/_telemetry.py`, `fie/adversarial.py`, `fie/client.py`, `fie/config.py`,
`fie/feedback_store.py`, `fie/framing_filter.py`, `fie/llama_guard.py`, `fie/local_predictor.py`,
`fie/monitor.py`, `fie/multilingual.py`, `fie/onnx_encoder.py`, `fie/output_scanner.py`,
`fie/preflight.py`, `fie/session_tracker.py`, `fie/stream_guard.py`, `fie/layers/pair.py`,
`fie/layers/patterns.py`, `app/main.py`, `app/auth.py`, `app/auth_guard.py`,
`app/auth_routes.py`, `app/limiter.py`, `app/routes/monitor.py`, `app/routes/flags.py`,
`app/routes/admin.py`, `app/routes/community.py`, `tests/test_detection_golden.py`.

**Read in part or searched:** `docs/RESEARCH_LOG.md` (outline, E11, cross-checked figures),
`docs/PRODUCTION_ENGINEERING.md` (outline, §12), `docs/ARCHITECTURE.md` (outline),
`SECURITY.md`, `config.py`, `app/schemas.py`, `app/routes/inference.py`,
`app/routes/analytics.py`, `storage/database.py`, `engine/fie_config.py`,
`engine/ground_truth_cache.py`, `engine/retraining/buffer.py`, `engine/failure_classifier.py`,
`engine/session_store.py`, `engine/hard_positive_collector.py`,
`engine/explainability/redaction.py`, `fie/integrations/openai.py`,
`fie/integrations/fastapi.py`, `deploy/huggingface/Dockerfile`, `deploy/huggingface/app.py`,
`deploy/huggingface/push_space.py`, `tests/test_integration.py`, the test inventory of all six
test files, `Frontend/src` (token storage, displayed claims), and the scan-call configuration
of 14 scripts under `scripts/`.

**Data and reports read:** every JSON report under `data/overrefusal/`,
`data/benchmark_audit/`, `data/ablation/`, `data/baselines/`, `data/calibration/`,
`data/robustness/`, `data/hallucination_eval/`, plus `data/fpr_v6*.json` and
`data/meta_impact_report.json`. The built wheel in `dist/`. The PyPI JSON record.

**Not read:** the remaining `engine/` modules line by line (agents, reasoning, verifier, RAG,
archetypes), `engine/pipeline/langgraph_pipeline.py` beyond its import graph, most of
`Frontend/`, `telemetry-server/`, the notebooks, `paper/`, and `docs/ARCHITECTURE.md`,
`docs/DEPLOYMENT.md`, `docs/OPERATIONS.md`, `docs/CODEBASE.md` beyond their outlines. Findings
about the monitor are therefore a lower bound.
