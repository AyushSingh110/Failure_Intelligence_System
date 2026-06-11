# Changelog

All notable changes to FIE (Failure Intelligence Engine) are documented here.

---

## [Unreleased] — production-hardening pass (2026-06-11)

### Security

- **Removed the unverified `POST /api/v1/auth/google` endpoint.** It accepted
  a raw `{email, name}` body with no token verification and returned a session
  JWT + API key for any address — including the admin's. Login now goes
  exclusively through the Google OAuth code flow (`/auth/google-callback`).
  The dashboard already used the OAuth flow; only the dead `loginGoogle`
  helper in `Frontend/src/lib/api.js` was removed.
  `tests/stress_test_suite.py` now authenticates via `FIE_API_KEY` (X-API-Key
  header) instead.
- API keys are no longer written to application logs on user creation
  (`app/auth.py`).
- All `/auth/*` endpoints are rate-limited per client IP (slowapi; no-op when
  slowapi is not installed): `google-callback` 10/min, `me` 60/min,
  `users` 30/min, `regenerate-key` 5/min.

### Fixed

- **PAIR v4 was bundled but never loaded.** `fie/adversarial.py` preferred
  v3 > v2 > v1 — the wheel shipped `pair_intent_classifier_v4.pkl` while the
  runtime silently used v3. The preference chain is now v4 > v3 > v2 > v1, so
  detection behavior matches the published v4 results (natural 0.50 threshold).
- **Version drift.** README said 1.13.0, `pyproject.toml` 1.12.0,
  `fie/__init__.py` 1.10.1, `fie/client.py` 1.4.1, server `config.py` 3.0.0.
  `pyproject.toml` is now the single source of truth (1.13.0); all other
  locations resolve it from package metadata at runtime.
- `requirements.txt` was missing runtime dependencies that CI hand-installed
  (`xgboost`, `scikit-learn`, `joblib`, `pandas`, `deep-translator`,
  `langdetect`) — a fresh clone now gets a fully functional server from
  `pip install -r requirements.txt`.
- `app/auth.py` created a new `MongoClient` on every auth lookup, bypassing
  connection pooling; it now reuses one module-level client.

### Added

- **Model artifact distribution** (`scripts/download_models.py` +
  `scripts/model_manifest.json`): trained models and the FAISS index are
  distributed as GitHub Release assets pinned by SHA-256. Wired into the
  Dockerfile and CI (best-effort) and into `publish-pypi.yml` in `--strict`
  mode — a wheel without bundled models can no longer be published.
  Procedure: `docs/OPERATIONS.md`.
- `docs/OPERATIONS.md` — runbook for model releases, SDK releases, UptimeRobot
  uptime monitoring, Sentry error tracking, Codecov, Cloud Run secrets
  hygiene, git history cleanup, and a Hugging Face Hub mirror.
- Opt-in Sentry error tracking in `app/main.py` (active only when
  `SENTRY_DSN` is set; `send_default_pii=False` hard-coded).
- Coverage reporting in CI (`pytest --cov` → Codecov upload, never fails the
  build).
- `.env.example` and `Frontend/.env.local.example` (CONTRIBUTING referenced
  them, but they did not exist).
- `docs/ARCHITECTURE.md` and `docs/CODEBASE.md` are now tracked — the README
  architecture link previously 404'd for anyone cloning the repo.

### Changed

- README and CONTRIBUTING now state explicitly that the `evaluation/` harness
  is private (red-team datasets) and how to request access, instead of
  pointing at a directory that is not in the repo.
- Untracked from git: root-level `node_modules/` (29 files), accidental root
  `package.json`/`package-lock.json`, and `.DS_Store`.

---

## [1.13.0] — 2026-06-09

### Added

#### Layer 3d — Cross-lingual romanisation detector

- `engine/agents/adversarial/multilingual_romanisation.py`: dedicated n-gram fingerprint detector for five romanised scripts — no external library dependency
- Five script scorers: `_score_pinyin`, `_score_arabizi` (digit-as-letter substitution), `_score_romaji`, `_score_korean`, `_score_iast`
- `_HARM_VOCAB_RE`: harm-vocabulary regex covering romanised harmful terms across all five scripts
- Public API: `run_romanisation_detection(prompt) → (root_cause | None, confidence, evidence)`
- Confidence range: 0.42–0.72 for script signal alone; +0.15 harm-vocab boost; hard cap 0.87
- Skip condition: `non_ascii_ratio > 0.35` — prompts already handled upstream by non-ASCII detectors
- Smoke test: 93% hit rate on first 30 multilingual bench prompts, 0 false positives on benign English
- Closes the Pinyin detection gap documented in v1.12.0 Known Limitations

#### UnknownBench-v3 extended to 200 prompts per category

- All four v3 datasets extended from 39–47 → 200 prompts each via `scripts/extend_benchmarks.py`
- Groq-powered generation (llama-3.3-70b-versatile) with per-family system prompts that preserve attack strategy (framing, mechanism, romanisation script) — only topic and phrasing varies
- 800 novel held-out prompts total across four structural attack families

### Fixed

- **Encoder lazy-load bug** (`engine/encoder.py`): `SentenceEncoder.available` now calls `_get_model()` before returning
  - Root cause: `_loaded` is only set inside `_get_model()`. Without this call, `available` always returned `False` on cold start even when `sentence-transformers` was installed and functional
  - Effect before fix: `consistency.py`, `ensemble.py`, and `embedding.py` all silently fell back to TF-IDF or exact string matching on every request
- **Removed silent encoder fallback catches** in `engine/detector/consistency.py`, `engine/detector/ensemble.py`, `engine/detector/embedding.py`
  - Broad `except Exception` blocks were swallowing encoder load failures and masking the bug above
  - Encoder errors now propagate correctly — no more invisible degradation

### Changed

- `engine/agents/adversarial/specialist.py`: Layer 3d wired into the priority detection chain; `CROSS_LINGUAL_ROMANISATION_ATTACK` added to mitigation map
- Multilingual Tier 2.5 (`langdetect`) is now a complement to Layer 3d, not the sole romanisation path
- README Known Limitations: "Romanised Pinyin" gap removed (closed by Layer 3d)
- UnknownBench-v3 benchmark counts updated in README table: 47/46/45/39 → 200 each
- `pyproject.toml`: version `1.12.0` → `1.13.0`

---

## [1.12.0] — 2026-06-06

### Added

**PAIR v4 semantic classifier**

- Retrained with 3× hard-positive weighting (Experiment 8 finding: 3× strictly outperforms 5×)
- 97.18% TPR on UnknownBench-v3 (first contact, never trained on)
- 98.5% TPR on UnknownBench-v2 (held-out validation)
- Threshold drops from 0.80 (manual override) to 0.50 (natural calibration) — better model honesty
- Training corpus: 789 examples (180 hard positives, 10 hard negatives, 39 inline academic hard negatives)
- Ships as `fie/models/pair_intent_classifier_v4.pkl` + `pair_intent_meta_v4.json`

**UnknownBench-v3 — 177 novel attack prompts**

- `unknown_virtualization_bench_v3.py` (47 prompts): Epistemic reframing — attacks framed as meta-level knowledge queries, not scenario construction
- `unknown_indirect_bench_v3.py` (46 prompts): Context poisoning — false conversational precedents before the harmful request
- `unknown_multilingual_bench_v3.py` (45 prompts): Cross-lingual Romanisation — Pinyin, Arabizi, Romaji, IAST Hindi, Esperanto, code-switching
- `unknown_many_shot_bench_v3.py` (39 prompts): Socratic consistency traps — logical entailment structures toward harmful conclusions
- All 12 benchmarks (v1+v2+v3) re-frozen with SHA-256 manifests via `evaluation/datasets/freeze_benchmarks.py`

**FSV feature ablation (SHAP)**

- `evaluation/ablation/run_fsv_ablation.py`: SHAP analysis of XGBoost failure classifier
- Finding: 10 of 560 features explain 100% of predictive performance (F1 = 0.8963 at top-10 vs 0.8960 full)
- Top features: `agreement_score`, `jury_verdict_FACTUAL_HALLUCINATION`, `jury_confidence`, `entropy_score`, `high_failure_risk`
- Knee curve saved to `evaluation/ablation/results/fsv_ablation/fsv_ablation_curve.png`

**TruthfulQA hallucination evaluation harness**

- `evaluation/hallucination/run_eval.py`: Complete harness for 817-question TruthfulQA evaluation
- Resume-safe JSONL cache at `evaluation/hallucination/results/raw_responses.jsonl`
- Two experiments: Exp H1 (Full FSV + XGBoost) vs Exp H2 (ensemble disagreement only)
- Labeling: substring containment check + ROUGE-1 recall on correct answers only
- Built-in XGBoost probability threshold sweep (0.10–0.50) in the report output
- Shadow ensemble: `llama-3.3-70b-versatile` + `deepseek-r1-distill-llama-70b` + `qwen-qwq-32b`

**Hard-positive collection pipeline**

- `engine/hard_positive_collector.py`: Stages blocked prompts (full text) for human review
- `FIE_COLLECT_HARD_POSITIVES=1` opt-in env var — disabled by default
- UNCERTAIN-zone blocks now enter the feedback review queue (previously invisible)
- CLEAR_ATTACK blocks now also staged with full prompt text for PAIR retraining
- `POST /flags/{id}/label` with `true_positive` → `confirm_hard_positive(event_id)`
- `GET /flags/hard-positives/stats` — staged/confirmed counts
- `GET /flags/hard-positives/export` — download confirmed prompts for next retraining run
- Storage: `data/hard_positive_candidates.jsonl` + `data/hard_positives_confirmed.jsonl`

**Multilingual Tier 2.5 — Romanised script detection**

- `langdetect` language detection for all-Latin prompts (new dependency)
- When non-English detected: translate → re-run Tier 2 phrase patterns on English translation
- Closes the Romanised injection gap identified in UnknownBench-v3 multilingual bench
- Arabizi (Romanised Arabic) now enters UNCERTAIN zone (conf=0.58) → routes to LlamaGuard
- Pinyin remains a documented limitation (langdetect cannot distinguish Pinyin from Latin syllables)

### Changed

- `pyproject.toml`: version `1.11.0` → `1.12.0`
- `pyproject.toml`: added `langdetect>=1.0.9` to core dependencies
- `pyproject.toml`: artifacts updated — PAIR v4 model replaces v2 as the shipped artifact
- `fie/adversarial.py`: UNCERTAIN-zone blocks now call `_fb_record` (feedback event creation)
- `fie/adversarial.py`: CLEAR_ATTACK blocks now capture `event_id` for hard-positive staging
- `app/routes/flags.py`: `POST /flags/{id}/label` now calls `confirm_hard_positive` / `dismiss_candidate`
- `scripts/retrain_pair_v4.py`: production training script with Exp 8 weighting

### Fixed

- UNCERTAIN-zone blocks were invisible to the human review queue — now recorded as feedback events
- PAIR v4 threshold calibration: 3× weighting no longer requires manual threshold override

---

## [1.11.0] — 2026-05-20

### Added

- **PAIR v3** — retrained on 169 hard-positive unknown attack prompts across 4 novel categories
- **UnknownBench-v1 + v2** — 400 novel attack prompts (200 each), SHA-256 frozen
- **GCG false positive fix** — `_is_natural_language_prose()` guard reduces FPR from 72% to 6.7%
- **Exp 7 threshold sweep** — empirical sweep across 0.50–0.90 to find operating point (t=0.80)
- **Exp 8 weight comparison** — 3× vs 5× hard-positive weighting comparison

### Changed

- PAIR v3 production threshold set to 0.80 (manual override based on sweep data)
- GCG suffix detector: LOW-range entropy signals disabled on predominantly alphabetic text

---

## [1.10.0] — 2026-04-15

### Added

- Three-zone confidence routing (CLEAR SAFE / UNCERTAIN / CLEAR ATTACK)
- LlamaGuard Tier-3 tiebreaker for UNCERTAIN zone (Groq-hosted)
- Crescendo trajectory boost (up to +0.20 based on session history)
- `FIE_UNCERTAIN_ALLOW` env var for dev/test pass-through mode

---

## [1.9.0] — 2026-03-10

### Added

- Session-aware multi-turn crescendo detection
- DiagnosticJury: 3-agent specialist panel (AdversarialSpecialist, LinguisticAuditor, DomainCritic)
- Feedback fast-path: O(1) SHA-256 hash lookup for confirmed attacks
- Flags review API (`/api/v1/flags`)

---

## [1.8.0] — 2026-02-01

### Added

- XGBoost failure classifier (v1) with Failure Signal Vector
- Shadow ensemble: 3 Groq models for hallucination disagreement scoring
- Ground truth pipeline: Wikidata + Serper cross-check
- Fix engine: retrieval-augmented correction for factual failures
