# Changelog

All notable changes to FIE (Failure Intelligence Engine) are documented here.

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
