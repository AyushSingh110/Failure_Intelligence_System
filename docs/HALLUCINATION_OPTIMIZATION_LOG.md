# FIE Hallucination Pipeline — Optimization Log

Running log of the phased effort to make the hallucination-detection pipeline
faster, cheaper, and offline-capable **without losing detection quality**. Newest
first. Every entry is dated, reproducible (script + command recorded), and every
performance/accuracy claim is backed by measured before→after numbers on a frozen,
decontaminated reference set. Tradeoffs and regressions are reported openly.

Design analysis this plan executes: see the pipeline design in
[HALLUCINATION_PIPELINE.md](HALLUCINATION_PIPELINE.md).

---

## Phase 0 — Measurement harness + decontaminated detection baseline

**Date:** 2026-07-14
**Status:** harness built and validated; **authoritative baseline numbers pending your run** (commands below).
**Behavior change to the pipeline:** none (instrumentation is opt-in/inactive by default), except one **required bug fix** disclosed below.

### 1. What was built / changed

- **`engine/instrumentation.py`** (new) — opt-in, single-request metrics collector.
  Records per-request: per-node wall-clock, Groq round-trips (API vs cache) + input/output
  tokens, and external HTTP round-trips (Wikidata/Serper/Wikipedia). **Inactive by
  default** — the production `/monitor` path never activates it, so behavior/output are
  unchanged. It is a lock-guarded process global (so Groq's thread-pool fan-out is
  captured) and is therefore valid for **sequential benchmarking only**, not concurrent serving.
- **Instrumentation hooks (additive, behavior-preserving):**
  - `engine/groq_service.py` — parse `usage` tokens from each Groq response, carry them
    on `GroqModelResponse`, and record every call (API + cache replay) at the single
    choke point `_call_single_model`.
  - `engine/verifier/serper_verifier.py`, `engine/verifier/wikidata_verifier.py`,
    `engine/rag/retriever.py` — record each external HTTP call.
  - `engine/pipeline/langgraph_pipeline.py` — `_timed()` wraps every node in
    `_build_graph`; node bodies are untouched. Timing is appended to the existing
    `pipeline_trace` only while measuring.
- **`scripts/build_hallucination_eval_set.py`** (new) — builds the frozen, decontaminated
  eval set (leakage audit below).
- **`scripts/measure_pipeline_baseline.py`** (new) — drives the **real** pipeline over the
  frozen set; reports latency / calls / tokens + ROC-AUC / recall@FPR / ECE with bootstrap
  95% CIs. Resumable (append-only cache). Silences persistence (Mongo/signal-log/session/
  email) for the benchmark process only — no synthetic records leak into production data.

### 2. Leakage audit + decontamination (measured)

The XGBoost meta-classifier was trained on a pool that **includes TruthfulQA and
HaluEval** (`data/labeled/synthetic_*.jsonl`; `data/training_results.json` records
truthfulqa 849, halueval 398, mmlu 490). Evaluating on those same questions would be
inflated by train/test leakage, so we decontaminate against the training pool
(1093 unique training questions) with the same discipline as the adversarial side
(`scripts/audit_benchmark_leakage.py`): **EXACT** (normalized) + **NEAR** (SBERT cosine ≥ 0.92).

| Benchmark | Unique Q | Exact-leaked | Clean | Clean % |
|---|---|---|---|---|
| TruthfulQA | 817 | 425 | 392 | 48.0% |
| HaluEval | 10 000 | 412 | 9 588 | 95.9% |

**Frozen eval set** (`data/hallucination_eval/eval_set_frozen.jsonl`, seed 20240617):
- Source: **TruthfulQA clean split only** (see finding 4b for why HaluEval was excluded).
- 200 clean questions sampled (1 further dropped as a near-dup at cosine ≥ 0.92).
- Labeled (question, candidate, label) pairs: **400 total — 200 hallucination (label=1,
  a TruthfulQA `incorrect_answer`) + 200 correct (label=0, a TruthfulQA `correct_answer`)**.
- Decontam manifest: `data/hallucination_eval/decontam_report.json`.

This set is **frozen** — all later phases are measured against this exact file.

### 3. Findings surfaced while building the harness

**3a. Latent crash in `finalize` (fixed).** The harness immediately hit
`ValueError: Invalid format specifier` on every request that reached `finalize` on the
normal (non-guard-blocked) path. Cause: an invalid f-string at
[langgraph_pipeline.py:1113](../engine/pipeline/langgraph_pipeline.py#L1113):
```python
f"finalize: xgb_prob={xgb_prob:.4f if xgb_prob else 'N/A'} "   # format spec is not a valid conditional → always raises
```
Verified it raises for both `float` (`ValueError`) and `None` (`TypeError`) inputs. Fixed by
computing the string first (`xgb_prob_str = f"{xgb_prob:.4f}" if xgb_prob is not None else "N/A"`).
This is a **required correctness fix**, not an optimization — the detection baseline
cannot be produced without `finalize` completing. **Implication worth noting:** on the
current code, every non-guard-blocked `/monitor` request aborts inside `finalize` after the
XGBoost step, so final persistence/trace on that path was failing in production. Flagging
for your awareness; the fix is minimal and isolated.

**3b. HaluEval local file is unusable for the positive class.** `data/datasets/halueval.json`
stores `wrong_answers` as degenerate `"yes"/"no"` tokens for all 10 000 rows (0 usable
hallucinated answer strings) — the downloader (`data/download_datasets.py`) landed a yes/no
verification framing instead of the real `hallucinated_answer`. HaluEval can therefore only
supply *correct* answers, which would imbalance the set (observed: 120 pos / 240 neg on a
mixed build). We use **TruthfulQA-only** for a balanced 200/200 set (the spec allows
"TruthfulQA and/or HaluEval"). Re-enabling HaluEval cleanly is a small future task (re-download
with the correct field) if we want a second benchmark.

**3c. Adversarial guard false-positives on benign QA (observed, not yet quantified).**
In the 6-pair smoke, 2 benign TruthfulQA questions were guard-blocked as attacks (routed
straight to `finalize`, `xgb_probability=None`). These are excluded from detection metrics
and counted separately (`n_excluded_no_prob_or_blocked`). Worth quantifying on the full run —
it is an over-refusal signal on the hallucination path.

### 4. Harness validation (mechanics only — NOT the baseline)

A 6-pair Groq run confirmed the harness works end-to-end: 0 errors after the fix,
`xgb_probability` populated, persistence silenced, per-node timing + tokens + call counts
recorded, detection math runs. **These n=6 numbers are not a baseline and must not be cited.**
Indicative only: end-to-end ranged ~2.3–24 s/request (cold cache, full jury), with
`jury_deliberate`, `shadow_inference`, `reasoning_verify`, and `signal_extract` the largest
nodes, ~6 Groq calls and ~2.4k tokens/request. The first request carried a one-time
model-load cost (SBERT/FAISS) in `adversarial_guard`.

### 4c. Harness hardening (added after a flaky-network run)

A first attempt to run the baseline hit intermittent `getaddrinfo failed` on `api.groq.com`
(DNS resolved 1/3 on retry — a flaky local connection, not a code bug; the pipeline itself
completed each request). To keep a degraded run from silently poisoning the baseline, the
harness now:
- **Preflight check** — resolves `api.groq.com` ×3 and warns loudly if the network is
  down/flaky before a long run starts.
- **Shadow-success tracking** — records shadow models responded per request; the report's
  **DATA QUALITY** section shows mean shadows/req + full-shadow rate, and requests with
  **0 shadows (network-degraded) are excluded** from detection metrics.
- **Local warmup** — loads SBERT + adversarial guard once before timing, so the first
  request's cold-start (the 31 s `adversarial_guard` seen initially) doesn't pollute per-node numbers.
- **Quieter logs** — engine per-node exception tracebacks are silenced during measurement
  (the harness records per-request errors + shadow counts itself).
- **Exclusion breakdown** — detection now reports why rows were dropped
  (`guard_blocked` / `degraded_0_shadow` / `no_probability`), so guard false-positives
  (finding 3c) are visible in how much they shrink the scored sample.

None of this changes pipeline behavior; it is all harness-side.

### 5. Authoritative baseline — FROZEN (2026-07-14)

**Command:** `python scripts/measure_pipeline_baseline.py --cold` (env `failure-engine`, full jury, cold cache)
**Run:** 400 pairs, 0 errors, 93% full-shadow (mean 2.93 shadows/req, 0 fully-degraded).
**Raw:** `data/hallucination_eval/baseline_raw.jsonl` · **Report:** `data/hallucination_eval/baseline_report.json`

**Performance** (2 of 400 requests are latency artifacts — one ran 5.4 h because the
machine slept mid-`jury_deliberate`; the raw `mean 60 243 ms` / `max 19 334 720 ms` reflect
that stall. Percentiles and the outlier-excluded stats are the honest numbers):

| Metric | Value |
|---|---|
| End-to-end latency **p50 / p95** | **10.5 s / 27.6 s** |
| End-to-end mean / max (excl. 2 sleep/stall outliers) | 11.8 s / 48.4 s |
| Groq API calls / request | **6.45** (cache hits 0.11) |
| External HTTP calls / request | 2.7 |
| Tokens per request (in / out / total) | 1313 / 1012 / **2325** |

**Per-node p50 (ms)** — the four Groq-bound nodes dominate:

| Node | p50 | p95 | n | Makes Groq call? |
|---|---|---|---|---|
| jury_deliberate | **7186** | 16 708 | 280 | **Yes — hidden** (explanation humanizer, see 5a) |
| reasoning_verify | 3010 | 12 663 | 280 | Yes (decompose + socratic) |
| shadow_inference | 2477 | 3087 | 400 | Yes (3 models, parallel) |
| gt_verify | 2185 | 6604 | 115 | Yes (claim + Wikidata/Serper) |
| signal_extract | 351 | 704 | 280 | No (local SBERT) |
| adversarial_guard | 62 | 88 | 400 | No (offline) |

**Detection quality** (decontaminated TruthfulQA, score = `xgb_probability` vs frozen label):

| Metric | Value |
|---|---|
| Scored / excluded | 280 / 120 (**all 120 = guard-blocked**, see 5b) |
| Class balance (scored) | 140 hallucination / 140 correct |
| **ROC-AUC (95% CI)** | **0.497 [0.424, 0.562]** — i.e. chance |
| Recall @ FPR=0.10 / 0.05 | 0.129 / 0.086 |
| ECE 10-bin (95% CI) | 0.267 [0.218, 0.332] |
| Production operating point | recall 0.257, FPR 0.236 (tp 36 / fp 33 / fn 104 / tn 107) |

### 5a. Correction to the Phase-0 design analysis (result contradicts hypothesis)

My initial trace claimed `jury_deliberate` was offline/cheap (agents are regex + local
SBERT). **Measurement contradicts this:** it is the *most* expensive node (p50 7.2 s). Cause:
`failure_agent.run_diagnostic()` ends with `attach_explanations_to_diagnostic()`, which calls
`engine/explainability/humanizer.py` → `_generate_with_groq()` (`groq.complete(...)`,
[humanizer.py:50](../engine/explainability/humanizer.py#L50)) to write a human-readable
explanation. So every full-jury request pays an **extra, cosmetic Groq round-trip on the
critical path**. This is a prime Phase-1 target (defer/async/skip for detection). The jury's
*diagnostic* agents themselves remain local; the cost is the explanation LLM call.

### 5b. Two headline findings

1. **Decontaminated detection is at chance (AUC 0.497).** On held-out, leakage-free
   TruthfulQA the meta-classifier's probability does **not** separate correct from
   hallucinated answers (CI straddles 0.50; ECE 0.27 = poorly calibrated; production point
   recall 0.26 at FPR 0.24 sits on the diagonal). This starkly contradicts
   `data/training_results.json` (AUC 0.65–0.95), which was measured on the **same questions
   the classifier trained on**. This mirrors the adversarial-side lesson (RESEARCH_LOG E1):
   in-distribution/contaminated metrics are not evidence of real-world capability.
   *Caveat:* TruthfulQA is adversarially designed to elicit *plausible* falsehoods that
   strong models (incl. the 70B shadows) often share — so a consistency-based ensemble sees
   agreement on the wrong answer and cannot flag it. AUC 0.50 here means "cannot detect
   TruthfulQA-style confident falsehoods," not "detects nothing anywhere." A second,
   non-adversarial benchmark would bound the other end (blocked today by the broken local
   HaluEval file, finding 3b).

2. **The adversarial guard blocks 30% of benign QA (120/400).** Every guard-blocked pair is
   dropped from detection, shrinking the sample from 400→280. This is the hallucination-path
   analogue of the over-refusal problem already documented on the adversarial side. It needs
   quantifying/fixing but is separate from the detection-quality question.

**Frozen reference:** all later phases are measured against this run. The targets to beat are
p50 10.5 s / 6.45 Groq calls / 2325 tokens per request at **no loss** below AUC 0.497 (and
ideally *recovering* real detection signal, which the baseline shows is currently absent).

### 6. Next

Phase 1 — structural latency wins, revised in light of the measured baseline (biggest
Groq-bound nodes first):
- **Defer/skip the explanation humanizer Groq call** in `jury_deliberate` (5a) — it is the
  #1 node (p50 7.2 s) and is cosmetic for detection. Expected the largest single latency win.
- Guard-before-shadow so guard-blocked prompts (30%, finding 5b) don't first pay 3 shadow calls.
- Drop the nested per-step GT in `reasoning_verify`; early-exit routing on stable answers;
  lower Groq timeout / max_tokens; static socratic probes.

Each change validated against this frozen set, reporting the latency/calls/tokens delta and
confirming AUC does not drop below the 0.497 baseline. **Open detection question** (bigger than
latency): the decontaminated AUC is at chance — Phase 2's shadow/semantic-entropy redesign is
where we try to *recover* real detection signal, not just preserve it. The 30% guard over-block
(5b) and the broken HaluEval file (3b) are tracked as separate follow-ups.

---

## Phase 0.5 — Detection finding: LOCKED

**Date:** 2026-07-14
**Status:** decisive. The fused XGBoost hallucination detector is **at chance on held-out data.**

### The finding (two benchmarks, decontaminated, frozen)

| Benchmark | ROC-AUC (95% CI) | Verdict |
|---|---|---|
| TruthfulQA (decontaminated, end-to-end) | **0.497** | at chance |
| HaluEval (decontaminated, end-to-end) | **0.38** — 95% CI **entirely below 0.50** | at chance / inverted, **stable across 10 repeated runs** |

The old **0.65–0.95** headline numbers are **confirmed train/test contamination**: the
classifier was trained on TruthfulQA + HaluEval + MMLU questions
(`data/labeled/synthetic_*.jsonl`; `data/training_results.json`), and those metrics were
measured on the same questions it trained on. Decontaminated, held-out, the fused detector
does not separate hallucinated from correct answers on **either** benchmark.

Note the HaluEval end-to-end result (0.38, below chance) **contradicts** the offline
question-disjoint per-source number (0.956) computed on the *old April feature snapshot*
(§Phase 0.5 audit). That contradiction — a signal that separates in isolation on old features
but whose *current fused output* is at/below chance — is exactly what the Phase 0.5b
signal-level ablation is built to resolve: is a working signal being buried/inverted by the
fusion (→ retrain), or does nothing separate end-to-end (→ de-scope)?

### Supporting audits (Phase 0.5, offline, no Groq)

- **Contamination audit** (`scripts/audit_classifier_contamination.py` →
  `classifier_contamination_report.json`): shipped-model in-sample AUC 0.586; a fresh
  question-disjoint retrain reaches 0.743 overall but is entirely carried by HaluEval
  (per-source question-disjoint AUC: HaluEval 0.956, TruthfulQA 0.581, MMLU 0.536) on the
  **old** feature snapshot. The shipped slim model also looks under-fit vs. a clean retrain.
- **Guard over-refusal** (`scripts/audit_guard_overrefusal.py` →
  `guard_overrefusal_report.json`): the adversarial guard blocks 29.5% of benign TruthfulQA
  and 13.5% of HaluEval questions, **100% via the PAIR intent-classifier layer** firing
  `JAILBREAK_ATTEMPT` on ordinary factual questions — the documented PAIR benign-FPR / OOD
  problem surfacing on the hallucination path.

**Consequence:** optimizing the latency of a detector that is at chance is premature. The next
step (Phase 0.5b) is the signal-level ablation that decides **rebuild** (a signal works, fix
the fusion/contamination) vs. **de-scope** (nothing works; keep only deterministic checks and
document the rest as non-functional). No behavior changed in this phase — measurement only.

---
