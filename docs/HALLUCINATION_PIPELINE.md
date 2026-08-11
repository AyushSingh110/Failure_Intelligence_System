# FIE Hallucination Pipeline — Current Design

*Kept in sync with the code as the optimization phases land. Source of truth for
what's been measured is [HALLUCINATION_OPTIMIZATION_LOG.md](HALLUCINATION_OPTIMIZATION_LOG.md).*

Last updated: 2026-07-14 (Phase 0 — instrumentation only; no architecture change).

## Overview

The hallucination path is a 12-node LangGraph flow in
[engine/pipeline/langgraph_pipeline.py](../engine/pipeline/langgraph_pipeline.py).
It is separate from the offline adversarial guard ([fie/adversarial.py](../fie/adversarial.py), ~48 ms).

Node order (linear, with conditional skips):

```
load_session → shadow_inference → adversarial_guard
  → (blocked? finalize) → signal_extract → provenance_gate
  → reasoning_verify → jury_deliberate → security_checks
  → (high_risk? gt_verify) → (escalate | auto_correct) → finalize
```

| Node | External I/O | Notes |
|---|---|---|
| load_session | Mongo read | prior turns |
| shadow_inference | 3 Groq (parallel) | `llama-3.3-70b`, `gpt-oss-120b`, `qwen3-32b`; `max_tokens=500`, `timeout=30s` |
| adversarial_guard | none | offline scan; runs *after* shadow today |
| signal_extract | SBERT (local) | entropy / agreement / ensemble / embedding |
| provenance_gate | none | provenance category + label |
| reasoning_verify | Groq + **nested GT per factual step** | decompose + step-verify + socratic |
| jury_deliberate | **1 Groq** (explanation humanizer) | 3 local agents (regex + SBERT) **+** a cosmetic explanation LLM call — the #1 latency node at p50 7.2 s (see log 5a) |
| security_checks | none | multi-turn + extraction trackers |
| gt_verify | Groq + Wikidata + Serper | factual/temporal ground truth |
| escalate / auto_correct | Groq / Wikipedia | correction strategies |
| finalize | XGBoost (local) + Mongo | meta-classifier + persistence |

The XGBoost meta-classifier ([engine/failure_classifier.py](../engine/failure_classifier.py))
loads `models/failure_classifier_slim.pkl` (10 features) and produces the final
`xgb_probability` / `xgb_is_failure` decision.

## Instrumentation (Phase 0)

[engine/instrumentation.py](../engine/instrumentation.py) is an **opt-in**, single-request
metrics collector. It is INACTIVE by default — production `/monitor` requests never
activate it, so behavior and output are unchanged. When a measurement harness calls
`begin()`/`end()` around a request it records:

- per-node wall-clock (nodes wrapped by `_timed()` in `_build_graph`; node bodies untouched)
- Groq round-trips (API vs cache) + input/output tokens (hook in `groq_service._call_single_model`)
- external HTTP round-trips (hooks in Serper / Wikidata / Wikipedia retriever)

It is a process-global guarded by a lock (so Groq's thread-pool fan-out is captured)
and is therefore **only** valid for sequential benchmarking, not concurrent serving.

## Measurement harness

- `scripts/build_hallucination_eval_set.py` — builds the frozen, **decontaminated**
  detection eval set (see log Phase 0 for the leakage audit).
- `scripts/measure_pipeline_baseline.py` — drives the real pipeline over the frozen
  set and reports latency / calls / tokens + ROC-AUC / recall@FPR / ECE with bootstrap CIs.
