# Production Engineering

Every production decision in FIE, what problem it solves, and why it was made
this way rather than the obvious alternative.

This is the document to read before changing anything in the request path. It is
organised by concern: **correctness**, **failure handling**, **scalability**,
**observability**, **operator control**, **overhead**, and **maintainability**.

> Conventions used throughout: a **degraded** component still returns an answer
> with reduced quality. A **failed** component returns no answer. The difference
> matters more than it sounds — see [Fail-secure vs fail-open](#2-fail-secure-vs-fail-open).

---

## Table of contents

1. [Correctness under concurrency](#1-correctness-under-concurrency)
2. [Fail-secure vs fail-open](#2-fail-secure-vs-fail-open)
3. [Error handling policy](#3-error-handling-policy)
4. [Scalability](#4-scalability)
5. [Observability](#5-observability)
6. [Operator control surface](#6-operator-control-surface)
7. [Overhead and cost](#7-overhead-and-cost)
8. [Startup, readiness and shutdown](#8-startup-readiness-and-shutdown)
9. [Maintainability](#9-maintainability)
10. [Regression safety](#10-regression-safety)
11. [Dependency pinning](#11-dependency-pinning)
12. [Known open issues](#12-known-open-issues)

---

## 1. Correctness under concurrency

### 1.1 Atomic model publication

**Problem.** `_load_pair_classifier()` assigned two artifacts to module globals
in sequence — the sklearn classifier, then the sentence embedder. The readiness
check tested only the first:

```python
if _pair_load_attempted:
    return _pair_clf is not None      # embedder not checked
```

If the embedder load failed or was abandoned mid-flight, `_pair_clf` was already
set. Every later call passed the readiness check and then raised
`AttributeError: 'NoneType' object has no attribute 'encode'` on the embedder —
permanently, for the life of the process.

**Why it mattered.** The failure was invisible. `_run_pair_classifier` caught the
exception and returned `(None, 0.0, {})`, which is indistinguishable from "PAIR
looked at this prompt and saw nothing". The project's own ablation study
identifies PAIR as the layer carrying the common case, so this silently disabled
most of the detection while every scan still returned a confident-looking
verdict.

**Fix.** Build into locals, publish to globals only when both succeed, and guard
on both. See `fie/layers/pair.py`.

```python
local_clf      = joblib.load(clf_path)
local_embedder = SentenceTransformer(embed_model)
# Only now, with both in hand:
_pair_clf, _pair_embedder = local_clf, local_embedder
```

**Why not just check both in the guard?** That fixes the symptom for this pair of
variables. Atomic publication fixes the class of bug: any future third artifact
is covered by the same discipline.

### 1.2 Locking the load path

Layers run in parallel, so the first scan has twelve threads racing into lazy
loaders simultaneously. `_pair_lock` makes the load-and-publish sequence atomic;
double-checked locking keeps the lock off the hot path once loading has settled.

`_pair_load_attempted` is set *before* the attempt, not after. A failed load must
not be retried on every subsequent call — under load that becomes a retry storm
where every request thread pays a multi-second model-load timeout.

### 1.3 Path resolution anchored to the package root

`fie/layers/pair.py` derives model paths from the **package root**, not from
`Path(__file__).parent`:

```python
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent   # .../fie
```

When the layer code moved from `fie/adversarial.py` into `fie/layers/pair.py`,
`Path(__file__).parent` silently changed from `fie/` to `fie/layers/`. The
meta-classifier stopped loading. Individual layer scores were unaffected, so
every existing test still passed — only the golden-output test caught it
(§10). Anchoring to the package root makes these paths survive future moves.

---

## 2. Fail-secure vs fail-open

A guardrail that fails open is worse than no guardrail, because it reports
success while providing no protection.

### 2.1 The three failure points, and what each does

| Failure | Old behaviour | New behaviour | Control |
| --- | --- | --- | --- |
| One detection layer errors/times out | dropped from results, invisible | recorded in `ScanResult.degraded_layers` | — |
| LlamaGuard tiebreaker unavailable | fell through | blocks (fail-secure) | `FIE_UNCERTAIN_ALLOW=1` to allow |
| `scan_prompt()` itself raises | **allowed through, indistinguishable from clean** | flagged via `scan_failed`; blocks if configured | `FIE_SCAN_FAILURE_MODE=closed` |

### 2.2 "Scanned and safe" vs "not scanned"

This distinction is the core of the design. Previously both produced
`(False, "", 0.0, [])` and no caller could tell them apart.

`GuardResult.scan_failed` and `ScanResult.degraded_layers` now carry it:

```python
result = preflight_check(prompt)
if result.scan_failed:
    # The prompt was NEVER EXAMINED. Not a safety verdict.
    # Apply your own policy here.
```

Layers that time out are materialised as explicit `LayerResult`s with
`status=TIMEOUT` rather than being omitted. Two reasons:

1. The meta-classifier consumes `{layer_name: confidence}` as a feature vector.
   A vanished layer meant the model silently received a **short vector** rather
   than an explicit zero — an undetected inference bug.
2. A caller cannot otherwise distinguish "PAIR scored this 0.0" from "PAIR never
   ran".

### 2.3 Defaults, and why `FIE_SCAN_FAILURE_MODE` ships as `open`

Flipping the default to `closed` would mean any bug in the scanner takes the
caller's entire application down. That is a decision for the operator, not for a
library upgrade. So the default preserves historical behaviour, but:

- the failure now logs at **ERROR** with a traceback (it was `debug`),
- `scan_failed=True` is visible on the result,
- **`FIE_SCAN_FAILURE_MODE=closed` is the recommended production setting** and is
  documented in the README and `.env.example`.

If you accept untrusted input, set it to `closed`.

---

## 3. Error handling policy

### 3.1 What was wrong

A previous hardening pass mechanically converted ~105 `except Exception: pass`
blocks into:

```python
except Exception:
    logger.warning("Suppressed exception in some_function()", exc_info=True)
```

Better than `pass`, but it still fails three ways:

1. **It does not say what was lost.** "Suppressed exception in `scan_prompt()`"
   does not tell an operator whether detection recall dropped or a metrics write
   failed.
2. **It does not say why continuing is safe.** Each block encodes a safety
   argument that existed only in the author's head.
3. **It logs tracebacks at WARNING for routine events** — a missing optional
   dependency in a lite install is expected, not a problem. Real failures drown
   in the noise.

### 3.2 The replacement

`fie/_degrade.py` provides two primitives that force the author to state the
safety argument at the call site:

```python
from fie._degrade import degraded, attempt

with degraded("feedback_store",
              provides="instant verdict for known prompts",
              impact="adds latency, verdict unchanged"):
    record_block(prompt)

dampen = attempt(
    lambda: get_dampening_factor(prompt, fired),
    default    = 1.0,                       # safe direction: no dampening
    capability = "framing_filter",
    impact     = "no benign dampening — fails toward blocking",
)
```

Design points:

- **`impact` is mandatory.** If you cannot state a benign impact, the block
  should not be swallowing the exception at all.
- **`default` must be the *safe* value, not the neutral one.** For anything
  scaling a confidence score, safe means failing toward blocking. Losing a
  component must never make a prompt look *safer* than it was shown to be.
- **Expected-absence exceptions** (`ImportError`, `FileNotFoundError`) log at
  INFO with no traceback. Everything else logs at WARNING with one.
- **`capability=` is a stable grep key.** `grep 'capability=pair_classifier'`
  finds every degradation of that component across the codebase.

### 3.3 What must never be wrapped

Detection layers. If a layer fails, the scan is degraded and the caller must be
told. Swallowing that converts "not scanned" into "looks safe" — the exact
failure mode a guardrail cannot have. `_run_pair_classifier` deliberately
**raises** on inference failure so the runner marks the scan degraded.

### 3.4 Log levels

| Level | Meaning | Example |
| --- | --- | --- |
| `INFO` | Expected optional component absent | lite install without `[ml]` extras |
| `WARNING` | Degraded but serving | one layer timed out |
| `ERROR` | A guarantee was broken | `scan_prompt()` raised — prompt not scanned |

Tracebacks attach only at ERROR, or at WARNING when DEBUG logging is enabled
(`exc_info=logger.isEnabledFor(logging.DEBUG)`). This keeps production logs
readable while preserving full detail on demand.

---

## 4. Scalability

### 4.1 The thread pool problem

**Before:** each `scan_prompt()` created one outer `ThreadPoolExecutor(12)` plus
one nested single-worker executor *per layer*. That is **13 thread pools and ~24
OS threads created and destroyed per scan**, so thread count scaled with request
rate rather than with cores.

**After:** one process-wide pool, created lazily, bounded, with named threads:

```python
_LAYER_POOL_SIZE = max(4, int(os.environ.get("FIE_LAYER_POOL_SIZE", "16")))
_layer_pool = ThreadPoolExecutor(max_workers=_LAYER_POOL_SIZE,
                                 thread_name_prefix="fie-layer")
```

**Measured effect:** warm median scan latency **28.9 ms → 22.3 ms** on this
machine, with no change to any detection output. Thread creation was roughly a
quarter of the scan budget.

Named threads matter operationally: `fie-layer-3` in a stack dump is
diagnosable; `Thread-47` is not.

### 4.2 The timeout that never worked

The nested per-layer pool was intended to enforce a 2-second per-layer timeout:

```python
with ThreadPoolExecutor(max_workers=1) as ex:      # ← the bug
    fut = ex.submit(layer_fn)
    root, conf, ev = fut.result(timeout=2.0)
```

`with ThreadPoolExecutor(...)` calls `shutdown(wait=True)` on exit. So even after
`fut.result(timeout=2.0)` gave up, the `with` block **blocked until the hung
layer finished anyway**. The per-layer timeout could never fire.

The replacement enforces a single deadline across the whole layer set and is
honest about the limit: **Python cannot kill a running thread.** A timeout means
"stop waiting and mark the layer degraded", not "cancel it". The abandoned thread
returns its worker to the shared pool when it eventually completes.

### 4.3 Pool sizing

Default 16 workers for 12 layers, so a single scan never queues internally while
still bounding total threads under concurrency. Tune with `FIE_LAYER_POOL_SIZE`.

On a small free-tier instance (1 vCPU), lower it — 12 threads contending for one
core adds context-switch overhead without parallelism. On a multi-core host
serving concurrent traffic, raise it. The layers are I/O- and regex-bound rather
than GIL-bound in the common case, so oversubscription is mildly beneficial.

### 4.4 Import resolution caching

`_get_attack_threshold()` ran `from engine.fie_config import ...` on **every
scan**. In an SDK-only install `engine` does not exist, so that raised
`ImportError`, built a traceback, and logged a warning **on every single call**.

Now resolved once and memoised in both directions (`_server_config()`). This is
both a correctness fix (the hot-config path was entirely dead — the imported
symbol did not exist) and a hot-path cost removal.

### 4.5 Caching

- **Scan cache** — TTL-aware LRU keyed on `SHA-256(prompt)`. Raw prompt text is
  never stored. The domain is part of the key so `domain="medical"` and
  `domain="developer"` cannot collide on the same prompt.
- **Feedback store** — confirmed verdicts short-circuit the full pipeline. This
  is a latency optimisation only; if it is unreadable the scan falls through to
  the full pipeline, which is strictly more thorough.

---

## 5. Observability

### 5.1 Structured logging

Log lines are `key=value` so they can be parsed without regex gymnastics:

```
layer=pair_classifier status=ready model=pair_intent_classifier_v6_3b.pkl threshold=0.50
scan status=degraded reason=layer_timeout deadline_s=10.0 layers=gcg_suffix,perplexity_proxy
degraded capability=session_tracker impact='multi-turn history not recorded' reason=OSError: ...
```

Stable keys — `capability=`, `status=`, `layer=`, `reason=`, `impact=` — mean an
operator can build alerts without knowing every message string.

### 5.2 Request correlation

Every request gets an `X-Request-ID` (accepted from the caller or generated) and
binds it to a `ContextVar`, so every log line emitted while handling that request
carries the same ID.

### 5.3 Progress bars suppressed

`sentence-transformers` writes a tqdm progress bar to **stderr** by default. In a
container that interleaves with structured logs and corrupts line-based parsers.
Every `encode()` call passes `show_progress_bar=False`.

### 5.4 Health surface

Three endpoints with three distinct jobs — conflating them is a classic
production failure:

| Endpoint | Question | Cost | Use for |
| --- | --- | --- | --- |
| `/health` | Is the process alive? | no network, no model load | liveness probe |
| `/ready` | Should it receive traffic? | reads a flag | readiness probe / LB |
| `/health/deep` | What exactly is wrong? | pings every dependency | on-call, dashboards |

**Why liveness must stay cheap:** a liveness probe that touches slow dependencies
eventually times out under load and gets the container killed — turning a partial
outage into a restart loop.

**Why readiness must exist separately:** without it, traffic routes to a cold
instance whose classifier has not loaded. Those users get scanned by a degraded
pipeline that still returns a confident verdict. `/ready` returns **503 until
warm-up completes**, which is also what makes a zero-downtime rolling deploy
possible.

`fie.adversarial.health()` never triggers a model load — a health probe must
never be the thing that pays the load cost.

---

## 6. Operator control surface

### 6.1 Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `FIE_SCAN_FAILURE_MODE` | `open` | `closed` blocks when the scanner fails. **Recommended: `closed`** |
| `FIE_UNCERTAIN_ALLOW` | unset | `1` allows UNCERTAIN prompts when LlamaGuard is unavailable |
| `FIE_LAYER_POOL_SIZE` | `16` | Shared layer thread pool size |
| `FIE_LAYER_DEADLINE_S` | `10.0` | Deadline for the whole layer set |
| `FIE_PAIR_VERSION` | `v6_3b` | Pin a specific PAIR model for A/B comparison |
| `FIE_LITE` | unset | `1` runs the 4-layer no-ML pipeline |
| `SCAN_THRESHOLD` | `0.65` | Global fallback confidence floor |
| `PREFLIGHT_BLOCK_ENABLED` | `true` | `false` = warn-only (detect and log, do not block) |
| `FIE_NO_TELEMETRY` | unset | `1` disables the anonymous version ping |
| `SENTRY_DSN` | unset | Enables error tracking; `send_default_pii=False` always |

### 6.2 Hot configuration

Per-attack-type thresholds are operator-tunable at runtime via MongoDB without a
restart (`engine/fie_config.py`).

**Precedence:** operator override → compiled per-type calibration → global
`SCAN_THRESHOLD`.

The override dict starts **empty by design**. The compiled defaults live next to
the detection code so the SDK works with no server, and an empty override map
therefore reproduces the shipped calibration exactly — which is what every
published benchmark number was measured against. A `pip install` with no
database gives you the numbers in the README.

**Validation is asymmetric, deliberately:**

- The **DB loader clamps** out-of-range values into `[0.05, 0.99]` and logs
  loudly. One bad key must not break a config reload for every other key.
- The **interactive updater rejects** them with `ValueError`. An operator typing
  a threshold at an admin endpoint needs to be told they were wrong, not silently
  corrected.

A threshold of `72` (meaning `0.72`) would otherwise disable a layer or block all
traffic.

### 6.3 Domain-aware thresholds

Per-request threshold multipliers, inferred from prompt text or passed
explicitly: medical `0.80`, finance `0.82`, legal `0.83`, education `0.88`,
default `1.00`, developer `1.12`.

Below 1.0 = stricter blocking (a missed attack in a clinical context is costlier
than a false positive). Above 1.0 = more permissive (security research and CTF
work produce prompts that legitimately resemble attacks).

---

## 7. Overhead and cost

### 7.1 Measured request-path costs

| Item | Before | After |
| --- | --- | --- |
| Warm median scan | 28.9 ms | **22.3 ms** |
| Thread pools per scan | 13 | **0** (shared) |
| Threads created per scan | ~24 | **0** (reused) |
| `ImportError` + traceback per scan | 1 | **0** |
| Cold-start first scan | degraded (~10 s) | warm-up gated by `/ready` |

### 7.2 Dependency weight — resolved

`torch` was 4,668 MB in the working environment (CUDA build) and existed solely
to compute 384-dim MiniLM embeddings. It has been replaced with ONNX Runtime:

| | torch | ONNX | Change |
| --- | --- | --- | --- |
| Disk | 4,668 MB | ~113 MB | ~41x |
| Embedding | 16.64 ms | 6.80 ms | 2.4x |
| Warm-up | 12.6 s | 1.97 s | 6.4x |
| Scan (median) | 38.2 ms | 24.6 ms | 36% |
| Output | reference | cosine 1.000000 | identical |

Removing the full stack reclaimed **4.5 GB**. int8 quantisation was tested and
rejected — cosine 0.879, failing the >0.999 gate — because it would have moved
the PAIR decision boundary while still returning confident probabilities.

Details and the reusable verification protocol: [`docs/DEPLOYMENT.md`](DEPLOYMENT.md).

### 7.3 Logging overhead

Tracebacks are formatted only when they will be used
(`exc_info=logger.isEnabledFor(logging.DEBUG)`). Formatting a traceback on a hot
path that discards it is pure waste, and the previous code did it on every scan.

---

## 8. Startup, readiness and shutdown

### 8.1 Warm-up

Model loading is lazy. Without an explicit warm-up the *first request* pays it —
about 10 s against a 10 s layer deadline, so a cold container's first real
request is served with PAIR degraded.

`_warm_models_in_background()` runs in a named daemon thread during the FastAPI
lifespan. The port accepts connections immediately (platform health checks do not
wait for a transformer to load) while `/ready` returns 503 until warm-up
completes. Measured warm-up: **~3.9 s**.

SDK users get the same control via `fie.adversarial.warmup()`:

```python
status = warmup()
if status["pair_classifier"] != "ready":
    log.warning("serving with reduced recall: %s", status)
```

### 8.2 Graceful shutdown

The lifespan handler calls `shutdown_layer_pool(wait=False)` before exit. Without
it the 16 workers are reclaimed only by the `atexit` hook, which **does not run
on SIGKILL** and can leave a container lingering past its termination grace
period during a rolling deploy.

---

## 9. Maintainability

### 9.1 The monolith split

`fie/adversarial.py` was **3,016 lines / 126 KB** — large enough that navigating
it was the main barrier to anyone contributing a detection change.

A dependency analysis found the layers were already independent: the only
cross-layer edge was `prompt_guard` reusing `patterns._normalize_for_detection`.
Everything else was a pure function of the prompt. So the split is a **move, not
a redesign**.

| Module | Lines | Layer |
| --- | --- | --- |
| `fie/adversarial.py` | 1,386 | orchestration, routing, aggregation, public API |
| `fie/layers/patterns.py` | 494 | 1 — regex patterns |
| `fie/layers/pair.py` | 332 | 8 — PAIR + meta-classifier |
| `fie/layers/many_shot.py` | 253 | 3 — many-shot |
| `fie/layers/perplexity.py` | 206 | 6 — perplexity proxy |
| `fie/layers/gcg.py` | 164 | 5 — GCG suffix |
| `fie/layers/prompt_guard.py` | 151 | 2 — grouped semantic |
| `fie/layers/copyright.py` | 113 | 12 — copyright |
| `fie/layers/indirect.py` | 109 | 4 — indirect injection |
| `fie/layers/direct_harm.py` | 108 | 7 — direct harm |

All names are re-exported from `fie.adversarial`, so existing imports in
evaluation scripts, the ablation study and notebooks keep working unchanged.

### 9.2 Dead code removed

`app/routes.py` (1,871 lines) was tracked in git but **permanently shadowed** by
the `app/routes/` package — Python resolves the package, so the module was
unreachable. Verified the package is a strict superset (every endpoint present,
plus 8 more) before deleting.

### 9.3 Test suite structure

Nine of fourteen files in `tests/` were **standalone scripts**, not pytest
modules. Several called `sys.exit()` at module scope, which aborts the entire
pytest session. That is why CI could only ever run a hand-picked subset — and why
`test_monitor_rag_fix.py` sat broken since the routes refactor without anyone
noticing.

Scripts moved to `scripts/manual_checks/` with a README. `tests/` now contains
only real pytest modules, and CI runs `pytest tests/ -m "not network"` — new test
files are covered the day they are added.

---

## 10. Regression safety

### 10.1 Golden-output testing

FIE's published numbers are properties of **exact confidence values**, not just
`is_attack` booleans. A refactor that shifts a threshold by 0.01 can keep every
conventional test green while invalidating every number in the README.

`tests/test_detection_golden.py` pins the full decision surface — attack type,
confidence to 4 dp, layers fired, and every individual layer score — across a
22-prompt corpus spanning all twelve layers.

**This is not theoretical.** During the monolith split it caught two regressions
that no other test detected:

1. `gcg` and `perplexity` layers raising `NameError` from missing stdlib imports.
2. The meta-classifier silently failing to load after the path change (§1.3) —
   every layer score identical, only the blended confidence moved.

Regenerate deliberately, never casually:

```bash
python -m tests.test_detection_golden --update
```

Then **read the diff**. Every changed line is a change in detection behaviour and
belongs in the changelog with re-measured benchmarks.

The corpus deliberately includes benign prompts. Over-refusal is FIE's documented
weak spot, so locking in current false-positive behaviour is how a "harmless"
tweak that starts blocking safe prompts gets caught.

### 10.2 Layer-completeness test

`test_no_layer_silently_missing` asserts every layer reports a score for every
prompt and that `degraded_layers` is empty. This is what keeps "scanned and safe"
distinct from "not actually scanned" at the test level.

---

## 11. Dependency pinning

`scikit-learn` is pinned to **1.7.2** because the shipped `.pkl` classifiers were
serialized with it and the decision boundary is version-sensitive — a mismatch
previously caused a real FPR regression.

> **Open issue.** The current development environment has scikit-learn **1.6.1**
> installed, producing `InconsistentVersionWarning` on every model load. Local
> results may not reproduce the published benchmarks. Fix with
> `pip install scikit-learn==1.7.2` before measuring anything. See §12.

`xgboost` is pinned to 2.1.4, but the serialized model predates it and emits an
upgrade warning on load. Re-saving via `Booster.save_model()` would clear it —
and requires re-validating the decision boundary.

---

## 12. Known open issues

Honest list of what is **not** fixed.

| Issue | Impact | Suggested fix |
| --- | --- | --- |
| scikit-learn 1.6.1 installed vs 1.7.2 pinned | Local runs may not reproduce published numbers | `pip install scikit-learn==1.7.2`, re-run benchmarks |
| XGBoost model serialized by an older version | Warning on every load; potential boundary drift | Re-save with `Booster.save_model()`, re-validate |
| ~100 generic `Suppressed exception` handlers remain in `engine/` and `app/` | Poor diagnosability in the server path | Migrate to `fie._degrade` (SDK path is done) |
| `torch` = 1.32 GB | Blocks every free hosting tier | ONNX migration — [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) |
| `FIE_SCAN_FAILURE_MODE` defaults to `open` | Scanner crash silently allows traffic | Set `closed` in production |
| `datetime.utcnow()` deprecation warnings | Will break on a future Python | Migrate to `datetime.now(datetime.UTC)` |
| Virtualization layer misses its own golden case | Known false negative, recorded in the golden file | Layer tuning |
| `.git` is 151 MB (~136 MB is four stale sdist tarballs) | Slow clones deter contributors | `git filter-repo --path dist/ --invert-paths` |

---

## Quick reference for reviewers

Changing anything in the request path? Check:

- [ ] Does it change detection output? → run `pytest tests/test_detection_golden.py`
- [ ] Does it add a swallowed exception? → use `fie._degrade`, state the `impact`
- [ ] Does it add a fallback? → is the default the **safe** value, not the neutral one?
- [ ] Does it add a lazy load? → is publication atomic, and is the lock held?
- [ ] Does it add a thread? → use the shared pool, name the thread
- [ ] Does it add a dependency? → what does it cost in MB, and is it optional?
- [ ] Does it change a threshold? → re-measure benchmarks, update README and RESEARCH_LOG
