# PLAN_003 — Truthful Scan Result (WP-003)

# 1. Plan status

| | |
| --- | --- |
| Work package | WP-003 |
| Status | **PLAN ONLY. Nothing is implemented. Awaiting owner review** |
| Date | 2026-10-09 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` at `b077aea` ("security: isolate server tenant state and harden authentication (WP-002)") |
| WP-002 committed? | Yes, by the owner, as `b077aea` |
| Working tree | Clean before this file was written. This file is the only change |
| Environment | Python 3.10.19, conda `failure-engine`; tests run with the CI-equivalent environment used in WP-001 and WP-002 |
| Measurement foundation | `evals/`, baseline `BL-0001_pair-v6.3b_fie-5d5ed90d` |

Evidence labels. CODE-READ: verified at the cited line in the current code. MEASURED: a number
from the pinned WP-001 baseline. Nothing in this plan was run against a live service.

# 2. Objective

Every scan result must say truthfully what ran, which decision zone the verdict belongs to,
and which model produced it. A consumer must be able to tell "this prompt looked safe to the
full pipeline" from "the classifier never loaded and four pattern layers saw nothing".

This package changes what a result **reports**. It does not change what FIE **decides**: no
threshold, no layer logic, no model, no aggregation rule.

# 3. Current behavior

All CODE-READ in [fie/adversarial.py](../../../fie/adversarial.py) and
[fie/layers/pair.py](../../../fie/layers/pair.py).

**The root cause of the false "full coverage" report is one line.** When the classifier is
not loaded, the PAIR layer returns the same value as "looked and found nothing":

```text
fie/layers/pair.py:325   if not _load_pair_classifier():
fie/layers/pair.py:326       return None, 0.0, {}
```

The layer wrapper sees a normal return and records `status = "ok"`
([adversarial.py:689-691](../../../fie/adversarial.py#L689)). `degraded_layers` is built from
layers whose status is not `ok` ([:1086](../../../fie/adversarial.py#L1086)), so the missing
classifier is not in it. The field's own comment says "Empty list = full pipeline ran"
([:333-337](../../../fie/adversarial.py#L333)). That statement is false whenever the
classifier is absent.

| Aspect | Today |
| --- | --- |
| Coverage | One field, `degraded_layers: list[str]`, filled only for a layer that **raised** or **timed out**. Not filled for: classifier or its dependencies missing; a layer the caller disabled (`disabled_layers` removes it from the task list, so it leaves no trace); the meta-classifier missing (returns 0.0); translation unavailable (returns `None`); the tiebreaker unavailable (visible only as the evidence string `llama_guard: "unavailable_blocked"`) |
| Layer states that exist internally | `LayerStatus`: `ok`, `timeout`, `error`, `skipped` ([:634-639](../../../fie/adversarial.py#L634)). `skipped` is defined and never assigned. The per-layer status never reaches the result; only the derived list does |
| Zone | Computed, not exposed. The router has three branches: clear safe, clear attack, uncertain ([:1197-1350](../../../fie/adversarial.py#L1197)). The uncertain branch has four outcomes: tiebreaker confirms attack, tiebreaker says safe, tiebreaker unavailable and blocked (default), tiebreaker unavailable and allowed (`FIE_UNCERTAIN_ALLOW=1`). All collapse into `is_attack`. The only trace is `evidence["llama_guard"]` |
| Fast paths | A prompt whose hash was labelled returns before any layer runs: `evidence={"feedback": "whitelisted"}` or `"confirmed_tp"`, `layer_scores={}`, `degraded_layers=[]` ([:1036-1052](../../../fie/adversarial.py#L1036)). It reads as a full, clean scan |
| Model identity | None on the result. The loader logs the file name once. `_pair_state()` exposes `loaded`, `threshold` and an error string that contains a filesystem path ([pair.py:264-267](../../../fie/layers/pair.py#L264)) |
| Confidence when allowed | Always `0.0` for an allowed prompt, including one that sat just under the block threshold or was cleared by the tiebreaker |
| Cache | A cached `ScanResult` object is returned as is for 5 minutes ([:1069-1071](../../../fie/adversarial.py#L1069)). Whatever it says about coverage is what the first scan saw |
| Async | `scan_prompt_async` wraps `scan_prompt` and returns the same object. It does not accept `disabled_layers` |
| `scan_prompt_lite` | A separate function with its own `LiteScanResult`. Runs 4 layers by design; `degraded_layers` lists only layers that raised |
| Serialization | No `to_dict`. The CLI builds JSON by hand, twice, with different field sets, and omits `degraded_layers` from both ([__main__.py:52-62, 113-122](../../../fie/__main__.py#L52)) |
| Pre-flight | `GuardResult` carries `blocked`, `attack_type`, `confidence`, `layers_fired`, `scan_failed`. It drops `degraded_layers` |

# 4. Evidence from WP-001

MEASURED, baseline `BL-0001`, profile `lite-simulated` (the ML packages unimportable, which is
the code path of a base `pip install fie-sdk`):

| Quantity | Value |
| --- | --- |
| Scans | 1,848 |
| Classifier loaded | No ("missing dependency: 'joblib' …") |
| Scans whose result reported full coverage (`degraded_layers == []`) | **1,848 of 1,848** |
| Attack recall, lite vs canonical | JailbreakBench 14 vs 130 of 134; HarmBench 17 vs 326 of 387; StrongREJECT 15 vs 218 of 242; SORRY-Bench 12 vs 316 of 387 |

The metric, kept exactly as WP-001 defined it
([evals/metrics.py:422-434](../../../evals/metrics.py#L422)): **false-full-coverage** = the
number of scans in the lite suite whose record has an empty `degraded` list while
`pair_classifier` did not run. Today 1,848 of 1,848 (100%). Target after WP-003: **0 of
1,848 (0%)**.

Also from WP-001: the harness derives the zone itself from a private evidence string, because
the result has no zone ([evals/subject.py:339](../../../evals/subject.py#L339)), and reads 22
private names in `fie`, several of them only to learn what the result should have said.

# 5. Scope

In scope, all inside `fie/` plus the harness adapter and tests:

1. A per-layer state on the result, with seven distinguishable states (§8).
2. A coverage summary derived from it.
3. A public `zone` and how the verdict was reached (§9).
4. The identity of the model artifacts actually loaded (§10).
5. One serialization method used by every output path (§13).
6. The same fields on the pre-flight result and the lite result.
7. A new pinned harness baseline that differs from `BL-0001` only where this package
   intends (§16, §22).

# 6. Non-goals

No PAIR retraining, new classifier, threshold change, new layer, multilingual, long-input,
adaptive-attack or homoglyph work; no model-format or pickle migration; no packaging change;
no dependency; no network use; no change to `app/`, `engine/` or `storage/`; no README or
marketing text; no change to what any layer returns or to how verdicts are aggregated.

The roadmap row for WP-003 also lists `decision`, `risk_score` and a warning for inert
parameters. The brief for this plan lists coverage, zone and model identity. The extras are
put to the owner as decisions W3-9 and W3-10, not assumed.

# 7. Current ScanResult contract

Definition: [fie/adversarial.py:320-372](../../../fie/adversarial.py#L320), a plain mutable
`@dataclass`, exported from `fie/__init__.py`.

| Field | Type | Meaning today |
| --- | --- | --- |
| `is_attack` | bool | The verdict |
| `attack_type` | str or None | Winning type; `None` when allowed |
| `category` | str or None | Always `None` |
| `confidence` | float | Aggregated confidence when blocked; `0.0` when allowed |
| `layers_fired` | list[str] | Layers behind the winning type when blocked; every layer that fired when allowed |
| `matched_text` | str or None | Regex excerpt |
| `mitigation` | str | Advice text |
| `evidence` | dict | Per-layer raw detail of the winning type, plus routing notes |
| `layer_scores` | dict | Confidence per layer that ran |
| `degraded_layers` | list[str] | Layers that raised or timed out |
| `is_degraded` | property | `bool(degraded_layers)` |

Construction sites: eight, all in `scan_prompt` — two fast paths and six routing outcomes.

Consumers (complete inventory, CODE-READ by search over the repository):

| Consumer | Reads | Notes |
| --- | --- | --- |
| `fie/preflight.py` `_safe_scan` | `is_attack`, `attack_type`, `confidence`, `layers_fired` | Drops everything else; feeds `GuardResult` |
| `fie/monitor.py`, `fie/integrations/openai.py`, `anthropic.py` | `GuardResult` fields | Log lines only |
| `fie/integrations/fastapi.py` | `attack_type`, `confidence`, `is_attack` | — |
| `fie/__main__.py` `detect`, `explain` | hand-picked fields, `evidence`, `layer_scores` | Two JSON shapes |
| `fie/_lite.py` | own `LiteScanResult` | Parallel contract |
| `fie/adversarial.py` `_record_session`, cache | `attack_type`, `confidence`, `is_attack` | Internal |
| `deploy/huggingface/app.py`, `space_app.py` | `is_attack`, `attack_type`, `confidence`, `layers_fired`, `degraded_layers`, `layer_scores`, `evidence` | Already shows a "degraded scan" note when the list is non-empty |
| `app/routes/monitor.py`, `playground.py` | `GuardResult` only | No `ScanResult` field reaches a server response |
| `engine/pipeline/langgraph_pipeline.py` | `layers_fired`, `evidence` | Not routed |
| `evals/subject.py` | `is_attack`, `attack_type`, `confidence`, `layers_fired`, `layer_scores`, `degraded_layers`, `evidence` | Writes the per-prompt records |
| `tests/test_detection_golden.py` | `is_attack`, `attack_type`, `confidence`, `layers_fired`, `layer_scores`, `degraded_layers` | Pins values to 4 decimals; asserts `degraded_layers` is empty with models loaded |
| `tests/test_sdk.py`, `test_integration.py` | attribute presence | `hasattr` checks |

No consumer compares two `ScanResult` objects for equality, serializes one with
`dataclasses.asdict`, or constructs one positionally outside `scan_prompt`. New fields with
defaults are therefore additive for every consumer found.

# 8. Coverage model

**Layer states.** One per layer, per scan. Seven values, because they mean different things
to a caller:

| State | Meaning | Today |
| --- | --- | --- |
| `ok` | Ran and returned a normal result (signal or no signal) | `ok` |
| `unavailable_dependency` | A package the layer needs cannot be imported | reported as `ok` |
| `unavailable_model` | The model file is not present | reported as `ok` |
| `unavailable_load_failed` | The model exists and failed to load | reported as `ok` |
| `error` | Raised while running | `error` |
| `timeout` | Still running at the scan deadline | `timeout` |
| `disabled` | Removed by the caller (`disabled_layers`) | no trace |

"Did not run" is not a state of its own: it is the union of the last six, and the state says
why.

**How a layer reports "unavailable" without changing any verdict.** The scanner-side wrapper
`_layer_pair` checks the loader's settled state after the call. If the classifier is not
usable it raises a small internal exception, `LayerUnavailable(reason)`, which
`_run_layer_safe` turns into `attack_type=None, confidence=0.0` — exactly the values the
layer contributes today — with the new state. Aggregation, the meta-classifier's input
vector (an explicit 0.0) and routing are unchanged. The multilingual layer calls the
classifier function directly as a booster, not through `_layer_pair`, and is untouched.

**Result fields.**

```text
result.coverage            ScanCoverage (frozen)
    .status                "full" | "partial" | "bypassed"
    .layers                {layer_name: state}          all twelve, always
    .classifier            state of "pair_classifier"   convenience
    .optional              {"meta_classifier": ..., "tiebreaker": ..., "translation": ...}
result.degraded_layers     list[str]                    kept; see below
result.schema_version      2
```

| `coverage.status` | Rule |
| --- | --- |
| `full` | All twelve layers are `ok` |
| `partial` | At least one layer is not `ok`, for any reason including `disabled` |
| `bypassed` | No layer ran: the verdict came from a labelled-prompt fast path |

`optional` reports components that are allowed to be absent and do not make coverage
`partial`: the meta-classifier (`ok` / `unavailable`), the tiebreaker (`not_needed` /
`confirmed` / `cleared` / `unavailable` / `disabled`), and translation (`not_needed` /
`ok` / `unavailable`). Whether a missing optional component should lower the status is
decision W3-3.

**`degraded_layers`.** Kept, same type. Its documented meaning — "empty list = full pipeline
ran" — is made true: it now also lists layers in an `unavailable_*` state. It still does not
list `disabled` layers, because the caller asked for that; `coverage.status` covers them.
This is the one existing field whose observed value changes, and only in installs where the
classifier is missing: there it goes from `[]` to `["pair_classifier"]`. That is the fix.
The alternative, leaving the field as it is and reporting only through `coverage`, is
decision W3-2.

The smallest honest check for a consumer is one line: `result.coverage.status == "full"`.

# 9. Zone model

The zone is a routing result. It is assigned in exactly one place, the router at the end of
`scan_prompt`, from the aggregated confidence and the per-type threshold. A detection layer
cannot assign it.

**Validated against the production paths**, not copied from the harness:

| Production path | `is_attack` | Harness's private rule gives | Proposed `zone` | Proposed `decided_by` |
| --- | --- | --- | --- | --- |
| Below the safe ceiling, or no layer fired | False | `allow` | `allow` | `pipeline` |
| At or above the threshold | True | `clear_block` | `clear_block` | `pipeline` |
| Uncertain band, tiebreaker unavailable or off, default policy | True | `uncertain_block` | `uncertain_block` | `fail_secure` |
| Uncertain band, tiebreaker confirms | True | `clear_block` | `uncertain_block` | `tiebreaker` |
| Uncertain band, tiebreaker clears | False | `allow` | `uncertain_allow` | `tiebreaker` |
| Uncertain band, `FIE_UNCERTAIN_ALLOW=1` | False | `allow` | `uncertain_allow` | `config` |
| Labelled false positive (fast path) | False | `allow` | `allow` | `feedback_override` |
| Labelled attack (fast path) | True | `clear_block` | `clear_block` | `feedback_override` |

The harness's three-value rule is correct for the configuration it measures (tiebreaker off,
no override), where rows 4 to 8 cannot occur. It is not a complete public contract: it would
call a prompt the tiebreaker confirmed a "clear" block and would hide that an allowed prompt
had been in the uncertain band. Hence four zone values and a second field for who decided.

```text
result.zone         "allow" | "uncertain_allow" | "uncertain_block" | "clear_block"
result.decided_by   "pipeline" | "tiebreaker" | "fail_secure" | "config" | "feedback_override"
```

Answers to the brief's questions.

| Question | Answer |
| --- | --- |
| Exact values | The four above. A `Literal` type; construction with any other value raises |
| Owner | The router in `scan_prompt` |
| Can a layer assign a zone? | No |
| Aggregation or routing result? | Routing |
| Information unavailable? | The zone is always known when a result exists. If the scan itself fails there is no `ScanResult`; the pre-flight result already says `scan_failed=True` and gets `zone=None` |
| `allow` while degraded? | Yes. `zone="allow"` with `coverage.status="partial"` is the honest description of a lite install that found nothing. It is the combination this package exists to make visible |
| Can a result be `uncertain_block`? | Yes: 63 of 250 XSTest-safe and 41 of 250 OR-Bench-hard prompts in the baseline |
| Stable across async | Yes: `scan_prompt_async` returns the same object |
| Part of the SDK contract | Yes, documented and exported |
| Invariant | `is_attack == (zone in {"uncertain_block", "clear_block"})`, enforced at construction |

# 10. Model identity model

```text
result.models       {role: ModelIdentity}   roles: "pair_classifier", "meta_classifier", "encoder"

ModelIdentity (frozen)
    loaded          bool
    version         str | None     from the artifact's own metadata, e.g. "v6.3b"
    digest          str | None     first 16 hex characters of the SHA-256 of the file that was loaded
    threshold       float | None   classifier roles only
    backend         str | None     encoder only: "onnx" | "sentence-transformers"
```

| Consideration | Decision in this plan |
| --- | --- |
| What is hashed | The bytes of the file actually opened, at load time, once. Not the manifest's declared value: identity must describe what ran |
| Full or abbreviated hash | 16 hex characters. The full hashes are already public in `scripts/model_manifest.json`; the abbreviation is for brevity in logs and records, not secrecy. 64 bits is ample to tell artifacts apart. Full hash is decision W3-6 |
| Per-layer identity | Only three artifacts exist: the PAIR classifier, the meta-classifier, the encoder. Other layers are code; their identity is the package version, which the result does not need to repeat |
| Model missing | `loaded=False`, everything else `None`. Never a default or expected version |
| Lite profile | `pair_classifier.loaded=False`, `encoder.loaded=False`; `coverage.classifier="unavailable_dependency"` |
| Fast-path result | `models` still reports what is loaded in the process; `coverage.status="bypassed"` says it was not consulted |
| Not exposed | File names, directories, the loader's error string (it contains a path), environment, host |

Security: nothing here is secret. The model files and their hashes are published. A digest
lets an operator confirm which artifact a deployment runs, and lets the harness check the
result against the file it verified, without reading a log line.

Cost: hashing the classifier and meta-classifier files is milliseconds. Hashing the 90 MB
encoder is a few hundred milliseconds, once, during load or warm-up. Whether to hash the
encoder is decision W3-7.

# 11. Error/timeout semantics

| Event | Verdict effect (unchanged) | In the result |
| --- | --- | --- |
| Layer raises | That layer contributes nothing | `coverage.layers[name]="error"`; in `degraded_layers` |
| Layer times out | Same | `"timeout"`; in `degraded_layers` |
| Classifier package missing | Same | `"unavailable_dependency"`; in `degraded_layers` |
| Classifier file missing | Same | `"unavailable_model"`; in `degraded_layers` |
| Classifier load fails | Same | `"unavailable_load_failed"`; in `degraded_layers` |
| Caller disables a layer | Same | `"disabled"`; not in `degraded_layers` |
| Meta-classifier missing | Its blend is skipped | `coverage.optional["meta_classifier"]="unavailable"` |
| Translation unavailable | Non-English path loses its boost | `coverage.optional["translation"]="unavailable"` when a translation was attempted and returned nothing |
| Tiebreaker unavailable | Uncertain band is blocked (default) | `zone="uncertain_block"`, `decided_by="fail_secure"`, `optional["tiebreaker"]="unavailable"` |
| Partially completed pool | Finished layers count | Each unfinished layer is `"timeout"` |
| Scan raises entirely | No `ScanResult` | `GuardResult.scan_failed=True` (exists today) |

States are fixed codes. No exception text, class name or message is placed in a public
field. Today the wrapper puts `"<ExceptionType>: <message>"` into the errored layer's
internal evidence; that stays internal (it only reaches a result for a layer that fired,
which an errored layer cannot) and a test will pin that it never appears in the result.
Detail continues to go to the log.

A warning is logged **once per process** when the classifier is unavailable, instead of the
current per-load line that is easy to miss, with the action to take.

# 12. Lite-install behavior

CODE-READ and MEASURED.

| Question | Finding |
| --- | --- |
| Which layers remain | Eleven of twelve run: everything except the PAIR classifier. They are regex and heuristic layers with no ML dependency |
| Which imports fail | `joblib` first, inside the classifier loader. Then `numpy`, `onnxruntime`, `tokenizers` for the encoder |
| How it is swallowed | The loader catches `ImportError`, logs a warning, records an error string, returns `False`. The layer then returns "no signal" (§3) |
| Is `is_attack` meaningful | As a positive, yes: a block from a pattern layer is real. As a negative it is weak: the layer that carries 130 of 134 JailbreakBench detections did not run |
| Is `confidence` meaningful | For a block, it is the firing layers' confidence and is as meaningful as before. For an allow it is `0.0`, as always |
| Is `attack_type` meaningful | Yes when blocked. It is the pattern layer's type |
| Is the state recoverable | Not within the process: the load is attempted once. Installing the extras and restarting recovers it |
| Coverage status | Needed, and it is `partial` |

**Desired result for a benign prompt on a lite install:**

```text
is_attack        False
zone             "allow"
decided_by       "pipeline"
coverage.status  "partial"
coverage.classifier            "unavailable_dependency"
coverage.layers["pair_classifier"]  "unavailable_dependency"
degraded_layers  ["pair_classifier"]
models["pair_classifier"].loaded    False
```

The verdict does not change. What changes is that the result no longer claims a capability
that is not present. Whether a lite install should instead refuse to scan, or block, is a
policy question for WP-004 and WP-007 and is not decided here.

`scan_prompt_lite` (the explicit four-layer function) gets the same `coverage`, with the
eight layers it never runs reported as `disabled` and status `partial` always.

# 13. Serialization/API impact

- New method `ScanResult.to_dict()`: the single JSON-safe representation, fixed key order,
  every public field, no object that `json.dumps` cannot handle. `LiteScanResult` and
  `GuardResult` get the same.
- `fie detect --output json` and `fie explain --output json` add `zone`, `decided_by`,
  `coverage`, `models`, `degraded_layers`, `schema_version` to their existing keys. No key is
  removed or renamed, so a script reading today's keys keeps working (decision W3-8 covers
  switching both commands to `to_dict()` outright).
- Human-readable CLI output gains one line when coverage is not full.
- `GuardResult` gains `zone` and `coverage_status`, defaulted.
- Server: no `ScanResult` field reaches any API response today, and none is added. Persisting
  the zone with an inference record is decision W3-11; this plan recommends leaving the
  server alone until the scan API package.
- Harness records: §16.

# 14. Backward compatibility

| Item | Effect |
| --- | --- |
| Existing fields `is_attack`, `attack_type`, `category`, `confidence`, `layers_fired`, `matched_text`, `mitigation`, `evidence`, `layer_scores` | Unchanged in type, meaning and value, in every configuration |
| `degraded_layers` | Type unchanged. Value unchanged whenever the classifier is loaded. Gains `"pair_classifier"` when it is not. **Intended contract change**, documented |
| `is_degraded` | Follows `degraded_layers`; therefore becomes `True` on a lite install |
| New fields | All have defaults; positional construction of the existing ten fields still works |
| `ScanResult(...)` built by third-party code | Works; new fields default to a neutral "unknown" form (`coverage=None`, `zone` derived from `is_attack` as `allow` / `clear_block`, `models={}`) so old constructors do not claim anything |
| Golden test | Byte-identical: it records six existing fields with the classifier loaded |
| The 87 existing tests | Expected unchanged. No assertion depends on `degraded_layers` being empty without models |
| Space demo | Already prints a "degraded scan" note from `degraded_layers`; on a Space without models it would now show it, correctly |
| Legacy clients | Ignore fields they do not read. A client that treats `is_degraded` as an error will now see it on a lite install, which is the point |
| `scan_prompt_async` | Same object, same fields |
| Result schema version | `schema_version = 2` on the result and in `to_dict()`; absent means 1 |

# 15. Security analysis

| Field | Could it leak? | Control |
| --- | --- | --- |
| `coverage.layers`, `.status`, `.classifier`, `.optional` | Fixed codes only | Closed sets; a test builds a result under an exception whose message contains a path, a token-like string and an e-mail, and scans `to_dict()` for them |
| `models[*].version` | Published metadata | From the artifact's own metadata file |
| `models[*].digest` | Published hash, abbreviated | Hex characters only |
| `models[*].backend` | Class of encoder | Closed set |
| `zone`, `decided_by` | Routing outcome | Closed sets |
| Not added to the result | Paths, file names, the loader's error string, environment, host, tenant, session id, prompt text beyond what `evidence` already holds | — |

Public result metadata versus internal diagnostics: the result carries states and
identities; the reason text stays in logs and in `health()`.

Relation to WP-002: no `app/`, `engine/` or `storage/` file changes; no result field carries
tenant data; authentication, route policy and scoping are untouched. One exposure noticed
while reading, outside this package's files: `GET /health/deep` returns
`components.detector.detail`, which includes `_pair_state()["error"]`, a string with a local
directory path, to anonymous callers when the classifier is missing. It is listed as
decision W3-12.

# 16. Test strategy

New file `tests/test_scan_result_contract.py`; additions to `tests/evals/`. Everything local,
no network (the harness guard already enforces it for the suites).

| Area | Tests |
| --- | --- |
| Full pipeline | With models loaded: `coverage.status=="full"`, twelve layers `ok`, `models["pair_classifier"].loaded`, `version=="v6.3b"`, digest equals the first 16 hex of the manifest hash; `zone` present |
| Missing classifier | Loader state forced to each of the three unavailable reasons: status `partial`, the exact state, `"pair_classifier" in degraded_layers`, `models[...].loaded is False`, and **`is_attack`, `attack_type`, `confidence`, `layers_fired`, `layer_scores` equal to today's values for the same prompts** |
| "Available" is not inferred from imports | Packages importable but the model file absent → `unavailable_model`, not `ok`; classifier present but inference raising → `error` |
| Individual layer failure | One layer patched to raise an exception whose message holds a path and a secret-like string: state `error`, result valid, neither string anywhere in `to_dict()` |
| Timeout | One layer patched to sleep past a shortened deadline: state `timeout`, other layers `ok`, result returned |
| Disabled layer | `disabled_layers={"regex"}`: state `disabled`, status `partial`, not in `degraded_layers` |
| Lite install | Harness suite `lite.std`: **false-full-coverage 0 of 1,848**; counts identical to `BL-0001` (14 / 17 / 15 / 12 / 12 / 4 / 6) |
| Model identity | Forced `FIE_PAIR_VERSION=v6`: version and digest change accordingly; missing → `loaded False`, no version |
| Zone | Each of the four zones and five `decided_by` values produced by a deterministic input (fake tiebreaker for the two tiebreaker rows, environment for `config`, labelled hash for the override rows); invalid value raises; the `is_attack` ↔ zone invariant holds for every record of the baseline |
| Harness cross-check | For every scanned prompt, the public `zone` equals the harness's derived zone under the canonical profile; the result's digest equals the verified file hash |
| Backward compatibility | Golden file byte-identical; a `ScanResult` built with only the ten old fields still works and claims nothing |
| Sync / async | Same prompt through both: `to_dict()` equal |
| Serialization | `to_dict()` round-trips through `json`; key set and order fixed; CLI JSON contains the old keys plus the new ones; `LiteScanResult` and `GuardResult` agree on shared keys |
| Fast paths and cache | Labelled prompt → `bypassed` + `feedback_override`; a cached result keeps its fields |
| No network, no detector change | Static check that the diff adds no import of a network or ML package; `evals baseline` (below) |

**Baseline.** `python -m evals baseline` compares eleven record files with `BL-0001`. Ten come
from the canonical profile, where the classifier is loaded: they must stay **byte-identical**.
The eleventh, `lite.std`, will differ in exactly one record field, `degraded`
(`[]` → `["pair_classifier"]`), in every record. That is the intended change, and the command
will therefore exit with "baseline mismatch" until a new baseline is pinned. Procedure: run
`verify-determinism --all`; confirm by script that the ten files are identical and that
`lite.std` differs in `degraded` only; pin `BL-0002`; point `CANONICAL` at it; keep `BL-0001`.
No harness logic is redesigned: the self-report metric already reads `degraded`.

# 17. Acceptance criteria

| # | Criterion | Proof |
| --- | --- | --- |
| AC-1 | `ScanResult` has an explicit public `zone` (four values) and `decided_by` | Zone tests; invariant test |
| AC-2 | `ScanResult` exposes truthful coverage | Coverage tests for all seven layer states |
| AC-3 | A missing classifier cannot report full coverage | Three unavailable-reason tests |
| AC-4 | Lite false-full-coverage is 0 of 1,848 | Harness `lite.std` |
| AC-5 | Model identity is accurate and safe | Digest equals manifest prefix; forced-version test; no path in any field |
| AC-6 | Layer failure is represented | Error test |
| AC-7 | Timeout is represented | Timeout test |
| AC-8 | No exception text, secret or path in a public field | Planted-string scan of `to_dict()` |
| AC-9 | Sync and async agree | Equality test |
| AC-10 | Serialization is deterministic | Fixed key order; round-trip |
| AC-11 | Existing fields are compatible | Golden file byte-identical; old-constructor test |
| AC-12 | Canonical guard baseline unchanged | The ten canonical-profile record files byte-identical to `BL-0001`; all eight counts equal. `lite.std` differs in `degraded` only, by design |
| AC-13 | Existing tests green | 87 existing, 230 harness (plus adapted contract tests listed in §18), 318 security |
| AC-14 | No network dependency introduced | Static check; harness guard: 0 non-canary events |
| AC-15 | No detector-quality work | Diff review: no change to any layer's return values, thresholds, weights or models; verdict-equality tests |
| AC-16 | *(new)* Verdicts are identical with the classifier missing, before and after | Lite counts equal `BL-0001`; per-record equality of every field except `degraded` |
| AC-17 | *(new)* `is_attack` and `zone` can never disagree | Enforced at construction; checked over all baseline records |
| AC-18 | *(new)* The harness no longer needs a private rule for the zone | Adapter reads the public field and cross-checks the old rule |
| AC-19 | *(new)* The unavailable-classifier warning is logged once per process | Log-capture test |
| AC-20 | *(new)* Scope: only the files in §18 changed | `git diff --stat` |

# 18. File change matrix

**MODIFY**

| File | Responsibility | Problem | Change | Compatibility risk | Security risk | Tests | Rollback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fie/adversarial.py` | Result type, layer runner, router | Result cannot express coverage, zone or model | Add `ScanCoverage`, `ModelIdentity`, the new `ScanResult` fields and `to_dict()`; `LayerStatus` gains the unavailable and disabled states; `_run_layer_safe` maps `LayerUnavailable`; `_layer_pair` raises it when the classifier is not usable; `_run_all_layers_parallel` returns a state for disabled layers; the eight construction sites pass zone, `decided_by`, coverage, models. No change to aggregation or thresholds | Low: additive fields | None: fixed codes | All of §16 | Revert file |
| `fie/layers/pair.py` | Classifier and meta-classifier loading | Loaded artifact has no recorded identity; failure reason is free text | Record version and digest at load; expose a reason **code** next to the existing error string; `_pair_identity()`, `_meta_identity()` accessors. Inference unchanged | Low | Digest and version only | Identity tests; golden test | Revert file |
| `fie/onnx_encoder.py` | Encoder loading | Same | Record digest and backend at load (subject to W3-7) | Low | Same | Identity tests | Revert file |
| `fie/_lite.py` | Four-layer scanner | Own result type without coverage | Add `coverage`, `zone`, `to_dict()` | Low | None | Lite tests | Revert file |
| `fie/preflight.py` | Guard wrapper | Drops coverage and zone | `GuardResult` gains `zone`, `coverage_status`; `_safe_scan` passes them | Low: defaults | None | Guard tests | Revert file |
| `fie/__main__.py` | CLI | JSON omits coverage | Add the new keys to both JSON outputs; one line in text output when not full | Low: keys added only | None | CLI tests | Revert file |
| `fie/__init__.py` | Exports | — | Export `ScanCoverage`, `ModelIdentity` | None | None | Import test | Revert file |
| `evals/subject.py` | The harness's only `fie` adapter | Derives the zone privately | Read the public zone; keep `derive_zone` as a cross-check that fails the run on disagreement; compare the result's digest with the verified hash. Record format unchanged | Records for `lite.std` change in `degraded` | None | `tests/evals/test_subject_contract.py` | Revert file |
| `tests/evals/test_subject_contract.py` | Private-name contract test | Lists names that become unnecessary | Update the list if it shrinks (**an existing test file; needs approval**, W3-13) | — | — | — | Revert |
| `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Log | — | At implementation | — | — | — | — |

**CREATE**

| File | Purpose |
| --- | --- |
| `tests/test_scan_result_contract.py` | §16 |
| `evals/baselines/BL-0002_…/` and updated `evals/baselines/CANONICAL` | The re-pinned baseline (§16) |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_003_truthful-scan-result.md` | Report |

**DELETE:** none. **Not touched:** `app/`, `engine/`, `storage/`, `deploy/`, `Frontend/`,
models, manifests, dependency files, `README.md`, existing tests other than the one named.

Estimated size: about 250 production lines, 400 test lines.

# 19. Step-by-step implementation sequence

| Step | Work | Files | Expected result | Acceptance | Rollback point |
| --- | --- | --- | --- | --- | --- |
| 0 | Baseline: record HEAD, 635 tests, `evals baseline` exit 0. Write the contract tests first; record which fail on today's code | new test file | Coverage, zone and identity tests fail; compatibility tests pass | Failures are the ones predicted in §3 | Delete the test file |
| 1 | Result contract: types, new fields with neutral defaults, `to_dict()`, exports | `adversarial.py`, `__init__.py` | Old-constructor and serialization tests pass; golden file identical | AC-10, AC-11 | Revert two files |
| 2 | Coverage semantics: layer states, `LayerUnavailable`, disabled layers, `coverage`, `degraded_layers` | `adversarial.py` | Coverage tests pass; verdict-equality tests pass | AC-2, 3, 6, 7, 8, 16 | Revert |
| 3 | Zone and `decided_by` at the eight construction sites | `adversarial.py` | Zone tests pass | AC-1, 17 | Revert |
| 4 | Model identity | `pair.py`, `onnx_encoder.py`, `adversarial.py` | Identity tests pass | AC-5 | Revert |
| 5 | Propagation: pre-flight, lite, CLI | `preflight.py`, `_lite.py`, `__main__.py` | Guard, lite, CLI, sync/async tests pass | AC-9 | Revert |
| 6 | Harness adapter cross-checks | `evals/subject.py`, its contract test | 230 harness tests pass | AC-18 | Revert |
| 7 | Regression: full suite; `verify-determinism --all`; diff of all eleven record files against `BL-0001` | — | Ten identical; `lite.std` differs in `degraded` only; false-full-coverage 0 of 1,848 | AC-4, 12, 13, 14, 16 | — |
| 8 | Pin `BL-0002`; execution report; master log | `evals/baselines/`, docs | `evals baseline` exit 0 against `BL-0002` | AC-12, 20 | Remove the new baseline directory, restore `CANONICAL` |

The assistant commits nothing; each rollback point is the working tree at the end of a step.

**Stop conditions.** Stop and report if: a detector's behaviour must change; a threshold must
change; a model must be retrained; a dependency must be added; network access is needed; any
of the ten canonical-profile record files differs from `BL-0001`, or `lite.std` differs in
anything but `degraded`; an existing API contract must be broken beyond the `degraded_layers`
change approved here; a field would carry sensitive information; the changes in `fie/` grow
beyond result semantics; the work starts to become multilingual, long-input or
adaptive-attack work; a file outside §18 must change.

# 20. Rollback plan

Revert the listed files; restore `evals/baselines/CANONICAL` to `BL-0001`. Nothing is
persisted anywhere by this package: no stored data, no cache format, no database field.
A result object from the new code is never written to a store that old code reads.

# 21. Performance impact

| Item | Cost |
| --- | --- |
| Per scan | Building a twelve-entry dict and two small frozen objects: microseconds. Budget: no measurable change in the harness latency suite's mean (24.5 ms at 46 characters) beyond run-to-run noise |
| Once per process | SHA-256 of the classifier and meta-classifier files: milliseconds. Encoder (90 MB): a few hundred milliseconds during load or warm-up (W3-7) |
| Cache | Unchanged |

# 22. Deployment considerations

- SDK: additive fields; a patch or minor release when the owner chooses. No packaging change.
- Server and Space: no server code changes. The Space's demo will show its existing
  "degraded scan" note if it ever boots without models, which today it would hide.
- Harness: `CANONICAL` moves to `BL-0002`. Anyone comparing against `BL-0001` must know that
  the lite suite's `degraded` field changed by design.
- No database, no migration, no secret, no environment variable.

# 23. Documentation impact

Execution report and master log at implementation. The `ScanResult` docstring is rewritten
to describe the contract. README, FACT_SHEET and the hosted demo text are not edited in this
package; the claims ledger in the audit lists what should change there later.

# 24. Risks

| # | Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | A verdict changes by accident while the layer wrapper is edited | Low | High | Verdict-equality tests with the classifier present and absent; ten canonical record files must stay byte-identical; golden file |
| R2 | A consumer breaks on `degraded_layers` becoming non-empty in a lite install | Low | Medium | It is the documented meaning of the field; only lite installs are affected; decision W3-2 offers the alternative |
| R3 | Re-pinning the baseline hides an unintended change | Low | High | The re-pin is allowed only after a scripted diff shows the single expected field difference |
| R4 | The contract grows into an observability schema | Medium | Medium | Three objects, closed value sets, nothing free-text |
| R5 | Identity hashing slows cold start | Low | Low | Measured in step 4; encoder hashing is optional |
| R6 | A cached result is mistaken for a fresh statement about coverage | Low | Low | Coverage describes the scan that produced the verdict, which is what was cached; documented |
| R7 | Third-party code subclasses or reconstructs `ScanResult` | Low | Low | Defaults for every new field |
| R8 | The optional-component states (translation, tiebreaker) are read as coverage | Medium | Low | They live under `optional`, separate from `layers`; W3-3 |

# 25. Open decisions

Listed with recommendations in §26. None is hidden in an implementation detail.

# 26. Exact owner decisions required

| # | Decision | Recommendation | Alternative |
| --- | --- | --- | --- |
| W3-1 | Zone values | Four: `allow`, `uncertain_allow`, `uncertain_block`, `clear_block`; plus `decided_by` with five values | The harness's three values and no second field |
| W3-2 | `degraded_layers` gains layers that are unavailable | Yes: it makes the documented meaning true and makes the WP-001 metric read 0 without touching the harness | Leave the field as it is and report only through `coverage`; the harness metric and every record format would then have to change |
| W3-3 | Coverage shape | An object: `status` (`full` / `partial` / `bypassed`), `layers`, `classifier`, `optional`. Optional components do not lower the status | A single enum; or a boolean; or count optional components toward `partial` |
| W3-4 | May a degraded result have `is_attack=False`? | Yes. Report, do not change the verdict. Blocking on missing coverage is a policy for WP-004 / WP-007 | Block or raise when the classifier is unavailable |
| W3-5 | Are `confidence` and `attack_type` meaningful without the classifier? | Yes for a block (they come from the layers that fired); unchanged. Documented as "from the layers that ran" | Null them out under partial coverage |
| W3-6 | Model digest length | 16 hex characters | Full 64 |
| W3-7 | Hash the encoder file | Yes, once at load | Classifier and meta-classifier only |
| W3-8 | CLI JSON | Keep today's keys, add the new ones | Replace both outputs with `to_dict()` (changes key sets) |
| W3-9 | `risk_score` (roadmap): a non-zero strength for allowed prompts that had a signal | Include: one additive float, the pre-routing aggregated confidence. It answers "how strong is the evidence" for an allow, which `confidence=0.0` hides | Defer |
| W3-10 | `decision` field and the warning for the inert `threshold=` argument (roadmap) | Defer both. `decision` duplicates `zone`. The inert argument is a control that does nothing, not a result field; it fits the policy package | Include |
| W3-11 | Persist `zone` with server inference records / return it from `/monitor` | No in this package; the server is untouched | Add fields to `GuardResult` consumers in `app/` |
| W3-12 | `/health/deep` returns the loader's error string, with a local path, to anonymous callers when the classifier is missing | Fix in this package by reporting the new reason code instead (one line in `app/main.py`, which is otherwise out of scope) | Defer to a later package |
| W3-13 | Edit `tests/evals/test_subject_contract.py` if the private-name list shrinks | Yes, that file only | Keep the adapter's private reads and leave the test alone |
| W3-14 | Result schema version | `schema_version = 2` on the result | None |
| W3-15 | Re-pin the baseline as `BL-0002` under the procedure in §16 | Yes | Keep `BL-0001` canonical and teach `evals baseline` to ignore `degraded` in the lite suite (a harness change) |
| W3-16 | `scan_prompt_lite` and `GuardResult` get the same fields | Yes | `ScanResult` only |
| W3-17 | Public versus internal | Public: states, zone, `decided_by`, version, digest, threshold, backend. Internal (logs, `health()`): reason text, paths, exception detail | — |
| W3-18 | How legacy clients read the new fields | They ignore them. `is_degraded` becoming `True` on a lite install is the only visible change, and is intended | — |

# 27. Final recommendation

Proceed with WP-003 as planned above, with the recommendations in §26.

The package is small because the defect is small: one layer reports "nothing found" when it
means "did not run". Fixing that at the source, giving the result a state per layer, a zone
and the identity of what was loaded, and re-pinning the one harness file that is supposed to
change, takes the false-full-coverage figure from 1,848 of 1,848 to 0 of 1,848 without
moving a single verdict.
