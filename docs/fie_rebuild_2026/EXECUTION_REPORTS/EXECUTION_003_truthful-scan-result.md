# EXECUTION_003 — Truthful Scan Result (WP-003)

| | |
| --- | --- |
| Work package | WP-003 |
| Status | **COMPLETE WITH ONE OWNER-GATED STEP AND DOCUMENTED DEVIATIONS — awaiting owner review.** Everything is implemented and verified except pinning `BL-0002`, which the harness refuses to do from an uncommitted tree (section K, DV-5) |
| Commit status | **NOT COMMITTED. The owner reviews and commits manually.** Nothing was staged, committed, pushed, stashed, reset or tagged |
| Date | 2026-10-10 |
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD (unchanged throughout) | `550226da5691ada35a19835059ec17e22632d2c6` — the owner's commit of the WP-003 plan |
| Contract | [PLAN_003_TRUTHFUL_SCAN_RESULT.md](../IMPLEMENTATION_PLANS/PLAN_003_TRUTHFUL_SCAN_RESULT.md), decisions W3-1 to W3-18 as approved, plus the owner's fast-path clarification (`bypassed`) |
| Deployed? | **No.** Nothing was deployed, no live service was called, no production database was touched |

What this report does **not** claim: that FIE is secure, or that its detector is more accurate.
This package changes what a result *says about itself*. It was built so that no verdict moves,
and the evidence below is that none did.

---

## A. Executive summary

**The defect.** When the PAIR classifier could not load, its layer returned "no signal" and the
scan result reported `degraded_layers == []`. A scan that had skipped the layer carrying most
detections was indistinguishable from a full scan. WP-001 measured it: 1,848 of 1,848 scans in
the base-install simulation reported full coverage.

**What changed.** Every scan result now states:

1. **what ran** — `coverage.layers`, one fixed state for each of the twelve layers;
2. **what was available** — `coverage.status`: `full`, `partial` or `bypassed`, plus the optional
   components (meta-classifier, tiebreaker, translation) reported separately;
3. **how the verdict was reached** — `zone` (four values) and `decided_by` (five values);
4. **which artifacts were loaded** — `models`, with the declared version and a 16-character
   SHA-256 prefix of each file that was actually loaded.

`degraded_layers` now includes a layer that was unavailable, so its documented meaning is true.
The same fields reach `LiteScanResult`, `GuardResult` and both CLI JSON outputs. The harness
reads the public zone and cross-checks it. `/health/deep` no longer returns a local directory
path to anonymous callers.

**Result.**

| Question | Answer |
| --- | --- |
| Did any detection verdict change? | **No.** 56 literal comparisons in tests; 8,032 canonical-profile harness records byte-identical to `BL-0001`; 1,848 lite records identical in every field except `degraded` (sections H, J) |
| Is the canonical baseline intact? | **Yes.** The ten canonical-profile record files are byte-identical to `BL-0001`; all eight counts are equal. `BL-0001` is untouched and `CANONICAL` still points at it |
| Lite false-full-coverage | **0 of 1,848** (was 1,848 of 1,848) ; every lite record differs from `BL-0001` in `degraded` only |
| Path disclosure in `/health/deep` | **Fixed**, one block in `app/main.py`, with a regression test |
| Tests | **785 passed, 0 failed, 0 skipped** (87 existing + 239 harness + 318 security + 141 new contract) |
| Scope | Only the approved files changed (section M) |
| `BL-0002` | **Not pinned.** Proven ready; the pin command requires a committed tree. Three commands for the owner are in section J |

---

## B. Starting state

| Item | Value |
| --- | --- |
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD | `550226da5691ada35a19835059ec17e22632d2c6` ("docs/fie_rebuild_2026/IMPLEMENTATION_PLANS/PLAN_003_TRUTHFUL_SCAN_RESULT.md") |
| Previous commit | `b077aea` (WP-002) |
| Working tree | Clean |
| Baselines present | `BL-0001_pair-v6.3b_fie-5d5ed90d`, `REF-0001_pair-v6.2_fie-5d5ed90d`; `CANONICAL` → `BL-0001…` |
| Interpreter | Python 3.10.19, conda env `failure-engine` |
| Tests before any change | **635 passed, 0 failed, 0 skipped** (87 existing + 230 harness + 318 security) |
| `python -m evals baseline` before any change | **Exit 0.** Eight counts equal; 11 record files byte-identical to `BL-0001` (run `20261010T061718Z_5d5ed90d_d233a245`) |

**Phase 0, tests first.** `tests/test_scan_result_contract.py` was written before any production
file was touched and run against the unmodified code: **67 passed, 70 failed** (137 tests at
that point).

| On the unmodified code | Tests |
| --- | --- |
| **Passed** — the behaviour this package must not move | all 8 verdicts with the classifier loaded; all 24 with it missing (8 prompts × 3 reasons); 8 uncertain-band routing outcomes; 8 lite verdicts; 8 pre-flight verdicts; legacy constructor; decision constants; layer return contracts; import guard (7 files); `/health/deep` with the classifier loaded |
| **Failed** — the defects | `assert 'pair_classifier' in []` for each of the three unavailable reasons; `ScanResult` has no `zone` (first failure in 20 tests), `coverage` (14), `models` (3) or `to_dict` (6); `ScanCoverage` / `ModelIdentity` do not exist; `LiteScanResult` and `GuardResult` carry neither; CLI JSON has no new keys; `/health/deep` returned the loader's text, e.g. `no PAIR classifier found in C:\Users\…\no-models-here — run …` |

The file was refined after Phase 0: four tests added (loading race, state codes, copy/pickle,
concurrency), one test restructured because it undid shared fixtures, the two `/health/deep`
tests given an isolating fixture, and assertions added for canonical serialization. No
assertion was weakened or removed, and no failing case was deleted. Final count: 141.

---

## C. Files

**Created**

| File | Lines | Purpose |
| --- | --- | --- |
| [tests/test_scan_result_contract.py](../../../tests/test_scan_result_contract.py) | 1,272 | 141 contract tests |
| This report | — | — |

**Modified — production**

| File | + / − | Change |
| --- | --- | --- |
| [fie/adversarial.py](../../../fie/adversarial.py) | 551 / 23 | `ScanCoverage`, `ModelIdentity`, closed value sets, new `ScanResult` fields, invariant, `to_dict()`; layer states; `_pair_task`; coverage builder; once-per-process warning; `zone` / `decided_by` at the eight construction sites; `model_identities()`, `classifier_state()` |
| [fie/layers/pair.py](../../../fie/layers/pair.py) | 146 / 2 | Digest and declared version recorded at load; fixed reason code beside the existing error text; non-blocking state and identity accessors; per-call "did the meta-classifier really run" flag |
| [fie/onnx_encoder.py](../../../fie/onnx_encoder.py) | 44 / 0 | Digest of `model.onnx` once per file per process; `identity()` (no path) |
| [fie/_lite.py](../../../fie/_lite.py) | 58 / 1 | `LiteScanResult` gains `zone`, `decided_by`, `coverage`, `schema_version`, `to_dict()` |
| [fie/preflight.py](../../../fie/preflight.py) | 49 / 5 | `GuardResult` gains `zone`, `coverage_status`, `schema_version`, `to_dict()` |
| [fie/__main__.py](../../../fie/__main__.py) | 29 / 0 | Six keys appended to both JSON outputs; one text line when coverage is not full |
| [fie/__init__.py](../../../fie/__init__.py) | 6 / 1 | Exports `ScanCoverage`, `ModelIdentity` |
| [app/main.py](../../../app/main.py) | 5 / 0 | The W3-12 fix, in the detector block of `health_deep` only |

**Modified — harness and its test**

| File | + / − | Change |
| --- | --- | --- |
| [evals/subject.py](../../../evals/subject.py) | 72 / 8 | Records the public `zone`; stops the run if it disagrees with the old rule; compares each result's digest with the verified file; `identity_report()`. Record format unchanged |
| [tests/evals/test_subject_contract.py](../../../tests/evals/test_subject_contract.py) | 92 / 1 | Nine tests added, one assertion added (W3-13). No existing assertion changed |

**Documentation:** this file; a new §11 and a session-log row in
`MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md`.

**Deleted:** none. **Not created:** `evals/baselines/BL-0002…` and the `CANONICAL` update
(section J).

**Not touched:** `engine/`, `storage/`, `deploy/`, `Frontend/`, model files, manifests,
dependency files, `README.md`, `fie/multilingual.py`, every other file in `app/` and `fie/`,
`tests/security/`, the other existing tests, the golden file.

Every removed line in `fie/adversarial.py` (23) is a comment, a docstring, or a statement
replaced by an equivalent one. No line of `_weighted_aggregate`, `_ATTACK_THRESHOLDS`,
`_LAYER_WEIGHTS`, `_FAST_PATH_LAYERS`, `_DOMAIN_MULTIPLIERS` or the routing comparisons
changed; a test pins the four constant tables.

---

## D. Implementation

**The root cause was one line.** `_run_pair_classifier` begins
`if not _load_pair_classifier(): return None, 0.0, {}`. That line is unchanged, because the
multilingual layer and two evaluation scripts call the function directly and rely on it.

**How "unavailable" is reported without touching a value.** The scanner now runs the
classifier layer through `_pair_task`:

```text
usable = _load_pair_classifier()        # the same call the layer makes first
out    = _layer_pair(prompt)            # unchanged; its values are returned as they are
if usable: return out
return the same tuple, tagged with the reason
```

The tag rides on a tuple subclass (`_LayerOutput`). It unpacks and compares as a plain tuple,
so the three values a layer returns are used exactly as before. `_run_layer_safe` reads the
tag into `LayerResult.status`. Aggregation, the meta-classifier's input vector and routing
never see the tag. This is what makes the verdict identical by construction rather than by
test: the code path that produces the numbers is the one that was already there.

**Reason codes.** The loader sets a fixed code in each of its three failure branches, next to
the error text it already kept:

| Loader branch | Code | Layer state |
| --- | --- | --- |
| `ImportError` | `dependency` | `unavailable_dependency` |
| No classifier file found | `model` | `unavailable_model` |
| Any other exception | `load_failed` | `unavailable_load_failed` |

**Coverage is derived, then checked.** `_coverage_for` builds the layer map from what the layer
runner returned. `ScanCoverage` recomputes the status from the map on construction and refuses
a mismatch, an unknown value, a missing layer, or free text. A coverage object cannot say
`full` while a layer did not run.

**The zone is assigned in one place.** The router at the end of `scan_prompt` passes `zone` and
`decided_by` at each of its six outcomes; the two fast paths pass theirs. `ScanResult` refuses
to be constructed when `is_attack` and `zone` disagree.

**Identity is hashed from the file, at load.** The classifier and meta-classifier files are
hashed immediately before `joblib.load`; the encoder file after the ONNX session is built. The
version comes from the artifact's own metadata file and must match a short-token pattern, or it
is dropped.

**Warning.** One `WARNING` per process on logger `fie.adversarial` when a scan runs without the
classifier (`scan coverage=partial reason=classifier_unavailable state=…`). It names the state
code and the action, and no path.

**Order of work.** Phases 1 to 4 were applied together, because the scanner imports the
loader's new accessors, then verified by test group; Phase 5 separately. After Phases 1–4: 123
of 137 contract tests passed, the 14 failures all being Phase 5 items. After Phase 5: all pass.

---

## E. Coverage contract

```text
result.coverage            ScanCoverage (read-only)
    .status                "full" | "partial" | "bypassed"
    .layers                {layer: state}, all twelve, always, in pipeline order
    .classifier            state of "pair_classifier"
    .optional              {"meta_classifier": …, "tiebreaker": …, "translation": …}
result.degraded_layers     list[str]
result.schema_version      2
```

**Layer states**

| State | Meaning | In `degraded_layers` | Before this package |
| --- | --- | --- | --- |
| `ok` | Ran and returned a normal result | No | `ok` |
| `unavailable_dependency` | A package the layer needs cannot be imported | **Yes** | reported as `ok` |
| `unavailable_model` | The model file is not present | **Yes** | reported as `ok` |
| `unavailable_load_failed` | The model exists and failed to load | **Yes** | reported as `ok` |
| `error` | Raised while running | Yes | same |
| `timeout` | Had not produced an answer when the scan needed it | Yes | same |
| `disabled` | Removed from this scan by the caller | No | no trace |
| `bypassed` | No layer ran: a labelled-prompt fast path decided | No | no trace |

**Status**

| `coverage.status` | Rule |
| --- | --- |
| `full` | All twelve layers are `ok` |
| `partial` | At least one layer is not `ok`, for any reason including `disabled` |
| `bypassed` | All twelve layers are `bypassed` |

**Optional components** — reported, never lower the status (W3-3)

| Component | Values |
| --- | --- |
| `meta_classifier` | `ok` · `unavailable` (not loaded, or the call failed) · `disabled` (`FIE_DISABLE_META`) · `bypassed` (fast path) |
| `tiebreaker` | `not_needed` · `confirmed` · `cleared` · `unavailable` (asked, no verdict) · `disabled` (`use_llama_guard=False`) |
| `translation` | `not_needed` · `ok` · `unavailable` |

**Fast-path clarification (owner instruction).** A prompt with a stored human label is decided
without running any layer. Marking the layers `ok` would say they looked and found nothing;
`disabled` would say the caller switched them off. Neither is true. Such a result has
`coverage.status = "bypassed"`, all twelve layers `bypassed`, `classifier = "bypassed"`,
`decided_by = "feedback_override"`, and `zone` `allow` or `clear_block`. `bypassed` is
whole-scan only: `ScanCoverage` refuses a map in which some layers are `bypassed` and others
are not, and a test confirms it never appears on a scan that ran. Its existing fields are
unchanged, including `degraded_layers == []`. One judgement call is recorded as DV-3: the
meta-classifier entry under `optional` is also `bypassed` in such a result.

**A base install, benign prompt** (measured, `joblib` unimportable):

```text
is_attack False   zone "allow"   decided_by "pipeline"
degraded_layers ["pair_classifier"]
coverage.status "partial"   coverage.classifier "unavailable_dependency"
coverage.optional {"meta_classifier": "unavailable", "tiebreaker": "not_needed", "translation": "not_needed"}
models: all three roles loaded=False, nothing else
```

The verdict is the same as before. What changed is that the result no longer claims a
capability that is not present (W3-4).

---

## F. Zone and decision source

| Production path | `is_attack` | `zone` | `decided_by` | `optional.tiebreaker` |
| --- | --- | --- | --- | --- |
| Below the safe ceiling, or no layer fired | False | `allow` | `pipeline` | `not_needed` |
| At or above the threshold | True | `clear_block` | `pipeline` | `not_needed` |
| Uncertain band, tiebreaker confirms | True | `uncertain_block` | `tiebreaker` | `confirmed` |
| Uncertain band, tiebreaker clears | False | `uncertain_allow` | `tiebreaker` | `cleared` |
| Uncertain band, tiebreaker gave no verdict, default policy | True | `uncertain_block` | `fail_secure` | `unavailable` |
| Uncertain band, tiebreaker off, default policy | True | `uncertain_block` | `fail_secure` | `disabled` |
| Uncertain band, no verdict, `FIE_UNCERTAIN_ALLOW=1` | False | `uncertain_allow` | `config` | `unavailable` / `disabled` |
| Labelled false positive (fast path) | False | `allow` | `feedback_override` | `not_needed` |
| Labelled attack (fast path) | True | `clear_block` | `feedback_override` | `not_needed` |

A tiebreaker verdict wins over `FIE_UNCERTAIN_ALLOW`, as it always did; two test rows pin that.

Invariant, enforced in `ScanResult.__post_init__`:
`is_attack == (zone in {"uncertain_block", "clear_block"})`.

Each row is produced by a deterministic test: the uncertain band by fixing the aggregate at
0.50 for `JAILBREAK_ATTEMPT` (band 0.39 to 0.65) with a stand-in tiebreaker; the override rows
by a labelled hash; the first two by real prompts. A separate test asserts that all four zones
and all five sources were reached. For each uncertain row a second test asserts the
*pre-existing* fields (`is_attack`, `confidence`, `attack_type`, the `llama_guard` evidence
marker) and passed on the unmodified code.

`confidence` and `attack_type` keep their meaning (W3-5). `risk_score`, a `decision` field and
the warning for the inert `threshold=` argument were **not** added (W3-9, W3-10).

---

## G. Model identity and hashing

```text
result.models   {"pair_classifier": ModelIdentity, "meta_classifier": …, "encoder": …}

ModelIdentity (frozen, validated)
    loaded      bool
    version     from the artifact's metadata, e.g. "v6.3b"; None if not declared
    digest      first 16 hex characters of the SHA-256 of the file that was loaded
    threshold   classifier roles only
    backend     encoder only: "onnx" | "sentence-transformers"
```

**Measured on this machine, shipped artifacts:**

| Role | Version | Digest | Manifest SHA-256 begins | Threshold | Backend |
| --- | --- | --- | --- | --- | --- |
| `pair_classifier` | `v6.3b` | `9c682b285a9fa205` | `9c682b285a9fa205…` | 0.5 | — |
| `meta_classifier` | none declared | `be6673d094a879f9` | `be6673d094a879f9…` | 0.41 | — |
| `encoder` | none declared | `57eb46cc82cd048d` | `57eb46cc82cd048d…` | — | `onnx` |

Tests compare each digest with an independent hash of the file and with
`scripts/model_manifest.json`. With `FIE_PAIR_VERSION=v6` forced, the identity becomes version
`v6.2` with the digest of that file. A role that is not loaded reports `loaded=False` and
nothing else; `ModelIdentity` refuses a version or digest on an unloaded role.

**Hashing cost (W3-7).**

| Artifact | Size | Time to hash |
| --- | --- | --- |
| Classifier | 11,349 bytes | 0.6 ms |
| Meta-classifier | 212,131 bytes | 0.8 ms |
| Encoder `model.onnx` | 90,293,328 bytes | 106 ms in isolation (file in the OS cache) |

**Cold-start impact, measured.** Ten interleaved pairs of fresh interpreters, `warmup()` timed,
identity hashing as shipped versus stubbed out by the measuring script:

| | Median | Min | Max |
| --- | --- | --- | --- |
| Hashing on (as shipped) | 2.265 s | 2.115 s | 2.429 s |
| Hashing off | 2.008 s | 1.890 s | 2.257 s |
| **Difference** | **+0.257 s** | +0.225 s | — |

About a quarter of a second, once per process, roughly 12% of warm-up on this machine. It is
more than the 106 ms of pure hashing because a fresh process also pays to read the file. The
digest is cached per (path, size, modification time), so the server's second encoder instance
of the same file does not hash it again (0.16 ms lookup). This is the plan's estimate ("a few
hundred milliseconds") at its upper end. The owner approved hashing the encoder (W3-7); if the
cost is unwelcome on a constrained host it can be removed without touching anything else, at
the price of `encoder.digest` being `None`.

**Limits.** The hash is taken just before the load, not atomically with it. The torch fallback
encoder loads a directory from the Hugging Face cache, so it reports `backend` with no digest.

---

## H. Verdict equivalence, before and after

The reference values were captured from the unmodified code at `550226d` before any change
(`scan_prompt(prompt, use_llama_guard=False)`, empty result cache) and are literals in the test
file. Compared fields: `is_attack`, `attack_type`, `confidence` to four decimals,
`layers_fired`, and every layer's score.

| Evidence | Scope | Result |
| --- | --- | --- |
| Captured literals, classifier loaded | 8 prompts: benign ×2, injection, DAN, harmful, soft-harm, copyright, many-shot | 8 of 8 identical |
| Captured literals, classifier missing | the same 8 × 3 unavailable reasons, driven through the **real** loader | 24 of 24 identical |
| Uncertain-band routing | 8 rows (tiebreaker confirms / clears / down / off, with and without `FIE_UNCERTAIN_ALLOW`) | 8 of 8 identical |
| Fast paths | whitelisted, known attack | identical |
| `scan_prompt_lite` | 8 prompts | 8 of 8 identical |
| `preflight_check` | 8 prompts | 8 of 8 identical |
| Concurrency | 32 scans on 8 threads | 32 of 32 identical |
| Golden file `tests/data/detection_golden.json` | 22 prompts, 6 fields each | File not modified; test passes |
| The 635 pre-existing tests | by name | All 635 pass; none renamed, none removed |
| Harness, canonical profile | 8,032 records in 10 files | Byte-identical to `BL-0001`, all 10 files |
| Harness, lite profile | 1,848 records | Every field identical except `degraded`, in all 1,848 |

The verdict tests **passed on the unmodified code too** (Phase 0). They are a fixed reference,
not something written to fit the new code.

`soft_harm` ("How can I secretly track my partner's phone…") is blocked at 0.95 with the
classifier and allowed without it, before and after. This package reports that difference; it
does not close it (W3-4).

**Would the tests notice a mistake?** Nine deliberate faults were introduced one at a time, the
named tests run, and the file restored and verified by SHA-256.

| # | Fault introduced | Tests that failed |
| --- | --- | --- |
| M1 | An unusable classifier is reported as `ok` (the original defect) | 7 |
| M2 | A fail-secure block is labelled `clear_block` | 3 |
| M3 | An allowed prompt is given a block zone | 2 (construction refuses) |
| M4 | Digest taken from the wrong end of the hash | 4, including the harness adapter test |
| M5 | File name offered as the version | 3 (validation drops it) |
| M6 | One threshold moved by 0.05 | 1 |
| M7 | A missing classifier made to block | 24 |
| M8 | The `/health/deep` fix removed | 3 |
| M9 | Exception text written into an errored layer's state | 0 — see below |

M9 is not a gap. The coverage builder maps any value that is not a known state to `error`, so
the planted text never reached the result and the leak test correctly still passed. The fault
was absorbed by the design rather than detected by a test.

---

## I. Tests

**Commands** (CI-equivalent environment; every key in `.env` blanked; no real credential):

```text
python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA
python -m evals baseline
python <scratch>/wp3_diff_records.py <run> evals/baselines/BL-0001_pair-v6.3b_fie-5d5ed90d
python -m evals verify-determinism --all
```

**Counts**

| Suite | Before | After |
| --- | --- | --- |
| Existing tests (`tests/test_*.py` before this package) | 87 | 87 |
| Harness tests (`tests/evals/`) | 230 | 239 (+9) |
| Security tests (`tests/security/`) | 318 | 318 |
| Contract tests (`tests/test_scan_result_contract.py`) | — | 141 |
| **Total** | **635 passed** | **785 passed, 0 failed, 0 skipped, 0 errors** |

Every one of the 635 earlier test names still passes.

**What the 141 contract tests cover**

| Area | Tests |
| --- | --- |
| Result contract: exports, legacy constructor, closed sets, invariant, validated value objects | 10 |
| Coverage: full; missing classifier ×3; importable-but-no-model; loaded-but-raises; loading race; state codes; layer error; timeout; disabled layer; disabled classifier; translation ×2; meta-classifier; warning once ×3 | 18 |
| Verdicts and zones: loaded ×8, missing ×24, real-prompt zones ×8, uncertain band ×16, all values reachable | 57 |
| Fast paths (`bypassed`) | 5 |
| Model identity: describes the files, matches the manifest, follows a forced version, no path | 4 |
| Serialization: key order, canonical bytes, non-JSON evidence, copy / pickle / `asdict`, concurrency, sync = async, cache | 7 |
| Propagation: lite ×10, guard ×13, shared keys, CLI ×3 | 27 |
| `/health/deep` | 4 |
| Scope guards: no new network or ML import ×7, decision constants, layer return contracts | 9 |
| Planted path / token / e-mail in an exception message, searched for in every public output | at nine points in the tests above |

**No network.** Every contract test runs under a guard that fails it if a non-loopback
connection is opened; none was. The tiebreaker and translator are local stand-ins. A static
test pins the network-capable and ML imports of the seven touched `fie/` files to what they
were; this package added `hashlib` and `re` (standard library) and nothing else. No dependency
was added.

---

## J. Baseline preservation

**The run.** `python -m evals baseline` on the final tree: run
`20261010T070206Z_70de7053_090da7e3`. All thirteen suites completed; both workers exited 0; the
network guard recorded 0 violations in the orchestrator and in each worker. The `fie` tree
hash is now `70de7053` (was `5d5ed90d`).

The command exited with code **6, "baseline mismatch", by design**. Its output:

```text
standard counts match the approved canonical counts
per-prompt records differ from BL-0001_pair-v6.3b_fie-5d5ed90d in: ['lite.std.jsonl']
```

Plan §16 predicted exactly this: `evals baseline` compares eleven files with `BL-0001` and one
of them is supposed to change. The run is flagged `canonical=False` for one reason,
"uncommitted changes in measured paths" (D-013).

**Scripted record-by-record comparison with `BL-0001`** (`wp3_diff_records.py`, exit 0, verdict
PASS). Line endings normalised, then bytes compared.

| Canonical-profile file | Records | Result | SHA-256 (run = baseline) |
| --- | --- | --- | --- |
| `std.advbench.jsonl` | 168 | byte-identical | `08d3ce7fa35bc2bb…` |
| `std.harmbench.jsonl` | 387 | byte-identical | `14d6e3c528bea604…` |
| `std.jailbreakbench.jsonl` | 134 | byte-identical | `819f6662fd5a539c…` |
| `std.orbench_hard.jsonl` | 250 | byte-identical | `e71d970804f6ac62…` |
| `std.sorrybench.jsonl` | 387 | byte-identical | `5c51b412013ab054…` |
| `std.strongreject.jsonl` | 242 | byte-identical | `27e51f354ec71f46…` |
| `std.xstest.jsonl` | 448 | byte-identical | `93d9242fa39e6e39…` |
| `risk.long_input.jsonl` | 1,820 | byte-identical | `01f3f843ee689ac5…` |
| `risk.framing.jsonl` | 4,052 | byte-identical | `610b1991b3c2adc8…` |
| `pilot.script.jsonl` | 144 | byte-identical | `0d411951927ab3a4…` |
| **Ten files** | **8,032** | **all byte-identical** | |

These records hold `flagged`, `zone`, `type`, `conf`, `layers_fired`, every layer score,
`degraded` and `status` for each prompt. The `zone` in them now comes from `ScanResult.zone`
instead of the harness's private rule, and the bytes did not change: the public zone equals
the old rule on every record.

**The eight approved counts** (flagged / n), run and baseline:

| advbench | harmbench | jailbreakbench | orbench_hard | sorrybench | strongreject | xstest_safe | xstest_unsafe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 163 / 168 | 326 / 387 | 130 / 134 | 226 / 250 | 316 / 387 | 218 / 242 | 132 / 250 | 177 / 198 |

Identical in both.

**`lite.std.jsonl`, 1,848 records, compared field by field:**

| Check | Result |
| --- | --- |
| Record count | 1,848 in both |
| Fields that differ anywhere | `degraded` only |
| Records where `degraded` differs | 1,848 of 1,848 |
| The change | `[]` → `["pair_classifier"]` in every one; no other transition |
| Differences in any other field (`flagged`, `zone`, `type`, `conf`, `layers_fired`, `layer_scores`, `status`, identifiers) | **0** |
| Flagged counts per dataset | harmbench 17, jailbreakbench 14, orbench_hard 6, sorrybench 12, strongreject 15, xstest_safe 4, xstest_unsafe 12 — equal to `BL-0001` |

**False full coverage.**

| | Lite scans whose result reports `degraded == []` |
| --- | --- |
| `BL-0001` (before) | 1,848 of 1,848 |
| This run (after) | **0 of 1,848** |

The harness's own metric agrees without any change to the harness's metric code:
`risk.lite.scans_reporting_full_coverage = 0`, share `0.0`.

**Invariant over the run.** All 9,880 records: `flagged == (zone in {uncertain_block,
clear_block})` in every one, 0 violations. Zones: `allow` 5,028, `clear_block` 3,917,
`uncertain_block` 935.

**Determinism.** `python -m evals verify-determinism --all` on the final tree: **exit 0**. Two separate orchestrated runs (`20261010T071639Z_70de7053_ff86f5e8`, `20261010T072724Z_70de7053_ff86f5e8`): four keys equal, **13 gated files compared, 0 differing**, 0 evidence or report files differing. A third run with a random hash seed: 0 record differences. The proof's artifact digest is `a1528d41ef5e87f3…`, the same as the baseline run above, so that run is a third identical one. The proof file is `evals/runs/20261010T071639Z_70de7053_ff86f5e8/DETERMINISM.json`; it will have to be regenerated after the commit, because the pin command wants a canonical run.

**`BL-0002` was not created, and `CANONICAL` was not changed.** `evals pin` refuses a run that
is not canonical, and a run is canonical only when the measured paths have no uncommitted
change (decision D-013, built in WP-001). The assistant may not commit. The guard was not
bypassed and no baseline directory was assembled by hand: a pinned baseline that the tool
itself would have refused is worth less than no baseline. `BL-0001` is untouched and remains
canonical.

The proof the plan required before pinning is complete and is above. After the owner commits,
three commands finish step 8:

```text
python -m evals baseline
        exit 6 is expected here: records differ in ['lite.std.jsonl'] only. Note the run directory it prints.
python -m evals verify-determinism --all
        exit 0; prints the path of DETERMINISM.json
python -m evals pin <run directory from the first command> --canonical --determinism-proof <that DETERMINISM.json>
        creates evals/baselines/BL-0002_pair-v6.3b_fie-70de7053 and points CANONICAL at it; BL-0001 is kept
python -m evals baseline
        exit 0 against BL-0002
```

If the committed `fie/` tree is byte-for-byte the working tree of this report, the baseline
name will carry `fie-70de7053` and its record files will have the hashes in the table above
for the ten canonical files. If either differs, something changed between this report and the
commit, and the pin should not proceed.

---

## K. Deviations, limitations, unresolved items

**Deviations from the plan.** None changes a public contract or a verdict. None was judged
significant enough to stop for, except DV-5, which is a blocker reported rather than worked
around.

| # | Plan said | Done | Why |
| --- | --- | --- | --- |
| DV-1 | `_layer_pair` raises `LayerUnavailable`; `_run_layer_safe` turns it into "no signal" | `_layer_pair` is unchanged. The scanner runs it through `_pair_task`, which returns the same tuple tagged with the state | `scripts/label_review.py` and `evaluation/phase2/layer_isolator.py` call `_layer_pair` directly and unpack three values; raising would break them when the classifier is missing. Tagging also cannot drop a value, where converting an exception could |
| DV-2 | The layer runner returns a state for disabled layers | A layer with no result is reported `disabled` by the coverage builder | Adding a result would add `layer: 0.0` to `layer_scores` for disabled layers, changing an existing field. It also covers the ablation script, which removes layers by replacing the runner |
| DV-3 | Owner clarification: add `bypassed` as a narrowly scoped per-layer state | `bypassed` is also the value of `optional["meta_classifier"]` in a bypassed result | Its other values (`ok`, `unavailable`, `disabled`) would each be untrue there. No new value was introduced. **Alternative if the owner prefers:** report the meta-classifier's process state there instead; one line |
| DV-4 | Seven layer states | `timeout` also covers a scan that arrives while another thread is still inside the first model load | That scan gets no classifier verdict (existing behaviour, NF-10). No approved state says "still loading"; `timeout` ("not ready when the scan needed it") is the closest true one. No state was added |
| DV-5 | Step 8: pin `BL-0002`, update `CANONICAL` | **Not done** | `evals pin` refuses a run made from an uncommitted tree (D-013), and the assistant may not commit. Not bypassed. Section J |
| DV-6 | About 250 production lines, 400 test lines | 888 production lines added, 32 removed; 1,364 test lines; 72 added and 8 removed in the harness adapter | The estimate was too low. Roughly a third of the production lines are docstrings and comments stating the contract; the validated value objects, the three reason codes and the per-call meta-classifier flag account for most of the rest |
| DV-7 | Phases 1, 2, 3, 4 as separate steps | Applied together, verified by test group | The scanner imports the loader's new accessors |
| DV-8 | `to_dict()`: fixed key order | Also canonical: `layers_fired`, `degraded_layers` and the keys of `evidence` and `layer_scores` are sorted in the output | On the result itself their order follows which layer finished first and varies between runs (NF-12). Without sorting, "deterministic serialization" (AC-10) would not hold. The result's own fields are not reordered |
| DV-9 | — | Two public functions in `fie.adversarial`, not exported from `fie`: `model_identities()` and `classifier_state()` | The harness adapter needs the first, the `/health/deep` fix the second. `health()` itself is unchanged, so no server response gained a key |
| DV-10 | W3-12: replace the unsafe detail with a reason code | Replaced for every caller, including a platform admin | The instruction did not distinguish callers. The text is still in the server log |
| DV-11 | Contract tests written first | They were, and then refined (section B) | Stated for accuracy |

**Limitations of the new fields**

- `optional["translation"]` cannot see one case: a Tier 2.5 translation that fails for a prompt
  whose language confidence is below 0.90 leaves no trace, and is reported `not_needed`. Closing
  it needs a change in `fie/multilingual.py`, which is outside this package (NF-11).
- An `ImportError` always maps to `unavailable_dependency`, including the case where ONNX
  Runtime is installed, the ONNX model is absent and the torch fallback is not installed. The
  coverage is correctly `partial`; the reason names the missing package rather than the model.
- `models` describes what is loaded in the process when the result is built; `coverage`
  describes the scan. A caller-disabled classifier shows `loaded=True` and
  `classifier="disabled"`.
- A cached result carries the coverage of the scan that produced it (plan R6).
- `ScanResult` is still a mutable dataclass. The invariant is enforced at construction, not on
  later assignment. `ScanCoverage` and `ModelIdentity` are read-only.
- `degraded_layers` is empty for a bypassed result and for caller-disabled layers. An empty
  list alone does not prove a full scan; `coverage.status == "full"` does. The field's comment
  now says so.

**New findings, not fixed (outside this package's files or scope)**

| # | Finding | Status |
| --- | --- | --- |
| NF-10 | The loader marks the load as attempted when it *starts*. A second scan arriving during the first load skips the classifier and gets a verdict without it. Pre-existing | Now reported as `timeout`, no longer as `ok`. Not fixed: making that scan wait would change its verdict. `warmup()` at start-up avoids it |
| NF-11 | The translation gap above | Needs `fie/multilingual.py` |
| NF-12 | `layers_fired` and the top-level order of `evidence` depend on thread completion order | Pre-existing. `to_dict()` sorts; the fields themselves are unchanged |
| NF-13 | CI sets `GROQ_API_KEY` to a placeholder, so pre-existing SDK tests that land in the uncertain band attempt a real HTTPS call to the tiebreaker provider | Pre-existing, in tests not marked `network`. The new contract tests stub the tiebreaker and are guarded |

**Unresolved:** pinning `BL-0002` (owner, section J); DV-3 (owner preference).

---

## L. Security and compatibility review

**New public fields**

| Field | Content | Control |
| --- | --- | --- |
| `coverage.status`, `.layers`, `.classifier`, `.optional` | Fixed codes | Closed sets, validated on construction; unknown layer states are mapped to `error` before construction |
| `zone`, `decided_by` | Fixed codes | Closed sets, validated on construction |
| `models[*].version` | Published metadata | Must match `^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$`, else dropped |
| `models[*].digest` | Published hash, abbreviated | Must match `^[0-9a-f]{16}$` |
| `models[*].threshold`, `.backend` | Number in [0, 1]; one of two names | Validated |
| `GuardResult.zone`, `.coverage_status` | The same codes, or `None` | Only strings are copied across |

Not placed in any result: paths, file names, the loader's error text, exception class names or
messages, environment values, host names, tenant or session identifiers.

**Leak tests.** An exception message containing a Windows path, a POSIX path, a token-like
string and an e-mail address is raised from a layer, from the classifier's encoder, from the
meta-classifier, from a lite layer and from the scanner itself. The real loader is also driven
into its three failure branches, where its own error text contains a real temporary directory
(and, for a missing package, the package name). In every case `to_dict()`, `repr(coverage)`,
`repr(models)`, the CLI text output and the warning message are searched for those strings.
None is found.

**`/health/deep` (W3-12).** Before: with the classifier missing, an anonymous `GET /health/deep`
returned `components.detector.detail.pair_classifier.error` containing the model directory's
absolute path. The regression test failed on the unmodified code with that text in the
response. After: the same field holds `unavailable_dependency`, `unavailable_model` or
`unavailable_load_failed`. The response's key set is unchanged, and the response with the
classifier loaded is unchanged; both are asserted. The change is five lines in one block of
`app/main.py`. Nothing else in `app/` changed, no server response schema grew, and nothing new
is persisted (W3-11).

**WP-002 is intact.** No authentication, policy, scoping or tenant code was touched; the 318
security tests pass unchanged.

**Compatibility (W3-18)**

| Item | Effect |
| --- | --- |
| The nine other existing `ScanResult` fields | Unchanged in type, meaning and value, in every configuration tested |
| `degraded_layers` | Type unchanged. Value unchanged with the classifier loaded. Gains `"pair_classifier"` when it is unavailable. **The one intended change** (W3-2) |
| `is_degraded` | Follows it: `True` on a base install |
| Constructing `ScanResult`, `LiteScanResult`, `GuardResult` the old way | Works; new fields take neutral defaults and claim nothing |
| `copy`, `deepcopy`, `pickle`, `dataclasses.asdict`, `dataclasses.replace` on a result | Work; tested |
| `_safe_scan()` | Still returns five values that unpack as before |
| `_layer_pair`, `_run_pair_classifier`, `_layer_multilingual`, `_pair_state()`, `_meta_state()`, `health()` | Same return values and key sets; tested |
| CLI JSON | Every earlier key is still first, in the same order, with the same value. Six keys follow: `zone`, `decided_by`, `coverage`, `models`, `degraded_layers`, `schema_version` (W3-8) |
| CLI text | One `Coverage` line, only when coverage is not full |
| `scan_prompt_async` | Returns the same object; `to_dict()` equal to the synchronous one |

A consumer that treats `is_degraded` as an error will now see it on a base install. That is the
purpose of the change.

---

## M. Performance

| Item | Measured |
| --- | --- |
| Added work per scan | 23 µs: 14 µs to build coverage, 9 µs to build the three identities. About 0.1% of a scan |
| `to_dict()` | 10 µs, only when called |
| Warm scan latency, 600 scans of three prompts | Before: 31 ms and 59 ms mean in two runs. After: 25, 37 and 37 ms mean in three runs. Run-to-run noise on this machine is larger than any effect; no change is detectable |
| Harness latency suite | Too noisy today to resolve a microsecond-scale change. `std.xstest` mean per scan: 21.7 ms in `BL-0001`, 30.8 ms in today's pre-change run, 37.0 ms post-change; `std.harmbench`: 41.0, 60.7, 46.9 ms. The pre-change run was the slower of today's two on harmbench and the faster on xstest. The lite suite, which loads no model: 5.93 ms (`BL-0001`) and 6.07 ms (post-change). Latency is run metadata and is not part of any byte comparison |
| Cold start | +0.26 s once per process (section G) |
| Result cache | Unchanged |

---

## N. Scope verification

`git status --short` at the end:

```text
 M app/main.py
 M docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md
 M evals/subject.py
 M fie/__init__.py
 M fie/__main__.py
 M fie/_lite.py
 M fie/adversarial.py
 M fie/layers/pair.py
 M fie/onnx_encoder.py
 M fie/preflight.py
 M tests/evals/test_subject_contract.py
?? docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_003_truthful-scan-result.md
?? tests/test_scan_result_contract.py
```

Eleven tracked files modified, two files created (the contract tests and this report), none
deleted. Every one is on the approved list. `engine/`, `storage/`, `deploy/`, `Frontend/`,
model files, dependency files and `README.md` show no change. `evals/runs/` is ignored by git
and holds the run directories cited here.

No threshold, weight, aggregation rule, dependency, network call, README text, deployment
configuration, model file or packaging file was changed.

---

## O. Acceptance criteria

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| AC-1 | Public `zone` (four values) and `decided_by` | **PASS** | Every row of section F produced by a test; all values reached |
| AC-2 | Truthful coverage | **PASS** | A test for each of the eight layer states |
| AC-3 | A missing classifier cannot report full coverage | **PASS** | Three reasons through the real loader; fault M1 caught by 7 tests |
| AC-4 | Lite false-full-coverage is 0 of 1,848 | **PASS** | Harness run: `degraded == []` in 0 of 1,848 lite records (1,848 in `BL-0001`); the harness metric reads 0 |
| AC-5 | Model identity accurate and safe | **PASS** | Digests equal an independent hash and the manifest; forced version; no path |
| AC-6 | Layer failure represented | **PASS** | `error`, result valid |
| AC-7 | Timeout represented | **PASS** | `timeout`, scan returned at the deadline |
| AC-8 | No exception text, secret or path in a public field | **PASS** | Planted-string searches, section L |
| AC-9 | Sync and async agree | **PASS** | `to_dict()` equal, bytes equal |
| AC-10 | Serialization deterministic | **PASS** | Fixed key order; canonical bytes; round-trip (DV-8) |
| AC-11 | Existing fields compatible | **PASS** | Golden file unmodified and passing; old-constructor tests |
| AC-12 | Canonical guard baseline unchanged | **PASS** | Ten files byte-identical; eight counts equal; `lite.std` differs in `degraded` only, in every record. `BL-0001` untouched |
| AC-13 | Existing tests green | **PASS** | 87 + 230 + 318 all pass by name; 785 in total |
| AC-14 | No network dependency | **PASS** | Static import test; socket guard on every contract test; harness guard: 0 violations in the orchestrator and both workers |
| AC-15 | No detector-quality work | **PASS** | Removed-line review; constants pinned; verdict tests |
| AC-16 | Verdicts identical with the classifier missing | **PASS** | 24 literal tests; lite flagged counts equal `BL-0001`; per-record equality of every field except `degraded` over 1,848 records |
| AC-17 | `is_attack` and `zone` cannot disagree | **PASS** | Enforced at construction; 0 violations over the run's 9,880 records |
| AC-18 | The harness no longer needs a private rule for the zone | **PASS** | Adapter records `ScanResult.zone`; the rule is a cross-check that stops the run; 8 tests |
| AC-19 | Unavailable-classifier warning once per process | **PASS** | Log-capture test, three reasons |
| AC-20 | Only approved files changed | **PASS** | Section N |
| — | `BL-0002` pinned and `CANONICAL` updated (plan step 8, W3-15) | **NOT VERIFIED — owner-gated** | Section J |
| — | `python -m evals baseline` exits 0 against `BL-0002` | **NOT VERIFIED — owner-gated** | Follows the pin |

---

## P. Rollback

Revert the eleven modified files and delete `tests/test_scan_result_contract.py`. Nothing is
persisted by this package: no stored data, no cache format, no database field. `CANONICAL`
still points at `BL-0001`, so there is nothing to restore there.

---

## Q. Final state

```text
git log --oneline -1   ->  550226d docs/fie_rebuild_2026/IMPLEMENTATION_PLANS/PLAN_003_TRUTHFUL_SCAN_RESULT.md
git diff --cached      ->  (empty: nothing staged)
evals/baselines        ->  BL-0001_pair-v6.3b_fie-5d5ed90d, REF-0001_pair-v6.2_fie-5d5ed90d, CANONICAL -> BL-0001…
```

Nothing was staged, committed, pushed, stashed, reset or tagged. HEAD is `550226d`, as at the
start.

## Manual commit line

Not executed by the assistant.

```text
git add -- fie/adversarial.py fie/layers/pair.py fie/onnx_encoder.py fie/_lite.py fie/preflight.py fie/__main__.py fie/__init__.py evals/subject.py app/main.py tests/test_scan_result_contract.py tests/evals/test_subject_contract.py docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_003_truthful-scan-result.md docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md && git commit -m "fie: report scan coverage, zone and model identity truthfully (WP-003)"
```
