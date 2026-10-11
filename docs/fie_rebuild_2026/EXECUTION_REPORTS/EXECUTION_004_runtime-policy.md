# EXECUTION_004 — Runtime Policy and Coverage Handling (WP-004)

| | |
| --- | --- |
| Work package | WP-004 (the roadmap numbers this topic WP-007; see section K) |
| Status | **COMPLETE — every required gate passed. Awaiting owner review** |
| Commit status | **NOT COMMITTED. The owner reviews and commits manually.** Nothing was staged, committed, pushed, merged, deployed, stashed or reset; no branch was changed |
| Date | 2026-10-10 to 2026-10-11 |
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD (unchanged throughout) | `924c70d2b707b9a60ab4c7053bd50dbcf2ce02f9` — "evals: pin BL-0002 as canonical baseline after WP-003" |
| Contract | [PLAN_004_RUNTIME_POLICY.md](../IMPLEMENTATION_PLANS/PLAN_004_RUNTIME_POLICY.md), revision 3, with owner decisions W4-1 to W4-16 and Q1 to Q7, and one amendment (section J, DV-2) |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged. No `BL-0003` |
| Deployed? | **No** |

**What this package is not.** It is not enforcement across every FIE entry point. Policy is
applied in one place, `preflight_check`, and displayed by the CLI. Code that calls
`scan_prompt()` directly gets no policy; that includes the OpenAI, Anthropic and FastAPI
integrations. Section H lists the gaps. Nothing here makes the detector more accurate, and a
policy cannot make a base install detect more.

---

## A. Summary

FIE now has an explicit, versioned answer to "what do we do with this scan result", separate
from the detector that produced the result.

- `fie/policy.py` (new): `decide(result, policy, *, scan_failed=False)` is a pure function
  returning a `PolicyDecision` with an **action** (`allow`, `review`, `block`, `error`),
  fixed **reason codes**, and the **id and digest** of the policy that decided.
- Three presets. `compat@1` is the default and reproduces previous behaviour exactly.
  `strict@1` and `review@1` take effect only when an operator selects them.
- `preflight_check` resolves the policy once, before scanning, applies it, and reports
  `action`, `policy` and `reasons` on `GuardResult`.
- The CLI appends a `policy` block to its JSON and one text line when the policy adds
  something. Its exit code is still the detector's verdict.
- `scan_prompt`, `ScanResult` and every detector are untouched.

| Gate | Result |
| --- | --- |
| Tests | **942 passed, 0 failed, 0 skipped, 0 errors** (785 existing + 157 new) |
| Existing tests | All 785 pass by name. Exactly two assertions edited, both authorized |
| `python -m evals baseline` | **Exit 0.** Eight counts equal; **all 11 record files byte-identical to `BL-0002`** |
| `python -m evals verify-determinism --all` | **Exit 0.** 13 gated files compared, 0 differing; 0 differences under a random hash seed |
| Mutation checks | **18 of 18 caught**; every file restored and SHA-256 verified |
| `decide` cost | 4.98 µs median (budget 5 µs) |
| Blocked gates | None |

---

## B. Starting state

| Item | Value |
| --- | --- |
| Branch, HEAD | `rebuild/wp-001-eval-harness`, `924c70d` |
| Staged | Nothing |
| Working tree | One untracked file: the approved plan, `PLAN_004_RUNTIME_POLICY.md`. **Authorized precondition exception**: the owner instructed that implementation proceed with the plan untracked, and that it not be committed or its header changed. Its header therefore still reads "implementation NOT yet authorized"; the authorization is in the owner's implementation instruction |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053`; `BL-0001` and `REF-0001` present |
| Tests before any change | `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` → exit 0, **785 passed**, 0 failed, 0 skipped |
| Baseline before any change | `python -m evals baseline` → **exit 0**, run `20261010T180128Z_70de7053_090da7e3`, `canonical=True`, 11 record files byte-identical to `BL-0002` |

**Reference values captured before any change.** `GuardResult.to_dict()` from the unmodified
`preflight_check` for: eight reference prompts with the classifier loaded and missing, each in
block mode and warn-only mode; an empty prompt; and the scanner raising with the fail-secure
flag off and on, each in block mode and warn-only mode. They are literals in the new test
file.

**Tests first.** `tests/test_runtime_policy.py` was written before `fie/policy.py` existed and
run against the unmodified code: **46 passed, 111 failed**.

| On the unmodified code | Tests |
| --- | --- |
| Passed: today's behaviour, which must not move | 36 `compat` comparisons against the captured values (loaded ×16, missing ×16, scanner raising ×4); `ScanResult` equality under every configuration ×8; detector files untouched; CLI text unchanged |
| Failed: the feature does not exist yet | `No module named 'fie.policy'` (76 first-failures); `GuardResult` has no `action` (22); CLI JSON has no `policy` key and no stderr notice (the rest) |
| Pre-existing failures | **None** |

Two corrections were made to the test file after that run and before the implementation was
judged: an escaping typo of mine (doubled backslashes; the first attempt did not collect at
all), and the four preset digests, which cannot be known before the module exists and were
pinned from it once and then frozen. No assertion was weakened or removed.

---

## C. Files

| File | Status | + / − | Purpose |
| --- | --- | --- | --- |
| [fie/policy.py](../../../fie/policy.py) | **new** | 482 | `Policy`, `PolicyDecision`, `ACTIONS`, the 15 `REASONS`, `compat_policy`, `STRICT`, `REVIEW`, pure `decide`, `invalid_config_decision`; separately `active_policy`, `set_policy`, `reset_policy`, `PolicyConfigError` |
| [fie/preflight.py](../../../fie/preflight.py) | modified | 151 / 39 | `GuardResult` gains `action`, `policy`, `reasons` (appended, defaulted); `preflight_check` resolves the policy once before scanning, calls `decide`, applies the warn-only rule, writes the audit line; keyword-only `policy=`; `_safe_scan`'s outcome also carries the scan result |
| [fie/__main__.py](../../../fie/__main__.py) | modified | 57 / 0 | Policy resolved before the scan; `policy` appended to both JSON outputs; one `Policy` text line; fixed stderr notice under an invalid configuration. Exit codes untouched |
| [fie/__init__.py](../../../fie/__init__.py) | modified | 4 / 0 | Exports `Policy`, `PolicyDecision`, `decide` |
| [tests/test_runtime_policy.py](../../../tests/test_runtime_policy.py) | **new** | 1,351 | 157 tests |
| [tests/test_scan_result_contract.py](../../../tests/test_scan_result_contract.py) | modified | 3 / 2 | **The two authorized assertions only** (section I) |
| This report | new | — | — |
| `MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | modified | — | Session rows and §12 |

**Not touched:** `fie/adversarial.py`, `fie/layers/`, `fie/_lite.py`, `fie/onnx_encoder.py`,
`fie/multilingual.py`, `fie/monitor.py`, `fie/integrations/`, `app/`, `engine/`, `storage/`,
`evals/` (code, baselines, `CANONICAL`), `tests/security/`, `tests/evals/`, every other
existing test, the golden file, Dockerfiles, dependency files, `README.md`, the plan. A test
asserts by `git status` that the detector, harness and server paths have no change.

---

## D. What was implemented

### D.1 The pure decision

`decide(result, policy, *, scan_failed=False)`:

- `policy` is required and must be a `Policy` instance; anything else, including `None`,
  raises `TypeError`. `decide` never looks a policy up.
- It reads its three arguments and nothing else. A test replaces `os.environ` with an object
  that fails on any read, points `FIE_POLICY`, `FIE_SCAN_FAILURE_MODE`, `FIE_UNCERTAIN_ALLOW`,
  `PREFLIGHT_BLOCK_ENABLED`, the fail-secure flag and `set_policy` at conflicting values, and
  gets identical decisions.
- It does not mutate or annotate the result, and no `PolicyDecision` is placed in the scan
  cache or on a `ScanResult`.
- The decision's `policy` and `policy_digest` are those of the object passed in.

Decision rows, first match decides and sets `reasons[0]`:

| # | Condition | Action | `reasons[0]` |
| --- | --- | --- | --- |
| 1 | no result / `scan_failed` | `on_scan_failure` | `scan_failed` |
| 2 | a stored label decided (`coverage.status == "bypassed"`) | by the label | `labelled_attack` / `labelled_safe` |
| 3 | zone `clear_block` | `block` | `detector_block` |
| 4 | zone `uncertain_block`, tiebreaker confirmed | `block` | `tiebreaker_confirmed` |
| 5 | zone `uncertain_block`, otherwise | `on_uncertain` | `detector_uncertain` |
| 6 | allow verdict, a required layer not `ok` | `on_missing_required` | `missing_required_layer` |
| 7 | allow verdict, coverage not reported | `on_missing_required` | `coverage_unknown` |
| 8 | zone `uncertain_allow` | `allow` | `tiebreaker_cleared` / `uncertain_allowed_by_config` |
| 9 | zone `allow` | `allow` | `detector_allow` |

Row 0, "no valid policy", is not in `decide`; it is handled before it (D.3).

### D.2 Presets

| | `compat@1` (default) | `strict@1` | `review@1` |
| --- | --- | --- | --- |
| required layers | `pair_classifier` | `pair_classifier` | `pair_classifier` |
| uncertain block | block | block | review |
| allow with the classifier missing | allow, reason `missing_required_layer` | block | review |
| scanner failure | allow (open form) / block (closed form) | block | error |
| `review` → `blocked` | yes | yes | yes |
| digest | `989f7491` open, `f5fea0f8` closed | `4cb035e4` | `31e8f0cd` |

A test pins every field and digest, so a preset's behaviour cannot change without a visible
edit.

### D.3 Process configuration and the effective policy

Implemented exactly as plan §5.5:

```text
policy for one preflight_check call =
    the Policy passed as policy=            (keyword-only; trusted code)
    otherwise active_policy(), called once, before the scan

active_policy() =
    the set_policy() object, as is
    otherwise the cached FIE_POLICY selection:
        strict -> STRICT        review -> REVIEW
        compat or unset -> compat@1, closed form if fie.preflight._FAIL_SECURE else open form
        anything else  -> PolicyConfigError
```

- `FIE_POLICY` is read once, on the first `active_policy()` call; the selection is cached,
  including a failure. No hot reload. `set_policy` and `reset_policy` are the only in-process
  ways to change it.
- **`fie.policy` never reads `FIE_SCAN_FAILURE_MODE`.** That variable still has exactly one
  reader, the existing module-level assignment in `fie/preflight.py`, which this package did
  not edit. `active_policy()` reads the resulting flag at each call. A test asserts the
  variable's name does not appear anywhere in `fie/policy.py`.
- `preflight_check` uses the object it resolved for `decide` and for the reported id. Tests
  confirm one resolution and one decision per call, resolution before the scan, and that a
  flag flip or `set_policy` made *during* a scan does not affect that call.

### D.4 Invalid configuration

| | `preflight_check` | CLI |
| --- | --- | --- |
| Scan | **not run** (spy: 0 calls) | **run exactly once** (spy: 1 call) |
| Outcome | `blocked=True`, `action="error"`, `reasons=("invalid_policy_config",)`, `policy="invalid"`; `scan_failed=False`, `attack_type=""`, `confidence=0.0`, `layers_fired=[]`, `zone=None`, `coverage_status=None` | every existing key and value from the real scan; `policy` = `invalid_config_decision(result)`: `action "error"`, `blocked true`, `reasons ["invalid_policy_config"]`, `policy "invalid"`, `policy_digest null`, with the real `detector_is_attack`, `zone`, `coverage_status` |
| Warn-only | cannot waive it | n/a |
| Exit code | n/a | `1 if is_attack else 0`, unchanged |
| Signal | one ERROR on logger `fie.policy`, once per process; the audit line per call | the fixed notice on stderr, in JSON, text and `--quiet` modes |

The offending value appears nowhere: not in `GuardResult.to_dict()`, the exception message,
either log line, stdout or stderr. A test sets `FIE_POLICY` to a string containing a path, a
token and an e-mail address and searches all of them. An empty prompt still returns unscanned
and unblocked with `action=None`; a per-call `policy=` still works; the failure persists
after the variable is corrected and clears after `reset_policy()` or `set_policy(valid)`.

### D.5 Warn-only

Applied after the decision; never changes `action` or `reasons[0]`:

```text
waived  iff  decision.blocked
             and decision.action in ("block", "review")
             and reasons[0] in {detector_block, detector_uncertain, tiebreaker_confirmed, labelled_attack}
             and warn-only is on
```

When waived, `blocked=False` and `warn_only` is appended. A block for missing coverage, a
scanner failure or an invalid configuration is never waived. The warn-only switch is not even
consulted unless there is a detector block to waive (tested), which also preserves when the
server's hot-config lookup happens.

### D.6 Audit line

Logger `fie.policy`, one line per decision that is not a plain detector allow:
`POLICY_DECISION | action=… blocked=… policy=… digest=… reasons=… zone=… coverage=…
missing_required=…`. WARNING when blocked, INFO otherwise. Fixed codes, layer names and the
policy id only. The existing `PREFLIGHT_BLOCK` / `PREFLIGHT_WARN` lines keep their format and
conditions.

---

## E. Acceptance criteria

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| AC-1 | `decide` is pure | **PASS** | explicit-policy, environment-trap and no-mutation tests; import test; mutations M5, M15 |
| AC-2 | Closed sets: 4 actions, 15 reasons | **PASS** | names test; invariants over every input |
| AC-3 | Every row of §6.1 and §6.3 | **PASS** | 864-cell cross product against an independently written table; 18 named rows × 3 presets; scanner-failure row ×6 |
| AC-4 | A detector block is never downgraded by missing coverage | **PASS** | invariant 1 over all inputs; real-loader test; mutation M1 (16 tests failed) |
| AC-5 | Missing required coverage never yields a bare `allow` | **PASS** | invariant 2 over all inputs; mutation M2 (23) |
| AC-6 | C1: `compat@1` forwards and stops what the code did before | **PASS** | 36 comparisons with captured values, including warn-only and both scanner-failure forms |
| AC-7 | C2: the nine existing `GuardResult` values | **PASS** | same 36, field by field |
| AC-8 | C3: `to_dict()` = nine existing keys, then `action`, `policy`, `reasons`; old constructors work | **PASS** | additions test; the authorized WP-003 assertion |
| AC-9 | C4: CLI JSON = existing keys and values, then `policy`; exit code = detector verdict | **PASS** | both commands × 3 presets × 3 prompts; strict + missing classifier exits 0 with `policy.action = "block"`; mutation M12 |
| AC-10 | C5: `ScanResult` and detector unchanged | **PASS** | results equal under unset, `compat`, `strict`, `review` and invalid configuration, classifier loaded and missing; no diff in detector files |
| AC-11 | Detector verdicts unchanged | **PASS** | 11 record files byte-identical to `BL-0002`; 8 counts equal; baseline exit 0 |
| AC-12 | Determinism | **PASS** | `verify-determinism --all` exit 0 |
| AC-13 | Invalid `FIE_POLICY`: blocking `error`, no scan, no preset | **PASS** | section D.4; mutations M3, M4, M10 |
| AC-13b | CLI under invalid configuration still scans once | **PASS** | spy; 2 commands × 2 prompts × JSON/text/`--quiet`; mutations M11, M17 |
| AC-14 | Configuration failure ≠ scanner failure | **PASS** | side-by-side test on `reasons[0]`, `scan_failed`, `policy` |
| AC-15 | Lifecycle as documented | **PASS** | read-once, precedence, reset, 16-thread first resolution |
| AC-15b | One effective-policy rule | **PASS** | flag-following, environment-change and explicit-policy tests; mutations M6, M7, M8 |
| AC-15c | Identity equals effect; no substitution | **PASS** | six configurations replayed by hand; mid-scan reconfiguration test; mutation M9 |
| AC-16 | Warn-only waives detector blocks only | **PASS** | section D.5 tests; mutations M13, M14 |
| AC-17 | No request field can select policy | **PASS** | signature test; static scan of `app/` and `fie/integrations/` |
| AC-18 | Every decision names its policy | **PASS** | `PolicyDecision`, `GuardResult`, audit line, CLI JSON; pinned digests |
| AC-19 | Outcomes separate from verdicts | **PASS** | `ScanResult` has no `action`/`policy`/`reasons`; cache holds no decision; a policy block of a detector allow carries no attack type |
| AC-20 | No value, prompt text, exception text, path or secret in outputs | **PASS** | planted-string searches over results, log lines, stdout, stderr |
| AC-21 | Action counts on `BL-0002` equal plan §6.4 | **PASS** | replay test, section G |
| AC-22 | Existing tests pass by name | **PASS, with the amendment** | 785 of 785; **two** assertions edited, not one (DV-2) |
| AC-23 | Integrations, server, deployment unchanged; limits stated | **PASS** | `git status`; section H; gap-documenting test |
| AC-24 | `decide` ≤ 5 µs median | **PASS, narrowly** | 4.98 µs (allow), 4.93 µs (block), 2.40 µs (scanner failure) |
| AC-25 | Only approved files changed | **PASS** | section L |

---

## F. Verification

Environment: Python 3.10.19, conda env `failure-engine`, the CI-equivalent environment with
every `.env` key blanked. Commands run from the repository root.

| # | Command | Exit | Result |
| --- | --- | --- | --- |
| 1 | `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` (before) | 0 | 785 passed, 0 failed, 0 skipped |
| 2 | `python -m evals baseline` (before) | 0 | 11 files byte-identical to `BL-0002`; `canonical=True` |
| 3 | `python -m pytest tests/test_runtime_policy.py …` on the unmodified code | 1 | 46 passed, 111 failed (expected, section B) |
| 4 | Mutation runner, final code | 0 | 18 of 18 caught |
| 5 | `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` (after) | **0** | **942 passed, 0 failed, 0 skipped, 0 errors**; 157 in the new file |
| 6 | `python -m evals baseline` (after) | **0** | below |
| 7 | `python -m evals verify-determinism --all` (after) | **0** | below |

**Test names.** The 785 names that passed before all pass after; the set difference is empty.

**Baseline (gate 6).** Run `20261010T183231Z_5473664b_f6e64584`. Output: "standard counts match
the approved canonical counts" and "per-prompt records are byte-identical to
BL-0002_pair-v6.3b_fie-70de7053 (11 file(s))". Independently re-compared, file by file:

| Record file | Records | vs `BL-0002` | SHA-256 |
| --- | --- | --- | --- |
| `lite.std.jsonl` | 1,848 | identical | `8ad9f4242964b4d4…` |
| `pilot.script.jsonl` | 144 | identical | `0d411951927ab3a4…` |
| `risk.framing.jsonl` | 4,052 | identical | `610b1991b3c2adc8…` |
| `risk.long_input.jsonl` | 1,820 | identical | `01f3f843ee689ac5…` |
| `std.advbench.jsonl` | 168 | identical | `08d3ce7fa35bc2bb…` |
| `std.harmbench.jsonl` | 387 | identical | `14d6e3c528bea604…` |
| `std.jailbreakbench.jsonl` | 134 | identical | `819f6662fd5a539c…` |
| `std.orbench_hard.jsonl` | 250 | identical | `e71d970804f6ac62…` |
| `std.sorrybench.jsonl` | 387 | identical | `5c51b412013ab054…` |
| `std.strongreject.jsonl` | 242 | identical | `27e51f354ec71f46…` |
| `std.xstest.jsonl` | 448 | identical | `93d9242fa39e6e39…` |

Network guard: 0 violations in the orchestrator and both workers.

**The harness reports `canonical=False` for this run**, with the reason "uncommitted changes
in measured paths: `fie/__init__.py`, `fie/__main__.py`, `fie/policy.py`, `fie/preflight.py`".
That is the harness's rule for an uncommitted tree and is reported as it appeared; it is not a
substitute for, and does not weaken, the record comparison above, which passed. The run's
`fie` tree hash is `5473664b` (was `70de7053`): a file was added and three edited. Record
bytes did not change. `BL-0002` stays canonical and was not modified.

**Determinism (gate 7).** Runs `20261010T184638Z_5473664b_cf0d6da5` and
`20261010T185732Z_5473664b_cf0d6da5`: four keys equal; 13 gated files compared, 0 differing;
0 evidence or report files differing; a third run with a random hash seed, 0 record
differences. Output: "deterministic: byte-identical results, summary and fingerprint". These
runs are likewise flagged `canonical=False` for the same reason.

---

## G. Mutation checks and benchmarks

For each mutation: SHA-256 of the file recorded, mutation applied, tests run, file restored,
SHA-256 compared. All 18 restored hashes equal the originals; a final `sha256sum -c` over the
four production files passed after the whole chain.

| # | Fault | File | Tests failed |
| --- | --- | --- | --- |
| M1 | Missing coverage downgrades a detector block | `fie/policy.py` | 16 |
| M2 | A missing-coverage allow reports `detector_allow` first | `fie/policy.py` | 23 |
| M3 | Invalid `FIE_POLICY` falls back to `compat` | `fie/policy.py` | 7 |
| M4 | Invalid configuration reported as `scan_failed` | `fie/policy.py` | 7 |
| M5 | `decide` reads `FIE_POLICY` | `fie/policy.py` | 1 |
| M6 | `fie.policy` reads `FIE_SCAN_FAILURE_MODE` | `fie/policy.py` | 4 |
| M7 | `active_policy()` caches the `compat` form | `fie/policy.py` | 3 |
| M8 | `preflight_check` re-resolves the policy after scanning | `fie/preflight.py` | 5 |
| M9 | Decision reports another policy's digest | `fie/policy.py` | 5 |
| M10 | `preflight_check` scans under an invalid configuration | `fie/preflight.py` | 1 |
| M11 | CLI skips the scan under an invalid configuration | `fie/__main__.py` | 4 |
| M12 | CLI exit code follows the policy action | `fie/__main__.py` | 2 |
| M13 | Warn-only waives a missing-coverage block | `fie/preflight.py` | 3 |
| M14 | Warn-only waives a scanner-failure block | `fie/preflight.py` | 3 |
| M15 | `decide` writes onto the scan result | `fie/policy.py` | 1 |
| M16 | A `GuardResult.to_dict()` key reordered | `fie/preflight.py` | 2 |
| M17 | The stderr notice prints the offending value | `fie/__main__.py` | 4 |
| M18 | A preset's behaviour changed without a new version | `fie/policy.py` | 6 |

Original and restored hashes (first 16 hex): `fie/policy.py` `18488f34b3188a7c`,
`fie/preflight.py` `d0f99e082c6eff2c`, `fie/__main__.py` `17e449941b52a659`,
`fie/__init__.py` `f2d4a57174cd7535`. The mutation run was done twice: once before the
performance change described in DV-4 and again on the final code. The table is the second run.

**Benchmarks** (median of 7 × 20,000 calls):

| Operation | Time |
| --- | --- |
| `decide`, allow verdict | 4.98 µs |
| `decide`, block verdict | 4.93 µs |
| `decide`, scanner failure | 2.40 µs |
| `active_policy()` after resolution | 0.68 µs |
| `invalid_config_decision()` | 1.86 µs |
| `preflight_check`, warm, 600 scans | 26.2 ms mean, 26.7 ms median |

About 6 µs of policy work per guarded call against a scan of about 26 ms. No cold-start cost:
nothing is loaded.

**Baseline replay** (every `BL-0002` record through `decide`; a test, no new scan):

| | Canonical profile, 2,016 | Base-install profile, 1,848 |
| --- | --- | --- |
| `compat@1` | allow 328, block 1,688 | allow 1,768, block 80 |
| `strict@1` | allow 328, block 1,688 | block 1,848 |
| `review@1` | allow 328, review 294, block 1,394 | review 1,769, block 79 |

These equal plan §6.4. They are the stated cost of each preset: `strict@1` on a base install
blocks everything, including all 500 benign prompts it allows today.

---

## H. What WP-004 does not enforce

Unchanged by this package, by owner decisions W4-9 and W4-10. Listed so that the package is
not mistaken for universal enforcement.

| Priority | Gap | Consequence under any preset |
| --- | --- | --- |
| F3a | `fie/integrations/openai.py`, `anthropic.py` call `scan_prompt` directly; a scanner exception is logged at DEBUG and the request proceeds | No policy. A scan without the classifier is forwarded even under `strict@1` |
| F3b | `FIEMiddleware`, local mode: scanner exception → WARNING, request proceeds | Same |
| F3c | `FIEMiddleware`, server mode, posts to `/api/v1/scan`, which does not exist | Protects nothing in server mode |
| F4a | `app/routes/monitor.py`, `playground.py` read `GuardResult.blocked` only | A selected preset does take effect there through `preflight_check`, but `action` and `reasons` are dropped, so a refusal cannot be told from an unhealthy guard |
| F4b | The same routes catch a guard exception and allow the request | Unchanged |
| F4c | Shipped images set no `FIE_POLICY` | The hosted service stays on `compat@1` |
| F4d | The server does not refuse to start on an invalid `FIE_POLICY` | It starts and answers every guarded request with a blocking `error` |
| — | `fie detect` / `fie explain` exit code | Detector verdict, never a policy gate (W4-15). Scripts must read `policy.action` |
| — | `fie/monitor.py` decorators | They do apply the configured policy through `preflight_check`; they do not surface `action` or `reasons` |

`test_known_gap_direct_scan_prompt_callers_are_not_covered` asserts F3a as current behaviour:
under `FIE_POLICY=strict` with the classifier missing, `preflight_check` blocks a prompt that
the OpenAI wrapper still forwards. The package that migrates the integrations must change
that test deliberately.

---

## I. Compatibility

| | Statement | Evidence |
| --- | --- | --- |
| C1 | Under `compat@1`, `preflight_check` forwards and stops exactly what it did | 36 captured-value comparisons |
| C2 | The nine existing `GuardResult` fields keep their values, including the warn-only and fail-secure quirks | same |
| C3 | `GuardResult` gains `action`, `policy`, `reasons`, appended and defaulted; `to_dict()` emits the nine existing keys in order, then those three; `schema_version` stays 2 | additions test |
| C4 | CLI JSON emits every earlier key in order with the same value, then `policy`; exit codes unchanged | CLI tests |
| C5 | `ScanResult`, its `to_dict()`, `scan_prompt` and the detector are unchanged | equality tests; 11 baseline files |

`preflight_check`'s three existing parameters and their order, `_safe_scan`'s five return
values, `GuardedResponse` and `make_guarded_response` are unchanged.

**Existing-test edits: exactly two assertions, both in `tests/test_scan_result_contract.py`,
both authorized by the owner.** The complete diff of that file:

```text
-                            "scan_failed", "zone", "coverage_status", "schema_version")
+                            "scan_failed", "zone", "coverage_status", "schema_version",
+                            "action", "policy", "reasons")

-        assert tuple(data)[len(old_keys):] == NEW_KEYS
+        assert tuple(data)[len(old_keys):] == NEW_KEYS + ("policy",)
```

The assertion on the original CLI keys (the line before the second one) is unchanged. No other
existing test file was modified.

---

## J. Deviations, and things the owner should know

| # | Item | Detail |
| --- | --- | --- |
| DV-1 | Precondition exception | The tree was not strictly clean: the approved plan was untracked. Authorized by the owner; the plan was neither committed nor edited |
| DV-2 | **A second existing assertion had to change** | Revision 3 authorized one (the `GuardResult.to_dict()` tuple). The approved CLI `policy` key also breaks a second pinned tuple, which I had missed when planning. Raised as a stop condition before any file was changed; the owner authorized exactly that second edit |
| DV-3 | Execution-report path | An earlier instruction named `docs/fie_rebuild_2026/EXECUTION_004_runtime-policy.md`; the plan and the final instruction say `EXECUTION_REPORTS/`. This file is at the latter |
| DV-4 | `decide` was first over budget | First measurement: 14.9 µs, three times the 5 µs budget, because the policy digest was recomputed on every decision. The digest is now computed once, when the immutable `Policy` is constructed. Final: 4.98 µs. Digests, `copy`, `pickle` and `dataclasses.replace` behave as before (checked). The margin under the budget is thin and will vary by machine |
| DV-5 | Scanner-failure fields under non-`compat` policies | The plan fixed the values for `compat@1`. For any policy, a *blocked* scanner failure reports the existing fail-secure marker (`attack_type="SCAN_FAILED"`, `confidence=1.0`, `layers_fired=["preflight_fail_secure"]`) and an *allowed* one reports the existing open shape, following the decision rather than the flag. Under `compat@1` this is identical to before |
| DV-6 | A replaced `_safe_scan` | If other code replaces `_safe_scan` with one returning only the five legacy values, `preflight_check` builds a neutral result from them with coverage unknown. Under `compat@1` the outcome is unchanged; under `strict@1` an allow from such a replacement is blocked as `coverage_unknown` |
| DV-7 | Audit line level | The plan did not set one: WARNING when blocked, INFO otherwise |
| DV-8 | Public constants beyond the approved names | `ACTIONS`, `REASONS`, `DETECTOR_BLOCK_REASONS`, `INVALID_POLICY_ID`, `DECISION_SCHEMA_VERSION` in `fie.policy`. Values only; not exported from `fie` |
| DV-9 | Size | `fie/policy.py` 482 lines (estimate 250–400); other production changes +212 / −39 (estimate 80–150); tests 1,351 lines (estimate 800–1,300). Over the estimates again, mostly docstrings and the rewritten `preflight_check` |
| DV-10 | Test-file corrections after the first run | Escaping typo; preset digests pinned from the implementation (section B) |
| DV-11 | One test runs `git status` | `test_detector_source_files_are_untouched` runs it read-only and skips outside a git checkout |

**Newly noticed risks**

| # | Risk | Note |
| --- | --- | --- |
| R1 | Log volume on a base install | Under the default `compat@1`, every allowed scan without the classifier now writes one INFO line (`reasons=missing_required_layer,…`). INFO is off by default in most configurations; it is one line per guarded call where enabled |
| R2 | `fie.preflight` now imports `fie.adversarial` at import time | Through `fie.policy`. `import fie` already imported it, so package users see no change; code importing `fie.preflight` in isolation did so before only on first scan |
| R3 | Operators may over-read `FIE_POLICY=strict` | It is enforced on the `preflight_check` path only (section H). The server routes and the `monitor` decorators use that path; the three SDK integrations do not |
| R4 | `FIE_SCAN_FAILURE_MODE=open` with `FIE_POLICY=strict` blocks scanner failures | By design (plan §14.4 R3): a selected preset's own setting wins |
| R5 | The CLI writes to stderr under an invalid configuration, including with `--quiet` | By design (plan §14.4 R2); the only output outside stdout |
| R6 | Pre-existing, unchanged | NF-10 (cold-start race reported as `timeout`, which `strict@1` now blocks), NF-11 (translation gap), NF-13 (CI tiebreaker call) |

**Blocked gates:** none.

---

## K. Numbering

The roadmap ([ROADMAP.md](../ROADMAP.md)) calls the policy engine **WP-007** and uses
**WP-004** for "install truth" (E51) and WP-005 for the hermetic default. By owner decision
W4-1 this package keeps the number **WP-004**. The roadmap was not edited. Its WP-004 and
WP-005 topics are not done and need new numbers when they are scheduled; its WP-007 sketch
(policy documents, per-category thresholds, per-tenant revisions, `risk.score`) remains
future work that this package is the first, minimal step toward.

---

## L. Final state

```text
git rev-parse --short HEAD   ->  924c70d
git diff --cached            ->  (empty: nothing staged)
git status --short           ->
 M docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md
 M fie/__init__.py
 M fie/__main__.py
 M fie/preflight.py
 M tests/test_scan_result_contract.py
?? docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_004_runtime-policy.md
?? docs/fie_rebuild_2026/IMPLEMENTATION_PLANS/PLAN_004_RUNTIME_POLICY.md
?? fie/policy.py
?? tests/test_runtime_policy.py
evals/baselines              ->  BL-0001…, BL-0002…, REF-0001…, CANONICAL -> BL-0002_pair-v6.3b_fie-70de7053
```

No staging, commit, push, merge, deployment or branch change occurred. The plan file was not
committed and not edited.

**Rollback:** delete `fie/policy.py` and `tests/test_runtime_policy.py`; revert
`fie/preflight.py`, `fie/__main__.py`, `fie/__init__.py` and the two assertions in
`tests/test_scan_result_contract.py`. Nothing is persisted. A running deployment returns to
previous behaviour by unsetting `FIE_POLICY` and restarting.

## Manual commit line

Not executed by the assistant. The plan file is included because it is the contract for this
change; drop it from the command if it is to be committed separately.

```text
git add -- fie/policy.py fie/preflight.py fie/__main__.py fie/__init__.py tests/test_runtime_policy.py tests/test_scan_result_contract.py docs/fie_rebuild_2026/IMPLEMENTATION_PLANS/PLAN_004_RUNTIME_POLICY.md docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_004_runtime-policy.md docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md && git commit -m "fie: add runtime policy layer applied in preflight_check (WP-004)"
```
