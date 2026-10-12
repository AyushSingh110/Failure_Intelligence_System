# EXECUTION_021, Phase 2 — CI release gate (WP-021)

| | |
| --- | --- |
| Work package | WP-021, **Phase 2 only** |
| Status | **IMPLEMENTED AND VALIDATED LOCALLY. NOT YET RUN ON A CI RUNNER.** The gate passes the corrected wheel and fails the pre-fix wheel on Windows. The workflow change has never executed: CI runs only for `main`, and nothing was pushed (section I, 1) |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, merged, published or deployed |
| Date | 2026-10-12 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `630e75f`, unchanged throughout |
| Contract | [PLAN_021_PACKAGING_BASE_IMPORT.md](../IMPLEMENTATION_PLANS/PLAN_021_PACKAGING_BASE_IMPORT.md) revision 1, §5 Levels 1 to 3, §6 Phase 2, with the owner's Phase 2 authorization |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase2/` |

---

## A. Summary

| | Result |
| --- | --- |
| Phase 1, after the owner's commit | Guard test passes; 974 tests pass; baseline byte-identical |
| `scripts/release_check.py` | Now checks two targets, T-base and T-fastapi, one environment at a time, with per-target criteria and exit codes |
| Positive control: wheel from `630e75f` | **Exit 0.** T-base 11 of 11 criteria pass, T-fastapi 10 of 10 |
| Negative control: wheel from `becc2c7` | **Exit 1.** `import fie` fails in both targets with `ModuleNotFoundError: No module named 'starlette'` |
| `ci.yml` | The `package` job runs the gate on the wheel it built and uploads the report |
| Tests after the change | 997 passed, 0 failed (974 + 23 new) |
| `python -m evals baseline` | Exit 0; all 11 record files byte-identical to `BL-0002` |

A passing gate means the wheel installs, imports, and its FastAPI integration answers two
fixed requests as documented. **It is not evidence of detection quality, and it does not show
that the middleware is safe to rely on in production** (section J).

---

## B. Phase 1 verification, before any change

First check, at `0fa17a9`: stopped. Phase 1's source, tests and extra were committed, but
`scripts/release_check.py` was not tracked in any commit (`.gitignore` line 153 excludes
`scripts/*`). Reported; the owner committed it as `630e75f`.

Second check, at `630e75f`:

| Gate | Result |
| --- | --- |
| Branch, HEAD, tree | `rebuild/wp-001-eval-harness`, `630e75f`, clean, nothing staged |
| `630e75f` against `0fa17a9` | One file added, `scripts/release_check.py`, byte-identical to the Phase 0 version (`e6e77199…`). No other difference |
| Tracked | `scripts/release_check.py`, `tests/test_base_import.py`, `tests/test_fastapi_integration.py`, `fie/integrations/__init__.py`, `fie/integrations/fastapi.py`; `pyproject.toml` has the `fastapi` extra |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |
| Phase 0 and Phase 1 evidence | 19, 19 and 39 files verify |
| `python -m pytest tests/test_runtime_policy.py::test_detector_source_files_are_untouched -p no:cacheprovider` | exit 0, 1 passed |
| `python -m pytest tests/ -m "not network" -p no:cacheprovider` | exit 0, **974 passed**, 0 failed, 0 skipped |
| `python -m evals baseline` | exit 0, 11 record files byte-identical to `BL-0002` |

The full suite and the baseline were run at `0fa17a9`. `630e75f` differs from it only by
adding a script that no test imported at that point, so they were not repeated before the
Phase 2 edits; both were run again after them (section G). The guard test's prediction from
Phase 1 held: it passes once the files are committed.

The script is tracked now but still matched by the ignore rule. That is harmless for a
tracked file; `.gitignore` was not edited.

---

## C. Files changed

| File | Change |
| --- | --- |
| `scripts/release_check.py` | Extended: two targets, Level 3 criteria, per-target reporting, repeatable `--wheelhouse`, `--targets`, POSIX layout and environment |
| `.github/workflows/ci.yml` | `package` job: two steps added after "Check wheel contents" |
| `tests/test_release_check.py` (new) | 23 tests of the gate's classification and exit codes |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_021_phase2-ci-release-gate.md` (new) | This report |
| `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Section 14.3 |
| `docs/fie_rebuild_2026/evidence/WP021/phase2/**` (new) | Evidence |

Not changed: `fie/`, `pyproject.toml`, the model manifest and loader, `.gitignore`,
`publish-pypi.yml`, `dist/`, baselines, every existing test.

---

## D. The gate

```text
python scripts/release_check.py --wheel <file> --out <dir>
    [--targets base,fastapi] [--wheelhouse <dir>]... [--workdir <dir>] [--keep]
```

It remains the one release check. `phase1/installed_checks.py`, the temporary script used in
Phase 1, is superseded by it and is kept only as that phase's evidence.

**How it runs.** For each target in turn: create a `venv` under the work directory, install
the wheel file (with `[fastapi]` for T-fastapi), run every check through that environment's
interpreter with `-I` from an empty directory, record, remove the environment, then start the
next. Only one environment exists at a time. The work directory is printed at the start and
must be outside the repository. Every command's stdout and stderr go to
`<out>/commands/<target>/`, written once and kept whatever the outcome; `--keep` also keeps
the environments. Commands that run the installed package get `FIE_NO_TELEMETRY=1` and
`FIE_NO_AUTO_DOWNLOAD=1`, a temporary home directory, and an explicit environment.

**Criteria.**

| Criterion | T-base | T-fastapi |
| --- | --- | --- |
| AC-1 wheel installs (and, with an extra, the wheel provides it) | yes | yes |
| AC-2 `pip check` | yes | yes |
| AC-3 installed set equals the closure the wheel declares for this target | yes | yes, including the extra |
| AC-4 installed files equal the wheel's | yes | yes |
| AC-5 `import fie` from this environment's `site-packages` | yes | yes |
| AC-6 every name in `fie.__all__` resolves | yes | yes |
| AC-7 `python -m fie --help` and `detect … --output json` | yes | yes |
| AC-8 `import fie` imports no optional package (`starlette`, `fastapi`, `anyio`, `openai`, `anthropic`, `joblib`, `sklearn`, `xgboost`, `onnxruntime`) | yes | yes |
| Starlette and FastAPI are not installed | yes | — |
| AC-10 three ways of asking for the integration each raise `ModuleNotFoundError`, `name == "starlette"`, message containing `pip install "fie-sdk[fastapi]"` | yes | — |
| AC-12 the SDK wrappers import; building a client without its SDK is an `ImportError` naming it | yes | — |
| AC-11 the three imports succeed | — | yes |
| AC-11 the middleware comes from this environment, is the documented class, Starlette was not imported by `import fie`; a known injection is answered 400 with `adversarial_input_blocked` and the documented message; a benign request reaches the handler | — | yes |

Not implemented: AC-9 (a repository test), Level 4, Level 5, release hygiene.

**Verdicts.** Each criterion is `PASS`, `FAIL`, `NOT_EVALUATED` or `TOOL_FAILED`. Each target
gets its own exit code; the run's exit code is 1 if any target has a failure, otherwise 2 if
any target could not run or left a criterion unevaluated, otherwise 0. A missing target, or a
target with no criteria, is never a pass. A `pip` that warns that the wheel does not provide
the requested extra fails AC-1 even though pip exits 0. Every line of output names the target
and the criterion.

**Portability.** The runner is `ubuntu-latest` with Python 3.11. The script previously had
Windows paths in its import check and an untested POSIX branch. Changed: the venv layout comes
from one function for both systems; the site-packages test uses resolved paths and no
Windows-only prefix; the POSIX environment passes through `LD_LIBRARY_PATH` (a relocated
interpreter such as GitHub's needs it to start) and sets `LANG` and `TMPDIR`; the user name is
read from `USER` as well as `USERNAME` for redaction. **None of this has been executed on
Linux** (section I, 1).

---

## E. CI

| | |
| --- | --- |
| Workflow | `.github/workflows/ci.yml` |
| Triggers | `push` to `main`; `pull_request` targeting `main`. Unchanged |
| Job | `package`, `runs-on: ubuntu-latest`, Python 3.11, `needs: test`. Unchanged |
| Existing steps | checkout; set up Python; install `build`, `hatchling`; download model artifacts (best effort); `python -m build --wheel`; check wheel contents. Unchanged |
| New step | "Release check (installed wheel — T-base and T-fastapi)": requires exactly one wheel in `dist/`, prints its SHA-256, runs `python scripts/release_check.py --wheel <that file> --out "$RUNNER_TEMP/release-check" --workdir "$RUNNER_TEMP"` |
| New step | "Upload release-check report": `actions/upload-artifact@v4`, `if: always()`, so the report and per-command output survive a failure |
| Downstream | `deploy-space` already `needs: [test, package, health-check]`, so a failing gate now also stops the deployment to the Space |

The wheel is built once, by the existing step, and the gate is given that file. No
`--wheelhouse` is passed, so on CI dependencies come from PyPI: this is the path
`pip install fie-sdk` takes, including the source build of `langdetect`.

`publish-pypi.yml` is unchanged. It does not run the gate yet; that is Phase 4.

---

## F. Controls

Both built with `python -m build --wheel --no-isolation` from a `git archive` export in a
scratch directory, `fie/models/` holding only `.gitkeep`, no working-tree file. Dependencies
from the two verified local wheelhouses, `--no-index`.

| | Positive | Negative |
| --- | --- | --- |
| Source commit | `630e75fec0b97b03ff49c1b91a12ddedbd6b62f5` | `becc2c746df205e7cf39e90ed1e4c985ba705355` |
| Wheel | `fie_sdk-1.18.0-py3-none-any.whl`, 152,716 bytes | same name, 151,936 bytes |
| SHA-256 | `eb251ccb7f9385d4af4fb728a7c0528b9bdc3c61674aef890b9b464414b3e37d` | `cfdf18598463f36956e5b765b3209ed05a20f20b7385297a3b3586e9dceaf4e4` |
| Cross-check | Differs from the Phase 1 wheel in one file, by line endings only (section I, 3) | **Byte-identical** to the Phase 0 wheel |
| **Run exit code** | **0** | **1** |
| T-base exit code | 0 | 1 |
| T-fastapi exit code | 0 | 1 |
| Evidence | `phase2/control_positive/` | `phase2/control_negative/` |

**Positive, T-base** (14 distributions; no `starlette`, `fastapi` or `anyio`): all 11 pass.
AC-10's message: `fie.integrations.fastapi needs Starlette, which is not installed. Install it
with: pip install "fie-sdk[fastapi]"`.

**Positive, T-fastapi** (17 distributions; `starlette` 0.52.1, `anyio` 4.15.1,
`exceptiongroup` 1.3.1): all 10 pass. Injection answered 400, `PROMPT_EXTRACTION`; benign
request reached the handler.

**Negative, T-base:** AC-1 to AC-4 pass. **AC-5 FAIL: `import fie raised ModuleNotFoundError:
missing module 'starlette'`.** AC-7 FAIL, same error. AC-6, AC-8, AC-10, AC-12 `NOT_EVALUATED`
because the import did not complete. Captured traceback ends:

```text
  File "<WORK>\T-base\env\lib\site-packages\fie\integrations\fastapi.py", line 28, in <module>
    from starlette.middleware.base import BaseHTTPMiddleware
ModuleNotFoundError: No module named 'starlette'
```

**Negative, T-fastapi:** AC-1 FAIL, `the wheel does not provide the extra 'fastapi'` (pip
installs it anyway, without Starlette); AC-5 and AC-7 FAIL with the same `starlette` error;
the AC-11 criteria `NOT_EVALUATED`.

Both controls behaved as expected and for the expected reason. The negative control's
T-fastapi result is a consequence of the same defect plus the missing extra, not a second,
independent defect.

---

## G. Tests and baseline, after the change

| Command | Exit | Result |
| --- | --- | --- |
| `python scripts/release_check.py --selftest` | 0 | 49 of 49 |
| `python -m pytest tests/test_release_check.py -p no:cacheprovider` | 0 | 23 passed |
| `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` | 0 | **997 passed**, 0 failed, 0 skipped |
| `python -m evals baseline` | 0 | Standard counts match; all 11 per-prompt record files byte-identical to `BL-0002_pair-v6.3b_fie-70de7053` |

`tests/test_release_check.py` covers the deciding, not the installing: only all-pass gives
exit 0; every target must pass; a missing target is not a pass; the E51 import failure is a
failed criterion naming the module; a failed tool says nothing about packages; the base target
rejects a Starlette nobody asked for; the FastAPI target requires the extra's packages; AC-10
accepts only the documented error; the middleware must block the injection and pass the benign
request; usage errors exit 2. It creates no environment.

---

## H. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase2/`

| Path | Content |
| --- | --- |
| `build_positive.json`, `build_negative.json` | Source commit, build command, wheel identity, comparison with the earlier wheel of the same source |
| `control_positive/` | `report.json` (per-target criteria, exit codes, installed versions, every command), `commands/T-base/`, `commands/T-fastapi/`, `SHA256SUMS.txt` (45 files verify) |
| `control_negative/` | The same for the pre-fix wheel (37 files verify), including the captured `starlette` traceback |

No file contains the user name. Phase 0 and Phase 1 evidence is unchanged and still verifies.

Scratch, kept: `C:\Users\ASUS\fie-e51-scratch\wp021\phase2\` (two source exports, two
wheels, an empty work directory). The gate removed its own environments (`removed: true` for
each target in both runs). No existing scratch content was changed.

---

## I. Limitations and deviations

| # | Point |
| --- | --- |
| 1 | **The CI change has not run.** The workflow triggers on `main` only; this work is on `rebuild/wp-001-eval-harness` and nothing was pushed. The plan's Phase 2 exit condition, "the CI job fails on a deliberately broken branch and passes on the fix", is therefore **not met on a runner**. It is met locally, on Windows, by the two controls. The POSIX paths, `LD_LIBRARY_PATH` handling, the install from PyPI and the `langdetect` source build on Python 3.11 are all first exercised when a pull request to `main` runs. Docker was not available here to try a Linux container. The first CI run should be read as a test of the gate as much as of the wheel |
| 2 | On CI the wheel may contain model files (the job downloads them, best effort, before building). The gate does not check wheel contents against the manifest; that is Phase 3 |
| 3 | **Line endings.** This checkout has `core.autocrlf=true`, so `git archive` writes CRLF. The positive wheel differs from the Phase 1 wheel in `fie/integrations/__init__.py`, by line endings only (58 CRLF against 0); the Python is the same. A wheel built on Linux from the same commit will have LF and a different SHA-256. Cross-platform build identity is not established and matters for Phase 3 and 4 |
| 4 | Local controls installed dependencies from local wheelhouses with `--no-index`; CI will use PyPI. The index path of the tool is untested |
| 5 | Python 3.10.19 locally; CI's `package` job uses 3.11. Starlette 0.52.1 only; on CI `starlette>=0.52.1` resolves to whatever PyPI has that day, which may be newer and is untested |
| 6 | The gate's middleware check uses two fixed prompts. In a base install the injection is caught by a layer that needs no model. This is a functional check, not a measurement |
| 7 | AC-12 was added to T-base (the plan lists it under Level 3; the authorization's list of T-base checks does not name it). It only asserts existing behaviour |
| 8 | One criterion is named "T-base: starlette and fastapi not installed", which repeats the target label in the output. Cosmetic; left as it is so the evidence matches the script |

---

## J. Unresolved security findings in the middleware

Recorded here because a green gate could otherwise be read as covering them. **Neither was
changed in this phase.**

| # | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| SF-1 | **The middleware fails open when the scan raises.** `FIEMiddleware._scan` wraps the scan in `except Exception`, logs a warning, and returns "not an attack"; `dispatch` then forwards the request | `fie/integrations/fastapi.py`, `_scan` (STATIC) | Any error inside `scan_prompt`, including one an attacker can provoke with a crafted input, turns the middleware into a pass-through for that request. The caller is not told |
| SF-2 | **The middleware bypasses the runtime policy of WP-004.** It calls `scan_prompt` directly and blocks only on `is_attack`. It does not call `preflight_check` or `fie.policy.decide` | the same function (STATIC) | `coverage`, `zone` and the configured policy are ignored. In a base install the classifier layer is `unavailable_dependency` and coverage is `partial`; under the `STRICT` preset `preflight_check` would not allow that, but the middleware forwards whatever the remaining layers do not flag. `FIE_POLICY` has no effect on it |
| SF-3 | Requests the middleware does not inspect at all: methods other than POST, PUT, PATCH; bodies that are not JSON; JSON without one of the six recognised field names or a `messages` list; prompts shorter than 10 characters; any path in `exempt_paths` | `dispatch`, `_extract_prompt` (STATIC); the pass-through cases are asserted as current behaviour in `tests/test_fastapi_integration.py` | The protection covers a specific request shape, not "every route" as the module docstring says |

What the gate does establish about the middleware: it imports from the installed wheel, and
for one known injection in a JSON `prompt` field it returns the documented 400. That is all.
SF-1 and SF-2 are the integration counterpart of the gap the WP-004 plan deferred ("enforcement
only in `preflight_check`; integrations bypass"), and belong in a package of their own.

## K. Recommended manual commit command

Not executed by the assistant.

```text
git add scripts/release_check.py .github/workflows/ci.yml tests/test_release_check.py docs/fie_rebuild_2026 && git commit -m "ci: run the installed-wheel release gate on T-base and T-fastapi (WP-021 phase 2)"
```
