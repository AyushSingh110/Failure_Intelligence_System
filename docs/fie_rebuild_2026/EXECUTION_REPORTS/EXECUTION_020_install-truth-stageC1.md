# EXECUTION_020 — Clean-Environment Installation Truth, Stage C1 (WP-020 / E51)

| | |
| --- | --- |
| Work package | WP-020, experiment E51. **Stage C1 only: base installs.** |
| Status | **STAGE C1 EXECUTED. Every base cell installed and passed its gates; no cell could import `fie`; zero scans completed.** Awaiting owner decision |
| Commit status | **NOT COMMITTED by the assistant.** The owner commits manually. Nothing was staged, pushed, merged, deployed, stashed or reset by the assistant |
| Dates | 2026-10-11 to 2026-10-12 |
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD | `2f62ecd` at the start of Stage C1; `3f823ad` at the end. The two commits in between (`af051a7`, `3f823ad`) are the owner's, and add only documents and evidence under `docs/fie_rebuild_2026/`. `fie/`, `pyproject.toml` and `scripts/model_manifest.json` are identical to `2f62ecd` |
| Contract | [PLAN_020_INSTALL_TRUTH.md](../IMPLEMENTATION_PLANS/PLAN_020_INSTALL_TRUTH.md), revision 6, §0.2, with the owner's Stage C1 authorization and its four later, narrower authorizations (section J) |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Kind of work | Measurement only. No change to `fie/`, `tests/`, packaging, dependencies, models, workflows, `dist/` or baselines |
| Not done, by scope | Both `[ml]` dependency sets: **not provisioned: awaiting explicit owner approval** (O-21, O-22). No product fix |

Labels used below. **MEASURED**: observed in an offline, gated run. **BLOCKED**: could not be
measured, with the reason. **STATIC**: read from a file or archive without running it.
**PREDICTED**: implied by the code; not observed. **DIAGNOSTIC**: observed while connected,
outside the measurement window; never used as a gate result.

---

## A. Summary

1. **MEASURED. A base install of `fie-sdk` cannot be imported.** In all four base cells
   `import fie` raises `ModuleNotFoundError: No module named 'starlette'`. The chain is
   `fie/__init__.py` → `fie/integrations/__init__.py` line 25 → `fie/integrations/fastapi.py`
   line 28 (`from starlette.middleware.base import …`). This holds for the published PyPI
   wheel `1.18.0` and for wheels built from the current tree (`2f62ecd`), with and without
   model files. The command line (`python -m fie detect|explain`) fails the same way, exit 1.
2. **MEASURED. Installation itself works**, offline, from a hash-pinned wheelhouse, in both
   install modes. Every post-install gate passed in all four cells.
3. **MEASURED. `pip install fie-sdk` needs a source build.** With the venv's bundled pip
   23.0.1, `langdetect` is installed from PyPI's source archive by the legacy
   `setup.py install` path: no wheel is built and no isolated build environment is used. pip
   prints a deprecation warning for that path.
4. **MEASURED. C9 holds for the payload.** The 69 installed `langdetect` package files are
   byte-identical between the source-built environment and all three pre-built-wheel
   environments. Only installer-generated metadata differs.
5. **BLOCKED. Every runtime question for base installs.** Scan records produced: **0** in
   every cell. Comparison C1 against `BL-0002` could not be made. Nothing here is evidence
   about detection quality, in either direction.

| Gate | Result |
| --- | --- |
| Tests (`tests/`) | 942 passed, 0 failed, exit 0: at the start, after the runner correction, and after the harness correction |
| `python -m evals baseline` | Exit 0, 11 record files byte-identical to `BL-0002`, all three times |
| E51 harness tests | 58 passed, exit 0 |
| Isolation, runs that measured | Offline at the start and end; TCP refused through every launcher and the base interpreter; recovery verified |
| Evidence checksums | Attempts 4, 5, 7 and all three diagnostics verify |

---

## B. State

| | Start of Stage C1 | End |
| --- | --- | --- |
| HEAD | `2f62ecd` | `3f823ad` (owner commits; docs and evidence only) |
| Staged | nothing | nothing |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` | unchanged |
| `dist/fie_sdk-1.18.0-py3-none-any.whl` | `e0b415d4…` | unchanged; inspected statically only |
| Stage B and remediation evidence | 46 files | unchanged, hash-compared |

---

## C. Artifacts and provenance

| Artifact | SHA-256 | Origin | Verified |
| --- | --- | --- | --- |
| A-pypi-1.18.0, `fie_sdk-1.18.0-py3-none-any.whl` (270,149 bytes) | `be504dc748822699df5dc5353ad0038617ef639a4c1840ed18786bc8579e963d` | PyPI | Equal to PyPI's published digest on 2026-10-11 and twice on 2026-10-12. Latest release is `1.18.0` each time |
| A-clean (151,936 bytes) | `8710486706d5c243957e077de60e1e9906b3d0317ea8c9e870d278162b839e99` | Local build from `git archive 2f62ecd`, `python -m build --wheel --no-isolation` | Built twice, byte-identical |
| A-release (314,838 bytes) | `17b23a9e719591a349cc1baeb9efc4d79b8e336f6e38d18612df8a8a7ca6400c` | The same, plus the manifest-verified files of `fie/models/` | Built twice, byte-identical |
| `langdetect-1.0.9.tar.gz` | `cbc1fef89f8d062739774bd51eda3da3274006b3661d199c2655f6b3f6d605a0` | PyPI source archive (mode A only) | Equal to PyPI's digest |
| `langdetect-1.0.9-py3-none-any.whl` | `9acdfda7ee22ffdaaaf87f993cec0eb186d772fe0f3b93469832d83be6e9c8a4` | Built locally in Stage B remediation from that archive (mode B only) | Not a PyPI wheel; never given a PyPI digest |

Model files, **STATIC**, from the wheel archives:

| Artifact | Verified | Unmanifested | Hash mismatch |
| --- | --- | --- | --- |
| A-pypi-1.18.0 | 9 | 0 | 1: `fie/models/meta_clf.pkl` |
| A-clean | 0 | 0 | 0 (ships no model file) |
| A-release | 10 | 0 | 0 |
| A-dist (not installed) | 9 | 4 | 1 |

No pickle or model file was opened in any run; the tripwire never fired.

Dependencies. The base set is 11 distributions. It was re-resolved three times in this stage
(2026-10-11, and twice on 2026-10-12) and equalled the approved pins each time. Mode A wheelhouse:
14 files (10 PyPI wheels, the source archive, 3 build tools), no `langdetect` wheel. Mode B
wheelhouse: 11 wheels (10 from PyPI, the local `langdetect` wheel), no source archive. Every
file equals its ledger. Only `pypi.org` and `files.pythonhosted.org` were contacted; no pip
configuration, proxy or `hosts` redirection was found.

---

## D. Results by cell

All four: Python 3.10.19, a `venv` of the existing interpreter, extra = base.

| | modeA-pypi-base-r3 | modeB-pypi-base-m1 | modeB-clean-base-m1 | modeB-release-base-m1 |
| --- | --- | --- | --- | --- |
| Artifact | A-pypi-1.18.0 | A-pypi-1.18.0 | A-clean | A-release |
| Install mode | **A**, source build | B, pre-built local wheel | B | B |
| Installer | bundled pip 23.0.1 | repository pip 26.0.1 via `--python` | same | same |
| Evidence | `offline_attempt5/` | `offline_attempt7/` | `offline_attempt7/` | `offline_attempt7/` |
| pip exit, seconds | 0, 10.1 | 0, 5.6 | 0, 7.3 | 0, 5.1 |
| Source build | `langdetect`, legacy `setup.py install` | none | none | none |
| `pip inspect` | `match`, 14 (12 pinned + `pip`, `setuptools`) | `match`, 12 | `match`, 12 | `match`, 12 |
| `pip check` | pass | pass | pass | pass |
| Interpreter identity | pass | pass | pass | pass |
| 11 dependency imports | all succeed | all succeed | all succeed | all succeed |
| Installed `fie` vs wheel | 53 files identical | 53 identical | 37 identical | 55 identical |
| Start-up code | `distutils-precedence.pth` (bundled setuptools) | none | none | none |
| Model files installed | 9 verified, 1 mismatch | 9 verified, 1 mismatch | none | 10 verified |
| Gates | **PASS** | **PASS** | **PASS** | **PASS** |
| `import fie` | `ModuleNotFoundError` | `ModuleNotFoundError`, `starlette` | same | same |
| Mode S, mode PC | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| CLI, 4 invocations | exit 1 each, same error | same | same | same |
| **Scan records** | **0** | **0** | **0** | **0** |
| Outcome | `IMPORT_BLOCKED_NO_SCAN_COMPLETED` | same | same | same |

Notes.

- The attempt 5 probe recorded only the exception class; the module name for
  `modeA-pypi-base-r3` comes from that run's four CLI tracebacks, which all name `starlette`.
  The corrected probe recorded name, message and import chain for the three mode B cells.
- The outcome label for `modeA-pypi-base-r3` was assigned afterwards from its evidence. The
  runner that produced it printed `RESULT: COMPLETED`, which was wrong (section J, DV-6).
- Repeat runs were not attempted: when the first run of a mode produces no measurement the
  runner does not start the second. There are no repeat-run differences to report.
- Mode A was also installed once in `modeA-pypi-base-r1` (attempt 4): same build path, same
  `langdetect` payload. That cell was not measured (section J, DV-4).
- `starlette` appears in the wheel's metadata only through `fastapi`, under the `dev` and
  `server` extras. It is not a base dependency. **STATIC.**
- Mode P is not applicable to base cells (no pickle can load without `joblib`).
- **PREDICTED, not measured:** installing `fie-sdk[server]`, or any extra that brings
  `starlette`, would get past this import. What the package then does is unknown.

Blocked questions, each cell: RQ4 (coverage and self-report), RQ5 (default verdicts), RQ9
(warnings), RQ10 and H1 (agreement with `BL-0002`, comparison C1), RQ11 (published versus
release behaviour, C4), H5, H8 (policy actions, C5), and attempted egress at scan time. None is
filled in from another mode, cell or artifact.

---

## E. Installation path (mode A)

The evidence required by PLAN_020 §6.4:

| # | Evidence | Result |
| --- | --- | --- |
| F-1 | Artifact is the PyPI wheel, by digest | Yes |
| F-2 | `langdetect` installed from the PyPI source archive, not a wheel | Yes: the log names the archive; the wheelhouse holds no `langdetect` wheel |
| F-3 | Build path and setuptools | Legacy `setup.py install`, with the setuptools 79.0.1 already in the venv. No wheel built, no isolated build, wheelhouse build tools unused |
| F-4 | Every other distribution from a pinned wheel | Yes |
| F-5 | Network off | Yes (section G) |
| F-6 | Installer | pip 23.0.1, bundled with the venv |
| F-7 | Installed set equals pins; `pip check` clean | Yes |
| F-8 | `import langdetect` and `import fie` succeed | `langdetect`: yes. **`fie`: no** |

So: the published wheel was installed by pip, offline, from a verified wheelhouse, through a
source build of `langdetect`. It differs from an ordinary user's install in three ways: the
index was a local wheelhouse; the installer was pip 23.0.1 (pip 23.1 and later removed the
legacy path, so a newer pip takes a different one, **not measured**); and the build-tool
versions were those of this machine. F-8 fails for `fie`, which is finding 1, not an
installation failure.

---

## F. Comparisons

| # | Outcome |
| --- | --- |
| C1 | **Not possible.** No scan record exists to compare with `BL-0002` `lite.std` |
| C4 | **Not possible at runtime.** The import outcome is the same for A-pypi-1.18.0 and A-release |
| C7, C8 | Not applicable: mode P does not exist for base cells |
| C9 | **Payload identical.** 69 files under `langdetect/`, same paths and SHA-256, in mode A (`r3`) and in each of the three mode B cells. Metadata differs and is listed separately: `langdetect-1.0.9-py3.10.egg-info/` (6 files) in mode A, `langdetect-1.0.9.dist-info/` (8 files) in mode B, plus `__pycache__` in both |

C9 removes `langdetect`'s files as a difference between the install modes. It licenses no
runtime statement yet, because no cell reached runtime.
Files: `evidence/E51/stageC1/comparisons/`.

---

## G. Isolation and recovery

Airplane mode, switched by the owner by hand. No script changed a network setting.

| | Attempt 5 (mode A) | Attempt 7 (mode B) |
| --- | --- | --- |
| Positive control, connected | launcher + base interpreter | 3 launchers + base interpreter |
| Window (UTC, 2026-10-12) | 05:07:06 to 05:07:29 | 06:08:25 to 06:09:09 |
| Adapters up, start / end | none / none | none / none |
| Usable default routes, start / end | 0 / 0 | 0 / 0 |
| TCP to two PyPI addresses, port 443, start / end | refused through 2 / 2 executables | refused through 4 / 4 |
| Launchers changed during the window | none | none |
| Recovery | verified 05:07:47 | verified 06:10:59 |

Limits. This shows that no physical interface was up, that no usable default route existed,
and that these executables could not open those connections, at the start and at the end of
each window. It is not a packet capture. One stale default-route entry remained listed on the
disconnected Wi-Fi interface throughout; it is recorded and counted as unusable. Every venv
launcher re-executes the base interpreter, so the operating system reports the base
`python.exe` as the process image for all of them. The probe's in-process socket recorder saw
no attempt, which says only that import did not get far enough to try.

---

## H. Tests

| Command | Exit | Result | When |
| --- | --- | --- | --- |
| `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` | 0 | 942 passed, 0 failed | Start of C1; after the runner correction; after the harness correction |
| `python -m evals baseline` | 0 | 11 record files byte-identical to `BL-0002` | The same three times |
| `python -m pytest docs/fie_rebuild_2026/experiments/E51/test_offline_runner.py -p no:cacheprovider` | 0 | 41 passed, then 58 passed after the second correction | After each harness change |

No test failed at any point. `tests/` was not modified. The harness tests live outside `tests/`
and do not change the 942 count.

---

## I. Files

Created or changed in Stage C1, all under `docs/fie_rebuild_2026/`:

| File | Purpose |
| --- | --- |
| `IMPLEMENTATION_PLANS/PLAN_020_INSTALL_TRUTH.md` | Revisions 5 and 6: Stage C1 instructions (§0.2); superseded firewall and private-interpreter outline marked as such |
| `experiments/E51/README.md` | The runnable procedure, gate categories, cell outcomes, corrections |
| `experiments/E51/prepare_c1.py`, `prepare_retry.py`, `prepare_modeb.py` | Connected preparation |
| `experiments/E51/offline_runner.py` | The owner-run offline runner |
| `experiments/E51/probe.py`, `cli_guard.py`, `egress_selftest.py`, `inspect_wheel.py`, `export_inputs.py` | Measurement and inspection |
| `experiments/E51/test_offline_runner.py` | 58 regression tests for the harness |
| `evidence/E51/stageC1/**` | Provisioning, seven offline attempts, three diagnostics, comparisons |
| `EXECUTION_REPORTS/EXECUTION_020_install-truth-stageC1.md` | This report |
| `MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Section 13 |

Evidence index (`evidence/E51/stageC1/`):

| Directory | Content |
| --- | --- |
| `provisioning/`, `provenance/`, `pins/`, `artifacts/`, `isolation/` | Connected provisioning of 2026-10-11 |
| `offline/`, `offline_attempt2/`, `offline_attempt3/` | Stopped before anything was installed (DV-1, DV-2) |
| `offline_attempt4/` | Mode A installed in `r1`; verification tool failed (DV-3) |
| `diagnostic_pip_inspect_attempt4/`, `diagnostic_pip_inspect_utf8_attempt4/` | DIAGNOSTIC: the failure and the fix |
| `retry1/`, `offline_attempt5/` | **Mode A, measured** |
| `diagnostic_pip_inspect_modeB_command/` | DIAGNOSTIC: the mode B inspection command |
| `modeb1/`, `offline_attempt7/` | **Mode B, measured** |
| `offline_attempt6/` | Interrupted before the offline window (DV-7) |
| `comparisons/` | C9 and the per-cell summary |

---

## J. Deviations and corrections

| # | What happened | Handling |
| --- | --- | --- |
| DV-1 | Attempt 1 stopped: Wi-Fi was still up when the run continued | Correct stop. Nothing installed |
| DV-2 | Attempts 2 and 3 stopped although airplane mode was on: the runner required the default-route list to be empty, and Windows keeps route entries on a disconnected interface | **Runner defect, mine.** Corrected to "no usable route" (the authorization's wording), failing closed on anything unclassifiable. Changed without asking first; reported to the owner before the next run. TCP controls unchanged |
| DV-3 | Attempt 4: install succeeded; `pip inspect` exited 2 and the runner reported all 12 distributions missing | **Runner defect, mine.** `pip inspect` died of a `UnicodeEncodeError` writing package descriptions to cp1252 output; `-I` ignores `PYTHONUTF8`. Diagnosed with one authorized connected invocation, confirmed with a second. Fixed with `-X utf8`; four-way classification so a tool failure is never read as a package finding; per-gate output kept. Owner-authorized |
| DV-4 | `modeA-pypi-base-r1` was installed but never measured | Kept, not reused. The retry used a fresh `r3` with its own positive control |
| DV-5 | Mode A was planned to run twice (`r1`, `r2`) to observe reproducibility | Not done as planned. Reproducibility of the install path is observed across `r1` and `r3` instead: same path, identical `langdetect` payload. `r2` was never used |
| DV-6 | Attempt 5 printed `RESULT: COMPLETED` with zero scan records, and the probe did not record the missing module | **Harness defect, mine.** Corrected before mode B, owner-authorized: cell outcomes, `blocked.json`, module name, message and import chain |
| DV-7 | Attempt 6 was interrupted at the keyboard before the offline window; it has no checksum file | Nothing installed. Kept as it is |
| DV-8 | Mode B cells were made independent (an install stop ends that cell only) | My change, stated to the owner before the run. No mode B install stopped, so it had no effect |
| DV-9 | Two connected diagnostics store `pip inspect` output with the home-path prefix replaced, not byte-exact | The repository is public. Byte-exact files are in the scratch directory, tied by SHA-256. Confirmed by the owner |
| DV-10 | The runner was started from the `base` conda environment, not `failure-engine` | No effect on the measurement: every interpreter path comes from the saved plan |

Owner authorizations after the Stage C1 authorization, in order: (1) one connected
`pip inspect` diagnostic; (2) one more with `-X utf8`; (3) runner correction and a mode A
retry in a fresh environment; (4) harness reporting correction, a mode B inspection-command
diagnostic, and the three mode B base cells.

---

## K. Limitations

- One machine, Windows 11, Python 3.10.19, one pip version per install mode.
- The mode A result is for pip 23.0.1. Newer pip versions take a different build path.
- Installs were from a local wheelhouse, not from PyPI.
- No runtime behaviour of any artifact was observed. In particular nothing is known from this
  stage about verdicts, coverage reporting, policy actions, warnings, or network use.
- `[ml]` installs were not attempted.
- The import failure was observed with the network off and with instrumentation loaded.
  Neither can make a module that is not installed importable; `pip inspect` confirms
  `starlette` is absent from all four environments.

---

## L. Recommendations (none acted on)

1. Treat "a base install of `fie-sdk` cannot be imported" as the Stage C1 finding and decide
   its consequence for the published `1.18.0` release. This is a product decision.
2. The fix belongs to WP-021 (packaging), not E51: either import the integrations lazily or
   behind a guard, or declare the dependency. Whichever is chosen, a clean-environment
   `import fie` in a base install is the regression test this experiment shows was missing.
3. Decide whether E51 continues to `[ml]` cells before or after that fix. **STATIC:** neither
   resolved `[ml]` set contains `starlette` or `fastapi` (Stage B remediation resolutions:
   current-tree `[ml]` 47 distributions, PyPI `[ml]` 56), and the wheel metadata lists
   `fastapi` only under `dev` and `server`. So the same import failure is **PREDICTED** for
   every `[ml]` cell, and the 209 MB and 338 MB downloads would likely measure nothing new
   until the import is fixed.
4. Remove the experiment's environments under `C:\Users\ASUS\fie-e51-scratch\env\` when the
   owner agrees; nothing was deleted in this stage.

## M. Recommended manual commit command

Not executed by the assistant.

```text
git add docs/fie_rebuild_2026 && git commit -m "docs: E51 Stage C1 base-install results - fie-sdk base install cannot be imported (WP-020)"
```
