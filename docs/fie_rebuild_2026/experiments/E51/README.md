# E51 — clean-environment installation truth: Stage C1 execution

Specification: [PLAN_020_INSTALL_TRUTH.md](../../IMPLEMENTATION_PLANS/PLAN_020_INSTALL_TRUTH.md),
§0.2 for this stage. This file is the runnable form. Where the two differ, the plan's §0.2 and
the owner's Stage C1 authorization govern.

**Scope of Stage C1.** Four base cells: one mode A cell (run twice) and three mode B cells.
Neither `[ml]` dependency set is downloaded or installed; those cells are recorded as
`not provisioned: awaiting explicit owner approval`. Nothing in `fie/`, tests, packaging,
`dist/`, baselines or Git history is changed.

## Fixed points

| | |
| --- | --- |
| Interpreter | The existing Python 3.10.19 (`failure-engine` environment). Environments are `venv`s of it. **No interpreter copy.** |
| Isolation | **Airplane mode, switched by the owner by hand.** No script changes a network setting. The per-program firewall design in the plan (§5.2, §5.3) is superseded and is not executed |
| Scratch | `C:\Users\ASUS\fie-e51-scratch` |
| Evidence | `docs/fie_rebuild_2026/evidence/E51/stageC1/` |

## Cells

| Environment | Artifact | Install mode | Installer |
| --- | --- | --- | --- |
| `modeA-pypi-base-r1`, `-r2` | Published `fie-sdk==1.18.0` wheel | **A**: pip builds `langdetect` from PyPI's source archive | the venv's bundled pip |
| `modeB-pypi-base` | Published wheel | **B**: pre-built local `langdetect` wheel | repository pip via `--python` |
| `modeB-clean-base` | A-clean, built from `git archive 2f62ecd` | B | same |
| `modeB-release-base` | A-release, the same plus manifest-verified model files | B | same |

`r1` and `r2` are one cell installed twice, to observe whether mode A is reproducible.

## The boundary

| Phase | Network | Who | What |
| --- | --- | --- | --- |
| 1. Connected provisioning | **up** | `prepare_c1.py` | Re-resolve the base set (must equal the approved pins); download and verify; assemble wheelhouse-A and wheelhouse-B with ledgers; build A-clean and A-release twice; inspect wheels statically; create the five **empty** environments; positive TCP control through each launcher and the base interpreter; export inputs |
| 2. Offline measurement | **airplane mode** | owner runs `offline_runner.py` | Negative controls; every install; every post-install gate; every import of `fie`; every probe; C9 hashing; closing negative controls |
| 3. Connected again | **up** | owner switches back; then `compare_c1.py` | Recovery check; comparisons C1, C4, C9; report |

Nothing is installed into a measurement environment in phase 1. Nothing is downloaded in
phase 2: every install uses `--no-index`.

## Controls

The same test, `egress_selftest.py`, started with `-I -S -B` so that no installed code runs,
through **six executables**: the base interpreter and the five environment launchers. It opens
a TCP connection to the literal IPv4 addresses that `pypi.org` and `files.pythonhosted.org`
resolved to during phase 1, port 443, and sends nothing.

| Control | When | Must |
| --- | --- | --- |
| Positive | Phase 1, after the environments exist | connect, through all six |
| Negative, opening | Phase 2, before any install | fail, through all six |
| Negative, closing | Phase 2, after the last probe | fail, through all six |
| Recovery | Phase 3 | connect, through the base interpreter |

The launchers are created before the positive control and are not recreated, moved or
replaced afterwards; the runner checks each launcher's SHA-256 against the value recorded with
its positive control, before the first install and again at the end. The runner also reads
(never writes) adapter state and default routes and stops unless no physical adapter is up and
no **usable** default route exists.

*Correction after attempts 1 to 3 (2026-10-11/12).* Attempt 1 stopped correctly: Wi-Fi was
still up. Attempts 2 and 3 stopped although airplane mode was on: every adapter reported
`Disconnected`, but Windows still listed the two default routes on the disconnected Wi-Fi
interface, and the runner required the list to be empty. That requirement was wrong: a listed
route on a disconnected interface carries no traffic. The runner now counts a default route as
usable unless its interface is positively known to be down (adapter status not `Up` **and**
every IP interface of that name reporting `Disconnected`), and records listed, usable and stale
counts. An interface it cannot classify counts as usable, so the check still fails closed. The
TCP controls through all six executables are unchanged and remain the decisive test. Nothing
was installed or measured in attempts 1 to 3; their evidence is kept.

**What this supports:** no network interface was up, and these six executables could not open
those connections, at the start and at the end of the window. **What it does not:** it is not
a packet capture. The probe's in-process socket recorder is an observation of attempts, not
isolation.

## Running it

Phase 1 (connected), from the repository root:

```text
python docs\fie_rebuild_2026\experiments\E51\prepare_c1.py
python -I -B docs\fie_rebuild_2026\experiments\E51\offline_runner.py --selftest
```

Phase 2 (owner), from the repository root, in an ordinary terminal that does not need the
network:

```text
python -I -B docs\fie_rebuild_2026\experiments\E51\offline_runner.py
```

The runner pauses and asks for airplane mode to be switched **on**; verifies it; runs; pauses
and asks for it to be switched **off**; verifies connectivity. It stops, keeping everything
written so far and still running the closing controls, if: the machine is not offline; any
control fails; a launcher changed; pip exits non-zero; pip builds anything other than
`langdetect` in mode A or anything at all in mode B; the pickle tripwire fires; the measured
code starts a process; or the probe cannot prove it measured the installed copy.

**If the mode A install fails, the runner stops and mode B is not run.** That is a result
(mode A not achieved), to be decided by the owner under O-23. Mode B is never substituted.

A second attempt writes to `offline_attempt2/`; it refuses to reuse an environment that
already has `fie` installed.

## Post-install gates: tool failure is not a package finding

*Added after offline attempt 4 (2026-10-12).* In attempt 4 the mode A install succeeded, but
`pip inspect` exited 2 and the runner reported all 12 pinned distributions as missing. They
were not missing. pip prints each package's description; several contain characters that
cp1252 cannot encode; with output redirected, pip's console writer raised
`UnicodeEncodeError`. The runner set `PYTHONUTF8=1`, but `-I` makes Python ignore that
variable. Evidence: `offline_attempt4/`, `diagnostic_pip_inspect_attempt4/` (the traceback)
and `diagnostic_pip_inspect_utf8_attempt4/` (the same command with `-X utf8`: exit 0, complete
JSON). Both diagnostics were connected and are not gate results.

**UTF-8 requirement.** `pip inspect` is run as `python -I -X utf8 -m pip inspect`. `-X utf8`
is a command-line option, which `-I` does not ignore. pip itself is unchanged (bundled
23.0.1). No other measurement command is given this flag. In particular the mode S command
line (`python -m fie …`) runs without it: if the installed CLI cannot write to redirected
cp1252 output, that is as-shipped behaviour and is recorded as a result.

**Categories.** `pip inspect` is classified as exactly one of:

| Status | Meaning | Cell outcome |
| --- | --- | --- |
| `tool_failed` | the command did not exit 0, timed out, or could not start | `verification tool failed` |
| `output_unparseable` | exit 0, but stdout is empty, not JSON, or not a complete report | `verification tool failed` |
| `match` | parsed; every pin present at its version, nothing extra, pre-existing packages unchanged | this gate passes |
| `mismatch` | parsed; a real difference, listed by name | `installed but not verified` |

In the first two cases the missing / wrong-version / extra lists are `not evaluated`. `pip
check` is classified the same way: exit 0 passes, exit 1 is broken requirements, anything else
is `tool_failed`. Gate G7 is unchanged: only `match`, with every other gate clean, passes.

**Evidence per gate.** Written before the output is parsed, never overwritten (exclusive
creation): `<cell>/gates/<gate>.stdout.txt`, `.stderr.txt`, `.run.json` (command, interpreter,
exit code, seconds), then `.result.json` (the classification).

Regression tests, outside `tests/` and run explicitly:

```text
python -m pytest docs\fie_rebuild_2026\experiments\E51\test_offline_runner.py -p no:cacheprovider
```

## Mode A retry (after attempt 4)

Scope: **mode A only**, one fresh environment, `modeA-pypi-base-r3`. `modeA-pypi-base-r1` was
installed in attempt 4; it is kept and not reused. Mode B and both `[ml]` sets are outside this
retry.

Connected, from the repository root:

```text
python docs\fie_rebuild_2026\experiments\E51\prepare_retry.py
python -I -B docs\fie_rebuild_2026\experiments\E51\offline_runner.py --plan C:\Users\ASUS\fie-e51-scratch\c1_retry_plan.json --selftest
```

`prepare_retry.py` downloads no package. It re-checks the release, pip configuration, proxy
and hosts; re-resolves the base set against the approved pins; re-hashes the mode A wheelhouse
against its ledger; creates the one environment; and takes the positive control through its
launcher and the base interpreter. Evidence: `stageC1/retry1/`.

Offline, owner:

```text
python -I -B docs\fie_rebuild_2026\experiments\E51\offline_runner.py --plan C:\Users\ASUS\fie-e51-scratch\c1_retry_plan.json
```

`--cells a,b` restricts a run to named cells of a plan; without it every cell of the plan runs.
On any stop the runner keeps what it has written, runs the closing controls if the offline
window was opened, pauses for the owner to restore the network, and verifies recovery.

## Cell outcomes: a blocked probe is not a completed measurement

*Added after offline attempt 5 (2026-10-12).* In attempt 5 the mode A install passed every
gate, but `import fie` raised `ModuleNotFoundError` (`starlette`), no scan ran, and the runner
still printed `RESULT: COMPLETED`. The probe recorded the exception class but not the module.

The probe's `import.json` now records, on failure: `exception`, `missing_module`
(`ModuleNotFoundError.name`), a sanitized `exception_message`, `import_completed: false`,
`scan_records: 0` and the import chain (file names relative to the environment). The full
traceback goes to the probe's stderr, which the runner keeps, redacted, in `run.json`.

Each probe process is classified `scan_completed` (exit 0, import completed, at least one
record), `import_blocked`, `no_scan_records` or `probe_failed`. Each cell then gets one outcome:

| Outcome | Meaning |
| --- | --- |
| `SCANS_COMPLETED` | all four runs (S ×2, PC ×2) completed scans |
| `IMPORT_BLOCKED_NO_SCAN_COMPLETED` | `import fie` raised; zero scan records. `blocked.json` holds the error and the unanswered questions |
| `NO_SCAN_COMPLETED` | zero scan records for another reason |
| `PARTIAL_SCANS_COMPLETED` | some runs completed, not all |
| `NOT_MEASURED_GATES_NOT_PASSED` | post-install gates did not pass; no probe was started |

The runner's last line is `COMPLETED` only when every planned cell is `SCANS_COMPLETED`.
Otherwise it is `FINISHED, MEASUREMENT NOT COMPLETE - <cell>: <outcome>; …` or `STOPPED - …`.
An import failure is an observed result of the installed artifact. It is not repaired (no
dependency is added to an environment), it is not a detector pass or failure, and a blocked
mode is never filled in from another mode or cell.

## Mode B base cells

Three fresh environments, `modeB-pypi-base-m1`, `modeB-clean-base-m1`, `modeB-release-base-m1`,
with no pip inside; the repository's pip installs through `--python`. The mode B inspection
command (`python -I -X utf8 -m pip --python <env> inspect`) was checked once, connected, in
`diagnostic_pip_inspect_modeB_command/`; that check is not a gate result.

```text
python docs\fie_rebuild_2026\experiments\E51\prepare_modeb.py                       # connected
python -I -B docs\fie_rebuild_2026\experiments\E51\offline_runner.py --plan C:\Users\ASUS\fie-e51-scratch\c1_modeb_plan.json   # owner, offline
```

`prepare_modeb.py` downloads and builds nothing: it re-hashes the base wheelhouse and the three
`fie` wheels against their ledgers, requires `fie/`, `pyproject.toml` and the manifest to be
unchanged since the build commit, re-resolves the base set against the approved pins, inspects
the wheels statically, creates the environments and takes a positive control through each
launcher. Evidence: `stageC1/modeb1/`. Mode B cells are independent: an install that stops one
cell does not stop the next. Mode B results are about pre-built-local-wheel environments and
are never described as what `pip install fie-sdk` does.

## Modes measured

| Mode | Base cells | Inputs |
| --- | --- | --- |
| S, as-shipped configuration | run, twice | 8 reference prompts + 40 smoke prompts; CLI `detect` and `explain` on two prompts |
| P, manifest-pinned | **not applicable**: no pickle can load without `joblib`, so there is nothing to pin | — |
| PC, comparable | run, twice | 1,848 `lite.std` inputs, each checked against `BL-0002` by SHA-256 |

A base install has no classifier. Nothing measured here is evidence about classifier
performance.

## Files

| File | Runs in | Purpose |
| --- | --- | --- |
| `provision.py`, `sdist_check.py` | repository env, connected | PyPI-only resolve, pin, download, verify (Stage B) |
| `prepare_c1.py` | repository env, connected | Phase 1 |
| `inspect_wheel.py` | repository env | Static wheel inspection; executes nothing from the wheel |
| `export_inputs.py` | repository env | Inputs as plain JSONL, verified against the baseline |
| `egress_selftest.py` | every launcher | TCP control |
| `offline_runner.py` | repository interpreter, offline | Phase 2 |
| `prepare_retry.py` | repository env, connected | One fresh mode A environment for the retry |
| `test_offline_runner.py` | repository env | Regression tests for the gate classification |
| `probe.py`, `cli_guard.py` | each measurement environment | The installed `fie`, under tripwire and recorders |
