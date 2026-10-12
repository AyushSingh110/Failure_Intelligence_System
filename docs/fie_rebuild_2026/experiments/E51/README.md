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
| `probe.py`, `cli_guard.py` | each measurement environment | The installed `fie`, under tripwire and recorders |
