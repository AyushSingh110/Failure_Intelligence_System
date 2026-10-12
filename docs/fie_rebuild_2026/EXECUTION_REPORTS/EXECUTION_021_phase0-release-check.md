# EXECUTION_021, Phase 0 — Release check reproduces the base-import failure (WP-021)

| | |
| --- | --- |
| Work package | WP-021, **Phase 0 only** |
| Status | **PHASE 0 COMPLETE. The negative control behaved as expected: the wheel installs, `import fie` fails, the release check exits 1.** No product fix was made |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, merged, published or deployed |
| Date | 2026-10-12 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `becc2c7`, unchanged throughout |
| Contract | [PLAN_021_PACKAGING_BASE_IMPORT.md](../IMPLEMENTATION_PLANS/PLAN_021_PACKAGING_BASE_IMPORT.md), revision 1, §5 and §6 Phase 0, with the owner's Phase 0 authorization |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase0/` |

---

## A. Summary

A new tool, `scripts/release_check.py`, installs a built wheel into a fresh environment and
tests what was installed. Run on a wheel built from a clean export of `becc2c7`, it reports:

| Criterion | Result |
| --- | --- |
| AC-1 wheel installs | PASS |
| AC-2 `pip check` | PASS |
| AC-3 installed set equals the declared base closure | PASS |
| AC-4 installed files equal the wheel | PASS (37 files, 0 differ) |
| **AC-5 `import fie`** | **FAIL: `ModuleNotFoundError`, missing module `starlette`** |
| AC-6 every name in `fie.__all__` resolves | NOT EVALUATED (the import did not complete) |
| AC-7 command line starts | FAIL: the same error, for `--help` and for `detect` |
| AC-8 no optional package imported | NOT EVALUATED (the import did not complete) |
| **Tool exit code** | **1** |

This is acceptance criterion AC-24 of the plan: the gate fails on the known defect, for the
documented reason. It reproduces E51 Stage C1 with an independent tool, in a fifth
environment. Nothing was fixed.

---

## B. State

| | Start | End |
| --- | --- | --- |
| Branch, HEAD | `rebuild/wp-001-eval-harness`, `becc2c7` | unchanged |
| Staged | nothing | nothing |
| Working tree | one untracked file, the WP-021 plan | plus the files in section C |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` | unchanged |
| E51 Stage C1 evidence | checksums verify (7 directories, 159 files) | not touched |
| `fie/`, `pyproject.toml`, `scripts/model_manifest.json` | identical to `2f62ecd`, the E51 build commit | unchanged |
| The WP-021 plan | `e43abbc8…` | unchanged |

---

## C. Files created or changed

| File | Kind |
| --- | --- |
| `scripts/release_check.py` | New. The Phase 0 tool |
| `docs/fie_rebuild_2026/evidence/WP021/phase0/**` | New. Build record, release-check report, per-command output, checksums |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_021_phase0-release-check.md` | New. This report |
| `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Section 14 added |

Nothing else. Not `fie/`, not `pyproject.toml`, not `tests/`, not a workflow, not the manifest.

---

## D. The tool

`scripts/release_check.py --wheel <file> --out <dir> [--wheelhouse <dir>] [--workdir <dir>] [--keep]`

- Takes a wheel **file**. It never imports the working tree.
- Creates a fresh `venv`, installs the wheel with no extra, and runs every check through that
  environment's interpreter with `-I`, from an empty directory outside the repository.
- Proves the installed copy: `fie.__file__` must lie under the environment's prefix, in
  `site-packages`.
- Records each command with its exit code, duration, working directory and the names of the
  environment variables it ran with; stdout and stderr go to separate files, written once.
- Runs the installed package with `FIE_NO_TELEMETRY=1` and `FIE_NO_AUTO_DOWNLOAD=1`, and with
  a temporary home directory. The environment is an explicit list; nothing is inherited that
  could redirect pip or configure `fie`.
- Implements Level 1 (AC-1 to AC-4) and Level 2 (AC-5 to AC-8) only. The report lists what is
  not implemented.
- Statuses: `PASS`, `FAIL`, `NOT_EVALUATED`, `TOOL_FAILED`. Exit 0 only if every implemented
  criterion passed; 1 if any failed; 2 if the check could not run. A failed tool, or a
  criterion that could not be evaluated, is never a pass.
- Adds no package, downloads no model, contacts no service of its own, and has no special case
  for the known defect.

`--wheelhouse` makes pip install dependencies from one directory with `--no-index`. Without it
pip uses its index. Phase 0 used E51's verified mode B wheelhouse, re-hashed against its ledger
first (11 wheels, all equal), so the run needed no network and built nothing from source.

Built-in self-test, `python scripts/release_check.py --selftest`: 16 checks of the
classification functions, including "a failed import fails AC-5 and names the module", "not
evaluated is never a pass" and "a tool failure is not a package finding".

---

## E. The artifact

| | |
| --- | --- |
| Source commit | `becc2c746df205e7cf39e90ed1e4c985ba705355` |
| Source | `git archive` of that commit, extracted to a scratch directory. No working-tree file; the export's `fie/models/` holds only `.gitkeep` |
| Build command | `python -m build --wheel --no-isolation --outdir <scratch>/wheel <scratch>/src`, `SOURCE_DATE_EPOCH=1791785731` (the commit time) |
| Build result | exit 0 |
| Wheel | `fie_sdk-1.18.0-py3-none-any.whl` |
| Size | 151,936 bytes, 42 files, **no model file** |
| SHA-256 | `cfdf18598463f36956e5b765b3209ed05a20f20b7385297a3b3586e9dceaf4e4` |
| Against E51's A-clean (`87104867…`) | All 37 package files byte-identical. The wheel digests differ because the build timestamp differs |

---

## F. Commands and exit codes

| Command | Exit |
| --- | --- |
| `python scripts/release_check.py --selftest` | 0 (16 of 16) |
| build (section E) | 0 |
| `python scripts/release_check.py --wheel <scratch>/wheel/fie_sdk-1.18.0-py3-none-any.whl --out docs/fie_rebuild_2026/evidence/WP021/phase0/release_check --wheelhouse <wheelhouse-B> --workdir <scratch>/work` | **1** |

Inside the release check (environment: Python 3.10.19, bundled pip 23.0.1 and setuptools
79.0.1 before install):

| Step | Command (in the new environment) | Exit |
| --- | --- | --- |
| create | `python -m venv <WORK>/env` | 0 |
| AC-1 | `python -I -m pip install --isolated --disable-pip-version-check --no-cache-dir --no-index --find-links <WHEELHOUSE> <wheel>` | 0 |
| AC-2 | `python -I -m pip check --disable-pip-version-check` | 0, "No broken requirements found." |
| AC-3 | `python -I -X utf8 -m pip inspect --disable-pip-version-check` | 0; 14 distributions: `fie-sdk` 1.18.0, its 11 dependencies, and the pre-existing `pip` and `setuptools` |
| AC-5 | `python -I -c "import fie …"` | **1** |
| AC-7 | `python -I -m fie --help` | **1** |
| AC-7 | `python -I -m fie detect "What is the capital of France?" --output json` | **1** |

The error, as captured (`commands/20_AC5_import.stderr.txt`):

```text
    from fie import integrations
  File "<WORK>\env\lib\site-packages\fie\integrations\__init__.py", line 25, in <module>
    from fie.integrations import openai, anthropic, fastapi
  File "<WORK>\env\lib\site-packages\fie\integrations\fastapi.py", line 28, in <module>
    from starlette.middleware.base import BaseHTTPMiddleware
ModuleNotFoundError: No module named 'starlette'
```

Both command-line invocations end in the same last line.

---

## G. Regression

| Command | Exit | Result |
| --- | --- | --- |
| `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` | 0 | 942 passed, 0 failed |
| `python -m evals baseline` | 0 | Standard counts match; all 11 per-prompt record files byte-identical to `BL-0002_pair-v6.3b_fie-70de7053` |

Both were run after `scripts/release_check.py` was written and the release check had run.

---

## H. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase0/`

| File | Content |
| --- | --- |
| `build.json` | Source commit, build command, wheel identity, comparison with E51's A-clean |
| `release_check/report.json` | Scope, wheel identity, environment, every criterion with its status, every command, verdict, exit code, cleanup |
| `release_check/commands/*.stdout.txt`, `*.stderr.txt` | Output of each command, unedited apart from path redaction |
| `release_check/SHA256SUMS.txt` | Checksums of the above; all verify |

Paths are replaced by `<WORK>`, `<BASE_PYTHON>`, `<WHEEL_DIR>`, `<WHEELHOUSE>`, `<HOME>`. No
file in the evidence directory contains the user name.

Cleanup. The tool removed its own temporary environment (`temporary_environment_removed:
true`). Kept in the scratch directory `C:\Users\ASUS\fie-e51-scratch\wp021\phase0\`: the
source export, the built wheel, and the now empty work directory. No existing scratch content
was deleted or changed.

---

## I. Deviations and open points

| # | Point |
| --- | --- |
| 1 | Dependencies were installed from E51's verified local wheelhouse (`--no-index`), not from PyPI. So `langdetect` came from the locally built wheel and this run says nothing about the source-build path. The tool installs from pip's index when `--wheelhouse` is not given; that path has not been exercised |
| 2 | AC-6 and AC-8 are reported `NOT_EVALUATED`. They can only be evaluated once the import completes, which is Phase 1 |
| 3 | AC-9 is a repository test proposed for Phase 1 and is not part of this tool |
| 4 | The tool has run on Windows with Python 3.10.19 only. It has POSIX paths for CI, untested |
| 5 | The tool's own checks are a built-in `--selftest`, not a file in `tests/`, because adding tests was outside this phase's authorization. Moving them into `tests/` is a Phase 1 or 2 decision |
| 6 | The tool's passing path (exit 0) has been exercised only by the self-test's classification checks, never by a real wheel, because no wheel that imports exists yet |
| 7 | **The new script is git-ignored.** The repository ignores everything in the scripts directory except files it lists by name, and editing the ignore file was outside this phase. So the script does not appear in git status, and committing it needs a forced add (section K). Whether to add it to the ignore file exceptions is an owner decision; CI cannot use a file that is not tracked |

Point 6 matters for Phase 1: the first real pass must be read with the same care as this
first real failure.

---

## J. What this does and does not establish

- **Established:** a wheel built from `becc2c7` installs cleanly and cannot be imported in a
  base environment; the release check detects this and fails.
- **Not established:** anything about detection. No scan ran. A future pass of this check will
  mean that the SDK imports and its command line starts, and nothing more.

## K. Recommended manual commit command

Not executed by the assistant.

```text
git add -f scripts/release_check.py && git add docs/fie_rebuild_2026 && git commit -m "scripts: add installed-wheel release check; reproduces base-import failure (WP-021 phase 0)"
```
