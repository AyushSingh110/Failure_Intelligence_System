# EXECUTION_021, Phase 3, Stage 3.3c — CI packaging gate, archive validation, publish protection (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.3c** |
| Status | **Implemented and verified locally.** All local acceptance criteria met. **Hosted CI has not run these changes** (section H) |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, published, tagged or deployed |
| Date | 2026-10-17 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `fe8f6c9` (Stage 3.3b, committed by the owner) |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3c/` (63 files, `SHA256SUMS.txt` verified) |

Labels: **observed** (measured here, in the evidence), **inferred**, **unverified**.

---

## A. Preflight (observed)

| Check | Result |
| --- | --- |
| Tree | Clean at `fe8f6c9`; nothing staged |
| Stage 3.3b committed | Yes: `fe8f6c9` contains `hatch_build.py`, the `pyproject.toml` changes and `tests/test_build_artifact_allowlist.py` |
| CI for `fe8f6c9` | **None found.** The latest runs are for `49b7a1f` (PR, success) and `e8ac325` (main push, success). Stage 3.3b has therefore not run on a hosted runner either |
| Inventory | Unchanged since Stage 3.1; 16 `in_wheel: true` entries; derived manifest `fbab28ab…` (5,112 B); local artifacts verify 25/25 |
| Why 14 packaging tests skip in CI | `ci.yml`'s `test` jobs install `requirements.txt` and pytest only. `build` and hatchling are installed only in the `package` job, which ran no tests. The 13 real-build tests and the hatchling class check call `pytest.skip` when either is missing |

## B. Decisions recorded

These are implementation decisions within the Stage 3.3c authorization, recorded in master log
§14.10.

| ID | Decision |
| --- | --- |
| D-053 | One archive gate, `scripts/verify_package_artifacts.py`, used by both CI and publication; its expectations derived only from the inventory |
| D-054 | The CI `package` job mirrors publication: strict download, sdist, gate, wheel from that sdist, gate on both, release check on that wheel |
| D-055 | The packaging tests run in the `package` job with `build` and `hatchling==1.29.0`; any skip fails the job |
| D-056 | The publish workflow builds and gates in the order of section E; publication is reachable only after both archives pass |

## C. Files changed

| File | Change | Why |
| --- | --- | --- |
| `scripts/verify_package_artifacts.py` (new, **git-ignored path**: commit with `git add -f`) | The archive checker | Section D |
| `tests/test_package_artifact_gate.py` (new) | 47 tests | Section F |
| `.github/workflows/ci.yml` | `package` job only | Section E |
| `.github/workflows/publish-pypi.yml` | Build-and-verify steps only | Section E |
| `MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | §14.10 | — |
| This report and `evidence/WP021/phase3/stage3_3c/` | New | — |

Unchanged (observed with git): `hatch_build.py`; `pyproject.toml`; `fie/`, including
`_model_integrity.py`, the loader and selection; `scripts/model_manifest.json`;
`scripts/download_models.py`; `scripts/release_check.py`; models; experiment files; baselines;
`Dockerfile`; all other CI jobs.

## D. The archive checker

`python scripts/verify_package_artifacts.py --sdist S.tar.gz --wheel W.whl [--report out.json]`

**Expectations:** `validate_production(scripts/model_manifest.json)` gives the `in_wheel: true`
entries (16 today, with size and SHA-256), and `derive_package_manifest()` gives the exact
`MANIFEST.json` bytes. `fie/_model_integrity.py` is loaded by file path; neither `fie` nor any ML
package is imported (a test checks this). There is no second list.

**Reading:** member names and bytes only (bytes are hashed). Nothing is extracted to disk,
imported or deserialized.

- **Wheel:** every file member.
- **sdist:** exactly one top-level directory is required and stripped. Files outside it, links
  and special files are refused.
- **Names:** every member name must be a plain relative path. Absolute paths, `..`, `.`, empty
  segments, backslashes and drive letters are refused. Duplicate names are refused, and so are
  names that differ only by case.

**Checks per archive:** the model area is every member whose case-folded path starts with
`fie/models/`.

1. All 16 artifacts are present.
2. Nothing else is present except `MANIFEST.json`: no legacy file, `.gitkeep`, experiment file,
   side-car, nested file, encoder by-product or case variant.
3. Each artifact has the inventory's size and SHA-256.
4. `MANIFEST.json` is present and byte-identical to the derivation.
5. For an sdist: `pyproject.toml`, `hatch_build.py`, `fie/_model_integrity.py` and
   `scripts/model_manifest.json` are present, and that inventory derives the same manifest.

The rest of the sdist is not judged; narrowing it is W21-29.

**Exit codes:** 0 pass; 1 any failure (each listed by archive file name and member path); 2 an
unusable inventory or argument. The optional JSON report holds every member name, the model-area
digests and the verdict.

## E. Workflow changes (reviewed locally; not yet run on a hosted runner)

**`ci.yml`, `package` job** (other jobs untouched):

1. Install `build`, `hatchling==1.29.0`, pytest, and the base dependencies read from
   `pyproject.toml` (the tests import `fie`).
2. Confirm that `build` and hatchling import, and that hatchling is exactly 1.29.0.
3. `download_models.py --strict` (was best-effort).
4. Run `test_build_artifact_allowlist.py` and `test_package_artifact_gate.py` with `--junitxml`.
   A check then fails the step if any test was skipped, failed or errored.
5. `python -m build --sdist`, then the gate on the sdist.
6. Unpack that sdist and build the wheel from it.
7. The gate on the sdist and wheel together; SHA-256 of both recorded.
8. The existing release check (T-base, T-fastapi) on that wheel.
9. Upload `package-gate` (JUnit XML, both gate reports with named members, digests) and the
   existing `release-check`, both `if: always()`.

The count-only "Check wheel contents" step is removed: the gate supersedes it.

**`publish-pypi.yml`:** trigger, permissions, Python, the strict download, `twine check` and the
publish step are unchanged. `python -m build` and "Verify wheel bundles models" are replaced by:
build the sdist; gate it; unpack it and build the wheel from it; check that `dist/` holds exactly
those two files; gate both. Every step must succeed for the job to reach `twine check` and the
publish step. The published wheel is the one built from the verified sdist.

## F. Tests (observed)

**`tests/test_package_artifact_gate.py`: 47 passed.** All archives are synthetic and written by
the test. The inventory comes from the same rules helper as the 3.3b tests.

| Requirement | Tests |
| --- | --- |
| Valid wheel and sdist pass | `test_valid_wheel_and_sdist_pass` |
| Unrelated files outside `fie/models/` not judged | `test_unrelated_files_outside_the_model_area_are_not_judged` |
| Missing artifact / missing manifest / legacy / experiment classifier / experiment side-car / nested / encoder by-product / `.gitkeep` / wrong bytes / wrong size / stale manifest / case variant | `test_negative_controls_fail[...]`, 12 cases × wheel and sdist = 24 |
| Duplicates | `test_duplicate_wheel_member_fails`, `test_duplicate_sdist_member_fails`, `test_members_differing_only_by_case_fail` |
| Traversal and non-canonical names | `test_traversal_and_non_canonical_paths_fail[5]` |
| Links | `test_links_in_the_model_area_fail` |
| sdist structure and inputs | `test_sdist_must_have_one_top_level_directory`, `test_sdist_missing_a_build_input_fails[4]`, `test_sdist_with_a_different_inventory_fails` |
| Expectations from the inventory | `test_expectations_are_the_in_wheel_entries_and_the_derived_manifest`, `test_changing_the_inventory_changes_what_is_expected`, `test_real_inventory_expects_sixteen_artifacts_and_the_established_manifest` |
| CLI | `test_cli_exit_codes_and_report`, `test_cli_unusable_inventory_or_arguments_exit_2`, `test_checker_imports_neither_fie_nor_ml_packages` |

One platform note. On Windows, Python's `zipfile` converts a backslash in a member name to `/`
when writing and reading. A backslash name therefore arrives as `fie/models/win.pkl` and is
refused as an unexpected member; on Linux it is refused as a non-canonical path. The test asserts
the right message per platform.

**The CI packaging step, reproduced locally:**

- Same pytest command and the same JUnit check: **79 collected, 0 skipped, 0 failed**; the check
  passes.
- **Negative control:** with `build` hidden (`sys.modules['build'] = None`, as on a runner without
  it), pytest exits **0** with **13 skipped**. That is the silent green this stage removes. The
  check fails with `packaging tests were skipped; every one must run in this job`.

**Full suite:** 1500 passed, 0 failed, 2 skipped (Windows platform cases). Before this stage:
1453 passed.

**Lint:** CI's ruff rule set passes on both new files. `git diff --check` is clean.

## G. Offline rehearsal of the CI and publish sequence (observed, `rehearsal.json`, `reports/`)

Temporary `git archive` exports of `fe8f6c9`, with LF and CRLF variants. Inventory artifacts were
verified 25/25 before copying, and a stale `MANIFEST.json` was planted in every populated tree.
Builds used `python -m build --no-isolation` under the socket-refusing audit hook. **Zero network
attempts.**

| Setup | sdist build | Gate on sdist | Wheel from sdist | Gate on sdist + wheel | Direct wheel | Gate on direct wheel |
| --- | --- | --- | --- | --- | --- | --- |
| M1 LF (twice) | ok | **pass** | ok | **pass** | ok | **pass** |
| M1 CRLF | ok | pass | ok | pass | ok | pass |
| M2 LF (+14 experiment files, 5 by-products) | ok | pass | ok | pass | ok | pass |
| M2 CRLF | ok | pass | ok | pass | ok | pass |
| M0 (no models) | **refused by the hook** (`16 of 16 … missing`) | — | not applicable | — | **refused** | — |

**Equality:**

| Variant | sdist | Wheel |
| --- | --- | --- |
| LF | `7fb4514baeb2…`, 1,131 members | `810d5446a801…`, 60 members |
| CRLF | `0506578d0128…` | `9646a7a39f50…` |

- M1 equals M2, and repeats are equal.
- The wheel from the sdist is **byte-identical** to the direct wheel in every setup.
- The model area is identical across sdist, both wheels, both scenarios and both line-ending
  variants. These are exactly the 16 approved artifacts plus `MANIFEST.json` (`fbab28ab…`); the
  per-member digests are in `reports/*.json`.
- The LF wheel digest equals the Stage 3.3b acceptance wheel. The sdist differs from 3.3b's
  because HEAD now includes the committed 3.3b files.
- No cross-platform claim is made.

**Negative controls on real archives from before Stage 3.3b** (Stage 3.3a outputs): every archive
was refused, exit 1.

| Archive | Failures |
| --- | --- |
| M1 wheel | 3: `.gitkeep`, both legacy pickles |
| M1 sdist | 4: those three, plus `missing build input: hatch_build.py` |
| M2 wheel | 17: plus the 14 experiment files |
| M2 sdist | 18 |

## H. Acceptance

| Criterion | Status |
| --- | --- |
| Expectations derived from the inventory | Met (tests in section F) |
| sdist and wheel judged by actual members | Met |
| Exactly 16 artifacts + derived manifest; sizes and hashes match | Met (rehearsal, reports) |
| Rejects missing, extra, duplicate and tampered members | Met (47 tests; pre-3.3b controls) |
| Packaging tests run without dependency skips in CI | **Implemented; verified locally** (simulation, plus the hidden-`build` control). **Unverified on a hosted runner** |
| Regular CI validates the wheel built from the sdist | **Implemented; unverified on a hosted runner** |
| Publication unreachable until both archives pass | **Implemented** (step order, default fail-fast); the publish workflow was **not run** (no release) |
| No source-tree rule, sdist narrowing, line-ending policy, research migration or published-sdist download | Met |
| Baseline, models, inventory, loader, selection unchanged | Met |
| No commit, push, merge, publication, deployment or release | Met |

**Hosted CI status: not run.** The first push or PR containing this change will show it. The
`package` job log should show `packaging tests: 79 collected, 0 skipped, 0 failed`, two
`verify_package_artifacts: PASSED` lines, and a `package-gate` artifact.

## I. Risks and deferred work

- **First hosted run.** The workflow steps were checked by YAML parse, by review, and by running
  the same commands locally on Windows bash. The runner is Ubuntu: `tar`, globbing and
  `tomllib` (Python 3.11) are standard there, but this is inferred, not observed.
- **CI cost.** The `package` job now installs the base dependencies and runs 79 tests, including
  13 real builds of a small synthetic project.
- **Isolated builds** in CI and publication fetch `hatchling==1.29.0` from PyPI, as before (pinned
  in 3.3b).
- **Deferred:**
  - W21-19: line endings;
  - W21-29: the sdist is still about 67 MB and includes third-party music files, which should be
    decided before the next publication;
  - W21-32: the source-tree rule;
  - W21-20-R: relocating the experiment files;
  - W21-14: inspecting the published sdist;
  - Phase 4: version and release.

## J. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3c/`; verify with `sha256sum -c SHA256SUMS.txt`.

| File | Content |
| --- | --- |
| `rehearsal.json` | Every rehearsal build and gate run: commands, exit codes, `dist/` digests; the pre-3.3b controls |
| `reports/*.json` | Gate reports: expected set, every member name, model-area sizes and digests, failures |
| `logs/*.log` | All build and gate logs, local paths scrubbed; network log (empty) |
| `ci_simulation/` | JUnit XML and logs for the CI packaging step (79/0/0) and the hidden-`build` control (13 skipped) |
| `workflow_diffs/ci.diff`, `publish-pypi.diff` | The exact workflow changes |
| `test_results.json` | Full-suite counts and skips; the gate and allowlist test names |
| `repository_state.json` | HEAD, changed files with digests, protected-path status, `CANONICAL` |
| `scripts/rehearse.py`, `scripts/package_evidence.py` | The scripts that produced this |

---

## K. Correction (2026-10-17, after merge of PR #25)

The Stage 3.3c commit `8bc61e7` was merged without `scripts/verify_package_artifacts.py`: the
file is under the git-ignored `scripts/` and was not force-added. CI on PR #25 (run
`39014383051`) and on the merge commit `ed3921f` (run `39014885011`) therefore failed:
`tests/test_package_artifact_gate.py` could not load the checker (`FileNotFoundError`). The test
matrix failed or was cancelled; `package`, `health-check` and `deploy-space` were skipped, so
nothing was deployed or published.

Two evidence defects were found by the pre-commit audit, verifying each `SHA256SUMS.txt` in a
clean LF export of HEAD:

1. The `*.log` evidence files of Stages 3.3a, 3.3b and 3.3c (63, 40 and 37) were excluded by
   `.gitignore`'s `*.log` rule, although each `SHA256SUMS.txt` lists them.
2. The 17 `stage3_3c/reports/*.json` files had been written with CRLF line endings by the
   checker's `--report` option on Windows, and Git stored them as LF, so their recorded digests
   did not match a clean checkout.

Fix, in one follow-up commit:

- the checker is force-added;
- the ignored evidence logs are force-added;
- `--report` now writes with `newline="\n"`;
- the 17 reports were converted from CRLF to LF, content otherwise unchanged;
- `stage3_3c/SHA256SUMS.txt` was regenerated over those LF bytes.

No other evidence file changed. The sections above describe the gate as implemented and are
unchanged.
