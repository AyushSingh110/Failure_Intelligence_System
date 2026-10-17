# EXECUTION_021, Phase 3, Stage 3.3d — W21-29: source distribution narrowed (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.3d (W21-29a, b, c)**, authorized by the owner |
| Status | **COMPLETE locally.** Every local acceptance criterion met. Hosted CI has not run this change (it is not committed) |
| Commit status | **NOT COMMITTED** by the assistant. Nothing staged, pushed, published or deployed |
| Date | 2026-10-17 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `7c7e3b1` (merged to `main` as `5a6e54b`) |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Plan | `EXECUTION_021_phase3-stage3_3d-sdist-narrowing-plan.md` and `evidence/WP021/phase3/stage3_3d_plan/`: unmodified, checksums re-verified before work began |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3d/` (56 files, `SHA256SUMS.txt` verified) |

Labels: **measured** (here, in the evidence), **inferred**.

---

## A. Decisions implemented (master log §14.11)

| ID | Decision |
| --- | --- |
| D-057 (W21-29a) | Narrow the sdist with an explicit `only-include` allowlist |
| D-058 (W21-29b) | Ship no tests in the sdist |
| D-059 (W21-29c) | Adjust the 3.3b guard: allow exactly `/scripts/model_manifest.json` in the sdist `only-include`, pin the list, keep rejecting model patterns |

W21-29d (a CI-level allowlist in the checker) was not approved and was not done.

## B. Files changed

| File | Change |
| --- | --- |
| `pyproject.toml` | Added the sdist allowlist below, with an explanatory comment; nothing else changed (`pyproject.diff`) |
| `tests/test_build_artifact_allowlist.py` | (1) The guard test rewritten as approved: the exception, a pin, and 8 negative controls. (2) 8 new narrowed-sdist tests. (3) A supporting change: `make_project()` now also copies `LICENSE` into the synthetic project, because the real `pyproject.toml` it copies now names `/LICENSE` |
| `MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | §14.11 |
| This report and `evidence/WP021/phase3/stage3_3d/` | New |

**Unchanged** (observed with git): `hatch_build.py`; `scripts/verify_package_artifacts.py`; both
workflows; `scripts/model_manifest.json`; the loader; models; experiment files; baselines; every
repository file. Nothing was deleted or moved; the demo media and MP3s stay in git.

**The exact sdist allowlist:**

```toml
[tool.hatch.build.targets.sdist]
only-include = [
    "/fie",
    "/hatch_build.py",
    "/scripts/model_manifest.json",
    "/README.md",
    "/LICENSE",
]
```

The resulting sdist also contains:

- the 16 approved artifacts and `fie/models/MANIFEST.json`, added by the hook;
- `pyproject.toml`, `PKG-INFO` and `.gitignore`, which hatchling adds itself.

## C. Before and after (measured)

| | Before (`stage3_3d_plan/measure.json` CURRENT; the same sdist hosted CI built on PR #26) | After (`rehearsal.json`, M1 LF) |
| --- | --- | --- |
| sdist digest | `05394995c946151e…` | `0cc9174f39de780e…` |
| sdist size (compressed) | **67,956,881 B** | **267,222 B** (−99.6%) |
| sdist files | 1,160 | **62**: exactly the allowlist set |
| Wheel from the sdist | `810d5446a8018c20…` | **`810d5446a8018c20…`: byte-identical** |
| Direct wheel | `810d5446a8018c20…` | `810d5446a8018c20…` |
| CRLF variant wheel | `9646a7a39f50…` (Stage 3.3c) | `9646a7a39f50…`: byte-identical |

The narrowed sdist is 211 bytes larger than the plan's probe (267,011 B), because the real
`pyproject.toml` carries the explanatory comment.

## D. Offline M0/M1/M2 rehearsal (measured; `rehearsal.json`, `reports/`, `logs/`)

The narrowed `pyproject.toml` was overlaid on a `git archive` export of `7c7e3b1`, in LF and CRLF
variants. The script asserts that the overlay is the narrowed configuration; the overlay digest
is in `rehearsal.json`. The 25 inventory artifacts were verified before copying, and a stale
`MANIFEST.json` was planted in every populated tree. Builds used `--no-isolation` under the
socket-refusing guard. **Zero network attempts.**

| Setup | sdist | Exactly the allowlist | < 2 MB | Gate on sdist | Wheel from sdist | Gate on sdist + wheel | Direct wheel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 LF ×2 | 267,222 B / 62 | yes | yes | PASS | `810d5446…` | PASS | `810d5446…`, PASS |
| M1 CRLF | 268,123 B / 62 | yes | yes | PASS | `9646a7a3…` | PASS | `9646a7a3…`, PASS |
| M2 LF (+14 experiment files, 5 by-products) | identical to M1 LF | yes | yes | PASS | `810d5446…` | PASS | PASS |
| M2 CRLF | identical to M1 CRLF | yes | yes | PASS | `9646a7a3…` | PASS | PASS |
| M0 (no models) | **refused**: `16 of 16 approved model artifacts cannot be packaged` | — | — | — | not applicable | — | **refused** |

The pre-3.3b archives from Stage 3.3a are still refused by the checker (exit 1).

**Process note:** a first rehearsal attempt silently ran **without** the overlay. A script edit
failed to apply after the command was moved to the background, so it built HEAD's old
configuration. It was detected because the new membership fields were empty. Its output was set
aside, outside the repository, and is not used anywhere. The run reported here is the second,
with the overlay asserted in the script.

## E. Tests (measured)

**`tests/test_build_artifact_allowlist.py`: 48 passed** (32 before; 16 new).

| Requirement | Tests |
| --- | --- |
| Guard: only the inventory is allowed, the list is pinned | `test_no_pattern_selects_model_files` |
| Guard still rejects model patterns | `test_guard_still_rejects_model_patterns[8]`: `/fie/models`, `/fie/models/meta_clf.pkl` and `/fie/models/MANIFEST.json` added to the sdist list; `/data/…json` in the sdist list; the inventory in the **wheel's** `only-include` (the exception is sdist-only); `artifacts` with `*.pkl`; sdist `include` with `*.json`; a `force-include` of a pickle |
| 1. Exact sdist membership, with demo media, third-party MP3s, data, notebooks, experiment evidence, tests, server, frontend and deployment files planted | `test_narrowed_sdist_holds_exactly_the_allowlist`, `test_repository_only_files_do_not_leak_into_the_sdist` |
| 2. Wheel from the sdist passes the checker | `test_wheel_from_the_narrowed_sdist_passes_the_archive_checker` |
| 3, 4. Missing and tampered models fail closed | `test_narrowed_sdist_build_still_fails_closed[missing, tampered]`; the existing `test_build_fails_closed[…]` (4) now also build under the narrowed configuration |
| 5. Experiment files and evidence excluded | `test_experiment_files_and_evidence_do_not_leak_into_the_sdist` |
| 6. Size ceiling (2 MB) | `test_narrowed_sdist_is_below_the_size_ceiling`: an incompressible 5 MB video is planted and the sdist stays below 2 MB |
| Models and build inputs present | `test_narrowed_sdist_carries_the_approved_models_and_build_inputs` |

The real-build tests use the real `pyproject.toml`, `.gitignore`, `hatch_build.py` and integrity
module around a synthetic package, so the narrowed configuration itself is what they test.

**Full suite:** **1,516 passed, 0 failed, 2 skipped** (the two Windows platform cases). Before
this stage: 1,500 passed.

**Lint:** CI's ruff rule set passes on the test file. `git diff --check` is clean.

## F. Hosted CI

Inferred:

- The CI `package` job runs `test_build_artifact_allowlist.py` with `build` and hatchling
  installed, and fails on any skip, so the 16 new tests will run there.
- Its gate will judge the narrowed sdist and the wheel built from it; that wheel should equal
  `810d5446…` on the Linux runner only if the tools match. Archive equality across platforms was
  never claimed.
- The test matrix, which installs no build tools, will skip 22 of these tests instead of 14: the
  8 new narrowed-sdist tests build archives. The 9 guard tests need no build tools and run
  everywhere.

Not yet observed. The first push or PR will show it.

## G. Acceptance

| Criterion | Status |
| --- | --- |
| sdist contains exactly the allowlist-derived set | Met (tests; M1/M2 LF and CRLF) |
| sdist below 2 MB | Met (267,222 B; test with a 5 MB planted file) |
| Wheel from the sdist passes the archive checker | Met |
| Wheel byte-identical to the established baseline | Met (`810d5446…` LF, `9646a7a3…` CRLF) |
| Missing or tampered models fail closed | Met |
| Experiment evidence and prohibited files excluded | Met |
| Guard allows only the inventory and still rejects model patterns | Met (pin plus 8 negative controls) |
| Hosted-CI assumptions and unrelated tests not weakened | Met: no workflow or checker change; full suite green |
| Protected files unchanged | Met |

## H. Final working tree

| Item | Status |
| --- | --- |
| Modified | `pyproject.toml`, `tests/test_build_artifact_allowlist.py`, `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` |
| Untracked | this report, `evidence/WP021/phase3/stage3_3d/`, and the plan report with its evidence (from the planning run, uncommitted) |
| Ignored files added | None: logs are stored as `.txt` |
| Commits, pushes, publishes, deployments | None |

The packaging-refinement workstream stops here. W21-19, W21-32, W21-20-R, W21-14 and Phase 4
were not started.

## I. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3d/`; verify with `sha256sum -c SHA256SUMS.txt`.

| File | Content |
| --- | --- |
| `summary.json` | The allowlist, before and after sizes, counts and digests, wheel comparison, full-suite counts, the new and changed test names |
| `rehearsal.json` | Every rehearsal build and gate: commands, exit codes, `dist/` digests, the exact sdist member list and allowlist verdict per setup, the overlay digest |
| `reports/*.json` | Archive-checker reports with every member and the model-area digests |
| `logs/*.txt` | All build and checker logs, local paths scrubbed; network log (empty) |
| `pyproject.diff` | The exact configuration change |
| `scripts/rehearse.py` | The rehearsal script |
