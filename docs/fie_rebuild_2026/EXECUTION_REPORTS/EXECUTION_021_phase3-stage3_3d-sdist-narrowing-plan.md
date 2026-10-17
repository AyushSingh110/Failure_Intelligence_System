# EXECUTION_021, Phase 3, Stage 3.3d — W21-29 source-distribution narrowing: audit and plan (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **W21-29 (planned as Stage 3.3d): planning only** |
| Status | **PLAN. No implementation change was made.** Narrowing is shown feasible by an offline probe; owner decisions needed (section 8) |
| Commit status | Not committed by the assistant. Outputs: this report and `evidence/WP021/phase3/stage3_3d_plan/` |
| Date | 2026-10-17 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `7c7e3b1`, merged to `main` as `5a6e54b` |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3d_plan/` (16 files, `SHA256SUMS.txt` verified) |

Labels: **measured** (this run, in the evidence), **inferred**, **recommendation**.

The master decision log was not edited, because only this report and its evidence were
authorized. Section 8 lists the decisions to record when the owner approves.

---

## 1. Verified state

| Check | Result |
| --- | --- |
| Tree | Clean at `7c7e3b1`; nothing staged |
| Hosted CI | PR #26 run `39015387722` (`7c7e3b1`): every job success. `package`: `packaging tests: 79 collected, 0 skipped, 0 failed`; gate PASSED on the sdist and on the sdist plus the wheel from it (17 members under `fie/models/` each); T-base and T-fastapi PASSED. `main` push run `39015659895` (`5a6e54b`): every job success, including `deploy-space` |
| Packaging configuration | `requires = ["hatchling==1.29.0"]`; `[tool.hatch.build] exclude = ["/fie/models/**"]`; custom hook `hatch_build.py`; `[tool.hatch.build.targets.wheel] packages = ["fie"]`; **no sdist target configuration**, so the sdist takes every non-ignored file in the repository |
| Publish workflow | Strict download → sdist → gate → wheel from that sdist → gate on both → `dist/` must hold exactly two files → `twine check` → publish |
| Checker | `scripts/verify_package_artifacts.py` judges `fie/models/` and the sdist build inputs only; by design it does not judge the rest of the sdist |
| Local `dist/` | Only the stale 2026-08-10 archives; **not used** here because they predate the current configuration. A fresh offline sdist of HEAD was built instead (section 2) |

## 2. Measured: today's sdist (offline build of HEAD; `measure.json`)

**Method:**

- A `git archive` export of `7c7e3b1` (LF), with the 25 inventory artifacts copied in after a
  size and SHA-256 check (never deserialized).
- Built with `python -m build --no-isolation` (build 1.4.0, hatchling 1.29.0) under the
  socket-refusing audit hook. **Zero network attempts.**
- The wheel was built from the unpacked sdist, and the archive checker judged both.

**Size:** `fie_sdk-1.18.0.tar.gz` is **67.96 MB compressed, 100.41 MB uncompressed, 1,160
files**. The wheel built from it has 60 files, `810d5446a801…`, the same digest as the Stage
3.3b/3.3c acceptance wheel.

| Category | Files | Uncompressed | Needed to build the wheel? |
| --- | ---: | ---: | --- |
| Demo media bundles `brag-output*/` (two MP4s, two **third-party MP3s** credited to ende.app, composition assets) | 74 | 41.64 MB | No |
| Media elsewhere (`docs/media/*.mp4`, which duplicate the two brag MP4s byte for byte in size; images) | 10 | 20.26 MB | No |
| Rebuild reports and evidence (`docs/fie_rebuild_2026/`) | 580 | 12.52 MB | No |
| Evaluation baselines (`evals/baselines/`) | 46 | 11.59 MB | No |
| Datasets (`data/`) | 103 | 7.01 MB | No |
| Notebooks | 6 | 3.33 MB | No |
| Server, frontend, deployment (`app/`, `engine/`, `storage/`, `Frontend/`, `deploy/`, `telemetry-server/`, `Dockerfile`, …) | 158 | 1.61 MB | No |
| Tests | 48 | 0.67 MB | No (section 3.2) |
| Package source `fie/` | 38 | 0.47 MB | **Yes** |
| Documentation (other `docs/`, root `*.md` except README) | 21 | 0.46 MB | No |
| Model artifacts `fie/models/` (16 approved + derived `MANIFEST.json`) | 17 | 0.31 MB | **Yes** (D-049) |
| Evaluation harness code and registry (`evals/`) | 24 | 0.25 MB | No |
| Scripts (research and tooling) | 15 | 0.18 MB | Only `scripts/model_manifest.json` |
| Build inputs (`pyproject.toml`, `hatch_build.py`, `scripts/model_manifest.json`, `PKG-INFO`, `.gitignore`) | 5 | 0.05 MB | **Yes** |
| README, LICENSE | 2 | 0.02 MB | **Yes**: the README is the project's long description; the LICENSE becomes `dist-info/licenses/LICENSE` |
| CI workflows, other | 13 | 0.04 MB | No |

**Largest files:**

| File | Size |
| --- | ---: |
| `brag-output-2026-09-29-113038/brag.mp4` | 15.55 MB |
| `docs/media/fie-explainer.mp4` | 15.55 MB |
| `brag-output/brag.mp4` | 4.29 MB |
| `docs/media/fie-overview.mp4` | 4.29 MB |
| `…/happy-beats-business-moves-vol-1-by-ende-dot-app.mp3` | 3.94 MB |
| `data/meta_clf_trainset.jsonl` | 3.35 MB |
| `…/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` | 2.82 MB |
| `evals/baselines/BL-000{1,2}…/records/risk.framing.jsonl` | 2.34 MB each |

The full per-member list, with sizes and digests, is in `measure.json`.

**Not determined:** whether any downstream consumer relies on the sdist for these files. No
in-repository workflow or the Dockerfile builds from the sdist (inferred from the workflow and
Dockerfile review in Stages 3.3a to 3.3c). The published 1.18.0 sdist is W21-14.

## 3. What the sdist must contain, and the probes

### 3.1 Minimal inclusion (measured)

Probe **NARROW_MIN**, applied to a throw-away copy of `pyproject.toml` only:

```toml
[tool.hatch.build.targets.sdist]
only-include = ["/fie", "/hatch_build.py", "/scripts/model_manifest.json", "/README.md", "/LICENSE"]
```

| Measure | Current | NARROW_MIN |
| --- | --- | --- |
| sdist | 67.96 MB / 1,160 files | **0.27 MB / 62 files** (99.6% smaller) |
| sdist contents | — | `fie/` source, the 16 approved artifacts plus `MANIFEST.json` (added by the hook), `hatch_build.py`, `pyproject.toml`, `scripts/model_manifest.json`, README, LICENSE, `PKG-INFO`, `.gitignore` (hatch adds the last two itself) |
| Wheel built from the sdist | `810d5446…`, 60 files | **`810d5446…`: byte-identical** |
| Archive checker (sdist + wheel from sdist) | PASS | **PASS**: all build inputs present, exactly 17 model members |

`only-include` does not interfere with the hook: the hook's force-includes still apply, and the
`/fie/models/**` exclusion still applies. The packed `MANIFEST.json` is still derived at build
time.

### 3.2 Shipping tests (measured)

Probe **NARROW_TESTS**: NARROW_MIN plus the four packaging and integrity test files and
`scripts/verify_package_artifacts.py`. The sdist is 0.30 MB; the wheel is again byte-identical
and the checker passes.

`only-include` packs an explicitly named file even if `.gitignore` excludes it, as the checker
is. Run inside the unpacked sdist, `pytest tests` gives **383 passed, 4 failed, 24 errors,
3 skipped** (`tests_in_narrow_sdist.json`):

- 24 setup errors and 2 failures need `scripts/download_models.py`, and one needs the `evals`
  package. Neither is in the sdist.
- **`test_no_pattern_selects_model_files` fails** (Stage 3.3b guard). It forbids any
  include-type key in `tool.hatch` from naming `.json`, `.pkl` or `models`, and `only-include`
  must name `/scripts/model_manifest.json`. **This failure would occur in the repository and in
  CI as well, not only in the sdist.** Any W21-29 implementation must adjust that guard
  (section 5).

A test subset is therefore not self-consistent without also shipping the download script and the
harness. Shipping a coherent suite would pull back much of what is being removed. Recommendation:
ship **no tests** (option A); option B is recorded in section 8.

## 4. Proposed policy (recommendation)

Policy: **an explicit allowlist** (`only-include`) for the sdist, not a growing exclude list. A
new top-level file then stays out until someone deliberately adds it, which suits a security
package.

| Category | Policy | Compatibility risk |
| --- | --- | --- |
| Demo videos and music (`brag-output*/`, `docs/media/`) | Exclude | None for building or installing. **Licensing:** the MP3s are third-party music. The narrowed sdist stops redistributing them on PyPI; whether they belong in the public git repository is a separate owner question, not part of W21-29 |
| Research datasets, notebooks, research scripts, `evaluation/` | Exclude | Researchers use the repository, not the sdist. No in-repo consumer of the sdist found |
| Baselines and evaluation outputs (`evals/`) | Exclude | The baseline gate runs from the repository. `CANONICAL` and `BL-0002` are untouched: only distribution contents change |
| Historical reports and evidence (`docs/fie_rebuild_2026/`) | Exclude | Evidence stays in git, where its checksums are verified |
| Tests and development tooling | Exclude (option A) | Downstream packagers that run tests from an sdist would find none. Today's sdist tests are not runnable offline either (they need the download script, the harness and the network). To be documented |
| Server, frontend, deployment, CI workflows | Exclude | The SDK wheel never contained them. The server is deployed from the repository |
| Other documentation | Exclude | README links to `docs/…` already do not resolve on PyPI; inside the sdist they will also dangle (minor) |
| `fie/`, `hatch_build.py`, `pyproject.toml`, `scripts/model_manifest.json`, README, LICENSE, the 16 artifacts plus derived `MANIFEST.json` | **Include** | Required: section 3.1 |

No tracked file is deleted or moved. Only what the sdist contains changes.

## 5. Implementation stages (proposed; each separately authorized)

### 5a — Narrow the sdist

| | |
| --- | --- |
| Allowed files | `pyproject.toml` (add `[tool.hatch.build.targets.sdist] only-include`, exactly the five entries of §3.1); `tests/test_build_artifact_allowlist.py` (adjust the guard, below); a new or extended test module for the sdist member policy; report, evidence, master-log entry |
| Guard adjustment | `test_no_pattern_selects_model_files` keeps failing on any include-type key that names `fie/models`, `.pkl` or a model JSON. It permits exactly `/scripts/model_manifest.json` in `targets.sdist.only-include`, and a separate assertion pins that list to the approved five entries |
| New tests | (1) The sdist member set equals: the `fie/` package files, the 17 model members, `hatch_build.py`, `pyproject.toml`, `scripts/model_manifest.json`, `README.md`, `LICENSE`, `PKG-INFO`, `.gitignore`. Nothing else, whatever exists in the source tree (built from a synthetic project with demo media, data, notebooks, evidence, tests and server files planted). (2) The wheel from that sdist passes the checker and equals the direct wheel's member set and digests. (3) M0, missing and tampered artifacts still fail closed. (4) Experiment and unlisted model files still do not ship. (5) An sdist size ceiling as a regression guard (for example under 2 MB) |
| Not changed | `hatch_build.py`, the checker, CI and publish workflows, the inventory, the loader, models, baselines |
| Acceptance | Section 6 |

### 5b — Optional checker extension (only if the owner wants a CI-level guard beyond tests)

Teach `verify_package_artifacts.py` an optional `--sdist-allowlist` mode that rejects top-level
entries outside the approved set, and call it in the `package` job and the publish workflow.
This needs explicit authorization, because it changes the checker and the workflows.
Recommendation: not needed if the 5a tests run in the `package` job, which they will through
`test_build_artifact_allowlist.py`.

**Dependencies on other open items:**

| Item | Relationship |
| --- | --- |
| W21-19 (line endings) | Independent. A smaller sdist makes the CRLF-versus-LF archive difference smaller, but does not settle it |
| W21-32 (source-tree rule) | Independent. `only-include` removes the incidental route for non-package files, but files under `fie/` are still selected by `packages = ["fie"]`, and the model area is still governed by the hook |
| W21-20-R (experiment relocation) | Independent. Experiment files were already kept out of archives by 3.3b |
| W21-14 (published sdist) | Useful context for release notes; not needed to implement 5a |
| Phase 4 (version and release) | **5a should land before the next release**, so the next published sdist is the narrow one and does not redistribute the third-party MP3s |

## 6. Acceptance tests and success criteria (for 5a)

| # | Criterion | How |
| --- | --- | --- |
| A1 | The sdist contains exactly the approved file set (§5a test 1) | New test plus an offline M1/M2 rebuild of the real tree |
| A2 | The wheel built from the narrow sdist passes `verify_package_artifacts.py`; exactly 16 artifacts plus derived `MANIFEST.json` | Checker on sdist plus wheel, in tests and in CI `package` |
| A3 | That wheel is byte-identical to the wheel built from today's configuration, on the same platform and tools | Measured feasible: `810d5446…` = `810d5446…` |
| A4 | M0, a missing artifact and a tampered artifact still fail the build | Existing 3.3b tests plus the offline rehearsal |
| A5 | Experiment, legacy, nested unlisted and `.gitkeep` files do not ship | Existing tests plus an M2 rebuild |
| A6 | The publish workflow still publishes the wheel built from the validated sdist | Workflow unchanged; CI `package` mirrors it |
| A7 | No weakening: `test_no_pattern_selects_model_files` still fails on any model artifact pattern (negative controls in the test) | New guard cases |
| A8 | Size | sdist **≤ 2 MB** (measured 0.27 MB, against 67.96 MB today: about 99.6% smaller) |
| A9 | Full suite passes; CI `package` shows `0 skipped`; `BL-0002` per-prompt records byte-identical | Standard gates |

## 7. Risks and open questions

1. Unknown downstream reliance on the sdist's extra files. None was found in the repository; it
   would surface as a request after release.
2. The guard adjustment must stay as strict as today for model files (A7).
3. Hatchling behaviour: `only-include` was measured on 1.29.0 only. The pin (D-052) keeps this
   stable.
4. Whether the demo media and third-party MP3s should stay in the public git repository at all.
   This is outside W21-29 and is an owner decision.

## 8. Decisions needed

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| W21-29a | Approve narrowing by an explicit `only-include` allowlist | Allowlist / exclude list / leave as is | **Allowlist**, exactly §3.1 |
| W21-29b | Tests in the sdist | (A) none; (B) a curated, self-consistent subset (needs the download script and `evals`, so larger) | **(A)** |
| W21-29c | Guard adjustment in the 3.3b test (§5a) | Approve / not | **Approve**: required by any allowlist that names the inventory |
| W21-29d | CI-level sdist allowlist in the checker (§5b) | Now / later / never | Later, only if wanted |
| — | Demo media and third-party MP3s in git | Separate | Raise separately; not part of W21-29 |
| — | Authorization of Stage 3.3d (5a) | — | — |

## 9. Confirmation

No implementation change was made:

- `pyproject.toml`, `hatch_build.py`, the checker, the CI and publish workflows, the inventory,
  the download script, the loader, models and baselines are unchanged;
- no tracked file was deleted or moved;
- no download, publication, deployment, commit or push.

All builds and probes ran in temporary directories outside the repository, offline (zero network
attempts), and nothing was deserialized.

## 10. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3d_plan/`; verify with `sha256sum -c SHA256SUMS.txt`.

| File | Content |
| --- | --- |
| `measure.json` | For CURRENT, NARROW_MIN and NARROW_TESTS: commands, exit codes, sdist digest, sizes, category breakdown, largest files and directories, every sdist and wheel member with size and digest; wheel-equality results |
| `reports/*.json` | Archive-checker reports for each configuration |
| `tests_in_narrow_sdist.json` | Test outcome counts inside the NARROW_TESTS sdist, with each non-passing test and its reason |
| `logs/*.txt` | Build and checker logs, local paths scrubbed (`.txt`, so that `.gitignore`'s `*.log` rule cannot drop them); network log (empty) |
| `scripts/measure.py` | The measurement script |
