# EXECUTION_021, Phase 3, Stage 3.3b — Manifest-driven model packaging allowlist (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.3b** |
| Status | **COMPLETE.** Every acceptance gate passed (15/15); the full suite passes |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, published or deployed |
| Date | 2026-10-14 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `49b7a1f`, unchanged |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3b/` (52 files, `SHA256SUMS.txt` verified) |

Labels: **observed** (measured, in the evidence), **inferred**, **unverified**.

---

## A. Decisions recorded

Recorded in `MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` §14.9 before any implementation file was
edited.

| ID | Owner decision | Implemented as |
| --- | --- | --- |
| **D-048** (W21-16) | A build fails if any `in_wheel: true` artifact is missing or fails size or SHA-256; no model-free flag | `plan()` raises `ModelArtifactError` naming every failing path and its code; sdist and wheel both stop |
| **D-049** (W21-28) | The sdist carries the same 16 approved artifacts and the generated manifest | The hook runs for both targets with the same inclusion map |
| **D-050** (W21-31) | The packed manifest is derived at build time from `scripts/model_manifest.json` | `derive_package_manifest()` bytes are written to a temporary file and force-included. The working-tree `fie/models/MANIFEST.json` is never packed |
| **D-051** (inclusion mechanism) | A custom Hatchling build hook replaces the broad globs | `hatch_build.py`; `artifacts` removed; `/fie/models/**` excluded for every target |
| **D-052** (W21-33) | Pin `hatchling==1.29.0` | `[build-system] requires = ["hatchling==1.29.0"]` |

## B. Files changed

| File | Change | Purpose |
| --- | --- | --- |
| `pyproject.toml` | `requires = ["hatchling==1.29.0"]`; `artifacts` globs removed; `[tool.hatch.build] exclude = ["/fie/models/**"]`; `[tool.hatch.build.hooks.custom] path = "hatch_build.py"` | The hook becomes the only way model files enter a distribution |
| `hatch_build.py` (new) | Build hook plus a plain `plan()` function | Verify and include exactly the approved artifacts; derive the manifest |
| `tests/test_build_artifact_allowlist.py` (new) | 32 tests | Section E |
| `MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | §14.9 | Decisions and result |
| This report and `evidence/WP021/phase3/stage3_3b/` | New | — |

Unchanged (observed with git): `scripts/`, including `model_manifest.json` and `download_models.py`;
`fie/`, including `_model_integrity.py`, the loader, thresholds and selection; `.github/`;
`evals/baselines/`; `Dockerfile`; `requirements.txt`; model and experiment files; `dist/`.

## C. Design

```text
python -m build (sdist or wheel; from the checkout or from the sdist)
  pyproject.toml: exclude /fie/models/**     → no file under fie/models/ is selected by pattern
  hatch_build.py ModelAllowlistHook.initialize (target sdist or wheel)
    plan(root):
      load fie/_model_integrity.py by file path       (no `import fie`, standard library only)
      validate_production(scripts/model_manifest.json) (invalid inventory → build fails)
      derived = derive_package_manifest(inventory)     (deterministic bytes)
      for each entry with in_wheel: true (16 today):
          read_verified(root, entry)                   (one read; size and SHA-256; never deserialized)
      any missing, unreadable, path violation or mismatch → ModelArtifactError naming every one
      unlisted files under fie/models/ → reported, never included
    force_include: each verified artifact → its fie/models/ path
                   derived bytes, in a temporary file → fie/models/MANIFEST.json
    a warning names unlisted files; an info line names present in_wheel:false files
  finalize: the temporary directory is removed (also registered with atexit)
```

- **Why it works** (measured in 3.3a): hook-supplied `force_include` overrides `exclude`, and is
  applied by both the sdist and the wheel builder. Removing `artifacts` and excluding
  `/fie/models/**` removes every other route by which a model-directory file could be selected,
  including files that `.gitignore` does not cover (`.gitkeep`, nested files).
- **sdist → wheel:** the sdist contains `hatch_build.py`, `pyproject.toml`, `.gitignore`,
  `scripts/model_manifest.json`, `fie/_model_integrity.py` and the 16 artifacts, so the hook runs
  again on the unpacked sdist and verifies the same files.
- **Editable builds** (`pip install -e .`): the hook verifies the same way (and fails the same
  way), but includes nothing, because an editable install runs from the source tree. Measured
  only through the code path; no editable install was performed, as no installation was allowed.
- **Diagnostics** name manifest-relative paths and a fixed code (`missing`, `mismatch`,
  `read_failed`, `path_violation`), for example: `model packaging refused: 1 of 16 approved model
  artifacts cannot be packaged (fie/models/pair_intent_classifier_v6_3b.pkl: mismatch). Fetch and
  verify them with python scripts/download_models.py. The build does not continue without them.`

## D. Acceptance experiment (observed, `acceptance.json`, `builds.json`)

**Setup:** exports from `git archive` of HEAD, with the uncommitted `pyproject.toml` and
`hatch_build.py` overlaid byte for byte (digests in `exports.json`). Builds ran with
`python -m build --no-isolation`, offline: `PIP_NO_INDEX`, proxies to a closed port, and the
socket-refusing audit hook (self-test `BLOCKED`). **Zero network attempts.** Tools: Python 3.10.19,
build 1.4.0, hatchling 1.29.0.

**Inputs:** the 25 inventory artifacts were verified 25/25 before copying, and the 19 extras were
found 19/19. The CRLF variant converts 875 text files: the 874 from 3.3a plus `hatch_build.py`.
Protected model bytes are identical in both variants.

| Setup | sdist | Direct wheel | Wheel from sdist | Gate |
| --- | --- | --- | --- | --- |
| M0 (no models) LF ×2, CRLF ×2 | **refused** (exit 1, `16 of 16 … missing`) | **refused** | not applicable: no sdist | pass |
| M1 (inventory) LF ×2, CRLF ×2 | exactly the allowlist | exactly the allowlist | exactly the allowlist | pass |
| M2 (inventory + 14 experiment files + 5 encoder by-products) LF ×2, CRLF ×2 | exactly the allowlist; warning names the 19 files | exactly | exactly | pass |
| M1S (stale `MANIFEST.json` + nested `sub/*.json` + `notes.txt`, which `.gitignore` does not cover) LF | exactly; derived manifest packed, stale one not; warning names the 2 files | exactly | exactly | pass |
| MT (approved classifier, one byte flipped) LF | **refused** (`…v6_3b.pkl: mismatch`) | **refused** | not applicable | pass |
| MM (approved side-car removed) LF | **refused** (`meta_clf.json: missing`) | **refused** | not applicable | pass |

**The allowlist found in every successful archive** (sdist, direct wheel and wheel from sdist,
LF and CRLF). These are exactly the 16 `in_wheel: true` paths plus the derived manifest, with
digests equal to the inventory:

```text
fie/models/MANIFEST.json                     fbab28ab75ac…  (5,112 B, = derive_package_manifest)
fie/models/meta_clf.json                     989091d0b229…
fie/models/meta_clf.pkl                      be6673d094a8…  (approved, D-034)
fie/models/pair_intent_classifier.pkl        ebea67fdc69b…  (v1)
fie/models/pair_intent_classifier_v3.pkl     a52c79b44959…
fie/models/pair_intent_classifier_v4.pkl     bd50eab9ba84…
fie/models/pair_intent_classifier_v5.pkl     3036809f955d…
fie/models/pair_intent_classifier_v6.pkl     25c1a421b03f…
fie/models/pair_intent_classifier_v6_3.pkl   4eea43fec568…
fie/models/pair_intent_classifier_v6_3b.pkl  9c682b285a9f…  (default)
fie/models/pair_intent_meta.json             ee2245119d8a…
fie/models/pair_intent_meta_v3.json          97df4a1acca8…
fie/models/pair_intent_meta_v4.json          00cf2aa50df6…
fie/models/pair_intent_meta_v5.json          d1b52dd543be…
fie/models/pair_intent_meta_v6.json          b83c0e7c5677…
fie/models/pair_intent_meta_v6_3.json        bb35fc9c6edf…
fie/models/pair_intent_meta_v6_3b.json       5babb7ec71c8…
```

**Absent from every archive** (named-member check): `.gitkeep`; both legacy pickles; all 14
experiment files; the 5 encoder by-products; `minilm-onnx/model.onnx` and `tokenizer.json`; the
5 server files; the M1S fixtures; the working-tree `MANIFEST.json`.

**Determinism and equality (observed):**

| Variant | sdist | Wheel |
| --- | --- | --- |
| LF | `6d2c55bf73d62d8a…`, 1,116 members | `810d5446a8018c20…`, 60 members (was 63: `.gitkeep` and the legacy pair are gone) |
| CRLF | `d9644d6603ba33b4…` | `9646a7a39f50f3b8…` |

- The archive bytes are **identical across M1, M2 and M1S** for each variant. Extra files in the
  source tree no longer change the output at all.
- Repeat builds are byte-identical.
- In every setup, the **direct wheel and the wheel from the sdist are byte-identical**.
- The `fie/models/` member digests are equal across the sdist, both wheels and both line-ending
  variants. As in 3.3a, CRLF changes only packaged source bytes, never model bytes. No
  cross-platform claim is made.

## E. Tests (observed)

**`tests/test_build_artifact_allowlist.py` — 32 passed.**

| Requirement | Tests |
| --- | --- |
| 1. Exactly the manifest-selected entries | `test_selection_is_exactly_the_in_wheel_entries`, `test_real_inventory_selects_its_sixteen_in_wheel_entries`, `test_real_tree_verifies_against_the_tracked_inventory` |
| 2. Size and digest checked | `test_altered_approved_artifact_fails[flip_byte, append_byte, truncate, replace]` |
| 3. Tampered artifact fails the build | the above, plus `test_build_fails_closed[tampered-sdist/wheel]` (real builds) |
| 4. Missing artifact fails the build | `test_missing_approved_artifact_fails`, `test_no_model_files_at_all_fails`, `test_build_fails_closed[missing-sdist/wheel]` |
| 5. Off-wheel legacy entries never included | `test_off_wheel_entries_are_neither_required_nor_included`, `test_nothing_unapproved_ships[…]` |
| 6. Root-level and nested extras excluded | `test_extras_experiments_and_gitkeep_are_never_selected`, `test_nothing_unapproved_ships[…]` |
| 7. Experiment files excluded | same |
| 8. `.gitkeep` excluded | same |
| 9. Manifest freshly derived despite a stale one | `test_manifest_is_derived_even_when_a_stale_one_is_present`, `test_archive_model_members_are_exactly_the_allowlist[…]` |
| 10. Derivation deterministic, established bytes | `test_derivation_is_deterministic_and_keeps_the_established_bytes` (`fbab28ab…`, 5,112 B) |
| 11. Wheel from sdist = direct wheel | `test_wheel_from_sdist_has_the_same_model_members_as_the_direct_wheel`, `test_archive_model_members_are_exactly_the_allowlist[sdist, wheel, from_sdist]` |
| 12. No `fie` or ML import | `test_hook_imports_neither_fie_nor_ml_packages` (child process with `-I`) |
| Globs cannot return | `test_no_pattern_selects_model_files` (no `artifacts`, `include`, `only-include` or `force-include` in `tool.hatch` names a model path), `test_fie_models_is_excluded_and_the_hook_is_registered` (pin, exclude, hook path) |
| Inputs travel with the sdist | `test_sdist_carries_the_hook_inputs`; `test_build_warns_about_unlisted_files` |

The real-build tests build a **synthetic** project that uses the real `pyproject.toml`,
`.gitignore`, `hatch_build.py` and `fie/_model_integrity.py`, so a configuration regression fails
them. They skip only when `build` or hatchling 1.29.0 is not installed. The CI test jobs do not
install them, so **in CI 14 of the 32 tests will skip**: the 13 real-build tests and the
hatchling class check (inferred from `ci.yml`). The other 18 `plan()`-level and configuration
tests still run there. The archive gate in CI is Stage 3.3c.

**Full suite** (`python -m pytest tests/ -m "not network" -p no:cacheprovider`): **1453 passed,
0 failed, 2 skipped.** The skips are the two Windows platform cases in `test_model_integrity.py`.
Before this stage: 1421 passed. The watched-source guard passes, because no file it watches
changed.

After the full run, one unused local variable was removed from the new test file (ruff F841).
The 32 focused tests were re-run and pass. The CI ruff rule set passes on both new files.

## F. Consequences to note

1. **Any build without the verified models now fails**, by design (D-048). This includes the CI
   `package` job, whose download step is not `--strict`: an unreachable release now fails the
   build instead of producing a model-free wheel with a warning. It also includes `pip install .`
   and `pip install -e .` from a checkout before `python scripts/download_models.py`. No workflow
   was changed. The CI and release gates are Stage 3.3c.
2. Building now needs `hatchling==1.29.0`. Isolated builds, as in CI and publication, fetch that
   exact version.
3. The published 1.18.0 and the stale `dist/` archives are unaffected and untouched.

## G. Deferred (not done)

- **Stage 3.3c:** the archive gate in `release_check.py`, CI and `publish-pypi.yml`, and a CI
  negative control.
- **W21-29:** narrowing the sdist. It still carries about 67 MB of non-SDK material, including
  third-party music files.
- **W21-14:** the published sdist.
- **W21-19:** line endings.
- **W21-32:** the source-tree rule for unlisted files.
- **W21-20-R:** the experiment-file relocation.
- **Out of scope:** cross-platform reproducibility; WP-022 (ONNX encoder, the sentence-transformers
  fallback, `session_tracker.py`).

## H. Integrity check (observed)

| Check | Result |
| --- | --- |
| `git status` | Modified: `pyproject.toml`, the master log. New: `hatch_build.py`, `tests/test_build_artifact_allowlist.py`, this report, the evidence directory. Nothing staged |
| `git diff --check` | Clean |
| Broad globs | Gone: no `artifacts` key; the word remains only in the explanatory comment |
| Inventory, download script, loader, thresholds, models, workflows, baseline, `CANONICAL` | Unchanged |
| Generated or ignored content | None new in the repository. The local `fie/models/MANIFEST.json` is unchanged (`fbab28ab…`). Build outputs stayed in temporary directories; the hook's temporary manifest directory is removed after each build |
| Network, install, download, deserialization | None |
| Commit, push, publish, deploy | None |

## I. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3b/`; verify with `sha256sum -c SHA256SUMS.txt`.

| File | Content |
| --- | --- |
| `acceptance.json` | Each of the 15 gates, its expectation and result; the allowlist with digests |
| `builds.json` | Every build: command, exit code, outputs and digests, `fie/models/` members, classification |
| `members/<sha16>.json` | Full member lists of the 4 distinct archives |
| `exports.json` | Archive digest, overlay digests, CRLF conversion list |
| `scenario_inputs.json` | 25/25 inventory checks, 19/19 extras, derived-manifest check |
| `test_results.json` | Focused and full-suite counts, skips, test names |
| `repository_state.json` | HEAD, changed files with digests, protected-path status, `CANONICAL` |
| `logs/*.log` | All 39 build logs (refusal messages included), local paths scrubbed; network log (empty) |
| `scripts/run_acceptance.py`, `scripts/package_evidence.py` | The scripts that produced this |
