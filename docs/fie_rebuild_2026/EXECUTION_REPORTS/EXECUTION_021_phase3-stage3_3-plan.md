# EXECUTION_021, Phase 3, Stage 3.3 — Planning audit: wheel allowlist, archive validation, sdist feasibility (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.3: planning only** |
| Status | **PLAN. Nothing implemented.** Stage 3.3 is **not ready for implementation**. It is ready for one bounded feasibility experiment (3.3a), subject to the decisions in section 9 |
| Commit status | Not committed by the assistant. This file is the only output of this run |
| Date | 2026-10-14 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `56335e28ae34048b87f5939e27b81f40b3b38dfe` |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |

Labels: **confirmed** (seen directly: a file, a named archive member, a log line), **inferred**
(reasoned from configuration or counts), **unverified** (needs a later, authorized check).

This run did not build, install, download, publish, move or modify anything. Archives were
inspected by listing member names and timestamps only; no member was extracted or opened. CI
information was read as GitHub run metadata and log text. No artifact was downloaded.

---

## 1. Prerequisites

| Check | Result |
| --- | --- |
| Stage 3.2 committed | **Confirmed.** `56335e2`, on top of `1443cc7`; working tree clean, nothing staged |
| Stage 3.2 CI | **Confirmed.** Pull-request run `38525827576` on `56335e2`: every job succeeded. `test (3.10)`, `(3.11)`, `(3.12)` each 1422 passed, 1 skipped. The worktree guard test now passes. `package` built `fie_sdk-1.18.0-py3-none-any.whl` (`4690cfdc…2a190e41`); T-base and T-fastapi both passed |
| Push to `main` after the merge | Run `38526409887` (`d44c390`) was **queued** when checked; its result is unverified |
| Release-check artifact of `38525827576` | Exists in CI; **not downloaded** (no network in this run) |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |

Stage 3.2 is therefore not a blocker. The decisions in section 9 are.

## 2. How packaging works today (confirmed from the files)

**`pyproject.toml`:**

```toml
[build-system]
requires      = ["hatchling"]          # unpinned; 1.29.0 locally
build-backend = "hatchling.build"

[tool.hatch.build]                     # applies to EVERY target: sdist and wheel
artifacts = ["fie/models/*.pkl", "fie/models/*.json"]

[tool.hatch.build.targets.wheel]
packages = ["fie"]
```

- There is no sdist target configuration, no `MANIFEST.in`, no build hook and no `exclude`.
- Hatch selects files by its VCS ignore rules (`.gitignore`). `artifacts` re-includes ignored
  files matching its globs. So **whatever `.pkl` or `.json` file is present in `fie/models/`
  at build time is packed, listed or not.** The globs are one level deep: `fie/models/minilm-onnx/`
  is ignored (`.gitignore:202`) and not matched.
- `fie/models/.gitkeep` is tracked and not ignored, so it is packed as an ordinary package file.

**Workflows:**

| Workflow | Download | Build | Content check |
| --- | --- | --- | --- |
| `ci.yml`, `package` job | `download_models.py`, **not** `--strict` (line 176) | `python -m build --wheel`: wheel straight from the checkout | Counts `fie/models/*.pkl|*.json`; **warns** if zero; then the release gate (installation and import only, AC-1 to AC-12) |
| `ci.yml`, test jobs | `--strict` | — | — |
| `publish-pypi.yml` (tag `v*` or manual) | `--strict` | `python -m build`: **sdist first, then the wheel built from that sdist** | Fails only if the wheel has **zero** model files, then `twine check` |

So what the published wheel contains is decided by what the sdist contains, which is decided by
what was in `fie/models/` on the runner.

**The runtime manifest (Stage 3.1/3.2).** `fie/models/MANIFEST.json` is written by
`download_models.py` only after all 25 inventory entries verify. It is git-ignored and matched
by the `*.json` artifact glob. Since Stage 3.2 the loader cannot load anything without it.

## 3. Inventory against package contents

Evidence columns:

- **PyPI:** the published `fie-sdk 1.18.0` wheel. Named members, hashed in Stage 3.0
  (`evidence/WP021/phase3/stage3_0/provenance/model_inventory.json`).
- **dist:** the stale local `dist/fie_sdk-1.18.0-py3-none-any.whl`, built 2026-08-10. Member
  names listed in this run.
- **CI:** the Stage 3.2 CI wheel. **Counts only**: `Files: 63, bundled models: 19`. No names.

"Inc" means a named member is present; "—" means named evidence shows it absent; "?" means no
named evidence.

| Artifact | Role / version | `in_wheel` | Git status; local file | PyPI | dist | CI | Stage 3.3 action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fie/models/pair_intent_classifier_v6_3b.pkl` | pair_classifier / v6_3b (default) | true | ignored; present | Inc | Inc | ? | Allowlist; verify digest at build |
| `fie/models/pair_intent_meta_v6_3b.json` | pair_meta / v6_3b | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_classifier_v6_3.pkl` | pair_classifier / v6_3 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_meta_v6_3.json` | pair_meta / v6_3 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_classifier_v6.pkl` | pair_classifier / v6 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_meta_v6.json` | pair_meta / v6 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_classifier_v5.pkl` | pair_classifier / v5 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_meta_v5.json` | pair_meta / v5 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_classifier_v4.pkl` | pair_classifier / v4 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_meta_v4.json` | pair_meta / v4 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_classifier_v3.pkl` | pair_classifier / v3 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_meta_v3.json` | pair_meta / v3 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_classifier.pkl` | pair_classifier / v1 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/pair_intent_meta.json` | pair_meta / v1 | true | ignored; present | Inc | Inc | ? | same |
| `fie/models/meta_clf.pkl` | meta_classifier | true | ignored; present (`be6673d0…`) | Inc | Inc | ? | same |
| `fie/models/meta_clf.json` | meta_classifier_meta | true | ignored; present (725 B) | Inc, **different bytes** (602 B, Stage 3.0) | Inc (602 B) | ? | same. Build-time verification makes a recurrence fail |
| `fie/models/minilm-onnx/model.onnx` | encoder | false | ignored directory; present | — | — | ? | Must stay out; assert absent |
| `fie/models/minilm-onnx/tokenizer.json` | tokenizer | false | ignored directory; present | — | — | ? | Must stay out; assert absent |
| `fie/models/failure_classifier_v4.pkl` | legacy | false | ignored; present | **Inc** | **Inc** | ? (inferred Inc) | **Exclude.** Today it is packed against `in_wheel: false` |
| `fie/models/feature_columns_v4.pkl` | legacy | false | ignored; present | **Inc** | **Inc** | ? (inferred Inc) | **Exclude** (same) |
| `models/failure_classifier_slim.pkl` | server | false | ignored; present | — | — | ? | Out by path (not under `fie/`); assert absent |
| `models/feature_columns_slim.pkl` | server | false | ignored; present | — | — | ? | same |
| `models/failure_classifier_slim_meta.json` | server | false | ignored; present | — | — | ? | same |
| `storage/faiss_adversarial.index` | server | false | ignored; present | — | — | ? | same |
| `storage/faiss_adversarial_meta.json` | server | false | ignored; present | — | — | ? | same |
| `fie/models/MANIFEST.json` | generated, not an inventory entry | — | ignored; generated by Step 0 (5,112 B, `fbab28ab…`) | — (did not exist) | — | ? (inferred Inc) | **Required in the wheel**; must equal the derivation of the tracked inventory (section 6, 3.3b) |
| `fie/models/.gitkeep` | — | — | **tracked**, not ignored | ? | **Inc** | ? | **Exclude** from the wheel (D-040) |
| 7 experiment classifiers + 7 side-cars (`_v2`, `_v3_w3x`, `_v6_1_contaminated`, `_v6_noaug`, `_v64_harmaug`, `_v64_harmaug_capped`, `_v65_benignaug`) | unlisted | — | ignored; present locally only | — | **Inc for 4 of 7**: `v2`, `v3_w3x`, `v6_1_contaminated`, `v6_noaug` and their side-cars. The three `v64`/`v65` files postdate the build | ? | **Exclude** by allowlist; never by location alone (section 5) |
| 5 encoder by-products (`minilm-onnx/config.json`, `ort_config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt`) | unlisted (D-038) | — | ignored directory; present | — | — | ? | Must stay out; assert absent |

**Count check (inferred).** CI's 19 bundled `.pkl`/`.json` files match exactly: 14 pair
artifacts, 2 meta artifacts, the 2 legacy files and `MANIFEST.json`. The CI runner has no
experiment files, because they are never in git or the release. This fits the configuration, but
it is **not named evidence**. Stage 3.3a must confirm it by member name.

**What the evidence establishes:**

1. **Confirmed:** the current configuration packs files that are not on the allowlist. The legacy
   pair is in both the published and the stale wheel. Four experiment classifiers are in the
   stale local wheel.
2. **Confirmed:** whether experiment files are packed depends on the build machine's local files.
   The same commit yields different wheels on this machine and on CI.
3. **Inferred:** the CI wheel at `56335e2` contains the legacy pair and `MANIFEST.json`.

## 4. Source-distribution feasibility

### 4.1 What happens today

**Confirmed from the stale sdist** (`dist/fie_sdk-1.18.0.tar.gz`, 345 members, names listed):

- It contains almost the **whole repository**: `app/`, `engine/`, `storage/*.py`, `tests/`,
  `docs/`, `notebooks/`, `.github/`, `Frontend/`, `Dockerfile` and so on, plus `.gitignore`,
  `pyproject.toml` and `scripts/download_models.py`.
- `fie/models/` holds **every** `.pkl`/`.json` present at build time, including four experiment
  classifiers and the legacy pair.
- There is no ONNX directory and no server model (`models/` has only `.gitkeep`; `storage/` has
  no index).

**The published sdist** (W21-14) has never been downloaded or inspected. **Unverified.**

**What a clean checkout produces:**

- **Confirmed, Phase 1 evidence** (`evidence/WP021/phase1/build.json`): a `git archive` export
  (no ignored files) builds a wheel with **no model files at all**, and its `fie/models/`
  contains only `.gitkeep`. Two builds were byte-identical (`24962aae…`) on one machine.
- **Inferred:** an sdist from a clean checkout without the download has no models, so a wheel
  built from it has none. Since Stage 3.2 that wheel fails closed at run time (`unavailable_model`).
- **Inferred:** an sdist from a checkout after `download_models.py` contains the 16 SDK artifacts,
  the legacy pair and `MANIFEST.json`, and a wheel built from it contains the same.

### 4.2 Answers to the feasibility questions

| Question | Answer | Status |
| --- | --- | --- |
| Which artifacts are available at build time | Only what is on disk. Git provides none; `download_models.py` provides all 25 inventory entries and `MANIFEST.json`; a research machine also has the experiment files | Confirmed |
| Does the build need a previously generated runtime manifest | Today, no: it packs it if present and omits it if not. A wheel without it loads no model since Stage 3.2 | Confirmed (configuration); runtime effect confirmed by Stage 3.2 tests |
| Does it depend on the downloader running first | Yes, for any model content. Without it the build silently produces a model-free wheel (the `package` job only warns) | Confirmed |
| Can network access, downloads or local ignored files change the result | **Yes.** Ignored local files are packed, and the download decides what exists. The build itself makes no network call (inferred), but CI's isolated build installs an **unpinned** hatchling from PyPI | Confirmed (local files); inferred (hatchling) |
| What an sdist must contain to build the intended wheel | Under the plan in section 6: `pyproject.toml`, the build hook, `fie/` including `fie/_model_integrity.py`, `scripts/model_manifest.json`, and the 16 `in_wheel` artifacts. Whether the sdist should carry the artifacts is decision W21-28 | Proposed |
| Does wheel-from-sdist honour `.gitignore` without `.git` | Hatch documents that it reads the root `.gitignore`, and the sdist contains `.gitignore` | **Unverified**; test in 3.3a |
| Does `exclude` override `artifacts` in hatchling 1.29 | Needed for the allowlist; behaviour not checked | **Unverified**; test in 3.3a |
| What reproducibility is justified | Same machine, same inputs: byte-identical wheels (Phase 1). Archive timestamps are normalized to 2020-02-02 (seen in the stale wheel). Nothing else | Confirmed, narrowly |

### 4.3 How to test sdist → wheel safely (Stage 3.3a)

All of it happens in a temporary directory outside the repository, with no network, using
`python -m build --no-isolation` with the locally installed `build` 1.4.0 and hatchling 1.29.0
(as in Phase 1).

1. Export the commit with `git archive`. It has no ignored files, and the canonical working tree
   is never used as a build directory.
2. **Scenario M0 (no models):** build sdist, wheel and wheel-from-sdist. Record every member
   name.
3. **Scenario M1 (inventory models):** copy into the export only the 25 inventory files and the
   generated `MANIFEST.json`, each verified against the manifest by hash before copying (read,
   never unpickled). Build the same three archives.
4. **Scenario M2 (dirty tree):** M1 plus copies of the seven experiment files and the encoder
   by-products, verified only by hash and never opened otherwise. This shows what a research
   machine produces today.
5. For each scenario and archive, record the member names, the `fie/models/` subset, digests, and
   whether `.gitkeep`, the legacy pair, experiment files and `MANIFEST.json` are present. Build
   each twice and compare bytes.
6. Do the M1 builds once from an LF export and once from a CRLF export (`git -c core.autocrlf=true
   archive`, or a CRLF-converted copy) to measure the line-ending effect (W21-19).

Outputs go to an evidence directory only. No canonical file changes.

## 5. The experiment-file boundary

**The facts:**

- The seven experiment classifiers and their side-cars live in `fie/models/` on this machine.
  They are unlisted, never in git and never in a release (Stage 3.0).
- They are written or read by these research scripts:
  - `scripts/train_pair_harmaug.py` and `scripts/eval_harmaug.py` (v64/v65 variants);
  - `scripts/retrain_pair_v2.py` (v2), `scripts/retrain_pair_v3.py` (v3 and v3_w3x) and
    `scripts/retrain_pair_exp8.py`;
  - `evaluation/phase2/exp6_before_after/run_eval.py` and
    `evaluation/phase2/exp8_weight_comparison/run_eval.py`.
  At least eleven other scripts use `ROOT / "fie" / "models"` as their model directory.
- Since Stage 3.2 the SDK can never load them.

**Two rules, kept separate:**

- **Rule A — the archive check (recommended now).** Fail the build or the gate if any archive
  member under `fie/models/` is not on the allowlist. It does not care what else is on disk, so
  experiment files are excluded without moving them and without breaking any research script.
- **Rule B — a source-tree check.** Fail the build if any unlisted model file exists in
  `fie/models/` at all. Enabled today, it would fail every build on this machine and on any
  research machine. It must wait.

**Recommended sequence:**

1. Stage 3.3b: the allowlist build hook. Experiment files are **excluded** and named in a build
   **warning**.
2. Stage 3.3c: archive validation (Rule A) in the build hook and in the release gate.
3. A separate stage, **W21-20-R** (research-workflow migration): agree a research directory
   outside the package (for example `research/models/`). Give the research scripts one
   configurable model directory, defaulting to the new location with a clear message. Move the
   fourteen files. Re-run the research scripts' dry-run or help paths only.
4. Only after W21-20-R: optionally make Rule B an error (W21-32).

Prerequisite for an unconditional Rule B: W21-20-R completed, and an owner decision on whether
local developer builds (as opposed to release builds) may still warn.

## 6. Proposed sub-stages

Each sub-stage is separately authorized. Each ends with an execution report, evidence, the
canonical baseline unchanged and the owner's manual commit.

### 3.3a — Sdist and wheel feasibility experiment (A)

| | |
| --- | --- |
| Objective | Replace every "unverified" in sections 3 and 4 with named evidence |
| Allowed files | An evidence directory and an execution report. Temporary build trees outside the repository |
| Work | Section 4.3, scenarios M0, M1 and M2, with LF and CRLF exports. Also test `exclude` against `artifacts`, `.gitignore` handling without `.git`, and `force-include` of a generated file, each in a throw-away copy of `pyproject.toml` inside the temporary export |
| Evidence | Member lists of every archive, the `fie/models/` subset, digests, repeat-build comparison, LF/CRLF comparison |
| Non-goals | No change to `pyproject.toml`, workflows or any tracked file; no download (the local verified files are used); no install; no publish |
| Acceptance | Each row of section 3 has a named status for M0, M1 and M2. The CI-count inference is confirmed or refuted. The three hatch behaviours are answered |
| Depends on | Owner authorization only |

### 3.3b — Manifest-driven allowlist and the runtime manifest in the wheel (B and C)

| | |
| --- | --- |
| Objective | Wheel and sdist model content comes from the inventory, never from the directory |
| Allowed files | `pyproject.toml` (build section only), a new `hatch_build.py` (custom build hook), new tests, report and evidence |
| Work | (1) Remove the `artifacts` globs. (2) The hook loads `fie/_model_integrity.py` by path (standard library only, as the download script does). (3) It validates `scripts/model_manifest.json` with `validate_production` and derives the package manifest in memory. (4) If `fie/models/MANIFEST.json` exists, it must be byte-identical to the derivation. (5) For each `in_wheel: true` entry, it reads the file once, verifies size and SHA-256, and force-includes it. (6) It force-includes the **derived** manifest bytes as `fie/models/MANIFEST.json`. (7) It excludes every other file under `fie/models/`, including `.gitkeep`, the legacy pair and anything unlisted, and names the unlisted ones in a warning. (8) Missing or mismatched `in_wheel` artifacts follow W21-16 |
| Tests | Synthetic temporary projects built with the hook (no real models deserialized): an extra unlisted file, an extra legacy file, a tampered artifact, a missing artifact, a stale `MANIFEST.json`, `.gitkeep`. Each asserts the exact `fie/models/` member set or a named build failure. A deterministic test that the same inputs give the same member list |
| Non-goals | No change to the loader, the inventory, the download script, thresholds or the baseline. No CI or publish change (3.3c). No experiment relocation. No line-ending normalization (3.3d) |
| Acceptance | From an M1 tree, both the wheel and wheel-from-sdist contain exactly the 16 `in_wheel` artifacts and `MANIFEST.json` under `fie/models/`, with the manifest's digests. From an M2 tree the output is identical, with a warning naming the experiment files. A tampered artifact fails the build with its path. The `BL-0002` per-prompt records are byte-identical; the full suite passes |
| Depends on | 3.3a results; W21-16, W21-28, W21-31 |

### 3.3c — Exact archive validation in the gate and CI (D)

| | |
| --- | --- |
| Objective | A wheel whose model content is not exactly the allowlist cannot pass CI or publication |
| Allowed files | `scripts/release_check.py` (new criteria AC-13 to AC-18, as PLAN_021 §5.2 numbers them, adapted), its tests, `ci.yml` (`package` job: `--strict` download, or the W21-16 alternative), `publish-pypi.yml` (replace the "at least one model" check with the gate), report and evidence |
| Work | The gate checks the wheel's `fie/models/` member set equals the allowlist plus `MANIFEST.json`. It checks each member's digest against the packed manifest, and the packed manifest against the derivation of the tracked inventory. It asserts the absence of `.gitkeep`, the legacy pair, encoder files and by-products, server files and unlisted files. It runs on the wheel and on a wheel built from the sdist |
| Tests | Gate unit tests with synthetic wheels for each failure. A CI negative control on a branch that packs an extra file, as in Phase 2 |
| Non-goals | No publication. No version change (Phase 4). Rule B is not enabled |
| Acceptance | The positive CI run passes; the negative control fails, naming the member. The release-check report lists every member |
| Depends on | 3.3b; owner approval for workflow changes |

### 3.3d — Line endings and reproducibility (E)

| | |
| --- | --- |
| Objective | Say exactly what is reproducible, and remove the line-ending variable if the owner agrees |
| Allowed files | `.gitattributes` (new) if W21-19 approves it, a renormalization commit made by the owner, report and evidence |
| Work | Use the 3.3a LF/CRLF comparison. Propose `.gitattributes` with `* text=auto eol=lf`, and `*.pkl`, `*.onnx`, `*.index` marked `binary`. Model JSON is downloaded, not checked out, so it is unaffected; only tracked text such as `.py` and `scripts/model_manifest.json` is. Pin the hatchling version in `[build-system]` if W21-33 agrees. Repeat the M1 builds on Windows and on CI (Linux) and compare |
| Non-goals | No promise of byte-identical cross-platform wheels unless the comparison shows it |
| Acceptance | A recorded statement for each pair of platforms: identical or not, and why. The `fie/models/` content is identical by digest on all platforms (the hook verifies it) |
| Depends on | 3.3a; W21-19, W21-33 |

**Separate and later:**

- **W21-20-R**, the research-workflow migration (section 5).
- **Rule B** (W21-32).
- **Narrowing the sdist** to the SDK (it currently carries the whole server and repository, W21-29).
- **W21-14**, inspecting the published sdist (needs a download).

## 7. Proposed acceptance tests (consolidated)

| # | Test | Sub-stage |
| --- | --- | --- |
| T1 | Wheel from an M1 tree: `fie/models/` members equal the 16 `in_wheel` paths plus `MANIFEST.json`, exactly | 3.3b |
| T2 | Wheel from the sdist of an M1 tree: same set and same digests as T1 | 3.3b |
| T3 | M2 tree (experiment files present): identical `fie/models/` content to T1; a warning names each excluded unlisted file | 3.3b |
| T4 | A tampered `in_wheel` artifact: the build fails, naming the path; no archive is produced | 3.3b |
| T5 | A missing `in_wheel` artifact: behaviour per W21-16 (fail, or an explicit model-free mode that is marked as such) | 3.3b |
| T6 | A stale `fie/models/MANIFEST.json`: the build fails | 3.3b |
| T7 | Absent from every archive: `.gitkeep`, legacy pair, encoder and tokenizer, five by-products, five server files | 3.3b, 3.3c |
| T8 | The packed `MANIFEST.json` equals `derive_package_manifest(scripts/model_manifest.json)` | 3.3c |
| T9 | Gate negative control: a wheel with one extra member fails in CI, naming it | 3.3c |
| T10 | An installed wheel from T1, with `[ml]` in an isolated environment: the default classifier loads through the verified path. Needs the `[ml]` install (about 209 MB) and owner approval; otherwise it stays unverified | 3.3c or Phase 4 |
| T11 | Two builds of the same tree on one machine give byte-identical archives | 3.3a, 3.3d |
| T12 | LF and CRLF exports: archive comparison recorded | 3.3a, 3.3d |
| T13 | Full suite passes; `BL-0002` per-prompt records byte-identical | every stage |

## 8. Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| A research machine publishes or shares a wheel with experiment pickles | Unreviewed models distributed (has happened: the stale wheel) | 3.3b allowlist; 3.3c gate |
| Legacy pair keeps shipping against `in_wheel: false` | Unneeded pickles in every install (has happened: PyPI 1.18.0) | 3.3b |
| `publish-pypi.yml` checks only "at least one model" | A wrong, partial or extra model set passes publication | 3.3c |
| The CI `package` job builds after a non-strict download | A model-free wheel passes CI with only a warning | 3.3c and W21-16 |
| Wheel built from an sdist lacking `scripts/model_manifest.json` or the hook | The hook cannot derive the manifest | 3.3b requires those files in the sdist; T2 tests it |
| Hatch precedence of `exclude`, `artifacts` and `force-include` differs from expectation | Allowlist leaks | 3.3a tests it before 3.3b is written |
| Unpinned hatchling | Builds change with upstream releases | W21-33 |
| CRLF checkouts on Windows | Wheels differ by platform in `.py` bytes; model content unaffected (verified at build) | 3.3d, W21-19 |
| Rule B enabled too early | Breaks every build on research machines | Section 5 sequence |
| The published 1.18.0 carries a different `meta_clf.json` and the legacy pair | Already published; cannot be changed | Changelog in Phase 4; yank decision W21-6 |

## 9. Decisions needed

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| **W21-16** | Build strictness when an `in_wheel` artifact is missing | (a) Always fail. (b) Fail unless an explicit `FIE_BUILD_MODEL_FREE=1`, which builds a wheel with no models and no `MANIFEST.json` and says so loudly. (c) Today's silent model-free build | **(b)**: releases and CI fail, and a deliberate model-free developer build stays possible and fail-closed at run time |
| **W21-19** | Line endings | (a) `.gitattributes` forcing LF for text and marking binaries, plus a one-time renormalization. (b) Leave as is and document | (a), decided after the 3.3a measurement |
| **W21-28** (new) | Does the sdist carry the 16 `in_wheel` artifacts | (a) Yes, so an sdist builds the intended wheel offline (about 0.3 MB of pickles). (b) No, and wheels are built only from a checkout after the download; publication changes to `--wheel` from the checkout | (a): the publish workflow already builds the wheel from the sdist |
| **W21-29** (new) | Narrow the sdist to the SDK (exclude `app/`, `engine/`, `tests/`, `docs/` and so on) | Now or later | Later, separately; record it as a known oddity |
| **W21-31** (new) | Source of the packed `MANIFEST.json` | (a) The hook derives it from the tracked inventory at build time and requires any existing generated file to match. (b) Pack the downloader's file as is | (a): one source of truth, checked again at build |
| **W21-32** (new) | Rule B (fail on unlisted files in the source tree) | After W21-20-R only; release builds only, or all builds | Release builds only, after W21-20-R |
| **W21-33** (new) | Pin hatchling in `[build-system]` | Pin, or leave unpinned | Pin to the version 3.3a measures |
| W21-20-R | Research-workflow migration plan and relocation | Separate plan | Plan before Rule B |
| W21-14 | Inspect the published sdist (one 3.3 MB download) | — | Before Phase 4 |
| — | Authorization for 3.3a, then each of 3.3b, 3.3c, 3.3d | — | — |

## 10. Known unknowns

1. CI wheel member **names** at `56335e2` (only counts are known). This can be resolved by
   reading the uploaded `release-check` artifact, if it lists members, or by 3.3a.
2. Hatchling 1.29: precedence of `exclude`, `artifacts` and `force-include`, and whether
   `.gitignore` is honoured in an unpacked sdist without `.git`.
3. The content of the published 1.18.0 sdist (W21-14).
4. Whether Windows (CRLF) and Linux builds of the same commit differ, and in which members.
5. The result of the queued `main` push run `38526409887`.
6. Whether the release gate's `[ml]` level (Stage 3.5) will be authorized; until then T10 is
   unverified.

## 11. Verdict

**Not ready for implementation of 3.3b to 3.3d.** The exact allowlist is unambiguous: the 16
`in_wheel: true` entries plus `MANIFEST.json`. But how it is enforced depends on hatch behaviour
not yet observed, and on W21-16, W21-28 and W21-31. Sdist behaviour is inferred, not observed.
The experiment-file boundary has a safe sequence (section 5), but its final step depends on
W21-20-R.

**Ready for 3.3a**, a no-network feasibility experiment in temporary directories, once it is
authorized.
