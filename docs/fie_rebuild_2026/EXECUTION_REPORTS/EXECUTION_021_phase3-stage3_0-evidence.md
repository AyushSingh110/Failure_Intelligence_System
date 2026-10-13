# EXECUTION_021, Phase 3, Stage 3.0 — Evidence and provenance (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.0 only**: evidence, provenance, read-only investigation |
| Status | **COMPLETE. Two owner decisions are required before Stage 3.1: W21-7 and W21-20** |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, published or deployed |
| Date | 2026-10-13 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `28683d4`, unchanged |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_0/` |
| What was changed | Documentation and evidence only. No source, test, manifest, build, workflow or model file |

Labels. **VERIFIED**: read directly from an artifact, a log, git, or a digest. **INFERRED**:
follows from verified facts but was not observed. **UNKNOWN**: not established.

No model file was deserialized. Every statement about a model file rests on its size, its
SHA-256, its JSON side-car, git, or metadata about it.

---

## A. Repository state

| Check | Result |
| --- | --- |
| Branch, HEAD | `rebuild/wp-001-eval-harness`, `28683d4b8e0ca137589ded73013a2ffbbef09a2a` |
| Working tree before this stage | Clean; nothing staged; no unexpected modification |
| Phase 2 commit | Present: `28683d4` "ci: run the installed-wheel release gate on T-base and T-fastapi" |
| CI integration | `.github/workflows/ci.yml` runs `scripts/release_check.py`; both tracked, with `tests/test_release_check.py` |
| `origin/main` | `d2b0c16`, the merge of pull request #19 (merged 2026-10-12T17:12:58Z). HEAD is an ancestor of it |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |
| Existing WP-021 evidence | All five checksum sets verify: 19, 39, 19, 37 and 45 files |
| Match with the approved Phase 3 plan | Yes |

---

## B. CI evidence

Three different things, kept apart:

| Kind | What exists |
| --- | --- |
| Local test | Phase 2's two controls on Windows, Python 3.10.19, from local wheelhouses (`evidence/WP021/phase2/`) |
| CI job conclusion | GitHub's recorded conclusion for each job of each run (below) |
| Uploaded CI artifact | The gate's own `release-check` report uploaded by each run, downloaded here and re-verified against its own checksum list |

### B.1 Run `38282217538` — pull request

| | |
| --- | --- |
| Event, branch, commit | `pull_request`, `rebuild/wp-001-eval-harness`, `28683d4b8e0ca137589ded73013a2ffbbef09a2a` |
| Status, conclusion | `completed`, **`success`** |
| Jobs | `secret-scan`, `lint`, `dependency-audit`, `test (3.10)`, `test (3.11)`, `test (3.12)`, `health-check`, `package`: success. `deploy-space`: **skipped** (not a push to `main`) |
| `package` job steps | All success, including "Release check (installed wheel — T-base and T-fastapi)" and "Upload release-check report" |
| Runner | `ubuntu-24.04`, image version `20261004.327.1` |
| Python | 3.11.17 (host and both environments); pip 24.0 |
| Wheel | `fie_sdk-1.18.0-py3-none-any.whl`, 314,670 bytes, 60 files, 18 bundled model files |
| **Wheel SHA-256** | **`1b149e77fda073783af47d77caef9d3fc2e16e8a1f846f5b3407e0c51274bf40`** |
| Digest sources | `sha256sum` in the job log, the gate's line in the job log, and the uploaded `report.json`: all three equal |
| Dependencies | From PyPI ("pip's configured index") |
| **T-base** | **11 of 11 criteria PASS**, exit 0 |
| **T-fastapi** | **10 of 10 criteria PASS**, exit 0; `starlette` **1.7.0**, `anyio` 4.15.1 |
| Gate exit code | 0 |
| Uploaded artifact | `release-check`, 117,321 bytes, digest `sha256:079dd898f50ef352df10bcc294a8c5c4c86dbd1a67d81f5d1c8919ccc1e2cb54`, expires 2027-01-10. 45 files, all verify against the artifact's own checksum list |

### B.2 Run `38282733913` — push to `main`

Rechecked after completion; the earlier snapshot showed `deploy-space` still running.

| | |
| --- | --- |
| Event, branch, commit | `push`, `main`, `d2b0c168543a9ca3c012c6be1365ee6034cf0259` |
| Status, conclusion | `completed`, **`success`** |
| Jobs | All nine success: `secret-scan`, `lint`, `dependency-audit`, `test (3.10)`, `test (3.11)`, `test (3.12)`, `package`, `health-check`, **`deploy-space`** |
| `package` job steps | All success, including the release check and the upload |
| Runner | `ubuntu-24.04`, image version `20261004.327.1` |
| Wheel SHA-256 | `1b149e77fda073783af47d77caef9d3fc2e16e8a1f846f5b3407e0c51274bf40`; three sources equal |
| T-base, T-fastapi | 11 of 11 and 10 of 10 PASS; exit 0; `starlette` 1.7.0 |
| Uploaded artifact | `release-check`, 117,300 bytes, digest `sha256:86e62c8ece816ae4e45c303b1d5c658a169cbb262a3be1a0dc5fc4a3df79db0b`. 45 files verify |

### B.3 What the CI runs add

- **The gate's Linux path works.** It had never run outside Windows.
- **Two independent CI builds produced the same wheel, byte for byte** (`1b149e77…` in both
  runs). VERIFIED for these two builds on one runner image.
- **The CI wheel contains model files** (18), because the `package` job downloads them before
  building. Local control wheels contained none. So the gate has passed on a wheel with models
  and on one without. It does not yet check which models; that is Phase 3.
- **`starlette>=0.52.1` resolved to 1.7.0** on CI. The middleware check passed on it. Versions
  between 0.52.1 and 1.7.0 are untested.
- The CI wheel's digest differs from every locally built wheel (models, and line endings).

Limits. These are GitHub's records and the gate's own report. The per-command output files of
the artifacts and the raw job logs are not copied into the repository; the artifacts expire on
2027-01-10. `deploy-space` succeeded, which is a statement about the deployment job, not about
this work package.

---

## C. Model inventory (R1)

39 files appear under `fie/models/` in at least one of: manifest, working tree, PyPI wheel,
`dist/` wheel. Full table with sizes and full digests: `provenance/model_inventory.json`.

| Group | Manifest | Working tree | Release assets | PyPI wheel | `dist/` wheel |
| --- | --- | --- | --- | --- | --- |
| Classifiers `v6_3b` (default), `v6_3`, `v6`, `v5`, `v4`, `v3`, original; their side-cars | listed | equal | equal | equal | equal |
| `failure_classifier_v4.pkl`, `feature_columns_v4.pkl` | listed | equal | equal | equal | equal |
| `meta_clf.pkl`, `meta_clf.json` | **B** | B | B | **A** | **A** |
| `minilm-onnx/model.onnx`, `tokenizer.json` | listed | equal | equal | absent | absent |
| Four other `minilm-onnx/*.json` files | not listed | present | absent | absent | absent |
| Seven experiment classifiers and side-cars (section D) | **not listed** | present | absent | absent | four of them present |

Not inspected: the published source distribution `fie_sdk-1.18.0.tar.gz` (never downloaded);
the original `meta_clf` release assets (replaced, no longer exist).

One new static finding: four JSON files in `fie/models/minilm-onnx/` (`config.json`,
`ort_config.json`, `special_tokens_map.json`, `tokenizer_config.json`) are in neither the
manifest nor the release. They are not model-type files and the build glob does not reach that
subdirectory, so they are not packaged today. Stage 3.1 needs a rule for them.

---

## D. The seven unmanifested experiment files

Common to all seven, VERIFIED: untracked now; ignored by `.gitignore` line 15
(`**/models/*.pkl`) and line 18 for the side-cars; absent from the manifest, from the GitHub
release and from the PyPI wheel; each 11,349 bytes with a JSON side-car.

| File | SHA-256 | Side-car says | Git history | Tracked references |
| --- | --- | --- | --- | --- |
| `_v2` | `cfca34c732c61f217c871d301dd45112ca4ab3ceb7c69147cd9a5796bd832dc3` | trained 2026-05-17, threshold 0.4127 | **Committed 2026-05-17 (`0a4ef8f`, `ab681c3`), removed from tracking 2026-05-24 (`935983a`, `38e21ef`). The bytes on disk equal the committed blob `14a317b3…`** | SDK loader's version list; `scripts/retrain_pair_v2.py`, `retrain_pair_v3.py`; `docs/ARCHITECTURE.md`, `docs/CODEBASE.md` |
| `_v3_w3x` | `d802d62c6de8a97127ff04f569d89a681dc644dcf2dd0a64d95cb1c3dd17473d` | trained 2026-06-02, threshold 0.4182 | Never committed; bytes not in the object database | None |
| `_v6_1_contaminated` | `9d7f3a1959785f2b7b7044fdefcd1f1e5591b6e5d3752a424e6902cbc2002e45` | trained 2026-06-20, `version: "v6"`, threshold 0.50, sklearn 1.7.2 | Never committed | By name in `docs/RESEARCH_LOG.md:367` only: "Backups kept: v6_noaug, v6_1_contaminated" |
| `_v6_noaug` | `759e8d8ea3d937951ad2cd25cf04b9568b9cc569fa37473f128d7afbc09a58d3` | trained 2026-06-20, `version: "v6"`, threshold 0.45, sklearn 1.7.2 | Never committed | The same line; a results row "v6-noaug @ 0.45" |
| `_v64_harmaug` | `b00a8f220ab2bdc86f769fdc1ab87fdd26cabc5fa12989779de9232da011a400` | trained 2026-08-12, `version: "v64_harmaug"`, threshold 0.50 | Never committed | `scripts/eval_harmaug.py:71` (default); `scripts/train_pair_harmaug.py` (writes it); three reports in `data/benchmark_audit/` |
| `_v64_harmaug_capped` | `e085b93750d668936b4e9af6a88e3767290d00529727d27f4ff5af0a2b1defa0` | trained 2026-08-12, threshold 0.60 | Never committed | `scripts/eval_harmaug.py:72` (default); the same reports; `docs/RESEARCH_LOG.md` |
| `_v65_benignaug` | `8929a7e339a4a1e1c1cc2cf90a67722be062be0e9ba7a22f09201e8d63cae5cc` | trained 2026-08-12, threshold 0.65 | Never committed | `scripts/eval_harmaug.py:101` (`--model` example); `docs/RESEARCH_LOG.md:1799-1804`; `scripts/benignaug_build_trainset.py:376`; two reports |

The `dist/` wheel contains `_v2`, `_v3_w3x`, `_v6_1_contaminated` and `_v6_noaug`, with the
same digests as on disk. It was not modified.

**How they are loaded** (source read statically; nothing was run):

- **Through the SDK and `FIE_PAIR_VERSION`: only `_v2`.** The loader's fixed list in
  `fie/layers/pair.py` is `v6_3b, v6, v6_3, v5, v4, v3, v2, v1`. The other six are not in it and
  cannot be selected through the SDK at all.
- **`_v64_harmaug`, `_v64_harmaug_capped`, `_v65_benignaug`: by `scripts/eval_harmaug.py`'s
  own `joblib.load`** (line 134) on `ROOT/fie/models/<file>`. The first two are in its default
  list; the third is passed with `--model TAG=FILE`. `FIE_PAIR_VERSION` is not involved. A
  missing file makes the script print `ERROR: missing <file>` and exit 1.
- `scripts/train_pair_harmaug.py` and the two `retrain_pair_*` scripts **write** into
  `fie/models/`; they do not read these files.
- `_v3_w3x`, `_v6_1_contaminated`, `_v6_noaug`: no tracked code reads them.
- No workflow, Dockerfile, deploy script, test or harness profile refers to any of the seven.
  The harness profiles pin `v6_3b` and `v6`, both manifest-listed.

**UNKNOWN for all seven:** the data and code revision each was trained from; whether the
side-car describes the file beside it; whether any is safe to deserialize. For `_v2` it is
VERIFIED that the bytes are the ones committed on 2026-05-17, which establishes that the file
has not changed since, not where it came from.

Two details worth noting. `_v6_1_contaminated` and `_v6_noaug` both declare `version: "v6"`,
the same version string as the shipped, manifest-listed `v6`; a result reporting only the
declared version could not tell them apart, though the digest could. And the research script
loads unmanifested pickles directly; that is the owner's research workflow, outside the
package, and is not changed by WP-021.

---

## E. `meta_clf.pkl` (R2)

| | A | B |
| --- | --- | --- |
| SHA-256 | `01be1995d70f1963ae4f912468642f79f27fd0d68d9135187f0592329c162c9e` | `be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6` |
| Size | 179,675 | 212,131 |
| Side-car | 11 features, threshold 0.50, 18,601 training rows | 12 features, threshold 0.41, 2,370 training rows; `989091d0b2296c837d1a3bc3ac6857600fe15439e68a7ecd96de1d6ca8ec76ce` |
| Found in | PyPI wheel 1.18.0; `dist/` wheel; the manifest from v1.14.0 through `d1565d1`, including at tag `v1.18.0` | Manifest at HEAD; working tree; the release asset as it is now; BL-0002's fingerprint |

Timeline, UTC:

| When | Event | Status |
| --- | --- | --- |
| 2026-06-19 | Manifest first lists `meta_clf.pkl`, as A (`89ea5a3`, `models-v1.14.0`) | VERIFIED |
| 2026-08-10 03:54 | Commit `1cc4036`; tags `v1.18.0` and `models-v1.18.0` both point to it; its manifest lists A | VERIFIED |
| 2026-08-10 03:59 | GitHub release `models-v1.18.0` published | VERIFIED |
| 2026-08-10 04:38 | PyPI upload of `fie_sdk-1.18.0-py3-none-any.whl` (`be504dc7…579e963d`), containing A | VERIFIED |
| 2026-08-10 05:29 | `model.onnx`, `tokenizer.json` added to the release; commit `d1565d1`, manifest still A | VERIFIED |
| 2026-08-11 18:03 | The release's `meta_clf.pkl` and `meta_clf.json` assets carry this creation time; the pickle's digest is B | VERIFIED |
| 2026-08-11 18:07 | Commit `08d5474`, "fix+measure: meta-classifier read 6 of 11 features as constant zero": manifest A → B, `release_tag` unchanged; also changes `fie/adversarial.py`, the training script and the golden test data | VERIFIED |
| 2026-10-10 | BL-0002 pinned (`924c70d`); its fingerprint records `meta_classifier` = B, `in_manifest: true` | VERIFIED |

**VERIFIED.** The published wheel's file is A, and A is what the manifest listed at the
wheel's release tag. The release was changed in place after publication: the tag
`models-v1.18.0` still points to a commit whose manifest lists A, while the release's asset is
now B. B is the manifest's file, the working tree's file, the release's file and the file the
canonical baseline was measured with.

**INFERRED.** That the asset replaced on 2026-08-11 was A (it no longer exists). That A is the
model "before the fix" of `08d5474` (the side-cars are consistent with that; the pickles were
not opened). That the PyPI wheel was built by the publish workflow (its content is what that
workflow would produce; no build log was examined).

**UNKNOWN.** Whether A or B is safe to deserialize. What the published package does with A at
run time. What the published source distribution contains.

The recommendation in the plan, that B is the intended file, is supported by four independent
VERIFIED records (manifest at HEAD, release asset, working tree, BL-0002 fingerprint) and
contradicted by none. It is a statement about which file the project intends to ship. It is
not a finding that B is correct as a model: the baseline's detection numbers are the evidence
for that, and they are a separate matter.

A consequence worth recording: checking out tag `v1.18.0` and running
`scripts/download_models.py --strict` would now fail on `meta_clf.pkl`, because that tag's
manifest expects A and the release serves B. **INFERRED** from the two digests; not run.

---

## F. Evidence written

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_0/`

| Path | Content |
| --- | --- |
| `ci/ci_runs_summary.json` | Both runs: metadata, every job's conclusion, the `package` job's steps, runner image, the wheel digest from three sources, per-target criteria and installed versions, artifact listing, re-verification result, the gate's log lines |
| `ci/ci_run_38282217538/`, `ci/ci_run_38282733913/` | The uploaded `report.json` byte for byte; the artifact's own checksum list; run metadata; artifact listing |
| `provenance/repository_state.json` | Section A |
| `provenance/model_inventory.json` | Section C, with full digests |
| `provenance/release_models-v1.18.0.json` | The release's 25 assets: size, digest, creation and update times |
| `provenance/seven_unmanifested_files.json` | Section D: digests, blob identities, path history, side-car metadata, every tracked reference, how each is loaded |
| `provenance/meta_clf_provenance.json` | Section E: candidates, where each is found, manifest history, tags, PyPI upload, baseline identity; facts, inferences and unknowns listed separately |
| `COMMANDS.md` | The exact commands and sources, and what was not done |
| `SHA256SUMS.txt` | Checksums of all of the above |

No earlier report or evidence file was edited. No file contains the user name or home path.

---

## G. Unresolved questions

1. The origin (data, code revision, author) of each of the seven experiment files.
2. Whether any model file, listed or not, is safe to deserialize. Integrity work in Phase 3
   establishes that a file is the one named in the inventory, not that it is benign.
3. The contents of the published source distribution.
4. Whether `scripts/eval_harmaug.py` still runs; it was read, not executed.
5. The four unlisted `minilm-onnx/*.json` files: whether the encoder needs them, how they
   reached this directory, and whether they belong in the inventory. Not traced in this stage.
6. Starlette versions between 0.52.1 and 1.7.0.
7. Whether a release can be made immutable; the models release was changed in place once.

---

## H. Owner decisions required before Stage 3.1

### W21-7 — which `meta_clf.pkl` is intended

| | |
| --- | --- |
| Evidence for B (`be6673d0…`) | Manifest at HEAD; current release asset; working tree; BL-0002's fingerprint (`in_manifest: true`). All VERIFIED, section E |
| Evidence for A (`01be1995…`) | It is what PyPI 1.18.0 ships, and what the manifest listed at that release's tag |
| What choosing B means | The corrected release ships B. The inventory keeps B's digest; nothing is edited toward A. PyPI 1.18.0 is documented as carrying A, the superseded file |
| What choosing A would mean | Reverting the manifest, the release asset and the baseline to the pre-`08d5474` model, and re-pinning the baseline |
| Not decided by this | Whether either file is safe to load; whether 1.18.0 should be yanked (W21-6) |
| Recommendation | **B.** Not acted on |

### W21-20 — the seven experiment files

| | |
| --- | --- |
| Today | They sit in `fie/models/`. The current build rule (`fie/models/*.pkl`, `*.json`) packs every one of them into any wheel built from this directory. The `dist/` wheel shows it has happened: it contains four. CI is unaffected, because a CI checkout does not have them |
| After Phase 3, if they stay | The loader refuses them (unlisted). The build fails, naming them, whenever it runs from this directory. A wheel could then be built only from a clean export |
| After Phase 3, if the owner moves them out of `fie/models/` | Local builds from this directory work again. Nothing unlisted is beside the trusted files |
| Research-script impact of moving | `scripts/eval_harmaug.py` reads `fie/models/` by default and would exit with "missing" for the two `v64` defaults and the `v65` example until its directory is pointed elsewhere. `scripts/train_pair_harmaug.py` and the `retrain_pair_*` scripts would keep writing new files into `fie/models/`, recreating the situation, until their output directory changes. Both are script edits outside this stage |
| SDK impact | `FIE_PAIR_VERSION=v2` stops working once `v2` is unlisted, wherever the file is. The other six were never selectable |
| What I did | Nothing. Not moved, deleted, renamed, modified, loaded or added to any inventory |
| Recommendation | The owner moves the seven classifiers and their seven side-cars to a directory outside the package, and decides separately whether the research scripts follow. Not acted on |

Neither recommendation is an authorization. Stage 3.1 does not start until the owner confirms
both.

## I. Recommended manual commit command

Not executed by the assistant.

```text
git add docs/fie_rebuild_2026 && git commit -m "docs: WP-021 phase 3 stage 3.0 - CI evidence and model provenance"
```
