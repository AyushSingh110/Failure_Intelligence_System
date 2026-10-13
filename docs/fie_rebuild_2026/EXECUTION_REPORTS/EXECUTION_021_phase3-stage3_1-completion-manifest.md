# EXECUTION_021, Phase 3, Stage 3.1 completion — Production manifest, inventory validation, downloader (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.1, remaining work** |
| Status | **COMPLETE.** Stage 3.1 is now complete as a whole. The model loader is not connected; that is Stage 3.2 and is not started |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, published or deployed |
| Date | 2026-10-13 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `d7587542ff2b9ffe520db377f04b558cc201de37`, unchanged |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1_completion/` |

No model file was opened by anything in this stage. The tracked inventory was read as JSON.
Nothing was downloaded, built, installed, moved or deleted.

---

## A. Preflight

| Check | Result |
| --- | --- |
| Branch, HEAD, tree | `rebuild/wp-001-eval-harness`, `d758754`, clean, nothing staged |
| Committed | The Stage 3.1 integrity module (`5bac8cd`) and the I/O-error follow-up (`d758754`) |
| Linux CI for the follow-up | Run `38435849880` (pull request, `d758754`): success; `test (3.10)`, `(3.11)`, `(3.12)` each 1178 passed, 1 skipped; `package` success. Run `38436436575` (push to `main`, `b2b3c66`): now completed, success, all nine jobs including `deploy-space` |
| Earlier evidence | All eight WP-021 checksum sets verify |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |
| `meta_clf.pkl` | Manifest and working-tree file both `be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6` |

## B. Owner decisions implemented

| # | Decision | Where it is implemented |
| --- | --- | --- |
| W21-7 | Intended `meta_clf.pkl` is `be6673d0…f5cbe9a6` | Digest unchanged in the inventory; a test pins it |
| W21-8 | Supported SDK models ship in the one wheel | `in_wheel: true` on 16 entries; the validator requires it for the four SDK roles |
| W21-15 | The two `fie/models/` v4 files stay listed, `in_wheel: false` | Role `legacy`; the validator ties that role to exactly those two paths and forbids `in_wheel: true` |
| W21-22 | Five encoder by-products are outside the inventory | Not added; a test asserts none is listed |
| W21-23 | Schema extended in place; one authoritative inventory; runtime manifest derived | Sections D and E |
| — | Roles `server` and `legacy` | In the role vocabulary |
| — | `version` is the `FIE_PAIR_VERSION` selection key | Validator maps each key to its file; a test compares it with the loader's source |
| — | Singleton roles selected by role only, no invented version | `sole_entry`; the validator rejects a version on those roles |
| — | All seven classifier versions ship | The validator requires exactly the seven |
| W21-20 | Experiment files to move out of `fie/models/` | **Not executed**, as instructed. None was moved or listed |
| — | `.gitkeep` excluded from the wheel | **Not implemented**, as instructed; reserved for Stage 3.3 |

These decisions say which files the project ships. They do not establish that any pickle is
safe to deserialize.

---

## C. Files changed

| File | Change |
| --- | --- |
| `fie/_model_integrity.py` | `sole_entry`, `validate_production`, `derive_package_manifest`, and the constants they use |
| `scripts/model_manifest.json` | `schema: 1`; `role`, `version` (where applicable), `default`, `in_wheel` on every entry. 90 lines added, 0 deleted |
| `scripts/download_models.py` | Writes the derived manifest after a fully verified run |
| `tests/test_model_manifest.py` | New, 153 tests |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_1-completion-manifest.md` | This report |
| `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Section 14.7 |
| `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1_completion/` | New |

Not changed: `tests/test_model_integrity.py` and every other existing test; `fie/layers/pair.py`;
`fie/onnx_encoder.py`; `pyproject.toml`; workflows; Dockerfiles and deploy scripts; `.gitignore`;
`.gitattributes`; every model, side-car and by-product file; the seven experiment files;
`dist/`; baselines. `fie/models/MANIFEST.json` was not created.

---

## D. The production inventory

**Schema.** Top level: `_comment`, `schema` (1), `release_tag`, `repo`, `artifacts`. The
existing keys keep their values. Per entry: `path`, `sha256`, `size`, `role`, `version` (pair
roles only), `default`, `in_wheel`, `used_by`.

**What did not change** (`manifest_change.json`): all 25 entries are present in the same
order, and every `path`, `sha256`, `size` and `used_by` is identical to the committed file;
`release_tag`, `repo` and `_comment` are identical.

**Entries by role.**

| Role | Entries | `version` | `in_wheel` | Root |
| --- | --- | --- | --- | --- |
| `pair_classifier` | 7 | `v6_3b`, `v6_3`, `v6`, `v5`, `v4`, `v3`, `v1` | true | `fie/models/` |
| `pair_meta` | 7 | the same keys | true | `fie/models/` |
| `meta_classifier` | 1: `meta_clf.pkl` | none | true | `fie/models/` |
| `meta_classifier_meta` | 1: `meta_clf.json` | none | true | `fie/models/` |
| `encoder` | 1: `minilm-onnx/model.onnx` | none | false | `fie/models/` |
| `tokenizer` | 1: `minilm-onnx/tokenizer.json` | none | false | `fie/models/` |
| `legacy` | 2: `failure_classifier_v4.pkl`, `feature_columns_v4.pkl` | none | false | `fie/models/` |
| `server` | 5 | none | false | `models/` (3), `storage/` (2) |

`default: true` on exactly one entry, `fie/models/pair_intent_classifier_v6_3b.pkl`. `v1` is
the original `pair_intent_classifier.pkl`. 16 entries ship in the wheel; 9 do not.

**`validate_production(source)`.** Takes the inventory as text, bytes or a decoded object and
returns the parsed manifest, or raises `ManifestError` (`manifest_invalid`) naming the first
rule broken. It reads no file and repairs nothing. Rules, beyond those `parse_manifest` already
enforces (schema 1, field types, digest format, path safety, unique paths, unique
role/version pairs):

- `default` is stated on every entry.
- `role` is one of the eight.
- Every path is under `fie/models/`, `models/` or `storage/`.
- `server` entries are under `models/` or `storage/`; every other role is under `fie/models/`.
- `legacy` is exactly the two approved paths, in both directions: nothing else may be
  `legacy`, and those two may be nothing else.
- `in_wheel` is fixed by role: true for the four SDK roles, false for `encoder`, `tokenizer`,
  `server`, `legacy`. So a legacy or server file cannot be promoted into the wheel, and an SDK
  model cannot be dropped from it, by editing the inventory alone.
- A pair entry's `version` is one of the seven keys and sits on that key's own file.
- No other role carries a `version`.
- Only the `v6_3b` `pair_classifier` may be the default, and exactly one entry is.
- The classifier versions are exactly the seven, and each has its `pair_meta`.
- Each singleton role has exactly one entry.
- The derived manifest's own path cannot be listed.

The rules are written into the module on purpose. Changing what ships, or the default, now
needs a change to this file as well as to the inventory, in a reviewed commit.

**`sole_entry(manifest, role)`.** The one entry of a role. None: `NotListed`. Several:
`ManifestError` naming them. It never returns the first.

---

## E. The derived manifest

`derive_package_manifest(source) -> bytes`

| | |
| --- | --- |
| Input | The tracked inventory, and nothing else. No file system, clock or environment |
| Step 1 | `validate_production`; an invalid inventory derives nothing |
| Kept | Entries under `fie/models/`, with that prefix removed: 20 of the 25 |
| Left out | The five `server` entries |
| Retained although not shipped | The four `in_wheel: false` entries (encoder, tokenizer, two legacy) |
| Fields | `path`, `sha256`, `size`, `role`, `default`, `in_wheel`, and `version` on pair entries. `used_by`, `release_tag`, `repo` and `_comment` are dropped |
| Form | `{"schema": 1, "artifacts": [...]}`; sorted by path; sorted keys; two-space indentation; ASCII; LF; one final newline |
| Self-reference | Impossible: the validator rejects an inventory that lists `fie/models/MANIFEST.json` |

For today's inventory the output is 5,112 bytes, SHA-256
`fbab28ab75acad2cfa8d178617caad112c32adafc6048d2a8483cd4a393bf2f5`, identical across repeated
calls and independent of the source's entry order, key order, indentation and line endings. It
parses with `parse_manifest` and works with `default_entry`, `select_entry`, `sole_entry` and
`read_verified`. A copy is kept as evidence (`derived_package_manifest.json`).

**`fie/models/MANIFEST.json` itself does not exist yet.** It is written only by a successful
download run, and none was made. `.gitignore` line 18 already ignores it; a test asserts it is
not tracked.

---

## F. The download script

| | Before | After |
| --- | --- | --- |
| Which entries | Every entry of the inventory | Unchanged. `in_wheel`, `role` and the other new fields are not consulted |
| Verification | SHA-256 against the inventory; a corrupt download is deleted | Unchanged |
| Release selection | `--tag`, `FIE_MODELS_RELEASE`, else `release_tag` | Unchanged |
| Exit codes | 1 on a digest mismatch; 1 on `--check` with a missing file; 1 on unreachable with `--strict`; otherwise 0 | Unchanged, plus: 1 if the package manifest cannot be derived (an inventory that breaks the rules) |
| `--check` | Read-only | Read-only. Does not create or replace the derived manifest |
| Derived manifest | — | Written once, atomically, **only when every one of the entries has been verified** in a run that is not `--check` |
| Partial verification | — | A missing, unreachable or corrupt artifact means no manifest is written and an existing one is left as it was. This includes the lenient non-strict case that still exits 0 |
| Dependencies | Standard library | Standard library. `fie/_model_integrity.py` is loaded by file path; the `fie` package is not imported, since the SDK's dependencies are not installed when this script runs |

Run for real, read-only: `python -I scripts/download_models.py --check` exits 0 with
"verified 25/25 artifacts (0 missing, 0 unreachable, 0 corrupt)", no network, no file written.

Server workflow: both Dockerfiles and the Space deploy run this script with `--strict` and are
untouched. They download and verify the same 25 files as before, and now also get the derived
manifest in the image.

---

## G. Tests

`tests/test_model_manifest.py`, new: **153 passed**. `tests/test_model_integrity.py`,
unmodified: 180 passed, 2 skipped, as before.

| Area | Covered |
| --- | --- |
| Roles | The vocabulary is exactly the eight; six unapproved names rejected |
| Fields | Each required field missing; `default` missing on six kinds of entry; values of the wrong type rejected, not normalised |
| Versions | Exactly the seven keys; `version` required on classifiers and side-cars; eight other values rejected, including the side-cars' declared `v6.2` and `v6.3b` and the unlisted `v2`; a key on the wrong file rejected; `v1` is the original file; a version on any other role rejected |
| Identity | Duplicate role/version and duplicate paths rejected |
| Default | Exactly one, and `v6_3b`; none rejected; a second rejected; another version taking it rejected; five non-classifier roles claiming it rejected |
| Side-cars | A missing version, a classifier without its side-car, and a side-car without its classifier each rejected |
| Singletons | `sole_entry` on each of the four; zero candidates is `NotListed`; several is `ManifestError` and never the first; a missing or doubled singleton rejects the inventory |
| Roots and shipping | Seven paths outside the roots rejected; five unsafe paths rejected; the 16 shipped entries and their roles; six non-shipped entries cannot be marked shipped; five shipped entries cannot be unmarked; `server` and `legacy` placement; a legacy file cannot take another role or enter the wheel |
| Derivation | Exactly the 20 package entries; server entries excluded; non-shipped entries retained; free text and top-level metadata removed; values carried unchanged; byte-identical across calls and across source order, key order and line endings; stable serialisation; nothing environmental; never lists itself; valid for the runtime helpers; verifies real bytes in a temporary package directory and rejects a tampered one; an invalid inventory derives nothing |
| The tracked file | Passes the rules; keeps its top-level keys; the approved `meta_clf.pkl` digest; roles as decided; the 16 wheel entries; none of the seven experiment files or five by-products listed; **selection keys agree with the loader's own source**; still readable by the harness's reader; nothing generated is tracked |
| Downloader | Downloads and verifies all 25 whatever `in_wheel` says; release selection unchanged; manifest written after success and equal to a fresh derivation; a second run downloads nothing and rewrites the same bytes; `--check` read-only, and does not overwrite an existing manifest; `--check` with a missing or wrong-digest file exits 1; a stale file is replaced; a wrong-digest download exits 1 for four kinds of entry and writes nothing; an unreachable artifact exits 1 under `--strict` and 0 otherwise, writing nothing either way; a failed or corrupt run does not replace an existing manifest; the manifest does not exist while any of the 25 downloads is in progress; server entries still strictly verified; an inventory that breaks the rules fails the run; no temporary file left; the script does not import `fie` |

The downloader tests run the real script against a temporary tree with a synthetic inventory
and a replacement download function; any call to the real network function fails the test.

One defect of my own in the first version of the tests, not in the code: a helper's parameter
name collided with a field name. Renamed. No assertion was weakened.

| Command | Exit | Result |
| --- | --- | --- |
| `python -m pytest tests/test_model_manifest.py tests/test_model_integrity.py -p no:cacheprovider` | 0 | 333 passed, 2 skipped |
| `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` | 0 | **1330 passed, 0 failed, 2 skipped** (1177 before; the same two platform skips) |
| `python -m evals baseline` | 0 | Standard counts match; **all 11 per-prompt record files byte-identical to `BL-0002_pair-v6.3b_fie-70de7053`** |

No existing test was edited. The WP-004 guard test passes with these files uncommitted; none
is on its watched paths.

**Harness fingerprint.** The harness hashes `scripts/model_manifest.json` as part of the
subject it measures, so its identity fields change with this edit; the run directory name
shows that. Compared field by field with the pinned baseline's fingerprint, the run differs in six fields, all identity metadata: `git/commit`, `git/dirty_subject`, `git/dirty_other`, `subject/fie_tree_sha256`, `subject/fie_py_files` (35 to 37) and the derived `subject_key`. The model identities are identical, and the per-prompt records are byte-identical. So the change is to what the harness says it measured, not to any measurement. The baseline was not rewritten and `CANONICAL` was not moved.

---

## H. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1_completion/`

| File | Content |
| --- | --- |
| `manifest_change.json` | Before and after: keys, counts, the proof that no path, digest, size or note changed, every entry's new fields, entries by role, the 16 wheel entries |
| `derivation.json` | The derivation's contract and today's output: digest, size, counts, determinism |
| `derived_package_manifest.json` | The derived bytes, as evidence. Not the generated file, which was not written |
| `test_results.json` | Commands, exit codes, counts, skips, the 153 new tests by name, file digests |
| `SHA256SUMS.txt` | Checksums of the above |

All eight earlier evidence sets are unchanged and still verify.

---

## I. Unresolved decisions and limitations

| # | Point |
| --- | --- |
| 1 | **The loader does not use any of this.** Models are still loaded by `joblib.load` on the first preferred file that exists. Until Stage 3.2, the inventory and the derived manifest protect nothing at run time |
| 2 | W21-9 (how a refused model is reported), W21-17 (an unavailable `FIE_PAIR_VERSION`), W21-18 (the fallback directory) are needed for Stage 3.2. W21-16 (build strictness) and W21-19 (line endings) for Stage 3.3 |
| 3 | W21-20 is approved and not executed: the seven experiment files are still in `fie/models/`. The research-workflow compatibility plan has not been written |
| 4 | No build rule enforces `in_wheel` yet. Today's build still packs whatever `fie/models/*.pkl` and `*.json` exist, including the two `legacy` files and `.gitkeep`. That is Stage 3.3 |
| 5a | Every test in this stage ran with `fie/models/MANIFEST.json` absent. CI's `test` job runs the download first, so there the file will be present. I expect no test to depend on its absence, but that is not verified until CI runs |
| 5 | **A side effect to be aware of.** Once `fie/models/MANIFEST.json` is written by a download run, today's build glob (`fie/models/*.json`) will include it in the wheel. That is the intended end state, but it arrives before the loader reads it and before the release gate checks it. CI's `package` job runs the download, so the next CI wheel will have one more file |
| 6 | The validator pins the role-to-`in_wheel` table, the seven keys and the default. That is deliberate, and it means WP-022's encoder decision, if it puts the encoder in the wheel, needs a change here |
| 7 | The docstring of `download_models.py` describes release asset names as `fie__models__foo.pkl`; the code uses the plain file name. Pre-existing; not touched |
| 8 | Python 3.10.19 on Windows only for this stage. The two platform-dependent tests are skipped here, as before |
| 9 | Integrity is not safety: a verified file is the file the inventory names, nothing more |

## J. Recommended manual commit command

Not executed by the assistant.

```text
git add fie/_model_integrity.py scripts/model_manifest.json scripts/download_models.py tests/test_model_manifest.py docs/fie_rebuild_2026 && git commit -m "fie: production model inventory, validator and derived package manifest; downloader writes it after full verification (WP-021 phase 3 stage 3.1)"
```
