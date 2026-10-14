# EXECUTION_021, Phase 3, Stage 3.2 — Verified model loading (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.2**: production model-loader integration and verified deserialization |
| Status | **COMPLETE**, with one worktree-only test failure that clears on commit (section F) |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, published or deployed |
| Date | 2026-10-14 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `1443cc7af474cdd0e24beec2329ccdd637530114`, unchanged |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged; not rewritten |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_2/` |

Labels: **measured** (run here and recorded in the evidence), **inferred** (reasoned, not run),
**unverified** (not established).

The stage ran in two sessions. The first stopped at the Step 3 gate on an existing-test conflict
and was then audited (verdict: partially complete). The owner chose option B, authorized two
retired Stage 3.1 tests to be updated, and authorized the remaining work. This report covers the
whole stage.

---

## A. Preflight (measured)

| Check | Result |
| --- | --- |
| Branch, HEAD, tree | `rebuild/wp-001-eval-harness`, `1443cc7`, clean, nothing staged |
| Stage 3.1 completion commit | Present (`1443cc7`) |
| CI for it | Run `38447572967` (pull request, `1443cc7`): success |
| Earlier evidence | All nine WP-021 checksum sets verify, before and after this stage |
| Tracked inventory | Passes `validate_production`, 25 entries |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |
| Approved meta-classifier digest | `be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6` in the tracked inventory |

## B. Owner decisions implemented

| # | Decision | Implementation |
| --- | --- | --- |
| W21-9 | Integrity failures reuse `unavailable_load_failed`, with an internal detail code | `LoadRefused.reason` is the public reason; `.detail` is internal. `LAYER_STATES` unchanged |
| W21-17 | No silent fallback for `FIE_PAIR_VERSION` | Empty or unset selects the manifest default. A listed key with a missing file gives `unavailable_model` / `missing`; an unlisted key gives `unavailable_load_failed` / `not_listed` |
| W21-18 | Repository-root `models/` fallback removed | `_resolve_models_dir` always returns the package directory; `_REPO_MODELS` is gone |
| W21-24 | Trusted runtime manifest is `<package>/models/MANIFEST.json` | `_verified_models.MANIFEST_PATH`, read nowhere else |
| W21-25 | Missing or invalid side-car makes the layer unavailable | Threshold, `embed_model` and `layer_names` are required from verified bytes; no default |
| W21-26 | Sanitized structured log and `health()` detail; no new audit event | Log lines carry `reason=`, `detail=` and a manifest-relative `artifact=`; `health()`'s existing `error` text carries the detail code. No new key, no new event type |
| W21-27 | Missing manifest gives `unavailable_model`; invalid gives `unavailable_load_failed` / `manifest_invalid` | As specified. The meta-classifier reports `unavailable` and records the same detail internally |
| Option B | Health-deep precondition updated, no absolute path reintroduced | Section F |
| Test updates | The two retired Stage 3.1 tests are replaced by stronger ones | Section F |

W21-7, W21-8, W21-15, W21-20, W21-22 and W21-23 stand. The manifest, digests, default version,
thresholds and scoring are unchanged.

## C. Step 0: the runtime manifest (measured)

1. `python -I scripts/download_models.py --check`: exit 0, `verified 25/25 artifacts (0 missing,
   0 unreachable, 0 corrupt)`. It wrote nothing.
2. The source was read first: a regular run fetches only artifacts that are missing or fail
   verification. So the regular run was then made once: exit 0, `verified 25/25`, no
   `downloading` line, `wrote fie/models/MANIFEST.json`.
3. The generated file is **byte-identical to a fresh derivation** of `scripts/model_manifest.json`:
   5,112 bytes, SHA-256 `fbab28ab75acad2cfa8d178617caad112c32adafc6048d2a8483cd4a393bf2f5`. It is
   git-ignored (`.gitignore:18`) and not committed. It is a derived local file, not a second
   source of truth. A test now checks the derivation equality on every run.

No download happened, and `--strict` was not used.

---

## D. Files changed

| File | Purpose |
| --- | --- |
| `fie/layers/_verified_models.py` (new) | The only model deserializer in `fie/`. Reads the package manifest once; exact selection; verified reads; `unpickle` from an in-memory buffer of verified bytes; side-car parsing from verified bytes; required side-car fields; `LoadRefused` with a public reason and an internal detail |
| `fie/layers/pair.py` | Both loaders go through the module. Side-cars are verified before use. Identity is computed from the deserialized bytes. Root fallback, `_file_digest`, the fixed version list and two unused imports removed. New internal detail codes and accessors `_pair_unavailable_detail_code()` and `_meta_unavailable_detail()`. Public functions, locks and the readiness log format are unchanged |
| `fie/adversarial.py` | `warmup()` classifies by detail code (`_WARMUP_UNAVAILABLE_DETAILS`) instead of matching message text, and logs the code on failure |
| `fie/_model_integrity.py` | Docstring only: the obsolete "the loader does not use this module yet" corrected. No code changed |
| `tests/test_verified_loading.py` (new) | Section G |
| `tests/test_scan_result_contract.py` | One precondition (option B) |
| `tests/test_model_integrity.py` | One test replaced (authorized) |
| `tests/test_model_manifest.py` | One test rewritten (authorized); its now-unused `import re` removed |
| This report, master-log section 14.8, evidence directory | Documentation |

Not changed (verified with git): `scripts/model_manifest.json`, `scripts/download_models.py`,
`scripts/release_check.py`, `pyproject.toml`, `.github/`, `fie/onnx_encoder.py`,
`fie/session_tracker.py`, `fie/integrations/`, `app/`, `evals/baselines/`.

## E. How a model is loaded now

```
pair._load_pair_classifier_locked / pair._load_meta_classifier   (inside the existing locks)
  import joblib                          a missing package is reported as a dependency, first
  _vm.package_manifest()                 <fie>/models/MANIFEST.json only; parsed once; outcome cached
  _vm.select_pair(requested or None)     the exact key, or the manifest default
  _vm.select_meta()                      the single meta_classifier + meta_classifier_meta (ambiguity refused)
  _vm.read_artifact(classifier)          one open; size and SHA-256 checked; nothing deserialized
  _vm.read_artifact(side-car) → parse_json    only verified bytes are parsed
  required_threshold / required_text / required_names    no defaults
  _vm.unpickle(entry, data)              checks the bytes again, then joblib.load(io.BytesIO(data))
  identity digest = sha256(data)[:16]    the same bytes that were deserialized
  publish all-or-nothing
  pair only: _build_embedder(embed_model from the verified side-car)
```

The classifier is read and checked before its side-car, so a planted file is classified by its
own content. Both are verified before anything is deserialized or any field is used.

**Failure outcomes:**

| Condition | Public state (pair) | Internal detail | Log level | `warmup()` |
| --- | --- | --- | --- | --- |
| `joblib` (or a package the model needs) not importable | `unavailable_dependency` | `dependency` | WARNING | unavailable |
| Runtime manifest absent | `unavailable_model` | `manifest_missing` | WARNING | unavailable |
| Listed file (classifier or side-car) absent | `unavailable_model` | `missing` | WARNING | unavailable |
| Manifest invalid, or meta role ambiguous | `unavailable_load_failed` | `manifest_invalid` | ERROR | failed |
| Requested version not listed | `unavailable_load_failed` | `not_listed` | ERROR | failed |
| Size or SHA-256 differs | `unavailable_load_failed` | `mismatch` | ERROR | failed |
| OS refuses to read | `unavailable_load_failed` | `read_failed` | ERROR | failed |
| Not a regular file, or outside the directory | `unavailable_load_failed` | `path_violation` | ERROR | failed |
| Verified bytes do not deserialize, or a side-car field is missing or invalid | `unavailable_load_failed` | `deserialize_failed` (logged with traceback) | ERROR | failed |
| Any other exception (a defect, or no encoder) | `unavailable_load_failed` (as before) | `unexpected` (logged with traceback) | ERROR | failed |

The meta-classifier is optional: in each case it reports `unavailable` and records the same
detail through `_meta_unavailable_detail()` and its WARNING line. No message contains an absolute
path; artifacts are named by manifest-relative path.

## F. The four failures from the audit (measured)

| Test | Final disposition |
| --- | --- |
| `test_runtime_policy.py::test_detector_source_files_are_untouched` | **Still fails, worktree-only. Not modified.** It asserts that `git status --porcelain` over the detector paths is empty. It lists exactly `M fie/adversarial.py`, `M fie/layers/pair.py` and `?? fie/layers/_verified_models.py`, the three authorized source files, and nothing else. It compares the working tree with HEAD, so once these files are committed it compares HEAD with itself. **Recheck after the commit:** `python -m pytest "tests/test_runtime_policy.py::test_detector_source_files_are_untouched" -p no:cacheprovider` |
| `test_scan_result_contract.py::test_health_deep_reports_a_reason_code_not_the_loader_error[model]` | **Passes.** Option B: the precondition no longer requires an absolute path. For `model` and `load_failed` it now requires the internal error to start with the fixed detail code (`missing:` or `mismatch:`), the detail accessor to agree, and the temporary location, its name and both fixture directory names to be absent. Every `/health/deep` leak assertion is unchanged |
| `test_model_integrity.py::test_the_loader_does_not_use_the_module_yet` | **Replaced** by `test_the_loader_uses_the_module_through_one_door_only`, which passes. It checks two things. First, exactly one file in `fie/` refers to the integrity module, both by text and by import (`_verified_models.py`), and `pair.py` imports that file. Second, in a child process, importing the loader loads the module but attempts no model load and reads no manifest, and the manifest path is `fie/models/MANIFEST.json` |
| `test_model_manifest.py::test_tracked_selection_keys_agree_with_the_loader` | **Rewritten**, passes. It reads the keys and files from `scripts/model_manifest.json`, derives the package manifest, and runs the loader's own `select_pair` on it. Each key selects exactly its classifier and side-car; no key selects the default; the seven experiment names and other strings are refused as `not_listed`. It also checks that `pair.py` calls `_vm.package_manifest`, `_vm.select_pair` and `_vm.select_meta`. It no longer parses source text with a regex |

## G. The verified-loading tests (`tests/test_verified_loading.py`, measured: 91 passed, 0 failed, 0 skipped)

Fixtures are synthetic: temporary model directories, manifests written by the test, and
test-authored pickles. No released model is deserialized by this file; section 5 only hashes the
released files.

| Area | Tests |
| --- | --- |
| Canary | Positive controls: the canary fires when deserialized directly and through `unpickle`. Every refusal test asserts it did not fire |
| Module | Manifest missing, malformed, wrong schema, not UTF-8, duplicate default, oversized, unreadable; outcome cached; unknown detail code rejected; `unpickle` and `parse_json` re-check their bytes; non-pickle gives `deserialize_failed`; `ImportError` propagates; required fields never defaulted; exact selection; side-car must be listed; meta ambiguity refused |
| Meta-classifier | Loads from verified bytes with identity from those bytes. Missing or invalid manifest, ambiguity, tampered pickle, modified, missing or incomplete side-car all leave it unavailable with sentinel threshold and features untouched. Dependency detail; sanitized log; one attempt per process |
| Pair classifier | Default for unset, empty and whitespace values, with the exact readiness log line. All seven keys select exactly their files (audit hook: the selected pair opened once, every other version zero times). `v2` refused although its file exists. Removed `v5` not replaced by the default. Tampered classifier refused before deserialization. Unlisted files never opened. Deserializer receives a buffer whose hash is the manifest digest, from one open. File swapped after verification changes nothing. Altered threshold or `embed_model` refused, encoder never built. Missing side-car not defaulted. Required side-car fields. Parsed bytes are verified bytes. Nine failure categories each with their state, detail, log level and no location. A programming error reported as `unexpected`. Concurrent first loads deserialize once |
| Boundary (new) | AST scan of every `fie/*.py`, following import aliases, with a positive control. The only deserializing calls are `joblib.load` in `_verified_models.py` (inside `unpickle` only) and the known `pickle.loads` in `session_tracker.py`, which is pinned separately as out of scope |
| Real artifacts (new) | The runtime manifest is byte-identical to the derivation of the tracked inventory. The 16 pair and meta artifacts in `fie/models/` pass `read_artifact` against entries equal to the tracked ones. The meta digest equals the approved digest; the default is `v6_3b`. A missing runtime manifest **fails** the test with instructions; it never skips |
| Repository-root isolation (new) | `_REPO_MODELS` is gone and `_resolve_models_dir` returns the package directory for any sentinel. In a synthetic checkout, canaries under every name at `<root>/models` are never opened. A root copy carrying the exact listed bytes of a classifier missing from the package does not load: the result is `unavailable_model` / `missing` |
| `warmup()` (new) | Nine cases: ready; `manifest_missing`, `missing` and `dependency` give unavailable; `mismatch`, `not_listed`, `manifest_invalid`, `deserialize_failed` and `unexpected` give failed, with the logged detail. A structural check confirms `warmup()` no longer reads `_pair_state()` or any message text |
| Contracts (new) | Through the real scanner, a tampered classifier or side-car gives `unavailable_load_failed` in `coverage.layers` and `coverage.classifier`, `partial` status, `degraded_layers == ["pair_classifier"]`, identity `loaded=False` with no digest, and nothing internal in the public result. `preflight_check` with STRICT blocks; with compat it allows, with `missing_required_layer` |

## H. Results (measured, final tree)

| Command | Result |
| --- | --- |
| `python -m pytest tests/test_verified_loading.py -p no:cacheprovider` | 91 passed, 0 failed, 0 skipped |
| Affected suites (integrity, manifest, runtime policy, scan-result contract, golden, SDK, harness subject contract) | All pass except the worktree-only guard |
| `python -m pytest tests/ -m "not network" -p no:cacheprovider` | **1420 passed, 1 failed (the worktree-only guard, section F), 2 skipped** |
| Before Stage 3.2, same runtime manifest present | 1330 passed, 0 failed, 2 skipped |

The two skips are platform cases in `test_model_integrity.py`: symbolic links cannot be created,
and file modes do not deny reading on Windows.

**Canonical baseline (measured).** `python -m evals baseline` exit 0, run
`20261014T062737Z_f0212fe0_4df7b578`, `canonical=False`:

- Standard counts match the approved counts.
- **11 of 11 per-prompt record files are byte-identical** to `BL-0002_pair-v6.3b_fie-70de7053`,
  checked independently by SHA-256 (`baseline.json`).
- The fingerprint differs only in identity metadata: `git/commit`, `git/dirty_other`,
  `git/dirty_subject`, `subject/fie_py_files`, `subject/fie_tree_sha256` and `keys/subject_key`.
  Model identities are identical.

This shows that deserializing from verified in-memory bytes gives the same detector behaviour as
loading by path. The golden confidence tests agree. The baseline was not rewritten and
`CANONICAL` was not moved.

## I. Scope

- **Not done:** no download, commit, push, merge, tag, publish or deploy; no wheel or source
  distribution built; no dependency installed; no Stage 3.3 or 3.5 work.
- **Experiment files:** the seven files and their side-cars were not moved, edited or listed.
  Their sizes, SHA-256 digests and modification dates are recorded in `repository_state.json`
  for future comparison. The dates all precede this stage.
- **Deserialization:** this stage's new tests deserialize only test-authored pickles. The
  existing contract and golden suites and the baseline deserialize the released models, as
  before, now through the verified path.

## J. Limitations and open work

1. **Worktree-only guard** (section F): clears on commit; recheck then.
2. **CI not yet run on this change** (unverified). In CI the test jobs generate the runtime
   manifest through their download step. The real-artifact tests fail, rather than skip, in any
   job where it is absent.
3. **Developer checkouts** without `fie/models/MANIFEST.json` now report both models unavailable
   until `python scripts/download_models.py` has been run once. This is fail-closed and logged.
4. **A locally retrained model no longer loads** until the inventory is updated in a reviewed
   change, because its digest no longer matches. This is intended.
5. **Not covered by this stage:** the `SentenceTransformer` fallback can still download and
   deserialize weights (its model name now comes only from a verified side-car); ONNX encoder
   files are not verified; `session_tracker.py` unpickles session data; the server's own model
   loading is separate. These belong to WP-022 or are separately scoped.
6. **Meta-classifier detail** is not exposed through `health()`, because existing tests pin the
   key sets of `_meta_state()` and `/health/deep`. It is in the log and in
   `_meta_unavailable_detail()`.
7. **Stage 3.3 is still pending:** `in_wheel` enforcement, `.gitkeep` exclusion, sdist
   feasibility. Also pending: the research-workflow plan and the relocation of the experiment
   files (W21-20).

## K. Rollback

Each change is local and uncommitted. Reverting the three source files and the three edited test
files, and deleting the two new files, restores the Stage 3.1 state exactly. No manifest,
artifact or baseline was changed, so nothing else needs restoring. The generated
`fie/models/MANIFEST.json` is ignored and harmless to the Stage 3.1 code.

## L. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_2/`, checksums in `SHA256SUMS.txt`:

| File | Content |
| --- | --- |
| `repository_state.json` | Branch, HEAD, changed and new files with LF-normalized digests, protected paths, `CANONICAL`, experiment-file record |
| `runtime_manifest.json` | Step 0 outputs; generated manifest digest; equality with the derivation |
| `deserializer_boundary.json` | AST scan result over `fie/`, call graph, out-of-scope loaders |
| `test_results.json` | Focused and full-suite counts, test names, skips, the four failures' dispositions |
| `baseline.json` | Console output, per-record digests and equality, differing fingerprint fields |

```text
66d33342bcb1548d603d916ae967890e67652c6e0f2db69b8e0bbf3b5e765e8c  baseline.json
43d59d67bd55c6e8399eddfa08684483db53c4e1cc00008d80ccfc7e71077a85  deserializer_boundary.json
cbd6159842aeffa828e1b4168af12d35b7d01d6a2f09050a2b53799189a2650f  repository_state.json
76e3dfbf44c98f24fcbe4a29f32f939c2adc358b5c6614654eaa488ece27c9b6  runtime_manifest.json
d4e24fe7b28b5ed60871394a622e3ee53f6aa54330be37249c6fcfcb1c84b22f  test_results.json
```
