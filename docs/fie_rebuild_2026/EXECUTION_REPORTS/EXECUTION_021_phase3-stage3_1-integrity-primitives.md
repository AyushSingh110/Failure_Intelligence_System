# EXECUTION_021, Phase 3, Stage 3.1 — Model-integrity primitives (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.1 only** |
| Status | **PARTIALLY COMPLETE, WITH A DECISION BLOCKER.** The standalone module and its tests are done and pass. The production manifest and the download script were **not** changed: they cannot be completed without owner decisions that have not been made (section F) |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, published or deployed |
| Date | 2026-10-13 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `25ec566`, unchanged |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1/` |

The model loader is not touched and does not use the new module. No model file was read by
anything in this stage, and nothing was deserialized.

---

## A. Preflight

| Check | Result |
| --- | --- |
| Stage 3.0 committed | Yes: `25ec566` "docs: WP-021 phase 3 stage 3.0 - CI evidence and model provenance"; its report and 16 evidence files are tracked |
| Branch, HEAD, tree | `rebuild/wp-001-eval-harness`, `25ec566`, clean, nothing staged |
| Evidence checksums | All six WP-021 sets verify: 19, 39, 19, 37, 45 and 15 files |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |
| Stage 0 to 3.0 reports and evidence | Present and unmodified |
| Approved `meta_clf.pkl` digest | The tracked manifest lists `be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6`, equal to the owner-approved digest; the working-tree file hashes to the same value |

## B. Owner decisions recorded

| # | Decision | Status |
| --- | --- | --- |
| W21-7 | The intended `meta_clf.pkl` is `be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6` | **Approved by the owner** |
| W21-20 | The seven unmanifested experiment classifiers and their side-cars are to be moved outside `fie/models/` | **Approved by the owner.** Not executed in this stage: the files were not moved, and no research script was changed. Research-workflow compatibility is a separately planned and reviewed step |
| W21-8, W21-9, W21-15, W21-16 to W21-19 | — | Not approved. Nothing in this stage assumes them |

---

## C. Files

| File | Change |
| --- | --- |
| `fie/_model_integrity.py` | New |
| `tests/test_model_integrity.py` | New, 144 tests |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_1-integrity-primitives.md` | New, this report |
| `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Section 14.5 |
| `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1/` | New |

Not changed: `scripts/model_manifest.json`, `scripts/download_models.py`, `fie/layers/pair.py`
and every other source file, every existing test, `pyproject.toml`, workflows, `.gitignore`,
`.gitattributes`, the seven experiment files, `dist/`, baselines.

---

## D. The module

`fie/_model_integrity.py`. Standard library only: `dataclasses`, `hashlib`, `json`, `os`,
`pathlib`, `re`. It imports no deserializer and contains no deserialization call.

| Function | Does |
| --- | --- |
| `parse_manifest(source)` | Validates a manifest given as JSON text, bytes or a decoded object, and returns a `Manifest` of immutable `Entry` values. The whole manifest is checked first; one bad entry rejects all of it |
| `select_entry(manifest, role, version)` | The entry for exactly that role and version, or `NotListed`. Never anything else |
| `default_entry(manifest, role)` | The entry the manifest marks `default` for that role, or `NotListed`. Not the first entry, not the newest |
| `resolve_path(models_dir, entry)` | The file's location inside the directory, links resolved. Refuses a location outside it |
| `verify_bytes(entry, data)` | Raises unless the bytes have the entry's size and SHA-256 |
| `read_verified(models_dir, entry)` | Opens the file once, reads at most one byte more than the listed size, checks size then digest, and returns **those bytes**. Returns nothing on failure |

**Manifest, schema 1.** `{"schema": 1, "artifacts": [...]}`. Per entry: `path`, `sha256`, `size`,
`role`, `in_wheel` required; `version` and `default` optional. Rules enforced:

- `path`: relative, forward slashes, plain segments. No `..`, `.`, empty segment, backslash,
  colon, NUL, leading or trailing slash. Unique, also when only letter case differs.
- `sha256`: exactly 64 lowercase hexadecimal characters.
- `size`: a non-negative integer (not a boolean, not a string).
- `role`, `version`: short tokens. `(role, version)` is unique.
- `in_wheel`, `default`: real booleans.
- At most one default per role; a default must name its version.
- Unknown extra keys (the tracked manifest's `used_by`, for instance) are tolerated and ignored.

**Failures.** Five exception classes under `ModelIntegrityError`, none a subclass of another,
each with a fixed `code`:

| Class | `code` | Meaning |
| --- | --- | --- |
| `ManifestError` | `manifest_invalid` | The manifest cannot be trusted |
| `PathViolation` | `path_violation` | The path leaves the model directory, or is not a regular file |
| `NotListed` | `not_listed` | The role or version is not in the manifest (carries `role`, `version`) |
| `ArtifactMissing` | `missing` | Listed, but the file is not there |
| `IntegrityMismatch` | `mismatch` | The file is there and is not the listed one (carries `what` = `size` or `sha256`, `expected`, `actual`) |

No message contains an absolute path; errors carry the manifest-relative path.

**Deliberately not here.** No deserialization. No mapping of a failure to a detector-layer
state or to a policy action (W21-9 is open; the caller decides). No fallback of any kind. No
decision about where a manifest lives or which directory is trusted: the caller passes both.

**Limit, stated in the module.** This establishes that a file is the one the manifest names.
It does not establish that the file is benign, and it does not defend against someone who can
write to the installation, who can change the manifest as well.

---

## E. Tests

`tests/test_model_integrity.py`: 144 tests, **143 passed, 1 skipped**, exit 0. All fixtures are
synthetic byte strings in a temporary directory. No model file is read.

| Requirement | Covered by |
| --- | --- |
| Valid manifest parsing | Typed entries with exact field values; text, bytes and decoded object parse identically; lookups; entries immutable; extra keys ignored |
| Malformed JSON, wrong shapes | 12 cases, each asserting the message fragment and `code == "manifest_invalid"`: not JSON, empty, not UTF-8, not an object, missing or wrong schema, `artifacts` not a list, an entry that is not an object |
| Invalid entry schema | Each of the five required fields missing; 12 invalid metadata values across `role`, `in_wheel`, `version`, `default`; one bad entry invalidates the whole manifest, naming its index |
| Invalid SHA-256 | 12 forms: empty, short, long, uppercase, non-hex, prefixed `sha256:`, wrong type |
| Invalid size | 7 forms, including booleans, strings, negatives, floats |
| Wrong size and digest on disk | Same-size tampering rejected with `what == "sha256"` and both digests; four wrong sizes rejected with exact expected and actual; a manifest with the wrong digest or size rejects an untouched file |
| Missing file | `ArtifactMissing`, `code == "missing"`, and explicitly not an `IntegrityMismatch`; a missing directory likewise |
| Duplicate paths | Rejected, also when only case differs |
| Unsafe paths | 19 forms rejected at parse time; the same 19 rejected again at resolve time for a hand-built entry; a correct file placed outside the directory is never read; a directory junction leaving the directory is refused; a directory is not an artifact |
| Default from an explicit marker | The marked entry is returned whichever position it is in; each role has its own; a role with no marker has no default |
| Exact version selection | Each listed version returns its own entry |
| Absent version, no substitution | 7 cases raise `NotListed` with the requested role and version; a dedicated test that a missing version never yields the default |
| The exact bytes checked are returned | Returned bytes equal the fixture and hash to the manifest digest; an audit hook counts **exactly one** open of the file; replacing the file afterwards does not change the bytes already returned |
| A changed file is rejected before bytes are returned | The result variable is asserted unassigned after the raise |
| Inconsistent or multiple defaults | Two defaults for a role rejected, naming both; a default without a version rejected; the same version twice for a role rejected; the same version under two roles allowed |
| Distinct failure kinds | Five classes, five codes, none a subclass of another |
| Stage boundary | The module imports only the standard library; contains no `load`, `Unpickler`, `eval`, `exec` or `__import__` call, and `loads` only as `json.loads`; importing it loads no model library; **no file in `fie/` refers to it, and `import fie` does not import it** |

The skip: "a symbolic link leaving the model directory is refused" cannot create a symbolic
link on this Windows machine without elevated rights. The same protection is exercised here by
the directory-junction test, which passed, and the symbolic-link test runs on CI's Linux.
Until it has run there, that one case is unverified on a real symbolic link.

One defect of my own in the first version of the tests, not in the module: a check forbade any
call named `compile`, which matched `re.compile`. Narrowed to the builtin. No assertion about
the module was weakened.

---

## F. Production manifest and download script: blocked

Reviewed before any edit, as instructed. Result: **neither file was changed.**

| Field | Can it be filled without an unapproved decision? |
| --- | --- |
| `path`, `sha256`, `size` | Yes. All three are already on all 25 entries of the tracked manifest |
| `role` | Yes for the files `fie/` loads. Not for `failure_classifier_v4.pkl` and `feature_columns_v4.pkl`, which nothing in `fie/` loads: their role depends on W21-15 |
| `version` | Yes for classifiers: the loader's own names and the side-cars |
| `default` | Yes: `v6_3b` (loader's first choice, the manifest's own note, BL-0002's identity) |
| **`in_wheel`** | **No.** It needs W21-8 for every `fie/models` entry and W21-15 for two. The schema requires it on every entry, so no valid production manifest can be written without it |

Also open: the four files in `fie/models/minilm-onnx/` that are in neither the manifest nor the
release (`config.json`, `ort_config.json`, `special_tokens_map.json`, `tokenizer_config.json`).
The derived package manifest has to say whether they are listed, out of scope, or forbidden.
They were neither ignored nor added.

A partial edit, adding `role`, `version` and `default` but not `in_wheel`, was rejected: it
would produce a tracked manifest the new module itself refuses, and would change a tracked
input of the evaluation harness for no usable result.

`scripts/download_models.py` depends on the same decisions (it would write the manifest derived
from those fields), so it is untouched too.

**Remaining Stage 3.1 work, blocked on the decisions in section H:** the fields in
`scripts/model_manifest.json`; deriving `fie/models/MANIFEST.json` from it, with the test that
the two agree; `scripts/download_models.py` writing the derived file.

---

## G. Regression

| Command | Exit | Result |
| --- | --- | --- |
| `python -m pytest tests/test_model_integrity.py -p no:cacheprovider` | 0 | 143 passed, 1 skipped |
| `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` | 0 | **1140 passed, 0 failed, 1 skipped** (997 before; the skip is the one above) |
| `python -m evals baseline` | 0 | Standard counts match; all 11 per-prompt record files byte-identical to `BL-0002_pair-v6.3b_fie-70de7053` |

No existing test was edited or disabled. The WP-004 guard test
(`test_detector_source_files_are_untouched`) passes with these files uncommitted, because it
watches a list of paths that does not include the new module.

---

## H. Owner decisions still needed

| # | Decision, stated precisely | Needed for |
| --- | --- | --- |
| W21-8 | Do the SDK's model files ship inside the one wheel (`in_wheel: true`), or is the wheel model-free? | `in_wheel` on every `fie/models` entry |
| W21-15 | `failure_classifier_v4.pkl` and `feature_columns_v4.pkl`, loaded by nothing in `fie/`: shipped in the wheel, or listed for the server only (`in_wheel: false`)? | `in_wheel` and `role` on two entries |
| W21-22 (new) | The four unlisted files in `fie/models/minilm-onnx/`: add them to the inventory with digests, or declare non-model JSON beside the encoder outside its scope? | The derived package manifest |
| W21-23 (new) | Confirm that the new fields go in place into `scripts/model_manifest.json`, and the role names, including for side-car and tokenizer files | The manifest edit itself |
| W21-9 | How a refused model is reported by a layer | Stage 3.2, not this stage |
| W21-16 to W21-19 | Build strictness; unavailable `FIE_PAIR_VERSION`; the fallback directory; line endings | Stages 3.2 and 3.3 |
| — | The research-workflow compatibility plan, then the relocation of the seven files (W21-20, approved) | Before any build from this working directory under the new rules |

---

## I. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1/`

| File | Content |
| --- | --- |
| `module.json` | The module's digest, imports, public functions, classes, error codes; that nothing in `fie/` uses it; that it does not deserialize |
| `test_results.json` | Focused and full-suite commands, exit codes, counts, the skip, the test file's digest, every passing test by name |
| `production_manifest_decision_review.json` | Section F: each field, its status and basis; the decisions needed; what remains |
| `SHA256SUMS.txt` | Checksums of the three files above |

Digests of source files are taken over LF-normalised bytes, because this checkout converts
line endings. All earlier evidence is unchanged and still verifies.

## J. Recommended manual commit command

Not executed by the assistant.

```text
git add fie/_model_integrity.py tests/test_model_integrity.py docs/fie_rebuild_2026 && git commit -m "fie: add standalone model-integrity primitives, not yet used by the loader (WP-021 phase 3 stage 3.1)"
```
