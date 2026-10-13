# EXECUTION_021, Phase 3, Stage 3.1 follow-up — I/O error handling in the integrity primitives (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.1 follow-up only** |
| Status | **COMPLETE.** The gap is closed and tested. Stage 3.1 as a whole remains partially complete: the production manifest and download script are still blocked on owner decisions, and were not touched |
| Commit status | **NOT COMMITTED by the assistant.** Nothing staged, pushed, published or deployed |
| Date | 2026-10-13 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `5bac8cd`, unchanged |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1_followup/` |

The model loader still does not use the module. No model file was read, and nothing was
deserialized.

---

## A. Preflight

| Check | Result |
| --- | --- |
| Branch, HEAD, tree | `rebuild/wp-001-eval-harness`, `5bac8cd` (the Stage 3.1 commit), clean, nothing staged |
| Module and tests tracked | `fie/_model_integrity.py`, `tests/test_model_integrity.py`: yes |
| Evidence checksums | All seven WP-021 sets verify, including Stage 3.0 and Stage 3.1 |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |

## B. What was wrong

The audit named one gap: a `PermissionError` or other `OSError` while reading could escape as a
raw exception. Inspecting every filesystem call showed the gap was wider than that:

| Call (before) | What an operating-system refusal did |
| --- | --- |
| `open(path, "rb")` | Raw `PermissionError` escaped, with the absolute path in its text |
| `f.read(...)`, `os.fstat(...)` | Raw `OSError` escaped, likewise |
| `os.path.lexists(candidate)` | Swallows every error and returns `False`: a file the system would not let us examine was reported as **`missing`** |
| `os.path.isfile(candidate)` | Swallows every error and returns `False`: the same file could be reported as **`path_violation`** ("not a regular file") |
| `os.path.isdir(root)` | Swallows every error: an unreadable model directory was reported as "does not exist" |

So two cases leaked a raw exception and three were silently misclassified. Reproduced against
the committed module with injected failures (`before_after.json`):

| Injected failure | Before (`5bac8cd`) | After |
| --- | --- | --- |
| Permission denied on open | raw `PermissionError`, path in the message | `ReadFailed`, `read_failed` |
| I/O error on read | raw `OSError`, path in the message | `ReadFailed`, `read_failed` |
| Permission denied examining the file (`lstat`) | `ArtifactMissing`, **`missing`** | `ReadFailed`, `read_failed` |
| Permission denied examining the file (`stat`) | `PathViolation`, **`path_violation`** | `ReadFailed`, `read_failed` |
| File removed between check and open | `ArtifactMissing`, `missing` | unchanged |
| `TypeError` from open (a bug) | `TypeError` | unchanged: not caught |

## C. The change

`fie/_model_integrity.py` only.

**New class `ReadFailed(ModelIntegrityError)`, `code = "read_failed"`.** It carries the
manifest-relative `path`, `operation` (`"resolve"`, `"open"` or `"read"`), `reason` (the
exception's class name, such as `"PermissionError"`) and `errno`. Its message is
`<relative path>: could not <operation> the artifact (<class name>)`.

**Error categories.**

| Code | Class | Before | After |
| --- | --- | --- | --- |
| `manifest_invalid` | `ManifestError` | yes | unchanged |
| `path_violation` | `PathViolation` | yes | unchanged in meaning; no longer produced by a refusal to examine the file |
| `not_listed` | `NotListed` | yes | unchanged |
| `missing` | `ArtifactMissing` | yes | unchanged in meaning; no longer produced by a refusal to examine the file |
| `mismatch` | `IntegrityMismatch` | yes | unchanged |
| `read_failed` | `ReadFailed` | — | **new** |
| *(none)* | raw `OSError` | escaped | no longer escapes |

**How each filesystem step is handled now.**

| Step | `FileNotFoundError`, `NotADirectoryError` | Any other `OSError` | Anything else |
| --- | --- | --- | --- |
| Model directory: `realpath`, `os.stat` | `missing` (as before) | `read_failed`, operation `resolve` | propagates |
| File location: `realpath` | — | `read_failed`, `resolve` | propagates |
| Is anything there: `os.lstat` | `missing` | `read_failed`, `resolve` | propagates |
| Is it a regular file: `os.stat` | not regular → `path_violation` (a link to nowhere, as before) | `read_failed`, `resolve` | propagates |
| `open` | `missing` (removed after the check) | `read_failed`, `open` | propagates |
| `read`, `os.fstat` | — | `read_failed`, `read` | propagates |

The swallowing helpers (`os.path.isdir`, `lexists`, `isfile`) are replaced by explicit `os.stat`
and `os.lstat` calls, so that a refusal is visible instead of being read as "no".

**Only `OSError` is classified.** A `TypeError`, `ValueError`, `RuntimeError`, `MemoryError` or
`KeyboardInterrupt` passes through untouched: those are bugs or signals, not I/O conditions.

**No absolute path leaves the module.** The operating system's own message names the full
path, so it is not copied into the new error, and the new error is raised `from None`, so the
original exception does not appear in a logged traceback either. What is kept is the class
name and the error number. The existing "removed between check and open" case (`missing`) had
chained the original `FileNotFoundError`; it is now raised `from None` too, for the same
reason. Its class, code and message are unchanged.

**Unchanged.** The file is still opened exactly once and the bytes returned are the bytes
checked. No mapping to a detector-layer state or policy action was added; W21-9 remains open.
No deserialization, no fallback.

---

## D. Tests

`tests/test_model_integrity.py`: 38 tests added (182 in the file). **No existing test was
modified**; the 144 from Stage 3.1 pass unchanged against the new module.

| Requirement | Tests |
| --- | --- |
| 1. Permission failure while opening | 8 parametrised cases (`PermissionError` with two error numbers, and `OSError`, `IsADirectoryError`, `TimeoutError`, `BlockingIOError`, `InterruptedError` on open): `ReadFailed`, operation `open`, exact reason, error number, relative path, exact message; nothing returned |
| 2. I/O failure while reading bytes | A file object whose `read` raises: operation `read`; the read was attempted once; **the file was closed**. A failing `fstat` on an oversized file: operation `read` |
| 3. Failure during path resolution | Refusal on the directory's `stat`, the file's `lstat`, and the file's `stat`, each through both `resolve_path` and `read_verified`: operation `resolve`, explicitly neither `missing` nor `path_violation`, and `open` was never called. A failure inside link resolution (`ELOOP`) likewise |
| 4. Missing versus unreadable | One test producing both and asserting each is not an instance of the other. `FileNotFoundError` and `NotADirectoryError` from the system during resolution stay `missing`. Removal between check and open stays `missing`. An absent directory is `missing`; an unreadable one is `read_failed` |
| 5. Existing classifications preserved | Five unsafe paths still raise `PathViolation` **without any open being attempted**; tampered content and wrong size still raise `IntegrityMismatch`, not `ReadFailed`; an invalid manifest is still `ManifestError`; a readable file still returns the exact bytes with exactly one open |
| 6. Sanitization | Every injected error carries a planted path (`victim-home-dir/private-user-name/…`). For each resulting error the formatted traceback, `repr`, and arguments are checked to contain neither the planted names, nor the temporary directory, nor the real home directory; and the cause is `None` with context suppressed |
| Not disguising bugs | Six non-I/O exceptions from `open` and two from `read` propagate as the **same object**; four wrong argument types give `TypeError`, not an integrity error |
| The contract | Six classes, six codes, none a subclass of another, none an `OSError`. A sweep of eight `OSError` subclasses at both the open and the read step: none escapes raw |

Failures are injected (the module's `open`, and `os.stat` / `os.lstat` / `os.fstat` /
`os.path.realpath` for one path only), so the tests are deterministic and identical on Windows
and Linux.

| Command | Exit | Result |
| --- | --- | --- |
| `python -m pytest tests/test_model_integrity.py -p no:cacheprovider` | 0 | **180 passed, 2 skipped** |
| `python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA` | 0 | **1177 passed, 0 failed, 2 skipped** (1140 passed, 1 skipped before) |
| `python -m evals baseline` | 0 | Standard counts match; all 11 per-prompt record files byte-identical to `BL-0002_pair-v6.3b_fie-70de7053` |

The WP-004 guard test passes with these files uncommitted; the module is not on its watched
paths.

---

## E. Limitations and platform gaps

| # | Point |
| --- | --- |
| 1 | **A real unreadable file was not produced on this machine.** `test_a_real_unreadable_file_is_read_failed` uses `chmod 0`, which does not deny reading on Windows, so it is skipped here. The injected cases exercise the same code path deterministically. The real case runs on CI's Linux (unless CI runs as the superuser, where it also skips) |
| 2 | The symbolic-link escape test from Stage 3.1 is still skipped on this machine; unchanged |
| 3 | A Windows sharing violation (a file locked by another process) arrives as `PermissionError` and is covered as such; it was not produced for real |
| 4 | `os.path.realpath` on Windows suppresses most errors itself, so the "failure in link resolution" case is reachable there only by injection |
| 5 | A `ValueError` from an embedded NUL in the **directory** argument (the caller's, not the manifest's) is treated as a caller error and propagates. Manifest paths cannot contain NUL |
| 6 | The error still says which operation failed and the system's error class. That is deliberate; it is not location information |
| 7 | Nothing here decides what a detector layer reports when it receives `read_failed`. That is W21-9, for Stage 3.2 |

---

## F. Files

| File | Change |
| --- | --- |
| `fie/_model_integrity.py` | `ReadFailed`; explicit `stat`/`lstat` in `resolve_path`; classified open and read in `read_verified`; `from None` on the two re-raises; docstrings |
| `tests/test_model_integrity.py` | 38 tests appended |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_1-followup-io-errors.md` | This report |
| `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Section 14.6 |
| `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1_followup/` | New |

Not changed: `scripts/model_manifest.json`, `scripts/download_models.py`, `fie/layers/pair.py`,
every other source file, every existing test, packaging, workflows, `.gitignore`,
`.gitattributes`, any model or side-car file, the seven experiment files, `dist/`, baselines.
`fie/models/MANIFEST.json` does not exist and was not created.

## G. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_1_followup/`

| File | Content |
| --- | --- |
| `before_after.json` | Six injected failures against the module at `5bac8cd` and against the working tree: exception class, code, whether the message leaked the planted path, the cause |
| `test_results.json` | Commands, exit codes, counts, skips, the new tests by name, the module's and test file's digests |
| `SHA256SUMS.txt` | Checksums of the two files above |

All earlier evidence is unchanged and still verifies.

## H. Still open

Unchanged from the decision audit: W21-8, W21-15, W21-22, W21-23, the meta-classifier's
version name, and whether all seven classifier versions ship (to finish Stage 3.1); W21-9,
W21-16 to W21-19 (from Stage 3.2). W21-7 and W21-20 are owner-approved; the relocation under
W21-20 has not been carried out.

## I. Recommended manual commit command

Not executed by the assistant.

```text
git add fie/_model_integrity.py tests/test_model_integrity.py docs/fie_rebuild_2026 && git commit -m "fie: classify filesystem failures in the model-integrity primitives as read_failed (WP-021 phase 3 stage 3.1 follow-up)"
```
