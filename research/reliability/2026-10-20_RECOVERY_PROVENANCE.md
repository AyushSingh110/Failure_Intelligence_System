# Recovery provenance: the 2026-10-20 reliability reports

**Restored:** 2026-10-27 under OD-S0-5. **Neither original survives anywhere:** not in the working tree and not in any git ref.

## What happened

| Time (IST, 2026-10-27) | Event |
|---|---|
| about 11:50 | The audit session printed both originals with raw `cat`. First, `wc -l` gave **251** (audit) and **102** (plan) lines. The outputs were kept in the session transcript and its tool-result cache. |
| 11:51:15 | The `research/` directory changed and both files were gone. The audit session issued only read commands. Another interactive Claude session open in the same repository is the likely cause; this was not confirmed. |
| — | The files had never been committed: `git log --all -- research/reliability/*` lists no 2026-10-20 path. |

## Files

| Restored file | Size | SHA-256 | Status |
|---|---|---|---|
| `2026-10-20_output_reliability_audit.md` | 43,878 B, 251 lines | `5b11f8ac02b9b54250361d0da520de3db1260712483c28d5e5af7a0b702d200c` | **Content-identical to the raw capture.** The capture minus its 3-line `wc` header hashes to exactly this value, and the line count matches the original's 251. It is restored under the original filename, unedited. |
| `2026-10-20_verifier_replay_experiment_plan.RECONSTRUCTED.md` | 8,357 B | — | **Reconstruction.** A provenance header is followed by the body. |
| ↳ body below the header's rule | 7,372 B, 102 lines | `63b9d9a95ee1b9287321465af47c872c6bd058779a52cd3c6fdc980fdbceae1b` | Character-identical to the raw capture of the original. The capture lacks a final newline; one was added, consistent with the original's 102-line count. |

## Limits: what cannot be proven

- **Line endings.** Neither capture contains a carriage return, so the originals' line-ending style is unobservable. LF is assumed.
- **Original hashes.** Neither original's hash was recorded before deletion. Byte-identity with the originals therefore cannot be proven; only identity with the captures can.
- **Who deleted the files.** That is inferred, not established.

## Why the two files are labelled differently

Both are equally close to their captures. The owner directed (OD-S0-5) that the plan be restored under an explicit reconstructed filename, because it was described as rebuilt from a transcript. The audit came from the tool-result cache, which held the full raw output. The distinction is kept as directed.

Scratchpad copies (outside the repository, session `87822716`): `recovered_2026-10-20/`. They are unchanged.
