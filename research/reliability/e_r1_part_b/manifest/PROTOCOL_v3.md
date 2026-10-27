# E-R1 Part B — Protocol v3: NEI Construction Remediation (Option A)

**Written:** 2026-10-27, before `fever_nei_v3.py` was run and before any v3 passage was read.
**Supersedes:** only `PROTOCOL.md` §3, the NEI rule, and §4, which gets a new audit draw. Everything else in `PROTOCOL.md`, including its §10 addendum, stands unchanged. `PROTOCOL.md`, `sample_manifest.json`, `nei_audit_sample.json` and `nei_noise_audit_result.json` are retained as historical evidence, as is the v2 derived data in `data/fever/derived/`.
**Owner decision:** Option A from `worklog/reports/2026-10-27_E-R1_partB_FEVER_download_and_audit.md`.

## 1. What does not change

- The seed, the deduplication, the quotas, the sampled claims, the FEVER labels and the smoke/calibration/test assignments. `fever_nei_v3.py` imports these from `fever_materialize/2`; it does not re-implement them.
- **Gold SUPPORTS/REFUTES items:** copied byte-for-byte from the v2 derived files.
- **Sealed-test labels:** kept in a separate file.
- **Arms, prompts, metrics, thresholds and acceptance criteria** (PROTOCOL §5–§7).

## 2. New NEI rule (`fever_nei/3`)

1. **Spans.** Every claim span of 1–6 tokens that starts with a capitalised token (v2 tokenisation, NFC keys).
2. **Candidates.** All dump pages whose title equals a span, **or** equals a span followed by a parenthetical disambiguator: `Span_-LRB-…-RRB-`, for example `Mud_-LRB-2012_film-RRB-`. They are found in one streaming pass over every title in the archive.
3. **Not candidates.**
   - pages whose disambiguator contains "disambiguation";
   - pages whose first non-artefact lead sentence matches `may (also) refer to`.
4. **Lead.** The first two non-artefact sentences, by index.
   - **Artefacts:** empty lines, and image-markup residue matching `^\s*(thumb|left|right|upright|frame)\b[^|]*\|`.
   - Artefact lines are **skipped and their indices recorded**. No sentence text is edited.
   - Headless lead sentences (such as "is a 1997 …") are kept unedited.
5. **Selection score.** `|content(claim) − content(span)| ∩ content(lead + disambiguator text)|`.
   - `content` means lower-cased alphanumeric tokens, a fixed 90-word stop list, and a fixed suffix stemmer (strip `ing`, `ed`, `es` or `s` when the word is longer than 4 characters).
   - The matched span's own tokens are excluded from the claim side.
6. **Choice.** The highest score wins. Ties go to the longer span, then the undisambiguated title, then title order.
7. **No defensible page.** If the best score is 0, or no candidate exists, the item is **excluded and counted**. It is never forced and never re-sampled.
8. **Provenance per item:**
   - the source page id as written in the dump, and its NFC key;
   - the matched span and sentence indices;
   - the skipped artefact indices;
   - the score and the number of candidates considered.

**Known risk, stated in advance.** Choosing the page with the most overlap with the claim may raise the rate of *accidental support*, because a page that discusses the claim's content is more likely to settle it. The audit measures this.

## 3. Fresh audit (stop rule unchanged)

- **Pool:** v3 NEI items in smoke and calibration, **excluding the 50 item ids audited under v2** (`nei_audit_sample.json`). No sealed-test passage is inspected.
- **Draw:** 50 items, `random.Random("20261027:nei-audit-v3")`, from the pool sorted by item id. If the pool holds fewer than 50 items, all of them are audited and the denominator is reported.
- **Classes:** OK / ACCIDENTAL_SUPPORT / ACCIDENTAL_REFUTE / CONSTRUCTION_FAIL, as in PROTOCOL §4.
- **STOP** if CONSTRUCTION_FAIL ≥ 5, or if ACCIDENTAL_SUPPORT + ACCIDENTAL_REFUTE ≥ 3.
- **Reviewer:** the assistant. This is an **LLM review, not human validation**. Every flagged and borderline item goes into a human-confirmation table.
- **Before/after record (not part of the pass/fail count).** The 10 items flagged under v2 are re-shown with their v3 passages.

## 4. After a pass

- **Packet.** A human review packet is prepared for the sealed-test v3 NEI passages:
  - randomised order, seed `20261027:test-nei-review`;
  - stable review ids;
  - a private mapping file;
  - no labels, no model output.
- **Before scoring.** Scoring waits until the human review is complete and adjudicated.
- **Adjudication.** Excluded items are not replaced.
