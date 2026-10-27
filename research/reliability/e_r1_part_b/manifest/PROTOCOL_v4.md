# E-R1 Part B — Protocol v4: Human-Selected NEI Passages (Option F)

**Written:** 2026-10-27, before any v4 candidate was generated.
**Owner decision:** Option F (see `worklog/reports/2026-10-27_E-R1_partB_NEI_remediation_v3.md` §8).
**Supersedes:** only the NEI passage rule, which was PROTOCOL.md §3 and PROTOCOL_v3.md §2, and its audit, PROTOCOL.md §4 and v3 §3.
**Retained unchanged as historical evidence:** `PROTOCOL.md`, `PROTOCOL_v3.md`, both failed audits (`nei_noise_audit_result.json`, `nei_noise_audit_v3_result.json`), all manifests, and the v2/v3 derived data.
**Not authorised here:** model scoring of any kind.

## 1. Unchanged

- **Sampling:** the sampled claims, labels, seeds, deduplication and smoke/calibration/test assignment, imported from `fever_materialize/2`.
- **Gold items:** SUPPORTS/REFUTES gold items as in v2.
- **Evaluation:** arms, prompts, metrics, thresholds and acceptance criteria (PROTOCOL §5–§7).
- **Sealed-test labels:** stay in `test_sealed_labels.jsonl`. **That file is not opened by any v4 step.** The NEI claims for the test packet come from the same deterministic sampling over `paper_test.jsonl`. Every claim in any packet is a FEVER NOT ENOUGH INFO claim, so the packets carry no other label information.

## 2. The decision the reviewer makes

For each NEI claim, the reviewer chooses the one candidate passage that is **about the claim's intended entity or subject** and **does not, by itself, establish that the claim is true or false**. If no candidate qualifies, the reviewer chooses **NONE**.

### 2.1 Definitions

| Term | Definition |
|---|---|
| **Intended entity or subject** | The thing the claim is about, as a competent reader understands the claim. Example: in "Moonlight was filmed in a shack", the subject is the film *Moonlight*, not moonlight the phenomenon. When a claim names several things, the subject is what the predicate is asserted of. "Tatum O'Neal was Time's Woman of the Year" is about Tatum O'Neal. A page about a *different referent with the same name* is wrong, and so is a page about a *work or event connected with the subject* rather than the subject itself; the reviewer judges which. A page about the right subject in a different form (for example the *Marnie* novel when the claim is about the film) is a **wrong sense** unless the claim is clearly about both |
| **Accidental support** | Read with ordinary inference, the passage establishes the claim: paraphrases, synonyms, dates and durations ("ran from 1968 to 1973" settles "for five years"), numbers, entailed categories ("debut studio album" settles "a work of music") and contextual implication all count. **The absence of the exact wording is not enough to call a passage non-supporting** |
| **Accidental refutation** | Read with ordinary inference, the passage establishes the claim is false: a contradicting value ("Governor of Texas" vs "of the Moon"), an incompatible category, an explicit negation, or a current-state contradiction ("currently a free agent" vs "plays for the 49ers"). A merely *different* fact that is compatible with the claim does not refute it ("Indian cricketer" does not logically exclude "soccer player"). Mark it uncertain if unsure |
| **Insufficiently relevant** | About the right subject, but so generic that it would not be a credible place to look. Example: a one-line stub that only restates the name |
| **Unusable** | Markup residue that obscures meaning (image captions, `thumb \|`); a headless lead that lacks the subject *and* is ambiguous without it; a hatnote ("For the theatre, see …"); a disambiguation list; empty or truncated text. Cosmetic issues such as PTB escapes converted for display, or IPA pronunciation brackets, do **not** make a passage unusable |

### 2.2 When to choose NONE

- **Must choose NONE** when every candidate is wrong-entity or sense, accidentally supports or refutes, is unusable, or is insufficiently relevant.
- **Do not force a choice.** Choosing NONE is expected and is not a failure. The sample size is not a goal.

### 2.3 Uncertainty

- `reviewer_confidence` is one of `high`, `medium` or `low`.
- A selection made with `low` confidence is recorded as `adjudication_status = UNRESOLVED` unless a second reviewer resolves it (§6).
- UNRESOLVED items are **excluded** from scoring, and they are counted.

## 3. Candidate generation (`fever_nei_v4_candidates.py`)

Candidates are **proposals only**. No automatic rule decides acceptance, and the reviewer never sees a rank.

1. **Spans.** Every 1–6-token claim span that starts with a capitalised token (v2 tokenisation, NFC title keys).
2. **Matched spans.** Spans whose title exists in the dump, either exactly or with a `_-LRB-…-RRB-` disambiguator.
3. **Maximal spans only.** A matched span strictly contained in another matched span is dropped, so "Rupert" yields to "Rupert Murdoch". Disjoint matched spans are all kept; the reviewer decides which is the subject.
4. **Pages for each span:**
   - the exact page;
   - disambiguated variants, ordered first by a **type-cue match** between the claim's words and the disambiguator (a fixed table: film, music, TV, book, play, game, sports, place, person), then by title.
   - Disambiguation pages are not candidates.
5. **Passages for each page:**
   - **lead window:** the first two non-artefact sentences;
   - **later window:** the next two non-artefact sentences, offered for the first page of the longest span, so the reviewer can avoid a lead that settles the claim.
6. **Selection of up to 5 candidates**, round-robin over the maximal spans (longest first):
   - span 1 page 1 lead;
   - span 1 page 1 later window;
   - span 2 page 1 lead;
   - span 1 page 2 lead;
   - and so on.
   - Fewer than 5 are shown when fewer exist. Zero candidates means the item is "no candidate", excluded and counted.
7. **Artefacts.**
   - Passage text is shown with PTB escapes converted for readability; the raw text is kept in the private mapping.
   - Flags in a separate column: image-line skipped, headless lead, hatnote, trailing colon or list, IPA. Artefact lines are skipped by index, never edited.
8. **The token-overlap ranking of v3 is not used.** Overlap is recorded as metadata in the private mapping only.

## 4. Packets and blindness

- **Three separate packets** (`smoke`, `calibration`, `test`).
  - Each has a randomised claim order, seed `20261027:v4-order:<role>`.
  - Each claim has stable review ids `R4-<role>-NNN` and candidate letters A–E in a randomised display order, seed `20261027:v4-cand`.
- **The reviewer sees:** the review id, the claim text, and for each candidate its letter, source title, page id, passage and artefact notes.
- **The reviewer does not see:**
  - FEVER ids;
  - ranks or the generation reason;
  - overlap scores;
  - model output of any kind;
  - gold labels;
  - the sealed-test label file.
- **Private mappings.** `mapping_<role>.json` maps review ids to FEVER ids and candidate letters to full provenance. It is not given to the reviewer.
- **Blank annotation columns.** They are left blank, and the blank packet's SHA-256 is recorded at generation.

## 5. Annotation schema (one row per claim)

| Field | Values |
|---|---|
| `selected_candidate_id` | `A`–`E` or `NONE` |
| `correct_entity_or_subject` | `Y` / `N` / `UNSURE`, for the selected candidate; `N/A` if NONE |
| `accidental_support`, `accidental_refutation`, `wrong_sense_or_entity`, `unusable_passage` | `Y` / `N`. For a selection these describe the selected candidate and must all be `N`. For NONE, mark `Y` on every reason that applied across the candidates |
| `insufficiently_relevant` | `Y` / `N` (same rule) |
| `reviewer_confidence` | `high` / `medium` / `low` |
| `rationale` | free text, **required for every row** |
| `adjudication_status` | `ACCEPTED` / `EXCLUDED` / `UNRESOLVED` |
| `reviewer_id` | the reviewer's initials or id |

**Consistency rules, enforced by the validator:**
- `ACCEPTED` ⇔ a letter is selected, the subject is `Y`, every defect flag is `N`, and confidence is not `low`.
- `EXCLUDED` ⇔ `NONE` with at least one `Y` reason, or a letter with a defect flag.
- `UNRESOLVED` ⇔ `low` confidence or `UNSURE` subject.

**Attestation.** Each completed packet requires a signed attestation file (`ATTESTATION_<role>.txt`) stating:
- the reviewer's id;
- the dates;
- that the review was done by a person **without** model or LLM assistance.

## 6. Second review (optional)

- **Subset.** A deterministic subset of 50 claims from smoke and calibration (seed `20261027:v4-second`).
- **Packet.** It is issued as `second_review_packet.csv`, with its own order and the same candidate letters.
- **Agreement.** Measured as Cohen's κ on (selected letter or NONE) and on ACCEPTED vs not.
- **Disagreements.** They are adjudicated by discussion, and the outcome is recorded in `adjudication_log.csv`.
- **Not a prerequisite.** A second reviewer is optional and is never simulated.

## 7. Freeze and counting (after review)

`validate_nei_review.py` must pass with zero errors. It then writes `nei_v4_adjudicated.json` (ids, choices, reasons; no text in tracked output) and the representativeness audit (§8). The frozen v4 derived files are then built:
- the gold items unchanged from v2;
- the NEI items taken from the ACCEPTED selections only, with provenance.

**Exclusions are counted and never replenished.** Any replacement sampling would need a new preregistered protocol, drawn only from the eligible pool, before outcomes are seen.

## 8. Representativeness audit (computed by the validator)

- **Counts.** NEI candidates before and after selection, by role; the distribution of selected display letters and of hidden candidate rank; the NONE rate.
- **Reasons.** Counts for each reason, and the uncertain count.
- **Concentration.** Exclusions by title type: person, work, place, organisation, other. These come from the disambiguator and the type-cue table, a coarse proxy.
- **Narrowing.** Claim length; the share of claims with several capitalised spans; and retained versus excluded on these.
- **Stop rule before scoring.**
  - If more than 35% of the NEI claims in any role are excluded, or if one title type accounts for more than 50% of exclusions while making up less than 25% of claims, the selection bias is reported and a revised evaluation design is recommended **before** the smoke run.
  - This mirrors the owner's instruction; the thresholds are fixed now.

## 9. Time budget

- **About 410 claims, read with up to 5 candidates of about 2 sentences each.** Expected 45–60 s per claim, giving **5.1–6.8 h**, within the owner's 5–7 h budget.
- **Optional second review:** 50 claims, about 45 min.
- **Suggested order:** the smoke packet first (10 claims) as a calibration of effort, then calibration (120), then test (280).
