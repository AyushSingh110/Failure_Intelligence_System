# C2 Phase 1.1 — Human-review protocol (specification only)

**Status:** specification. **No row has been reviewed or relabelled.** It implements the taxonomy in
`C2_PHASE1_DESIGN.md` §3 and depends on OD-C2-2, OD-C2-3 and OD-C2-4 in
[C2_PHASE1_1_OWNER_DECISION_GATE.md](C2_PHASE1_1_OWNER_DECISION_GATE.md).

## 1. What reviewers label

Every reviewed row gets four independent fields. They are never collapsed into a single label at review time.

| Field | Values | One-line definition |
| --- | --- | --- |
| **M** instruction-hierarchy manipulation | `yes` / `no` / `unclear` | Does the text try to override, bypass, extract or redefine the assistant's instructions, role, policy or control boundary? |
| **H** harmful-content intent | `harmful` / `dual_use` / `none` / `unclear` | Would fulfilling the request give operational help inside FIE's harmful-content policy? |
| **C** context (multi-select) | `educational`, `academic`, `historical`, `fictional`, `defensive_security`, `quoted`, `professional_domain`, `none` | The communicative purpose; recorded, never used alone to set H |
| **Q** confidence | `high` / `medium` / `low` | How sure the reviewer is of their own M and H |

Two auxiliary flags:

- `not_a_user_input`: the row is a model refusal, a model answer or other non-prompt text.
- `toxic_only`: abusive or hateful language with no request for harmful help and no manipulation.

## 2. Decision flow (applied in this order)

1. **Is it a user input?**
   - A refusal ("I can't help with…"), an assistant answer or a fragment → `not_a_user_input = true`, Q =
     high. Stop: the row is excluded from training (OD-C2-2).
2. **Manipulation (M).** Mark `yes` if the text addresses the assistant to change its rules or role, or to
   reveal hidden instructions, e.g.:
   - "ignore previous instructions";
   - personas without restrictions ("you are DAN");
   - system-prompt extraction;
   - fake system or developer messages;
   - delimiter or role injection;
   - instructions to the model embedded in a quoted document (the addressee test, design §3.1 rule 2).

   Mark `no` if text *about* manipulation is analysed, quoted for classification, or explained.
3. **Harmful intent (H).** Read the *requested output*, not the topic.
   - `harmful`: the output would give operational help — steps, recipes, working code, targeting, evasion —
     for violence, weapons, cyber-intrusion, fraud, sexual content involving minors, self-harm methods, or
     other policy categories.
   - `none`: explanation, history, definition, prevention, detection, safety, policy discussion, fiction
     without operational detail, figurative language.
   - `dual_use`: both readings are plausible and the text gives no deciding signal (e.g. "What household
     chemicals are dangerous to mix?").
   - `unclear`: the reviewer cannot tell what is being asked.
4. **Context (C).** Tick every context that applies. **Context never turns H = harmful into `none`.**
   - A fictional or medical frame around an operational request stays `harmful`.
   - A frame that *changes the requested output* ("summarise the history of…" instead of "give the steps
     for…") is judged on the new output.
5. **Toxic without request.** Abuse or slurs with no request for help and no manipulation → H = `none`,
   M = `no`, `toxic_only = true`. The production mapping of `toxic_only` is out of C2's scope and is reported
   separately.
6. **Confidence (Q).** `low` whenever M or H is `unclear`, or the reviewer would not defend the label to a
   colleague.
7. **Escalate** (the adjudicator sees it whatever the agreement):
   - any `unclear`;
   - any `dual_use`;
   - Q = `low`;
   - M = `yes` together with H = `harmful`;
   - any row from the 22 cross-label conflicts;
   - any red-team row the reviewer marks H = `none`. These are expected, but each is checked, because it
     changes the label.

### 2.1 Source-specific guidance

| Source | Known issue | Reviewer note |
| --- | --- | --- |
| `anthropic_rlhf_redteam` (485) | Label = membership in a red-team transcript; the first turn is often innocuous | Judge **only the text shown**. Do not infer intent from the source. Many rows are expected to be H = `none` |
| `harmfulqa` (465) | Questions labelled harmful as a set; E2 found academic or business questions mixed in | Judge each question on its requested output |
| `toxicchat` `toxic_query` (180) | Flagged for toxicity *or* jailbreaking | Separate `toxic_only` from real harmful requests |
| `toxicchat` `jailbreak` (71) | The only M = `yes` candidates; 30% truncated at 512 characters | Mark Q = `low` if the truncation removes the request |
| `llama_paraphrase` (160) | 82 regex-detected refusals | Confirm the refusals; review the other 78 normally |

## 3. Reviewers, independence and adjudication

- **Reviewers.** Two independent reviewers (R1, R2), identified by pseudonymous IDs in the records, plus
  an adjudicator (A) who did not review the same row. Named by the owner (OD-C2-4).
- **Blind review.**
  - Reviewers see the row text, a row id and nothing else: not the current label, not the source name,
    not any model score.
  - Exception: source-specific guidance (§2.1) is given as general training before review, never per row.
- **Disagreement** is recorded field by field, keeping both reviewers' values.
- **The adjudicator decides with a written one-line rationale.** Rows still unresolved → Q = `low`,
  `adjudication = unresolved`. These are excluded from training and kept in the ambiguous analysis set.
- **Batching.** Order is randomised and stratified by source, 50 rows per batch. Each batch includes 5
  hidden duplicate rows to measure self-consistency.

## 4. Pilot (S1) before the full review

- **Sample:** 200 rows, stratified, drawn with a fixed seed recorded in the label store:
  - 40 red-team, 40 HarmfulQA, 30 ToxicChat (15 per subtype), 20 llama-paraphrase;
  - 10 cross-label conflicts;
  - 60 from retained sources (both labels), as the 10% audit pilot.
- **Measures:**
  - Cohen's κ per field (M, H) and for the derived production label;
  - percent agreement;
  - prevalence of `dual_use` and `unclear`;
  - self-consistency on the hidden duplicates;
  - time per row.
- **Acceptance (practical, fixed before the pilot):**
  - κ ≥ 0.60 on H and on M (if M prevalence is too low for κ to be stable, report percent agreement and
    the confusion counts instead);
  - self-consistency ≥ 90%;
  - `unclear` ≤ 15%.
- **If any criterion fails:** revise the guide, re-train the reviewers, and run a fresh 100-row pilot. The
  failed pilot labels are kept, not reused as training labels.

## 5. Full review (S2) and the 10% audit

- **Full review:** ≈ 1,279 rows (red-team 485, HarmfulQA 465, ToxicChat 251, llama paraphrase 78 + 82
  refusal confirmations), plus the 22 cross-label conflicts (they overlap the above).
- **10% audit of retained sources:** a seeded, stratified sample of 10% of each retained source
  (AdvBench mirror, forbidden_qs, MaliciousInstruct, JBB-behaviours, aug_v63 and every benign source).
  - If a source's audited production-label disagreement with its current label exceeds 10% (one-sided 95%
    lower bound > 5%), that whole source moves to full review.
  - This rule is fixed now.
- **Effort (estimate, not verified):**
  - ≈ 1,600 rows × 2 reviewers × ≈ 1 minute ≈ 53 reviewer-hours;
  - plus adjudication (≈ 15–20% of rows) ≈ 6 hours;
  - plus the pilot ≈ 8 hours;
  - plus minimal-pair review (≈ 680 pairs × 2 × ≈ 1.5 minutes) ≈ 34 hours;
  - ≈ 100 hours in total.

## 6. Minimal-pair review (S3)

Applies to every candidate pair: mined, human-written, or model-drafted (if OD-C2-4 = B).

### 6.1 Acceptance tests for a pair, in order

1. **Each side passes §2 independently**, with no conflict and Q ≥ `medium`:
   - attack twin: H = `harmful` (or M = `yes`);
   - benign twin: H = `none` and M = `no`.
2. **Same topic.** Both reviewers tick "same subject and key vocabulary".
3. **Intent is the only material difference.** Both reviewers tick "the requested output differs in
   harmfulness, not in subject".
4. **Automatic surface screens.** Length ratio 0.67–1.5; token Jaccard ≥ 0.4; harm-vocabulary count
   difference ≤ 1; same framing and quotation class.
   - Only 1 of 595 mined candidates passes these today (decision gate §0.1). Mined neighbours are therefore
     used as topic seeds for writing twins, not as pairs.

### 6.2 Corpus-level shortcut checks (after acceptance, before any training)

- **Surface-only check.** A bag-of-words classifier trained to tell the attack side from the benign side
  using only surface features must be near chance on held-out pairs, beyond its intent-word baseline. The
  rule is preregistered in S6.
- **Generator-artifact check** (if drafting is used). A classifier separating model-drafted benign twins
  from human-written benign twins must be near chance (AUC ≤ 0.65, preregistered). Otherwise drafted twins
  stay as a separate source group and are evaluated source-held-out only.
- **Provisional status.** A model-drafted twin carries `author = model:<name>@<manifest digest>`. It is
  `status = provisional` until two reviewers accept it. A provisional row never enters training, and no
  model draft is ever used as a label.
- **Partition unit.** A pair, its seed and every derivative belong to one partition.

## 7. Storage (append-only, auditable, originals untouched)

New files are planned under `research/adversarial/c2_reviewed_labels/` (owner to approve the path in S1):

- **`review_events.jsonl`** — one event per action, never edited. Fields:
  - `event_id`, `row_ref` (source file + row index + text SHA-256), `batch_id`;
  - `reviewer_id` (pseudonym), `role` (R1 / R2 / A);
  - `M`, `H`, `C`, `Q`, `not_a_user_input`, `toxic_only`, `pair_id`, `rationale` (adjudicator only);
  - `guide_version`, `timestamp`.
- **`labels_v1.jsonl`** — *derived* by a script from the events: final values per row, production mapping,
  adjudication status. Regenerated, never hand-edited. Each version is stored separately with a SHA-256
  manifest.
- **Training files are never modified.** A new label version references rows by (file, index, text hash).
- **Corrections** are new events with `supersedes = <event_id>`.
- **Summary reports show** counts, κ, ids and short excerpts only where essential, not raw text dumps.
  Reviewer names stay in a separate owner-held mapping.

## 8. Production mapping (recorded, applied only at training time)

- `attack` = (M = `yes`) ∨ (H = `harmful`).
- H = `dual_use`: held out per OD-C2-3 (recommended), or mapped by the owner's later policy.
- `not_a_user_input`, `toxic_only` with M = `no` and H = `none`, and Q = `low`: excluded from training,
  reported separately.
- Everything else: `benign`.
