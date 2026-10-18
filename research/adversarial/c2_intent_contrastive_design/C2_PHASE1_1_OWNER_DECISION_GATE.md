# C2 Phase 1.1 — Owner decision gate

| | |
| --- | --- |
| Date | 2026-10-18 |
| Repository | `rebuild/wp-001-eval-harness` @ `bf4450f` (C2 Phase 1 commit). Clean at start. |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Inputs inspected | `C2_PHASE1_DESIGN.md` (whole §10 and the sections it relies on), `design_spec.json`, `audit/training_data_audit.json`, `scripts/audit_training_data.py`, `logs/RUN_LOG.md`, and the E53–E55 reports and saved results |
| New evidence | `audit/phase1_1_pair_candidates.json`, from `scripts/inventory_pair_candidates_p1_1.py`. Ids and surface metrics only; no text, no scoring |
| Companion documents | [Human-review protocol](C2_PHASE1_1_HUMAN_REVIEW_PROTOCOL.md) · [Final-test options](C2_PHASE1_1_FINAL_TEST_OPTIONS.md) · [Readiness checklist](C2_PHASE1_1_READINESS_CHECKLIST.md) |

**These are recommendations. None is decided.** Each decision has a line for you to mark approve / reject / modify.

## 0. Preflight

- **Prior evidence verifies.** The E53, E54, E55 and C2 Phase 1 `SHA256SUMS.txt` files all verify.
- **The design's citations match saved evidence:**

  | Finding cited by the design | Saved source |
  | --- | --- |
  | E53: A3 − A0 XSTest −0.8; A3 − mean(R3) −19.7; A5 optimism 15.4–32.4 for 9 sources | `e53/results/metrics.json`, `e53/results/a5_loso.json` |
  | E54: no useful trade-off, H0 retained | `e54/results/metrics.json` |
  | E55: failed at α = 10% and 5% | `e55/results/metrics.json` |
  | All three integrity checks pass | `integrity_check.json` in each experiment |

- **Ignored outputs preserved:** the `results/` folders of E53, E54 and E55 (git-ignored by `.gitignore:82`).
- **Exact §10 wording:** reproduced verbatim at the start of each decision below.

## 0.1 One new fact that changes the plan

`audit/phase1_1_pair_candidates.json` inventories the 595 "near-neighbour pair candidates" of Phase 1 §4.1 route 1:

- **Only 1 of 595 passes the preregistered surface screens** (length ratio 0.67–1.5, token Jaccard ≥ 0.4,
  harm-vocabulary difference ≤ 1). That one candidate has a review-required attack side.
- **They are semantic neighbours, not minimal pairs.** Median token Jaccard is 0.091; cosine median is 0.83.
- **Benign partners repeat.** 195 unique benign partners serve 595 candidates; one partner is nearest to 25
  different attacks.
- **Many attack sides are themselves in question.** 355 of 595 need review, and 30 are refusals.

**Consequence:** mined pairs can serve only as topic-near *seeds*. Nearly all minimal pairs will need a newly
written benign twin. That raises the stakes of OD-C2-4.

---

## OD-C2-1 — artifact retention

> **Exact wording:** "Force-add the git-ignored `results/` folders of E53–E55 to version control?" — Blocking: **No**

**Why it matters.** The commits for E53–E55 do not contain their metrics, thresholds, per-arm scores or
integrity reports. Anyone reading git alone cannot check the conclusions.

| Option | Consequence |
| --- | --- |
| A. Force-add `results/` only (5.5 MB in total: E53 5.1 MB, E54 0.2 MB, E55 0.2 MB) | Conclusions become checkable from git. Contents are unchanged, so existing checksums stay valid. No raw benchmark text: `results/` holds metrics, thresholds and per-prompt scores keyed by hash. **Note:** E53 `results/models/` holds 20 pickled (joblib) research heads (0.25 MB). Committing pickles conflicts with the spirit of D-008, so a sub-option is to force-add `results/` minus `models/`; the heads are deterministic to refit (E53 D9) |
| B. Keep local-only and archive elsewhere (e.g. a release asset) | Smaller repo; evidence depends on a second store |
| C. Do nothing | Reports cite files that git does not hold |

**Recommended default: A**, for `results/` only, with no change to their contents. `cache/` (benchmark
texts) stays out of git.

**Required before:** nothing is blocked, but it should be settled before any C2 result cites E53–E55.

Owner: ☐ approve ☐ reject ☐ modify: ____

## OD-C2-2 — refusal rows

> **Exact wording:** "Approve excluding the 82 refusal rows from the attack class (no relabelling until approved)?" — Blocking: **For S2**

**Why it matters.**

- 82 of the 160 `llama_paraphrase:llama3.1:8b` rows labelled *attack* are model refusals ("I can't…").
- The detector screens user *inputs*. A refusal text is not an attack request, so these rows teach the head
  that apologetic phrasing is an attack signal.
- 30 of the 595 pair candidates have a refusal on the attack side.

| Option | Consequence |
| --- | --- |
| A. Exclude them from the attack class in the reviewed label version (originals untouched) | Removes a known label error. Excluded rows stay listed for audit |
| B. Relabel them as benign | Adds refusal-style text to the benign class. That is outside the input distribution and may create a new shortcut |
| C. Keep them | A known error stays in A1 and every later arm |

**Recommended default: A** (exclude, do not relabel). The exclusion is implemented in the new reviewed-label
file, never in the source data. A reviewer still confirms each of the 82 rows, since the regex could
misfire.

**Required before:** S2 (full review).

Owner: ☐ approve ☐ reject ☐ modify: ____

## OD-C2-3 — dual-use policy

> **Exact wording:** "Production policy for H = dual-use: block or allow?" — Blocking: **For S2**

**Why it matters.** It decides how H = dual-use maps to the production binary label (Phase 1 §3.2). It also
decides whether those rows train as positives or negatives, or are held out.

| Option | Consequence |
| --- | --- |
| A. Hold out of training; report them as a separate evaluation slice; production behaviour unchanged for now | No policy is baked into the data before the taxonomy pilot measures how often dual-use occurs and how reliably reviewers agree on it |
| B. Block (dual-use → attack) | Higher recall on ambiguous requests; reproduces the E4/E18 over-refusal pattern on medical, legal and security questions |
| C. Allow (dual-use → benign) | Lower over-refusal; known misses on soft-harm (SORRY "Unqualified advice", E22) |

**Recommended default: A** for C2 training. Production mapping is decided after the S1 pilot reports the
dual-use rate and κ. Either B or C remains open as an owner policy, but it is a product decision, not a
research shortcut.

**Required before:** S2 (labels); final mapping before S6 (preregistration).

Owner: ☐ approve ☐ reject ☐ modify: ____

## OD-C2-4 — local drafting and reviewers

> **Exact wording:** "May a local Ollama model write *benign* twins and frames under human review? Who reviews (2 people needed)?" — Blocking: **For S3**

**Why it matters.** §0.1 shows that almost every minimal pair needs a newly written benign twin.

**Feasibility (Verified from files only; nothing started, no inference):**

- `ollama.exe` is installed (27.8 MB, 2026-10-11).
- Manifests list `llama3.2` (2.0 GB), `mistral:7b` (4.4 GB), `qwen2.5:7b` (4.7 GB), `llama3.1:8b` (4.9 GB),
  `granite3.3:8b` (4.9 GB) and `qwen3:8b` (5.2 GB), each with licence layers.
- Only `llama3.2` fits the 4 GB GPU; the 7–8B models would need CPU offload.
- **Whether generation works, and how fast, is Unknown.**

| Option | Consequence |
| --- | --- |
| A. Humans write all benign twins (≈ 680) | No generator artifacts; slowest. About 680 twins × ≈ 3 minutes ≈ 34 hours of writing, before review |
| B. A local model drafts *benign* twins and frames; every draft is provisional until two humans accept it (protocol §6) | Faster. Needs artifact screens; drafts form their own source group and are evaluated source-held-out |
| C. No twins; use mined topic-near pairs | Fails the surface screens (§0.1); does not test the hypothesis |

**Recommended default: B**, but only with:

- benign drafts only;
- one named local model, pinned by its manifest digest;
- two reviewers;
- the artifact screens in the review protocol §6.

The restriction against generating new harmful text is kept. Changing it would be a separate decision
(OD-C2-4b below).

**Reviewers.** Two independent reviewers plus an adjudicator are needed. **Who they are is the owner's
decision.** This cannot be a single person reviewing their own work.

**Required before:** S3 (pair construction). The reviewer assignment is needed for S1.

Owner: ☐ approve ☐ reject ☐ modify: ____ · Reviewers: R1 ____ R2 ____ Adjudicator ____

> **OD-C2-4b (new, separate):** should harmful twins ever be drafted by a model? **Recommended: no.** Harmful
> twins come only from existing human-written requests in the training sources. This avoids creating new
> harmful text and avoids refusal artifacts (E24a).
> Owner: ☐ keep restriction ☐ modify: ____

## OD-C2-5 — final evaluation set

> **Exact wording:** "Approve downloading and freezing a final evaluation set (which sets, licences)?" — Blocking: **For S5**

**Why it matters.**

- Every set available locally is development evidence (Phase 1 §7.1).
- Without a sealed final set, no C2 result can be called a generalisation result.
- Options and trade-offs are in [C2_PHASE1_1_FINAL_TEST_OPTIONS.md](C2_PHASE1_1_FINAL_TEST_OPTIONS.md).

**Recommended default.** Approve a *complementary* set covering:

- (i) harmful-content intent,
- (ii) over-refusal,
- (iii) instruction-hierarchy manipulation, including over-defence.

Acquire it before S3, freeze it by hash, and keep it out of every selection step.

**Required before:** S5. It must be frozen before any C2 model is trained (S7).

Owner: ☐ approve ☐ reject ☐ modify: ____

## OD-C2-6 — second encoder for leakage checks

> **Exact wording:** "Which second encoder may be used for leakage and near-duplicate checks only? Needs a download; this is not the E40 comparison" — Blocking: **For S3/S5**

**Why it matters.** Deduplicating with the same MiniLM that C2 trains on can hide paraphrase leakage that
MiniLM maps apart.

| Option | Consequence |
| --- | --- |
| A. Download one small, commonly used sentence encoder, used for deduplication only, pinned by hash, never as a classifier | Better leakage detection; one download |
| B. Lexical deduplication only (MinHash or character n-grams, no download) + MiniLM | No download; misses semantic paraphrases |
| C. MiniLM only | Self-referential leakage check |

**Recommended default: B now, A when OD-C2-5's download happens**, in the same approved network window.
**Which encoder is the owner's choice.** It must not become an E40 comparison.

**Required before:** S3 (pair partitioning) and S5 (final-set deduplication).

Owner: ☐ approve ☐ reject ☐ modify: ____

## OD-C2-7 — head implementation

> **Exact wording:** "Accept numpy/scipy heads (no torch), or approve installing torch?" — Blocking: **For S7**

**Why it matters.**

- Verified: torch is not installed in `failure-engine` or `fie-eval-pinned`.
- The C2 objective (BCE + margin + consistency, plus optional group-DRO) on a 384→64 head has analytic
  gradients.

| Option | Consequence |
| --- | --- |
| A. numpy/scipy (L-BFGS) heads | No new dependency. A0 reproduces exactly (sklearn). Gradient code needs unit tests (finite differences) |
| B. Install torch in a separate research environment (not the eval environment) | Faster to write; adds a large dependency and another environment to pin |

**Recommended default: A**, with a finite-difference gradient test in the training preregistration.

**Required before:** S7 (training).

Owner: ☐ approve ☐ reject ☐ modify: ____

## OD-C2-8 — scope of the manipulation axis

> **Exact wording:** "Is a dedicated manipulation (M) head in scope, given only 71 local candidates, or should C2 focus on H?" — Blocking: **For S6**

**Why it matters.**

- The only local M = 1 candidates are the 71 ToxicChat `jailbreak` rows; 30% of that source is truncated at
  512 characters.
- JailbreakBench *prompts* (development) carry manipulation cues in 75% of rows, but they are evaluation
  data and cannot be used for training.

| Option | Consequence |
| --- | --- |
| A. C2 focuses on H; M is recorded in review and evaluated only on the final set's injection slice; no M claims | Honest scope; the M head is deferred |
| B. M head in scope with ≈ 71 examples | Underpowered; any result would be anecdotal |
| C. M head in scope with a new reviewed M training set from approved sources (needs OD-C2-5-style acquisition) | Larger project; separate data decision |

**Recommended default: A.**

**Required before:** S6 (preregistering the training experiment).

Owner: ☐ approve ☐ reject ☐ modify: ____

---

## Summary table

| ID | Recommended default | Needed before |
| --- | --- | --- |
| OD-C2-1 | Force-add `results/` only; contents unchanged | Before C2 cites E53–E55 (non-blocking) |
| OD-C2-2 | Exclude the 82 refusals from the attack class in a new label version; reviewer confirms each | S2 |
| OD-C2-3 | Hold dual-use out of training; decide production mapping after the S1 pilot | S2 / S6 |
| OD-C2-4 | Local model may draft **benign** twins and frames only, provisional until two humans accept; you name 2 reviewers + adjudicator | S1 (reviewers), S3 |
| OD-C2-4b | Keep the no-harmful-generation restriction | S3 |
| OD-C2-5 | Approve a complementary final set (intent, over-refusal, manipulation/over-defence), frozen before S3 | S5 (before any training) |
| OD-C2-6 | Lexical deduplication now; one dedup-only encoder in the same approved download window | S3 / S5 |
| OD-C2-7 | numpy/scipy heads with gradient tests | S7 |
| OD-C2-8 | Focus on H; no M claims | S6 |
