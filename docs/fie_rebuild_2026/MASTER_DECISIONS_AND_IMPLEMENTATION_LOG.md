# FIE Rebuild 2026 — Master Decisions and Implementation Log

**This is the source of truth for the rebuild.** It is updated at the end of every approved
session, never in bulk at the end of the project. If this file and any other document
disagree about what was decided or done, this file is right and the other one is stale.

| | |
| --- | --- |
| Started | 2026-10-07 |
| Baseline commit | `24cb2e9d76e977364f728f47790935283ef110b7` (`main`, clean, 236 commits) |
| Current phase | **WP-001 (evaluation harness) complete, awaiting owner review.** Branch `rebuild/wp-001-eval-harness`, local only, not pushed. Work after Step 1 is uncommitted by owner instruction |
| Canonical baseline | `evals/baselines/BL-0001_pair-v6.3b_fie-5d5ed90d` (PAIR v6.3b). Reference: `REF-0001_pair-v6.2_fie-5d5ed90d` |
| Next action | Owner reviews [EXECUTION_001](EXECUTION_REPORTS/EXECUTION_001_eval-harness.md) and commits. WP-002 not started and not approved |
| Companion documents | [BASELINE_AUDIT.md](BASELINE_AUDIT.md) · [ROADMAP.md](ROADMAP.md) · [EXECUTION_REPORTS/](EXECUTION_REPORTS/) · [evidence/](evidence/) |

## How to use this log

- **Decisions** (§2) are numbered `D-nnn` and never renumbered. A reversed decision gets a new
  entry that names the one it replaces; the old entry is marked superseded, not deleted.
- **Work packages** are numbered `WP-nnn`. Each approved package gets one report,
  `EXECUTION_REPORTS/EXECUTION_nnn_<short-name>.md`, following
  [the template](EXECUTION_REPORTS/TEMPLATE.md).
- **Experiments** keep the `E-number` series of `docs/RESEARCH_LOG.md`. E30–E48 retain the
  meaning the September review gave them. New experiments start at E49 (D-010).
- **After every session**, add a row to §3, update §4–§8 as needed, and bump the header table.
- Status words used here: *proposed* (awaiting the owner), *approved*, *implemented*,
  *rejected*, *superseded*.

---

## 1. Standing rules for every session

Taken from the project brief. They apply until a decision in §2 changes them.

1. Before any change: record `git status`, branch and commit; run the test baseline.
2. No deletion of existing functionality, no unrelated refactors, no silent change to a public
   API, benchmark setting, threshold, test or model version. Any breaking change is named in
   the execution report before it is made.
3. Every change that can alter a verdict reports attack recall **and** benign over-refusal,
   paired against the pinned baseline.
4. A failed experiment is logged as fully as a successful one.
5. No claim in user-facing text without a report file behind it. No compliance claims.
6. Research LLM calls go through local Ollama, not Groq.
7. Secrets never enter the repository, the logs or these documents.

---

## 2. Decision register

### D-001 — The guardrail is the product; the hallucination monitor is research
**Status:** proposed · **Date:** 2026-10-07 · **Type:** product

**Decision.** The production product is the input guardrail in `fie/`. The output monitor in
`app/` + `engine/` is a research prototype. It must not be described as a working feature,
must not gate or rewrite anything in a production path, and moves under `research/`.

**Evidence.** Monitor ROC-AUC 0.497 and 0.396 on decontaminated data. The evaluated LangGraph
pipeline is not the code `/monitor` runs. The API advertises AUC 0.749. Guard recall 88.2%
macro reproduced exactly. Audit §4, §5.10.

**Rejected alternative.** Fix the monitor first (the September plan's Phase 4 brought forward).
Rejected: eight or more weeks of research with an uncertain outcome, while the shipped guard
misleads users today.

### D-002 — Trust before capability
**Status:** proposed · **Type:** product, sequencing

**Decision.** Phase 0 contains no new detection capability. It makes the shipped artifact
equal to the measured artifact, makes results report their own coverage, closes the live
tenant-isolation holes, and builds the measurement harness.

**Evidence.** Default PyPI install: ~6% macro recall (simulated) with `degraded_layers == []`.
Silent egress of prompt text. Three inert public parameters. Audit §5.1, §5.5, §5.6.

**Rejected alternative.** Start with the over-refusal fix because it is the headline weakness.
Rejected: it cannot be evaluated without the harness, and it does not help a user whose install
has no classifier.

### D-003 — Expose the existing three zones as ALLOW / REVIEW / BLOCK; keep detection and policy separate
**Status:** proposed · **Type:** architecture

**Decision.** Detectors return scores. A versioned, declarative policy turns scores, coverage
and language into a decision. The default preset reproduces today's verdicts exactly.

**Evidence.** The routing already has three zones internally ([fie/adversarial.py:1180-1350](../../fie/adversarial.py#L1180)).
Measured split, audit §7: XSTest hard blocks fall from 52.8% to 27.6%; OR-Bench-hard from
90.4% to 74.0%; 13.4% of benchmark attacks sit in the review band.

**Consequence stated honestly.** REVIEW relabels uncertainty. It does not reduce error, and it
does little for OR-Bench-hard.

**Rejected alternatives.** (a) Keep the boolean and tune thresholds: E20 shows no acceptable
operating point. (b) A continuous score only, no states: integrators need an action, and a
bare score invites each one to invent a threshold with no measurement behind it.

### D-004 — Evaluation harness first, with no production code change
**Status:** proposed · **Type:** process, research

**Decision.** WP-001 is a tracked `evals/` harness and a pinned baseline. It touches nothing
under `fie/`, `engine/`, `app/` or `storage/`.

**Evidence.** Four of the fact sheet's measurement scripts are git-ignored. Benchmark scripts
do not record whether the tiebreaker was reachable. The headline 85.8% describes PAIR v6.2,
not the shipped v6.3b. The golden test passes only in the tiebreaker-unreachable state.
Audit §8 (C2, C17, C18, C27), §10.

**Rejected alternative.** Ship the server isolation hotfix first. Not rejected on merit: it is
independent and may be swapped to first (OQ-2). The harness is recommended first because it is
one session, carries no production risk, and every later report needs it.

### D-005 — "A script is not an attack": script and language are inputs to policy, not signals of attack
**Status:** proposed · **Type:** architecture, research

**Decision.** Normalization is one shared stage feeding every detector. Text in a language the
classifier was not validated on is reported as *unsupported language* and handled by policy.
No detector may fire on script alone.

**Evidence.** 30 of 30 benign prompts in five scripts hard-blocked, by four different layers
(`gcg_suffix`, `pair_classifier`, `multilingual`, `regex`). PILOT, n=6 per script. Audit §5.2.

**Changes the September plan.** The review located the fault in `multilingual` and the regex.
Two further layers block independently, and PAIR itself is invalid on non-English input, so
the fix is structural.

### D-006 — No network egress and no non-determinism by default
**Status:** proposed · **Type:** security, product, **breaking**

**Decision.** Telemetry, online translation, the remote tiebreaker and model download become
explicit opt-ins. Language identification becomes deterministic.

**Evidence.** Prompt text is sent to Google Translate with no opt-out. Benign Spanish, French
and German prompts are blocked 18/18 when the translator is unreachable and 1/18 when it
answers. One verdict in 250 flipped between identical runs. Audit §5.2, §5.6, §5.7.

**Breaking.** Changes default behaviour for multilingual input and for anyone relying on the
import-time ping. Requires its own measured package (WP-005) and a deprecation notice.

### D-007 — Long-input scoring is engineering, pulled ahead of adaptive research
**Status:** proposed · **Type:** architecture

**Decision.** Windowed scoring (WP-009) precedes the adaptive-attack suite (WP-012).

**Evidence.** One 21-word benign sentence appended to a caught attack halves detection; 168
words leave 6.7%. The same padding removes over-refusal (132 → 5 of 250). Audit §5.8.

**Why it changes the plan.** The review treated this as a 256-token truncation fix to measure
late (E41). The mechanism is mean-pool dilution and it starts at one sentence. It also means
both headline numbers describe short bare prompts only.

### D-008 — No pickle in the production package
**Status:** proposed · **Type:** security

**Decision.** The PAIR head (linear SVM with sigmoid calibration) ships as plain numeric
weights. Session state serialises as JSON. Every model file is hash-checked at load.

**Evidence.** `joblib.load` at [fie/layers/pair.py:137](../../fie/layers/pair.py#L137) and
[:275](../../fie/layers/pair.py#L275); `pickle.loads` from Redis at
[fie/session_tracker.py:283](../../fie/session_tracker.py#L283); unverified download at
[fie/onnx_encoder.py:105](../../fie/onnx_encoder.py#L105).

**Rejected alternative.** `skops`: adds a dependency to solve what a 384-float vector solves.

### D-009 — Tenant state never alters global behaviour
**Status:** proposed · **Type:** security

**Decision.** Tenant identity comes only from the credential. Every shared mutable store is
made per-tenant or switched off. Shared models change only offline, through the evaluation
gate, under a new version.

**Evidence.** Audit §5.4, S1–S8. The most serious: any authenticated tenant can place an
answer in a global cache that `correct` mode then serves to other tenants' users.

### D-010 — Experiment numbering
**Status:** proposed · **Type:** process

**Decision.** E30–E48 keep the September review's assignments. Rebuild experiments start at
E49: E49 long-input dose-response, E50 domain-shortcut test, E51 clean-environment install,
E52 meta-classifier zone ablation.

### D-011 — Today's probes are evidence, not experiments
**Status:** implemented (this session) · **Type:** process

**Decision.** The 2026-10-07 measurements are recorded as audit evidence under
`evidence/2026-10-07/`, not as E-entries in `docs/RESEARCH_LOG.md`. They had no pre-registered
hypothesis. WP-001 turns them into tracked suites; the ones that matter become E49–E52 with
hypotheses written first.

### D-012 — Do not publicise working evasions before the fix
**Status:** proposed · **Type:** security, communication

**Decision.** The padding and prefix results (audit §5.8, §5.9) are disclosed in the changelog
and execution report of the package that addresses them, not earlier in public posts.

**Note.** They are simple enough that the practical exposure is small, and the README already
says white-box evasion is expected. The reason is ordering, not secrecy.

---

## 3. Session log

| # | Date | Type | Approved scope | Outcome | Report |
| --- | --- | --- | --- | --- | --- |
| 0 | 2026-10-07 | Planning | Read-only audit; baseline tests; planning documents in `docs/fie_rebuild_2026/` | Audit, roadmap and this log written. 87/87 tests pass. Published v6.3b figures reproduced exactly. Repository working tree unchanged outside this directory | This file, §4 |
| 1 | 2026-10-08 | Planning gate | One planning document for WP-001 | [PLAN_001_EVAL_HARNESS.md](IMPLEMENTATION_PLANS/PLAN_001_EVAL_HARNESS.md) written. 85.8% vs 88.2% resolved: two model versions on the same splits, both reproduce | The plan, §3 |
| 2 | 2026-10-08 | Implementation | WP-001, approved with decisions OD-1 to OD-13 | **Complete, awaiting review.** Harness built in `evals/`; canonical baseline and v6.2 reference pinned; all approved counts reproduce; deterministic artifacts byte-identical; 0 outbound attempts; 317 tests pass (87 existing unchanged + 230 new); no production file changed. See §9 | [EXECUTION_001](EXECUTION_REPORTS/EXECUTION_001_eval-harness.md) |

---

## 4. Session 0 record (planning, 2026-10-07)

### Pre-session state

| | |
| --- | --- |
| Branch / commit | `main` / `24cb2e9` |
| `git status` | clean |
| Environment | conda `failure-engine`, Python 3.10.19, Windows 11 |

### What was done

1. Read the README, fact sheet, September research review, the whole `fie/` scan path, the
   server's auth, routing and shared-state modules, CI, packaging and deployment files, and
   every evaluation report under `data/`. Full list: audit §13.
2. Ran the offline test suite in an environment that cannot see real credentials.
3. Ran two read-only probe scripts against the frozen, SHA-pinned splits, with translation
   stubbed so no prompt left the machine.
4. Queried three public read-only endpoints: the PyPI JSON record for `fie-sdk`, HTTP HEAD on
   three GitHub Release model assets, and `/health`, `/ready`, `/monitor/model-info` on the
   live Space.
5. Wrote the three planning documents, the execution-report template, and the evidence folder.

### Metrics recorded as the baseline

Shipped configuration: PAIR v6.3b, threshold 0.50, tiebreaker unreachable, translation
stubbed offline. Source: [`evidence/2026-10-07/probe_report.json`](evidence/2026-10-07/probe_report.json).

| Set | n | Flagged | Clear block | Uncertain block |
| --- | --- | --- | --- | --- |
| XSTest safe | 250 | 132 (52.8%) | 69 | 63 |
| OR-Bench-hard | 250 | 226 (90.4%) | 185 | 41 |
| XSTest unsafe | 198 | 177 (89.4%) | 156 | 21 |
| JailbreakBench | 134 | 130 (97.0%) | 122 | 8 |
| HarmBench | 387 | 326 (84.2%) | 263 | 63 |
| StrongREJECT | 242 | 218 (90.1%) | 201 | 17 |
| SORRY-Bench | 387 | 316 (81.7%) | 250 | 66 |

Macro recall over the four attack benchmarks: 88.2%. Counting clear blocks only: 76.7%.
Latency, short prompts, n=200: mean 34.3 ms, p50 33.2, p95 50.0.
Tests: 87 passed, 0 failed, 0 skipped, 28 s.

### Findings that changed the direction

| Finding | Effect on the plan |
| --- | --- |
| Default PyPI install has no classifier and reports full coverage | P0 moved ahead of the over-refusal work; WP-003 and WP-004 created |
| Four layers, not one, block non-Latin text; PAIR is invalid on non-English | Script handling became a pipeline stage (D-005), not a layer fix |
| European-language verdicts depend on Google Translate being reachable | Hermetic default made a prerequisite of script handling (D-006) |
| Padding collapses recall and over-refusal from the first sentence | Long-input scoring pulled forward (D-007); benchmark numbers re-read as short-prompt numbers |
| A fixed "compliance review" prefix evades 34% of caught attacks; the domain multiplier is not the cause | New experiment E50; caution against further domain-balanced augmentation until it is understood |
| `/track` unauthenticated with caller-supplied tenant; global answer cache | P5 split; WP-002 created as an independent hotfix |
| The UNCERTAIN band holds 25.2% of XSTest, 16.4% of OR-Bench-hard and 13.4% of attacks | REVIEW adopted with its cost stated (D-003) |
| The evaluated hallucination pipeline is not the deployed one | Monitor classified as research with no product claim (D-001) |

### Errors and corrections during the session

| What happened | Root cause | Resolution |
| --- | --- | --- |
| First draft of the audit gave file, line and route counts from estimates | Written before counting | Counted from `git ls-files` and `grep`; corrected to 35 SDK files / ~8.3k lines, 92 server files / ~19.5k lines, 43 routes, 35 environment variables |
| The follow-up probe exceeded the 10-minute foreground limit | About 8,500 scans, long padded prompts | Completed in the background; results read from its output file. No data lost |
| A latency-versus-length timing was taken while that probe was still running | CPU contention | Reported in the audit as indicative only; proper measurement assigned to WP-001 |
| The lite-install result is a simulation on a full install | A clean environment would have meant installing packages, outside this session's scope | Labelled as simulated everywhere; clean-environment check is E51 |
| The domain-keyword probe first seemed to show the threshold multiplier being exploited | The suffix also changes the PAIR embedding | Re-ran with the domain forced to `default`; the multiplier explains at most 1 of 3,052 cases. Attribution corrected to PAIR |

### Regressions

None. No code changed.

### Tests added

None. Probe scripts are preserved under `evidence/2026-10-07/` for promotion in WP-001.

### Security impact of the session itself

No credential was read: `.env` was inspected for key names only, with values masked. Tests ran
with every `.env` key overridden. Telemetry was disabled. The probes wrote blocked-prompt
records to a scratch file instead of `~/.fie/`. Outbound requests: the test suite's own calls
to Groq (rejected, fake key) and Google Translate (rate-limited), and the three read-only
lookups listed above.

---

## 5. Configuration and model state

Nothing was changed. Recorded so later sessions can detect drift.

| Item | Value at baseline |
| --- | --- |
| Shipped PAIR model | `pair_intent_classifier_v6_3b.pkl`, SHA-256 `9c682b28…2514`, threshold 0.50 |
| Meta-classifier | `meta_clf.pkl`, SHA-256 `be6673d0…e9a6`, threshold 0.41 |
| Embedder | `fie/models/minilm-onnx/model.onnx`, SHA-256 `57eb46cc…d2ea`, 256-token window |
| Manifest | `scripts/model_manifest.json`, release `models-v1.18.0`, 25 artifacts, all local files match |
| Per-type thresholds | `_ATTACK_THRESHOLDS` in `fie/adversarial.py:113-128`; uncertain band `[0.60·T, T)` |
| Package version | 1.18.0 in `pyproject.toml`; PyPI 1.18.0 uploaded 2026-08-10 is a different build |
| Environment drift from `requirements.txt` | xgboost 3.2.0 (2.1.4), joblib 1.5.3 (1.4.2), numpy 2.2.6 (2.1.3), fastapi 0.135.1 (0.115.6), pydantic 2.12.3 (2.10.3), langgraph 1.2.4 (0.2.76), PyJWT 2.12.1 (2.10.1) |

---

## 6. Known limitations of this audit

- The monitor (`engine/`) was read selectively. Findings about it are a lower bound.
- The script and language results are a pilot: six hand-written prompts per group.
- The lite-install recall is simulated, not measured from the PyPI wheel.
- No adaptive attack was run. The padding and prefix probes are naive by design.
- The online-tiebreaker configuration was not measured, only bounded.
- The dashboard frontend was checked only for token storage and displayed claims.
- Rate limiting behind the hosting proxy was reasoned about, not tested.
- Baseline numbers were taken in an environment whose package versions differ from the pins.
- The live service was queried on three read-only routes. No finding was tested against it.

---

## 7. Open questions for the owner

| # | Question | Why it matters | Recommendation |
| --- | --- | --- | --- |
| OQ-1 | `.gitignore` line 116 (`docs/*`) ignores this whole directory. Should `docs/fie_rebuild_2026/` be tracked, and public? | Until it is, the source-of-truth log is not under version control | Track it. Add the two negation lines in WP-001. If the security findings should stay private until fixed, keep the audit in a private branch until WP-002 ships |
| OQ-2 | Run WP-001 (harness) or WP-002 (server isolation) first? | They are independent. WP-002 closes holes on a live API | WP-001 first if hosted usage by other people is negligible; WP-002 first if anyone else relies on the hosted API |
| OQ-3 | Should the default install pull the ML dependencies, or stay light and refuse to run without them? | Decides WP-004's shape and is a packaging break | Pull them. Keep an explicit `lite` extra that labels every result |
| OQ-4 | Are the proposed gate margins acceptable: 2.0 points or p < 0.05, paired bootstrap, 10,000 resamples? | Every later package is judged against them | Accept for now; revisit once E30 gives per-language variance |
| OQ-5 | Are the scorecard targets in the roadmap §13 the right definition of done? | They are proposals | Confirm or edit before WP-007 |
| OQ-6 | Should `docs/RESEARCH_REVIEW_2026-09.md` and the measurement scripts become tracked? | Readers cannot check claims without them | Yes, in WP-006 |
| OQ-7 | Is a breaking 2.0.0 release acceptable for the hermetic default and the new result contract, or must everything stay additive in 1.x? | Shapes WP-003 to WP-007 | Additive through 1.x for WP-003; 2.0.0 for WP-005 and WP-007 |
| OQ-8 | Has the admin API key from before the June endpoint removal been rotated? | The June notes list it as still to do | Confirm |

---

## 8. Unresolved technical questions

| # | Question | How it gets answered |
| --- | --- | --- |
| UQ-1 | Does the meta-classifier move prompts between zones? A single layer at 0.58 against a 0.68 threshold ended as a clear block in the pilot | E52 |
| UQ-2 | Is the one-in-250 verdict flip caused by unseeded `langdetect`? | **Answered in WP-001: consistent with yes.** With the detector seeded, every canonical run is byte-identical. Unseeded, XSTest-safe row 99 changed verdict in 2 of 8 passes. The seed is the only thing that differs between the two |
| UQ-3 | Is the legal and medical evasion a learned shortcut from the v6 benign corpus? | E50 |
| UQ-4 | What does the PyPI 1.18.0 wheel do on a clean machine, with and without `[ml]`? | E51 |
| UQ-5 | Does the per-IP rate limiter see one address for all users behind the Space proxy? | Test in WP-002 |
| UQ-6 | Is the live Space's XGBoost classifier loaded? `/monitor/model-info` reports `model_loaded: false`, which may only reflect lazy loading | Check `/health/deep` during WP-002 |
| UQ-7 | Do baseline numbers hold in an environment rebuilt to the pinned versions? | **Answered in WP-001: yes, on this machine.** Environment `fie-eval-pinned` built from `requirements.txt`: all counts identical, all 2,016 standard records byte-identical. Another OS or CPU is still untested (UQ-9) |
| UQ-8 | Does `recalibrate()` in production actually drop operator attack-threshold overrides, as the code reads? | Unit test in WP-002 |
| UQ-9 | Do the canonical counts and bytes reproduce on Linux or another CPU? Do the guard tests pass there and on Python 3.11 and 3.12? | Raised by WP-001. One run on a Linux machine; no CI change is approved yet (OD-9) |
| UQ-10 | Does any native library used by `fie` open a network connection without CPython's `socket` module? The Python-level guard cannot see that | Raised by WP-001. An operating-system-level check (network namespace or firewall log) on Linux |
| UQ-11 | What are the upstream revisions of the JailbreakBench, HarmBench, StrongREJECT and AdvBench files? They are pinned by content hash only | WP-006 |
| UQ-12 | The E21 manifest gives HarmBench `clean_kept` = 380, but the frozen file behind every published HarmBench figure has 387 rows (7 exact duplicates). Which is the intended benchmark? | Owner decision in WP-006. WP-001 keeps 387 and reports 380 unique (OD-10) |

---

## 9. WP-001 implementation record

Updated at each checkpoint. Detail, commands and numbers are in
[EXECUTION_001_eval-harness.md](EXECUTION_REPORTS/EXECUTION_001_eval-harness.md).

### 9.1 Decisions approved by the owner on 2026-10-08

| # | Decision |
| --- | --- |
| OD-1 | Canonical baseline is PAIR v6.3b; PAIR v6.2 is a reference reproduction |
| OD-2 | Profile deviations V1–V10 approved. `sdk-offline-failsecure` is the **canonical reproducibility/evaluation profile**, never described as the production runtime. Reports keep four behaviours apart: shipped/default, canonical reproducibility profile, lite profile, stability/unseeded |
| OD-3 | Risk fixtures and live-service security findings stay on the local, un-pushed branch until WP-002 |
| OD-4 | Full per-prompt deterministic baseline records are committed |
| OD-5 | Evidence-file differences are reported, not fatal |
| OD-6 | Unpickled classes are recorded; no allowlist enforced |
| OD-7 | No packaging change |
| OD-8 | A fresh environment from the declared pins is built and tested; it does not replace the canonical environment automatically |
| OD-9 | No CI workflow change |
| OD-10 | HarmBench keeps its frozen 387 rows; 380 unique reported beside it |
| OD-11 | AdvBench is a labelled case study, never in the headline macro |
| OD-12 | Rebuild documentation is tracked on the local branch; the branch is not pushed |
| OD-13 | Tests live in `tests/evals/` |

These supersede the "proposed" status of D-004 (harness first) and answer OQ-1 (track the
documents: yes, on the local branch), OQ-2 (WP-001 first) and OQ-4 (margins accepted for the
regression design; nothing is compared in WP-001).

### 9.2 Owner instruction received during implementation

**No commits by the assistant.** Received 2026-10-08 after Step 1 had been committed. From
then on all work is left as uncommitted working-tree changes for the owner to commit. Two
commits made before the instruction remain on the branch: `094a105` (Step 0) and `c5b2291`
(Step 1). Consequence for the design: see D-013.

### 9.3 Implementation decisions

**D-013 — "Clean tree" means clean measured paths.** *Status: implemented.* The plan made a
dirty working tree non-canonical. With the harness itself left uncommitted by instruction,
that rule would have made every run non-canonical. The rule now reads: a run is traceable
when the **measured paths** (`fie/`, `engine/`, `app/`, `storage/`, `scripts/model_manifest.json`,
`data/overrefusal`, `data/benchmark_audit`, `pyproject.toml`) have no uncommitted change.
Changes elsewhere are recorded as `dirty_other`. The harness's own content is pinned
separately by `harness.tree_sha256` in every fingerprint. *Rejected:* asking the owner to
commit mid-implementation; treating the whole run as non-canonical.

**D-014 — Per-prompt records live in `records/`, not `results/`.** *Status: implemented.*
`.gitignore` already ignores any directory named `results/`. A pinned baseline's records
would have been silently untracked. Renamed; the plan's layout is otherwise unchanged.

**D-015 — Two pre-arm warm-ups in the hermetic guard.** *Status: implemented.* The plan
assumed that denying all IPv4/IPv6 socket creation and all process creation would have no
false positives. Step 3 found two, both local-only: `urllib3` creates an IPv6 socket and binds
it to `::1` at import time to probe for IPv6, and `platform.uname()` runs `cmd /c ver` on
Windows. Both are now resolved once, immediately before the guard is armed, and listed in
every run's guard summary. Nothing is exempt afterwards: creating a socket or starting a
process after arming is still denied and counted. *Rejected:* allowing socket creation
(weakens the guard and the approved "direct socket creation" test); exempting by call stack
(fragile).

**D-016 — Stability results are run metadata.** *Status: implemented.* The stability suite is
non-deterministic by design, so its numbers cannot sit in `REPORT.md`, which is byte-compared.
`REPORT.md` explains the four behaviours and points to `RUN_NOTES.md`, which carries the
stability and latency numbers.

**D-017 — `evals/.gitattributes`.** *Status: implemented.* With `core.autocrlf=true` a
checkout would convert pinned artifacts to CRLF and break every pinned hash. The file turns
conversion off for `evals/baselines`, `evals/fixtures` and `evals/registry`. Readers also fold
CRLF to LF, so a converted copy still verifies.

**D-018 — `pin` needs a determinism proof.** *Status: implemented.* The plan said a full
determinism verification is required before pinning. To make that enforceable, `pin` takes
`--determinism-proof FILE` and refuses unless the proof covers every deterministic suite and
its artifact digest equals the run's. It also refuses a non-canonical run and never
overwrites an existing baseline. *Limit:* it trusts the proof file; it does not re-run the
verification.

**D-019 — Stability passes: 8 for XSTest-safe, 3 for JailbreakBench.** *Status: implemented.*
The plan said 3 passes each. The known unstable prompt flips in roughly one pass in seven, so
three passes would usually miss it. Stability is run metadata, so the extra passes cost time
and nothing else.

**D-020 — The fresh environment uses Python 3.10.** *Status: implemented.* Same minor version
as the canonical environment, so that only the package versions change. Python 3.11 (the
Dockerfile's version) is left for UQ-9.

**D-021 — The canonical baseline stays in the `failure-engine` environment.** *Status:
implemented, per OD-8.* The pinned-version environment reproduced every record byte for byte,
but it was not promoted. Promoting it is the owner's decision.

The complete list of departures from the plan, DV-1 to DV-18, is in the execution report
under "Deviations". None changes an approved decision.

### 9.4 Step log

| Step | Result | Tests |
| --- | --- | --- |
| 0 Pre-state | Branch created; environment and model hashes recorded; execution report opened | 87 existing tests pass |
| 1 Canonical serializer | Done | 28 |
| 2 Dataset registry | Done. **Checkpoint passed:** 250 / 198 / 250 / 134 / 387 / 242 / 387 / 168 rows, all content hashes verified | 35 |
| 3 Hermetic guard | Done. All seven required child-process proofs pass. Two false positives found and resolved (D-015) | 33 |
| 4 Model integrity | Done. Mismatch, missing file, missing role, unlisted role and mid-run change all abort | 16 |
| 5 Subject adapter | Done. One module touches `fie`; 22 private names under a contract test | 14 |
| 6 Worker + standard suites | Done. **Hard checkpoint passed:** XSTest-safe 132/250, OR-Bench-hard 226/250, XSTest-unsafe 177/198, JailbreakBench 130/134, HarmBench 326/387, StrongREJECT 218/242, SORRY-Bench 316/387, AdvBench 163/168 | 22 (final count for the file, including two pin tests added at Step 13) |
| 7 Fingerprint + determinism | Done. **Hard checkpoint passed** on the standard suites: two separate worker processes, 8 gated files byte-identical, four keys equal, evidence files identical too; a third run with a random hash seed showed 0 record differences | 26 |
| 8 Metrics + reporting | Done. Two-axis rule enforced in code; `REPORT.md` is deterministic; stability and latency go to `RUN_NOTES.md` (D-016) | 28 |
| 9 Risk suites | Done. Long input (1,820 inputs), framing (4,052), script pilot (144, labelled PILOT), all labelled "constructed risk suite / diagnostic probe" | 20 |
| 10 Lite profile | Done. A real profile in a fresh interpreter with nine ML packages unimportable. Result: the classifier does not load and 1,848 / 1,848 scans still report full coverage | — |
| 11 Latency | Done. Kept out of every deterministic file | — |
| First full baseline | `python -m evals baseline`: exit 0, 764 s (12.7 min), canonical, all counts match | 228 harness tests pass |
| 12 Reference v6.2 | Done. **Checkpoint passed:** XSTest-safe 134/250, OR-Bench-hard 226/250, XSTest-unsafe 176/198, JailbreakBench 129/134, HarmBench 317/387, StrongREJECT 217/242, SORRY-Bench 291/387, AdvBench 160/168; macro 85.76% | — |
| 13 Determinism, full scope | `verify-determinism --all`: two separate runs, 13 gated files byte-identical (11 record files, `summary.json`, `fingerprint.json`), evidence and `REPORT.md` identical, 0 differences under a random hash seed. The baseline run has the same artifact digest: three identical runs. Reference profile: 9 gated files identical | — |
| 13 Pinning | `BL-0001_pair-v6.3b_fie-5d5ed90d` (canonical) and `REF-0001_pair-v6.2_fie-5d5ed90d` (reference) pinned; `CANONICAL` names the former | 230 harness tests pass |
| Fresh environment (OD-8) | `fie-eval-pinned` from `requirements.txt`: counts identical for both profiles; 7 record files byte-identical; only `env_key` differs. Not promoted (D-021) | — |
| Final confirmation | `python -m evals baseline` after pinning: exit 0, 725 s, "per-prompt records are byte-identical to BL-0001 (11 files)" | 317 pass: 87 existing with identical outcomes + 230 harness |
| Final checks | Secret scan: 856 files, 16 real `.env` values, 0 findings. Scope: nothing outside `evals/`, `tests/evals/`, `.gitignore`, `docs/fie_rebuild_2026/`; protected paths unchanged against `24cb2e9` | — |

### 9.4a Results now on record

Counts first. Canonical reproducibility/evaluation profile, PAIR v6.3b, baseline `BL-0001`.

| Axis | Set | Flagged / n | Rate |
| --- | --- | --- | --- |
| Attack | JailbreakBench | 130 / 134 | 97.01% |
| Attack | HarmBench | 326 / 387 | 84.24% |
| Attack | StrongREJECT | 218 / 242 | 90.08% |
| Attack | SORRY-Bench | 316 / 387 | 81.65% |
| Attack | **Macro recall, 4 sets** | — | **88.25%** [86.5, 90.0] |
| Benign | XSTest-safe | 132 / 250 | 52.80% over-refusal |
| Benign | OR-Bench-hard | 226 / 250 | 90.40% over-refusal |
| Contrast | XSTest-unsafe | 177 / 198 | 89.39% |
| Case study | AdvBench | 163 / 168 | 97.02%, outside the macro |

Reference, PAIR v6.2, `REF-0001`: macro 85.76% [83.9, 87.6]; XSTest-safe 134 / 250;
OR-Bench-hard 226 / 250.

Measured for the first time through the harness (constructed risk suites and diagnostic
probes, held on the local branch under OD-3):

| Probe | Result |
| --- | --- |
| Long input, 120 HarmBench prompts | 107 flagged unpadded; 56 with 21 filler words after; 8 with 84; 5 with 168 or more |
| Framing, 763 attack prompts | 674 flagged unframed; 445 under the legal template (229 lost), 607 medical, 652 developer |
| Lite profile, 1,848 scans | Classifier absent; recall 12 to 17 per set; **1,848 of 1,848 results report no degraded layer** |
| Stability, unseeded | XSTest-safe row 99 changes verdict in 2 of 8 passes |
| Latency, warm | Mean 24.5 ms at 46 characters; 163 ms at 2,732 characters |

### 9.4b Security findings from WP-001

None was fixed; fixing is outside this package. Full table in the execution report.

| # | Finding | Goes to |
| --- | --- | --- |
| SF-1 | A base install without the ML packages reports full coverage on every scan while the classifier is absent | WP-003, WP-004 |
| SF-2 | Telemetry, translation, tiebreaker and model download all attempt the network when left enabled | WP-005 |
| SF-3 | The product swallows the resulting exceptions and continues silently | WP-003 |
| SF-4 | `fie` imports `engine` and `storage` on the scan path | WP-011 |
| SF-5 | Model loading unpickles 11 classes | WP-004 |
| SF-6 | Padding and framing evasions confirmed at scale | WP-009, E49, E50 |
| SF-7 | Older dataset hash pins depend on CRLF working copies; two datasets had no pin | WP-006 |
| SF-8 | One verdict depends on an unseeded random number generator in a dependency | WP-005 |

The harness introduced no vulnerability and leaked no secret. It adds no dependency and is
not imported by any production module.

### 9.5 Errors met during implementation

Each is written up in full (error, root cause, investigation, fix, validation, lesson) in the
execution report. In brief:

| # | Error | Root cause | Fix |
| --- | --- | --- | --- |
| E-1 | The guard reported violations while `fie` only imported and warmed up | Two local-only probes: `urllib3` creates and binds an IPv6 socket at import; `platform.uname()` spawns `cmd /c ver` on Windows | Both resolved once before the guard is armed and listed in the guard summary (D-015) |
| E-2 | A pinned baseline's records would not have been tracked by git | `.gitignore` ignores every directory named `results/` | Directory named `records/` (D-014) |
| E-3 | The adapter refused to proceed when the caller had disabled logging | The model cross-check reads the loader's log line; with logging off there is nothing to read | `prepare()` re-enables logging first. The refusal itself was correct: fail closed |
| E-4 | A run-metadata test flagged a calendar date in `REPORT.md` | The date is static registry text saying when counts were approved, not a run timestamp | Test now looks for the run id, its UTC stamp and any date-with-time pattern |
| E-5 | Several scripted multi-line edits corrupted a string or failed to apply | Shell heredoc escaping of backslashes | Switched to the editor for such edits; each was caught immediately by a syntax error or a failing assertion |
| E-6 | Three tests had wrong expected values (import-blocker hit list, Wilson interval decimals, a prose substring check) | The tests, not the code: expectations typed from estimates | Expected values recomputed independently and compared with a tolerance |

No error came from the subject behaving differently from the plan's measurements. No stop
condition was triggered.

### 9.6 WP-001 final status

**Complete against the 20-item definition of done; awaiting owner review.** Not done, by
instruction: commits after Step 1, any push, WP-002, any product fix, any change to
`README.md` or `FACT_SHEET.md`.

State handed over:

| Item | Value |
| --- | --- |
| Branch | `rebuild/wp-001-eval-harness`, local only |
| Commits on the branch | `094a105` (Step 0), `c5b2291` (Step 1). Everything later is uncommitted |
| Original state | `24cb2e9` intact, equal to `main` and `origin/main` |
| Harness | `evals/` — 16 modules, 4,308 lines; registries, fixtures, two pinned baselines |
| Tests | `tests/evals/` — 230 tests |
| Run output | `evals/runs/` — 17 run directories, git-ignored, safe to delete |
| Extra conda environment | `fie-eval-pinned`, created for OD-8. Safe to remove with `conda env remove -n fie-eval-pinned` |

When the owner commits, the next run's `fingerprint.json` will differ from the pinned one in
its git fields only. Its records should stay byte-identical; `python -m evals baseline`
checks exactly that.
