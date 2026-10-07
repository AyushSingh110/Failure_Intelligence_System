# FIE Rebuild 2026 — Master Decisions and Implementation Log

**This is the source of truth for the rebuild.** It is updated at the end of every approved
session, never in bulk at the end of the project. If this file and any other document
disagree about what was decided or done, this file is right and the other one is stale.

| | |
| --- | --- |
| Started | 2026-10-07 |
| Baseline commit | `24cb2e9d76e977364f728f47790935283ef110b7` (`main`, clean, 236 commits) |
| Current phase | Planning complete. **No implementation approved yet.** |
| Next action | Owner approves, amends or reorders WP-001 and answers the open questions in §7 |
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
| UQ-2 | Is the one-in-250 verdict flip caused by unseeded `langdetect`? | Seed it in the harness and re-run the determinism suite (WP-001) |
| UQ-3 | Is the legal and medical evasion a learned shortcut from the v6 benign corpus? | E50 |
| UQ-4 | What does the PyPI 1.18.0 wheel do on a clean machine, with and without `[ml]`? | E51 |
| UQ-5 | Does the per-IP rate limiter see one address for all users behind the Space proxy? | Test in WP-002 |
| UQ-6 | Is the live Space's XGBoost classifier loaded? `/monitor/model-info` reports `model_loaded: false`, which may only reflect lazy loading | Check `/health/deep` during WP-002 |
| UQ-7 | Do baseline numbers hold in an environment rebuilt to the pinned versions? | One pinned-environment run in WP-001 |
| UQ-8 | Does `recalibrate()` in production actually drop operator attack-threshold overrides, as the code reads? | Unit test in WP-002 |
