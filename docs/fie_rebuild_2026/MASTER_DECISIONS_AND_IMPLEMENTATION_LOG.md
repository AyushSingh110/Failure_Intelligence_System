# FIE Rebuild 2026 — Master Decisions and Implementation Log

**This is the source of truth for the rebuild.** It is updated at the end of every approved
session, never in bulk at the end of the project. If this file and any other document
disagree about what was decided or done, this file is right and the other one is stale.

| | |
| --- | --- |
| Started | 2026-10-07 |
| Baseline commit | `24cb2e9d76e977364f728f47790935283ef110b7` (`main`, clean, 236 commits) |
| Current phase | **WP-002 (server isolation) implemented, awaiting owner review. NOT COMMITTED — the owner reviews and commits manually.** WP-001 is committed by the owner as `1e074d6`. Branch `rebuild/wp-001-eval-harness`. **The branch is on the public remote** (see §10.7) |
| Canonical baseline | `evals/baselines/BL-0001_pair-v6.3b_fie-5d5ed90d` (PAIR v6.3b). Reference: `REF-0001_pair-v6.2_fie-5d5ed90d` |
| Next action | Owner reviews [EXECUTION_002](EXECUTION_REPORTS/EXECUTION_002_server-isolation.md), commits, completes the deployment checklist (§10.9) and deploys. WP-003 not started and not approved |
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
| 2 | 2026-10-08 | Implementation | WP-001, approved with decisions OD-1 to OD-13 | **Complete; accepted and committed by the owner as `1e074d6`.** Harness built in `evals/`; canonical baseline and v6.2 reference pinned; all approved counts reproduce; deterministic artifacts byte-identical; 0 outbound attempts; 317 tests pass (87 existing unchanged + 230 new); no production file changed. See §9 | [EXECUTION_001](EXECUTION_REPORTS/EXECUTION_001_eval-harness.md) |
| 3 | 2026-10-08 | Planning gate | One planning document for WP-002 | [PLAN_002_SERVER_ISOLATION.md](IMPLEMENTATION_PLANS/PLAN_002_SERVER_ISOLATION.md) written. 51 routes and 41 state items inventoried; S1–S12 confirmed by code reading; 20 further findings (N1–N20) | The plan |
| 4 | 2026-10-08 | Implementation | WP-002, approved with the recommendations of plan §31 (W2-1 to W2-22) | **Implemented, awaiting review. Not committed.** One principal, one policy per route, scoped data access. Every in-scope attack reproduced on the unmodified code and is closed. 635 tests pass (87 existing + 230 harness + 318 security). `fie/` untouched. See §10 | [EXECUTION_002](EXECUTION_REPORTS/EXECUTION_002_server-isolation.md) |
| 5 | 2026-10-10 | Planning gate | One planning document for WP-003 | [PLAN_003_TRUTHFUL_SCAN_RESULT.md](IMPLEMENTATION_PLANS/PLAN_003_TRUTHFUL_SCAN_RESULT.md) written; committed by the owner as `550226d` | The plan |
| 6 | 2026-10-10 | Implementation | WP-003, approved with decisions W3-1 to W3-18 and the `bypassed` clarification | **Implemented, awaiting review. Not committed.** Results report coverage, zone, decision source and model identity. No verdict moved: ten canonical record files byte-identical to `BL-0001`; lite false-full-coverage 1,848 → 0 of 1,848. 785 tests pass. `BL-0002` not pinned (needs the owner's commit). See §11 | [EXECUTION_003](EXECUTION_REPORTS/EXECUTION_003_truthful-scan-result.md) |
| 7 | 2026-10-10 | Baseline | Post-commit verification and pin for WP-003 | WP-003 committed by the owner as `a9db959`. `evals baseline` exit 6 (only `lite.std.jsonl` differs), `verify-determinism --all` exit 0, `BL-0002_pair-v6.3b_fie-70de7053` pinned and made canonical, `evals baseline` exit 0. Committed by the owner as `924c70d` | §11.8 |
| 8 | 2026-10-10 | Planning gate | One planning document for WP-004, three revisions | [PLAN_004_RUNTIME_POLICY.md](IMPLEMENTATION_PLANS/PLAN_004_RUNTIME_POLICY.md), revision 3. Owner decisions W4-1 to W4-16 and Q1 to Q7 | The plan, §14 |
| 9 | 2026-10-10/11 | Implementation | WP-004, approved revision 3 plus one amendment | **Implemented, awaiting review. Not committed.** A pure policy layer (`fie/policy.py`) applied in `preflight_check` and shown by the CLI; default `compat@1` reproduces previous behaviour. 942 tests pass; all 11 record files byte-identical to `BL-0002`; determinism verified. **Not universal enforcement** (§12.5). See §12 | [EXECUTION_004](EXECUTION_REPORTS/EXECUTION_004_runtime-policy.md) |

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

---

## 10. WP-002 implementation record

Detail, evidence and the full attack table are in
[EXECUTION_002_server-isolation.md](EXECUTION_REPORTS/EXECUTION_002_server-isolation.md).

### 10.1 Approval

The owner approved WP-002 for implementation on 2026-10-08, with every recommendation of
[PLAN_002](IMPLEMENTATION_PLANS/PLAN_002_SERVER_ISOLATION.md) §31 (W2-1 to W2-22) and a
standing rule: **the assistant makes no commit, stage, push, stash or reset.**

One further decision was taken during implementation. **D-022 — a tenth existing test may be
edited.** *Status: approved by the owner.* The plan said nine tests in
`tests/test_integration.py` would change. The first test of `tests/test_monitor_rag_fix.py`
also calls `/monitor` with no credential. Raised as a stop condition before any file was
modified; the owner approved editing that one test (a principal is passed in; assertions are
unchanged).

### 10.2 State

| Item | Value |
| --- | --- |
| Implementation date | 2026-10-08 |
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD before and after | `1e074d69e81d27932b47c01d57e78aff91c40c44` (the owner's WP-001 commit) |
| Commit status | **NOT COMMITTED — owner will review and manually commit** |
| Deployed | No |
| Status | Complete with documented deviations |

### 10.3 Files changed

Created (production): `app/security_events.py`, `app/tenancy.py`, `storage/tenant_store.py`.

Created (tests): `tests/security/` — `__init__.py`, `conftest.py`, `fakes.py`, and
`test_authn.py`, `test_route_matrix.py`, `test_tenant_spoofing.py`,
`test_inference_isolation.py`, `test_answer_cache_isolation.py`, `test_feedback_isolation.py`,
`test_session_isolation.py`, `test_shared_state.py`, `test_admin_boundary.py`,
`test_exposure.py`, `test_concurrency.py`, `test_failure_modes.py`, `test_static_boundary.py`,
`test_performance_budget.py`.

Modified (production, 25): `app/auth.py`, `app/auth_guard.py`, `app/auth_routes.py`,
`app/limiter.py`, `app/main.py`, `app/routes/_helpers.py`, `app/routes/admin.py`,
`app/routes/analytics.py`, `app/routes/community.py`, `app/routes/flags.py`,
`app/routes/inference.py`, `app/routes/monitor.py`, `app/routes/playground.py`,
`engine/agents/adversarial/specialist.py`, `engine/agents/failure_agent.py`,
`engine/archetypes/registry.py`, `engine/fie_config.py`, `engine/groq_service.py`,
`engine/ground_truth_cache.py`, `engine/multi_turn_tracker.py`, `engine/retraining/buffer.py`,
`engine/session_store.py`, `engine/verifier/ground_truth_pipeline.py`, `storage/database.py`,
`storage/signal_logger.py`.

Modified (existing tests, 2): `tests/test_integration.py`, `tests/test_monitor_rag_fix.py`.

Documentation: `EXECUTION_REPORTS/EXECUTION_002_server-isolation.md` (new), this file.

Deleted: none. Not touched: `fie/`, `models/`, `scripts/`, `data/`, `evals/`, `tests/evals/`,
`Frontend/`, `deploy/`, `.github/`, dependency files, `README.md`, `SECURITY.md`, `FACT_SHEET.md`.

Size: 321 new production lines; 1,525 added and 829 removed in modified production files;
3,235 lines of tests.

### 10.4 Security invariants proven

All seventeen of plan §15 hold, each with an automated test: I-1 no unauthenticated access to
tenant data; I-2 the credential determines the tenant; I-3 a caller-supplied tenant cannot
raise access; I-4 and I-5 no cross-tenant read of the cache or write of state; I-6 feedback
changes nothing outside its tenant; I-7 sessions are per tenant; I-8 no existence oracle;
I-9 shared state is immutable or platform-controlled; I-10 rows carry their tenant; I-11 no
tokens and no start without a strong secret; I-12 admin rights are read from the store at the
time; I-13 no response carries another tenant's text or identity; I-14 a missing tenant fails
and is never served as global; I-15 an admin's cross-tenant read is explicit and recorded,
with no cross-tenant write or delete; I-16 a key can be rotated; I-17 the playground cannot be
pointed at internal addresses (limit: a DNS answer could change between check and use).

### 10.5 Acceptance criteria and tests

| Item | Result |
| --- | --- |
| AC-1 to AC-12, AC-14 to AC-22 | Pass. AC-11 and AC-22 pass with D-022 (ten tests edited, not nine). AC-14 with two stated exceptions: one hypothesis did not reproduce, one behaviour was already correct |
| AC-13 baseline | Pass. `python -m evals baseline` on the final tree: exit 0, all eight counts equal, 11 per-prompt record files byte-identical to `BL-0001`, four fingerprint keys equal, `fie/` tree `5d5ed90d` unchanged. Flagged `canonical=False` only because WP-002 is uncommitted (D-013); expected to clear after the owner's commit |
| Security suite | 318 passed. On the unmodified code the same suite: 115 passed, 203 failed |
| Existing tests | 87 passed; 86 outcomes identical by name, 1 test renamed |
| WP-001 harness tests | 230 passed |
| Full suite | 635 passed, 0 failed, 0 skipped, 0 expected failures |

Attack reproduction on the unmodified code: every in-scope finding (S1–S12 except the
unknowable proxy question in S12, N1–N13, N16, the N18 header, N19's mechanism) reproduced.
Not reproduced: the session "existence" read-oracle of N10 (the write side did reproduce).
Not attempted: N19's claim that a crafted prompt can make an external verifier return a wrong
answer.

### 10.6 Deviations

Fifteen, DV-1 to DV-15, in the execution report. The ones that change what the owner can
expect: DV-1 (the tenth test); DV-2 (the three platform switches do **not** re-open
request-path recalibration, retraining or index growth, because the routes no longer call
those functions); DV-3 (one planned event has no trigger and is not emitted); DV-9 (the
playground also refuses every IP literal); DV-13 (the security suite blanks `.env` keys in the
test process).

### 10.7 New findings

| # | Finding | Status |
| --- | --- | --- |
| NF-1 | `fie.feedback_store.apply_label` reports success for an unknown event id on the MongoDB backend | Not fixed (`fie/`). Harmless: verified by test |
| NF-2 | **The branch is on the public GitHub repository**, with the audit and the WP-002 plan, which describe attacks on the live service | **Owner action.** Decision OD-3 of WP-001 and plan §29 said it would stay local. The assistant did not push it |
| NF-3 | Reasoning sub-step verification calls the answer pipeline without a scope; it now runs uncached and logs `cache.scope_missing` | Safe. Two-file change outside Artifact E; deferred |
| NF-4 | Helper calls to the provider (explanation writer and others) are no longer cached | Safe. More provider calls |
| NF-5 | The unrouted `engine/pipeline/langgraph_pipeline.py` now stores nothing | Safe |
| NF-6 | `/ready` and `/health/deep` returned warm-up exception text | Fixed |
| NF-7 | Spike-alert e-mail count was always 0 (wrong key) | Fixed in passing |
| NF-8 | The Space-only `GET /api` route has no declared policy and is outside the route-matrix test | Not fixed (`deploy/`). Static banner |
| NF-9 | A comment in `scripts/show_api_key.py` names a removed function | Not fixed (`scripts/`) |

### 10.8 Residual risks

- The live service is unchanged until the owner commits and deploys.
- A stolen session token works for up to 24 hours. API keys are plain text in the database.
  Both are WP-017.
- The platform admin is trusted and can read every tenant on request; that read is now
  recorded.
- The scan-verdict and translation caches in `fie/` stay global: timing shows that some
  caller recently scanned identical text.
- Learned entries already in the attack-pattern index still take part in matching.
- Past effects are not undone: thresholds already recalibrated, cache entries and records
  already written. Legacy entries without a tenant are never served.
- The suite runs against an in-repository fake of MongoDB, not against MongoDB.
- The playground check resolves a name once and the HTTP client resolves it again.
- Whether the Space's proxy presents one address to the rate limiter is still unknown;
  authenticated traffic no longer depends on the answer.

### 10.9 Deployment checklist (owner)

1. `JWT_SECRET_KEY` (32+ characters) is set on the Space, or the build will not start.
2. `FIE_ALLOW_INSECURE_DEV_SECRET` is not set on the Space.
3. `ADMIN_EMAIL` on the Space is the intended account.
4. Decide whether the server-side `FIE_API_KEY` should exist there.
5. Decide whether to reset stored thresholds and whether to remove learned FAISS entries.
6. On the production database, drop the unique index on `inferences.request_id` alone after
   checking that no (`tenant_id`, `request_id`) pair is duplicated.
7. Rotate the admin API key if it was ever shared.
8. Tell known API users that anonymous access has ended.
9. Do not set `FIE_AUTO_RECALIBRATE`, `FIE_AUTO_RETRAIN` or `FIE_FAISS_AUTOGROW`.
10. Decide what to do about the public branch (NF-2).

### 10.10 Deferred

N14, N15, N17, the rest of N18, N20; NF-1, NF-3, NF-8, NF-9; key hashing and token revocation
(WP-017); per-tenant provider metering; a pinned connection for the playground.

### 10.11 Recommended manual commit command

Not executed by the assistant.

```text
git add -- app/auth.py app/auth_guard.py app/auth_routes.py app/limiter.py app/main.py app/security_events.py app/tenancy.py app/routes/_helpers.py app/routes/admin.py app/routes/analytics.py app/routes/community.py app/routes/flags.py app/routes/inference.py app/routes/monitor.py app/routes/playground.py engine/agents/adversarial/specialist.py engine/agents/failure_agent.py engine/archetypes/registry.py engine/fie_config.py engine/groq_service.py engine/ground_truth_cache.py engine/multi_turn_tracker.py engine/retraining/buffer.py engine/session_store.py engine/verifier/ground_truth_pipeline.py storage/database.py storage/signal_logger.py storage/tenant_store.py tests/security tests/test_integration.py tests/test_monitor_rag_fix.py docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_002_server-isolation.md docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md && git commit -m "security: isolate server tenant state and harden authentication (WP-002)"
```

---

## 11. WP-003 implementation record

Detail and evidence are in
[EXECUTION_003_truthful-scan-result.md](EXECUTION_REPORTS/EXECUTION_003_truthful-scan-result.md).

### 11.1 Approval

The owner approved WP-003 for implementation on 2026-10-10 with decisions W3-1 to W3-18 of
[PLAN_003](IMPLEMENTATION_PLANS/PLAN_003_TRUTHFUL_SCAN_RESULT.md) §26, with these departures from
the plan's recommendations: **W3-9 `risk_score` deferred** (not added); W3-10 `decision` field
and the inert-`threshold=` warning deferred; W3-12 the `/health/deep` path disclosure fixed as a
single-file exception in `app/main.py`. Standing rule unchanged: the assistant makes no commit,
stage, push, stash, reset or tag.

**D-023 — fast-path results are `bypassed`.** *Status: approved by the owner with the
implementation gate.* A prompt decided by a stored label runs no layer. Its layers are reported
neither `ok` nor `disabled` but `bypassed`, with `coverage.status = "bypassed"`. `bypassed` is a
whole-scan state only. No other new status value may be introduced without approval.

### 11.2 State

| Item | Value |
| --- | --- |
| Implementation date | 2026-10-10 |
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD before and after | `550226da5691ada35a19835059ec17e22632d2c6` |
| Commit status | **NOT COMMITTED — owner will review and manually commit** |
| Deployed | No |
| Status | Complete, with one owner-gated step (pin `BL-0002`) and documented deviations |

### 11.3 Files changed

Created: `tests/test_scan_result_contract.py` (141 tests),
`EXECUTION_REPORTS/EXECUTION_003_truthful-scan-result.md`.

Modified (production, 8): `fie/adversarial.py`, `fie/layers/pair.py`, `fie/onnx_encoder.py`,
`fie/_lite.py`, `fie/preflight.py`, `fie/__main__.py`, `fie/__init__.py`, `app/main.py` (five
lines, the detector block of `health_deep`).

Modified (harness, 1): `evals/subject.py`. Modified (existing test, 1, W3-13):
`tests/evals/test_subject_contract.py` (nine tests and one assertion added; none changed).

Deleted: none. Not created: `evals/baselines/BL-0002…`; `CANONICAL` unchanged.

Size: 888 production lines added and 32 removed; 72 added and 8 removed in the harness adapter;
1,364 test lines.

### 11.4 The contract now on record

- `ScanResult.coverage`: `status` (`full` / `partial` / `bypassed`), `layers` (all twelve, one
  of `ok`, `unavailable_dependency`, `unavailable_model`, `unavailable_load_failed`, `error`,
  `timeout`, `disabled`, `bypassed`), `classifier`, `optional` (meta-classifier, tiebreaker,
  translation; never lower the status).
- `ScanResult.zone`: `allow`, `uncertain_allow`, `uncertain_block`, `clear_block`.
  `decided_by`: `pipeline`, `tiebreaker`, `fail_secure`, `config`, `feedback_override`.
  Invariant enforced at construction: `is_attack == (zone in {uncertain_block, clear_block})`.
- `ScanResult.models`: per role, `loaded`, `version`, 16-hex `digest` of the loaded file,
  `threshold`, `backend`.
- `degraded_layers` includes unavailable layers. `schema_version = 2`. `to_dict()` on
  `ScanResult`, `LiteScanResult`, `GuardResult`.
- An allowed result may have partial coverage (W3-4). Policy on that is not decided here.

### 11.5 Results

| Item | Result |
| --- | --- |
| Tests | 785 passed, 0 failed, 0 skipped: 87 existing + 239 harness + 318 security + 141 contract. All 635 earlier test names still pass |
| Verdicts | Unchanged. 56 literal comparisons captured from `550226d` (classifier loaded and missing, uncertain band, lite, pre-flight); golden file unmodified |
| Canonical profile | Ten record files (8,032 records) byte-identical to `BL-0001`; eight counts equal |
| Lite profile | 1,848 records identical in every field except `degraded`, which is `["pair_classifier"]` in all of them. False full coverage: 1,848 → **0 of 1,848** |
| `python -m evals baseline` | Exit 6 by design: differs from `BL-0001` in `lite.std.jsonl` only |
| `verify-determinism --all` | Exit 0: 13 gated files identical across two runs; 0 differences under a random hash seed |
| Network | 0 guard violations; no dependency and no network or ML import added |
| `/health/deep` | Returns a fixed state code instead of the loader's text; response schema unchanged |
| Mutation checks | 8 of 9 deliberate faults failed tests; the ninth is absorbed by validation |
| Cost | About 23 µs per scan; about 0.26 s once per process at cold start (hashing the 90 MB encoder) |

### 11.6 Deviations

Eleven, DV-1 to DV-11, in the execution report. The ones the owner should know: **DV-5**
(`BL-0002` not pinned: `evals pin` refuses an uncommitted tree under D-013, and the guard was
not bypassed); DV-1 (the classifier's state travels on a tagged tuple, not an exception, so
`_layer_pair` keeps its return contract for two scripts outside the package); DV-3 (`bypassed`
is also the meta-classifier's optional state in a bypassed result); DV-4 (`timeout` covers a
scan that arrives during another thread's first model load); DV-6 (three to four times the
estimated size); DV-8 (`to_dict()` output is canonical, not only fixed-order); DV-10 (the
`/health/deep` code replaces the text for admins too).

### 11.7 New findings

| # | Finding | Status |
| --- | --- | --- |
| NF-10 | The loader marks its load attempted when it starts, so a scan arriving during the first load runs without the classifier | Pre-existing. Now reported as `timeout` instead of `ok`. Not fixed: fixing it changes that scan's verdict |
| NF-11 | A failed Tier 2.5 translation with language confidence below 0.90 leaves no trace; reported `translation: not_needed` | Needs `fie/multilingual.py` |
| NF-12 | `layers_fired` and `evidence` key order follow thread completion order | Pre-existing. `to_dict()` sorts |
| NF-13 | CI's placeholder `GROQ_API_KEY` makes pre-existing uncertain-band SDK tests attempt a real call to the tiebreaker provider | Pre-existing, not fixed |

### 11.8 Owner actions

1. Review and commit (command below).
2. Pin `BL-0002`: `python -m evals baseline` (exit 6 expected), `python -m evals
   verify-determinism --all`, `python -m evals pin <run> --canonical --determinism-proof
   <DETERMINISM.json>`, then `python -m evals baseline` (exit 0). The baseline should be named
   `BL-0002_pair-v6.3b_fie-70de7053`.
3. Decide DV-3 if the default is not wanted.

### 11.9 Deferred

`risk_score`; the `decision` field; the inert `threshold=` warning; persisting the zone in
server records or returning it from `/monitor`; any policy on partial coverage (WP-004,
WP-007); NF-10 to NF-13.

### 11.10 Recommended manual commit command

Not executed by the assistant.

```text
git add -- fie/adversarial.py fie/layers/pair.py fie/onnx_encoder.py fie/_lite.py fie/preflight.py fie/__main__.py fie/__init__.py evals/subject.py app/main.py tests/test_scan_result_contract.py tests/evals/test_subject_contract.py docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_003_truthful-scan-result.md docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md && git commit -m "fie: report scan coverage, zone and model identity truthfully (WP-003)"
```

---

## 12. WP-004 implementation record

Detail and evidence are in
[EXECUTION_004_runtime-policy.md](EXECUTION_REPORTS/EXECUTION_004_runtime-policy.md).

### 12.1 Numbering

**D-024 — this package is WP-004, although the roadmap calls the policy engine WP-007.**
*Status: approved by the owner (W4-1).* [ROADMAP.md](ROADMAP.md) uses WP-004 for "install
truth" (E51), WP-005 for the hermetic default and WP-007 for the policy engine. Runtime policy
was pulled ahead and took the number WP-004. The roadmap was not edited. Its install-truth and
hermetic-default topics are not done and need new numbers when scheduled; its WP-007 sketch
(policy documents, per-category thresholds, per-tenant revisions, `risk.score`) is future work
that this package is the first step toward.

D-003 (detection and policy are separate; the default reproduces today's verdicts) moves from
*proposed* to **implemented in part**: the separation exists and the default preset is exact.
The three-way `allow` / `review` / `block` surface exists as policy actions; nothing was
re-calibrated.

### 12.2 Approval

Approved for implementation on 2026-10-10 on revision 3 of
[PLAN_004](IMPLEMENTATION_PLANS/PLAN_004_RUNTIME_POLICY.md), with decisions W4-1 to W4-16 and
Q1 to Q7 (plan §14). Standing rule unchanged: the assistant makes no commit, stage, push,
merge, deploy or branch change.

Two things were decided during execution:

- **The plan file stayed untracked.** The tree was not strictly clean at the start because the
  approved plan had not been committed. The owner authorized proceeding with it untracked.
- **D-025 — a second existing assertion may be edited.** *Status: approved by the owner.* The
  plan authorized one edit in `tests/test_scan_result_contract.py` (the
  `GuardResult.to_dict()` key tuple). The approved CLI `policy` key also breaks a second
  pinned tuple, missed in planning. Raised as a stop condition before any file was changed;
  the owner authorized exactly that second edit and no other.

### 12.3 State

| Item | Value |
| --- | --- |
| Implementation date | 2026-10-10 to 2026-10-11 |
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD before and after | `924c70d2b707b9a60ab4c7053bd50dbcf2ce02f9` |
| Commit status | **NOT COMMITTED — owner will review and manually commit** |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged. No `BL-0003` |
| Deployed | No |
| Status | Complete; all required gates passed; no blocked gate |

### 12.4 What exists now

- `fie/policy.py` (new): `Policy`, `PolicyDecision`, `decide(result, policy, *,
  scan_failed=False)` (pure), `invalid_config_decision`, presets `compat@1` (default, two
  forms told apart by digest), `strict@1`, `review@1`; separately `active_policy`,
  `set_policy`, `reset_policy`, `PolicyConfigError`.
- Actions `allow`, `review`, `block`, `error`; fifteen fixed reason codes; every decision
  names its policy id and digest and repeats the detector's verdict and zone beside the
  action.
- `preflight_check` resolves the policy once before scanning (per-call `policy=` >
  `set_policy` > cached `FIE_POLICY` > `compat@1`), applies it, and reports `action`,
  `policy`, `reasons` on `GuardResult`. An invalid `FIE_POLICY` gives a blocking `error`
  (`invalid_policy_config`) without scanning; there is no fallback.
- `FIE_SCAN_FAILURE_MODE` keeps its single reader in `fie/preflight.py`; the `compat` form
  follows that flag at each call.
- Warn-only waives detector-origin blocks only.
- CLI: `policy` appended to both JSON outputs; exit code is still the detector's verdict.
- Modified: `fie/preflight.py`, `fie/__main__.py`, `fie/__init__.py`; two assertions in
  `tests/test_scan_result_contract.py`. New: `fie/policy.py`, `tests/test_runtime_policy.py`
  (157 tests). Not touched: `fie/adversarial.py`, layers, integrations, `fie/monitor.py`,
  `app/`, `engine/`, `storage/`, `evals/`, baselines.

### 12.5 Enforcement limits — read before relying on a preset

**WP-004 is not enforcement across every entry point.** Policy is applied in
`preflight_check` only.

| Gap | Consequence |
| --- | --- |
| OpenAI and Anthropic integrations call `scan_prompt` directly and fail open at DEBUG | No policy, under any preset |
| `FIEMiddleware` local mode fails open; server mode posts to a route that does not exist | No policy; server mode protects nothing |
| Server routes read `GuardResult.blocked` only, and allow if the guard raises | A selected preset is enforced there through `preflight_check`, but `action` and `reasons` are dropped |
| Shipped images set no `FIE_POLICY`; the server does not refuse to start on an invalid one | Hosted service stays on `compat@1` |
| CLI exit code | Detector verdict, never a policy gate |

A test asserts the first gap as current behaviour so that the follow-up must change it on
purpose. Recommended next package: migrate the three integrations (follow-up F3), which
changes failure behaviour for existing users and needs its own review. Then the server
(F4). Token revocation and key hashing (WP-017) should precede any reliance on
admin-controlled policy.

### 12.6 Results

| Item | Result |
| --- | --- |
| Tests | 942 passed, 0 failed, 0 skipped: 785 existing (all by name) + 157 new |
| `python -m evals baseline` | Exit 0; eight counts equal; all 11 record files byte-identical to `BL-0002`. Run flagged `canonical=False` only because the change is uncommitted (D-013) |
| `verify-determinism --all` | Exit 0; 13 gated files, 0 differing; 0 differences under a random hash seed |
| `compat@1` versus previous behaviour | 36 comparisons with values captured before the change, including warn-only and both scanner-failure modes: identical |
| Mutation checks | 18 of 18 caught; files restored, SHA-256 verified |
| Cost | About 5 µs per decision; no cold-start cost |
| Baseline replay | `strict@1` on a base install blocks all 1,848 prompts (500 of 500 benign); `review@1` moves 294 of 2,016 canonical records from block to review |

### 12.7 Deviations and risks

Eleven deviations, DV-1 to DV-11, in the execution report. The ones to know: DV-2 (the second
assertion, D-025); DV-4 (`decide` first measured at 14.9 µs against a 5 µs budget; fixed by
computing the policy digest once at construction; final 4.98 µs, a thin margin); DV-5 (a
blocked scanner failure reports the existing fail-secure marker under every policy); DV-9
(larger than estimated again).

Risks noted: one INFO audit line per allowed scan on a base install under the default;
`FIE_POLICY=strict` is easy to over-read given §12.5; `FIE_SCAN_FAILURE_MODE=open` with
`FIE_POLICY=strict` blocks scanner failures by design; NF-10 means `strict@1` blocks requests
that arrive during the first model load, so `warmup()` before serving matters more.

### 12.8 Deferred

Integrations (F3); server and deployment (F4); per-tenant policy; required model digests;
`risk_score`, the `decision` field, the inert `threshold=` warning; NF-10, NF-11, NF-13.

### 12.9 Recommended manual commit command

Not executed by the assistant.

```text
git add -- fie/policy.py fie/preflight.py fie/__main__.py fie/__init__.py tests/test_runtime_policy.py tests/test_scan_result_contract.py docs/fie_rebuild_2026/IMPLEMENTATION_PLANS/PLAN_004_RUNTIME_POLICY.md docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_004_runtime-policy.md docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md && git commit -m "fie: add runtime policy layer applied in preflight_check (WP-004)"
```

---

## 13. WP-020 / E51 record: installation truth, Stages A to C1

Detail and evidence are in
[EXECUTION_020_install-truth-stageC1.md](EXECUTION_REPORTS/EXECUTION_020_install-truth-stageC1.md)
and under `evidence/E51/`. The plan is
[PLAN_020_INSTALL_TRUTH.md](IMPLEMENTATION_PLANS/PLAN_020_INSTALL_TRUTH.md), revision 6.
This is a measurement package: nothing in `fie/`, `tests/`, packaging, dependencies, models,
workflows, `dist/` or the baselines was changed.

### 13.1 Scope actually approved and executed

| Stage | Approved scope | Executed |
| --- | --- | --- |
| A | Read-only preflight | Yes |
| B, B remediation | Release and provisioning checks; a controlled source build of `langdetect` as a recorded exception | Yes |
| C1 | **Base installs only**: one mode A cell, three mode B cells | Yes, 2026-10-11 to 2026-10-12 |
| `[ml]`, current tree (O-21) | **Not approved** | Not provisioned |
| `[ml]`, PyPI (O-22) | **Not approved** | Not provisioned |

### 13.2 Decisions

- **D-026 — the package is WP-020.** *Status: approved by the owner (O-1).* The roadmap's
  "install truth" topic (its WP-004) runs under this number; see D-024.
- **D-027 — isolation is airplane mode, switched by the owner by hand, on the existing Python
  3.10.19 interpreter.** *Status: approved by the owner.* No script changes a network setting.
  The per-program firewall and private-interpreter design in the plan is superseded and was
  not executed.
- **D-028 — two install modes, kept apart.** *Status: approved by the owner (O-24).* Mode A:
  pip builds `langdetect` from PyPI's source archive (one cell). Mode B: a pre-built local
  `langdetect` wheel (the matrix). Mode B is never described as what `pip install fie-sdk` does.
- **D-029 — neither `[ml]` set is downloaded or installed without its own approval.**
  *Status: owner decision; O-21 and O-22 not approved.*
- **D-030 — a failed verification tool is not a package finding, and a blocked probe is not a
  completed measurement.** *Status: approved by the owner after offline attempts 4 and 5.* The
  runner classifies `pip inspect` as `tool_failed`, `output_unparseable`, `match` or
  `mismatch`, and each cell as `SCANS_COMPLETED`, `IMPORT_BLOCKED_NO_SCAN_COMPLETED` and so on.
  Gate G7 is unchanged.

### 13.3 Evidence (measured, offline, gated)

- **A base install of `fie-sdk` cannot be imported.** `import fie` raises
  `ModuleNotFoundError: No module named 'starlette'` in all four base cells: the published
  wheel `1.18.0` (`be504dc7…579e963d`) in both install modes, and A-clean and A-release built
  from `2f62ecd`. Chain: `fie/__init__.py` → `fie/integrations/__init__.py:25` →
  `fie/integrations/fastapi.py:28`. The command line fails the same way.
- **Scan records produced: 0 in every cell.** Every runtime question for base installs is
  blocked. No comparison with `BL-0002` was possible. This is not evidence about detection
  quality in either direction.
- Installation works in both modes and every post-install gate passed in all four cells.
- With bundled pip 23.0.1, `langdetect` is installed from source by the legacy
  `setup.py install` path (E-16 confirmed by measurement).
- C9: the 69 installed `langdetect` package files are byte-identical in mode A and in all
  three mode B cells; only installer-generated metadata differs.
- No pickle or model file was opened. The published wheel ships a `meta_clf.pkl` that does not
  match the manifest (static).
- Isolation: in both measuring runs no adapter was up, no usable default route existed and TCP
  was refused through every launcher and the base interpreter, at the start and the end;
  recovery was verified. Not a packet capture.
- Regression state: 942 tests pass and `python -m evals baseline` exits 0 against `BL-0002`,
  at the start and after each harness change. `CANONICAL` unchanged.

### 13.4 Not established

- Anything about what the installed package does once imported.
- Whether an install with an extra that brings `starlette` imports. Predicted, not measured.
- Mode A with a pip newer than 23.0.1.
- Any `[ml]` cell. Static reading: neither resolved `[ml]` set contains `starlette`, so the
  same import failure is predicted there.

### 13.5 Deviations

Ten, all in the execution report, section J. Three were defects in the experiment harness,
found by running it and corrected with the owner's knowledge: a route check that was too strict
(DV-2), a `pip inspect` encoding failure misread as missing packages (DV-3), and a run with
zero scans labelled completed (DV-6).

### 13.6 Open for the owner

1. The consequence of the import failure for the published `1.18.0` release.
2. The fix, which belongs to WP-021 (packaging): lazy or guarded integration imports, or a
   declared dependency; and a clean-environment `import fie` test for base installs.
3. Whether E51's `[ml]` cells run before or after that fix.
4. Removal of the experiment's scratch environments. Nothing was deleted.

### 13.7 Recommended manual commit command

Not executed by the assistant.

```text
git add docs/fie_rebuild_2026 && git commit -m "docs: E51 Stage C1 base-install results - fie-sdk base install cannot be imported (WP-020)"
```

---

## 14. WP-021 record: packaging and base-import correctness

Plan: [PLAN_021_PACKAGING_BASE_IMPORT.md](IMPLEMENTATION_PLANS/PLAN_021_PACKAGING_BASE_IMPORT.md),
revision 1, proposed. Each phase is authorized separately.

### 14.1 Phase 0 (2026-10-12): the release check reproduces the defect

Detail and evidence:
[EXECUTION_021_phase0-release-check.md](EXECUTION_REPORTS/EXECUTION_021_phase0-release-check.md),
`evidence/WP021/phase0/`.

- **Authorized scope:** a new `scripts/release_check.py` implementing Level 1 (installation)
  and Level 2 (SDK import) of the plan's §5, and one run of it on a wheel built from the
  unmodified tree. No product fix.
- **Result:** on a wheel built from a clean `git archive` of `becc2c7`
  (`fie_sdk-1.18.0-py3-none-any.whl`, 151,936 bytes, SHA-256 `cfdf1859…dceaf4e4`), AC-1 to
  AC-4 pass; **AC-5 `import fie` fails with `ModuleNotFoundError: No module named
  'starlette'`**; AC-7 (command line) fails the same way; AC-6 and AC-8 are not evaluated
  because the import did not complete. **The tool exits 1.** This is the plan's AC-24: the
  gate fails on the known defect for the documented reason.
- The wheel's 37 package files are byte-identical to E51's A-clean; the result is consistent
  with E51 Stage C1 and was obtained with an independent tool.
- Regression: 942 tests pass; `python -m evals baseline` exits 0, byte-identical to `BL-0002`.
  `CANONICAL` unchanged.
- Changed: only `scripts/release_check.py` (new), this section, the execution report and the
  evidence. `fie/`, `pyproject.toml`, `tests/`, workflows, models and the manifest are
  untouched.
- Not established: anything about detection. The tool's passing path has not yet been
  exercised by a real wheel.

**Open:** Phase 1 (lazy integration import and the `fastapi` extra) is not authorized. The
plan's owner decisions W21-1 to W21-14 are undecided. The remaining E51 `[ml]` cells stay
unrun pending a separate owner decision.

Recommended manual commit command, not executed by the assistant:

```text
git add -f scripts/release_check.py && git add docs/fie_rebuild_2026 && git commit -m "scripts: add installed-wheel release check; reproduces base-import failure (WP-021 phase 0)"
```

### 14.2 Phase 1 (2026-10-12): base import correctness

Detail and evidence:
[EXECUTION_021_phase1-base-import.md](EXECUTION_REPORTS/EXECUTION_021_phase1-base-import.md),
`evidence/WP021/phase1/`.

**Decisions, approved by the owner for this phase:**

- **D-031 (W21-1) — the FastAPI integration is loaded on request.** `import fie` no longer
  imports `fie.integrations.fastapi`. Starlette is not made a base dependency.
- **D-032 (W21-2) — `fie.integrations.__all__` is `["openai", "anthropic"]`.** The documented
  import `from fie.integrations.fastapi import FIEMiddleware` is unchanged.
- **D-033 (W21-3) — a `fastapi` extra, `starlette>=0.52.1`.** 0.52.1 is the only version the
  middleware has been tested against; the bound states that and nothing more.

**Result:**

- Changed: `fie/integrations/__init__.py`, `fie/integrations/fastapi.py`, `pyproject.toml`
  (one extra added); new `tests/test_base_import.py` (21 tests) and
  `tests/test_fastapi_integration.py` (11 tests). `fie/__init__.py` and every other source file
  are untouched.
- A wheel built from a clean export with these changes (`24962aae…c656c85018`, version still
  `1.18.0`, a local test artifact) **installs and imports in a base environment**:
  `scripts/release_check.py` exits 0 with all eight criteria passing, where the Phase 0 wheel
  exits 1.
- Two fresh environments. T-base: no Starlette installed; `import fie`, all 22 public names
  and the command line work; an explicit FastAPI request raises `ModuleNotFoundError` with
  `name == "starlette"` and the install command. T-fastapi (`[fastapi]`, Starlette 0.52.1):
  the middleware imports, blocks a known injection with the documented 400 response and passes
  a benign request.
- The new tests fail on the unfixed commit (14 failed, 7 errors) and pass on the fix.
- `python -m evals baseline`: exit 0, byte-identical to `BL-0002`. `CANONICAL` unchanged.
- **Full suite: 973 passed, 1 failed.** The failure is
  `tests/test_runtime_policy.py::test_detector_source_files_are_untouched`, a WP-004 test that
  asserts `git status --porcelain` is empty for a list of paths including `fie/integrations`.
  It fails because the authorized edits are uncommitted, and is predicted to pass once they are
  committed. The test was not edited.

**Not established:** detection quality on any artifact. The published `1.18.0` is unchanged
and still cannot be imported in a base install. Nothing was released; no version was changed.

**Open:** the failing guard test (commit and rerun; consider narrowing it as a separate
change); `scripts/release_check.py` is still git-ignored and untracked; Phases 2 to 5 are not
authorized; W21-4 to W21-14 are undecided; the remaining E51 `[ml]` cells stay unrun.

Recommended manual commit command, not executed by the assistant:

```text
git add -f scripts/release_check.py && git add fie/integrations/__init__.py fie/integrations/fastapi.py pyproject.toml tests/test_base_import.py tests/test_fastapi_integration.py docs/fie_rebuild_2026 && git commit -m "fie: load the FastAPI integration lazily so a base install imports (WP-021 phase 1)"
```

### 14.3 Phase 2 (2026-10-12): the release gate in CI

Detail and evidence:
[EXECUTION_021_phase2-ci-release-gate.md](EXECUTION_REPORTS/EXECUTION_021_phase2-ci-release-gate.md),
`evidence/WP021/phase2/`.

- **Phase 1 after commit:** the owner committed Phase 1 as `0fa17a9` and the release-check
  script as `630e75f` (it had been left out by the ignore rule `scripts/*`). At that state the
  WP-004 guard test passes, 974 tests pass and `python -m evals baseline` is byte-identical to
  `BL-0002`.
- **Changed:** `scripts/release_check.py` (two targets, Level 3 criteria, per-target exit
  codes, POSIX handling); `.github/workflows/ci.yml` (`package` job runs the gate on the wheel
  it built and uploads the report); new `tests/test_release_check.py` (23 tests).
- **The gate.** T-base: the wheel alone installs, imports, starts its CLI, has neither
  Starlette nor FastAPI installed or imported, and an explicit FastAPI request fails by name.
  T-fastapi: the wheel with `[fastapi]` imports the middleware from the installed copy, which
  answers a known injection with the documented 400 and passes a benign request. One
  environment at a time; `NOT_EVALUATED` and `TOOL_FAILED` are never a pass.
- **Controls, local, Windows.** Wheel from `630e75f` (`eb251ccb…14b3e37d`): exit 0, both
  targets pass. Wheel from `becc2c7` (`cfdf1859…dceaf4e4`, byte-identical to the Phase 0
  wheel): exit 1, `import fie` fails in both targets with `ModuleNotFoundError: No module named
  'starlette'`.
- Regression after the change: 997 tests pass; baseline exit 0, byte-identical to `BL-0002`.
  `CANONICAL` unchanged.

**Not established.**

- **The workflow change has not run on a CI runner.** CI triggers on `main` only and nothing
  was pushed. The plan's condition "the CI job fails on a broken branch and passes on the fix"
  is met locally only. The script's POSIX path is untested.
- Detection quality, on any artifact.
- Cross-platform build identity: this checkout exports CRLF, a Linux build would have LF.

**Security findings recorded, not fixed** (execution report, section J):

- SF-1: `FIEMiddleware` forwards the request when the scan itself raises.
- SF-2: `FIEMiddleware` calls `scan_prompt` directly; the WP-004 runtime policy, coverage and
  zone do not apply to it.
- SF-3: it inspects only POST/PUT/PATCH JSON bodies with recognised fields on non-exempt paths.

A passing release gate does not cover these and is not a statement that the integration is
safe to rely on in production.

**Open:** the first CI run; Phases 3 to 5 not authorized; W21-4 to W21-14 undecided; the
remaining E51 `[ml]` cells unrun.

Recommended manual commit command, not executed by the assistant:

```text
git add scripts/release_check.py .github/workflows/ci.yml tests/test_release_check.py docs/fie_rebuild_2026 && git commit -m "ci: run the installed-wheel release gate on T-base and T-fastapi (WP-021 phase 2)"
```

### 14.4 Phase 3, Stage 3.0 (2026-10-13): CI evidence and model provenance

Detail and evidence:
[EXECUTION_021_phase3-stage3_0-evidence.md](EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_0-evidence.md),
`evidence/WP021/phase3/stage3_0/`. Documentation and evidence only; no source, test, manifest,
build, workflow or model file was changed, and no model file was deserialized.

**CI, verified from GitHub's records and the uploaded reports.**

- Run `38282217538` (pull request, `28683d4`): success on `ubuntu-24.04`, Python 3.11.17.
  The release gate passed: T-base 11 of 11 criteria, T-fastapi 10 of 10 (Starlette 1.7.0).
  Wheel SHA-256 `1b149e77fda073783af47d77caef9d3fc2e16e8a1f846f5b3407e0c51274bf40`, 60 files,
  18 bundled model files. `deploy-space` skipped.
- Run `38282733913` (push to `main`, `d2b0c16`): success, all nine jobs including
  `deploy-space`. Same wheel digest, same gate result.
- This closes the open item of 14.3: the gate has now run on a CI runner, on Linux, installing
  from PyPI. Two independent CI builds produced a byte-identical wheel.

**`meta_clf.pkl`, verified.** The published 1.18.0 wheel carries
`01be1995d70f1963ae4f912468642f79f27fd0d68d9135187f0592329c162c9e`, which is what the manifest
listed at the wheel's own tag. The manifest at HEAD, the working tree, the current release asset
and BL-0002's fingerprint all carry
`be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6`. The release
`models-v1.18.0` was changed in place on 2026-08-11, the day after publication; commit
`08d5474` updated the manifest and left the release tag name unchanged. Inferred, not
verified: that the replaced asset was the first digest; that it is the pre-fix model. Unknown:
whether either file is safe to deserialize.

**Seven unmanifested classifiers in `fie/models/`, verified.** All untracked, git-ignored,
absent from the manifest, the release and the PyPI wheel. `_v2` equals a blob committed on
2026-05-17 and removed from tracking on 2026-05-24; the other six were never committed. Only
`_v2` is reachable through `FIE_PAIR_VERSION`. The two `v64` files and the `v65` file are read
by `scripts/eval_harmaug.py`'s own `joblib.load`. Three have no tracked reader. The stale
`dist/` wheel contains four of them. Their origin is unknown. None was moved, deleted, loaded
or added to an inventory.

**Owner decisions required before Stage 3.1, neither made:**

- **W21-7** — which `meta_clf.pkl` is intended. Recommended: the manifest's `be6673d0…`.
- **W21-20** — whether the owner moves the seven experiment files out of `fie/models/`.
  Recommended: yes. Consequences for `scripts/eval_harmaug.py` and the training scripts are in
  the report, section H.

Recommended manual commit command, not executed by the assistant:

```text
git add docs/fie_rebuild_2026 && git commit -m "docs: WP-021 phase 3 stage 3.0 - CI evidence and model provenance"
```

### 14.5 Phase 3, Stage 3.1 (2026-10-13): standalone model-integrity primitives

Detail and evidence:
[EXECUTION_021_phase3-stage3_1-integrity-primitives.md](EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_1-integrity-primitives.md),
`evidence/WP021/phase3/stage3_1/`.

**Status: partially complete, with a decision blocker.**

**Owner decisions recorded:**

- **D-034 (W21-7), approved by the owner** — the intended `meta_clf.pkl` is
  `be6673d094a879f9211318edec9750f144adf10b58788ecb855a2969f5cbe9a6`. The tracked manifest
  already lists it; nothing was edited.
- **D-035 (W21-20), approved by the owner** — the seven unmanifested experiment classifiers
  and their side-cars are to be moved outside `fie/models/`. **Not yet executed.**
  Research-workflow compatibility is a separately planned and reviewed step that comes first.

**Done:**

- `fie/_model_integrity.py` (new): standard library only. Parses and validates a manifest;
  selects an entry by exact role and version, or the entry explicitly marked default; resolves
  a path inside the model directory; reads a file once and returns the bytes it verified by size
  and SHA-256. Five distinct errors: invalid manifest, path violation, not listed, missing,
  mismatch. It deserializes nothing, falls back to nothing, and decides no layer state.
- **The model loader does not use it.** No file in `fie/` refers to it and `import fie` does
  not import it; a test asserts both.
- `tests/test_model_integrity.py` (new): 144 tests on synthetic fixtures, 143 passed, 1 skipped
  (a symbolic-link case that this Windows machine cannot set up; a directory-junction test
  covers the same protection here).
- Regression: 1140 tests pass, 0 fail, 1 skipped; `python -m evals baseline` exits 0,
  byte-identical to `BL-0002`. `CANONICAL` unchanged.

**Not done, blocked:** the new fields in `scripts/model_manifest.json`, the derived
`fie/models/MANIFEST.json`, and the change to `scripts/download_models.py`. The schema requires
`in_wheel` on every entry, and its value depends on decisions the owner has not made. Neither
file was touched.

**Decisions needed to finish Stage 3.1:** W21-8 (models in the one wheel or not); W21-15 (the
two pickles nothing in `fie/` loads); W21-22, new (the four unlisted JSON files beside the
encoder); W21-23, new (fields added in place to the tracked manifest, and the role names).
Still open for later stages: W21-9, W21-16 to W21-19.

Recommended manual commit command, not executed by the assistant:

```text
git add fie/_model_integrity.py tests/test_model_integrity.py docs/fie_rebuild_2026 && git commit -m "fie: add standalone model-integrity primitives, not yet used by the loader (WP-021 phase 3 stage 3.1)"
```

### 14.6 Phase 3, Stage 3.1 follow-up (2026-10-13): filesystem failures classified

Detail and evidence:
[EXECUTION_021_phase3-stage3_1-followup-io-errors.md](EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_1-followup-io-errors.md),
`evidence/WP021/phase3/stage3_1_followup/`.

- **The gap, as found.** In `fie/_model_integrity.py` a `PermissionError` or other `OSError`
  on open or read escaped raw, with the absolute path in its text. Inspection showed more: a
  refusal while examining the file was silently reported as `missing` or as `path_violation`,
  because `os.path.lexists`, `isfile` and `isdir` swallow every error.
- **The change.** A sixth error class, `ReadFailed`, code `read_failed`, carrying the relative
  path, the operation (`resolve`, `open`, `read`), the exception class name and the error
  number. Every filesystem step now distinguishes absence (`missing`) from refusal
  (`read_failed`). Only `OSError` is classified; other exceptions propagate unchanged. The
  operating system's message is not copied and the new errors are raised `from None`, so no
  absolute path reaches a message or a logged traceback.
- **Unchanged.** One open, and the bytes returned are the bytes checked. The other five codes
  keep their meaning. No layer-state or policy mapping (W21-9 open). The loader does not use
  the module. No manifest, downloader, build or workflow change.
- **Tests.** 38 added, none modified: 180 pass and 2 skip in the file (the symbolic-link case,
  and a real unreadable file, neither producible on this Windows machine; both run on Linux).
  Full suite 1177 passed, 0 failed, 2 skipped. `python -m evals baseline` exits 0,
  byte-identical to `BL-0002`. `CANONICAL` unchanged.

Stage 3.1 as a whole is still partially complete: the tracked manifest and the download script
wait on W21-8, W21-15, W21-22, W21-23 and the meta-classifier's version name.

Recommended manual commit command, not executed by the assistant:

```text
git add fie/_model_integrity.py tests/test_model_integrity.py docs/fie_rebuild_2026 && git commit -m "fie: classify filesystem failures in the model-integrity primitives as read_failed (WP-021 phase 3 stage 3.1 follow-up)"
```

### 14.7 Phase 3, Stage 3.1 completion (2026-10-13): production inventory and derived manifest

Detail and evidence:
[EXECUTION_021_phase3-stage3_1-completion-manifest.md](EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_1-completion-manifest.md),
`evidence/WP021/phase3/stage3_1_completion/`.

**Status: Stage 3.1 complete.** The loader is not connected (Stage 3.2, not started).

**Owner decisions recorded and implemented:**

- **D-036 (W21-8)** — the supported SDK model artifacts ship in the one wheel.
- **D-037 (W21-15)** — `fie/models/failure_classifier_v4.pkl` and `feature_columns_v4.pkl`
  stay in the inventory with role `legacy` and `in_wheel: false`.
- **D-038 (W21-22)** — the five encoder by-products in `fie/models/minilm-onnx/` (`config.json`,
  `ort_config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt`) are
  outside the model-artifact inventory.
- **D-039 (W21-23)** — `scripts/model_manifest.json` is extended in place and stays the only
  authoritative inventory; `fie/models/MANIFEST.json` is derived from it. Roles:
  `pair_classifier`, `pair_meta`, `meta_classifier`, `meta_classifier_meta`, `encoder`,
  `tokenizer`, `server`, `legacy`. `version` is the `FIE_PAIR_VERSION` selection key (`v6_3b`,
  `v6_3`, `v6`, `v5`, `v4`, `v3`, `v1`), not the side-car's declared version. Singleton roles
  are selected by role alone, with no invented version. All seven classifier versions ship.
- **D-040** — `.gitkeep` is to be excluded from the wheel. **Recorded, not implemented**:
  reserved for Stage 3.3.
- D-034 (W21-7) and D-035 (W21-20) stand. The relocation under W21-20 is still **not executed**.

**Done:**

- `fie/_model_integrity.py`: `sole_entry`, `validate_production`, `derive_package_manifest`.
  The validator pins the role vocabulary, the seven keys and their files, the single `v6_3b`
  default, the roots, and `in_wheel` by role, so that a legacy or server file cannot enter the
  wheel and an SDK model cannot leave it by editing the inventory alone.
- `scripts/model_manifest.json`: `schema: 1` and the new fields on all 25 entries; 90 lines
  added, none deleted; every path, digest, size and note identical to before. 16 entries are
  `in_wheel: true`.
- `scripts/download_models.py`: still downloads and strictly verifies every entry whatever
  `in_wheel` says; `--check` stays read-only; the derived manifest is written, atomically,
  only after every artifact is verified, and never on a partial or failed run.
- `tests/test_model_manifest.py`: 153 tests, synthetic inventories and a mocked download.
- The derivation is deterministic: 5,112 bytes, SHA-256 `fbab28ab…393bf2f5` for today's
  inventory.

**Verified:** 1330 tests pass, 0 fail, 2 platform skips. `python -m evals baseline` exits 0;
all 11 per-prompt record files byte-identical to `BL-0002`. The harness fingerprint differs
from the pinned one in six identity fields only (commit, dirty flags, `fie` tree, subject key);
model identities identical. `CANONICAL` unchanged; the baseline was not rewritten.

**Not established, not done:**

- Nothing is enforced at run time yet: the loader still uses `joblib.load` on the first
  preferred file that exists.
- No build rule enforces `in_wheel`; `fie/models/MANIFEST.json` has not been generated here.
- CI has not run on this change.

**Open:** W21-9, W21-17, W21-18 (Stage 3.2); W21-16, W21-19 (Stage 3.3); the research-workflow
compatibility plan and the relocation of the seven experiment files.

Recommended manual commit command, not executed by the assistant:

```text
git add fie/_model_integrity.py scripts/model_manifest.json scripts/download_models.py tests/test_model_manifest.py docs/fie_rebuild_2026 && git commit -m "fie: production model inventory, validator and derived package manifest; downloader writes it after full verification (WP-021 phase 3 stage 3.1)"
```

### 14.8 Phase 3, Stage 3.2 (2026-10-14): verified model loading

Detail and evidence:
[EXECUTION_021_phase3-stage3_2-verified-loading.md](EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_2-verified-loading.md),
`evidence/WP021/phase3/stage3_2/`.

**Status: Stage 3.2 complete**, with one worktree-only test failure that clears on commit. The
first session stopped at the Step 3 gate on an existing-test conflict. A read-only audit then
found the stage partially complete, and the owner authorized the decisions below and the rest of
the work.

**Owner decisions recorded and implemented:**

- **D-041 (W21-9)** — an integrity failure is `unavailable_load_failed`, with an internal detail
  code (`manifest_invalid`, `not_listed`, `mismatch`, `read_failed`, `path_violation`,
  `deserialize_failed`). No new public layer state.
- **D-042 (W21-17)** — no silent fallback for `FIE_PAIR_VERSION`. Unset or empty selects the
  manifest default. A listed key with a missing file gives `unavailable_model` / `missing`; an
  unlisted key gives `unavailable_load_failed` / `not_listed`.
- **D-043 (W21-18)** — the repository-root `models/` fallback is removed.
- **D-044 (W21-24)** — the runtime manifest is `<package>/models/MANIFEST.json`, derived from
  `scripts/model_manifest.json`, which stays the only authoritative inventory.
- **D-045 (W21-25)** — a missing or invalid side-car makes the layer unavailable; no default
  threshold, feature list or encoder name.
- **D-046 (W21-26, W21-27)** — sanitized structured log lines and the existing `health()` error
  text carry the detail code; no new audit event. A missing manifest gives `unavailable_model`;
  an invalid one gives `unavailable_load_failed` / `manifest_invalid`.
- **D-047** — health-deep test option B (precondition without an absolute path). The two
  Stage 3.1 tests whose premise Stage 3.2 retires are replaced by stronger ones.

**Done:**

- `fie/layers/_verified_models.py` (new) is the only model deserializer in `fie/`. It reads the
  package manifest once and selects exactly. Verified bytes go to `joblib.load(io.BytesIO(...))`;
  side-cars are parsed from verified bytes, with required fields.
- `fie/layers/pair.py`: both loaders use it. Identity is computed from the deserialized bytes;
  root fallback, path hashing and the fixed version list removed. The readiness log, public
  functions and locks are unchanged.
- `fie/adversarial.py`: `warmup()` classifies by detail code, not message text.
- `fie/_model_integrity.py`: docstring status corrected only.
- Tests: `tests/test_verified_loading.py` (91 cases), plus the three authorized test updates.
- Step 0: `fie/models/MANIFEST.json` generated locally without any download, byte-identical to
  the derivation (`fbab28ab…393bf2f5`). Ignored, not committed.

**Verified:** focused 91 passed. Full suite 1420 passed, 1 failed, 2 platform skips; the failure
is `test_detector_source_files_are_untouched`, which lists exactly the three authorized
uncommitted source files. `python -m evals baseline` exit 0: all 11 per-prompt record files
byte-identical to `BL-0002`. The fingerprint differs only in identity metadata; model identities
are identical. `CANONICAL` is unchanged and the baseline was not rewritten. The deserializer
boundary scan finds only `_verified_models.py` (`joblib.load`) and the out-of-scope
`session_tracker.py` (`pickle.loads`).

**Not established, not done:**

- CI has not run on this change.
- The guard test must be rechecked after commit.
- The `SentenceTransformer` fallback, ONNX encoder files and `session_tracker.py` remain
  unverified (WP-022 or separately scoped).

**Open:** Stage 3.3 (`in_wheel` enforcement, `.gitkeep` exclusion, sdist feasibility);
W21-16, W21-19; the research-workflow plan and the experiment-file relocation (W21-20).

Recommended manual commit command, not executed by the assistant:

```text
git add fie/layers/_verified_models.py fie/layers/pair.py fie/adversarial.py fie/_model_integrity.py tests/test_verified_loading.py tests/test_scan_result_contract.py tests/test_model_integrity.py tests/test_model_manifest.py docs/fie_rebuild_2026 && git commit -m "fie: load pair and meta classifiers only through the package manifest, from verified bytes; no fallback (WP-021 phase 3 stage 3.2)"
```

### 14.9 Phase 3, Stage 3.3b (2026-10-14): manifest-driven package allowlist

Background, committed in `49b7a1f`:
[EXECUTION_021_phase3-stage3_3-plan.md](EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_3-plan.md)
(planning audit) and
[EXECUTION_021_phase3-stage3_3a-packaging-feasibility.md](EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_3a-packaging-feasibility.md)
(offline experiment). 3.3a measured on hatchling 1.29.0 that:

- the `artifacts` globs pack every `fie/models/*.pkl` and `*.json` present, including experiment
  files, the legacy pair and `.gitkeep`;
- no `exclude` overrides `artifacts`;
- `force-include` overrides `exclude`;
- a wheel-only `force-include` does not reach the sdist.

**Owner decisions recorded for this stage** (recorded before implementation):

- **D-048 (W21-16)** — a normal build fails if any `in_wheel: true` artifact is missing or fails
  its size or SHA-256 check. No model-free build flag is introduced in Stage 3.3b.
- **D-049 (W21-28)** — the sdist carries the same 16 approved production model artifacts as the
  wheel, together with the generated runtime manifest.
- **D-050 (W21-31)** — the packaged `fie/models/MANIFEST.json` is derived at build time from the
  tracked `scripts/model_manifest.json` by the existing deterministic derivation
  (`derive_package_manifest`). Any generated file already in the working tree is never packed.
- **D-051 (inclusion mechanism)** — the broad `artifacts` globs are replaced by a custom
  Hatchling build hook that validates the inventory and verifies, then explicitly includes,
  exactly the approved artifacts.
- **D-052 (W21-33)** — the build requirement is pinned to `hatchling==1.29.0`, the version
  measured in Stage 3.3a.

Still open: W21-19 (line endings), W21-29 (narrowing the sdist), W21-32 (source-tree rule for
unlisted files), W21-20-R (experiment-file relocation), W21-14 (published sdist).

**Result** (details:
[EXECUTION_021_phase3-stage3_3b-manifest-driven-package-allowlist.md](EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_3b-manifest-driven-package-allowlist.md),
`evidence/WP021/phase3/stage3_3b/`). **Status: Stage 3.3b complete; not committed.**

- `pyproject.toml`: `hatchling==1.29.0`; `artifacts` globs removed; `/fie/models/**` excluded;
  custom hook `hatch_build.py`.
- `hatch_build.py` loads `fie/_model_integrity.py` by path, validates the inventory, verifies each
  `in_wheel: true` artifact by size and SHA-256 (no deserialization), and force-includes exactly
  those plus a temporary file holding the derived manifest. It runs for the sdist and the wheel.
  Unlisted files are warned about and never packed.
- Offline acceptance (M0/M1/M2 × LF/CRLF × 2, plus stale-manifest, tampered and missing cases):
  **15/15 gates pass**.
  - Every successful sdist, direct wheel and wheel from sdist contains under `fie/models/` exactly
    the 16 approved artifacts plus `MANIFEST.json` (`fbab28ab…`).
  - M0, tampered and missing builds are refused with a named diagnostic.
  - The archive bytes are identical across M1, M2 and M1S; the direct wheel equals the wheel from
    the sdist.
  - Zero network attempts.
- Tests: `tests/test_build_artifact_allowlist.py` 32 passed; full suite 1453 passed, 0 failed,
  2 platform skips. 14 of the 32 need `build` and hatchling, which the CI test jobs do not
  install; those will skip in CI.
- Unchanged: inventory, download script, loader, models, workflows, baseline, `CANONICAL`.

**Open:** Stage 3.3c (archive gate in CI, release check and publication); W21-29 (sdist scope,
about 67 MB including third-party music files); W21-19; W21-32; W21-20-R; W21-14.

Recommended manual commit command, not executed by the assistant:

```text
git add pyproject.toml hatch_build.py tests/test_build_artifact_allowlist.py docs/fie_rebuild_2026 && git commit -m "build: package exactly the verified in_wheel model artifacts via a manifest-driven hatch build hook; pin hatchling 1.29.0 (WP-021 phase 3 stage 3.3b)"
```
