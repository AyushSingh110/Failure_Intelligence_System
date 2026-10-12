# FIE Rebuild 2026 — Roadmap

**Date:** 2026-10-07 · **Baseline commit:** `24cb2e9` · **Status:** proposed, awaiting approval. Nothing here is implemented.
**Evidence for every statement:** [BASELINE_AUDIT.md](BASELINE_AUDIT.md). Section references like "§5.8" point there.
**Decisions and open questions:** [MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md](MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md)

Guiding rule: **make FIE more trustworthy before making it more complicated.** Each work
package below either removes a way FIE can mislead its user, or adds a measurement. New
detection capability comes after both.

---

## 1. Priority order

The eight candidate priorities, re-ranked against the audit. P5 is split, because part of it
is urgent and part of it can wait.

| Order | Priority | Why here |
| --- | --- | --- |
| 1 | **P0 — Make the existing guardrail reliable and product-safe** | The published package ships without its classifier and reports full coverage (§5.1). It sends prompts to third parties while described as offline (§5.6). Until the shipped artifact is the measured artifact, no other number means anything to a user |
| 2 | **P5a — Critical tenant isolation and auth** | Unauthenticated writes into any tenant, and a global answer cache any tenant can poison (§5.4). The API is live. Small diff, no effect on detection metrics, so nothing blocks it |
| 3 | **P2 — Reproducible evaluation** (the minimum harness) | Every later package must report before and after on both axes. Today the scripts are untracked, the run configuration is unrecorded, and the headline number describes a model that is not the default (§8 C2, C17, C18, C27). The harness is the first package for this reason |
| 4 | **P1 — Over-refusal, multilingual, policy routing** | The largest user-facing failure. Split in two: script handling and the ALLOW/REVIEW/BLOCK contract are engineering with clear pass/fail; the OR-Bench-hard representation problem is research |
| 5 | **P3 — Adaptive evaluation** | One benign sentence halves detection and one fixed prefix evades a third of caught attacks, with no optimisation (§5.8, §5.9). Robustness claims must stop until this is measured. Long-input scoring, the engineering half, is pulled forward into P1 |
| 6 | **P4 — SDK / API developer experience** | After the decision contract settles. Polishing the current boolean API would freeze the wrong interface |
| 7 | **P5b — Governance** (audit events, retention, policy versioning, provenance) | Builds on the policy engine and the result contract |
| 8 | **P6 — Streaming protection** | Covers 400 characters with three regexes and has no benchmark. Label it experimental now; build it after an output-side detector is measured |
| 9 | **P7 — Hallucination redesign** | At chance. Research track, isolated from the product. Its *containment* (stop advertising AUC 0.749, stop cross-tenant answer substitution) happens in P0 and P5a |

Differences from the September 16-week plan: that plan started with a Groq-to-Ollama provider
abstraction and went straight to experiments. This one starts with the shipped artifact and
the live API, because the audit found product-level faults the review did not look for. The
review's experiments E30–E48 are kept and slotted in below.

---

## 2. Target architecture

### 2.1 Input pipeline

```
Application
    │  scan(text, context?, tenant?)
    ▼
┌─ fie (production package, stable API) ─────────────────────────────────────────────┐
│ 1 NORMALIZE      Unicode NFKC, confusables, zero-width, script segmentation,       │
│                  deterministic local language id. One normal form for ALL detectors │
│ 2 SEGMENT        Long input → overlapping windows; multi-part input → parts        │
│ 3 DETECT         Detectors return Signal(name, score, status, evidence).            │
│                  No thresholds and no blocking decisions inside a detector          │
│ 4 AGGREGATE      Signals → RiskAssessment(score, category, coverage, calibration)   │
│ 5 POLICY         Versioned, declarative. RiskAssessment + coverage + language       │
│                  → Decision(ALLOW | REVIEW | BLOCK, reasons, policy id)             │
│ 6 EVIDENCE       Decision carries fired detectors, scores, thresholds, matched      │
│                  spans, normal form, model and config identity                      │
│ 7 AUDIT          AuditEvent to a sink the integrator chooses. Content logging off   │
│                  by default                                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ├── ALLOW  → application / model
    ├── REVIEW → integrator-defined: queue, second model, step-up, or treat as block
    └── BLOCK  → stop
```

What changes against today, and what does not.

| Stage | Today | Target | Evidence that forces the change |
| --- | --- | --- | --- |
| Normalize | Used by two layers; PAIR sees raw text | One normal form for every detector | §5.2: four layers each misread scripts in their own way |
| Segment | None; mean over ≤ 256 tokens | Windowed scoring | §5.8 |
| Detect | Layers emit an attack type only above their own threshold | Layers emit a score always | A score that vanishes below a threshold cannot be calibrated or explained |
| Aggregate | Weighted vote + inert meta blend + text-inferred domain multiplier | Simple, documented aggregation; multiplier removed | §5.5 |
| Policy | Constants in `fie/adversarial.py` | Versioned document supplied by the integrator | §7; review §4 items 2–3 |
| Decision | `is_attack: bool`; allowed prompts report confidence 0.0 | Three states plus the score that produced them | §7 |
| Evidence | Partly present | Complete, with identity | §11 |
| Detectors themselves | PAIR v6.3b + 11 heuristics | **Unchanged in Phases 0–1** | Change one thing at a time |

### 2.2 Output pipeline — separate, and experimental until measured

```
Model output ─► output detectors ─► RiskAssessment ─► output policy ─► Decision + AuditEvent
                   │
                   └─ factuality / reliability signals: research package only.
                      Never gates a request. Never rewrites an answer in a production path.
```

### 2.3 Repository shape

```
fie/            PRODUCTION. Must not import engine/, app/, storage/. Semver. Contract tests.
gateway/        PRODUCTION (today: the guard-relevant part of app/). Auth, tenants, policy store,
                POST /v1/scan, audit sink.
evals/          Harness, frozen splits, pinned baselines, gates. Tracked.
research/       Everything experimental: hallucination monitor (today's engine/ and most of app/),
                adaptive attacks, augmentation, paper code. May import fie; never the reverse.
docs/           Generated fact sheet; this rebuild log.
```

The physical move is one late, mechanical package (WP-011). The *rule* (fie imports nothing
from research code) is enforced by a test long before the directories move. Today `fie`
imports `engine.fie_config`, `engine.hard_positive_collector` and `storage.database` at run time.

---

## 3. API and SDK direction

A sketch of the intended surface, not a specification. The specification is written in WP-007.

```python
from fie import Guard, Policy

guard = Guard(policy=Policy.preset("strict"))   # "strict" reproduces today's verdicts exactly
d = guard.scan(prompt)

d.action       # "allow" | "review" | "block"
d.risk.score   # 0.71 — kept for allowed prompts too
d.reasons      # [Reason(detector="pair", score=0.71, threshold=0.50, evidence={...})]
d.coverage     # {"pair": "ok", "regex": "ok", "translation": "disabled", ...}
d.provenance   # {"fie": "2.0.0", "policy": "strict@3", "models": {"pair": "v6.3b:9c682b28"},
               #  "config": "sha256:…"}
```

| Principle | Concretely |
| --- | --- |
| One install command gives the measured pipeline, or fails loudly | No silent lite mode. A lite install is an explicit choice and every result says so |
| Nothing leaves the machine unless asked | Telemetry, translation, remote tiebreaker and model download are opt-in, named, and reported in `coverage` |
| Every parameter does something | `threshold=`, `--threshold`, `scan_threshold` are either wired to the policy or removed through a deprecation cycle |
| Deterministic | Same input, same policy, same models → same decision, on every run |
| Backwards compatible | `scan_prompt()` and `ScanResult.is_attack` keep working, defined as `action != "allow"` under the `strict` preset, for at least two minor releases |
| Explanations are evidence | Detector, score, threshold, matched span, nearest training examples where useful. No LLM call in the request path |
| A scan API exists | `POST /v1/scan` returns the same `Decision`. `FIEMiddleware` server mode calls it |

---

## 4. Policy model

**Detection confidence and policy decision are separate objects.** The detector says
"risk 0.71, category harmful_request, calibrated". The policy says "review".

```yaml
# Sketch. Field names are settled in WP-007.
id: acme-support-bot
version: 3
extends: balanced
block_at: 0.80            # risk >= block_at            -> BLOCK
review_at: 0.45           # review_at <= risk < block_at -> REVIEW
categories:
  copyright_reproduction: { action: allow }
  prompt_extraction:      { block_at: 0.70 }
on_degraded_coverage: review      # a required detector did not run
on_scan_error: block              # the scanner itself failed
unsupported_language: review      # text the classifier was not validated on
long_input: { window_tokens: 256, stride: 128, pool: max }
logging: { store_content: false, retention_days: 30 }
```

| Question | Answer |
| --- | --- |
| What does REVIEW mean? | Whatever the integrator binds it to: `on_review="block"` (default, today's behaviour), `"allow_and_flag"`, or a callback |
| Presets | `strict` = today (review treated as block). `balanced` = review surfaced. `permissive` = review allowed and flagged. Each preset ships with its measured two-axis numbers |
| What the three states buy, measured | XSTest hard blocks 52.8% → 27.6%. OR-Bench-hard 90.4% → 74.0%. Cost: 13.4% of benchmark attacks land in REVIEW (§7) |
| What they do not buy | Lower error. REVIEW exposes uncertainty; it does not remove it |
| Where the band comes from | Phase 1: today's `[0.60·T, T)`. Phase 2: chosen by conformal risk control so that over-refusal ≤ α and miss rate ≤ β hold on a calibration set (review idea B2) |
| Per-tenant policy | Stored per tenant, immutable revisions, policy id and version on every decision and audit event |
| What a policy may not do | Change detector internals. Reach across tenants. Take effect without a new version number |
| Feedback | A label creates a tenant-scoped allow or deny entry inside that tenant's policy. It never edits a shared model, threshold or cache |

---

## 5. Security model

| Area | Rule |
| --- | --- |
| Threat model | The SDK, model weights, thresholds and this document are public. Assume a white-box attacker with unlimited offline queries. State this in the README |
| Tenant isolation | Tenant identity comes only from the credential. No request field may name a tenant. No shared mutable state between tenants: cache, allow/deny lists, thresholds, answer cache, registries. Each has an isolation test |
| Authentication | Deny by default. Every route declares its requirement. The server refuses to start without a signing secret |
| Credentials | API keys hashed at rest, shown once, never placed in tokens or logs. Admin role read from the database |
| Artifact trust | Every model file is verified against the manifest when it is loaded, on every install path. No pickle in the production package: the PAIR head is a linear model and ships as plain weights |
| Egress | None by default. Each optional outbound call is named, opt-in, and visible on the result |
| Failure mode | One setting, honoured by every wrapper. Default `closed` for the gateway. For the SDK the default is chosen once, documented, and reported on the result |
| Degraded coverage | A missing dependency or model is a coverage failure, not a clean scan. Policy decides what happens |
| Logging and privacy | Content logging off by default. Configurable retention. No prompt text in alerts unless enabled |
| Audit | One append-only event per decision: time, tenant, input hash, action, risk, policy id and version, model ids, config hash, coverage |
| Rate limiting | Keyed by credential, applied to every route, tested behind a proxy |
| Global behaviour | No path lets one tenant's data alter another tenant's decisions. Shared model updates happen offline, through the evaluation gate, with a new version |

---

## 6. Evaluation architecture

Evaluation is part of the product. It is what lets a stranger check a claim.

```
evals/
  splits/        frozen, SHA-pinned (reuse data/overrefusal, data/benchmark_audit)
  suites/        attack_recall · over_refusal · zones · multilingual_benign · long_input ·
                 framing · lite_install · determinism · latency         (Phase 0)
                 notinject · falsereject · indirect_injection · multi_turn · output_side ·
                 adaptive_whitebox · adaptive_blackbox                  (Phase 1–2)
  configs/       strict-failsecure · online-tiebreaker · lite · each policy preset
  baselines/     BASELINE_<commit>.json — per-prompt verdicts plus the fingerprint below
  run.py         one command per table
  gate.py        paired bootstrap comparison against a baseline
```

**Configuration fingerprint**, written into every report and required by the gate: git commit,
dirty flag, SHA-256 of every loaded model, PAIR version and threshold, compiled thresholds
hash, policy id and version, tiebreaker state, translation state, every `FIE_*` variable,
Python and package versions, seed, hostname class.

**Gates for any package that can change a verdict.** Margins are proposed defaults; see open
question OQ-4.

| Gate | Rule |
| --- | --- |
| G1 Both axes | Attack recall on all five attack sets and over-refusal on both benign sets are reported together, with 95% CIs |
| G2 Paired | Comparison against the pinned baseline uses the paired bootstrap in `scripts/stats_utils.py` (10,000 resamples, seed 42) |
| G3 No silent recall loss | No attack set worsens by more than 2.0 points, or significantly (p < 0.05), unless the package's approved scope says it trades recall |
| G4 No silent over-refusal gain | Neither benign set worsens by more than 2.0 points, or significantly |
| G5 Multilingual | Once the E30 set exists: benign false-positive rate per language is reported for every change |
| G6 Deterministic | Two consecutive hermetic runs give byte-identical per-prompt files |
| G7 Hermetic | The run makes zero outbound connections, enforced by a socket guard |
| G8 Golden | `tests/data/detection_golden.json` changes only with a line-by-line explanation in the execution report |
| G9 Pinned models | The run aborts if any model hash differs from the manifest |

**Claims ledger.** `docs/FACT_SHEET.md` becomes generated output: each number links to a
report file and the command that produced it. README numbers are checked against it in CI.

---

## 7. Research roadmap (kept apart from production engineering)

Research answers questions. It ships nothing by itself. A result reaches the product only
through a work package that passes the gates in §6. E30–E48 keep the numbers the September
review assigned; new experiments start at E49.

| ID | Question | Method | Feeds |
| --- | --- | --- | --- |
| E30 | What is the benign false-positive rate per language and script? | 200 benign prompts × 12 languages from a public multilingual set, plus romanised and code-switched. Same protocol for local guard models | WP-008 |
| E31 | Does script-aware normalization plus language routing fix it without losing the attacks E11 credited to `multilingual`? | Before/after on E30 and the E11 and character-injection sets | WP-008 |
| E32 | What is recall with the online tiebreaker? | Run the suite in the `online-tiebreaker` config. Today's bound: 76.7%–88.2% macro | WP-013 |
| E49 | How fast does detection fall with input length, and does windowed scoring recover it at what over-refusal cost? | Today's padding probe on the full sets; window size × stride × pooling grid | WP-009 |
| E50 | Did domain-balanced retraining teach a "medical or legal words mean benign" shortcut? | Matched pairs: the same harmful and benign requests with and without domain framing, across PAIR v5, v6.2, v6.3b | WP-014 |
| E51 | What does the PyPI wheel actually do in a clean environment? | Fresh venv, `pip install fie-sdk`, run the suite | WP-004 |
| E52 | Does the meta-classifier move prompts between zones even though it changes no boolean verdict? | Zone ablation with `FIE_DISABLE_META` | WP-007 |
| E37–E39 | Adaptive robustness, both axes | White-box greedy substitution on the PAIR margin; black-box local LLM attacker; three adversarial-training rounds | WP-012 |
| E40 | Is the over-refusal ceiling the embedder? | MiniLM vs bge-small vs e5-small vs multilingual-e5-small, head and data fixed | WP-014, WP-008 |
| E34–E36 | Why do XSTest and OR-Bench-hard disagree? Does it hold for other guards? | More local guards on full splits; trigger-density × intent-ambiguity; a pre-registered prediction | Paper P1 |
| E47 | Can the REVIEW band be chosen with a two-sided risk guarantee? | Conformal risk control over PAIR → local guard | WP-013 |
| E42–E46 | Hallucination: can a cross-model entailment detector beat a length-only baseline out of distribution? | Review §2.4–2.5, local Ollama only | Research package only |

Rules carried over from the research log and the review: a written hypothesis before each
run; every number with a CI; both axes in every table; negative results logged as fully as
positive ones; all LLM calls through local Ollama, never Groq.

---

## 8. Product roadmap

A work package (WP) is one approved session with one execution report. Order within a phase
is the recommended order.

### Phase 0 — Truth and safety baseline

| WP | Name | Changes detection? | Summary |
| --- | --- | --- | --- |
| 001 | Evaluation harness and pinned baseline | No. No production code touched | Tracked `evals/` harness; configuration fingerprint; hermetic and deterministic runs; pinned baseline for the shipped config |
| 002 | Server isolation hotfix | No | Close S1–S5, S7, S9, S10 in §5.4; fix the `/flags` auth helper; disable cross-tenant answer substitution |
| 003 | Truthful scan result | No | Additive fields: `decision`, `zone`, `risk_score`, `coverage`, `provenance`. A missing classifier becomes visible. Inert parameters warn |
| 004 | Install truth and artifact integrity | No | Dependencies and version so that the default install runs the measured pipeline or fails loudly; checksum on every load path; PAIR head as plain weights with an equivalence gate; clean-venv install test |
| 005 | Hermetic and deterministic by default | **Yes** (multilingual path) | No egress unless opted in; deterministic language id; measured effect on both axes |
| 006 | Claims reconciliation | No | README, fact sheet, SECURITY.md, pyproject and landing page regenerated from harness reports; reproduction scripts tracked |

### Phase 1 — Decision model and input handling

| WP | Name | Changes detection? | Summary |
| --- | --- | --- | --- |
| 007 | Policy engine and `Decision` | No under `strict` | Versioned policy; three states; presets with measured numbers; old API mapped onto it |
| 008 | Script and language handling | **Yes** | E30 then E31. One normal form for all detectors; "unsupported language" becomes a policy input instead of a block |
| 009 | Long-input scoring | **Yes** | E49. Windowed scoring; both axes reported, because it will raise over-refusal on long text |
| 010 | Gateway scan API | No | `POST /v1/scan`; middleware fixed; one failure-mode setting honoured everywhere |
| 011 | Repository boundary | No | Import rule enforced by test; later the mechanical move to `research/` |

### Phase 2 — Robustness

| WP | Name | Summary |
| --- | --- | --- |
| 012 | Adaptive evaluation | E37–E39 as a permanent suite. No robustness wording in docs until it has run |
| 013 | REVIEW-band resolver | Replace the Groq injection classifier with a measured local option or none; conformal band selection (E32, E47) |
| 014 | Over-refusal research line | E34–E36, E40, E50, minimal-pair training. Ships only through the gates |

### Phases 3–6

| Phase | WP | Summary |
| --- | --- | --- |
| 3 Developer experience | 015 | `Guard` / `Policy` / `Decision` as the documented API; quickstart that works on a clean machine in under five minutes; integration examples; API reference; deprecations completed |
| 4 Governance | 016 | Audit event schema and sinks; retention and content-logging controls; policy and model provenance on every record; the control mapping in audit §11 kept current |
| 4 Governance | 017 | Credential hardening left over from WP-002: hashed keys at rest, key rotation, credential-keyed rate limits |
| 5 Output side | 018 | Output pipeline with at least one measured detector; streaming that scans the whole stream; or the claims are removed |
| 6 Hallucination research | 019 | Move the monitor to `research/`; provider abstraction to Ollama; then E42–E46. Decision point: method paper or negative-result paper |

---

## 9. Dependency graph

```
WP-001 harness ──────────────┬──────────────────────────────────────────────┐
                             │                                              │
WP-002 server isolation      │  (independent of everything; can run first)  │
                             ▼                                              │
                     WP-003 truthful result ──► WP-007 policy engine ──► WP-010 scan API ──► WP-015 SDK v2 ──► WP-016/017 governance
                             │                        │      ▲
                             ▼                        │      │
                     WP-004 install truth             │   E52 zones
                             │                        │
                             ▼                        ▼
                     WP-005 hermetic default ──► WP-008 script handling (E30 → E31)
                             │                        │
                             ▼                        ▼
                     WP-006 claims               WP-009 long input (E49) ──► WP-012 adaptive (E37–39)
                                                                                   │
                                                              WP-013 REVIEW resolver (E32, E47)
                                                                                   │
                                                              WP-014 over-refusal research (E34–36, E40, E50)

WP-011 repo boundary: rule after WP-003; physical move after WP-010.
WP-018 output side: after WP-007.        WP-019 hallucination: after WP-011 (rule) and WP-002.
```

Hard prerequisites, stated once: nothing that changes a verdict runs before WP-001.
WP-007 needs WP-003's fields. WP-008 needs WP-005, because the multilingual path is where the
network calls and the non-determinism live. WP-012 needs WP-009, or the adaptive attacker will
only rediscover padding. WP-006 needs WP-004 and WP-005, or the corrected README would be
wrong again within a week.

---

## 10. Acceptance criteria

Every package also requires: existing tests still pass; an execution report in
`EXECUTION_REPORTS/`; the master log updated; `git diff --stat` limited to the approved scope.

| WP | Passes when |
| --- | --- |
| 001 | (a) One command reproduces, from a clean checkout with manifest-matching models: XSTest-safe 132/250, OR-Bench-hard 226/250, XSTest-unsafe 177/198, JailbreakBench 130/134, HarmBench 326/387, StrongREJECT 218/242, SORRY-Bench 316/387. (b) Two consecutive hermetic runs produce byte-identical per-prompt files. (c) A socket guard proves zero outbound connections. (d) Every report contains the full configuration fingerprint and the run aborts on a model-hash mismatch. (e) Suites exist for zones, long input, framing, lite simulation, script pilot, latency. (f) No file under `fie/`, `engine/`, `app/`, `storage/` changes. (g) Full baseline runs in ≤ 20 minutes on the development laptop |
| 002 | (a) A test per finding: tenant A cannot write to, read from, or change the decisions of tenant B through any route. (b) Unauthenticated `POST /track`, `/monitor`, `/diagnose`, `/analyze*` and `DELETE /clusters/reset` return 401. (c) The server exits at start-up with no signing secret. (d) The session token contains no API key. (e) `/flags` routes return 200 for an authorised caller and are tenant-scoped. (f) Harness baseline unchanged |
| 003 | (a) Golden file byte-identical for every existing field. (b) With the classifier's dependencies absent, `coverage` reports it and a warning is logged once. (c) For every prompt in the baseline, `zone` and `decision` are consistent with `is_attack`. (d) `risk_score` is non-zero for allowed prompts that had a signal. (e) Passing `threshold=` emits a deprecation warning. (f) Harness baseline unchanged |
| 004 | (a) In a clean virtual environment on Python 3.10–3.12, the documented install command followed by a scan of the JailbreakBench split gives 130/134, or the import fails with an actionable message. (b) Plain-weight PAIR head matches the pickled model: max absolute probability difference ≤ 1e-6 over the full baseline; verdicts identical. (c) No `joblib.load` or `pickle.load` remains under `fie/`. (d) A tampered model file is refused at load. (e) The version number is new and names exactly one artifact |
| 005 | (a) With default settings and no environment variables, a scan makes zero outbound connections (socket-guard test). (b) Determinism suite: 0 flips in 3 passes over all seven sets. (c) Both axes reported against the WP-001 baseline; any change is explained per prompt. (d) Benign Spanish, French and German pilot prompts are not blocked for want of a translator |
| 006 | (a) A CI check fails if a number in README or the fact sheet has no matching harness report. (b) Every "regenerate" command in the fact sheet runs from a clean clone. (c) Audit §8 entries C1–C28 are each resolved or explicitly kept with a reason |
| 007 | (a) `strict` preset reproduces the WP-001 baseline verdict for all 1,848 prompts. (b) Each preset ships a two-axis table generated by the harness. (c) Policy id and version appear on every decision. (d) An invalid policy is rejected at load. (e) E52 result recorded |
| 008 | (a) E30 dataset frozen and SHA-pinned before any fix is written. (b) Benign false-positive rate for each non-English language ≤ the English rate on the same template set + 5 points. (c) Recall on the E11 multilingual attacks and the character-injection set not lower than baseline. (d) Gates G1–G4 hold on the English sets |
| 009 | (a) Recall on the padded HarmBench set at 336 words of filler ≥ 90% of unpadded recall, filler before and after. (b) Over-refusal on padded XSTest-safe reported; the change in over-refusal on unpadded sets ≤ 2 points. (c) p95 latency reported for 256, 1k and 4k tokens |
| 010 | (a) `POST /v1/scan` returns the same `Decision` as the SDK for the baseline sets. (b) `FIEMiddleware` server mode blocks a known attack end to end. (c) With the scanner forced to raise, every wrapper follows the single failure-mode setting |
| 011 | (a) A test fails if any module under `fie/` imports `engine`, `app` or `storage`. (b) README states which directories are production and which are research |
| 012 | (a) Attack success rate with CIs at a fixed query budget, white-box and black-box. (b) Over-refusal reported after every defence round. (c) Results published whatever they are |
| 013 | (a) E32 measured. (b) The chosen resolver's recall and over-refusal on the REVIEW band reported with CIs. (c) If conformal selection is used, empirical risk on a held-out set is within the nominal bound |
| 014 | Research. Each experiment has a pre-registered hypothesis and a logged outcome. Anything shipped passes G1–G9 |
| 015 | A new user on a clean machine reaches a correct first scan in ≤ 5 minutes following the quickstart, verified by a scripted clean-environment test |
| 016 | Every decision produces one audit event matching the schema; content is absent when content logging is off; retention deletes on schedule in a test |
| 017 | No API key is recoverable from the database; rotation invalidates old keys and tokens |
| 018 | An output-side benchmark is chosen and measured on both axes; the stream guard scans beyond the first window; or the README claims are removed |
| 019 | Monitor code imports nothing from Groq; every hallucination result reports a length-only baseline next to it |

---

## 11. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | Fixing script handling removes the "robustness" that came from blocking any odd character, and real homoglyph attacks get through | High | High | E31 measures both; flag only intra-word script mixing; normalise before PAIR; keep the character-injection set as a gate |
| R2 | Windowed scoring restores recall on long input and brings over-refusal back with it | High | High | Report both in WP-009; pair it with the REVIEW state; pooling choice is a measured parameter |
| R3 | Adaptive attacks break PAIR outright | Medium–High | High | That is a result. Publish it with the defence rounds. Position FIE as one layer, not a wall |
| R4 | The OR-Bench-hard problem is a property of a frozen 23M encoder and no cheap fix exists | Medium | High | E40 tests it early. If true, say so and make REVIEW plus a cascade the product answer |
| R5 | Exposing REVIEW moves 13.4% of attacks to a path integrators treat as "allow" | Medium | High | Default binding is `block`. Presets carry their measured recall. Docs state the cost |
| R6 | Making the default install heavy (ML dependencies) loses users who wanted a light package | Medium | Medium | Keep an explicit lite extra that labels every result as reduced coverage |
| R7 | Changing defaults (telemetry, translation, failure mode) breaks existing integrations silently | Medium | Medium | Deprecation warnings for one release; changelog; contract tests |
| R8 | Removing online translation lowers multilingual attack recall | Medium | Medium | Measured in WP-005; opt-in remains; E40 evaluates a multilingual encoder as the real fix |
| R9 | The plain-weight PAIR export drifts from the pickled model | Low | High | Equivalence gate at 1e-6 over the whole baseline before the pickle is removed |
| R10 | Solo maintainer; the plan is long | High | Medium | Each package is one session and leaves the tree shippable. Phase 0 alone is a worthwhile release |
| R11 | Auth fixes break the dashboard or existing API clients | Medium | Medium | WP-002 lists every changed route as a breaking change before implementation; frontend checked in the same package |
| R12 | Research and product timelines pull against each other (paper P1 deadline) | Medium | Medium | WP-001's harness serves both. Research packages do not block Phase 0–1 |
| R13 | Baseline numbers shift when the environment is rebuilt to the pinned versions (xgboost 3.2.0 vs 2.1.4 today) | Medium | Medium | WP-001 records versions in the fingerprint; re-run once in a pinned environment and record any delta |
| R14 | This rebuild log is not version-controlled because `docs/*` is git-ignored | Certain until fixed | Medium | Decide in OQ-1 before WP-001 |
| R15 | 4 GB VRAM limits local guard models and adaptive attackers | High | Low–Medium | One model at a time; cached generations; small guards first |

---

## 12. Do not do — yet

| Tempting | Why not now |
| --- | --- |
| Retrain PAIR to fix over-refusal | E19, E20, E24–E29 show augmentation moves along a trade-off curve. Without the harness and the long-input and shortcut findings understood, a new model is a new unknown |
| Tune thresholds to improve the XSTest number | No operating point exists (E20). It would be optimising a benchmark while §5.8 shows the benchmark is unrepresentative of real prompt lengths |
| Add a thirteenth detection layer | Eleven of twelve add about zero on benchmarks (E10). More layers are more false-positive surface, as §5.2 shows |
| Swap in a bigger or multilingual encoder immediately | It is E40, a controlled ablation. Doing it first changes every number at once |
| Add an LLM judge to the request path | Latency, cost, a new injection surface, and "Strong but Brittle" shows reasoning guards are easily subverted |
| Rewrite `fie/adversarial.py` around the new architecture in one go | The golden test and the baseline exist to make small, verified steps possible. Use them |
| Delete the eleven heuristic layers | Some may matter on vectors no benchmark covers. Demote them to signals under the policy engine first, then prune with evidence |
| Port the monitor from Groq to Ollama as step one | It is the September plan's Phase 0, and it polishes the component that is at chance while the shipped guard misleads users |
| Build the streaming guard out | No output detector has been measured. Label it experimental |
| Enterprise features: SSO, RBAC, billing, SOC 2 wording | Tenant isolation is broken at the first route. Fix isolation, then build on it |
| Claim alignment with OWASP, NIST or any regulation | The control map in the audit shows gaps in most rows. Publish the map as a gap analysis only |
| Publish new headline numbers | Not until WP-006. The next README must be generated from reports |
| Rewrite git history to shrink the repository | Destructive, needs a force-push, unrelated to trust |
| Put the padding and prefix results in a LinkedIn post before the fix lands | They are working evasions of a deployed guard. Disclose with the fix, in the execution report and changelog |

---

## 13. Scorecard: where FIE is, and what "done" is measured against

Targets are proposals to be confirmed (OQ-5). "Measure" means no target is set until a baseline exists.

| Dimension | Metric | Today | Label | Proposed target |
| --- | --- | --- | --- | --- |
| Security | Macro attack recall, 4 sets, shipped model | 88.2% | REPRODUCED | No regression |
| Security | Recall in the default PyPI install | ~6% (simulated) | REPRODUCED sim. | Equal to the full pipeline, or the install fails loudly |
| Security | Recall with 336 words of benign padding | 6.7% | REPRODUCED | ≥ 90% of unpadded |
| Security | Caught attacks evaded by one fixed prefix | up to 34.0% | REPRODUCED | Measure in E50; target after |
| Security | Adaptive attack success rate | — | UNMEASURED | Measure |
| Security | Indirect injection, multi-turn | — | UNMEASURED | Measure or declare out of scope |
| Utility | XSTest-safe flagged / hard-blocked | 52.8% / 27.6% | REPRODUCED | Hard-blocked ≤ 27.6% at once via REVIEW; lower is research |
| Utility | OR-Bench-hard flagged / hard-blocked | 90.4% / 74.0% | REPRODUCED | Research; no honest target yet |
| Utility | Benign non-Latin blocked | 30/30 | PILOT | Each language ≤ English + 5 points |
| Utility | Benign European languages blocked with no translator | 18/18 | PILOT | 0 attributable to translator absence |
| Reliability | Verdict flips between identical runs | 1 / 250 | REPRODUCED | 0 |
| Reliability | Missing classifier visible on the result | No | REPRODUCED | Yes |
| Reliability | Uncertain routing visible on the result | No | Code read | Yes |
| Reliability | Wrappers honouring the failure-mode setting | 1 of 4 | Code read | 4 of 4 |
| Performance | Latency, short prompt, mean / p95 | 34.3 / 50.0 ms | REPRODUCED | No regression > 20% |
| Performance | Latency vs input length; memory; throughput | — | UNMEASURED | Measure |
| Developer experience | Clean-machine install gives the measured pipeline | No | REPRODUCED | Yes, ≤ 5 minutes to first correct scan |
| Developer experience | Public parameters with no effect | ≥ 3 | REPRODUCED | 0 |
| Enterprise readiness | Cross-tenant state channels | ≥ 6 (S1, S4, S5, S7, S8, S3) | Code read | 0, each with a test |
| Enterprise readiness | Unauthenticated compute or write routes | ≥ 6 | Code read | 0 |
| Enterprise readiness | Policy and model identity on a decision | None | Code read | Always |
| Enterprise readiness | Pickle load sites in the production package | 3 | Code read | 0 |
| Research quality | Fact-sheet measurement scripts present in the repository | 0 of 4 | REPRODUCED | All, one command per table |
| Research quality | Results with a recorded run configuration | 0 | REPRODUCED | All |
| Research quality | Leakage audit, frozen splits, CIs, negative results logged | Yes | MEASURED | Keep |
