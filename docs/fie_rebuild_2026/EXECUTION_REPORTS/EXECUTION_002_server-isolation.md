# EXECUTION_002 — Server Isolation and Tenant Security Hotfix (WP-002)

| | |
| --- | --- |
| Work package | WP-002 |
| Status | **COMPLETE WITH DOCUMENTED DEVIATIONS — awaiting owner review** |
| Commit status | **NOT COMMITTED. The owner reviews and commits manually.** Nothing was staged, committed, pushed, stashed or reset |
| Date | 2026-10-08 |
| Branch | `rebuild/wp-001-eval-harness` |
| HEAD (unchanged throughout) | `1e074d69e81d27932b47c01d57e78aff91c40c44` — the owner's WP-001 commit |
| Contract | [PLAN_002_SERVER_ISOLATION.md](../IMPLEMENTATION_PLANS/PLAN_002_SERVER_ISOLATION.md), recommendations of §31 approved |
| Deployed? | **No.** Nothing was deployed, no live service was called, no production database was touched |

Evidence labels used below. **REPRODUCES**: the attack test failed on the unmodified code for
the stated reason. **DOES NOT REPRODUCE**: the test passed on the unmodified code. **NOT
TESTABLE IN LOCAL HARNESS**: needs something the local fakes cannot stand in for.

---

## A. Scope

**Implemented:** the approved plan, steps 0 to 10. One authentication path that produces a
`Principal`; one declared policy on every route; a `TenantScope` that can only be built from a
principal; scoped data access for inferences, feedback and signal logs; tenant-scoped answer
cache, shadow-response cache, session context, conversation turns, clusters and trend; no
recalibration, retraining or attack-index growth from a request; database-checked admin
rights with explicit, recorded cross-tenant reads; security events; the exposure fixes
(`/health/deep`, error bodies, request id, CORS header, playground endpoints, rate-limit key).

**Extra findings fixed, as approved (W2-22):** N1–N13, N16, the CORS header from N18, N19.

**Not implemented, as instructed:** anything in `fie/`; detector, PAIR, multilingual,
long-input, hallucination, streaming, packaging, README or dashboard work; token revocation;
key hashing; a database migration; a deployment. Deferred findings N14, N15, N17, N20 and
the rest of N18 are untouched.

**Precondition check.** WP-001 was found committed (`1e074d6`) with a clean working tree, so
work began. Pre-state: Python 3.10.19 (conda `failure-engine`), fastapi 0.135.1,
starlette 0.52.1, slowapi 0.1.9, PyJWT 2.12.1, pymongo 4.16.0. Baseline tests before any
change: 317 passed (87 existing + 230 harness). `python -m evals baseline` before any change:
exit 0, records byte-identical to `BL-0001`.

**One stop condition was raised before any file was changed** (section J, DV-1): a tenth
existing test needed editing. The owner approved it.

## B. Files changed

**CREATE — production (3 files, 321 lines)**

| File | Lines | Role |
| --- | --- | --- |
| `app/security_events.py` | 72 | `emit()`, `tenant_ref()` |
| `app/tenancy.py` | 136 | `TenantScope`, `tenant_of`, `scope_key`, `TenantAnalytics`, `TenantRegistry` |
| `storage/tenant_store.py` | 113 | `TenantStore` and the `*_all_tenants` admin reads |

**CREATE — tests (17 files, 3,235 lines, 318 tests)**

`tests/security/`: `__init__.py`, `conftest.py`, `fakes.py`, `test_authn.py` (35),
`test_route_matrix.py` (106), `test_tenant_spoofing.py` (15), `test_inference_isolation.py`
(21), `test_answer_cache_isolation.py` (19), `test_feedback_isolation.py` (10),
`test_session_isolation.py` (10), `test_shared_state.py` (11), `test_admin_boundary.py` (7),
`test_exposure.py` (48), `test_concurrency.py` (3), `test_failure_modes.py` (19),
`test_static_boundary.py` (10), `test_performance_budget.py` (4).

**CREATE — documentation:** this file.

**MODIFY — production (25 files; +1,525 / −829 lines)**

| File | + / − | Change |
| --- | --- | --- |
| `app/auth.py` | 156 / 53 | Signing secret read at use and required; token without API key; key rotation fixed; empty e-mail refused; empty `ADMIN_EMAIL` grants nothing; constant-time env key; strict lookups; keys removed from the user list; `users.api_key` index requested |
| `app/auth_guard.py` | 227 / 46 | `Principal`, `authenticate`, the three policies, database-checked admin, cross-tenant read gate. Old helpers removed |
| `app/auth_routes.py` | 61 / 61 | Policies; verified e-mail required; generic errors |
| `app/limiter.py` | 25 / 2 | `rate_key`: tenant for authenticated requests |
| `app/main.py` | 88 / 29 | `enforce_startup_security`; CORS header removed; request-id validation; `/health/deep` and `/ready` exposure; `public` policy on root routes |
| `app/routes/_helpers.py` | 0 / 15 | Raw collection accessor removed |
| `app/routes/admin.py` | 25 / 23 | Admin policy; audit on change; digest through the store; generic errors |
| `app/routes/analytics.py` | 52 / 62 | Per-tenant trend and clusters; admin policy; generic errors; no raw database handle |
| `app/routes/community.py` | 5 / 7 | Policies declared |
| `app/routes/flags.py` | 47 / 43 | Dead helper removed; platform admin on all five routes; audit on label |
| `app/routes/inference.py` | 129 / 86 | Policies; `/track` tenant rule; scoped store; explicit `all_tenants`; deletes always scoped |
| `app/routes/monitor.py` | 123 / 154 | Policy; scope passed to every helper; no anonymous tenant; no index growth; feedback as plan §20 |
| `app/routes/playground.py` | 89 / 19 | Policy; endpoint restriction; own clusters and cache scope |
| `engine/agents/adversarial/specialist.py` | 5 / 1 | Learned prompt text never returned |
| `engine/agents/failure_agent.py` | 29 / 10 | Registry and tracker are arguments; no shared default |
| `engine/archetypes/registry.py` | 8 / 0 | Growth off unless the platform switch is set |
| `engine/fie_config.py` | 29 / 4 | Automatic recalibration off; `$set` instead of replace; event on change |
| `engine/groq_service.py` | 52 / 13 | Cache key includes tenant and system message; no scope → no cache |
| `engine/ground_truth_cache.py` | 91 / 37 | Tenant in key and in every query; schema 2; source class |
| `engine/multi_turn_tracker.py` | 15 / 3 | (tenant, conversation) |
| `engine/retraining/buffer.py` | 10 / 1 | Tenant on rows; trigger off |
| `engine/session_store.py` | 53 / 27 | (tenant, session) |
| `engine/verifier/ground_truth_pipeline.py` | 21 / 8 | Scope forwarded; trace without identity |
| `storage/database.py` | 125 / 112 | Tenant-namespaced document id; scoped upsert; `*_all_tenants` names; index requests |
| `storage/signal_logger.py` | 60 / 13 | Tenant on every row and in every lookup |

**MODIFY — existing tests (2 files; +44 / −11)**

| File | Tests touched | Change |
| --- | --- | --- |
| `tests/test_integration.py` | 9 (`TestMonitor` ×7, `TestDiagnose` ×2) | Fixture supplies a test principal through a dependency override. One test renamed: `test_monitor_requires_no_auth_returns_something` → `test_monitor_without_a_credential_is_401`, and now asserts 401 |
| `tests/test_monitor_rag_fix.py` | 1 (first test) | Passes a principal to the handler; its three fakes accept the new keyword arguments; the usage counter is stubbed. Four assertions unchanged. **Owner-approved addition, DV-1** |

**MODIFY — documentation:** `MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md`.

**DELETE:** none.

**Not touched:** `fie/`, `models/`, `scripts/`, `data/`, `evals/`, `tests/evals/`, `Frontend/`,
`deploy/`, `.github/`, `pyproject.toml`, `requirements.txt`, `README.md`, `SECURITY.md`,
`docs/FACT_SHEET.md`, `app/schemas.py`, `engine/model_extraction_tracker.py`.

## C. Step-by-step implementation

| Step | What was done | Result |
| --- | --- | --- |
| 0 | Wrote `tests/security/` (fake database, principals, network guard, marker scan, 14 test modules) before touching production code. Ran it against the unmodified code | **115 passed, 202 failed, 8 teardown errors** (317 tests at that point). Each failure examined; see section D |
| 1 | `Principal`, `authenticate`, policies, security events, token content, secret enforcement, key rotation, login checks | Unit tests pass |
| 2 | A policy dependency on every route; `/track` tenant rule; `/flags` repaired as admin-only; admin deletes scoped | Route-matrix tests pass: 51 routes, each with exactly one declared policy |
| 3 | `TenantStore`; tenant-namespaced inference ids; tenant on signal logs; routes no longer touch collections | Isolation tests pass |
| 4 | Answer cache and shadow-response cache scoped; trace without identity | Cache tests pass |
| 5 | Feedback route rewritten per plan §20; recalibration and retraining off from requests | Feedback tests pass |
| 6 | Session context and conversation turns keyed by (tenant, id) | Session tests pass |
| 7 | Per-tenant clusters and trend; index growth removed from the route and gated in the registry; learned text removed from evidence | Shared-state tests pass |
| 8 | `/health/deep`, `/ready`, error bodies, `/auth/users`, request id, CORS header, playground endpoint check, rate-limit key | Exposure tests pass |
| 9 | Whole suite; harness; baseline | Sections G and H |
| 10 | Final tests re-run against an extracted copy of the committed pre-fix code (`git archive HEAD`, no repository state changed); scope diff; this report | Section D, section I |

First run of the suite on the fixed code: 311 of 317 passed. The six failures were five
test defects and one behaviour of `fie/` that the test had assumed wrongly (NF-1); none was a
defect in the fix. They are listed in section J, DV-11.

## D. Attack reproduction matrix

"Pre" is the final test file run against the unmodified code (HEAD `1e074d6`). "Post" is the
same test on the fixed code. All 318 tests: pre 115 passed / 203 failed; post 318 passed.
No test passed before and fails now. (One test, for the blocked-prompt path, was added after
the first pre-fix run; it was then run against the pre-fix copy and fails there too: an
uncredentialled `/monitor` call answered 200.)

| # | Attack | Classification | Pre-fix evidence | Fix location | Post | Regression test |
| --- | --- | --- | --- | --- | --- | --- |
| 1a | S1 unauthenticated `/track` | **REPRODUCES** | 200, record stored | `inference.py` `track_inference` | 401 | `test_track_requires_credential` |
| 1b | S1 tenant A names B in the body | **REPRODUCES** | 200, stored under B | `_enforce_body_tenant` | 403, nothing stored, event | `test_track_body_tenant_mismatch_is_403` |
| 1c | N1 overwrite by `request_id` | **REPRODUCES** | "A's record is gone": 404 for A afterwards | `database.save_inference`, `TenantStore` | A's record intact | `test_track_cannot_overwrite_other_tenant_record` |
| 2a | S4 answer served across tenants | **REPRODUCES** | B's response contained A's correction; exact and reworded question | `ground_truth_cache.py` | miss | `test_cache_a_writes_b_misses`, `…_on_a_similar_question` |
| 2b | N2 submitter's e-mail in the trace | **REPRODUCES** | "verified by alpha.owner@tenant-a.test" in B's response | `ground_truth_pipeline.py`, cache stores a class | no identity | `test_cache_trace_has_no_identity` |
| 2c | N19 write-through shared | **REPRODUCES for the mechanism.** Hypothesis **not confirmed** for "a crafted prompt makes the external answer wrong": that part was not attempted | A system-verified entry had no tenant and any lookup returned it | `_cache_if_confident(scope=…)` | scoped | `test_cache_writethrough_is_tenant_scoped` |
| 3a | S5 feedback moves global thresholds | **REPRODUCES** | FACTUAL 0.40 → 0.25, UNKNOWN 0.45 → 0.25 after one feedback with 60 labels present | `monitor.py` feedback, `fie_config.maybe_recalibrate` | unchanged | `test_feedback_does_not_change_thresholds` |
| 3a | Retraining started by feedback | **REPRODUCES** | Retrain job started | `monitor.py`, `buffer.py` | not started | `test_feedback_does_not_start_retraining` |
| 3b | S6 overrides dropped | **REPRODUCES** | `attack_thresholds` missing from the stored document | `fie_config.recalibrate` | kept | `test_recalibration_preserves_attack_thresholds` |
| 3c | N8 admin feedback on another tenant | **REPRODUCES** | 200 | `monitor.py` feedback | 404 | `test_admin_cannot_write_feedback_cross_tenant` |
| 3d | N13 label on another tenant's log | **REPRODUCES** | "B's signal log was labelled by A" | `signal_logger.py` | untouched | `test_signal_log_lookup_is_tenant_scoped` |
| 4a | S7 prompt promoted to shared index | **REPRODUCES** | Growth called with A's prompt | `monitor.py`, `registry.py` | never called | `test_monitor_does_not_grow_attack_index` |
| 4b | N4 learned prompt returned | **REPRODUCES** | Marker present in the specialist's evidence | `specialist.py` | absent for learned entries | `test_faiss_evidence_never_returns_learned_prompt` |
| 4c | S8 label changes global hashes | **Latent, as the plan said.** The route answered 401 to everyone, admin included | 401 for tenant and admin | `flags.py` | tenant 403; admin 200; event | `test_flag_label_requires_platform_admin` |
| 5a | N10 session write across tenants | **REPRODUCES** | One document shared by both tenants | `session_store.py` | two documents | `test_same_session_id_two_tenants_is_two_sessions` |
| 5a | N10 session existence visible | **DOES NOT REPRODUCE.** The archetype label did not change. The plan's statement was not confirmed | Test passed on the unmodified code | — (covered by scoping anyway) | passes | `test_session_existence_is_not_observable` |
| 5b | N10 escalation across tenants | **REPRODUCES** | B's response reported `REPEATED_REFUSED` with 3 prior adversarial turns that were A's | `multi_turn_tracker.py` | `None` | `test_conversation_escalation_is_tenant_scoped` |
| 6a | S3 anyone resets clusters | **REPRODUCES** | 200 without a credential; B's reset emptied A's | `analytics.py`, `tenancy.py` | 401; own only | `test_cluster_reset_is_tenant_scoped` |
| 6b | N3 answers in `/clusters` | **REPRODUCES** | B received A's answer marker in a centroid | same | empty for B | `test_clusters_show_only_own_tenant` |
| 6c | S3 one trend | **REPRODUCES** | B saw 3 signals that were A's | same | 0 | `test_trend_is_tenant_scoped` |
| 6c | N16 alert blends tenants | **REPRODUCES, weaker than stated.** B's alert carried a 6% blend of A's signal (0.06 instead of 0.0). The "count" in the e-mail was always 0 because the code read a key that does not exist | `ema_entropy == 0.06` | `monitor.py` | 0.0; count now read from the right key | `test_spike_alert_reflects_only_the_callers_traffic` |
| 6d | N7 admin clear deletes all | **REPRODUCES** | 3 records of two tenants deleted | `inference.py` | 0 deleted | `test_admin_clear_deletes_only_own_tenant` |
| 6e | N7 admin sees all by default | **REPRODUCES** | Other tenants' records listed | `inference.py`, `authorize_cross_tenant_read` | own; explicit flag; 3 audit events | `test_admin_default_view_is_own_tenant`, `…_explicit_and_audited` |
| 7a | N9 shadow cache across tenants | **REPRODUCES** | B received "shadow answer number 1", A's | `groq_service.py` | separate calls | `test_groq_cache_is_scoped` |
| 7b | Cache key without tenant | **REPRODUCES** (by signature: the key function took no tenant) | — | `_question_id` | tenant in key | `test_cache_key_contains_tenant` |
| 8a | Existence oracle on inference id | **DOES NOT REPRODUCE — already correct**, as the plan stated | passes | — | passes | `test_no_existence_oracle_on_inference_id` |
| 8b | Cache hit reveals others' activity | **REPRODUCES** | B's `from_cache` changed after A's correction | cache scoping | unchanged | `test_cache_hit_flag_only_for_own_entries` |
| 8c | S12 shared rate-limit bucket | **REPRODUCES in the harness** when two tenants share a source address (HTTP 429 "60 per 1 minute" for B). Whether the Space's proxy presents one address is **NOT TESTABLE IN LOCAL HARNESS** | 429 in the concurrency test | `limiter.py` | independent limits | `test_two_tenants_behind_one_address_have_independent_limits` |
| 9 | S10 forged token without a secret | **REPRODUCES** | 200 for a token signed with the source constant | `auth.py`, `main.py` | 401; startup fails | `test_no_secret_no_tokens`, `test_startup_fails_without_secret` |
| 10 | S9 stale admin claim | **REPRODUCES** | 200 on an admin route for a non-admin | `auth_guard.py` | 403 | `test_admin_claim_in_a_token_is_not_believed`, `test_admin_flag_is_checked_in_database` |
| — | S9 API key in the token | **REPRODUCES** | `api_key` in the payload | `auth.py` | absent | `test_session_token_contains_no_api_key` |
| 11 | N5 key rotation fails | **REPRODUCES** | HTTP 500 | `auth.py` | 200; old key refused | `test_key_rotation_replaces_key` |
| 12 | N6 login without e-mail | **REPRODUCES** | Account created for an empty e-mail; **`is_admin: True`** when `ADMIN_EMAIL` is empty; a second such login got the same account | `auth.py`, `auth_routes.py` | refused | `test_login_requires_verified_email`, `test_empty_admin_email_grants_nothing` |
| 13 | N11 playground reaches inward | **REPRODUCES at the level tested:** the server issued the request to each of 19 internal or non-https targets (the HTTP call itself was a fake). Real reachability is **NOT TESTABLE IN LOCAL HARNESS** | Request recorded | `playground.py` | none issued | `test_playground_blocks_internal_endpoints` and three more |
| 14 | N12 `/health/deep` spends quota | **REPRODUCES** | Provider called for an anonymous request | `main.py` | not called | `test_health_deep_anonymous_makes_no_outbound_call` |
| — | S2 open routes | **REPRODUCES** | Eleven route/method pairs answered without a credential: exactly the eleven of W2-1 | policies | 401 | `test_no_credential_is_401` (39 cases) |
| — | S11 keys returned to admins | **REPRODUCES** for the response; keys at rest stay plain text (WP-017) | Keys in `/auth/users` | `auth.py` | absent | `test_user_list_never_contains_api_keys` |
| — | N18 CORS tenant header | **REPRODUCES** | `x-tenant-id` offered | `main.py` | not offered | `test_cors_does_not_offer_a_tenant_header` |

Three more routes answered 422 instead of 401 without a credential before the fix
(`/feedback/{id}`, `/flags/{id}/label`, `/playground`): they validated the body before checking
the credential. No data was exposed. They now answer 401 first.

## E. Security invariants

| # | Invariant | Status | Evidence |
| --- | --- | --- | --- |
| I-1 | No unauthenticated access to tenant data | **Holds** | 39 protected routes × no credential → 401; × unknown key → 401 |
| I-2 | The credential determines the tenant | **Holds** | `test_tenant_spoofing.py` (15) |
| I-3 | A caller-supplied tenant cannot raise access | **Holds** | Four mismatch values → 403, store empty |
| I-4 | A cannot read B's cached answer | **Holds** | `test_answer_cache_isolation.py` (19) |
| I-5 | A cannot write B's state | **Holds** | `test_inference_isolation.py` (21), `test_shared_state.py` (11) |
| I-6 | A's feedback cannot change B's thresholds or state | **Holds** | `test_feedback_isolation.py` (10) |
| I-7 | A's sessions are closed to B | **Holds** | `test_session_isolation.py` (10) |
| I-8 | No existence oracle | **Holds** | Identical status and body for foreign and missing ids on three routes; cache flags unchanged |
| I-9 | Shared state is immutable or platform-controlled | **Holds** | Static tests; growth, recalibration and retraining unreachable from routes |
| I-10 | Decisions attributable to a tenant | **Holds** | `test_rows_carry_tenant`: six collections |
| I-11 | No tokens and no start without a strong secret | **Holds** | Five tests |
| I-12 | Admin right read from the store at the time | **Holds** | Flag cleared → 403; store down → 503; store raising → 503 |
| I-13 | No response carries another tenant's text or identity | **Holds** | Marker scan on every response in the suite; zero leaks |
| I-14 | Missing tenant fails; never global | **Holds** | `test_failure_modes.py` (19) |
| I-15 | Admin cross-tenant read explicit and recorded; no cross-tenant write or delete | **Holds** | Six tests |
| I-16 | A credential can be revoked | **Holds** | Rotation test |
| I-17 | No server requests to tenant-chosen internal addresses | **Holds within the stated limit** | 22 tests. Limit: DNS answer could change between check and use |

## F. Acceptance criteria

| # | Criterion | Result |
| --- | --- | --- |
| AC-1 | Unauthenticated `/track` cannot modify state | **Pass** |
| AC-2 | A cannot modify B's state, body tenant notwithstanding | **Pass** |
| AC-3 | A's cache entry never returned to B | **Pass** |
| AC-4 | A's feedback cannot modify B's thresholds or configuration | **Pass** |
| AC-5 | A's sessions closed to B | **Pass** |
| AC-6 | All tenant-sensitive storage access has an explicit scope | **Pass** (static tests + query-filter test) |
| AC-7 | Tenant-sensitive cache keys carry the isolation context | **Pass** |
| AC-8 | No cross-tenant existence disclosure | **Pass** |
| AC-9 | No unclassified global mutable tenant state | **Pass** (10 module-level containers, each classified; a new one fails the test) |
| AC-10 | Existing valid tenant traffic still works | **Pass** (authenticated happy paths across the suite; dashboard call list covered by route tests) |
| AC-11 | Existing tests green except those asserting insecure behaviour | **Pass with DV-1**: 87 pass; 86 outcomes identical by name, 1 renamed; ten tests edited, not nine |
| AC-12 | WP-001 harness passes | **Pass**: 230 |
| AC-13 | Baseline metrics unchanged | **Pass**: `python -m evals baseline` exit 0; all eight counts equal; 11 record files byte-identical to `BL-0001`; four fingerprint keys equal; `fie/` unchanged. The run is flagged non-canonical only because the working tree is uncommitted (section H) |
| AC-14 | The suite reproduces every fixed vulnerability | **Pass with the two exceptions stated in D** (5a read oracle does not reproduce; 8a was already correct) |
| AC-15 | Every route has a declared policy | **Pass**, including detection of a route without one |
| AC-16 | Token and secret rules | **Pass** |
| AC-17 | Admin rights from the store | **Pass** |
| AC-18 | Key rotation | **Pass** |
| AC-19 | No foreign marker in any response | **Pass** |
| AC-20 | Events present on denials and free of secrets | **Pass**; every test scans its events for credentials, e-mails, raw tenant ids and planted text |
| AC-21 | Performance budget | **Pass**: authentication + scope ≤ 1 ms p95; key construction ≤ 0.1 ms p95; 0 extra user-store lookups on tenant routes, 1 on admin routes with a token |
| AC-22 | Only Artifact E files changed | **Pass with DV-1** (one extra test file, approved) |

## G. Test counts

| Suite | Command | Passed | Failed | Skipped | Expected failures |
| --- | --- | --- | --- | --- | --- |
| Security | `pytest tests/security` | **318** | 0 | 0 | 0 |
| Existing | the six `tests/test_*.py` files | **87** | 0 | 0 | 0 |
| WP-001 harness | `pytest tests/evals` | **230** | 0 | 0 | 0 |
| **Full** | `pytest tests/ -m "not network"` | **635** | 0 | 0 | 0 |

Before the change the full suite was 317 (87 + 230). The security suite on the unmodified
code: 115 passed, 203 failed.

The figures above are from the last full run, made after every change in this report.

## H. Baseline verification

Command: `python -m evals baseline`, run on the final working tree, after every change in
this report. Run directory `evals/runs/20261008T080344Z_5d5ed90d_d233a245` (git-ignored).

| Check | Result |
| --- | --- |
| Exit code | **0** |
| Output | "standard counts match the approved canonical counts" and "per-prompt records are byte-identical to BL-0001_pair-v6.3b_fie-5d5ed90d (11 file(s))" |
| JailbreakBench | 130 / 134 |
| HarmBench | 326 / 387 |
| StrongREJECT | 218 / 242 |
| SORRY-Bench | 316 / 387 |
| XSTest-safe (benign axis) | 132 / 250 |
| OR-Bench-hard (benign axis) | 226 / 250 |
| XSTest-unsafe | 177 / 198 |
| AdvBench (case study) | 163 / 168 |
| Fingerprint keys | `dataset_key`, `config_key`, `subject_key`, `env_key`: all four equal to the pinned baseline's |
| `fie/` source tree | `5d5ed90d`, unchanged. `git status -- fie` is empty |
| Hermetic guard | Workers exited 0 |

**One flag differs, and it is expected.** The run reports `canonical=False` with the reason
"uncommitted changes in measured paths: app/…, engine/…, storage/…". That is the harness
doing its job: WP-002 is uncommitted by instruction, and WP-001 (decision D-013) marks a run
non-canonical when the measured paths have uncommitted changes. For the same reason the
run's artifact digest (`0513d8cbc3cf6e63…`) is not the pinned one: `fingerprint.json` records the
git state. The measured results are identical: same counts, same 11 record files byte for
byte. After the owner commits, `python -m evals baseline` should report `canonical=True`.

The baseline was not regenerated and nothing under `evals/` was modified. The same command
was also run before any change (exit 0, records byte-identical) and once mid-way (same
result).

Why the result could not have changed: no file under `fie/` was modified, and the canonical
profile makes `engine`, `app` and `storage` unimportable inside the measured process.

## I. Scope verification

`git status --porcelain` at hand-over:

```text
 M app/auth.py
 M app/auth_guard.py
 M app/auth_routes.py
 M app/limiter.py
 M app/main.py
 M app/routes/_helpers.py
 M app/routes/admin.py
 M app/routes/analytics.py
 M app/routes/community.py
 M app/routes/flags.py
 M app/routes/inference.py
 M app/routes/monitor.py
 M app/routes/playground.py
 M docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md
 M engine/agents/adversarial/specialist.py
 M engine/agents/failure_agent.py
 M engine/archetypes/registry.py
 M engine/fie_config.py
 M engine/groq_service.py
 M engine/ground_truth_cache.py
 M engine/multi_turn_tracker.py
 M engine/retraining/buffer.py
 M engine/session_store.py
 M engine/verifier/ground_truth_pipeline.py
 M storage/database.py
 M storage/signal_logger.py
 M tests/test_integration.py
 M tests/test_monitor_rag_fix.py
?? app/security_events.py
?? app/tenancy.py
?? docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_002_server-isolation.md
?? storage/tenant_store.py
?? tests/security/
```

`git diff --stat` for code and tests: 27 files changed, 1,569 insertions, 840 deletions.
Untracked: 3 production files (321 lines), 17 test files (3,235 lines), this report.

`git status --porcelain -- fie models scripts data evals tests/evals Frontend deploy .github
pyproject.toml requirements.txt README.md SECURITY.md docs/FACT_SHEET.md` → empty.

HEAD before and after: `1e074d6`. No commit, no stage, no push, no stash, no reset.

## J. Deviations

Every departure from the approved plan. None weakens an invariant or an acceptance criterion.

| # | Plan | What was done | Why |
| --- | --- | --- | --- |
| DV-1 | Nine existing tests change, in one file | **Ten, in two files.** The first test of `tests/test_monitor_rag_fix.py` calls the `/monitor` handler directly with no credential | The plan missed it. No secure handler can process an uncredentialled call. Raised as stop condition 1 before any change; **approved by the owner** |
| DV-2 | `FIE_AUTO_RECALIBRATE`, `FIE_AUTO_RETRAIN`, `FIE_FAISS_AUTOGROW` "restore today's behaviour" | **They do not.** The routes no longer call these functions at all. The switches gate only callers outside any route | The implementation brief forbids request-path recalibration and growth and forbids a switch that weakens the boundary. This is the stricter reading |
| DV-3 | Event `authz.cross_tenant_denied` | Not emitted | It has no trigger: a foreign id is a plain 404 by design, and a body mismatch has its own event |
| DV-4 | `cache.scope_missing` for any cache call without a scope | Emitted for the answer cache. The shadow-response cache is silently uncached without a scope | Helper calls without a scope are normal there (explanation writer, claim extraction) and would flood the log |
| DV-5 | `save_inference` "takes the tenant from its caller" | Signature unchanged: the tenant is the record's, and `TenantStore` forces it to the scope's | Same effect; keeps the existing test's stub and a script's patch list valid |
| DV-6 | `run_full` / `run_diagnostic` take a registry and a tracker | Both receive one per-tenant holder that wraps the two behind a lock | Eight threads of one tenant updating a trend would otherwise race |
| DV-7 | — | Helpers not named in the plan: `verify_platform_admin`, `platform_admin_or_none`, `ensure_principal`, `is_current_admin`, `authorize_cross_tenant_read` | Needed to keep one authentication path |
| DV-8 | — | The usage-limit message no longer quotes the plan's call limit | The number came from the token or a second lookup; neither is used now |
| DV-9 | Playground: public `https` hosts | Also refuses every IP literal, even a public one, and single-label names | Closes decimal, hex and IPv4-mapped spellings without a parser for each |
| DV-10 | `/health/deep`: unchanged shape | For anonymous callers the `groq` component reports the new status `configured` | It is not probed, so `ok` would be untrue. Top-level keys and the `detector` component are unchanged; the CI deploy check reads only those |
| DV-11 | — | Six tests failed on the first run against the fixed code. Five were test defects: a count that included the user collection; two counts that included the explanation writer's provider calls; an annotation the limiter could not resolve; a static rule that flagged the login route reading its own new account. The sixth is NF-1. Each was corrected in the test; no production behaviour was changed to satisfy a test | Recorded so the corrections are visible |
| DV-12 | Reproduction "on commit `c5b2291`" | Done on `1e074d6` | The plan predates the owner's WP-001 commit. Production code is identical in both |
| DV-13 | — | The suite's `conftest.py` blanks every `.env` key in the process environment at import | Required so no real credential can load; in a full-suite run the existing API tests therefore stop making their stray calls to the provider. Their outcomes are unchanged |
| DV-14 | Size: about 450 new and 350 changed production lines, 1,800 test lines | 321 new, about 1,530 added / 830 removed in modified files, 3,235 test lines | Routes were rewritten around dependencies rather than patched; the test matrix is larger than estimated |
| DV-15 | — | A few log lines that this package already edited no longer print e-mails, tenant ids or exception text | Plan DL-10 |

## K. New findings

| # | Finding | Status |
| --- | --- | --- |
| NF-1 | `fie.feedback_store.apply_label` reports success for an event id it did not find when MongoDB is the backend. `POST /flags/{id}/label` therefore answers 200 for an unknown id. No hash is learned and no row changes | Not fixed: `fie/` is out of scope. Verified harmless by test |
| NF-2 | **The branch is on the public GitHub repository.** `rebuild/wp-001-eval-harness` tracks `origin` at `1e074d6`, and the repository is public. The audit and `PLAN_002`, which describe these attacks on the live service, are readable by anyone | Owner action. Nothing was pushed or changed on the remote by the assistant. See section N |
| NF-3 | Reasoning sub-step verification (`engine/reasoning/step_verifier.py`) calls the answer pipeline without a scope. It now runs uncached, and each confident external verification logs `cache.scope_missing` | Safe (fails closed). Threading the scope through `engine/reasoning/` is outside Artifact E |
| NF-4 | The explanation writer and other helper calls to the provider are no longer cached, because they pass no scope | Safe. More provider calls. Plan risk R8 |
| NF-5 | `engine/pipeline/langgraph_pipeline.py`, which no route uses, calls the changed helpers without a tenant. It now stores nothing: no session turns, no signal logs, no inference records | Safe. It was already unreachable |
| NF-6 | Before the fix, `/ready` and `/health/deep` returned the warm-up exception text to anyone | Fixed (exposure hardening) |
| NF-7 | The spike-alert e-mail's inference count was always 0: the code read `signals_count`, the tracker provides `signals_recorded` | Fixed in passing |
| NF-8 | The Space adds `GET /api` in `deploy/huggingface/space_app.py`. It has no declared policy and is not seen by the route-matrix test, which inspects `app.main` | Not fixed: `deploy/` is out of scope. It returns a static banner |
| NF-9 | `scripts/show_api_key.py` mentions a removed function in a comment | Not fixed: `scripts/` is out of scope |

## L. Rollback

Nothing is committed, so before the owner's commit: `git checkout -- <the 27 modified files>`
and delete `app/security_events.py`, `app/tenancy.py`, `storage/tenant_store.py`,
`tests/security/`. After the owner's commit: revert that commit.

After a deployment, rolling back re-opens every hole in section D. Two data points to know:

- Answer-cache entries written by the new code carry a tenant and `schema: 2`. The old code's
  similarity search would serve them to every tenant. Remove them before rolling back if that
  matters.
- Inference documents written by the new code have `_id = "<tenant>:<request_id>"`. The old
  code finds them through the `request_id` field. No conversion is needed.

Session and conversation state written before a deployment is ignored by the new code and
expires on its own (24 h and 2 h).

## M. Deployment checklist (owner only)

Pushing to `main` deploys. Before pushing:

| # | Action | Why |
| --- | --- | --- |
| 1 | The Space has a `JWT_SECRET_KEY` secret of at least 32 characters | Otherwise the new build refuses to start. The local `.env` value passes this check |
| 2 | `FIE_ALLOW_INSECURE_DEV_SECRET` is **not** set on the Space | It is a local-development switch only |
| 3 | The Space's `ADMIN_EMAIL` is the intended account | Empty grants admin to nobody; wrong locks the admin out |
| 4 | Decide whether the server-side `FIE_API_KEY` should exist on the Space | It is a database-independent platform-admin credential |
| 5 | Decide whether to reset thresholds that past recalibrations stored (`fie_config` document) | The package stops further movement; it does not undo past movement |
| 6 | Decide whether to remove learned entries from the FAISS files on the Space | Their text is no longer returned; they still take part in matching |
| 7 | On the production database: check that no (`tenant_id`, `request_id`) pair is duplicated, then drop the unique index on `request_id` alone | Until then a tenant reusing another tenant's request id on `/track` gets 409. The new unique index on the pair is requested by the application at start |
| 8 | Rotate the admin API key if it was ever shared | Rotation works again |
| 9 | Tell known API users that anonymous access has ended | Breaking change BC-1 |
| 10 | Do **not** set `FIE_AUTO_RECALIBRATE`, `FIE_AUTO_RETRAIN` or `FIE_FAISS_AUTOGROW` | They concern shared state |

After deploying, read-only: `/ready` is true; `POST /api/v1/monitor` with no credential
returns 401; the dashboard signs in and lists the owner's own inferences; the log has no
`startup.insecure_secret` event.

## N. Security disclosure status

The fixes exist only in the working tree of this machine. They are not committed, not pushed
and not deployed. **The live service is still vulnerable to everything in section D.**

The disclosure boundary of plan §29 was not held: the branch carrying the audit and the plan
is on the public repository (NF-2). The assistant did not push it and has not altered the
remote. Options for the owner, in order of effect: delete the remote branch; deploy this fix
promptly; consider whether hosted users need to be told, since the service stores other
people's prompts and the faults allowed cross-tenant reads of some of that text. Deleting
the branch does not remove copies already fetched.

This report and the test suite contain working reproductions. They should travel with the
fix, not ahead of it.

## Deferred items

Recorded, not started: N14 (tenant id embeds part of the e-mail), N15 (`load_from_db` never
loads two thresholds), N17 (usage quota), the rest of N18 (`X-API-Key` not allowed by CORS),
N20 (public demo writes to the review queue), NF-1, NF-3, NF-8, NF-9; key hashing and token
revocation (WP-017); per-tenant provider metering; a pinned connection for the playground.

## Manual commit line

Not executed by the assistant. It stages only the files of this package. The line uses
`&&`, which works in Git Bash, cmd and PowerShell 7; in Windows PowerShell 5.1 replace `&&`
with `;`.

```text
git add -- app/auth.py app/auth_guard.py app/auth_routes.py app/limiter.py app/main.py app/security_events.py app/tenancy.py app/routes/_helpers.py app/routes/admin.py app/routes/analytics.py app/routes/community.py app/routes/flags.py app/routes/inference.py app/routes/monitor.py app/routes/playground.py engine/agents/adversarial/specialist.py engine/agents/failure_agent.py engine/archetypes/registry.py engine/fie_config.py engine/groq_service.py engine/ground_truth_cache.py engine/multi_turn_tracker.py engine/retraining/buffer.py engine/session_store.py engine/verifier/ground_truth_pipeline.py storage/database.py storage/signal_logger.py storage/tenant_store.py tests/security tests/test_integration.py tests/test_monitor_rag_fix.py docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_002_server-isolation.md docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md && git commit -m "security: isolate server tenant state and harden authentication (WP-002)"
```
