# PLAN_002 — Server Isolation and Tenant Security Hotfix (WP-002)

| | |
| --- | --- |
| Work package | WP-002 |
| Status | **PLAN ONLY. Nothing is implemented. Awaiting owner approval** |
| Date | 2026-10-08 |
| Branch | `rebuild/wp-001-eval-harness` (local, not pushed). No production file was changed while writing this plan |
| Measurement foundation | WP-001 harness, baseline `BL-0001_pair-v6.3b_fie-5d5ed90d` |
| Inputs read | Read in full this session: `SECURITY.md`; every file under `app/` and `storage/`; `config.py`; the state-holding modules under `engine/` and `fie/` listed in §4; the Space Dockerfile and entrypoints; `tests/test_integration.py`; the dashboard's `api.js`. Re-read in part: [BASELINE_AUDIT.md](../BASELINE_AUDIT.md) §3, §5.4, §5.5, §5.12 and [ROADMAP.md](../ROADMAP.md) WP-002 rows (both written in the earlier planning session, as were the master log and EXECUTION_001). Searched, not read line by line: `README.md` (no tenant or auth claim found), the root Dockerfile, the CI workflow, the rest of `engine/`, `fie/`, `Frontend/src`, `scripts/`, `examples/` for the concepts in the brief |
| Disclosure | **This file describes exploitable faults in a live service. It stays on the local branch until the fixes are deployed and verified (§29)** |

**How to read the evidence labels.** CODE-READ: verified by reading the code at the cited
line. CHECKED: confirmed by a local, read-only command during this session. HYPOTHESIS: follows
from the code but has not been executed; Step 0 reproduces it locally before anything is
fixed. Nothing here was tested against the live service.

---

# 1. Objective

Make it impossible for one tenant to influence, retrieve, modify or observe another tenant's
state through the FIE server.

In scope: authentication, tenant identity, route authorization, and the scope of every piece
of server state that holds tenant data or changes a tenant's results.

Out of scope, by instruction: detector layers, PAIR, multilingual and long-input behaviour,
the policy engine, the hallucination method, streaming, model formats, pickle removal,
packaging, README text, and cleanup unrelated to a tenant boundary. **`fie/` is not modified by
this plan.** That keeps the WP-001 baseline untouched by construction (§23, AC-13).

The package is finished when every invariant in §15 has an automated test that fails on
today's code and passes on the fixed code, and the harness baseline is unchanged.

# 2. Security context

The server is one FastAPI process ([app/main.py](../../../app/main.py)) with 51 routes: 43
under `/api/v1`, plus `/`, `/health`, `/ready`, `/health/deep` and four documentation routes
(CHECKED by enumerating `app.routes`). The Hugging Face Space serves the same application and
mounts a Gradio demo at `/` in the same process
([deploy/huggingface/space_app.py:263](../../../deploy/huggingface/space_app.py#L263)). CI
deploys the Space on every push to `main`.

What the code does today, in one paragraph. A caller is identified by an `X-API-Key` header or
a session token. Most data routes for stored inferences check that identity and filter by
tenant. Everything around them does not: thirteen `/api/v1` routes need no credential, two
more treat it as optional, one write route takes the tenant from the request body, and almost
all derived state — the answer cache, thresholds, session history, clusters, trend, the
attack-pattern index, response caches — is one shared copy for every caller.
`SECURITY.md` line 74 states "all MongoDB queries are scoped to `tenant_id`". That statement
is false today for six of the thirteen collections the server writes (two of the thirteen
hold no tenant data).

The audit recorded twelve findings (S1–S12). This plan confirms all twelve by code reading
and adds twenty more, N1–N20 (Artifact C in §3; the rest in §6, §14 and §30). The most
serious additions:

- `POST /track` can overwrite any tenant's stored record, not only add to it.
- Two response fields return another caller's text: the attack-pattern evidence returns 120
  characters of someone else's prompt, and `GET /clusters` returns other tenants' model
  answers. Both are reachable without a credential.
- The answer cache returns the e-mail address of the person who submitted a correction.
- `POST /auth/regenerate-key` raises on every call when MongoDB is connected, so a leaked key
  cannot be rotated (CHECKED: the truth-test it performs on a collection object raises).
- A platform admin's "clear my inferences" call deletes every tenant's records.

# 3. Current threat model

## 3.1 Assets

| # | Asset | Where it lives |
| --- | --- | --- |
| AS-1 | Tenant prompts and model outputs | `inferences`, `signal_logs`, `conversation_turns`, `session_context`, `model_extraction_tracking`, `flagged_events`, FAISS metadata file, hard-positive files |
| AS-2 | Tenant corrections and labels | `feedback`, `ground_truth_cache`, `signal_logs` label fields |
| AS-3 | Tenant credentials | `users.api_key` (plain text), session token |
| AS-4 | Tenant decisions: what FIE tells a tenant about a prompt or an answer | Thresholds, answer cache, clusters, trend, attack-pattern index, learned hashes |
| AS-5 | Platform configuration | `fie_config` document, environment |
| AS-6 | Operator resources | Groq, Serper and SendGrid quotas; MongoDB; CPU |
| AS-7 | Identity data | E-mail, name, picture in `users`; e-mail inside the tenant identifier |

## 3.2 Trust boundaries

1. Internet → FastAPI process (the only boundary that is partly enforced today).
2. Tenant ↔ tenant, inside the process (not enforced for derived state).
3. Tenant → platform admin (enforced by `is_admin`, read from a 24-hour token).
4. Server → third parties: Groq, Google OAuth and Translate, Wikidata, Serper, SendGrid.
5. Request path → background threads (recalibration, retraining, index save, e-mail).

## 3.3 Attackers

| ID | Attacker | Surface | Capability today | Expected authorization | Expected denial | Audit record needed | Residual risk after WP-002 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Unauthenticated, external | 13 open routes + 2 optional-auth routes | Write records into any tenant, overwrite records by id, reset clusters, read cluster contents and trend, run the full monitor pipeline on the operator's Groq quota, grow the global attack index | None beyond health, docs, the public demo feedback and anonymous telemetry | 401, no body detail | `authn.missing` with route and request id | Public demo, `/telemetry` and `/community/feedback` remain open by design |
| B | Authenticated tenant A | All authenticated routes | Poison the global answer cache; move global thresholds with 50 labels; write into any session or conversation id; read other tenants' text through cluster and FAISS evidence | Own tenant's data only | 404 for another tenant's object, 403 for a tenant mismatch | `authz.tenant_mismatch`, `authz.cross_tenant_denied` | Global platform-controlled state still shared (model, thresholds). Timing of the scan-verdict cache |
| C | Malicious person holding tenant A's key | Same as B | Same as B. FIE has no user level inside a tenant | Same as B | Same | Same | **No intra-tenant separation exists or is added.** One key = whole tenant |
| D | Stolen key or token of tenant A | Same as B | Same as B, and the key cannot be rotated because the route fails | Same as B until rotation | 401 after rotation | `auth.key_rotated` | A stolen session token stays valid up to 24 h. Keys stay in plain text in the database (WP-017) |
| E | Tenant A reading tenant B | Read routes, response evidence, caches | Clusters, FAISS nearest prompt, cache trace with e-mail, trend, cache-hit flag | Never | 404 / absent field | `authz.cross_tenant_denied` | None intended |
| F | Tenant A poisoning tenant B | Feedback, cache, sessions, conversations, clusters, attack index | Replace B's answers in correct mode; shift B's thresholds; mark B's conversation as escalating; destroy B's records | Never | Write lands only in A's scope | `authz.tenant_mismatch` | Platform admin actions still affect all tenants, by design |
| G | Tenant A inferring B's activity | Timing, error text, flags in responses, public config | `from_cache`, cache trace, Groq-cache latency, `/monitor/model-info` threshold changes, global trend | No observable difference | Identical response for "not yours" and "does not exist" | — | Scan-verdict and translation caches remain global: a timing difference of milliseconds shows that some caller recently scanned the same text |
| H | Platform admin | Admin routes, plus every data route in "all tenants" mode | Read and delete every tenant's data, including by accident; read every API key; switch the guard to warn-only for everyone | Cross-tenant read only when asked for explicitly; no cross-tenant write or delete | 403 for a non-admin; admin flag checked against the database | `admin.cross_tenant_read`, `admin.config_change` | The admin is trusted. No separation of duties, no second approver |
| I | Background task using global state | Recalibration thread, retrain thread, index save | Rewrites global thresholds from all tenants' labels; drops operator overrides | Runs only on a platform admin's explicit action | Does not start from a request path | `platform.recalibration` | A manual recalibration still mixes all tenants' labels |

## 3.4 ARTIFACT C — Cross-Tenant Attack Matrix

Each row has a planned local reproduction (§22). "S" numbers are the audit's; "N" numbers are
new in this plan.

| # | Attack | Current result | Root cause | Target result | Test |
| --- | --- | --- | --- | --- | --- |
| 1a | S1. Unauthenticated `POST /track` with `tenant_id = B` | Record stored in B's history. CODE-READ | No dependency on the route; body field trusted. [inference.py:38-43](../../../app/routes/inference.py#L38), [schemas.py:36](../../../app/schemas.py#L36) | 401 | `test_track_requires_credential` |
| 1b | Tenant A sends `tenant_id = B` | Stored under B | Same | 403, nothing stored, event logged | `test_track_body_tenant_mismatch_is_403` |
| 1c | **N1.** `POST /track` with a `request_id` that already exists under B | B's record is replaced and re-homed. HYPOTHESIS | `save_inference` upserts on `_id = request_id`, a global namespace. [database.py:150-171](../../../storage/database.py#L150) | A's write cannot touch B's document | `test_track_cannot_overwrite_other_tenant_record` |
| 2a | S4. A submits a "correct answer"; B asks a similar question | B receives A's text as a verified answer at confidence 1.0; in correct mode it replaces B's model output | One global cache, matched at cosine ≥ 0.92. [monitor.py:959-966](../../../app/routes/monitor.py#L959), [ground_truth_cache.py:79-135](../../../engine/ground_truth_cache.py#L79) | B never sees A's entry | `test_cache_a_writes_b_misses` |
| 2b | **N2.** Same, B reads the pipeline trace | Trace contains "verified by `<A's e-mail>`" | `verified_by` is the submitter's e-mail and is written into the returned trace. [ground_truth_pipeline.py:149-152](../../../engine/verifier/ground_truth_pipeline.py#L149) | No identity in any trace | `test_cache_trace_has_no_identity` |
| 2c | **N19.** A crafts a prompt so that the external or self-consistency answer is wrong, without using feedback | Wrong answer cached for everyone. HYPOTHESIS | Write-through caching of system results is also global. [ground_truth_pipeline.py:427-446](../../../engine/verifier/ground_truth_pipeline.py#L427) | Entry stays in A's scope | `test_cache_writethrough_is_tenant_scoped` |
| 3a | S5. A submits 50 labels | Global per-question-type thresholds are recomputed and applied to every tenant's `/monitor` | `maybe_recalibrate()` runs after each feedback and writes process-global and persisted thresholds. [fie_config.py:456-606](../../../engine/fie_config.py#L456), [monitor.py:636](../../../app/routes/monitor.py#L636) | No threshold changes from a request path | `test_feedback_does_not_change_thresholds` |
| 3b | S6. Any recalibration | Operator attack-threshold overrides are dropped from the stored document | `replace_one` without the field. [fie_config.py:556-566](../../../engine/fie_config.py#L556) | Overrides survive | `test_recalibration_preserves_attack_thresholds` |
| 3c | **N8.** A platform admin submits feedback on another tenant's record | Cache entry and label written from an admin identity onto another tenant's inference | Admin branch uses the unscoped lookup. [monitor.py:947-951](../../../app/routes/monitor.py#L947) | Feedback only on the caller's own records | `test_admin_cannot_write_feedback_cross_tenant` |
| 3d | **N13.** Feedback updates a signal log found by `request_id` alone | Label lands on whichever log has that id | `signal_logs` has no `tenant_id`. [signal_logger.py:175-184](../../../storage/signal_logger.py#L175) | Lookup by tenant and id | `test_signal_log_lookup_is_tenant_scoped` |
| 4a | S7. Any caller's prompt judged adversarial at ≥ 0.85 | Added to the global FAISS index and saved to disk | Auto-growth on the request path. [monitor.py:298-311](../../../app/routes/monitor.py#L298) | No growth from requests | `test_monitor_does_not_grow_attack_index` |
| 4b | **N4.** B sends a prompt similar to A's stored one | Response evidence contains 120 characters of A's prompt. Reachable through `/diagnose` with no credential | `nearest_prompt` returned in jury evidence. [specialist.py:314-316](../../../engine/agents/adversarial/specialist.py#L314) | Only seed-corpus text is ever returned | `test_faiss_evidence_never_returns_learned_prompt` |
| 4c | S8. A label on a flagged event | Would whitelist or block that exact prompt for everyone. Unreachable today | Process-global hash sets. [feedback_store.py:19-20](../../../fie/feedback_store.py#L19) | Platform admin only, audited | `test_flag_label_requires_platform_admin` |
| 5a | **N10.** A sends `/monitor` with B's `session_id` | A appends turns to B's history; A learns whether the session exists through the archetype label | Key is the caller-supplied id alone. [session_store.py:133-213](../../../engine/session_store.py#L133) | A and B have separate histories under the same id | `test_same_session_id_two_tenants_is_two_sessions` |
| 5b | **N10.** A sends adversarial turns with B's `conversation_id` | B's next response reports `multi_turn_escalation` | Same. [multi_turn_tracker.py:84-129](../../../engine/multi_turn_tracker.py#L84) | Scoped by tenant | `test_conversation_escalation_is_tenant_scoped` |
| 6a | S3. Anyone calls `DELETE /clusters/reset` | Global registry wiped | No dependency. [analytics.py:28-33](../../../app/routes/analytics.py#L28) | 401; an authenticated caller resets only its own | `test_cluster_reset_is_tenant_scoped` |
| 6b | **N3.** Anyone calls `GET /clusters` | Returns each cluster's centroid, whose `answer_counts` keys are normalised model answers from all callers. HYPOTHESIS | Global registry, centroid dumped whole. [clustering.py:134-144](../../../engine/archetypes/clustering.py#L134), [consistency.py:215-249](../../../engine/detector/consistency.py#L215) | Own tenant's clusters only | `test_clusters_show_only_own_tenant` |
| 6c | S3, **N16.** `GET /trend`; spike e-mail | One trend for all tenants; the alert e-mail sent to one tenant carries all tenants' rate and count | Global tracker. [monitor.py:216-234](../../../app/routes/monitor.py#L216) | Per-tenant trend | `test_trend_is_tenant_scoped` |
| 6d | **N7.** Admin calls `DELETE /inferences` | Every tenant's records deleted | Admin branch loops over all records. [inference.py:203-217](../../../app/routes/inference.py#L203) | Deletes the caller's tenant only | `test_admin_clear_deletes_only_own_tenant` |
| 6e | **N7.** Admin opens the dashboard | Lists every tenant's prompts by default | Admin branch returns all. [inference.py:82-86](../../../app/routes/inference.py#L82) | Own tenant unless `all_tenants=true`; that read is audited | `test_admin_cross_tenant_read_is_explicit_and_audited` |
| 7a | **N9.** B sends a prompt A sent within the hour | B gets shadow answers generated for A's request; the canary check cannot fire | Cache key is model + prompt; tenant and system message are not in it. [groq_service.py:26-45](../../../engine/groq_service.py#L26), [:161](../../../engine/groq_service.py#L161) | Key carries the scope and the system message; no scope → no cache | `test_groq_cache_is_scoped`, `test_groq_cache_disabled_without_scope` |
| 7b | Answer-cache key | SHA-256 of the lower-cased question, nothing else | [ground_truth_cache.py:60-63](../../../engine/ground_truth_cache.py#L60) | Tenant in the key and in every query | `test_cache_key_contains_tenant` |
| 8a | Read of another tenant's inference id | 404, same as a missing id. **Already correct** | Scoped lookup. [database.py:253-270](../../../storage/database.py#L253) | Keep | `test_no_existence_oracle_on_inference_id` |
| 8b | Answer-cache hit | `from_cache: true`, use count and trace show that someone asked before | Global cache | Only the tenant's own history is visible | `test_cache_hit_flag_only_for_own_entries` |
| 8c | S12. A sends 60 requests a minute | Possibly every user behind the proxy is limited. HYPOTHESIS, not tested | Limiter keys on the socket address. [limiter.py:10](../../../app/limiter.py#L10) | Authenticated traffic is limited per tenant | `test_rate_limit_key_is_tenant_for_authenticated` |
| 9 | S10. Deploy without `JWT_SECRET_KEY` | Tokens signed with a constant in the source; anyone can forge an admin token | Fallback string. [auth.py:24](../../../app/auth.py#L24) | No token is accepted or issued; the server refuses to start | `test_no_secret_no_tokens`, `test_startup_fails_without_secret` |
| 10 | S9. Admin flag removed in the database | Old token still grants admin for up to 24 h; `/auth/users` trusts the token | Flag read from the token. [auth_guard.py:19-25](../../../app/auth_guard.py#L19), [auth_routes.py:190-193](../../../app/auth_routes.py#L190) | Admin checked against the database on each admin call | `test_admin_flag_is_checked_in_database` |
| 11 | **N5.** `POST /auth/regenerate-key` | HTTP 500 whenever MongoDB is connected. CHECKED for the failing expression | `if collection:` on a collection object raises. [auth.py:197-206](../../../app/auth.py#L197) | Key is replaced; old key stops working | `test_key_rotation_replaces_key` |
| 12 | **N6.** Google login that returns no e-mail | One shared account with an empty e-mail; platform admin if `ADMIN_EMAIL` is unset. HYPOTHESIS | Empty string compared with empty default; `verified_email` not checked. [auth.py:95](../../../app/auth.py#L95), [auth_routes.py:136-140](../../../app/auth_routes.py#L136) | Login refused | `test_login_requires_verified_email`, `test_empty_admin_email_grants_nothing` |
| 13 | **N11.** `/playground` with `custom_endpoint` pointing at an internal address | The server makes the request. HYPOTHESIS | No check on the URL. [playground.py:92-107](../../../app/routes/playground.py#L92) | Only public `https` hosts; no redirects | `test_playground_blocks_internal_endpoints` |
| 14 | **N12.** Anyone polls `/health/deep` | One outbound Groq completion per call; internal error text returned | Active probe on a public route. [main.py:374-389](../../../app/main.py#L374) | No outbound call and no error text for anonymous callers | `test_health_deep_anonymous_makes_no_outbound_call` |

# 4. Repository/code-path inventory

Every file that reads or writes identity, tenant, cache, feedback, threshold or session
state. Found by searching the whole repository for those concepts, then reading each hit.

| Area | File | Lines | Role in this package |
| --- | --- | --- | --- |
| Identity | [app/auth.py](../../../app/auth.py) | 236 | Users collection, key lookup, token create/verify, usage counter |
| Identity | [app/auth_guard.py](../../../app/auth_guard.py) | 59 | `resolve_user`, `require_user`, `require_admin` |
| Identity | [app/auth_routes.py](../../../app/auth_routes.py) | 206 | Google callback, `/auth/me`, `/auth/users`, key rotation |
| Entry | [app/main.py](../../../app/main.py) | 462 | Lifespan, CORS, middleware, health routes |
| Entry | [app/limiter.py](../../../app/limiter.py) | 41 | Rate limiter keyed on socket address |
| Routes | [app/routes/inference.py](../../../app/routes/inference.py) | 244 | `/track`, `/analyze*`, `/inferences*`, `/diagnose` |
| Routes | [app/routes/monitor.py](../../../app/routes/monitor.py) | 1040 | `/monitor`, `/feedback/{id}`, calibration, signal logs |
| Routes | [app/routes/analytics.py](../../../app/routes/analytics.py) | 473 | `/trend`, `/clusters*`, `/telemetry`, `/analytics/*` |
| Routes | [app/routes/admin.py](../../../app/routes/admin.py) | 150 | Guard config, digest e-mail |
| Routes | [app/routes/flags.py](../../../app/routes/flags.py) | 137 | Review queue; two routes with a broken auth helper |
| Routes | [app/routes/community.py](../../../app/routes/community.py) | 105 | Public demo feedback |
| Routes | [app/routes/playground.py](../../../app/routes/playground.py) | 321 | Playground, custom endpoint |
| Schemas | [app/schemas.py](../../../app/schemas.py) | 370 | `InferenceRequest.tenant_id`, `MonitorRequest.session_id` |
| Storage | [storage/database.py](../../../storage/database.py) | 375 | `inferences`, `feedback`, in-memory fallback |
| Storage | [storage/signal_logger.py](../../../storage/signal_logger.py) | 305 | `signal_logs` |
| State | [engine/ground_truth_cache.py](../../../engine/ground_truth_cache.py) | 215 | Answer cache |
| State | [engine/verifier/ground_truth_pipeline.py](../../../engine/verifier/ground_truth_pipeline.py) | 446 | Cache read, write-through, trace |
| State | [engine/fie_config.py](../../../engine/fie_config.py) | 606 | Thresholds, recalibration |
| State | [engine/retraining/buffer.py](../../../engine/retraining/buffer.py) | 225 | Retrain trigger (its data source does not exist; never completes) |
| State | [engine/session_store.py](../../../engine/session_store.py) | 238 | Session context |
| State | [engine/multi_turn_tracker.py](../../../engine/multi_turn_tracker.py) | 201 | Conversation turns |
| State | [engine/model_extraction_tracker.py](../../../engine/model_extraction_tracker.py) | 257 | Per-tenant rate and probe tracking |
| State | [engine/groq_service.py](../../../engine/groq_service.py) | 477 | Shadow-model response cache |
| State | [engine/archetypes/clustering.py](../../../engine/archetypes/clustering.py), [engine/evolution/tracker.py](../../../engine/evolution/tracker.py) | 175, 130 | Global cluster registry and trend |
| State | [engine/archetypes/registry.py](../../../engine/archetypes/registry.py) | 377 | FAISS attack-pattern index, auto-growth |
| State | [engine/agents/failure_agent.py](../../../engine/agents/failure_agent.py), [engine/agents/adversarial/specialist.py](../../../engine/agents/adversarial/specialist.py) | 322, 423 | Write the global registries; return FAISS evidence |
| State | [engine/hard_positive_collector.py](../../../engine/hard_positive_collector.py), [engine/demo_feedback.py](../../../engine/demo_feedback.py), [app/notifications.py](../../../app/notifications.py) | 227, 241, 267 | Files of raw prompts; public feedback; e-mail |
| SDK, read only | [fie/feedback_store.py](../../../fie/feedback_store.py), [fie/adversarial.py](../../../fie/adversarial.py), [fie/session_tracker.py](../../../fie/session_tracker.py), [fie/client.py](../../../fie/client.py) | — | Learned hashes, scan cache, session tracker, HTTP client. **Not modified** |
| Deploy | [Dockerfile](../../../Dockerfile), [deploy/huggingface/Dockerfile](../../../deploy/huggingface/Dockerfile), [space_app.py](../../../deploy/huggingface/space_app.py), [.github/workflows/ci.yml](../../../.github/workflows/ci.yml) | — | One worker; Space runs API and demo in one process; deploy job reads `/health/deep` |
| Dashboard | [Frontend/src/lib/api.js](../../../Frontend/src/lib/api.js), [auth.js](../../../Frontend/src/lib/auth.js) | — | Calls `/auth/*`, `/inferences`, `/trend`, `/monitor`, `/analyze`, `/diagnose`, `/playground`, `/analytics/usage`, always with a credential header when a session exists |
| Tests | [tests/test_integration.py](../../../tests/test_integration.py) | 306 | Two classes call routes that will need a credential (§21) |

Not on any server route: `engine/pipeline/langgraph_pipeline.py`, `engine/canary_tracker.store_canary`
(never called), `engine/instrumentation` (inactive unless a script starts it).

# 5. Authentication architecture

What exists today. CODE-READ throughout.

| Element | Behaviour | Location |
| --- | --- | --- |
| Credential 1: API key | Header `X-API-Key`. Looked up in `users` by equality on a plain-text field. No index is created on that field | [auth.py:122-147](../../../app/auth.py#L122) |
| Credential 2: session token | Header `Authorization: Bearer`. HS256, 24 h. Payload: e-mail, name, picture, `tenant_id`, **`api_key`**, `is_admin`, plan | [auth.py:210-236](../../../app/auth.py#L210) |
| Credential 3: environment key | If the header equals the server's `FIE_API_KEY` variable, the caller is a platform admin with `tenant_id` = `ADMIN_EMAIL`. Works with the database down. Compared with `==` | [auth.py:131-142](../../../app/auth.py#L131) |
| Signing secret | `JWT_SECRET_KEY`. If unset or short: a warning, and a constant from the source is used | [auth.py:15-24](../../../app/auth.py#L15) |
| Resolution | Token first. If the payload has `email`, `tenant_id` and `api_key` it **is** the user, with no database read. Otherwise the API key | [auth_guard.py:7-32](../../../app/auth_guard.py#L7) |
| Enforcement | A function called at the top of a route body. Three variants: `require_user`, `require_admin`, and `resolve_user` (optional). A route that forgets the call is open. Nothing checks that every route makes one | [auth_guard.py:35-54](../../../app/auth_guard.py#L35) |
| Second implementation | `/auth/me`, `/auth/users`, `/auth/regenerate-key` parse the header themselves | [auth_routes.py:159-206](../../../app/auth_routes.py#L159) |
| Third implementation | `/flags` and `/flags/{id}/label` import `verify_token`, which does not exist, so both always return 401 | [flags.py:131-137](../../../app/routes/flags.py#L131) |
| Account creation | First Google login creates the user, key and tenant. Admin if the e-mail equals `ADMIN_EMAIL` | [auth.py:75-119](../../../app/auth.py#L75) |
| Usage limit | Read-then-increment by tenant; allows the call when the database fails; not reached when the pre-flight guard blocks | [auth.py:171-194](../../../app/auth.py#L171), [monitor.py:110-128](../../../app/routes/monitor.py#L110) |

Roles that exist: platform admin (`is_admin`) and everyone else. There is no tenant admin, no
user inside a tenant, and no service account.

# 6. Tenant identity architecture

**How a tenant comes to exist.** One Google account = one user document = one API key = one
tenant. `tenant_id` is the first ten alphanumeric characters of the e-mail's local part plus
six hex digits ([auth.py:37-41](../../../app/auth.py#L37)). It therefore contains personal data and
has 24 bits of randomness (N14).

**Where the effective tenant comes from, per code path.**

| Path | Source of tenant | Can the caller choose it? |
| --- | --- | --- |
| `GET/DELETE /inferences*`, `/feedback/{id}`, `/notifications/digest` | The resolved user | No |
| `POST /track`, `/track-and-analyze` | **`tenant_id` in the request body**, default `"anonymous"` | **Yes** |
| `POST /monitor` | The resolved user, or the string `"anonymous"` when there is no credential | Shared bucket for every anonymous caller |
| Session context, conversation turns | None. The caller-supplied `session_id` / `conversation_id` is the whole key | **Yes**, by choosing the id |
| Answer cache, thresholds, clusters, trend, FAISS, shadow-response cache, signal logs, retraining buffer, flagged events | None | Not applicable: one copy for all |
| Env-key admin | `ADMIN_EMAIL`, or `local@fie.dev` | No |

CORS allows a request header named `X-Tenant-ID` ([main.py:190](../../../app/main.py#L190)). No
code reads it (N18). It is removed so that it cannot become a tenant source later.

**Can a caller impersonate another tenant by changing a field?** Yes, on `/track` and
`/track-and-analyze`, and for session and conversation state. Nowhere else: no query
parameter and no header selects a tenant.

**Target invariant: the caller does not choose their effective tenant identity.** The tenant
is a property of the verified credential and of nothing else. §16 and §17 give the design.

**The body `tenant_id` on `/track`: decision.** The field stays in the schema (removing it
would reject existing clients with 422). The server never uses it as the source.

| Body value | Response | Reason |
| --- | --- | --- |
| Absent | Stored under the caller's tenant | The normal case |
| Equal to the caller's tenant | Stored | A correct client that echoes its own id keeps working |
| Different from the caller's tenant, including `"anonymous"` sent explicitly | **403**, nothing stored, `authz.tenant_mismatch` logged | See below |

Why 403 and not "ignore": silently storing the record under a different tenant than the
client asked for changes what the call means without telling anyone, and it hides both
integration mistakes and attacks. Why not 422: the body is well formed; the caller is not
allowed to do what it asks, which is what 403 means. The response is identical whether or
not the named tenant exists. Absent and explicit are told apart with Pydantic's
`model_fields_set`. A platform admin gets the same 403: WP-002 adds no "act as tenant".
No first-party caller of `/track` exists in the SDK, the dashboard, the examples or the
scripts (CHECKED by search), so the expected breakage is limited to unknown third parties.

# 7. Route authorization matrix

## ARTIFACT A — Route Security Matrix

All 51 routes. Auth types: **K** API key, **J** session token, **E** environment key.
"Current authorization" is what the code enforces, taken from the route source (CHECKED by a
script that reads each endpoint). Paths are under `/api/v1` unless they start at the root.

| # | Method | Path | Auth required? | Auth type | Tenant source | Admin required? | Current authorization | Data read | Data write | Tenant-scoped? | Global by design? | Security risk | Target behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GET | `/` (root) | No | — | — | No | None | Name, version | — | n/a | Yes | None | Unchanged |
| 2 | GET | `/health` | No | — | — | No | None | DB up/down | — | n/a | Yes | None | Unchanged |
| 3 | GET | `/ready` | No | — | — | No | None | Warm-up state | — | n/a | Yes | None | Unchanged |
| 4 | GET | `/health/deep` | No | — | — | No | None | Component status, error text | Outbound Groq call | n/a | Partly | N12: quota burn, internal errors | Stays public with the same top-level shape. Anonymous: no outbound call, error class only. Full detail for a platform admin |
| 5–8 | GET | `/docs`, `/docs/oauth2-redirect`, `/openapi.json`, `/redoc` | No | — | — | No | None | Route list | — | n/a | Yes | Low. The code is public | Unchanged |
| 9 | POST | `/auth/google-callback` | No | Google code | Created here | No | Rate limit 10/min | Google profile | `users` | Own | — | N6; S9 key in token; Google's error text returned | Refuse missing or unverified e-mail. No key in the token. Generic error text |
| 10 | GET | `/auth/me` | Yes | J, K | Credential | No | Own parser | Own user | — | Yes | — | Second auth implementation | Shared dependency |
| 11 | POST | `/auth/regenerate-key` | Yes | J only | Credential | No | Own parser | — | `users.api_key` | Yes | — | N5: always fails | Works; old key invalid at once; event logged |
| 12 | GET | `/auth/users` | Yes | J only | — | Yes | `is_admin` from the token | **All users with API keys** | — | No | Admin view | S9, S11 | Admin checked in DB. `api_key` removed from the response |
| 13 | POST | `/track` | **No** | — | **Body** | No | **None** | — | `inferences` | **No** | No | S1, N1 | Credential required. Tenant from credential. Mismatch → 403. Write cannot touch another tenant's document |
| 14 | POST | `/track-and-analyze` | **No** | — | **Body** | No | **None** | — | `inferences` | **No** | No | S1, S2, N1 | Same as 13 |
| 15 | POST | `/analyze` | **No** | — | — | No | **None** | — | — (CPU) | n/a | No | S2: free compute | Credential required |
| 16 | POST | `/analyze/v2` | **No** | — | — | No | **None** | Global trend | Global clusters, trend | **No** | No | S2, S3 | Credential required. Tenant's own registry and trend |
| 17 | POST | `/diagnose` | Optional | J, K | — | No | Hides internal explanation from non-admins | FAISS index | Global clusters, trend | **No** | No | S2, N4 | Credential required. Tenant's own registry. Learned prompt text never returned |
| 18 | GET | `/inferences` | Yes | J, K, E | Credential | No | `require_user`; admin sees all | `inferences` | — | Yes; admin: all | — | N7 | Own tenant. `all_tenants=true` for a DB-verified admin, audited |
| 19 | GET | `/inferences/export/csv` | Yes | J, K, E | Credential | No | Same | Same | — | Same | — | N7 | Same as 18 |
| 20 | GET | `/inferences/grouped/by-question` | Yes | J, K, E | Credential | No | Same | Same | — | Same | — | N7 | Same as 18 |
| 21 | GET | `/inferences/{id}` | Yes | J, K, E | Credential | No | Scoped lookup; admin unscoped | One record | — | Yes | — | Low | Own tenant. Admin cross-tenant read explicit and audited |
| 22 | DELETE | `/inferences/{id}` | Yes | J, K, E | Credential | No | Scoped; admin unscoped | — | Delete one | Yes; admin: any | — | Admin deletes any tenant's record | Own tenant only, for everyone |
| 23 | DELETE | `/inferences` | Yes | J, K, E | Credential | No | Scoped; **admin deletes all tenants** | — | Delete many | Admin: no | — | N7 | Own tenant only, for everyone |
| 24 | POST | `/monitor` | Optional | J, K, E | Credential or `"anonymous"` | No | Rate limit 60/min per address | Cache, thresholds, sessions, FAISS, trend | `inferences`, `signal_logs`, sessions, turns, clusters, trend, FAISS, extraction log, e-mail | **Partly** | No | S2, S4, S5, S7, N3, N4, N9, N10, N16 | Credential required. Every read and write in the caller's scope |
| 25 | GET | `/monitor/status` | No | — | — | No | None | Ollama status | Outbound probe | n/a | — | Low | Credential required |
| 26 | GET | `/monitor/model-info` | No | — | — | No | None | Global thresholds, config version | — | n/a | Platform config | Shows threshold movement to anyone | Credential required |
| 27 | GET | `/monitor/calibration` | Yes | J, K, E | — | Yes | `require_admin` (token flag) | Aggregates of all tenants' labels | — | No | Admin view | S9 | Admin checked in DB |
| 28 | GET | `/monitor/signal-logs` | Yes | J, K, E | — | Yes | `require_admin` | **Raw prompts and outputs of all tenants** | — | No | Admin view | S9; no audit trail | Admin checked in DB; read audited; rows carry tenant |
| 29 | POST | `/feedback/{id}` | Yes | J, K, E | Credential | No | Scoped record lookup; admin unscoped | Record, signal log | `feedback`, **global cache**, signal log label, **thresholds**, retrain buffer | **Partly** | No | S4, S5, S6, N8, N13 | Own records only. Cache entry in own scope. No recalibration, no retrain trigger |
| 30 | GET | `/trend` | **No** | — | — | No | **None** | Global trend | — | **No** | No | S3 | Credential required. Own trend |
| 31 | GET | `/clusters` | **No** | — | — | No | **None** | Global clusters with answer text | — | **No** | No | S3, N3 | Credential required. Own clusters |
| 32 | DELETE | `/clusters/reset` | **No** | — | — | No | **None** | — | Wipes global registry | **No** | No | S3 | Credential required. Resets own registry |
| 33 | POST | `/telemetry` | No | — | — | No | Rate limit 30/min | — | `sdk_telemetry` | n/a | Yes, anonymous | Unbounded inserts within the limit | Unchanged |
| 34–39 | GET | `/analytics/usage`, `/model-performance`, `/calibration`, `/question-breakdown`, `/paper-metrics`, `/sdk-telemetry` | Yes | J, K, E | — | Yes | `require_admin` (token flag); exception text returned | Aggregates over all tenants | — | No | Admin view | S9; error text | Admin checked in DB; generic errors |
| 40 | GET | `/admin/guard/config` | Yes | J, K, E | — | Yes | `require_admin` | Global guard config | — | No | Platform config | S9 | Admin checked in DB |
| 41 | POST | `/admin/guard/config` | Yes | J, K, E | — | Yes | `require_admin` | — | **Guard on/off for every tenant** | No | Platform config | S9: a stale admin token can disable blocking for all | Admin checked in DB; change audited |
| 42 | POST | `/notifications/digest` | Yes | J, K, E | Credential | No | `resolve_user` + 401 | Own inferences | E-mail to self | Yes | — | Exception text returned | Shared dependency; generic errors |
| 43 | POST | `/playground` | Yes | J, K, E | Credential | No | `require_user` | FAISS | Global clusters, trend; outbound request to a caller URL | Partly | No | N11; operator Groq quota without metering | Own registry; endpoint restricted to public `https` hosts |
| 44 | GET | `/flags` | Yes | J | — | No | **Broken: always 401** | Flagged events of all callers | — | No | Review queue | Dead control; if "repaired" as written, any user would read all | Platform admin |
| 45 | POST | `/flags/{id}/label` | Yes | J | — | No | **Broken: always 401** | — | Label; global block/allow hash | No | Platform control | S8 if repaired as written | Platform admin; audited |
| 46 | GET | `/flags/export` | Yes | J, K, E | — | Yes | `require_admin` | Confirmed attacks | — | No | Admin view | S9 | Admin checked in DB |
| 47 | GET | `/flags/hard-positives/stats` | Yes | J, K, E | — | Yes | `require_admin` | Counts, file paths | — | No | Admin view | S9 | Admin checked in DB |
| 48 | GET | `/flags/hard-positives/export` | Yes | J, K, E | — | Yes | `require_admin` | Raw prompts of all callers | — | No | Admin view | S9 | Admin checked in DB; read audited |
| 49 | POST | `/community/feedback` | No | — | — | No | Rate limit 20/h | — | `demo_feedback` | n/a | Yes, public | Low | Unchanged |
| 50 | GET | `/community/stats` | No | — | — | No | None | Counts | — | n/a | Yes, public | None | Unchanged |
| 51 | GET | `/community/export` | Yes | J, K, E | — | Yes | `require_admin` | Public feedback set | — | n/a | Admin view | S9 | Admin checked in DB |

Space only: `GET /api` (banner) and the Gradio application mounted at `/`. The demo calls the
scanner inside the process, not through HTTP, so requiring credentials on the API does not
affect it. It shares the scan cache and the flagged-events queue with the API (N20).

Summary. 51 routes reviewed. 10 stay exactly as they are (1–3, 5–8, 33, 49, 50). 41 change:
11 gain a credential requirement (13–17, 24–26, 30–32); 2 are repaired and become admin
routes (44, 45); 15 other admin routes move to a database-checked admin (12, 27, 28, 34–41,
46–48, 51); and 13 change scope or response content (4, 9–11, 18–23, 29, 42, 43).

# 8. Complete mutable-state inventory

## ARTIFACT B — Tenant State Matrix

41 items. "X" = one tenant can affect or observe another through this item today. "t" = by
timing or availability only. "L" = latent: the path exists but cannot be reached today.

| ID | State | Location | Purpose | Current scope | Tenant key | Write path | Read path | Crosses | Risk | Target scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ST-01 | `users` collection | [auth.py:54-72](../../../app/auth.py#L54) | Accounts, keys, admin flag, usage | Per user | `tenant_id`, `email` | Login, key rotation, usage counter | Every authenticated call; `/auth/users` | — | Plain-text keys returned to admins | Tenant. Keys never returned in lists |
| ST-02 | `inferences` collection | [database.py:150-171](../../../storage/database.py#L150) | Stored prompt/answer records | Field `tenant_id`; `_id` global | `tenant_id` field | `/track`, `/track-and-analyze`, `/monitor` | `/inferences*`, `/feedback`, digest | X | S1, N1, N7 | Tenant. Document id namespaced by tenant |
| ST-03 | `feedback` collection | [database.py:326-341](../../../storage/database.py#L326) | Corrections | Field `tenant_id` | `tenant_id` | `/feedback/{id}` | No route | X | N8 | Tenant |
| ST-04 | `signal_logs` collection | [signal_logger.py:27-142](../../../storage/signal_logger.py#L27) | Raw signals, prompt and outputs, labels | **Global** | **None** | `/monitor`, `/feedback` | Recalibration, admin analytics, `/monitor/signal-logs` | X | N13, S5 | Tenant |
| ST-05 | `ground_truth_cache` collection | [ground_truth_cache.py:142-188](../../../engine/ground_truth_cache.py#L142) | Verified answers | **Global** | **None** | `/feedback`; write-through from the pipeline | `/monitor` | X | S4, N2, N19 | Tenant |
| ST-06 | `fie_config` document | [fie_config.py:104-115](../../../engine/fie_config.py#L104) | Thresholds, guard mode | Global | — | Admin route; **recalibration from feedback** | Startup load | X | S5, S6 | Global, admin controlled |
| ST-07 | `session_context` collection | [session_store.py:102-213](../../../engine/session_store.py#L102) | Conversation history, 24 h | **Global** | **None** (`session_id`) | `/monitor` | `/monitor` | X | N10 | Tenant + session |
| ST-08 | `conversation_turns` collection | [multi_turn_tracker.py:62-129](../../../engine/multi_turn_tracker.py#L62) | Escalation tracking, 2 h | **Global** | **None** (`conversation_id`) | `/monitor` | `/monitor` | X | N10 | Tenant + conversation |
| ST-09 | `model_extraction_tracking` collection | [model_extraction_tracker.py:84-141](../../../engine/model_extraction_tracker.py#L84) | Rate and probe window, 1 h | Per tenant; `"anonymous"` shared | `tenant_id` | `/monitor` | `/monitor` | X | Shared anonymous bucket | Tenant (no anonymous bucket once credentials are required) |
| ST-10 | `retraining_buffer` collection | [buffer.py:17-58](../../../engine/retraining/buffer.py#L17) | Counter of new labels | **Global** | **None** | `/feedback` | Retrain trigger | X L | Cross-tenant trigger of a job that cannot finish | Tenant field; trigger off |
| ST-11 | `flagged_events` collection | [feedback_store.py:97-147](../../../fie/feedback_store.py#L97) | Blocks awaiting review; prompt hash and 140-character excerpt | **Global** | **None** | Every block by the scanner, including the public demo | `/flags*` | X L | S8, N20 | Global, admin controlled |
| ST-12 | `demo_feedback` collection | [demo_feedback.py:65-168](../../../engine/demo_feedback.py#L65) | Public reports | Global | — | Public | Admin export, public counts | — | None | Global by design |
| ST-13 | `sdk_telemetry` collection | [analytics.py:38-59](../../../app/routes/analytics.py#L38) | Anonymous pings | Global | — | Public | Admin | — | None | Global by design |
| ST-14 | FAISS index and metadata files | [registry.py:266-280](../../../engine/archetypes/registry.py#L266) | Attack patterns with raw prompt text | **Global** | **None** | `/monitor` auto-growth | Jury on `/monitor`, `/diagnose`, `/playground` | X | S7, N4 | Global, immutable seed |
| ST-15 | Hard-positive files | [hard_positive_collector.py:14-16](../../../engine/hard_positive_collector.py#L14) | Raw blocked prompts for retraining; off unless a flag is set | Global | None | Scanner blocks | Admin export | X L | Raw text of all callers in one file | Global, admin controlled |
| ST-16 | `~/.fie/flagged_events.jsonl` | [feedback_store.py:40-49](../../../fie/feedback_store.py#L40) | Fallback for ST-11 when the database is down | Global | None | Scanner blocks | `/flags*` | X L | As ST-11 | Global, admin controlled |
| ST-17 | `data/demo_feedback.jsonl` | [demo_feedback.py:49](../../../engine/demo_feedback.py#L49) | Fallback for ST-12 | Global | — | Public | Admin | — | None | Global by design |
| ST-18 | `models/xgboost_retrained.pkl` | [buffer.py:212-217](../../../engine/retraining/buffer.py#L212) | Output of the retrain job. Nothing loads it | Global | — | Retrain thread | None | L | A model trained on all tenants' labels | Must not be produced from a request path |
| ST-19 | `_fallback_records` | [database.py:15](../../../storage/database.py#L15) | In-memory inferences when the database is down | As ST-02 | Field | As ST-02 | As ST-02 | X | As ST-02 | Tenant |
| ST-20 | `_fallback`, `_fallback_summaries` | [session_store.py:15-17](../../../engine/session_store.py#L15) | In-memory sessions | As ST-07 | None | As ST-07 | As ST-07 | X | As ST-07 | Tenant + session |
| ST-21 | `archetype_registry` | [clustering.py:175](../../../engine/archetypes/clustering.py#L175) | Failure clusters, with centroid answers | **Process-global** | **None** | `/monitor`, `/diagnose`, `/analyze/v2`, `/playground` | `/clusters`; reset by anyone | X | S3, N3 | Tenant |
| ST-22 | `evolution_tracker` | [tracker.py:130](../../../engine/evolution/tracker.py#L130) | Trend averages | **Process-global** | **None** | Same | `/trend`, spike e-mail | X | S3, N16 | Tenant |
| ST-23 | `adversarial_registry` in memory | [registry.py:377](../../../engine/archetypes/registry.py#L377) | As ST-14 | Global | None | As ST-14 | As ST-14 | X | As ST-14 | Global, immutable seed |
| ST-24 | `_response_cache` | [groq_service.py:22](../../../engine/groq_service.py#L22) | Shadow-model answers, 1 h | **Process-global** | **None** | Any Groq call | Any Groq call | X | N9 | Tenant; none when no scope is given |
| ST-25 | Live thresholds and guard mode | [fie_config.py:81-97](../../../engine/fie_config.py#L81) | In-memory copy of ST-06 | Global | — | Recalibration thread; admin route | `/monitor`, scanner | X | S5 | Global, admin controlled |
| ST-26 | Learned block/allow hash sets | [feedback_store.py:19-20](../../../fie/feedback_store.py#L19) | Instant verdict for labelled prompts | Process-global | None | Label route (unreachable); startup load | Every scan | X L | S8 | Global, admin controlled |
| ST-27 | Scan-verdict cache | [adversarial.py:298](../../../fie/adversarial.py#L298) | Verdict for identical text, 5 min, 512 entries | Process-global | None | Every scan | Every scan | t | Timing shows a recent identical scan | Global, derived. Accepted (§18) |
| ST-28 | Translation cache | [multilingual.py:46](../../../fie/multilingual.py#L46) | Translations, 512 entries | Process-global | None | Scanner | Scanner | t | As ST-27 | Global, derived. Accepted |
| ST-29 | SDK session tracker; optional Redis key `fie:session:{id}` (pickled) | [session_tracker.py:236-387](../../../fie/session_tracker.py#L236) | Multi-turn boost | Process or Redis | None | Scan with a `session_id` | Same | — | The server never passes a session id to the scanner. If it ever does, the key must carry the tenant. Unpickling from Redis belongs to WP-004 | Not on the server path. Recorded |
| ST-30 | `_memory_store` | [model_extraction_tracker.py:64](../../../engine/model_extraction_tracker.py#L64) | Fallback for ST-09 | Per tenant; anonymous shared | `tenant_id` | `/monitor` | `/monitor` | X | As ST-09 | Tenant |
| ST-31 | `_canary_store` | [canary_tracker.py:225](../../../engine/canary_tracker.py#L225) | Canary per conversation. `store_canary` is never called | — | — | None | None | — | Dead | Leave |
| ST-32 | `_spike_last_sent` | [notifications.py:181](../../../app/notifications.py#L181) | One alert per hour | Per tenant | `tenant_id` | `/monitor` | `/monitor` | — | Low | Tenant |
| ST-33 | `_seen_hashes` | [demo_feedback.py:62](../../../engine/demo_feedback.py#L62) | Duplicate filter for public reports; grows without bound | Global | — | Public | Public | — | Memory growth | Global by design |
| ST-34 | `_is_retraining` | [buffer.py:12](../../../engine/retraining/buffer.py#L12) | Single-run guard | Global | — | Retrain trigger | Same | — | None | Global |
| ST-35 | Rate-limit buckets | [limiter.py:10](../../../app/limiter.py#L10) | Request limits | Per socket address | — | Four limited routes | Same | t | S12 | Per tenant for authenticated traffic |
| ST-36 | Signing secret, admin e-mail, Mongo client | [auth.py:15-51](../../../app/auth.py#L15) | Read once at import | Global | — | Environment | Auth | — | S10 | Global, immutable |
| ST-37 | Loaded models, warm-up state, settings | [main.py:82](../../../app/main.py#L82), [config.py:209-211](../../../config.py#L209) | Model weights and static configuration | Global | — | Startup | Everything | — | None | Global, immutable |
| ST-38 | Instrumentation collector | [engine/instrumentation.py](../../../engine/instrumentation.py) | Counts and timings for benchmark scripts. Inactive on the server | Global | — | Scripts | Scripts | — | None | Request |
| ST-39 | Groq client and the operator's quota | [groq_service.py:446-477](../../../engine/groq_service.py#L446) | One key for all tenants | Global | — | Every monitor call | — | t | One tenant can exhaust the shared quota | Global. Metering deferred |
| ST-40 | In-memory inference fallback when the database fails mid-run | [database.py:155-161](../../../storage/database.py#L155) | Availability | As ST-19 | Field | — | — | — | Data silently kept only in memory | Tenant |
| ST-41 | Session token in the browser's `localStorage` | [Frontend/src/lib/auth.js](../../../Frontend/src/lib/auth.js) | Dashboard session | Per browser | — | Login | Every call | — | S9: the token carries the API key | Token without the key |

Counts, from the table. 32 items hold tenant data or change a tenant's results (all except
ST-12, 13, 17, 31, 33, 34, 36, 37, 38). **27 items cross tenants today** (the X, t and L
rows), which are **19 distinct channels** once each in-memory copy is paired with its
persisted form: inferences; feedback; signal logs; answer cache; thresholds; session context;
conversation turns; extraction tracking; retraining buffer; flagged events with learned
hashes; attack-pattern index; hard-positive files; clusters; trend; shadow-response cache;
scan-verdict cache; translation cache; rate-limit buckets; shared Groq quota. Four of the 19
are timing or availability only (the last four). Three are latent (retraining, learned
hashes, hard-positive files).

# 9. Cache architecture and isolation analysis

There are four caches. Only the first changes what a tenant is told.

## 9.1 Answer cache (`ground_truth_cache`): complete lifecycle

| Stage | Today | Location |
| --- | --- | --- |
| Write A: feedback | A tenant marks an answer wrong and supplies a correction. Stored with `source="user_feedback"`, `confidence=1.0`, `verified_by=<submitter e-mail>`. The question text is the stored inference's prompt | [monitor.py:958-970](../../../app/routes/monitor.py#L958) |
| Write B: write-through | The pipeline stores any answer it verified at ≥ 0.90: Wikidata, Serper, and shadow-model self-consistency for reasoning and code questions. `verified_by="system"` | [ground_truth_pipeline.py:427-446](../../../engine/verifier/ground_truth_pipeline.py#L427) |
| Key | `_id` = first 32 hex of SHA-256 of the trimmed, lower-cased question. Nothing else | [ground_truth_cache.py:60-63](../../../engine/ground_truth_cache.py#L60) |
| Stored | Question text, 384-dimension question vector, answer, source, confidence, `verified_by`, timestamps, use count | [:162-173](../../../engine/ground_truth_cache.py#L162) |
| Read | Exact id first. Otherwise load **every** document and take the best cosine match; hit at ≥ 0.92 | [:79-139](../../../engine/ground_truth_cache.py#L79) |
| Effect of a hit | The pipeline returns at once with the cached answer, `from_cache=true`, label "FULLY_PROVENANCED", and a trace line with the answer, `verified_by` and the use count. If the answer differs from the model's output, `/monitor` returns it as an applied fix | [ground_truth_pipeline.py:139-154](../../../engine/verifier/ground_truth_pipeline.py#L139), [monitor.py:457-480](../../../app/routes/monitor.py#L457) |
| Invalidation | None. A later write for the identical question overwrites; there is no delete | — |
| Expiration | None | — |

What is missing from the key and why it matters:

| Context | In the key today? | Needed? | Reason |
| --- | --- | --- | --- |
| Tenant | No | **Yes** | The whole finding |
| User inside the tenant | No | No | FIE has no such principal |
| Source class (feedback vs system) | No, a field only | Yes, as a field checked on read | A system write must not overwrite a tenant's own correction, and the reverse must be visible |
| Model identity, policy version | No | No | The cached value is a fact about the question, not a model verdict. A schema version is added so a later format change can drop old entries |
| Prompt | Yes, normalised | Yes | — |
| Mode (monitor / correct) | No | No | The mode decides what the client does with the answer |
| Question text and vector at rest | Stored in clear | Kept, inside the tenant's scope | Needed for the semantic match. No longer visible across tenants |

## 9.2 Shadow-response cache (`_response_cache`)

In-process, one hour, about 500 entries. Key: model name and prompt. The system message is
not in the key, although `/monitor` sends a fresh canary token in it on every request
([monitor.py:148-158](../../../app/routes/monitor.py#L148)). On a hit the caller receives answers
generated for someone else's request, and the canary comparison is made against a token that
was never sent. Latency also reveals the hit.

## 9.3 Scan-verdict cache and translation cache

Both live in `fie/`. The cached value is a pure function of the text, the shipped model and
platform configuration. No tenant input other than the text itself can change it, and it
cannot be written directly. The only cross-tenant effect is timing. Both are left as they
are and recorded as an accepted residual (§18).

## 9.4 The choice

| Option | Verdict |
| --- | --- |
| A. Tenant-scoped | **Chosen for the answer cache.** Every entry carries the tenant; every read filters on it; entries without a tenant are never served |
| B. User-scoped | Not applicable: no user level exists |
| C. Request-scoped | Would remove the feature. The cache exists so that a tenant's correction applies to that tenant's later traffic |
| D. Remove | Larger behaviour change than needed; the roadmap asks only to stop cross-tenant substitution |
| E. Hybrid: tenant entries plus a global set of system-verified answers | Rejected for WP-002. The "system" answers are derived from one tenant's prompt and model output (N19), and a global hit still tells one tenant what another asked. A platform-curated global set can be designed later with its own write path |

Consequence the owner should know: after deployment the cache is empty for every tenant,
because no existing entry has a tenant. Hit rate starts at zero and external lookups rise
until each tenant's own entries build up.

# 10. Feedback/recalibration isolation analysis

The chain today, CODE-READ end to end:

```text
POST /feedback/{id}
  → record looked up (tenant-scoped; admin: unscoped)              monitor.py:947
  → correction → GLOBAL answer cache                                monitor.py:959
  → signal log found by request_id alone → label written            monitor.py:973, signal_logger.py:145
  → feedback document (has tenant_id)                               monitor.py:988
  → maybe_recalibrate(): count ALL labelled logs; every 50 new,
      start a thread → recalibrate()                                fie_config.py:585
        → best-F1 threshold per question type from ALL labels,
          clamped to [0.25, 0.75]
        → replace the stored config document (drops attack overrides)
        → swap the process-global thresholds
  → retraining buffer +1 (global); at 500, start a retrain thread   buffer.py:90
        → imports a function that does not exist → stops
  → every later /monitor, any tenant: get_threshold(question_type)  monitor.py:636
```

| Question | Answer |
| --- | --- |
| Is feedback stored globally? | The feedback document has a tenant. The label on the signal log and the buffer row do not |
| Does it update global thresholds? | Yes, automatically, every 50 labels from any mix of tenants |
| Does it update tenant thresholds? | There are none |
| Reload and persistence | New thresholds are written to the database and swapped in memory at once. After a restart the counter starts at zero, so the first feedback triggers a recalibration if 50 labels exist |
| Other workers | One worker is configured. With more, each would recalibrate separately |
| Other tenants | Yes. A tenant controls its own prompts, its own "model answers" and its own labels, so it controls the points on the curve the thresholds are fitted to |
| Does it change the guard? | No. These thresholds belong to the hallucination monitor. The guard's thresholds change only through the admin route |
| Should safety thresholds change online at all? | Not from a request path. A change to shared decision thresholds should be a deliberate, attributable platform action with a recorded version |

**The safe temporary option, evaluated: disable automatic recalibration, keep collecting
feedback.** Chosen. Cost: thresholds stop adapting. They return to being what the operator
set. Feedback, labels and the buffer are still recorded, now with a tenant on each row, so a
later per-tenant or reviewed design has the data. `recalibrate()` stays in the code as a
function a platform admin can run deliberately; it is not exposed as a route in this
package, and it is corrected so that it no longer drops the attack-threshold overrides. The
retrain trigger is switched off explicitly, so that repairing its missing import later does
not silently start cross-tenant training.

Not decided here: what the right long-term design is (per-tenant thresholds, reviewed global
recalibration, or none). That belongs to the hallucination research track.

# 11. Session isolation analysis

Three things are called "session". They are different and must not be merged.

| Store | Identifier | Chosen by | Tenant in key? | Content | Reaches a response? |
| --- | --- | --- | --- | --- | --- |
| `engine/session_store` (ST-07, 20) | `session_id`, 1–128 of `[A-Za-z0-9_-]` | The caller | No | Up to 10 turns of prompt and answer, 4,000 characters each, plus a summary written by sending older turns to Groq | Indirectly. The fetched context is not sent to shadow models; only "has context" changes the archetype label ([monitor.py:204-214](../../../app/routes/monitor.py#L204)). So a read leaks existence, not text |
| `engine/multi_turn_tracker` (ST-08) | `conversation_id`, same pattern | The caller | No | 500 characters of each prompt, categories, adversarial flag | Yes: `multi_turn_escalation` in the response |
| `fie/session_tracker` (ST-29) | `session_id` argument of the scanner | SDK caller | No | Prompt hashes and verdicts; pickled if Redis is configured | Not used by any server route |

Answers to the brief's questions. Sessions are identified only by a caller-supplied string.
Tenant identity is not part of the key. Data is stored in one global collection (or one
dictionary). Serialization for the two server stores is plain documents; the SDK tracker
uses pickle with Redis, which is not on the server path. Text cannot be retrieved across
tenants through the session store, but existence can be inferred, and both stores can be
written across tenants. Identifiers are attacker-controlled and guessable if a client uses
simple values such as user ids.

Target: the key is the pair (tenant, id). The same `session_id` used by two tenants is two
unrelated sessions. The client-facing field and its format do not change. Session identity
stays what it is — a label the client chooses for a conversation — and is never treated as
proof of who the caller is.

# 12. Admin boundary analysis

| Question | Finding |
| --- | --- |
| How is admin identity established? | `is_admin` on the user document, set once at account creation when the e-mail equals `ADMIN_EMAIL`. Or the environment key. On token requests the flag is read from the token, valid 24 h, with no database check |
| What tenant context do admin requests use? | On data routes an admin has no tenant filter at all: list, export, read, delete and feedback act on every tenant |
| Can admin routes expose tenant data by accident? | Yes. The admin's ordinary dashboard view lists every tenant's prompts, and the "clear" action deletes all of them (N7) |
| Can tenant filters be bypassed? | Not by a non-admin on the inference routes. They are bypassed by design for admins, with no explicit request and no audit record |
| Do support or debug routes expose shared state? | `/monitor/signal-logs` returns raw prompts and outputs of all tenants. `/flags/hard-positives/export` returns raw blocked prompts. `/auth/users` returns every API key. `/monitor/model-info`, `/health/deep`, `/trend` and `/clusters` are public |

Roles, as they are and as they will be after this package:

| Role | Exists today? | After WP-002 |
| --- | --- | --- |
| Platform admin | Yes | Same people. Flag verified in the database on each admin call. Cross-tenant **read** only on explicit request and always audited. No cross-tenant write or delete through data routes |
| Tenant admin | **No** | Not added. Gap recorded |
| Normal user inside a tenant | **No.** One key is the whole tenant | Not added. Gap recorded |

No RBAC system is introduced. The change is that "admin" stops meaning "no tenant filter".

# 13. Rate limiting analysis

| Aspect | Finding |
| --- | --- |
| Mechanism | `slowapi`, in-process memory, optional: if the package is missing the server runs without limits and logs a warning |
| Key | `get_remote_address`: the socket peer address |
| Scope | Per address, per route, per process |
| Limited routes | `/monitor` 60/min, `/telemetry` 30/min, `/community/feedback` 20/h, `/auth/*` 5–60/min. The other 36 `/api/v1` routes have no limit |
| Behind a proxy | Unknown. If the Space's proxy presents one address, all users share one bucket. **Not tested; recorded as unknown, not guessed** |
| Tenant-isolation risk | If the bucket is shared, one tenant's traffic denies `/monitor` to all others. If it is not, limits are per address and unrelated to tenants |
| Usage quota | A separate monthly counter per tenant. It allows the call on a database error, is not atomic, and is skipped when the pre-flight guard blocks. It concerns a tenant's own usage, not another tenant's data; left alone here |

Proposed, small, and optional (decision W2-11): the limiter's key becomes the tenant for
authenticated requests and stays the address otherwise. That removes the shared-bucket
effect for authenticated traffic whatever the proxy does. Forwarded-address headers are not
trusted, because the proxy's behaviour is unknown. Everything else about rate limiting stays
for WP-017.

# 14. Data leakage analysis

| # | Where | What leaks | To whom | Plan |
| --- | --- | --- | --- | --- |
| DL-1 | Jury evidence `faiss_result.nearest_prompt` | 120 characters of another caller's prompt | Any caller of `/monitor`, `/diagnose` | Return text only for seed-corpus entries |
| DL-2 | `GET /clusters` centroid | Normalised model answers of all callers | Anyone | Per-tenant registry; credential required |
| DL-3 | Answer-cache trace | Submitter's e-mail, use count | Any tenant with a similar question | Trace states the source class only |
| DL-4 | Spike alert e-mail | All tenants' failure rate and inference count | The tenant that happened to send the request | Per-tenant trend |
| DL-5 | Error responses | `detail=str(exc)` on flags, admin and analytics routes; Google's response body on login failure | Callers | Generic message; detail goes to the log with the request id |
| DL-6 | `/health/deep` | Internal exception text, component inventory | Anyone | Error class only for anonymous callers |
| DL-7 | `/monitor/model-info` | Global thresholds and their version, which move when others submit feedback | Anyone | Credential required. Movement stops anyway once recalibration is manual |
| DL-8 | `/auth/users` | Every user's API key | Platform admin | Field removed from the response |
| DL-9 | Session token | The API key | Anyone who reads the token | Key removed from the payload |
| DL-10 | Logs | E-mail at login; tenant id (contains the e-mail's local part); corrected answer text; cached answer text; conversation ids | Log readers | Security events use a hashed tenant reference. Existing application log lines are reduced only where this package already edits the line |
| DL-11 | `signal_logs`, `conversation_turns`, `model_extraction_tracking`, `flagged_events`, FAISS metadata, hard-positive files | Raw or partial prompt text at rest with no tenant on the row | Database and file readers; admin routes | Tenant recorded on new rows. Retention and minimisation are WP-016 |
| DL-12 | Identifiers | `tenant_id` embeds part of the e-mail | Anywhere the id appears | Not changed: changing ids is a migration. Recorded for WP-017 |
| DL-13 | Existence of objects | Inference id: no leak (404 either way). Cache: `from_cache`. Sessions: label difference | Tenants | Removed by scoping |
| DL-14 | `X-Request-ID` | Client value copied into logs and the response without a length or character limit | Log readers | Accept only 1–64 of `[A-Za-z0-9_-]`; otherwise generate one |

Cache keys already hash the text they are built from. The new keys keep that property: no
raw prompt and no raw tenant id is used as a key where a hash will do.

# 15. Security invariants

## ARTIFACT D — Security Invariants

| # | Invariant | Current status | Future enforcement | Regression test |
| --- | --- | --- | --- | --- |
| I-1 | An unauthenticated caller cannot read or write tenant data | **Violated**: routes 13, 14, 16, 17, 24, 30–32 | Every route declares a policy; a test fails for a route without one | `test_route_matrix.py` (one case per route), `test_authn.py` |
| I-2 | The verified credential determines the tenant | **Violated** on `/track`, `/track-and-analyze`, sessions, conversations | The tenant exists only on the `Principal` built by the authentication dependency | `test_tenant_spoofing.py` |
| I-3 | A caller-supplied tenant id cannot raise access | **Violated** | A body tenant that differs → 403 | `test_track_body_tenant_mismatch_is_403` |
| I-4 | Tenant A cannot read tenant B's cached answer | **Violated** | Tenant in the cache key and in every query; entries without a tenant are not served | `test_answer_cache_isolation.py` |
| I-5 | Tenant A cannot write tenant B's state | **Violated** (records, sessions, turns, clusters, trend, FAISS, cache) | All tenant state reached through a scoped accessor that needs a `Principal` | `test_inference_isolation.py`, `test_shared_state.py` |
| I-6 | Tenant A's feedback cannot change tenant B's thresholds or state | **Violated** | No recalibration or retraining starts from a request | `test_feedback_isolation.py` |
| I-7 | Tenant A's sessions cannot be read or written by tenant B | **Violated** | Key = (tenant, id) | `test_session_isolation.py` |
| I-8 | An unauthorized request does not reveal that another tenant's object exists | Holds for inference ids; **violated** for cache and sessions | Same response for "not yours" and "missing" | `test_exposure.py` |
| I-9 | Shared state is immutable or changed only by a platform admin | **Violated**: thresholds, FAISS, clusters, trend | §18 classification; a static test forbids request-path writes to the listed global stores | `test_static_boundary.py`, `test_shared_state.py` |
| I-10 | Each stored decision is attributable to a tenant and a configuration version | Partly: inference and feedback rows have a tenant; signal logs do not | Tenant on every tenant-scoped row; `config_version` already on responses | `test_rows_carry_tenant` |
| I-11 | *(new)* Without a strong signing secret no session token is issued or accepted, and the server does not start | **Violated** | Check at token use and at startup | `test_no_secret_no_tokens`, `test_startup_fails_without_secret` |
| I-12 | *(new)* Admin rights are read from the database at the time of the admin action | **Violated** | Admin dependency reads the user document; database unavailable → 503 | `test_admin_boundary.py` |
| I-13 | *(new)* No response contains text or identity that came from another tenant | **Violated**: DL-1, 2, 3, 4 | Scoping, plus field removal | `test_exposure.py` |
| I-14 | *(new)* If the tenant cannot be determined the request fails; it is never served as global or anonymous | **Violated**: `"anonymous"` | No anonymous tenant exists. Constructing a scope without a tenant raises | `test_failure_modes.py` |
| I-15 | *(new)* A platform admin's cross-tenant read is explicit and recorded; data routes give an admin no cross-tenant write or delete | **Violated** | `all_tenants=true` plus an audit event; write and delete always scoped | `test_admin_boundary.py` |
| I-16 | *(new)* A credential can be revoked: after key rotation the old key is refused | **Violated** (rotation fails) | Fixed rotation | `test_key_rotation_replaces_key` |
| I-17 | *(new)* The server does not make requests to addresses chosen by a tenant inside its own network | **Violated** (HYPOTHESIS) | URL check before the request | `test_playground_blocks_internal_endpoints` |

# 16. Proposed target architecture

One path, used by every route, with no way around it:

```text
HTTP request
  │
  ▼
AUTHENTICATION      app/auth_guard.py: authenticate(request) -> Principal | 401
  │                 the only code that reads Authorization / X-API-Key
  ▼
IDENTITY            Principal (frozen): tenant_id, subject, role, credential_kind, request_id
  │                 tenant_id is never empty and never "anonymous"
  ▼
AUTHORIZATION       route dependency: require_tenant | require_platform_admin | public
  │                 a route with no declared policy fails the route-matrix test
  ▼
TENANT CONTEXT      TenantScope(principal): the value every scoped accessor needs
  │                 cannot be built from a string taken from a request
  ▼
DATA ACCESS         storage/tenant_store.py: TenantStore(scope)  — inferences, feedback, signal logs
                    engine stores take scope: answer cache, sessions, conversation turns
                    app/tenancy.py: TenantRegistry — per-tenant clusters and trend
                    platform-wide reads: separate functions named *_all_tenants, admin-only, audited
```

Design rules.

1. **One authentication function.** The three existing implementations collapse into the one
   in `app/auth_guard.py`. No second system is created beside it.
2. **Policy is a FastAPI dependency, not a call inside the body.** A route's signature shows
   its policy. Tests can swap the dependency for fixed principals without a database.
3. **A scope cannot be forged from request data.** `TenantScope` is constructed only from a
   `Principal`. Route modules never read `tenant_id` from a body, query or header. A static
   test enforces both.
4. **Tenant filters live in one layer.** Route modules stop touching `storage.database._db`
   and collection objects. The scoped store adds the tenant to every filter and every
   document itself. This is the answer to "do not sprinkle `if tenant_id ==`": there are no
   such comparisons in routes.
5. **The scope is passed explicitly, one level down.** No hidden context variable carries
   the tenant. Reason: this server's route functions and dependencies run in a thread pool,
   and shadow-model calls run in a second pool; an implicit per-thread or per-task tenant is
   easy to lose and impossible to see in a signature. A function that needs tenant state
   takes a `scope` argument. A function that receives none gets no tenant state.
6. **Fail closed.** No scope → no cache read, no cache write, no session read. Tenant lookup
   failure → 401 or 503, never a default tenant.
7. **Global state is listed, and nothing else may be global.** §18 is the list.
8. **`fie/` is untouched.** The guard's own process-global state stays as shipped and is
   classified in §18.

# 17. Proposed authorization model

**Principal.**

| Field | Meaning |
| --- | --- |
| `tenant_id` | From the user document (API key) or the verified token. Non-empty; the value `"anonymous"` is rejected |
| `subject` | The user's e-mail. Used for audit, never for scoping |
| `role` | `tenant` or `platform_admin` |
| `credential_kind` | `api_key`, `session`, `env_key` |
| `admin_verified` | True only when the admin flag was read from the database, or the credential is the environment key, during this request |

**Policies.** Each route has exactly one.

| Policy | Requirement | Used by (matrix rows) |
| --- | --- | --- |
| `public` | None. Declared explicitly | 1–8, 9, 33, 49, 50 |
| `require_tenant` | A valid credential | 10, 11, 13–26, 29–32, 42, 43 |
| `require_platform_admin` | A valid credential, admin flag confirmed in the database now. Database unavailable → 503 | 12, 27, 28, 34–41, 44–48, 51 |

**Decisions inside routes.**

| Case | Result |
| --- | --- |
| No or invalid credential on a non-public route | 401, body `{"detail": "Authentication required"}`. Same text for missing, malformed, expired and unknown |
| Valid credential, not admin, admin route | 403 |
| Object id that belongs to another tenant | 404, identical to a missing id |
| Body tenant differs from the principal's | 403 |
| Admin, data route, no flag | Own tenant, like anyone else |
| Admin, read route, `all_tenants=true` | All tenants; `admin.cross_tenant_read` logged. A non-admin sending the flag gets 403 |
| Admin, write or delete route | Own tenant only. No flag changes this |

**Credentials.**

| Item | Change |
| --- | --- |
| Session token payload | `email`, `name`, `picture`, `tenant_id`, `is_admin`, `plan`, `exp`. **No `api_key`.** Tokens issued before the change still verify until they expire |
| Token trust | Identity and tenant are taken from a verified token for its 24-hour life. `is_admin` in a token is a hint for the dashboard only; admin routes ignore it and read the database |
| Signing secret | Unset or shorter than 32 characters: no token is created or accepted, and startup fails. An explicit development switch allows the old behaviour for local work and is logged at ERROR on every start |
| API key | Lookup unchanged. Environment-key comparison becomes constant-time |
| Key rotation | Repaired. Old key refused immediately. The response returns the new key only if it was stored |
| Login | Refused when Google returns no e-mail or `verified_email` is not true. An empty `ADMIN_EMAIL` grants admin to nobody |

Not changed, recorded for WP-017: keys in plain text at rest, token revocation before
expiry, token storage in the browser, the shape of `tenant_id`.

# 18. Proposed state-scoping model

Classification of every item in Artifact B.

| Class | Items | Rule |
| --- | --- | --- |
| **GLOBAL IMMUTABLE** | ST-36 signing secret and admin e-mail; ST-37 models, warm-up state, settings; ST-14/23 attack-pattern index **after** auto-growth stops | Set at start. No request writes it |
| **GLOBAL ADMIN CONTROLLED** | ST-06/25 thresholds and guard mode; ST-11/16/26 flagged events and learned hashes; ST-15 hard-positive files; ST-12/17 public feedback set; ST-13 anonymous telemetry | Written by a platform admin action, or by an anonymous public path that holds no tenant data. Each admin change is audited |
| **TENANT SCOPED** | ST-01 users; ST-02/19/40 inferences; ST-03 feedback; ST-04 signal logs; ST-05 answer cache; ST-07/20 session context; ST-08 conversation turns; ST-09/30 extraction tracking; ST-10 retraining buffer rows; ST-21 clusters; ST-22 trend; ST-24 shadow-response cache; ST-32 alert timer; ST-35 rate-limit bucket for authenticated traffic | Reachable only with a scope. Tenant on every row and in every key |
| **USER SCOPED** | ST-41 browser session | One user = one tenant today, so no server state is user-scoped |
| **REQUEST SCOPED** | ST-38 instrumentation collector; the canary token of one `/monitor` call | Not kept after the response |
| **MUST NOT EXIST** | The `"anonymous"` tenant bucket (ST-09/30, and inference rows written under it); ST-18 a model retrained from a request; ST-31 the unused canary store (left in place, never written) | No code path creates them after this package |

Reasoning for the ambiguous ones.

| Item | Why this class |
| --- | --- |
| Scan-verdict and translation caches (ST-27, 28) | Derived from text and platform state only. They cannot be written with chosen content and cannot return another tenant's data. They stay global. The remaining timing signal — a repeat of a scan made within five minutes is faster — is accepted and documented, because removing it means changing `fie/` and its measured latency |
| Learned hashes (ST-26) | They change verdicts for everyone, so they cannot be tenant-writable. A tenant-scoped version needs the tenant inside the scanner, which is `fie/`. Until the result contract carries a tenant (WP-003, WP-007), labelling is a platform admin action |
| Flagged events (ST-11) | Rows hold a prompt hash and an excerpt with no tenant, so they cannot be shown per tenant without showing other tenants' excerpts. Admin-only |
| Attack-pattern index (ST-14) | Its seed corpus is static project data. Learned entries are tenant prompts. Stopping growth makes it immutable; what to do with entries already learned is decision W2-6 |
| Thresholds (ST-06) | Per-tenant thresholds would be a new feature. They are shared platform configuration, so only a platform admin changes them, and never as a side effect of feedback |
| Shared Groq quota (ST-39) | A resource, not data. One tenant can exhaust it. Per-tenant metering is real work and is not required to stop data crossing; recorded |
| Clusters and trend (ST-21, 22) | They summarise a tenant's own traffic and are shown to that tenant. A platform-wide view is a different object that does not exist yet and is not added |

Per-tenant in-memory holders are bounded (least recently used tenants are dropped) so that
creating many tenants cannot grow memory without limit.

# 19. Proposed cache-key model

| Cache | Key today | Key proposed | Stored tenant field | Miss behaviour |
| --- | --- | --- | --- | --- |
| Answer cache, exact | `sha256(norm(question))[:32]` | `sha256("gtc2" ‖ tenant_id ‖ 0x00 ‖ norm(question))[:32]` | `tenant_id`, `schema: 2`, `source_class` (`tenant_feedback` or `system`) | Query filter `{_id, tenant_id}` |
| Answer cache, semantic | Best cosine over **all** documents | Best cosine over documents with this `tenant_id` and `schema: 2` | Same | No tenant documents → miss |
| Shadow-response cache | `sha256(model ‖ ":" ‖ prompt)[:24]` | `sha256(scope_ref ‖ 0x00 ‖ model ‖ 0x00 ‖ sha256(system_message) ‖ 0x00 ‖ prompt)` | — (in memory) | `scope` argument absent → no read and no write |
| Session context | `session_id` | Document filter `{tenant_id, session_id}`; in-memory key `(tenant_id, session_id)` | `tenant_id` | Document without `tenant_id` is not matched |
| Conversation turns | `conversation_id` | Filter `{tenant_id, conversation_id}` | `tenant_id` | Same |
| Scan-verdict cache | `sha256(norm(prompt [+domain] [+disabled layers]))` | **Unchanged** (§18) | — | — |

Notes.

- The tenant is inside the hash and also a stored, indexed field. The field does the
  isolation (every query filters on it); the hash prevents two tenants' identical questions
  from colliding on one `_id`.
- `scope_ref` for the in-memory cache is the tenant id; nothing is persisted.
- Including the system message in the shadow-response key means `/monitor`, which sends a
  new canary each time, stops getting hits. That is correct: a response generated under a
  different canary cannot answer the canary check. The cost is more Groq calls on repeated
  prompts; the playground and the helper calls without a system message still benefit
  within one tenant. Removing the per-request canary from the key would need a different
  canary design and is not attempted here.
- A separate rule protects a tenant's own correction: a `system` write never overwrites an
  existing `tenant_feedback` entry for the same key.
- Existing documents are left in place and never served. Nothing is migrated or deleted by
  the code.

# 20. Proposed feedback/recalibration model

```text
POST /feedback/{id}                         policy: require_tenant
  → TenantStore(scope).get_inference(id)    404 if not this tenant's (admins included)
  → correction → answer cache, scope = caller, source_class = tenant_feedback
  → TenantStore(scope).label_signal(id)     found by (tenant_id, request_id)
  → TenantStore(scope).save_feedback(...)
  → buffer row with tenant_id               counting only
  ✗ no maybe_recalibrate()
  ✗ no maybe_trigger_retrain()
```

| Element | Rule |
| --- | --- |
| Collection | Unchanged in what it records; every row now has a tenant |
| Automatic recalibration | Off. A switch `FIE_AUTO_RECALIBRATE` exists and defaults to off; turning it on restores today's behaviour and logs a warning at start. It is an escape hatch for the owner, not a recommended setting |
| Manual recalibration | `recalibrate()` remains a function. It updates fields with `$set`, so attack-threshold overrides are kept. It logs a `platform.recalibration` event with the old and new version. It still uses all tenants' labels; that is visible in the event |
| Retrain trigger | Off behind `FIE_AUTO_RETRAIN`, default off |
| Thresholds in use | Whatever the `fie_config` document holds at deployment. Whether to reset values that earlier feedback produced is the owner's decision (W2-4) |
| Effect of one tenant's feedback on another | None: the answer cache entry is in the caller's scope, the label is on the caller's row, and nothing global is recomputed |

# 21. Backward-compatibility analysis

| # | Change | Class | Who is affected | Migration |
| --- | --- | --- | --- | --- |
| BC-1 | `/monitor`, `/diagnose`, `/analyze`, `/analyze/v2`, `/track`, `/track-and-analyze`, `/trend`, `/clusters`, `/clusters/reset`, `/monitor/status`, `/monitor/model-info` need a credential | **BREAKING** | Anyone calling the hosted API without a key. SDK server mode without `api_key`: the client logs a 401 warning and the decorator returns the model's answer unmonitored, as it does today when the server is unreachable | Sign in, copy the key, pass `api_key=`. The dashboard already sends its credential. The public demo is unaffected |
| BC-2 | `/track` body `tenant_id` that differs from the caller → 403 | **BREAKING** for a client that sends another id | Unknown third parties only | Omit the field or send your own id |
| BC-3 | `/trend`, `/clusters` show the caller's tenant | **BREAKING** in content | Dashboard trend panel: now the tenant's own trend, which is what its label says | None needed |
| BC-4 | Answer cache starts empty per tenant; trace no longer names a person | **BREAKING** in content | Tenants relying on answers verified for others | None. Own corrections rebuild it |
| BC-5 | Admin: list/export/read default to own tenant; `DELETE /inferences*` scoped for admins | **BREAKING** for the admin | The owner | Add `all_tenants=true` to read across tenants. There is no global delete |
| BC-6 | `/auth/users` without `api_key` | **BREAKING** for the admin | The owner | Look in the database if a key is ever needed |
| BC-7 | Session token without `api_key` | INTERNAL | The dashboard reads the key from the login response and `/auth/me`, not from the token (CHECKED: `auth.js:17`) | Old tokens work until they expire |
| BC-8 | `/flags`, `/flags/{id}/label` work for a platform admin | ADDITIVE | They return 401 for everyone today | — |
| BC-9 | `/auth/regenerate-key` works | ADDITIVE | It fails today | — |
| BC-10 | Server refuses to start without a strong `JWT_SECRET_KEY` | **BREAKING** for a deployment without the secret | The Space, if the secret is missing there | Confirm the secret before deploying (§28) |
| BC-11 | Login refused without a verified e-mail | **BREAKING** for such accounts | None expected | — |
| BC-12 | Automatic recalibration and FAISS growth stop | **BREAKING** in behaviour, not in interface | Hallucination-monitor thresholds stop moving; the attack index stops growing | Switches exist for both |
| BC-13 | `multi_turn_escalation`, session context: scoped by tenant | INTERNAL | A conversation continues only within one tenant, which is the only correct case. State older than the deployment is not carried over: 2 h and 24 h lifetimes | — |
| BC-14 | `/health/deep` for anonymous callers: no Groq probe, no error text | **BREAKING** in content | The CI deploy job reads `components.detector.mode` and the overall status; both stay | The `groq` component reports `configured` instead of a live result |
| BC-15 | Playground refuses non-public or non-`https` custom endpoints | **BREAKING** for local test endpoints | Users pointing the hosted playground at `localhost` — which never worked from a hosted server | — |
| BC-16 | Request and response schemas | No field removed or renamed. `InferenceRequest.tenant_id` stays | — | — |
| BC-17 | Rate-limit key per tenant (if approved) | INTERNAL | — | — |
| BC-18 | Stored documents | ADDITIVE fields (`tenant_id`, `schema`, `source_class`). New inference documents use a tenant-namespaced `_id`; the `request_id` field and all lookups by it are unchanged | — | The unique index on `request_id` alone is replaced by one on (`tenant_id`, `request_id`): an owner-run step (§28) |

Versioning. BC-1 alone justifies a server minor release with a "breaking for anonymous API
use" note in the changelog. The SDK package is not changed and needs no release. No
deprecation period is proposed for anonymous access: the reason for the change is that the
open routes are the vulnerability. If the owner wants a period, the only safe form is a
date, not a flag that re-opens the routes (W2-1).

**Existing tests that must change.** `tests/test_integration.py` only.

| Test | Today | After |
| --- | --- | --- |
| `TestMonitor::test_monitor_requires_no_auth_returns_something` | Asserts the status is not 404/405 with no credential. Its name and comment assert the insecure behaviour | Renamed; asserts 401 without a credential |
| `TestMonitor` — the other six, `TestDiagnose` — both | Call without a credential; three expect 422 from validation | Same assertions, sent with a test principal through a dependency override in the fixture |
| `TestTelemetry`, `TestHealth`, `TestScanPrompt`, both integration classes, `TestAnalytics` | — | Unchanged |

No other existing test file is touched. The 87 keep passing with these nine edited.

# 22. Test strategy

**Location.** `tests/security/`, new. It never talks to a real database or network.

**Isolation of the test environment.**

| Guard | How |
| --- | --- |
| No real database | `conftest.py` clears `MONGODB_URI` before the application is imported and installs an in-repository fake collection (`tests/security/fakes.py`) that implements exactly the operations the server uses. No new dependency |
| Refusal to run against production | A session-scoped check aborts the suite if a MongoDB client is connected to anything, or if `MONGODB_URI` is set to a non-local host |
| No network | Groq, Google, Serper, Wikidata and SendGrid are replaced by fakes. A socket guard like the harness's fails any test that opens a connection |
| No real secrets | Every `.env` key is overridden with an empty or dummy value before import, as in WP-001 |
| Fixed principals | Tenant A, tenant B, platform admin, and "no credential", injected with FastAPI dependency overrides; a second set of tests exercises the real `authenticate()` against the fake user collection |

**Step 0 writes the attack tests first.** Each row of Artifact C becomes a test that
demonstrates the fault on the unmodified code. They are recorded in the execution report as
"reproduces" or "does not reproduce". A hypothesis that does not reproduce is reported and
its fix is dropped or re-scoped before any code changes. After the fixes the same tests
assert the target result.

**Matrix.**

| Group | Cases |
| --- | --- |
| Authentication | No credential; malformed header; unknown key; expired token; token signed with the development constant; token signed with a wrong secret; token without `tenant_id`; valid tenant A by key and by token; environment key; no signing secret configured |
| Route matrix | One generated case per route and method: the declared policy is enforced. A route present in the application and absent from the table fails the test |
| Tenant spoofing | Body tenant absent / own / B's / `"anonymous"` / empty; `X-Tenant-ID` header ignored; query parameter `tenant_id` ignored; admin with B's id → 403 |
| Inference isolation | A writes, B cannot list, read, export, delete; same `request_id` under A and B are two documents; A cannot overwrite B's; clear affects only the caller |
| Answer cache | A writes, B misses (exact and semantic); B writes, A misses; same question, two tenants, two answers; a `system` write does not replace a `tenant_feedback` entry; a legacy document without a tenant is never served; no identity in the trace; `from_cache` only for own entries |
| Shadow-response cache | Hit within a tenant; miss across tenants; no scope → no caching; different system message → miss |
| Feedback | A's 50 labels leave thresholds and the config document unchanged; B's `/monitor` threshold identical before and after; the label lands only on A's signal log; a manual recalibration keeps the attack overrides |
| Sessions | Same `session_id`, two tenants: independent histories; guessed id: empty; A's turns never appear in B's context; conversation escalation is per tenant; legacy documents without a tenant are not read |
| Shared state | `/clusters`, `/trend`, reset are per tenant; `/monitor` does not add to the attack index; learned prompt text is never returned; per-tenant holders are bounded |
| Admin | Non-admin → 403 on each admin route; admin flag cleared in the database → 403 with an old token; database down → 503 on admin routes; admin default view is own tenant; `all_tenants=true` works and is audited; admin delete is scoped; label routes work for an admin |
| Storage | Static test: no route module touches `storage.database._db`, a collection object, or a function from the platform-wide set without the admin policy; every scoped query contains the tenant; every scoped insert carries it |
| Concurrency | 32 threads, tenants A and B interleaved: no record, cache entry, session turn or cluster in the wrong scope; two simultaneous writes to one id; simultaneous cache write and read |
| Failure behaviour | User collection unavailable → 401 for keys, 503 for admin, never a default tenant; cache store raising → the request proceeds without cache; signal-log store raising → feedback still recorded for the right tenant; Google returning no e-mail → login refused; scope built without a tenant → error |
| Exposure | Same status and body for another tenant's id and a missing id on every `{id}` route; no e-mail, key or other tenant's text in any response in the suite (asserted by scanning response bodies for planted markers); errors carry no exception text |
| Regression of the guard | `pytest tests/evals`, then `python -m evals baseline` |

Planted markers: tenant A's test prompts, answers, e-mail and key contain unique strings.
A final test scans every response B and the anonymous caller received during the whole
suite and fails if any marker appears.

# 23. Acceptance criteria

| # | Criterion | Proof |
| --- | --- | --- |
| AC-1 | Unauthenticated `/track` cannot modify state | `test_track_requires_credential`; fake store unchanged |
| AC-2 | A request authenticated as A cannot modify B's state even with B in the body | 403 and store unchanged; overwrite test |
| AC-3 | A's cache entry is never returned to B | Answer-cache group |
| AC-4 | A's feedback cannot modify B's thresholds or configuration | Feedback group |
| AC-5 | A's sessions cannot be accessed by B | Sessions group |
| AC-6 | All tenant-sensitive storage access has an explicit scope | Static storage test |
| AC-7 | All tenant-sensitive cache keys contain the isolation context | Key tests for the answer and shadow-response caches |
| AC-8 | Unauthorized requests do not reveal cross-tenant object existence | Exposure group |
| AC-9 | No global mutable tenant state remains on a production path unless classified in §18 | `test_static_boundary.py` compares module-level mutable state in `app/`, `engine/`, `storage/` with an allowlist taken from §18 |
| AC-10 | Existing valid tenant traffic still works | Authenticated happy-path tests for every route in rows 10–32, 42, 43; dashboard call list replayed against the test client |
| AC-11 | Existing tests stay green except those asserting insecure behaviour | 87 pass; the nine edits in §21 are the only changes |
| AC-12 | The WP-001 harness still passes | `pytest tests/evals`: 230 pass |
| AC-13 | Attack and benign baseline metrics unchanged | `python -m evals baseline` exits 0 with "per-prompt records are byte-identical to BL-0001". `fie/` tree hash still `5d5ed90d` |
| AC-14 | The security suite reproduces every fixed vulnerability | Each Artifact C test fails on commit `c5b2291` plus the test files alone, and passes at the end. Recorded per test |
| AC-15 | *(new)* Every route has a declared policy | Route-matrix test, including the failure when a route is added without one |
| AC-16 | *(new)* No session token contains an API key; none is accepted or issued without a strong secret; startup fails without one | Token tests |
| AC-17 | *(new)* Admin rights come from the database at call time | Admin group |
| AC-18 | *(new)* Key rotation works and revokes the old key | Rotation test |
| AC-19 | *(new)* No response in the suite contains another tenant's planted marker | Marker scan |
| AC-20 | *(new)* Each denial in the suite produced the expected security event, and no event contains a credential or prompt text | Event assertions |
| AC-21 | *(new)* Added cost is within the budget in §24 | Timing test with fakes |
| AC-22 | *(new)* Scope: only the files in Artifact E changed; `fie/`, models, manifests, dependency files, CI, README and FACT_SHEET untouched | `git diff --stat` against `c5b2291` |

# 24. Observability requirements

Only what is needed to verify the boundary. No new logging system.

**Event record**, one JSON object per line through the existing structured logger, logger
name `fie.security`:

| Field | Content |
| --- | --- |
| `event` | One of the names below |
| `severity` | `info`, `warning`, `error` |
| `ts` | UTC ISO-8601 |
| `rid` | The request id already bound by the middleware |
| `route`, `method` | Path template, not the concrete URL |
| `tenant_ref` | First 12 hex of SHA-256 of the tenant id. Never the id itself, which contains part of an e-mail |
| `target_tenant_ref` | Same form, when a second tenant is involved |
| `credential_kind` | `api_key`, `session`, `env_key`, `none` |
| `outcome` | `denied`, `allowed`, `changed` |
| `reason` | A fixed code, for example `missing_credential`, `tenant_mismatch` |

| Event | Severity | When |
| --- | --- | --- |
| `authn.missing` | info | Non-public route without a credential |
| `authn.invalid` | warning | Bad key, bad or expired token, token under the development secret |
| `authz.tenant_mismatch` | warning | Body tenant differs from the principal's |
| `authz.cross_tenant_denied` | warning | An id resolved to nothing in the caller's scope while the route was called with an explicit foreign reference. Not emitted for ordinary 404s, to avoid making the log an oracle or a flood |
| `authz.admin_denied` | warning | Admin route, caller not an admin now |
| `admin.cross_tenant_read` | info | `all_tenants=true` used |
| `admin.config_change` | warning | Guard configuration changed; old and new values |
| `admin.flag_labelled` | info | A flagged event labelled; label and event id |
| `auth.key_rotated` | info | Key replaced |
| `platform.recalibration` | warning | Thresholds recomputed; versions |
| `cache.scope_missing` | error | A cache call arrived without a scope and was served uncached. Indicates a code path that lost the tenant |
| `startup.insecure_secret` | error | Development switch in use |

Never logged: API keys, tokens, authorization headers, prompt or answer text, corrections,
e-mail addresses. A test asserts this by scanning captured events for the planted markers.

Retention and shipping: whatever the platform's log retention is. No collection is added.
Correlation: `rid`, already returned to the caller in `X-Request-ID`.

**Performance budget.** Security first; the budget exists to notice accidents.

| Quantity | Budget | How measured |
| --- | --- | --- |
| Authentication plus scope construction | ≤ 1 ms p95 with the fake store | Timing test, 1,000 iterations |
| Cache-key construction | ≤ 0.1 ms p95 | Same |
| Extra database round trips per tenant request | 0 for API-key callers (the lookup exists today); 0 for token callers | Call counter on the fake |
| Extra round trips per admin request | 1 | Same |
| Query shape | Every scoped query has an equality on `tenant_id` and uses an index that starts with it | Asserted against the list of indexes the code requests |
| `/monitor` end to end | No budget set: dominated by shadow-model calls measured in seconds | Reported, not gated |

Indexes requested by the code, following the existing pattern of `create_index(...,
background=True)` at first use: `inferences (tenant_id, timestamp)`, `inferences (tenant_id,
request_id)` unique, `signal_logs (tenant_id, request_id)`, `ground_truth_cache (tenant_id)`,
`session_context (tenant_id, session_id)`, `conversation_turns (tenant_id, conversation_id,
timestamp)`, `users (api_key)`.

# 25. File-by-file implementation plan

## ARTIFACT E — File Change Matrix

**CREATE**

| File | Action | Reason | Risk | Tests |
| --- | --- | --- | --- | --- |
| `app/tenancy.py` | Create | `TenantScope` (built only from a `Principal`), `scope_key()`, `TenantScopeError`, bounded `TenantRegistry` holding one cluster registry and one trend tracker per tenant | Low. New code, no caller until routes switch | `test_shared_state.py`, `test_failure_modes.py` |
| `storage/tenant_store.py` | Create | The only data-access object routes use for `inferences`, `feedback`, `signal_logs`. Adds the tenant to every filter and document. Separate `*_all_tenants` functions for admin reads | Medium. Central; a mistake here is a mistake everywhere, which is also why it is one place | `test_inference_isolation.py`, static storage test, concurrency |
| `app/security_events.py` | Create | `emit()` for the events in §24 | Low | Event assertions |
| `tests/security/__init__.py`, `conftest.py`, `fakes.py` | Create | Isolation of the suite; fake collection; principals; network guard | Low | Self-tests of the fake |
| `tests/security/test_authn.py`, `test_route_matrix.py`, `test_tenant_spoofing.py`, `test_inference_isolation.py`, `test_answer_cache_isolation.py`, `test_feedback_isolation.py`, `test_session_isolation.py`, `test_shared_state.py`, `test_admin_boundary.py`, `test_exposure.py`, `test_concurrency.py`, `test_failure_modes.py`, `test_static_boundary.py`, `test_performance_budget.py` | Create | §22 | Low | — |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_002_server-isolation.md` | Create at implementation | Required report | — | — |

**MODIFY**

| File | Current role | Security problem | Proposed change (functions) | Why | Tests | Risk | Compatibility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `app/auth_guard.py` | Three helper functions | Optional auth; admin from token; called inside bodies | Add `Principal`; `authenticate()`; dependencies `require_tenant`, `require_platform_admin`, `public`. Admin dependency reads the user document. Remove `resolve_user`, `require_user`, `require_admin`, and the unused `can_access_tenant_record` once no route uses them | One authentication path | `test_authn.py`, `test_admin_boundary.py` | Medium: every route depends on it | INTERNAL |
| `app/auth.py` | Users, keys, tokens | S9, S10, N5, N6 | `create_session_token` without `api_key`; secret check in `create_session_token` and `verify_session_token`; `regenerate_api_key` fixed and returning success; `get_or_create_user` refuses an empty e-mail and never grants admin on an empty `ADMIN_EMAIL`; constant-time environment-key comparison; `get_all_users` drops `api_key`; index on `users.api_key` | Credential integrity | Token, rotation and login tests | Medium | BC-6, BC-7, BC-10, BC-11 |
| `app/auth_routes.py` | Login and account routes | Own header parsing; unverified e-mail; error text | Use the dependencies; check `verified_email`; generic error bodies | One path; N6 | Login tests | Low | BC-11 |
| `app/main.py` | Application, CORS, health | S10; N12; N18; DL-14 | Lifespan aborts without a strong secret unless the development switch is set; drop `X-Tenant-ID` from allowed headers; `health_deep` makes no outbound call and returns no error text for anonymous callers; validate `X-Request-ID` | I-11; exposure | Startup, health and header tests | Medium: can stop the server from starting | BC-10, BC-14 |
| `app/limiter.py` | Limiter | S12 | Key function: tenant when a principal is on the request, address otherwise | Availability between tenants | Key test | Low | BC-17. Only if W2-11 is approved |
| `app/routes/inference.py` | Track, analyze, inferences, diagnose | S1, S2, N1, N7 | Dependencies on all 12 routes; `/track` and `/track-and-analyze` tenant rule; all reads and writes through `TenantStore`; admin `all_tenants` on the three list routes and the single read; deletes always scoped; `/analyze/v2` and `/diagnose` use the caller's registry | I-1, I-2, I-5, I-15 | Inference, spoofing and admin groups | Medium | BC-1, BC-2, BC-5 |
| `app/routes/monitor.py` | Monitor, feedback, calibration | S2, S4, S5, S7, N8–N10, N13, N16 | Dependency; remove the `"anonymous"` fallbacks; pass `scope` to session store, turn tracker, shadow-model calls, answer pipeline, signal logger; use the tenant's registry and tracker; remove the FAISS growth call; feedback as in §20 | Most cross-tenant paths start here | Cache, feedback, session and shared-state groups | **High**: 1,040 lines, many branches. Changes are substitutions at call sites, no restructuring | BC-1, BC-4, BC-12, BC-13 |
| `app/routes/analytics.py` | Trend, clusters, analytics | S3, N3 | Dependencies; `/trend`, `/clusters`, reset use the caller's registry; admin routes use the admin dependency; no exception text in bodies; signal-log access through the store | I-1, I-5, I-13 | Shared-state and admin groups | Low | BC-1, BC-3 |
| `app/routes/admin.py` | Guard config, digest | S9; DL-5 | Admin dependency; audit event on change; digest through `TenantStore`; generic errors | I-12 | Admin group | Low | — |
| `app/routes/flags.py` | Review queue | Dead helper; S8 if repaired naively | Remove `_require_auth`; all five routes use the admin dependency; audit event on label; generic errors | Roadmap item; I-9 | `test_flag_label_requires_platform_admin` | Low | BC-8 |
| `app/routes/community.py` | Public feedback | Admin from token on export | Admin dependency on the export route; the two public routes declare `public` | Consistency | Route matrix | Low | — |
| `app/routes/playground.py` | Playground | N11; global registry | Dependency; URL check before `_call_custom` (scheme `https`, host resolves only to public addresses, no redirects followed); jury uses the caller's registry | I-17, I-5 | Playground test | Low | BC-15. Only if W2-12 is approved |
| `app/routes/_helpers.py` | Signal builder, collection accessor | Hands a raw collection to routes | Remove `get_signal_logs_collection`; callers use the store | AC-6 | Static test | Low | INTERNAL |
| `storage/database.py` | Inference and feedback storage | N1; unscoped helpers used by routes | `save_inference` takes the tenant from its caller, writes with a filter on `_id` and `tenant_id`, and uses a tenant-namespaced `_id` for new documents; unscoped functions renamed `*_all_tenants`; compound indexes; `initialize_vault` stops requesting the unique index on `request_id` alone (it would otherwise be re-created at every start) | I-5 | Inference group | Medium: storage format of new documents | BC-18 |
| `storage/signal_logger.py` | Signal logs | N13 | `log_signal(tenant_id=...)`; `find_log_by_request_id(tenant_id, request_id)`; `update_signal_feedback` filtered by tenant; index | I-5, I-10 | Feedback group | Low | ADDITIVE field |
| `engine/ground_truth_cache.py` | Answer cache | S4, N2, N19 | `lookup_cache(question, *, scope)` and `save_to_cache(..., *, scope, source_class)` as in §19; no scope → miss / no write; `verified_by` stores a class, not an e-mail | I-4 | Cache group | Medium | BC-4 |
| `engine/verifier/ground_truth_pipeline.py` | Answer pipeline | Passes no scope; trace names a person | `run_ground_truth_pipeline(..., scope=None)` forwards the scope to lookup and write-through; trace line without identity or use count | I-4, I-13 | Cache group | Low | BC-4 |
| `engine/groq_service.py` | Shadow models | N9 | `_call_single_model`, `fan_out`, `fan_out_with_confidence`, `complete` accept `cache_scope=None`; key as in §19; no scope → uncached | I-5, I-8 | Shadow-cache tests | Low. Default is "no cache", the safe side | INTERNAL |
| `engine/session_store.py` | Session context | N10 | `store_turn`, `get_context`, `clear_session` take `tenant_id`; filter and in-memory key include it; index | I-7 | Session group | Low | BC-13 |
| `engine/multi_turn_tracker.py` | Conversation turns | N10 | `check_multi_turn_escalation(tenant_id=...)`; document field and filter; index | I-7 | Session group | Low | BC-13 |
| `engine/fie_config.py` | Thresholds | S5, S6 | `maybe_recalibrate` returns at once unless `FIE_AUTO_RECALIBRATE` is on; `recalibrate` uses `$set` and emits an event | I-6, I-9 | Feedback group | Low | BC-12 |
| `engine/retraining/buffer.py` | Retrain buffer | Global trigger | `add_to_buffer(tenant_id=...)`; `maybe_trigger_retrain` returns at once unless `FIE_AUTO_RETRAIN` is on | I-6 | Feedback group | Low | — |
| `engine/agents/failure_agent.py` | Runs phases; writes the two global registries | S3 | `run_full` and `run_diagnostic` take the registry and tracker to use; with none given they write nothing | I-5 | Shared-state group | Medium: shared by four routes | INTERNAL |
| `engine/archetypes/registry.py` | FAISS index | S7 | `add_confirmed_detection` returns `False` at once unless `FIE_FAISS_AUTOGROW` is on | I-9 | `test_monitor_does_not_grow_attack_index` | Low | BC-12 |
| `engine/agents/adversarial/specialist.py` | Jury evidence | N4 | `nearest_prompt` filled only when the record's source is `seed`; otherwise the label and category alone | I-13 | `test_faiss_evidence_never_returns_learned_prompt` | Low | Evidence text for learned entries disappears |
| `tests/test_integration.py` | Existing tests | Asserts anonymous access | The nine edits in §21 | AC-11 | — | Low | — |
| `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Log | — | Updated at implementation, not in this session | Required | — | — | — |

**DELETE**

None. No file is removed. The module-level singletons `archetype_registry` and
`evolution_tracker` stay defined for scripts that import them; no route uses them afterwards,
and the static test enforces that.

**Not modified, on purpose:** everything under `fie/`, `models/`, `scripts/`, `data/`,
`evals/`, `tests/evals/`, `Frontend/`, `deploy/`, `.github/`, `pyproject.toml`,
`requirements.txt`, `README.md`, `SECURITY.md`, `docs/FACT_SHEET.md`. `SECURITY.md` line 74
becomes true once this package ships; whether to reword it is a later decision (§29).

Size estimate: about 450 lines of new production code, about 350 changed lines across the
modified files, and about 1,800 lines of tests.

# 26. Step-by-step implementation sequence

The order follows one rule: nothing is made stricter before the test that proves the old
behaviour exists, and identity is in place before any scope depends on it. The assistant
commits nothing; each "rollback point" is the working-tree state at the end of a step, which
the owner may commit.

| Step | Work | Files | Prerequisite | Invariants | Tests | Expected result | Rollback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Baseline and reproduction. Record pre-state (branch, HEAD, 87 + 230 tests, `evals baseline`). Build the test fakes and fixtures. Write every Artifact C test against today's code and record which reproduce | `tests/security/*` only | Approval of this plan | — | Suite self-tests; attack tests marked "expected to demonstrate the fault" | A table of 30+ attacks: reproduces / does not. **Stop and report if a hypothesis fails or a new path appears** | Delete `tests/security/` |
| 1 | Identity. `Principal`, `authenticate`, dependencies, security events, token content, secret check, key rotation, login checks. No route uses the new dependencies yet | `app/auth_guard.py`, `app/auth.py`, `app/auth_routes.py`, `app/security_events.py`, `app/main.py` (lifespan check only) | Step 0 | I-11, I-12, I-16 | `test_authn.py`, token, rotation, login, startup | Old helpers still work; new ones tested alone; 87 still pass | Revert five files |
| 2 | Route policies. Attach a dependency to every route per Artifact A; `/track` tenant rule; repair `/flags`; admin deletes scoped; the nine edits to the existing test file | `app/routes/*.py`, `tests/test_integration.py` | Step 1 | I-1, I-2, I-3, I-15 | `test_route_matrix.py`, `test_tenant_spoofing.py`, `test_admin_boundary.py` | No route is open by omission. Shared state is still shared: the next steps fix that | Revert routes and the test file |
| 3 | Scoped storage. `TenantStore`; inference id namespace; tenant on signal logs; routes stop touching collections | `storage/tenant_store.py`, `storage/database.py`, `storage/signal_logger.py`, `app/routes/_helpers.py`, route call sites | Step 2 | I-5, I-10 | `test_inference_isolation.py`, static storage test | Records cannot cross or be overwritten | Revert; new documents written meanwhile stay readable by the old code through `request_id` |
| 4 | Answer cache and shadow-response cache; trace without identity | `engine/ground_truth_cache.py`, `engine/verifier/ground_truth_pipeline.py`, `engine/groq_service.py`, `app/routes/monitor.py`, `app/routes/playground.py` | Step 3 | I-4, I-8, I-13 | Cache groups | No cross-tenant answer; cache starts empty | Revert. Entries written with a tenant would then be served globally by the old code: note in the rollback plan |
| 5 | Feedback, recalibration, retraining | `app/routes/monitor.py` (feedback route), `engine/fie_config.py`, `engine/retraining/buffer.py` | Step 4 | I-6, I-9 | `test_feedback_isolation.py` | Feedback is collected; nothing global moves | Revert, or set the two switches |
| 6 | Sessions and conversations | `engine/session_store.py`, `engine/multi_turn_tracker.py`, `app/routes/monitor.py` | Step 2 | I-7 | `test_session_isolation.py` | Per-tenant histories | Revert |
| 7 | Remaining shared state: per-tenant clusters and trend; FAISS growth off; evidence without learned text | `app/tenancy.py`, `engine/agents/failure_agent.py`, `engine/archetypes/registry.py`, `engine/agents/adversarial/specialist.py`, routes | Step 2 | I-5, I-9, I-13 | `test_shared_state.py`, static state allowlist | Artifact B rows match §18 | Revert, or set the growth switch |
| 8 | Exposure and admin hardening: `/health/deep`, error bodies, `/auth/users`, request id, CORS header, and — if approved — the playground URL check and the limiter key | `app/main.py`, `app/routes/*.py`, `app/limiter.py`, `app/routes/playground.py` | Step 2 | I-13, I-17 | `test_exposure.py`, playground, limiter | No response carries foreign or internal detail | Revert |
| 9 | Regression. Whole security suite; 87 existing; 230 harness; `python -m evals baseline`; concurrency; performance budget | — | Steps 1–8 | All | Everything | All green; records byte-identical to `BL-0001`. **Stop if the baseline differs** | — |
| 10 | Final verification. Each Artifact C test shown failing on the pre-state and passing now; marker scan; secret scan of new files; scope diff; execution report; master log; deployment checklist handed to the owner | Reports | Step 9 | — | — | Report complete. Nothing deployed by the assistant | — |

Steps 4, 5, 6 and 7 are independent of each other once step 3 is done and can be reviewed
separately. `app/routes/monitor.py` is edited in steps 2, 4, 5, 6 and 7; each edit is a
small set of call-site substitutions, listed in the execution report.

Stop conditions during implementation: a file outside Artifact E must change; a dependency
must be added; `fie/` must change; the harness baseline differs; a test needs a real
database or the network; a reproduction contradicts this plan; a fix requires changing an
approved decision.

# 27. Rollback plan

| Change | Failure behaviour (what happens when it goes wrong) | Rollback | Data compatibility | Cache implication |
| --- | --- | --- | --- | --- |
| Credential required on 11 routes | Legitimate anonymous callers get 401 | Redeploy the previous commit. There is deliberately no switch that re-opens the routes | None | None |
| Database-checked admin | Database down → admin routes return 503. Tenant routes keep working with API keys only if the user lookup works; with the database down, only the environment key and session tokens authenticate, as today | Redeploy | None | None |
| Startup secret check | Server does not start; the Space shows not ready; the CI deploy job fails after its wait | Set the secret, or set the development switch as an emergency measure, or redeploy | None | None |
| Token without API key | An old dashboard build that read the key from the token would lose it. CHECKED: the current build does not | Redeploy | Old and new tokens both verify | None |
| Inference id namespace | A fault in the new write path → 500 on `/track` or failed save on `/monitor` (logged; the monitor response is still returned, as today) | Redeploy. Documents written with the new `_id` remain readable by old code through `request_id` | Additive. The old unique index on `request_id` must be replaced before two tenants can hold the same id; until then such a write fails with a generic 409 and is logged | None |
| Tenant on signal logs, sessions, turns, buffer | Lookup failure → feedback recorded without a label on the log; session treated as new | Redeploy | Additive fields. Old rows without a tenant are not read by new code; old code reads new rows | Session and turn state from before deployment is ignored and expires in 24 h and 2 h |
| Tenant-scoped answer cache | Scope missing → uncached request, `cache.scope_missing` logged | Redeploy. **After a rollback, entries written with a tenant are served to everyone again by the old semantic search.** If that matters, the owner removes `schema: 2` entries before rolling back | Additive. Old entries stay, unserved | Effectively a flush for every tenant at deployment, without deleting anything |
| Shadow-response cache scope | Scope missing → no caching, more Groq calls | Redeploy | In memory only | Emptied by the restart anyway |
| Recalibration and retraining off | Thresholds stay fixed | Set `FIE_AUTO_RECALIBRATE` / `FIE_AUTO_RETRAIN`, or redeploy | None | None |
| Per-tenant clusters and trend | Memory bound reached → oldest tenant's in-memory summary dropped | Redeploy | In memory only | Global registry content is not carried over |
| FAISS growth off; evidence redaction | Index stops learning | Set `FIE_FAISS_AUTOGROW`, or redeploy | The index file is not rewritten | None |
| `/health/deep` reduction | A monitor that read the `groq` latency loses it | Redeploy | None | None |

Fail-open paths that this package must not create, each with a test: a missing tenant
treated as global; a store error that falls back to an unscoped query; an admin check that
falls back to the token when the database is down; a cache error that falls back to the
global cache; a secret check that only warns.

One fail-open path that exists and is left as it is, by scope: `/monitor` continues when the
pre-flight guard raises (audit §5.12). It is a guard-policy matter for WP-003/WP-007, not a
tenant boundary.

# 28. Deployment safety

During this plan and its implementation the assistant does not call the live API with
anything that writes, does not use real tenants, does not rotate credentials, does not
delete or flush anything, and does not deploy. All reproductions run against in-process
fakes.

Deployment is the owner's action. Because CI deploys on every push to `main`, **pushing is
deploying**. Checklist for the owner, to be completed in the execution report:

| # | Before pushing | Why |
| --- | --- | --- |
| 1 | The Space has a `JWT_SECRET_KEY` secret of at least 32 characters | Otherwise the new build does not start |
| 2 | The Space's `ADMIN_EMAIL` is set to the intended account | An empty value now grants admin to nobody; a wrong value locks the admin out |
| 3 | Decide whether the server-side `FIE_API_KEY` variable should exist on the Space | It is a database-independent admin credential |
| 4 | Decide W2-4: keep or reset thresholds that past recalibrations stored | The package stops further movement; it does not undo past movement |
| 5 | Decide W2-6: keep or remove learned entries in the FAISS files on the Space | Their text stops being returned either way |
| 6 | Replace the unique index on `inferences.request_id` with a unique index on (`tenant_id`, `request_id`). Check first that no pair is duplicated | Lets two tenants hold the same client-chosen id. Until done, such a collision returns 409 |
| 7 | Rotate the admin API key if it was ever shared (open item OQ-8 from the audit) | Rotation works again after this package |
| 8 | Tell known API users that anonymous access ends | BC-1 |

| # | After deploying | Check, read-only |
| --- | --- | --- |
| 9 | `/ready` is true; `/health/deep` reports the detector in `full_pipeline` mode | The CI job already does this |
| 10 | `POST /api/v1/monitor` with no credential returns 401 | One request, no data written |
| 11 | The dashboard signs in, lists the owner's own inferences, and shows a trend | Normal use |
| 12 | The log shows no `cache.scope_missing` and no `startup.insecure_secret` | — |

Legacy data left in place by the code: inference rows under tenant `"anonymous"`; answer
cache entries, signal logs, sessions and turns without a tenant; learned FAISS entries.
None is reachable by a tenant afterwards. Removing any of it is an owner-run step and is not
scripted in this package.

# 29. Security disclosure boundary

Until WP-002 is implemented, deployed and verified:

- The branch holding this plan, the audit and the attack tests is not pushed.
- No exploit detail goes into the README, `SECURITY.md`, the changelog, a public issue, a
  pull request description or a commit message on a public branch. Commit messages describe
  the fix ("require credentials on monitor routes"), not the attack.
- The test suite contains working reproductions. It is pushed together with the fixes, not
  before.
- This plan and the execution report are internal. They may be published later in a reduced
  form.

After remediation the owner decides: whether to publish an advisory; whether hosted users
were exposed and need to be told (the service stores other people's prompts, and the faults
allowed cross-tenant reads of some text); how to reword `SECURITY.md`; and whether the
earlier `SECURITY.md` claim about tenant isolation needs a correction note.

Dependency between packages: decision OD-3 of WP-001 holds the WP-001 branch back until
WP-002 closes the live-service findings. Both are released together.

# 30. Risks

| # | Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | A cross-tenant path was missed | Medium | High | Whole-repository search behind Artifact B; the static test that fails on new module-level mutable state or raw collection access; the marker scan across all responses |
| R2 | The auth change locks out the dashboard or the owner | Medium | High | Dependency-level tests with real token and key flows; dashboard call list replayed; deployment checklist items 1–3; environment key kept |
| R3 | `monitor.py` is long and edited in five steps; a branch keeps using global state | Medium | High | Substitutions only; the static test forbids the old imports in route modules; shared-state tests drive every branch that writes |
| R4 | The scope is dropped in a thread pool or a helper | Medium | Medium | Explicit arguments, no implicit context; "no scope → no tenant state" plus the `cache.scope_missing` event makes a drop visible and safe |
| R5 | The fake collection behaves unlike MongoDB | Medium | Medium | The fake implements only what the code uses and is itself tested; an optional owner-run pass against a local, throwaway MongoDB with a database-name guard |
| R6 | Stricter startup takes the Space down on deploy | Low if the checklist is followed | High | Checklist item 1; the check is tested; an emergency switch exists |
| R7 | The harness baseline changes | Very low | High | `fie/` is not touched; the canonical profile blocks `engine`, `app` and `storage` from loading; step 9 verifies |
| R8 | More Groq calls after the shadow-response cache stops hitting on `/monitor` | High | Low to medium | Stated in §19; quota is the operator's; a different canary design could restore hits later |
| R9 | Unknown third parties depend on anonymous routes | Unknown | Medium | Changelog note; no silent partial behaviour |
| R10 | The proxy question (S12) stays open | Certain | Low | Per-tenant key removes the cross-tenant effect for authenticated traffic without needing the answer |
| R11 | Residual: a stolen session token works for 24 h; keys are plain text in the database; the platform admin can read everything | Certain | Medium | Recorded for WP-017; cross-tenant admin reads are now explicit and logged |
| R12 | Residual: timing on the scan-verdict and translation caches | Certain | Low | Accepted and documented in §18 |
| R13 | A reproduction in step 0 contradicts the plan | Medium | Low | Stop condition; plan amended before code changes |
| R14 | Past harm is not undone: thresholds already moved, cache entries already stored, records possibly written by others | Certain | Unknown | Legacy cache entries stop being served; thresholds and records need the owner's review (checklist 4, 5) |

Findings noticed and deliberately **not** fixed here, because they are not tenant
boundaries: `load_from_db` never loads the scan and consistency thresholds (missing `global`
declarations, N15); the usage quota is non-atomic, allows calls on a database error and is
skipped on a guard block (N17); `/monitor` continues when the guard raises; the dashboard's
API-key sessions send `X-API-Key`, which CORS does not allow cross-origin (N18); the public
demo writes anonymous blocks into the review queue (N20); `demo_feedback._seen_hashes` grows
without bound; `scan_threshold` is a control with no effect (audit §5.5).

# 31. Open decisions requiring owner approval

Each has a recommendation. The plan above is written with the recommendations applied.

| # | Decision | Recommendation | Alternative |
| --- | --- | --- | --- |
| W2-1 | End anonymous access on rows 13–17, 24–26, 30–32 of Artifact A | Yes, all eleven, with no re-opening switch and no grace period | Keep `/monitor/status` and `/monitor/model-info` public; or announce a date first |
| W2-2 | `/track` body tenant that differs from the caller | 403 | Ignore silently; or 422 |
| W2-3 | Answer cache | Tenant-scoped for every entry; existing entries never served | Hybrid with a global set of system-verified answers |
| W2-4 | Automatic recalibration | Off; manual platform action only. Owner decides separately whether to reset stored thresholds to defaults | Leave on |
| W2-5 | `/flags` | Platform admin only. This differs from the roadmap's wording "tenant-scoped": the stored events carry no tenant and cannot be shown per tenant safely | Leave returning 401 until WP-003 |
| W2-6 | Attack-pattern index | Stop growth; keep serving existing learned entries for matching, never return their text | Also drop non-seed entries when the index loads |
| W2-7 | Admin on data routes | Own tenant by default; explicit, audited `all_tenants` for reads; no cross-tenant write or delete; no feedback on other tenants' records | Keep today's behaviour and add only the audit record |
| W2-8 | Environment admin key (`FIE_API_KEY` on the server) | Keep, with constant-time comparison | Require an explicit switch to enable it |
| W2-9 | Refuse to start without a strong signing secret | Yes, with a development switch | Refuse tokens but start anyway |
| W2-10 | Session token | Remove the API key; keep 24 h; no revocation yet | Add per-request database verification now |
| W2-11 | Rate-limit key per tenant for authenticated requests | Include (about ten lines) | Leave for WP-017, as the roadmap lists it |
| W2-12 | Playground custom endpoint | Include the URL check | Disable custom endpoints on the hosted server; or defer |
| W2-13 | Test database | In-repository fake, no new dependency | Add `mongomock` as a development dependency; or require a local MongoDB |
| W2-14 | Edit `tests/test_integration.py` as listed in §21 | Yes, those nine tests only | Leave them failing and add replacements elsewhere |
| W2-15 | `/health/deep` | Public, same shape, no outbound call or error text for anonymous callers | Admin only (the CI deploy job would need a credential, and CI is out of scope) |
| W2-16 | Inference document id namespaced by tenant, with the index change as an owner-run step | Yes | Keep the global id and reject collisions with 409 permanently |
| W2-17 | Legacy rows without a tenant | Leave in place, unreachable; owner purges later | Purge in this package (needs a script that writes to the live database) |
| W2-18 | Indexes | Requested by the code at first use, as the existing code does | Owner creates them by hand before deployment |
| W2-19 | Leave `fie/` untouched, accepting global learned hashes (admin-controlled) and the timing residual on two caches | Yes | Thread a tenant into the scanner now |
| W2-20 | Commits | The assistant commits nothing. The owner chooses whether to commit after each step or once | — |
| W2-21 | Dashboard | No change planned. Confirm there is no page that calls the API before sign-in other than the landing page's links | Add a change if one is found in step 2 |
| W2-22 | Scope of the extra findings | Fix N1–N13, N16 and N19 in this package, and remove the unused CORS header from N18; record N14, N15, N17, N20 and the rest of N18 for later | Restrict the package to S1–S12 only |
