# PLAN_004 — Runtime Policy and Coverage Handling (WP-004)

| | |
| --- | --- |
| Status | **PLAN, revision 3 — directionally approved; implementation NOT yet authorized. Nothing is implemented.** |
| Date | 2026-10-10 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `924c70d` (baseline pin), on top of `a9db959` (WP-003) |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053` |
| Depends on | WP-003 (`ScanResult.zone`, `.decided_by`, `.coverage`, `.models`) |
| Owner decisions | W4-1 to W4-16 and Q1 to Q7 decided on 2026-10-10 and recorded in §14.1 and §14.3. Revision 2 resolved clarifications A to E; revision 3 resolves the effective scanner-failure policy and the CLI under invalid configuration (§14.2) |

Labels used below. **EXISTING**: what the code does today, read from the file named.
**PROPOSED**: what this plan recommends. **APPROVED**: decided by the owner and recorded in §14. PROPOSED items that are not listed
in §14 are design detail for review; approval of this plan is not authorization to implement.

---

## 0. Starting-state verification

| Check | Result |
| --- | --- |
| Branch, HEAD | `rebuild/wp-001-eval-harness`, `924c70d` "evals: pin BL-0002 as canonical baseline after WP-003" |
| WP-003 implementation commit | Present: `a9db959` |
| Baseline-pinning commit | Present: `924c70d` |
| Working tree | Clean; nothing staged |
| `evals/baselines/CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |
| `BL-0002` contents | 11 record files, `fingerprint.json`, `summary.json`, `MANIFEST.sha256`, `REPORT.md`, `PIN.json` (kind "canonical baseline", source commit `a9db959`, tree not dirty, determinism proof verified, scope `all`, 0 gated files differing, same artifact digest `c6dd511d…`) |
| `python -m evals baseline` | **Exit 0.** Run `20261010T172841Z_70de7053_090da7e3`, `canonical=True`: standard counts match; per-prompt records byte-identical to `BL-0002` (11 files) |
| Determinism suite | Not rerun. The proof inside `PIN.json` is present and consistent |

---

## 1. Executive summary

**The situation.** After WP-003 a scan result says truthfully what ran. Nothing acts on it. A
base install scans without the classifier, reports `coverage.status = "partial"`, and the
prompt is allowed exactly as before. In `BL-0002`, the base-install profile allows **1,278 of
1,348 attacks (94.8%)**, every one of them correctly labelled as a partial-coverage allow, and
every one forwarded.

**The decisions that are made today are scattered and implicit.** Five different places turn
a result into an action, with four different failure behaviours (§2.3). Three of them discard
coverage entirely. Two fail open silently.

**The recommendation.** One small, pure module, `fie/policy.py`, with one function:

```text
decide(scan_result | None, policy, *, scan_failed=False) -> PolicyDecision
```

It reads only its arguments: public fields of the WP-003 result and an explicit, validated
`Policy` object. It reads no environment variable, global, file or network resource, never
calls a detector, never changes or annotates a `ScanResult`, and holds no state. Which policy
a process uses is resolved separately, by `active_policy()`, once per guarded call and before
the scan (§5.5). It returns an **action** (`allow`,
`review`, `block`, `error`), the **reasons** as fixed codes, and the **policy id and
version** that produced it.

The default policy, `compat@1`, reproduces today's actions exactly, in every branch. Two more
presets make the alternatives explicit: `strict@1` (missing required coverage and scanner
failure both block) and `review@1` (uncertainty and missing coverage are surfaced as
`review`). The policy is selected by the deployment, never by a request.

WP-004 wires the module into **one** enforcement point, `preflight_check`, and into the CLI
for display. **It is not universal enforcement.** Code that calls `scan_prompt()` directly,
which includes the OpenAI, Anthropic and FastAPI integrations, does not pass through the
policy, and the server routes do not read its outcome. Those gaps are listed in §8.6 as
prioritized follow-up and are deliberately not changed here (W4-9, W4-10).

**What does not change.** `scan_prompt`, `ScanResult`, every detector, threshold, weight and
model. The harness scans through `scan_prompt`, so all eleven record files must stay
byte-identical to `BL-0002` and no new baseline is needed.

**What this plan does not claim.** A policy cannot make a base install detect more. Under
`strict@1` a base install blocks or errors on everything, including all 490 benign prompts it
allows today. That is the honest trade, and it is why blocking on partial coverage is a
preset and not the default (W4-4).

---

## 2. Current-state findings

All from reading the files at `924c70d`.

### 2.1 The four things, as they exist today

| Concept | Where it lives | State |
| --- | --- | --- |
| **Detection** | `fie/adversarial.py` `scan_prompt` → `ScanResult.is_attack`, `.attack_type`, `.confidence`, `.zone`, `.decided_by` | Explicit since WP-003 |
| **Coverage** | `ScanResult.coverage`, `.degraded_layers`, `.models` | Explicit since WP-003 |
| **Policy** | Spread over `scan_prompt`'s router, `fie/preflight.py`, three integrations, two server routes | **Implicit** |
| **Operational failure** | `GuardResult.scan_failed`; elsewhere an exception caught and dropped | **Inconsistent** |

### 2.2 Policy that already lives inside the detector

EXISTING, `fie/adversarial.py` `scan_prompt`, the uncertain-band branch:

- No tiebreaker verdict → **block**, unless `FIE_UNCERTAIN_ALLOW` is `1`/`true`/`yes` → allow.
- `ScanResult.is_attack` is therefore already a *policy outcome* for 294 of the 2,016
  standard records (the `uncertain_block` zone), not a pure detector verdict. WP-003 made
  this visible as `zone = "uncertain_block"`, `decided_by = "fail_secure"` or `"config"`.

This is the one place where detection and policy are fused. WP-004 does **not** move it:
moving it changes what `is_attack` means and would change `BL-0002`. The policy layer reads
`zone` and `decided_by` and can treat `uncertain_block` differently from `clear_block` without
touching the router (§5).

Also EXISTING there: the `threshold=` argument of `scan_prompt` only matters when no layer
fired, where the result is an allow regardless. It is inert. W3-10 deferred the warning; it
stays deferred (W4-14).

### 2.3 Every place a result becomes an action

| # | Call site | Reads | Incomplete scan (classifier unavailable) | Scanner raises | Coverage reaches the caller? |
| --- | --- | --- | --- | --- | --- |
| P1 | `fie/preflight.py` `preflight_check` → `GuardResult` | `is_attack`, `attack_type`, `confidence`, `layers_fired`; copies `zone`, `coverage.status` | **Allowed.** `coverage_status = "partial"` is set on the result and nothing acts on it | `FIE_SCAN_FAILURE_MODE`: `open` (default) → allowed with `scan_failed=True`; `closed` → blocked as `SCAN_FAILED` | Yes, as `coverage_status` |
| P2 | `fie/monitor.py` (three wrappers) | `guard.blocked`, `guard.attack_type` | Allowed | Follows P1; `scan_failed` is not read | No |
| P3 | `fie/integrations/openai.py`, `anthropic.py` `_monitored_create` | `scan_prompt(...).is_attack` directly | Allowed | **Fail-open, logged at DEBUG**: `except Exception: logger.debug(...)`. `FIE_SCAN_FAILURE_MODE` does not apply | No |
| P4 | `fie/integrations/fastapi.py` `FIEMiddleware._scan` | `attack_type`, `confidence`, `is_attack` | Allowed | **Fail-open**, logged at WARNING. Server mode posts to `/api/v1/scan`, a route that does not exist in `app/`; every such call returns "not an attack" | No |
| P5 | `app/routes/monitor.py`, `app/routes/playground.py` | `GuardResult.blocked` and three fields | Allowed | `preflight_check` does not raise; if it did, both routes catch and **allow** ("allowing request through"). The shipped Dockerfiles set `FIE_SCAN_FAILURE_MODE=closed` | No: `coverage_status` and `zone` are dropped |
| P6 | `fie/__main__.py` `detect`, `explain` | whole `ScanResult` | Exit code 0 ("safe"), with the WP-003 `Coverage` line | Traceback | Yes |
| P7 | `fie/_lite.py` `scan_prompt_lite` | own verdict | Always partial by construction; allowed | Layer errors recorded | Yes, to a direct caller |

Other EXISTING switches:

- **Warn-only.** `PREFLIGHT_BLOCK_ENABLED` (SDK env) and `engine.fie_config`
  `block_enabled` (server, global, set by a platform admin through
  `POST /admin/guard/config`, persisted). With it off, P1 returns `blocked=False` and keeps
  `attack_type`. A fail-secure scan failure is still blocked.
- **Layer deadline** `FIE_LAYER_DEADLINE_S`: a slow layer becomes `timeout`, lands in
  `degraded_layers`, and the scan is allowed or blocked on what remains.

### 2.4 What the baseline says about these branches

From `BL-0002` records (no new scan was run for this plan).

| Profile | Expected | `allow` | `uncertain_block` | `clear_block` | Coverage |
| --- | --- | --- | --- | --- | --- |
| Canonical, 2,016 standard records | attack (1,516) | 186 | 190 | 1,140 | full |
| | benign (500) | 142 | 104 | 254 | full |
| Base-install simulation, 1,848 | attack (1,348) | **1,278** | 1 | 69 | partial, all |
| | benign (500) | 490 | 0 | 10 | partial, all |

Readings that constrain the design:

1. Partial coverage is not rare noise on a base install; it is every scan. A policy that
   blocks on it turns the base install into a refusal machine: 490 of 500 benign prompts
   allowed today would be blocked.
2. A policy that allows on it forwards 94.8% of attacks. There is no setting that is right
   for both a production gateway and a developer trying the SDK. This has to be a choice.
3. On the canonical profile `uncertain_block` holds 104 of the 358 blocked benign prompts
   (29%) and 190 of 1,330 blocked attacks (14%). Surfacing it as `review` relabels
   uncertainty; it does not reduce error (D-003 said the same).

### 2.5 Coverage facts the policy must respect

- `coverage.status = "partial"` covers very different situations: classifier missing; one
  regex layer raised; a layer timed out; the caller passed `disabled_layers`. They are not
  equally serious. Per the ablation recorded in the audit, the classifier carries 130 of 134
  JailbreakBench detections.
- `timeout` on the classifier also means "another thread was still loading the model"
  (WP-003 DV-4, NF-10). It is transient and clears after warm-up.
- `coverage.optional["translation"]` under-reports one failure case (NF-11). It is not
  reliable enough to drive an action.
- `coverage.status = "bypassed"`: a stored human label decided. No layer ran, by design.
- `coverage` is `None` on a `ScanResult` built by older code (WP-003 neutral form).
- The scan result cache returns the same object for the same prompt, for any caller. Policy
  must therefore be applied *after* the cache, to the result, and must not be stored in it.

---

## 3. Problem statement, threat model, failure scenarios

**Problem.** FIE reports coverage and then ignores it. An integrator cannot say "do not treat
a scan without the classifier as safe" without writing their own branch on
`coverage.status`, and five call sites each implement a different, undocumented answer.

**What the policy layer protects against**

| # | Scenario | Today | Notes |
| --- | --- | --- | --- |
| T1 | Deployment silently loses the classifier (bad image, missing extras, model download blocked) | Recall collapses; every result still allows; only a log line and `coverage` say so | The WP-001 finding, now visible but not acted on |
| T2 | Attacker makes the scanner raise (crafted input hitting a parser bug) | P1 default `open`: allowed. P3/P4: allowed, DEBUG/WARNING log | `FIE_SCAN_FAILURE_MODE=closed` exists but covers P1 only |
| T3 | Attacker makes one layer slow (long or pathological input) so it times out | Layer drops out; the rest decide | With a required layer, a policy can refuse; that creates T6 |
| T4 | Tiebreaker outage | Uncertain band blocked (fail-secure) | Already handled inside the detector |
| T5 | Cold start: requests arrive during the first model load | Scans run without the classifier, reported `timeout` | NF-10 |
| T6 | **Policy as a denial-of-service lever**: under a blocking policy, anything that degrades coverage blocks all traffic | n/a | The cost of T1/T3 protection. Must be a deliberate choice, with `review`/`error` as the middle ground |
| T7 | Caller influences policy through request data | n/a today (no policy object) | Must be impossible by construction |
| T8 | Consumer mistakes a policy action for a detector verdict, or the reverse | `is_attack` already mixes them in the uncertain band | Naming and typing, §5.4 |
| T9 | Invalid or mistyped policy configuration silently yields a permissive policy | n/a | Must fail loudly, §7 |

**Out of the threat model for this package:** detector evasion, multilingual and long-input
recall, the WP-002 residual risks (§8.4), prompt content in logs.

---

## 4. Candidate designs and tradeoffs

| | Design | For | Against |
| --- | --- | --- | --- |
| A | **Do nothing in code; document "check `coverage.status`"** | Zero risk | Leaves five inconsistent call sites; every integrator re-invents the branch |
| B | **A boolean switch**: `FIE_REQUIRE_FULL_COVERAGE=1` blocks any non-full scan | Tiny | Cannot express `review`; treats a failed copyright regex like a missing classifier; a third env switch next to `FIE_SCAN_FAILURE_MODE` and `FIE_UNCERTAIN_ALLOW` with undefined interaction |
| C | **Policy inside `scan_prompt`**: the router emits the action | One object for callers | Fuses policy into the detector again; changes the cached object; `BL-0002` records would have to change or be proven unchanged field by field; violates D-003 |
| D | **A pure policy module applied at the enforcement point** (recommended) | Detection untouched, baseline untouched by construction; one decision table, testable exhaustively; actions and verdicts are different types | A second object for callers to learn; integrations that call `scan_prompt` directly do not get it until they are migrated (W4-9) |
| E | **The roadmap's full policy engine** (YAML, per-category thresholds, per-tenant revisions, `Guard` class, `risk.score`) | The eventual shape | Far larger; needs `risk_score` (deferred, W3-9), a scan API (WP-010), tenant storage. Not "minimal" |

**D, sized so that E can grow out of it.** The `Policy` object in D is the seed of the
roadmap's policy document: same idea, three fields instead of a schema. Nothing in D has to
be undone to reach E.

Within D, three sub-choices:

| Sub-choice | Options | Recommendation |
| --- | --- | --- |
| Where the action is carried | (i) new fields on `ScanResult`; (ii) a separate `PolicyDecision`; (iii) only on `GuardResult` | **(ii)**, and `GuardResult` carries a copy of its fields. Keeping it off `ScanResult` is what stops a verdict being mistaken for an action, and keeps the cached object policy-free |
| How "partial" is judged | (i) any non-`ok` layer; (ii) a set of *required* layers | **(ii)**. Default required set: `pair_classifier` only (W4-5). Other degraded layers are reported as a reason and do not change the action under any shipped preset |
| What `review` does at the enforcement point | (i) blocks; (ii) allows and flags; (iii) configurable | **(iii)** `on_review`, default `block` (W4-6). A boolean `blocked` must still exist for every current consumer |

---

## 5. Policy contract

Architecture, action names, required layer, `review` binding and the separate `error` action
are APPROVED (W4-2, W4-3, W4-5, W4-6, W4-7). The field and function names below are APPROVED as written (Q1).

### 5.1 Objects

```text
fie/policy.py        new. Imports: standard library only, plus the closed sets ALL_LAYERS
                     and ZONES from fie.adversarial. No network or ML import. One lazy
                     read of fie.preflight._FAIL_SECURE, inside active_policy() only (§5.5).

ACTIONS              ("allow", "review", "block", "error")

Policy               frozen dataclass, validated on construction; a plain value
    name                 str    short token, ^[a-z][a-z0-9_-]{0,31}$ ; "invalid" is reserved
    version              int    >= 1
    required_layers      tuple  subset of ALL_LAYERS
    on_uncertain         "block" | "review"
    on_missing_required  "allow" | "review" | "block" | "error"
    on_scan_failure      "allow" | "block" | "error"
    on_review            "block" | "allow"
    .id                  "<name>@<version>"
    .digest              first 8 hex of SHA-256 over the canonical field values

PolicyDecision       frozen dataclass
    action               one of ACTIONS
    blocked              bool   enforcement bit (§5.2); before any warn-only override
    reasons              tuple of fixed codes (§5.3), deciding reason first
    policy               "<name>@<version>", or the literal "invalid" (§5.6)
    policy_digest        Policy.digest of the exact Policy object given to decide();
                         None when policy == "invalid"
    detector_is_attack   bool | None    copied from the result; None when there is no result
    zone                 str | None     copied
    coverage_status      str | None     copied
    missing_required     tuple of layer names, in pipeline order
    schema_version       1
    .to_dict()           exactly these keys, in this order

decide(result, policy, *, scan_failed=False) -> PolicyDecision          pure
invalid_config_decision(result=None) -> PolicyDecision                  pure; the fixed §5.6 decision
compat_policy(*, scan_failure) / STRICT / REVIEW                         the shipped presets
active_policy() -> Policy            the exact effective process policy (§5.5);
                                     may raise PolicyConfigError
set_policy(policy) / reset_policy()  process configuration (§5.5)
PolicyConfigError                    fixed message; never contains the offending value
```

These names, and the `GuardResult` fields `action`, `policy`, `reasons`, are APPROVED as
written (Q1).

**`decide` is pure (clarification A).**

- `policy` is a required positional argument and must be a `Policy` instance. There is no
  default and no `None`: `decide(result, None)` raises `TypeError`. It never calls
  `active_policy()`.
- It reads nothing but its three arguments: no environment variable, module global, file,
  clock, random source or network resource. Same arguments, same decision, in any process.
- `PolicyDecision.policy` and `.policy_digest` are computed from the `policy` argument
  itself. The decision therefore names exactly the policy that produced it, and nothing
  replaces or adjusts the policy after the decision is constructed.
- It does not mutate `result`, does not attach anything to it, and nothing in this package
  stores a `PolicyDecision` in the scan cache or on a `ScanResult`. The decision is a new
  object returned to the caller.
- It is total for valid arguments: every `ScanResult` (including a legacy one with
  `coverage=None`), a `LiteScanResult`, or `None` yields a decision. It raises only for a
  programming error (wrong argument type).

### 5.2 Outcome semantics

| Action | Means | Does **not** mean | `blocked` |
| --- | --- | --- | --- |
| `allow` | The policy permits forwarding the input | That the input is safe or the scan complete. `reasons` and `coverage_status` say what it rests on | False |
| `review` | The policy will not decide alone: uncertain evidence, or required coverage missing | A detector verdict | `on_review == "block"` → True (default, W4-6) |
| `block` | The policy forbids forwarding | That a detector fired: a block can come from missing coverage or scanner failure; `reasons` says which | True |
| `error` | The input could not be evaluated to the standard the policy requires, or no valid policy exists | "Allow", or a refusal of this particular input | True |

`error` stays separate from `block` (W4-7): both stop the input; `block` says "this input was
refused", `error` says "the guard is not healthy".

The states that must not be confused:

| State | Recognised by | Handled by |
| --- | --- | --- |
| Detector looked and scored low | layer state `ok`, zone `allow` | evidence toward `allow` |
| Detector unavailable | layer state `unavailable_*`, `error`, `timeout`, or `disabled`, for a required layer | `on_missing_required` |
| Bypassed | `coverage.status == "bypassed"` | the stored label decides; coverage rules do not apply |
| Scanner failure | `scan_failed=True` or `result is None` | `on_scan_failure`; reason `scan_failed` |
| Policy configuration failure | no valid `Policy` could be resolved | enforcement point, before `decide` (§5.6); reason `invalid_policy_config` |

### 5.3 Reason codes (closed set, fifteen)

| Code | Origin | Set when |
| --- | --- | --- |
| `detector_block` | detector | zone `clear_block` |
| `detector_uncertain` | detector | zone `uncertain_block`, `decided_by == "fail_secure"` |
| `tiebreaker_confirmed` | detector | zone `uncertain_block`, `decided_by == "tiebreaker"` |
| `labelled_attack` | detector | bypassed, `is_attack` |
| `labelled_safe` | detector | bypassed, not `is_attack` |
| `tiebreaker_cleared` | detector | zone `uncertain_allow`, `decided_by == "tiebreaker"` |
| `uncertain_allowed_by_config` | detector | zone `uncertain_allow`, `decided_by == "config"` |
| `detector_allow` | detector | zone `allow` |
| `missing_required_layer` | coverage | a required layer is not `ok` |
| `coverage_unknown` | coverage | `coverage is None` |
| `degraded_optional_layer` | coverage | a non-required layer is `unavailable_*`, `error` or `timeout` |
| `layer_disabled_by_caller` | coverage | a non-required layer is `disabled` |
| `scan_failed` | operational | no result exists |
| `invalid_policy_config` | operational | no valid policy exists (§5.6). Never produced by `decide` |
| `warn_only` | enforcement | appended by `preflight_check` when it waives a block (§6.1) |

Fixed strings only. `reasons[0]` is the reason that decided the action; later entries are
context. No layer error text, prompt text, path, environment value or exception text.

### 5.4 Keeping verdict and action apart (W4-3)

- **`ScanResult.is_attack`, `.zone`, `.decided_by` are the detector's.** `ScanResult` gains
  no field and its `to_dict()` gains no key. The one existing fusion, the uncertain-band
  fail-secure block (§2.2), stays where it is and is documented, not moved.
- **`PolicyDecision.action` is the policy's.** It is a different type on a different object.
  `PolicyDecision.detector_is_attack` and `.zone` sit next to it, so every serialized
  decision and every log line shows both, and neither can be read as the other.
- The word `allow` is both a zone and an action (the approved names, W4-3). They never appear
  in the same field: `zone` holds zones, `action` holds actions. Tests assert that
  `ScanResult` has no `action` and `PolicyDecision.zone` is only ever a zone or `None`.
- `GuardResult.blocked` remains the enforcement bit. `GuardResult.action` is the policy
  action. With warn-only they can differ (`action="block"`, `blocked=False`), and that
  difference is the point.

### 5.5 Process configuration and the effective policy

Resolution is separate from decision, and there is exactly one resolution per guarded call.

**The effective-policy rule (authoritative).**

```text
effective policy for one preflight_check call  =
    the Policy passed as policy=                       if the caller's code passed one
    otherwise  active_policy()                         called exactly once, before the scan

active_policy()  =
    the Policy given to set_policy()                   if one is set          -> returned as is
    otherwise, by the cached FIE_POLICY selection:
        "strict"  -> STRICT                            (strict@1)
        "review"  -> REVIEW                            (review@1)
        "compat" or unset ->                           (compat@1)
              compat_policy(scan_failure="block")      if fie.preflight._FAIL_SECURE is true
              compat_policy(scan_failure="allow")      otherwise
        anything else -> raises PolicyConfigError
```

The object obtained this way is the one passed to `decide`, and its `id` and `digest` are the
ones on the decision. Nothing is substituted, merged or adjusted afterwards.

**Scanner-failure behaviour of `compat@1` has one source: the existing flag.**

- `FIE_SCAN_FAILURE_MODE` is read in exactly one place, where it is read today: the
  module-level assignment of `fie.preflight._FAIL_SECURE`, executed when `fie.preflight` is
  first imported (in practice at `import fie`, which imports it). `fie.policy` never reads
  that environment variable.
- `active_policy()` reads the *flag*, not the variable, each time it is called with a
  `compat` selection, and returns one of two prebuilt immutable constants. It is a module
  attribute lookup; nothing is cached about it, so there is no second copy that could
  disagree with the flag.
- This is today's behaviour, including its timing: today `_safe_scan` consults the same
  module flag on every call. Changing the environment variable after import has no effect
  today and has none under this plan. Code that replaces the flag (as existing tests do) is
  honoured on the next call, as today.
- The two forms share the id `compat@1` (Q3). Their digests differ, so the decision records
  which one was in force: the id says which preset the operator selected, the digest says the
  exact field values used.
- For `strict@1`, `review@1`, a `set_policy` object and a per-call `policy=`, the flag is
  **not** consulted. Each of those states its own `on_scan_failure`. An explicit
  `compat_policy(scan_failure="allow")` passed by code is used as given, whatever the flag
  says: an explicit `Policy` is never altered.

**Precedence, consistent across every entry.**

| # | Source | Scope | Flag consulted? |
| --- | --- | --- | --- |
| 1 | `preflight_check(..., policy=p)`, keyword-only, trusted code only (Q4) | That call. `active_policy()` is not called | No |
| 2 | `set_policy(p)` | Process, until replaced or reset | No |
| 3 | `FIE_POLICY` = `strict` or `review` | Process, selection read once | No |
| 4 | `FIE_POLICY` = `compat`, or unset | Process, selection read once | Yes, at each `active_policy()` call |

**Lifecycle of the selection.**

- Importing `fie.policy` reads nothing and resolves nothing.
- The first `active_policy()` call reads `FIE_POLICY` once and caches the **selection**: one
  of the three names, or the failure. Accepted values are `compat`, `strict`, `review` after
  `strip().lower()`; unset or empty selects `compat`.
- **Changing `FIE_POLICY` in a running process has no effect**, in either direction. There is
  no hot reload.
- `set_policy(p)` accepts only a `Policy` instance (otherwise `TypeError`), takes precedence
  over the cached selection from the next call on, and masks a cached failure.
- `reset_policy()` discards both the `set_policy` object and the cached selection; the next
  `active_policy()` re-reads `FIE_POLICY`. It does not re-read `FIE_SCAN_FAILURE_MODE`: that
  flag belongs to `fie.preflight` and is not this module's to reset.
- **Recovery from an invalid `FIE_POLICY`:** restart with a valid value, or `set_policy(valid)`,
  or correct the variable in-process and call `reset_policy()`.

**Thread-safety.** The selection is resolved under one lock with a double-checked fast path:
concurrent first calls read the environment once and agree. `Policy` objects are immutable.
`set_policy` and `reset_policy` swap references under the same lock. Each `preflight_check`
call obtains its policy once, before scanning, and uses that one object to the end; a
`set_policy` or a flag change during the call affects the next call, never the current one.

**What is deliberately absent:** per-request, per-tenant or per-session selection (W4-11);
selection by any data that arrives in a request; a file format; a database; hot reload.

### 5.6 Invalid policy configuration (clarification B, W4-8)

`FIE_POLICY` set to anything other than the three names makes `active_policy()` raise
`PolicyConfigError`, on the first call and on every later call until recovery (§5.5). The
error message is the fixed text `"FIE_POLICY is not a known policy name"` and does not
contain the value.

**At the enforcement point**, `preflight_check` resolves the policy before scanning and, on
failure, returns without scanning:

| Field of the returned `GuardResult` | Value |
| --- | --- |
| `blocked` | `True` |
| `action` | `"error"` |
| `reasons` | `("invalid_policy_config",)` |
| `policy` | the literal `"invalid"` (Q2). No name or version is reported because no valid policy exists; `Policy.name` cannot be `"invalid"`, so the literal is unambiguous. The decision's `policy_digest` is `None` |
| `scan_failed` | `False`. The scanner did not fail; it was not run |
| `attack_type` | `""` . No detector spoke |
| `confidence` | `0.0` |
| `layers_fired` | `[]` |
| `zone`, `coverage_status` | `None` |
| `refusal_message` | the standard refusal text |

- **No scan is run.** There is no policy to evaluate its result under.
- **Distinguishable from a scanner failure** by three fields: `reasons[0]` is
  `invalid_policy_config` rather than `scan_failed`; `scan_failed` is `False` rather than
  `True`; `policy` is `"invalid"`. A fail-secure scanner failure keeps today's shape
  (`attack_type="SCAN_FAILED"`, `layers_fired=["preflight_fail_secure"]`, `scan_failed=True`).
- **Warn-only cannot waive it** (§6.1).
- **The empty-prompt rule comes first** (W4-12): an empty or whitespace prompt returns as
  today, unscanned and unblocked, with `action=None`, whatever the configuration.
- **A per-call `policy=` argument is unaffected**: it does not consult `active_policy()`, so
  an integrator who passes an explicit policy is not blocked by a bad environment value.
- **Logging.** One `ERROR` on logger `fie.policy`, once per process, when the failure is
  first cached: fixed text naming the variable and the three accepted names. Each affected
  call also produces the ordinary per-decision audit line (§7.1) with
  `reason=invalid_policy_config`. Neither contains the value.
- **Nothing serialized can contain the value**: `GuardResult.to_dict()`,
  `PolicyDecision.to_dict()`, the CLI output, the exception message and both log lines are
  built from fixed strings. A test sets `FIE_POLICY` to a string holding a path, a token and
  an e-mail address and searches all five.

**In the CLI, the scan still runs.** `fie detect` and `fie explain` are not enforcement
points: they print what the detector found, and their exit code is defined as the detector's
verdict (W4-15). The two requirements "no scan under invalid configuration" and "exit code
unchanged, no fabricated findings" cannot both hold for the CLI, because without a scan there
is no verdict to report or to exit on. The no-scan rule therefore applies to
`preflight_check`, where a decision is enforced, and the CLI behaves as follows:

1. The policy is resolved first. `PolicyConfigError` is caught.
2. One line is written to **stderr**, in every output mode including `--quiet`:
   `fie: policy configuration is invalid (FIE_POLICY is not a known policy name); no policy
   was applied`. Fixed text; the value is not printed. Nothing was written to stderr by these
   commands before, so this is the one addition outside stdout.
3. `scan_prompt` is called exactly once, as today.
4. JSON output: every existing key, in the existing order, with the real scan's values. The
   appended `policy` key holds `invalid_config_decision(result).to_dict()`:

   ```text
   action "error"   blocked true   reasons ["invalid_policy_config"]
   policy "invalid"   policy_digest null
   detector_is_attack / zone / coverage_status   copied from the real scan result
   missing_required []   schema_version 1
   ```
5. Text output: the existing lines, then `Policy     : error (invalid_policy_config)`.
6. Exit code: `1 if result.is_attack else 0`, exactly as today.

Nothing is fabricated: every detector field is from a scan that ran, and the policy block says
that no policy was applied. No preset is selected, so nothing permissive is chosen silently.
`ScanResult` and its serialization are untouched.

**The CLI exit code is not a policy gate, under any configuration.** That is W4-15, and it
already holds for valid policies: under `strict@1` with the classifier missing, a benign
prompt exits 0 while `policy.action` is `block`. Under an invalid configuration it likewise
exits 0 or 1 by the detector's verdict, with the stderr line and `policy.action = "error"`
as the signal. A script that needs the policy outcome must read `policy.action` from the
JSON. The plan states this limit rather than hiding it (§8.6).

| | `preflight_check` | CLI |
| --- | --- | --- |
| Role | Enforcement point | Diagnostic display |
| Policy resolved | Before the scan | Before the scan |
| Scan under invalid configuration | **Not run** | **Run once** |
| Detector fields | `None` / empty: no detector spoke | Real, from the scan |
| Policy outcome | `action="error"`, `blocked=True` | `policy.action="error"`, `policy.blocked=true` |
| What stops the input | `GuardResult.blocked` | Nothing: the CLI does not forward input. Exit code = detector verdict |

## 6. Decision table

### 6.1 Inputs and precedence

Row 0 is evaluated by the enforcement point, because `decide` requires a valid policy. Rows 1
to 9 are `decide`, top to bottom; the first matching row sets the action and `reasons[0]`.
Later rows can only append context reasons.

| # | Input | Source | Decides | `reasons[0]` |
| --- | --- | --- | --- | --- |
| 0 | No valid process policy | `active_policy()` raised | `error` | `invalid_policy_config` |
| 1 | `scan_failed` or `result is None` | caller of `decide` | `on_scan_failure` | `scan_failed` |
| 2 | `coverage.status == "bypassed"` | result | `block` if `is_attack` else `allow` | `labelled_attack` / `labelled_safe` |
| 3 | `zone == "clear_block"` | result | `block` | `detector_block` |
| 4 | `zone == "uncertain_block"`, `decided_by == "tiebreaker"` | result | `block` | `tiebreaker_confirmed` |
| 5 | `zone == "uncertain_block"`, any other `decided_by` | result | `on_uncertain` | `detector_uncertain` |
| 6 | Verdict is an allow and a required layer is not `ok` | `coverage.layers` | `on_missing_required` | `missing_required_layer` |
| 7 | Verdict is an allow and `coverage is None` | legacy-built result | `on_missing_required` | `coverage_unknown` |
| 8 | `zone == "uncertain_allow"` | result | `allow` | `tiebreaker_cleared` / `uncertain_allowed_by_config` |
| 9 | `zone == "allow"` | result | `allow` | `detector_allow` |

Context reasons appended after `reasons[0]`, in this order, when true:
`missing_required_layer` (rows 3 to 5, so a block reached with the classifier missing still
says so), `degraded_optional_layer`, `layer_disabled_by_caller`, and for rows 6 and 7 the
detector's own reason (`detector_allow`, `tiebreaker_cleared` or
`uncertain_allowed_by_config`).

**Invariants the table guarantees, each with a test (clarification E):**

1. **A detector block is never downgraded by missing coverage.** Rows 3 and 4 precede row 6
   and do not consult coverage. Row 5 can yield `review` only because the policy says so for
   uncertainty, never because coverage is missing.
2. **Missing required coverage never yields a bare `allow`.** Rows 6 and 7 always set
   `reasons[0]` to `missing_required_layer` or `coverage_unknown` and fill
   `missing_required`. Under `compat@1` the action is `allow`; it is never an `allow` whose
   first reason is `detector_allow`.
3. **A required layer that the caller disabled is missing.** `disabled` on a required layer
   matches row 6.
4. **A label outranks coverage; a scanner failure outranks everything `decide` sees.**
5. **Configuration failure is not scanner failure.** Row 0 and row 1 have different reasons,
   different `scan_failed`, different `policy`.

Not inputs, deliberately: `confidence` and layer scores (`risk_score` is deferred, W4-14);
`coverage.optional` (translation under-reports, NF-11; the meta-classifier and tiebreaker are
optional by W3-3); `models` (required digests are deferred, W4-13); anything from a request.

**Warn-only, specified exactly.** Warn-only (`PREFLIGHT_BLOCK_ENABLED=false`, or the server's
`block_enabled=false`) is an enforcement override applied by `preflight_check` *after* the
decision. It never changes `action` or `reasons[0]`.

```text
waivable  =  decision.blocked
             and decision.action in ("block", "review")
             and decision.reasons[0] in DETECTOR_BLOCK_REASONS

DETECTOR_BLOCK_REASONS = {detector_block, detector_uncertain, tiebreaker_confirmed, labelled_attack}

if warn-only is on and waivable:   GuardResult.blocked = False; reasons += ("warn_only",)
otherwise:                          GuardResult.blocked = decision.blocked
```

| Decision | `reasons[0]` | Warn-only on → `blocked` | Today |
| --- | --- | --- | --- |
| `block` | `detector_block`, `tiebreaker_confirmed`, `labelled_attack` | False, `warn_only` appended | same: logged, allowed |
| `block` (`compat`, `strict`) or `review` (`review@1`) | `detector_uncertain` | False, `warn_only` appended | same: an uncertain-band block is waived today |
| `block` or `review` | `missing_required_layer`, `coverage_unknown` | **True. Not waived** | n/a (new branch) |
| `block` or `error` | `scan_failed` | **True. Not waived** | same: a fail-secure scan failure is not waived today |
| `error` | `invalid_policy_config` | **True. Not waived** | n/a (new branch) |
| `allow` | any | False; nothing to waive; `warn_only` not appended | same |

Warn-only exists to observe *detections* without enforcing them. It cannot be used to switch
off a coverage requirement or to hide an unhealthy guard. Under `compat@1` the first, second,
fourth and last rows are the only reachable ones, and each equals today's behaviour.

### 6.2 Presets

| | `compat@1` (default) | `strict@1` | `review@1` |
| --- | --- | --- | --- |
| `required_layers` | `pair_classifier` | `pair_classifier` | `pair_classifier` |
| `on_uncertain` | `block` | `block` | `review` |
| `on_missing_required` | `allow` | `block` | `review` |
| `on_scan_failure` | follows `fie.preflight._FAIL_SECURE` (set from `FIE_SCAN_FAILURE_MODE` at import): false → `allow`, true → `block` (§5.5) | `block` | `error` |
| `on_review` | `block` | `block` | `block` |

`compat@1` is defined as "exactly today" and is the default (W4-4). The stricter presets
take effect only when an operator selects them.

`Policy` objects are plain values, so `compat@1` exists in two forms that differ only in
`on_scan_failure`: `compat_policy(scan_failure="allow")` and
`compat_policy(scan_failure="block")`. Both report the id `compat@1`; `policy_digest` tells
them apart (Q3). Which one a call uses is fixed by the single rule in §5.5: `active_policy()`
returns the form matching `fie.preflight._FAIL_SECURE` at the moment it is called, once per
guarded call, before the scan. `preflight_check` performs no selection of its own and never
swaps the policy it obtained. `strict@1` and `review@1` do not depend on the flag.

### 6.3 Coverage × verdict × action

| Case | `compat@1` | `strict@1` | `review@1` |
| --- | --- | --- | --- |
| Full coverage, allow | allow | allow | allow |
| Full coverage, clear block | block | block | block |
| Full coverage, uncertain (fail-secure) | block | block | **review** |
| Partial: classifier unavailable (dependency / model / load failed), allow | allow + `missing_required_layer` | **block** | **review** |
| Partial: classifier unavailable, block | block | block | block |
| Partial: classifier `timeout` (deadline or cold-start race), allow | allow + reason | **block** | **review** |
| Partial: classifier `error`, allow | allow + reason | **block** | **review** |
| Partial: a non-required layer degraded, allow | allow + `degraded_optional_layer` | allow + reason | allow + reason |
| Caller disabled a non-required layer | allow + `layer_disabled_by_caller` | same | same |
| Caller disabled the classifier, allow | allow + reason | **block** | **review** |
| Labelled false positive (bypassed) | allow + `labelled_safe` | same | same |
| Labelled attack (bypassed) | block + `labelled_attack` | same | same |
| Tiebreaker unavailable, uncertain band | block (detector fail-secure) | block | review |
| Tiebreaker cleared / `FIE_UNCERTAIN_ALLOW` | allow + reason | allow + reason | allow + reason |
| Scanner raised | allow (`open`) or block (`closed`), `scan_failed` | block | **error** |
| Invalid `FIE_POLICY` (no preset is in force), `preflight_check` | `error`, `invalid_policy_config`, blocked, no scan — in every case | | |
| Invalid `FIE_POLICY`, CLI | scan runs; `policy.action = "error"`, `invalid_policy_config`; stderr notice; exit code by detector verdict — in every case | | |
| Empty or whitespace prompt | not scanned, not blocked, `action=None` — in every case, as today (W4-12) | | |
| `scan_prompt_lite` result | n/a: `decide` accepts it; classifier is `disabled` | block on allow | review on allow |
| Legacy result, `coverage=None`, allow | allow + `coverage_unknown` | block | review |

### 6.4 What each preset would do to the baseline (replayed from `BL-0002`, no scan)

| | Canonical profile, 2,016 | Base-install profile, 1,848 |
| --- | --- | --- |
| `compat@1` | allow 328, block 1,688 — today | allow 1,768, block 80 — today |
| `strict@1` | identical to `compat@1` | allow 0, **block 1,848** (benign: 500 of 500) |
| `review@1` | allow 328, review 294, block 1,394 | allow 0, review 1,769, block 79 |

These numbers are the acceptance targets for the replay test (§9). They also state the cost
of each preset before anyone chooses one.

**High-risk versus low-risk deployments** are expressed by the preset, chosen by the
operator: a production gateway picks `strict@1`; an integrator with a human queue picks
`review@1`; a developer evaluating the SDK gets `compat@1` and a visible reason on every
partial allow. The per-request `domain` argument continues to adjust *detector* thresholds
only; it does not select a policy (a request must not be able to).

---

## 7. Configuration and compatibility

### 7.1 Configuration

| Mechanism | Scope | Status |
| --- | --- | --- |
| `FIE_POLICY` = `compat` \| `strict` \| `review` | Process, read once (§5.5) | APPROVED in principle; unset → `compat@1` (W4-4) |
| `fie.policy.set_policy(Policy)` / `reset_policy()` | Process | APPROVED (Q1) |
| `preflight_check(..., policy=Policy)` keyword-only, for trusted integrator code | One call | APPROVED (Q4) |
| `FIE_SCAN_FAILURE_MODE` | As today: read once, when `fie.preflight` is imported, into `_FAIL_SECURE` | Unchanged and still the only reader. The flag selects the form of `compat@1` (§5.5). Ignored by `strict@1`, `review@1` and any explicit `Policy` |
| `FIE_UNCERTAIN_ALLOW`, `PREFLIGHT_BLOCK_ENABLED`, server `block_enabled` | As today | Unchanged. The first stays inside the detector; the other two stay the warn-only override (§6.1) |

No YAML, file format, database, hot reload or per-category thresholds.

**Invalid configuration fails closed.** A `Policy` cannot be constructed with an unknown value,
an unknown layer name, a malformed or reserved name, or a version below 1 (`ValueError`). An
unknown `FIE_POLICY` never selects a preset: it yields a blocking `error` with
`invalid_policy_config` (§5.6, W4-8). There is no fallback, permissive or otherwise.

**Scope of configuration.** Per process, set by whoever deploys it. Nothing is per request.
Per-tenant policy is deferred (W4-11); `decide` taking the policy as an argument, and the
decision never entering the shared scan cache, are what keep that door open.

**Versioning and audit.**

- Presets are `name@version`. Changing a preset's behaviour means a new version; the old one
  stays constructible. A test pins every shipped preset's field values and digest.
- Every `PolicyDecision` carries the policy id and digest. `GuardResult` carries the id.
- `preflight_check` writes one line on logger `fie.policy` for every decision whose action is
  not `allow`, and for every `allow` whose first reason is not `detector_allow`: `action`,
  `blocked`, `policy`, `policy_digest`, `reasons`, `zone`, `coverage_status`,
  `missing_required`. Fixed codes and layer names only; no prompt text, no attack type text
  beyond what the existing lines already log.
- The existing `PREFLIGHT_BLOCK` and `PREFLIGHT_WARN` lines keep their format and conditions.

### 7.2 Compatibility, in five separate statements (clarification C)

**C1. Enforcement behaviour under `compat@1` is today's.** For every input and every branch
that exists today, `preflight_check` returns the same enforcement outcome:

| Branch | `blocked` today and under `compat@1` |
| --- | --- |
| Empty or whitespace prompt | False (not scanned) |
| Detector allow, any coverage | False |
| Detector block (clear, uncertain, tiebreaker-confirmed, labelled), block mode | True |
| The same, warn-only | False |
| Scanner raised, `FIE_SCAN_FAILURE_MODE` open | False, `scan_failed=True` |
| Scanner raised, closed (with or without warn-only) | True, `scan_failed=True` |

No input that is forwarded today is stopped under `compat@1`, and none that is stopped is
forwarded. The scanner-failure rows follow `fie.preflight._FAIL_SECURE` with today's timing:
set from `FIE_SCAN_FAILURE_MODE` when the module is imported, consulted on each call, and
unaffected by later changes to the environment variable (§5.5).

**C2. Existing `GuardResult` fields keep their values.** `blocked`, `attack_type`,
`confidence`, `layers_fired`, `refusal_message`, `scan_failed`, `zone`, `coverage_status`,
`schema_version` have, under `compat@1`, exactly the values they have today for the same
input, including the quirks: `attack_type` kept in warn-only mode, `""` on an allow,
`"SCAN_FAILED"` / `1.0` / `["preflight_fail_secure"]` on a fail-secure scanner failure,
`schema_version == 2`. The dataclass keeps its field order; positional construction with the
existing fields still works.

**C3. `GuardResult` gains three fields, appended, defaulted (approved addition).**

| New field | Type | Default | Set to |
| --- | --- | --- | --- |
| `action` | `str \| None` | `None` | the policy action; `None` when nothing was decided (empty prompt, or a `GuardResult` built by other code) |
| `policy` | `str \| None` | `None` | `"<name>@<version>"`, or `"invalid"` |
| `reasons` | `tuple[str, ...]` | `()` | the decision's reasons, plus `warn_only` when applied |

`GuardResult.to_dict()` emits its nine existing keys first, in their existing order
(`blocked`, `attack_type`, `confidence`, `layers_fired`, `refusal_message`, `scan_failed`,
`zone`, `coverage_status`, `schema_version`), then exactly `action`, `policy`, `reasons`.
`schema_version` stays 2: the change is additive. The WP-003 test that pins the nine-key
tuple is the one existing assertion this package must update, to the twelve-key tuple; that
edit to `tests/test_scan_result_contract.py` is **authorized (Q5)**: that one assertion
changes from the nine-key to the twelve-key tuple, and no other assertion or test file may
change under this exception. It is listed in §11.

**C4. CLI JSON gains one key, appended (approved addition).** `fie detect --output json` and
`fie explain --output json` emit every key they emit today, in the same order with the same
values, then one more: `policy`, holding `PolicyDecision.to_dict()`. Exit codes are unchanged
and follow `is_attack` (W4-15), under every policy configuration including an invalid one;
the CLI exit code is a detector verdict and never a policy gate. The text output gains one
`Policy` line, printed only when the action is not the one `is_attack` alone implies, or a
non-detector reason is present. Under an invalid `FIE_POLICY` the CLI still scans, appends a
`policy` block with `action="error"`, and writes one fixed notice to stderr (§5.6); stdout
keeps every existing key and value.

**C5. `ScanResult` and the detector are unchanged.** No field is added to `ScanResult`,
`LiteScanResult`, `ScanCoverage` or `ModelIdentity`. `ScanResult.to_dict()` has the same
fifteen keys in the same order. `scan_prompt`, `scan_prompt_async`, `scan_prompt_lite`, every
layer, threshold, weight and model are untouched; `fie/adversarial.py` is not edited. The
object returned by `scan_prompt` is field-for-field the same whatever policy is configured.

**Also unchanged:** `preflight_check`'s three existing parameters and their order;
`_safe_scan`'s five return values; `GuardedResponse`; `make_guarded_response`;
`fie/monitor.py`; `fie/integrations/`; everything in `app/`. `fie.__init__` additionally
exports `Policy`, `PolicyDecision`, `decide`.

A user who installs the new version and sets nothing sees identical behaviour and three extra
keys in `GuardResult.to_dict()` and one in the CLI JSON.

## 8. Security analysis

### 8.1 Fail-open and fail-closed, stated per branch

| Branch | `compat@1` | Why that default | Safer setting |
| --- | --- | --- | --- |
| Scanner raises | open unless `FIE_SCAN_FAILURE_MODE=closed` | Existing default; changing it is a behaviour change for every SDK user | `strict`, `review` |
| Required layer missing | open, with a reason code | Blocking would refuse all traffic on every base install (§2.4) | `strict`, `review` |
| Uncertain band, no tiebreaker | closed | Existing detector behaviour | — |
| Invalid `FIE_POLICY` | closed: `error`, `invalid_policy_config`, under every preset | A typo must not select a permissive policy (W4-8) | — |

The default is not the secure setting, and this plan says so. Whether the *shipped server
images* should set `FIE_POLICY=strict`, as they already set `FIE_SCAN_FAILURE_MODE=closed`,
is a deployment-configuration change outside this package (W4-10) and is follow-up F4.

### 8.2 Denial of service

Under `strict@1`, anything that removes the classifier from a scan blocks the request.
Sources: a bad deploy (T1), the layer deadline on a pathological input (T3), the cold-start
race (T5). Mitigations available without detector work: `warmup()` before serving (exists);
`review@1`/`error` so the operator sees an outage as an outage, not as mass refusals; the
once-per-process coverage warning (exists). A single crafted input can block only itself: the
layer pool and deadline are per scan. Not solved here: a sustained flood of slow inputs
occupying the shared 16-worker pool. Recorded as a risk (R4).

### 8.3 Bypass

- **Request-controlled policy.** `decide` takes an explicit `Policy` object and no request
  data. The process policy comes from the environment or `set_policy`; a per-call policy is
  an object the integrator's own code constructs. The server routes pass nothing and are not
  modified. A static test confirms no file in `app/` passes `policy=` or calls `set_policy`.
- **Caller-disabled layers.** `disabled_layers` is an argument of `scan_prompt`. If
  integrator code forwards it from a request, a caller can switch the classifier off. Under
  `compat@1` that is allowed today and stays allowed, with a reason; under `strict@1` /
  `review@1` a disabled *required* layer is treated as missing. `preflight_check` does not
  expose `disabled_layers` at all.
- **Label fast path.** A whitelisted hash skips every layer and every coverage rule. That is
  existing behaviour and is a trust statement about whoever can write labels. Since WP-002
  labels are tenant-scoped on the server, but the SDK's in-memory whitelist is process-wide.
  Not changed here; the decision carries `labelled_safe` so it is auditable. Listed as
  follow-up F2.
- **Warn-only.** A global admin switch that turns blocks into logs. Unchanged; now visible on
  every decision as `warn_only`, and unable to waive a missing-coverage or scan-failure
  block.
- **Cache.** The decision is never cached and never written to the shared result object, so
  one caller's policy cannot leak into another's result.
- **Direct `scan_prompt` callers** bypass the policy entirely. That is a known, documented
  limit of this package (§8.6).

### 8.4 WP-002 residual risks reviewed, none fixed here

| Residual risk (EXECUTION_002 §10.8) | Bearing on WP-004 |
| --- | --- |
| The scan-verdict and translation caches in `fie/` are global; timing shows that someone scanned identical text | No new exposure: no policy data enters the cache |
| The platform admin can change the global guard switch | Warn-only stays global. A per-tenant policy would need this revisited (W4-11) |
| Learned entries in the attack-pattern index still take part in matching | Detector-side; untouched |
| Stolen session token valid 24 h; API keys in plain text (WP-017) | Unrelated to policy, but an attacker with an admin token can set warn-only. **Follow-up F1, to prioritise before any deployment that relies on policy** |
| Suite runs against a fake MongoDB | No server change here |

**Security follow-ups to schedule, not part of WP-004:**
F1 token revocation and key hashing (WP-017) before relying on admin-controlled policy;
F2 process-wide SDK label store; F3 the P3/P4 integrations failing open at DEBUG and the
non-existent `/api/v1/scan` route; F4 the P5 server routes dropping `coverage_status` and
allowing when the guard raises; F5 NF-10 cold-start race; F6 NF-13 CI tiebreaker call.

### 8.5 Information exposure

`PolicyDecision` holds fixed codes, layer names from a closed set, the policy id and an
8-hex digest. A custom `Policy.name` must match `^[a-z][a-z0-9_-]{0,31}$`, so it cannot carry
a path, an address or free text. Tests plant a path, a token and an e-mail address in three
places — the scanner's exception text, the `FIE_POLICY` value, and an attempted policy name
(rejected) — and search every serialized output and log line, as WP-003 did.

### 8.6 What WP-004 does not enforce (clarification D)

**This package is not universal enforcement, and must not be described as such** in the
execution report, the changelog, the README or anywhere else. Policy is applied in exactly
one place, `preflight_check`, and displayed by the CLI.

`scan_prompt()` returns a detector result. Code that calls it directly and branches on
`is_attack` gets no policy decision, no coverage rule and no configured failure handling,
whatever `FIE_POLICY` says. Known cases, all unchanged by this package (W4-9, W4-10):

| Priority | Gap | Today | Consequence under any preset |
| --- | --- | --- | --- |
| F3a | `fie/integrations/openai.py`, `anthropic.py` call `scan_prompt` directly | Any scanner exception is caught and logged at DEBUG; the request proceeds | No policy. A scan without the classifier is forwarded even under `strict@1`. A scanner crash is invisible at default log levels |
| F3b | `fie/integrations/fastapi.py` `FIEMiddleware`, local mode | Scanner exception → WARNING, request proceeds | Same |
| F3c | `FIEMiddleware`, server mode, posts to `/api/v1/scan` | No such route exists; every call returns "not an attack" | The middleware protects nothing in server mode |
| F4a | `app/routes/monitor.py`, `playground.py` read `GuardResult.blocked` only | `action`, `reasons`, `coverage_status` are dropped | The server enforces `blocked` (so a selected preset does take effect there through `preflight_check`), but cannot tell a refusal from an unhealthy guard, and returns neither to the caller |
| F4b | The same routes catch an exception from the guard and allow the request | Fail-open on an unexpected guard error | Unchanged |
| F4c | Shipped images set `FIE_SCAN_FAILURE_MODE=closed` and no `FIE_POLICY` | — | The hosted service keeps `compat@1` until a deployment change is separately approved |
| F4d | The server does not refuse to start on an invalid `FIE_POLICY` | — | It starts and answers every guarded request with a blocking `error` |
| — | `fie/monitor.py` decorators | Use `preflight_check`, read `guard.blocked` | They do enforce the configured policy; they do not surface `action` or `reasons` |
| — | `fie detect` / `fie explain` exit code (W4-15) | Detector verdict | Not a policy gate: exits 0 for a benign verdict even when `policy.action` is `block` or `error`. Scripts must read `policy.action` |
| — | Any third-party code calling `scan_prompt` | Its own logic | No policy |

F3 is recommended as the next package: it is where a configured `strict` policy is silently
not applied. It changes failure behaviour for existing users and needs its own review.

---

## 9. Test and evaluation plan

New file `tests/test_runtime_policy.py`. Local only; the WP-003 network guard is reused.

| Area | Tests |
| --- | --- |
| **Purity of `decide`** | `decide(result, None)` raises `TypeError`. With `FIE_POLICY`, `FIE_SCAN_FAILURE_MODE`, `FIE_UNCERTAIN_ALLOW` and `PREFLIGHT_BLOCK_ENABLED` each set to conflicting values, and with `set_policy` pointing elsewhere, `decide(result, p)` returns the same decision for the same `(result, p)`. `os.environ` access is trapped during `decide` and must not occur. The `ScanResult` passed in is equal, field for field, before and after, and gains no attribute. The scan cache holds no `PolicyDecision` |
| **Decision table, exhaustive** | Cross product of: 3 presets (and both `compat` forms) × {no result; every valid (zone, decided_by) pair; bypassed allow and block} × classifier state {`ok`, three `unavailable_*`, `error`, `timeout`, `disabled`} × {another layer degraded, disabled, or neither} × {`coverage=None`}. Each cell is compared with a table written out in the test, not derived from the implementation |
| **Invariants (§6.1)** | Property tests over every constructible input: (1) zone `clear_block` or tiebreaker-confirmed → `action == "block"` under every preset and every coverage state; (2) allow verdict with a required layer not `ok`, or `coverage=None` → `reasons[0] in {missing_required_layer, coverage_unknown}` and never `detector_allow`; (3) `blocked` equals the §5.2 rule; (4) reasons non-empty, all from the closed set, no duplicates; (5) `decide` never returns `invalid_policy_config` or `warn_only` |
| **Policy validation** | Every invalid field value, unknown layer, malformed name, the reserved name `invalid`, version 0 → `ValueError`. Instances are frozen. Shipped presets pinned by literal field values and digest |
| **Process configuration** | Unset → `compat@1`; each valid name, with surrounding whitespace and mixed case; `FIE_POLICY` read exactly once (changing it after the first call has no effect, valid→valid and valid→invalid); `set_policy` outranks the environment; a per-call `policy=` outranks both and `active_policy` is not called (spy); `reset_policy()` re-reads `FIE_POLICY`; `set_policy("strict")` raises `TypeError`; 16 threads calling `active_policy()` concurrently on a cold cache read the environment once and agree |
| **Effective policy (§5.5)** | (a) With the selection `compat`: `_FAIL_SECURE` False → `active_policy()` is the `allow` form; True → the `block` form; same id `compat@1`, different pinned digests. (b) Flipping the flag between two calls flips the returned form on the next call, without `reset_policy()`. (c) Setting `FIE_SCAN_FAILURE_MODE` in the environment *after* import, in either direction, changes nothing; `fie.policy` never reads that variable (environment-access trap on the name). (d) Under `strict`, `review`, `set_policy(p)` and per-call `policy=p`, flipping the flag changes nothing: scanner raising gives the policy's own `on_scan_failure`. (e) An explicit `compat_policy(scan_failure="allow")` passed per call with the flag True → scanner failure is allowed, digest is the `allow` form's. (f) `preflight_check` calls `active_policy` exactly once per call and before `scan_prompt` (call-order spy). (g) **Identity equals effect**: for every case above, `decision.policy == p.id` and `decision.policy_digest == p.digest` for the object `p` that was resolved, and re-running `decide(result, p, scan_failed=…)` by hand reproduces the decision exactly. (h) No substitution: the flag flipped and `set_policy` called from another thread *during* a scan do not change that call's decision or digest |
| **Invalid configuration** | `FIE_POLICY` set to a string containing a path, a token and an e-mail address: `preflight_check` returns exactly the §5.6 field table; `scan_prompt` is **not called** by `preflight_check` (spy, call count 0); warn-only on → still blocked; an empty prompt still returns unblocked with `action=None`; a per-call `policy=` still works; the failure persists after the variable is corrected and clears after `reset_policy()` and after `set_policy(valid)`; exactly one ERROR record across many calls; none of the planted strings in `GuardResult.to_dict()`, the exception message, the log records or the CLI output; `policy_digest` is `None` |
| **Invalid configuration, CLI** | For `detect` and `explain`, in JSON, text and `--quiet` modes, with a benign and an attack prompt: `scan_prompt` **is called exactly once** (spy); stdout JSON has every existing key in order with values equal to a direct scan, followed by `policy` equal to the §5.6 block with `detector_is_attack`, `zone`, `coverage_status` equal to the scan's; exit code is `1 if is_attack else 0` (0 for the benign prompt, 1 for the attack); stderr holds exactly the one fixed notice, also under `--quiet`; with a valid configuration stderr is empty; the planted `FIE_POLICY` value appears in neither stream; the `ScanResult` equals the one produced under a valid configuration |
| **Configuration failure ≠ scanner failure** | Side by side: scanner raising under `strict@1` versus invalid config. Assert the three distinguishing fields (`reasons[0]`, `scan_failed`, `policy`) differ as §5.6 states |
| **C1 enforcement under `compat@1`** | The §7.2 C1 table, branch by branch, against `preflight_check` as it is at `924c70d`: the eight WP-003 reference prompts with the classifier loaded and missing (three reasons), tiebreaker down; scanner raising with `_FAIL_SECURE` False and True; each of those with warn-only on and off |
| **C2 existing `GuardResult` values** | For every C1 case, the nine existing fields equal the values captured from the unmodified code (the WP-003 `PRE_GUARD` literals, extended before any change to the scanner-failure and warn-only branches) |
| **C3 `GuardResult` additions** | Old constructors (positional and keyword) work and give `action=None`, `policy=None`, `reasons=()`; `to_dict()` key tuple is the nine existing keys followed by `action`, `policy`, `reasons`; JSON round-trip |
| **C4 CLI** | For both commands: the key tuple is today's tuple followed by `policy`; every earlier value equals today's; exit code equals `1 if is_attack else 0` under all three presets, including the case where `policy.action` is `block` on a benign verdict (`strict@1`, classifier missing → exit 0); `Policy` text line present only when specified; stderr empty under every valid configuration |
| **C5 detector unchanged** | `ScanResult.to_dict()` key tuple is the WP-003 fifteen. For the eight reference prompts × {loaded, three unavailable reasons}, the result of `scan_prompt` under `FIE_POLICY=strict`, `=review`, unset and invalid is equal field for field. `fie/adversarial.py` has no diff. The WP-003 contract file passes with only the C3 key-tuple assertion updated |
| **Missing classifier, timeout, error, cold-start race** | Real loader in its three failure branches; classifier slowed past a shortened deadline; classifier encoder raising; the WP-003 "loading" stand-in. `compat`: allow with `missing_required_layer` first; `strict`: block; `review`: review. A non-required layer raising: allow with `degraded_optional_layer` under all three |
| **Labelled fast paths** | Whitelisted and known-attack hashes under each preset, with the classifier loaded and missing: the label decides |
| **Disabled layers** | Required and non-required, via `scan_prompt(disabled_layers=…)` then `decide` |
| **Lite** | `decide(scan_prompt_lite(...), p)` under each preset |
| **Warn-only (§6.1 table)** | Each of the six rows, under each preset where reachable. In particular: `strict@1` + classifier missing + warn-only → blocked; `strict@1` + scanner raises + warn-only → blocked; `review@1` + missing coverage + warn-only → blocked; invalid config + warn-only → blocked; detector block + warn-only → not blocked, `warn_only` last in `reasons`, `action` still `block` |
| **Verdict and action stay apart** | `ScanResult` has no `action`, `policy` or `reasons` attribute; `PolicyDecision.zone` is a zone or `None`; `PolicyDecision.action` is never placed in a `zone` field; a case where they visibly differ (`detector_is_attack=False`, `action="block"`) serializes both |
| **Untrusted input cannot select policy** | No parameter of `decide` or `preflight_check` accepts a mapping or a policy *name*; static check that nothing in `app/` or `fie/integrations/` passes `policy=`, calls `set_policy`, or forwards `disabled_layers` from a request |
| **Integrations unchanged (limitation is real)** | Existing integration tests pass unmodified. One new test documents the gap rather than hiding it: under `FIE_POLICY=strict` with the classifier missing, the OpenAI wrapper's scan path still forwards. It is asserted as *current behaviour* and named so that the follow-up package must change it deliberately |
| **No leak** | Planted-string search (§8.5) over `PolicyDecision.to_dict()`, `GuardResult.to_dict()`, both log lines, the exception message, CLI output |
| **Concurrency** | 32 `preflight_check` calls on 8 threads under each preset equal the single-threaded decisions; `set_policy` during a run never yields a decision mixing two policies |
| **Baseline replay** | `decide` over synthetic results rebuilt from every `BL-0002` record: action counts equal §6.4 exactly for all three presets |
| **Mutation checks** | Each must fail at least one test: swap rows 3 and 6; let row 6 emit `detector_allow` first; invalid `FIE_POLICY` falls back to `compat`; invalid config reported as `scan_failed`; `decide` reads `FIE_POLICY`; `fie.policy` reads `FIE_SCAN_FAILURE_MODE`; `active_policy()` caches the `compat` form instead of following the flag; `preflight_check` re-resolves or swaps the policy after scanning; the decision reports a digest other than its policy's; `preflight_check` scans under an invalid configuration; the CLI skips the scan, or changes its exit code, under an invalid configuration; warn-only waives a `missing_required_layer` block; warn-only waives a `scan_failed` block; `decide` writes an attribute on the result; a `GuardResult.to_dict()` key reordered |

**Performance.** `decide` micro-benchmark, budget 5 µs median. `active_policy()` after
resolution, budget 1 µs. `preflight_check` warm latency before and after, interleaved, with
the micro-benchmarks as the bound. No cold-start cost: nothing is loaded.

**Baseline and determinism.**

1. Before any change: `python -m evals baseline` exit 0 against `BL-0002`.
2. After: `python -m evals baseline` **exit 0**: eight counts equal, all **eleven** record
   files byte-identical to `BL-0002`. No file is expected to differ, because the harness
   calls `scan_prompt`, which this package does not touch.
3. The run's `fie` tree hash will differ from `70de7053` (a file is added and two are edited).
   That changes run names and the subject key, not records. `BL-0002` stays canonical and no
   `BL-0003` is created (W4-16).
4. `verify-determinism --all` exit 0 on the final tree.
5. If any record file differs, stop: the package touched detection.

## 10. Acceptance criteria

| # | Criterion | Proof |
| --- | --- | --- |
| AC-1 | `decide` is pure: it requires an explicit `Policy`, reads no environment variable or mutable global, performs no I/O, and does not mutate or annotate its input | Purity tests; environment-access trap; static import test; mutation check |
| AC-2 | Actions are a closed set of four and reasons a closed set of fifteen, both validated | Invariant tests |
| AC-3 | Every row of §6.1 and §6.3 is implemented as written | Decision-table test |
| AC-4 | **A detector block is never downgraded by missing coverage** | Invariant 1 over all inputs; mutation check |
| AC-5 | **Missing required coverage never yields a bare `allow`**: `reasons[0]` is `missing_required_layer` or `coverage_unknown` | Invariant 2 over all inputs; mutation check |
| AC-6 | **C1.** Under `compat@1`, `preflight_check` forwards and stops exactly the inputs it does today, in every existing branch, with the scanner-failure branch following `_FAIL_SECURE` with today's timing | §7.2 C1 table tests; effective-policy tests (a) to (c) |
| AC-7 | **C2.** Under `compat@1`, the nine existing `GuardResult` fields have today's values | Captured-literal comparison |
| AC-8 | **C3.** `GuardResult.to_dict()` is the nine existing keys in their existing order followed by exactly `action`, `policy`, `reasons`; old constructors work | Key-tuple and constructor tests |
| AC-9 | **C4.** CLI JSON is today's keys in today's order with today's values, followed by exactly `policy`; exit codes follow the detector verdict under every configuration, valid or invalid | CLI tests under every configuration |
| AC-10 | **C5.** `ScanResult.to_dict()` schema unchanged; `scan_prompt` returns the same result under every policy configuration; `fie/adversarial.py` has no diff | Key-tuple test; equality tests; `git diff` |
| AC-11 | Detector verdicts unchanged | Eleven `BL-0002` record files byte-identical; eight counts equal; `evals baseline` exit 0 |
| AC-12 | Determinism preserved | `verify-determinism --all` exit 0 |
| AC-13 | An invalid `FIE_POLICY` never selects a preset. In `preflight_check` it yields `action="error"`, `blocked=True`, `reasons=("invalid_policy_config",)`, `policy="invalid"`, `policy_digest=None`, and **no scan runs** | Invalid-configuration tests with a `scan_prompt` spy (0 calls); mutation check |
| AC-13b | In the CLI an invalid `FIE_POLICY` **does not suppress the scan**: `scan_prompt` runs once, existing output and exit code are those of the real scan, the appended `policy` block is the §5.6 error decision, and one fixed notice goes to stderr. No detector field is fabricated | CLI invalid-configuration tests with a spy (1 call); mutation checks |
| AC-14 | Configuration failure is distinguishable from scanner failure | Side-by-side test on `reasons[0]`, `scan_failed`, `policy` |
| AC-15 | The configuration lifecycle is as documented: `FIE_POLICY` read once, selection cached including failure, changed only by `set_policy` / `reset_policy` / restart; thread-safe | Process-configuration tests |
| AC-15b | **One effective-policy rule.** The policy for a call is the per-call argument, else `active_policy()` called once before the scan. `FIE_SCAN_FAILURE_MODE` is read only by `fie.preflight` at import; the `compat` form follows `_FAIL_SECURE`; no other policy consults it | Effective-policy tests (a) to (f); mutation checks |
| AC-15c | **Identity equals effect, no substitution.** `decision.policy` and `decision.policy_digest` are those of the exact `Policy` object passed to `decide`, and replaying `decide` with that object reproduces the decision | Effective-policy tests (g), (h); mutation checks |
| AC-16 | Warn-only waives only detector-origin blocks; it never waives a missing-coverage, scanner-failure or configuration block | §6.1 warn-only table tests; two mutation checks |
| AC-17 | No request field can select or weaken policy | Signature and static tests |
| AC-18 | Every decision names its policy (`name@version`, or `invalid` with a `None` digest); `PolicyDecision` carries the digest, which distinguishes the two forms of `compat@1` | Field presence on `PolicyDecision`, `GuardResult`, log line, CLI JSON; pinned digests |
| AC-19 | Policy outcomes are separate from detector verdicts: nothing is added to `ScanResult`, and no `PolicyDecision` is cached or stored on one | Attribute and cache tests |
| AC-20 | No invalid environment value, prompt text, exception text, path or secret in any policy output or log line | Planted-string tests |
| AC-21 | Action counts on `BL-0002` equal §6.4 for all three presets | Replay test |
| AC-22 | Existing tests pass: all 785 by name, with exactly one assertion updated (the `GuardResult.to_dict()` key tuple, C3) | Name comparison before and after; diff of the one test file |
| AC-23 | Integrations, server, deployment files and dependencies are unchanged, and the execution report states the enforcement limits of §8.6 | `git diff --stat`; report review |
| AC-24 | `decide` costs ≤ 5 µs median | Micro-benchmark |
| AC-25 | Only the files in §11 changed | `git diff --stat` |

## 11. File-by-file change matrix

**NEW**

| File | Purpose | Security | Compatibility | Tests | Rollback |
| --- | --- | --- | --- | --- | --- |
| `fie/policy.py` | `Policy`, `PolicyDecision`, presets, pure `decide`; separately `active_policy`, `set_policy`, `reset_policy`, `PolicyConfigError` | Closed sets; validated; `decide` has no I/O and no hidden state; the only environment read is in `active_policy` | New module; nothing imports it until wired | All of §9 | Delete |
| `tests/test_runtime_policy.py` | §9 | — | — | — | Delete |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_004_runtime-policy.md` | Report, at implementation | — | — | — | — |

**MODIFY**

| File | Existing responsibility | Proposed modification | Why | Security | Compatibility risk | Tests | Rollback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fie/preflight.py` | The enforcement point; builds `GuardResult` | Let `_safe_scan`'s outcome also carry the `ScanResult` as an attribute (it already carries `zone` and `coverage_status` that way; the five returned values are unchanged); resolve the policy once, before scanning (per-call argument, else `active_policy()`; `PolicyConfigError` → the §5.6 result, no scan); call `decide(result, policy, scan_failed=…)` with that same object; `_FAIL_SECURE` and its import-time assignment are not edited; apply the warn-only rule of §6.1; append `action`, `policy`, `reasons` to `GuardResult` and to the end of its `to_dict()`; keyword-only `policy=`; one audit log line | The only place that both sees coverage and enforces | Invalid config → blocking `error`; warn-only cannot waive non-detector blocks | Low: C1 and C2 proven against captured values | AC-6, 7, 8, 13, 14, 16 | Revert file |
| `fie/__main__.py` | CLI | Resolve the policy before scanning; append `policy` to both JSON outputs; one `Policy` text line when the action is not what `is_attack` alone implies; under an invalid configuration still scan, append the error decision and write one fixed stderr notice. Exit codes untouched | Consistency with the SDK path | Fixed codes only | Low: one key appended (C4) | AC-9 | Revert file |
| `tests/test_scan_result_contract.py` | WP-003 contract tests | **One assertion, authorized (Q5)**: the pinned `GuardResult.to_dict()` nine-key tuple becomes the twelve-key tuple, appending `action`, `policy`, `reasons`. No other assertion in this file, and no other test file, may change | The tuple is pinned exactly, so an approved additive key must be reflected | None | None | AC-22 | Revert file |
| `fie/__init__.py` | Exports | Export `Policy`, `PolicyDecision`, `decide` | Public contract | None | None | Import test | Revert file |
| `docs/fie_rebuild_2026/MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | Log | At implementation only | — | — | — | — | — |

**MUST NOT CHANGE**

`fie/adversarial.py`, `fie/layers/`, `fie/onnx_encoder.py`, `fie/_lite.py`,
`fie/multilingual.py`, `fie/llama_guard.py`, `fie/feedback_store.py`, `fie/monitor.py`,
`fie/integrations/` (W4-9), everything in `app/`, `engine/`, `storage/` (W4-10), `evals/`
including `evals/subject.py`, `BL-0001`, `BL-0002`, `REF-0001` and `CANONICAL` (W4-16),
`tests/security/`, `tests/evals/`, every other existing test, the golden file, model files,
manifests, dependency files, Dockerfiles, deployment settings, `README.md`.

If `fie/adversarial.py` turns out to need a change, that is a stop condition, not a detail.

---

## 12. Implementation sequence and stop conditions

| Step | Work | Files | Acceptance | Rollback point |
| --- | --- | --- | --- | --- |
| 0 | Record HEAD; 785 tests; `evals baseline` exit 0. Capture today's `GuardResult` values for the scanner-failure and warn-only branches. Write `tests/test_runtime_policy.py` first; record which tests fail | new test file | Tests needing the module fail (it is absent); the C1 and C2 tests against today's `preflight_check` pass | Delete the test file |
| 1 | `fie/policy.py`: types, validation, presets, pure `decide`; then `active_policy`, `set_policy`, `reset_policy` | new module | AC-1, 2, 3, 4, 5, 15, 20, 21, 24 | Delete the module |
| 2 | Wire into `preflight_check`; `GuardResult` fields; the one C3 assertion | `fie/preflight.py`, `tests/test_scan_result_contract.py` | AC-6, 7, 8, 13, 14, 16, 17, 18, 19 | Revert |
| 3 | CLI and exports | `fie/__main__.py`, `fie/__init__.py` | AC-9 | Revert |
| 4 | Mutation checks; performance | — | Every mutation fails a test | — |
| 5 | Full suite; `evals baseline`; `verify-determinism --all` | — | AC-10, 11, 12, 22, 23, 25 | — |
| 6 | Execution report; master log | docs | — | — |

The assistant commits nothing.

**Stop conditions.** Stop and report if: any `BL-0002` record file differs; `fie/adversarial.py`
or any detector file must change; `ScanResult` needs a new field; `compat@1` cannot reproduce
a current branch exactly; a dependency or network call is needed; a server or tenant change
becomes necessary; a file outside §11 must change; an acceptance criterion cannot be met
without widening scope; the decision table needs a row that §6 does not have; `decide`
cannot be kept pure; any existing test other than the one C3 assertion would have to change;
an integration, server route, Dockerfile or deployment setting would have to change.

---

## 13. Rollback

Delete `fie/policy.py` and `tests/test_runtime_policy.py`; revert `fie/preflight.py`,
`fie/__main__.py`, `fie/__init__.py` and the one assertion in
`tests/test_scan_result_contract.py`. Nothing is persisted: no stored data, no cache format,
no database field, no baseline. Operationally, a deployment returns to today's behaviour
without a code change by unsetting `FIE_POLICY` **and restarting the process**; the variable
is read once (§5.5), so unsetting it in a running process does nothing.

---

## 14. Owner decisions

### 14.1 Decided by the owner on 2026-10-10

| # | Decision | Recorded as |
| --- | --- | --- |
| W4-1 | Numbering | **Keep WP-004.** The roadmap ([ROADMAP.md](../ROADMAP.md)) calls runtime policy WP-007 and uses WP-004 for "install truth" (E51) and WP-005 for the hermetic default. This package takes the WP-004 number; the roadmap is not edited here. The discrepancy is to be recorded in the master log when the implementation is executed |
| W4-2 | Architecture | **Approved:** a separate pure policy module (design D) |
| W4-3 | Action values | **Lowercase `allow`, `review`, `block`, `error`.** `PolicyDecision.action` is the policy's outcome and is distinct from the detector's verdict (`ScanResult.is_attack`, `.zone`); §5.4 |
| W4-4 | Default policy | **`compat@1`.** Stricter presets take effect only when an operator selects them |
| W4-5 | Required layers | **`pair_classifier` only**, in this package |
| W4-6 | `review` and `blocked` | **`review` maps to `blocked=True` by default** |
| W4-7 | `error` | **Kept separate from `block`** |
| W4-8 | Invalid `FIE_POLICY` | **A blocking `error` with the dedicated fixed reason code `invalid_policy_config`.** Never a silent fallback to a permissive preset; §5.6 |
| W4-9 | Other SDK call sites | **Not modified in WP-004:** OpenAI, Anthropic, FastAPI middleware, `monitor`. Their policy bypass and existing fail-open behaviour are recorded as prioritized follow-up (§8.6, F3). WP-004 is not to be described as enforcing policy across all FIE entry points |
| W4-10 | Server | **Out of scope:** server routes, shipped deployment configuration, server start-up behaviour (§8.6, F4) |
| W4-11 | Per-tenant policy | **Deferred** |
| W4-12 | Empty prompt | **Existing behaviour preserved.** No policy decision is made; it is represented as `action=None` on `GuardResult` |
| W4-13 | Required model digest | **Deferred** |
| W4-14 | `risk_score`, `decision` field, inert `threshold=` warning | **Deferred** |
| W4-15 | CLI exit code | **Unchanged** |
| W4-16 | Baseline | **`BL-0002` stays canonical. No `BL-0003`** |

### 14.2 Clarifications resolved in revision 2

| | Ambiguity in revision 1 | Resolution | Where |
| --- | --- | --- | --- |
| A | `decide(result, policy=None)` implied a hidden lookup of process state | `policy` is required and explicit; `decide` reads only its arguments. Process configuration is a separate function with a stated lifecycle, precedence and thread-safety contract. The decision is never cached in or written to `ScanResult` | §5.1, §5.5 |
| A | `compat@1` "reads `FIE_SCAN_FAILURE_MODE`" contradicted purity | The preset is a value in two forms; the setting is supplied from outside `decide`. (The two-reader arrangement written here in revision 2 was itself ambiguous and is replaced by A′ below) | §6.2 |
| B | Invalid configuration was reported with reason `scan_failed` | Dedicated reason `invalid_policy_config`; exact `GuardResult` field values; `policy="invalid"`; no scan; `scan_failed=False` | §5.3, §5.6 |
| B | "Read once, at first use" sat beside an implied immediate effect | Explicit lifecycle: read once, cached including failure, no hot reload, three recovery paths | §5.5 |
| B | Exposure of the bad value | Fixed strings everywhere; test with a planted value | §5.6, §8.5, §9 |
| C | One compatibility table mixed five things | Five separate statements C1 to C5, each with its own acceptance criterion | §7.2, AC-6 to AC-10 |
| C | "Existing tests unmodified" conflicted with appending keys to a pinned tuple | One named assertion is updated; listed in the file matrix | §7.2 C3, §11, AC-22 |
| D | Integration gaps were an open decision | Recorded as limits, with a warning against claiming universal enforcement, and a test that documents the gap | §1, §8.6, §9, AC-23 |
| E | Warn-only "is not a row" left its interaction with strict blocks to prose | Exact predicate and a six-row table; only detector-origin blocks are waivable | §6.1, AC-16 |
| E | "Bare allow" was not testable as written | Defined as `reasons[0]`; invariant over all inputs | §6.1, AC-5 |
| A′ (rev. 3) | Two readers of the scanner-failure setting (`active_policy()` reading the variable once; `preflight_check` using the import-time flag) could disagree | One source: `fie.preflight._FAIL_SECURE`, read by `active_policy()` at each call. `fie.policy` never reads `FIE_SCAN_FAILURE_MODE`. `preflight_check` resolves once, before the scan, and never substitutes. Identity and digest are those of the object passed to `decide` | §5.5, §6.2, §7.1, AC-6, AC-15b, AC-15c |
| B′ (rev. 3) | "No scan under invalid configuration" contradicted the CLI scanning first | The no-scan rule is `preflight_check`'s. The CLI is a diagnostic, not an enforcement point: it still scans, reports real findings, appends the error decision, writes a stderr notice, and keeps its exit code. The irreconcilable alternative (skip the scan) is recorded with the reason it was rejected | §5.6, §6.3, §7.2 C4, AC-13, AC-13b |

### 14.3 Questions Q1 to Q7 — decided by the owner on 2026-10-10

| # | Question | Decision |
| --- | --- | --- |
| Q1 | API names: `Policy`, `PolicyDecision`, `decide`, `active_policy`, `set_policy`, `reset_policy`, `compat_policy`; `GuardResult` fields `action`, `policy`, `reasons` | **Approved as written** |
| Q2 | Identity when no valid policy exists | **`policy="invalid"` and `policy_digest=None`** |
| Q3 | The two scanner-failure forms of `compat@1` | **One shared id, `compat@1`; the effective configuration is distinguished by `policy_digest`** |
| Q4 | Per-call `preflight_check(..., policy=Policy)` | **Retained**: keyword-only, for trusted code only; never populated from request data |
| Q5 | The pinned `GuardResult.to_dict()` assertion in `tests/test_scan_result_contract.py` | **Authorized**: that single assertion changes from the nine-key to the twelve-key tuple, appending `action`, `policy`, `reasons`. No other assertion or test file may change under this exception |
| Q6 | Test documenting the OpenAI integration's policy bypass | **Kept** |
| Q7 | Hosted-demo preset; whether external users rely on integrations failing open | **Deferred** to the relevant follow-up packages (F4, F3) |

### 14.4 Remaining points for the implementation gate

No open conflict. Three consequences of the approved decisions are restated here so that they
are accepted knowingly, not discovered later:

| # | Point | Follows from |
| --- | --- | --- |
| R1 | The CLI exit code is never a policy gate. A script gating on `fie detect --quiet` is not protected by any preset, nor by the invalid-configuration `error` | W4-15 |
| R2 | The CLI writes one line to stderr under an invalid `FIE_POLICY`, including with `--quiet`. It is the only output outside stdout and occurs only in that error state. If any stderr output under `--quiet` is unwanted, the alternative is JSON/text only, leaving `--quiet` with no signal at all | Revision 3, §5.6 |
| R3 | An operator who sets `FIE_SCAN_FAILURE_MODE=open` together with `FIE_POLICY=strict` gets scanner failures **blocked**: a selected preset's own setting wins over the legacy variable, which shapes `compat@1` only | §5.5 |

## 15. Explicit non-goals

No change to detectors, thresholds, weights, aggregation, normalization, segmentation,
models or benchmark records. No RAG, retrieval, claim verification, answer correction,
hallucination monitoring or training. No authentication, authorization or tenant change. No
scan API. No YAML policy language, per-category thresholds, conformal calibration or
`risk_score`. No dependency. No network call. No README, marketing or deployment change. No
fix for NF-10, NF-11, NF-13 or any WP-002 residual risk.

---

## 16. Relationship to FIE Reliability

The reliability workstream (RAG grounding, claim verification, correction) will produce the
same shape of thing a detector does: evidence, with a statement of how much of the intended
checking actually ran. A grounding check that could not reach its retriever is the
reliability twin of a scan without its classifier.

The separation built here carries over unchanged:

- **Evidence producers report; they do not act.** A verifier returns findings plus coverage,
  as `scan_prompt` now does.
- **One pure function turns evidence and coverage into an action**, under a named, versioned
  policy. The same four actions fit: allow the answer, send it to review, block it, or report
  that it could not be checked.
- **"Could not check" is never "passed".** AC-5 is the rule a reliability policy needs most.

What WP-004 deliberately does not do is generalise `decide` for inputs it has not seen. A
later package can add a second evidence type next to `ScanResult`; nothing here has to be
undone for that.

---

## 17. Estimated complexity

| Item | Estimate | Uncertainty |
| --- | --- | --- |
| `fie/policy.py` | 250–400 lines including docstrings | Medium |
| `fie/preflight.py`, `fie/__main__.py`, `fie/__init__.py` | 80–150 lines changed | Low |
| Tests | 800–1,300 lines; 200–300 test cases, most of them generated table cells; one assertion edited in an existing file | Medium |
| Harness time | Two baseline runs and one determinism run, about an hour of machine time | Low |

WP-003's plan under-estimated its size by a factor of about 3.5. These ranges already allow
for that; the main unknown is how much `preflight_check` has to be restructured to keep
`compat@1` exact in the warn-only and fail-secure branches. Revision 2 adds the configuration
lifecycle and the invalid-configuration path; they are inside these ranges, toward the upper
end. The integrations and the server are out of scope (W4-9, W4-10), so the earlier caveat
about the package doubling no longer applies.

---

## 18. Final recommendation

The design is settled by the owner's decisions in §14.1: a pure policy function taking an
explicit policy, three named presets with `compat@1` as the default, applied in
`preflight_check` and shown by the CLI, with the detector, `ScanResult` and `BL-0002`
untouched.

Revision 3 fixes the last two: there is one rule for the effective policy, with a single
reader of the scanner-failure setting and a decision that names exactly the policy that
produced it; and an invalid configuration stops `preflight_check` without a scan while the
CLI, which enforces nothing, still reports the real scan beside an explicit policy error.

Revision 2 removed the contradictions that would otherwise have been decided during coding:
`decide` has no hidden inputs; configuration has a stated lifecycle; an invalid configuration
is its own outcome and not a scanner failure; compatibility is five separate, testable
statements; warn-only has an exact rule; and the package's limits are written down so that it
cannot be mistaken for enforcement across every entry point.

Two things remain true and should stay visible. The default still forwards a scan that ran
without the classifier, now with an explicit reason on the result. And an operator who
selects `strict@1` is protected only on the `preflight_check` path until the integrations are
migrated in a follow-up package.

**Implementation has not started and is not authorized by this revision.**
