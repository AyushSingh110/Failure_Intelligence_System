# FIE evaluation harness

Measures the guardrail under named, fingerprinted configurations, so that every
change to FIE can be compared with the exact previous version.

The harness measures. It is **not** part of the request-serving path, is not
shipped in the wheel, and changes no file under `fie/`, `engine/`, `app/` or
`storage/`.

## Run it

From the repository root, in the project's Python environment. No install step.

```bash
python -m evals selftest                 # guard canaries, model hashes, dataset hashes (seconds)
python -m evals baseline                 # THE one command: every suite, then check the counts
python -m evals run --plan standard      # the standard benchmarks only (~2 min)
python -m evals run --profile reference-v6.2     # reproduce E22 with the older PAIR v6.2
python -m evals verify-determinism       # run twice in separate processes, compare bytes
python -m evals verify-determinism --all # every deterministic suite; required before pinning
python -m evals show <run dir>           # print a run's report
python -m evals pin <run dir> --canonical --determinism-proof <DETERMINISM.json>
```

Output goes to `evals/runs/<UTC time>_<subject>_<plan>/` (git-ignored).
`python -m evals baseline` exits 0 only when the standard counts equal the
approved canonical counts and, once a baseline is pinned, the per-prompt records
are byte-identical to it.

## Four behaviours — keep them apart

| | Behaviour | Measured by |
| --- | --- | --- |
| **A** | **Shipped / default behaviour.** What a user gets from the code as shipped: online translation attempted, language detector unseeded, telemetry on | Not measured directly by a deterministic suite. C and D show where it differs |
| **B** | **Canonical reproducibility/evaluation profile** (`sdk-offline-failsecure`). Every network path closed, every known source of run-to-run variation fixed | The standard, risk and pilot suites |
| **C** | **Lite profile** (`lite-simulated`). The ML packages are unimportable: the code path of a base `pip install fie-sdk` | `lite.std` |
| **D** | **Stability / unseeded behaviour.** Profile B with the language-detector seed removed, repeated | `stability` (run metadata) |

**Profile B is a controlled measurement configuration. It is not the production
runtime environment.** It changes ten things at run time, inside the worker
process only (the list is in `registry/profiles.json` and printed in every report).
Two of them change what is measured: a fixed language-detector seed and a fixed
hash seed.

## What a run proves, and what it does not

| Claim | Mechanism | Limit |
| --- | --- | --- |
| **Zero outbound connection attempts through the Python runtime, with every known egress path closed at its source** | Egress closed by the profile; worker environment built from an allowlist; a Python audit hook that denies and records socket creation, DNS and process creation; canaries at start; accounting at the end. The run fails on the guard's *record*, because the product swallows the exception | Not operating-system or native-code isolation. Native code that calls the OS network API directly is invisible to a Python guard |
| The models measured are the published models | SHA-256 against `scripts/model_manifest.json`: before the worker starts, again before `fie` is imported, again after the run. Then the loader's own log line, threshold and declared version are cross-checked | Trusts the manifest at the checked-out commit |
| The datasets are the pinned datasets | Content hash over canonically serialized rows; row and unique counts | Upstream revisions of four benchmarks are not recorded locally |
| Two runs agree | `MANIFEST.sha256` over the deterministic files | Byte identity is promised only when all four fingerprint keys match: same machine class, same package versions |

Two local-only probes run immediately **before** the guard is armed, because they
would otherwise be denied although nothing leaves the machine: `platform.uname()`
(runs `cmd /c ver` on Windows) and `urllib3`'s import-time IPv6 check (creates a
socket and binds it to `::1`). Nothing is exempt after arming.

## Files in a run directory

| Deterministic result artifact (byte-compared) | Run metadata (never byte-compared) |
| --- | --- |
| `fingerprint.json` — identity, configuration, environment, four keys | `run.json` — run id, times, exit codes, guard accounting |
| `records/<suite>.jsonl` — one line per scanned input | `timing/<suite>.jsonl` — per-prompt milliseconds |
| `evidence/<suite>.jsonl` — the subject's raw evidence (reported, not gated) | `nondeterministic/` — latency and stability raw data |
| `summary.json` — counts, rates, intervals, zones | `latency.json`, `known_unstable.json`, `RUN_NOTES.md` |
| `REPORT.md` — the human report | `work/`, `logs/`, `status/`, `plan.json` |
| `MANIFEST.sha256` — hash of each of the above | |

No timestamp, duration, run id, host name, absolute path or process id appears
in the left column.

A record holds the hash of the scanned text, not the text: the prompts are
already in the repository.

## The fingerprint and comparability

| Key | Covers | If two runs differ |
| --- | --- | --- |
| `dataset_key` | dataset content, suite definitions, fixtures, schema version | **incomparable** |
| `config_key` | the profile and everything the subject reports about its configuration | comparable only as a declared configuration change |
| `subject_key` | hash of `fie/**/*.py` and every model hash | expected: this is the change being measured |
| `env_key` | Python minor version, platform, numpy / scikit-learn / onnxruntime / tokenizers / xgboost versions | verdicts comparable with a warning; latency is not |

A run is **canonical** — and only then may be pinned — when: the measured paths
(`fie/`, `engine/`, `app/`, `storage/`, the model manifest, the frozen datasets,
`pyproject.toml`) have no uncommitted change; every model is in the manifest; no
`--limit`, `--suites` or explicit model hash was used; every suite completed with
no scan error; and the hermetic proof holds. The harness's own source is pinned
by `identity.harness.tree_sha256`.

## Suites

| Suite | What it is | Label |
| --- | --- | --- |
| `std.*` | XSTest (safe + unsafe contrast), OR-Bench-hard, JailbreakBench, HarmBench, StrongREJECT, SORRY-Bench | frozen, leakage-audited benchmarks |
| `std.advbench` | AdvBench remainder | **case study** — never in a headline |
| `risk.long_input` | one fixed benign filler sentence before or after a position-chosen sample | **constructed risk suite / diagnostic probe** |
| `risk.framing` | four fixed hand-written templates around every attack prompt | **constructed risk suite / diagnostic probe** |
| `pilot.script` | 72 hand-written benign prompts, 12 groups of 6 | **PILOT** — counts only, no rate, no interval |
| `lite.std` | the standard benchmarks with the ML packages unimportable | **constructed risk suite / diagnostic probe** |
| `latency` | warm scan timing, with length buckets | run metadata |
| `stability` | unseeded passes; counts verdicts that change | run metadata |

The risk suites and the pilot are not representative benchmarks and support no
claim about any population of real prompts. The headline always shows attack
recall **and** benign over-refusal; a run that covers one axis opens with
"Single-axis run. Not a security result."

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | completed |
| 1 | usage error |
| 2 | a suite is incomplete, or a scan raised |
| 3 | model integrity failure |
| 4 | dataset integrity failure |
| 5 | hermetic violation, or the guard's self-test failed |
| 6 | counts or records differ from the canonical baseline |
| 7 | not deterministic |
| 8 | `--resume`: the plan or the subject changed |

## Baselines

`evals/baselines/CANONICAL` names the canonical baseline. A baseline directory
holds the deterministic files of one canonical run, plus `PIN.json` and the run's
notes as metadata. Baselines are never edited or overwritten: a correction is a
new baseline, and `CANONICAL` moves in a commit of its own.

`REF-…` directories are reference reproductions (summary and fingerprint only).
They check that the harness reproduces an independently published record; they
are not baselines.

## Layout

| Module | Responsibility |
| --- | --- |
| `cli.py`, `commands.py` | commands and exit codes |
| `orchestrator.py` | preconditions, sanitized environment, worker lifecycle, final assembly. Never imports `fie` |
| `worker.py` | one fresh interpreter per profile; runs suites inside the guard |
| `subject.py` | **the only module that touches `fie`**; every private name it reads is under a contract test |
| `hermetic.py` | audit-hook guard, import blocker, environment allowlist, self-test |
| `integrity.py` | model verification |
| `datasets.py` | dataset registry, parsing, content hashing |
| `transforms.py` | suite and profile registries; deterministic input construction |
| `fingerprint.py` | three blocks, four keys |
| `canonical.py` | the one serializer and the one hash |
| `metrics.py`, `report.py` | counts, intervals, the two-axis rule; `REPORT.md` and `RUN_NOTES.md` |
| `latency.py` | timing and stability statistics (metadata) |
| `registry/`, `fixtures/` | plain JSON and text; nothing in them is imported or executed |

Tests are in `tests/evals/`. Guard tests run in child processes, because the
audit hook cannot be removed once installed.

## Known limits

- Whether the canonical counts reproduce on Linux, or on another CPU, is not known.
- The harness depends on private names inside `fie` (`subject.PRIVATE_CONTRACT`).
- Upstream revisions for JailbreakBench, HarmBench, StrongREJECT and AdvBench are
  not recorded locally; they are pinned by content hash.
- The contamination audits cover the PAIR v6.2 training corpus and the v6.3 / v6.3b
  augmentation sets. For a model trained on other data the status is unknown.
- Baseline-versus-candidate comparison is designed but not built.
