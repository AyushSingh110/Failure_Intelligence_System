# PLAN_020 — Clean-Environment Installation Truth (experiment E51)

> **PROPOSED — EXECUTED IN OWNER-AUTHORIZED STAGES ONLY.**
> Revision 5. This is a plan for a measurement. Stage A (read-only preflight) and Stage B
> (release and provisioning checks, with one remediation) have been carried out under separate
> owner authorizations and are summarised in §0.1. **Stage C1 (base cells only) is authorized
> and its execution instructions are §0.2. Nothing beyond Stage C1 is authorized**: no `[ml]`
> dependency set may be downloaded or installed, and no script changes a network setting.
> Results are in the Stage C1 execution report, not in this plan.

| | |
| --- | --- |
| Proposed work package | **WP-020** (numbering: §13.2, decision O-1) |
| Experiment | E51, from [ROADMAP.md](../ROADMAP.md): "What does the PyPI wheel actually do in a clean environment?" |
| Date | 2026-10-11 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `2f62ecd` (WP-004) |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053` |
| Kind of work | Measurement only. No change to `fie/`, packaging, dependencies, models, workflows, baselines or defaults |
| Owner decisions | O-1 to O-24, §13.3. Decisions already taken during Stages A and B: §0.1 |
| Revision 4 changes | Installation modes for the `langdetect` source-build problem (§6.4), dependency verification gates (§7.5), sequential environments (§7.6), record of executed stages and of decisions that supersede earlier text (§0.1). Earlier revisions: §15 |

Evidence labels. **READ**: taken from a file in the repository at `2f62ecd`. **INSPECTED**:
taken from a build artifact or saved metadata without running it. **PREDICTED**: what the code
implies should happen; unverified until the experiment runs. **TO MEASURE**: unknown.

---

## 0. Starting-state verification

| Check | Result |
| --- | --- |
| Branch, HEAD | `rebuild/wp-001-eval-harness`, `2f62ecd` "add runtime policy layer applied in preflight_check (WP-004)" |
| WP-004 committed by the owner | Yes: `fie/policy.py` and `tests/test_runtime_policy.py` are tracked at `2f62ecd` |
| Working tree | Clean apart from this proposed plan (untracked); nothing staged |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053`; `BL-0001`, `REF-0001` present; no `BL-0003` |

Not re-run for this plan: the test suite and `python -m evals baseline`. Step 0 of the
experiment re-runs both on the committed tree before anything else.

### 0.1 Executed stages, and owner decisions that supersede text below

This plan is being executed in owner-authorized stages. The plan text is kept as the
specification; where a later owner decision differs from it, **the decision governs** and the
older text is left in place for the record.

**Executed so far** (evidence under `docs/fie_rebuild_2026/evidence/E51/`):

| Stage | Outcome |
| --- | --- |
| A — read-only preflight | Passed, except the network-off procedure, resolved by decision below |
| B — release and provisioning checks (`stageB/`) | Regression gate passed (942 tests; `evals baseline` exit 0). Latest PyPI release is `1.18.0`, wheel digest `be504dc7…579e963d`. No inherited pip configuration, proxy or `hosts` redirection. **Complete resolution failed**: `langdetect>=1.0.9` has no Python 3 wheel on PyPI, so `--only-binary :all:` cannot resolve it. Preserved as recorded |
| B remediation (`stageB_remediation/`) | Source archive `langdetect-1.0.9.tar.gz` verified against PyPI (`cbc1fef8…f6d605a0`); a wheel built from it twice, byte-identical (`9acdfda7…e6e9c8a4`), all 69 payload files equal to the archive's. With that wheel declared, the three sets resolve: base 11 packages (1.9 MB), current-tree `[ml]` 47 (209 MB), PyPI `[ml]` 56 (338 MB). Hash-required download verified for the base set only. Nothing installed into an experiment environment |

**Decisions that supersede earlier sections:**

| Decision | Supersedes |
| --- | --- |
| **Isolation is airplane mode for the entire measure phase**, switched by the owner; no script toggles a network setting. Positive connectivity controls are recorded before going offline and must fail afterwards; at the end the run pauses for the owner to restore connectivity, which is then verified | §5.2 and §5.3 (per-program firewall rules and their per-executable controls are **not used**); O-14, O-15, O-16, O-18. §5.4's limits on what may be claimed still apply in full |
| **No private interpreter copy.** Environments are created from the existing Python 3.10.19 interpreter; each environment's actual interpreter path and version are recorded | §5.2 item 1; F1 |
| Scratch directory is `C:\Users\ASUS\fie-e51-scratch` | `$X` throughout |
| **No `[ml]` dependency set is downloaded or installed without a separate explicit approval for that set** | O-4, extended to the current-tree set |
| Unmanifested model files: listing and hashing only | O-12, as recommended |
| A source build of `langdetect` is permitted as a narrowly scoped, recorded exception; `--only-binary :all:` is not relaxed for anything else | §5.6 step 2 |

The firewall design of §5.2 to §5.3 remains in the document as a considered alternative and as
the reasoning behind the limits in §5.4. It is not part of the execution.

### 0.2 Stage C1 execution instructions (revision 5)

Stage C1 is authorized by the owner as a limited execution: **the three mode B base cells and
the one mode A base cell.** Decisions O-24 approved; **O-21 and O-22 not approved**, so neither
`[ml]` set is downloaded or installed and every `[ml]` cell is recorded as
`not provisioned: awaiting explicit owner approval`. Where this section differs from §5, §6.4
or §7, this section governs for Stage C1. The runnable form is
[experiments/E51/README.md](../experiments/E51/README.md).

| Point | Stage C1 instruction | Replaces |
| --- | --- | --- |
| Interpreter | The existing Python 3.10.19 interpreter. Every environment is a `venv` of it; nothing is copied. Each launcher's path, SHA-256, version and base interpreter are recorded | §5.2 item 1, F1 |
| Isolation | Airplane mode, switched on and off by the owner by hand. No script changes a network setting; the runner only reads adapter and route state | §5.2, §5.3 |
| Launchers | **All five base-cell environments are created during connected provisioning, empty**, and their paths do not change afterwards. Nothing is installed into them while connected | §7.6, fourth rule |
| Positive control | While connected: a TCP connection to the literal addresses that `pypi.org` and `files.pythonhosted.org` resolve to, port 443, through **each of the five launchers and the base interpreter**, with `-I -S -B` so no installed code runs | §7.6, fourth rule ("one launcher") |
| Negative control | Offline: the same test, same addresses, same six executables, at the **start** of the window and again at the **end**. Any success stops the run | §5.3 |
| Boundary | *Connected*: resolve, download, verify, build, static inspection, environment creation, positive controls, input export. *Offline*: every install, every post-install gate, every import of `fie`, every probe, C9 hashing. *Connected again*: recovery check, comparison, report | §7.3 |
| Mode A cell | Run twice in two fresh environments (`r1`, `r2`) so that "installs reproducibly" is observed, not assumed. One cell, two repetitions | §6.4 |

**Why the earlier conflict is resolved this way.** O-2 asks for a positive control through each
measurement executable; §7.6 proposed creating environments inside the offline window and
controlling only one launcher. An environment that does not exist while the network is up
cannot have a positive control. Base environments are a few megabytes each, so creating all
five beforehand costs nothing and removes the conflict. The sequential create-and-remove rule
of §7.6 remains the rule for `[ml]` environments, which are outside Stage C1.

**Limits that still apply** (§5.4): airplane mode is machine-level and is evidenced here by
adapter state, the absence of a default route, and TCP failure to two previously reachable
addresses through six executables. That supports "these processes could not open those
connections, and no network interface was up". It is not a packet capture and does not by
itself prove that nothing left the machine by any channel.

---

## 1. Problem statement and evidence

### 1.1 The question

Every measurement in this rebuild so far was made on the **working tree**, in the developer's
conda environment, with models present. `BL-0002`'s base-install profile is a **simulation**:
the working tree's code with nine ML packages made unimportable. Nobody has yet installed the
package the way a user does and looked at what it does.

- With the classifier missing, the base-install simulation allows 1,278 of 1,348 attacks.
- The default policy, `compat@1`, still allows such a scan (with a reason code).
- Whether a real install lands in that state, with or without `[ml]`, is unknown.

### 1.2 What the repository already shows

| # | Finding | Label |
| --- | --- | --- |
| E-1 | **Three different artifacts carry the version `1.18.0`.** PyPI's wheel was 270,149 bytes when its metadata was saved on 2026-10-07 (uploaded 2026-08-10). The wheel in the repository's `dist/` is 313,989 bytes. The current tree would build a third: the `dist/` wheel has no `fie/layers/`, `fie/onnx_encoder.py` or `fie/policy.py`. `pyproject.toml` still says `1.18.0` | INSPECTED / READ |
| E-2 | **`[ml]` means different things on PyPI and in the repository.** PyPI 1.18.0: `joblib`, `pandas`, `scikit-learn`, `sentence-transformers`, `xgboost`. HEAD: `scikit-learn`, `xgboost`, `joblib`, `pandas`, `onnxruntime`, `tokenizers` | INSPECTED / READ |
| E-3 | **Model files are not in git.** `fie/models/` tracks only `.gitkeep`. `[tool.hatch.build] artifacts` forces `fie/models/*.pkl` and `*.json` into a wheel if they exist on the builder's disk | READ |
| E-4 | A wheel built from a clean checkout contains no model. A wheel built from this working tree would contain 33 files from `fie/models/` | READ / PREDICTED |
| E-5 | **The `dist/` wheel contains pickles that the manifest does not list**: `pair_intent_classifier_v2.pkl`, `…_v3_w3x.pkl`, `…_v6_1_contaminated.pkl`, `…_v6_noaug.pkl`. `scripts/model_manifest.json` lists `v6_3b`, `v6_3`, `v6`, `v5`, `v4`, `v3`, the original, `meta_clf.pkl` and two `failure_classifier` files. The PyPI wheel's model content is unknown until it is inspected | INSPECTED / TO MEASURE |
| E-6 | The loader picks the first classifier file that exists from a fixed preference list, `v6_3b` first. Which file a given artifact loads by default therefore depends on what that artifact contains, and on that artifact's own copy of the list | READ |
| E-7 | **The encoder is never in the wheel.** `model.onnx` (90 MB) and its `tokenizer.json` are ignored and not in `artifacts`. On first use `fie/onnx_encoder.py` tries to download both from the GitHub release `models-v1.18.0` unless `FIE_NO_AUTO_DOWNLOAD` is set | READ |
| E-8 | If that download fails, the loader falls back to `sentence-transformers`, which HEAD's `[ml]` no longer installs; the classifier then reports `unavailable_dependency` | PREDICTED |
| E-9 | `import fie` starts a background thread that sends a telemetry request unless `FIE_NO_TELEMETRY` is set | READ |
| E-10 | A non-English prompt can be sent to an online translator; an uncertain-band prompt is sent to a remote tiebreaker when `GROQ_API_KEY` is set | READ |
| E-11 | Dependencies are lower bounds only. The v6.2 classifier's metadata records scikit-learn 1.7.2; the shipped v6.3b metadata records no version | READ / TO MEASURE |
| E-12 | `requires-python = ">=3.9"`. Everything so far ran on 3.10.19 only | READ / TO MEASURE |
| E-13 | The harness cannot measure an installed copy: `evals/subject.py` refuses to proceed unless `fie` was imported from the working tree. E51 needs its own probe | READ |
| E-14 | `publish-pypi.yml` runs `scripts/download_models.py --strict` before building; CI's build job runs it without `--strict` | READ |
| E-16 | **`fie-sdk` cannot be installed from wheels alone.** Its unconditional dependency `langdetect>=1.0.9` is published on PyPI as a source archive and a `py2`-tagged wheel only. Every Python 3 install of `fie-sdk`, with or without `[ml]`, builds `langdetect` from source at install time, which executes its `setup.py` and needs a build tool (`setuptools`) to be present or fetched | **MEASURED** in Stage B (`stageB/resolve/base.pip.log`) |
| E-15 | In a base install no pickle can be loaded: the loader imports `joblib` before it opens any model file, and `joblib` is not a base dependency | READ |

### 1.3 Why a measurement before a fix

The predictions are unverified, and several depend on third-party packages at install time.
The right fix depends on which state real users are in, which only the published artifact can
show. And a packaging change alters what `pip install fie-sdk` does for every user, which is a
release decision the owner has not been asked to make.

Relevant standing decisions: **D-006** (no network egress and no non-determinism by default)
is the principle the isolation design in §5 has to honour for the measurement itself;
**D-008** (no pickle in the production package) is why loading a model file is treated here as
executing code; **D-013** ("clean tree" means clean measured paths) is why the experiment code
lives outside the measured paths.

---

## 2. Hypotheses and research questions

All hypotheses are PREDICTED. The last column says which measurement mode (§4) can test each;
a hypothesis is only as strong as the mode that tested it.

| # | Hypothesis | Tested in |
| --- | --- | --- |
| H1 | A base install imports, scans, and runs every layer except the classifier, which is `unavailable_dependency`. Its verdicts equal `BL-0002`'s `lite.std` records | S for behaviour; PC for the record comparison |
| H2 | The PyPI 1.18.0 wheel predates WP-003: its results have no `coverage`, `zone` or `models`, and there is no `fie.policy`. It cannot report that it is degraded | S |
| H3 | `pip install "fie-sdk[ml]"` from PyPI pulls `torch` through `sentence-transformers` | Provisioning dry run only; no execution needed |
| H4 | A wheel built from a clean checkout of HEAD has no models; with `[ml]` its classifier is `unavailable_model` | S |
| H5 | A release-procedure wheel from HEAD, installed with `[ml]`, has the classifier file and no encoder. With egress denied the classifier is `unavailable_dependency`; it would reach full coverage only by downloading from GitHub on first scan | S (the download itself is never allowed, so "would reach full coverage" stays PREDICTED) |
| H6 | `import fie` attempts a network connection in every cell unless telemetry is disabled | S only. P and PC disable or alter this and cannot test it |
| H7 | With the newest scikit-learn the manifest-listed classifier loads with a version warning, and its scores may differ from `BL-0002` | P and PC |
| H8 | Under the default `compat@1`, every degraded cell allows what its remaining layers allow | S for the default; replay for the other presets |
| H10 | *(rev. 4)* Installing the published wheel with pip succeeds offline from a verified wheelhouse when `langdetect` is built from its source archive, and pip takes a source-build path for it (which path depends on the pip version) | Install mode A (§6.4). Not a runtime mode |
| H11 | *(rev. 4)* The `langdetect` files installed by the source-build path are byte-identical to those installed from the pre-built local wheel, so runtime results obtained with the pre-built wheel hold for a source-built install | Comparison C9 |
| H9 | *(new)* The default loader choice in the PyPI wheel is a manifest-listed file | Static inspection, before anything is imported. If false, the `[ml]` S cell for that artifact is **blocked** (§4.3) |

**Research questions**

| # | Question |
| --- | --- |
| RQ1 | Per cell: does it install, and what exactly is installed (names, versions, hashes, size)? |
| RQ2 | Does `import fie` succeed? What does it import, how long does it take, what does it try to reach? |
| RQ3 | What is in each wheel: modules, model files with hashes and manifest status, metadata? |
| RQ4 | Per cell: state of the twelve layers, classifier, meta-classifier, encoder; `coverage.status`; `degraded_layers`; `models` |
| RQ5 | Per cell: verdicts on the pinned prompts, compared with `BL-0002` |
| RQ6 | Per cell: what `decide` returns under `compat@1`, `strict@1`, `review@1` (replayed; no default is changed) |
| RQ7 | What destinations are attempted at import, warm-up, first scan and over the suite? |
| RQ8 | Does the `fie` console script work? |
| RQ9 | What warnings and exceptions occur, including pickle version skew? |
| RQ10 | How faithful was `BL-0002`'s base-install *simulation* to a real base install? |
| RQ11 | Do the artifacts named `1.18.0` differ in behaviour? |
| RQ12 | *(new)* Is `1.18.0` the current published release? |
| RQ13 | *(new)* Which model file does each artifact load by default, and is it a verified one? |
| RQ14 | *(rev. 4)* What does pip actually do to install the published wheel's dependencies: which are wheels, which are built, by which mechanism, needing what at install time? |

---

## 3. Scope

**In scope:** building wheels in a scratch directory; creating disposable virtual environments
outside the repository; installing into them; running a read-only probe; if approved, adding
and removing named outbound firewall rules for the measurement interpreter (§5); writing
evidence and a report under `docs/fie_rebuild_2026/`.

**Out of scope, explicitly:** any change to `fie/`, `evals/`, `app/`, `engine/`, `storage/`,
tests, `pyproject.toml`, `requirements.txt`, model files, `scripts/model_manifest.json`,
workflows, Dockerfiles, `README.md`, `CHANGELOG.md`; any change to the default policy
(`compat@1` stays the default; `strict@1` is not selected anywhere, including by implication);
any publication or deployment; `CANONICAL`, every baseline, and any `BL-0003`; deleting or
replacing the stale wheel in `dist/`; staging, committing, pushing, merging or changing branch.

### 3.1 File matrix

Nothing existing is modified, except the master log at execution.

| Path | Status | Purpose |
| --- | --- | --- |
| `docs/fie_rebuild_2026/experiments/E51/README.md` | new | How to re-run; exact commands; the two elevated commands of §5 |
| `…/E51/probe.py` | new | Runs inside a clean environment. Standard library only. Imports the **installed** `fie` and records everything in §7 |
| `…/E51/observe.py` | new | Process-local instrumentation: socket/DNS recorder, pickle tripwire, subprocess recorder (§4.4, §5.2) |
| `…/E51/egress_selftest.py` | new | The independent control tests of §5.3, run through every measurement executable. Standard library only; imports nothing from `fie`, `observe.py` or any installed package |
| `…/E51/exe_inventory.py` | new | Lists every executable under the scratch directory with path and SHA-256; compares two inventories (§5.2) |
| `…/E51/provision.py` | new | The provisioning procedure of §5.6: configuration checks, explicit-source downloads, host and hash verification |
| `…/E51/e51_block.ps1`, `e51_unblock.ps1` | new | The two owner-run, elevated scripts (O-14). They add and remove rules for the paths in the inventory and nothing else |
| `…/E51/inspect_wheel.py` | new | Static inspection of a wheel: file list, hashes, manifest status, metadata, loader preference list read from the source text. Imports nothing from the wheel |
| `…/E51/run_matrix.py` | new | Run from the repository environment: builds wheels, creates environments, installs, invokes the probe, collects results |
| `…/E51/export_inputs.py` | new | Writes the pinned prompts to one JSONL with its SHA-256 |
| `…/E51/compare.py` | new | Compares probe records with `BL-0002` and between modes |
| `docs/fie_rebuild_2026/evidence/E51/<run-id>/…` | new | Results, `SUMMARY.json`, `MANIFEST.sha256` |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_020_install-truth.md` | new | The report |
| `MASTER_DECISIONS_AND_IMPLEMENTATION_LOG.md` | modified at execution | Session rows and a new section |

The experiment code lives under `docs/` because the harness treats `fie/`, `evals/` and other
code paths as measured (D-013); files added there change run identities (O-6).

Outside the repository, deleted at the end: scratch export, built wheels, wheel cache, the
private interpreter copy, virtual environments. Machine state touched outside the scratch
directory: the firewall rules of §5, removed at the end and verified removed.

---

## 4. Measurement modes

Revision 1 called a run "as-shipped" while proposing to force `FIE_PAIR_VERSION` in one cell.
That was wrong: an override that changes which model loads is not unmodified behaviour. The
modes are now separate, and every evidence file names its mode.

### 4.1 The three modes

| | **S — as-shipped configuration** | **P — manifest-pinned** | **PC — manifest-pinned, comparable** |
| --- | --- | --- | --- |
| Purpose | What does the artifact do when a user simply installs and calls it? | The same call with the model choice restricted to verified files | Verdicts that can be compared with `BL-0002` |
| Environment variables set for `fie` | **None.** No `FIE_*` variable at all | `FIE_PAIR_VERSION` = the manifest-listed default; `FIE_NO_AUTO_DOWNLOAD=1` | As P, plus `FIE_NO_TELEMETRY=1` |
| In-process changes to `fie` | **None** | None | The canonical profile's deviations from `evals/registry/profiles.json`: translator returns nothing, language detector seeded, result cache cleared per suite |
| Call | `scan_prompt(prompt)`, `preflight_check(prompt)`, CLI, with defaults | Same | `scan_prompt(prompt, use_llama_guard=False)`, as the harness calls it |
| Model selection | The artifact's own default | Forced | Forced |
| Encoder | As installed. Never pre-staged | As installed, and a second variant with the manifest-verified encoder files copied in | Both variants |
| Inputs | 8 reference prompts + 40-prompt smoke set | Same as S | 2,016 standard + 1,848 base-install |
| May be described as | "as-shipped configuration, under isolation and observation" | "manifest-pinned" | "manifest-pinned, harness-comparable" |
| May **not** be described as | "completely unmodified" (it runs with egress denied and with instrumentation present) | "as-shipped", "default", "unmodified" | "as-shipped", "default", "unmodified" |

**What S still changes, stated plainly.** Mode S sets no `fie` variable and patches nothing in
`fie`. Three things around it are not what a user has: outbound connections are denied (§5), a
recorder and a tripwire are loaded in the process (§4.4), and the process environment is a
minimal allowlist with a scratch home directory. So S measures the artifact's behaviour
*when its network calls fail*, which is one real user condition (firewalled, offline) and not
the only one. What a default install does when its downloads succeed is not measured by this
experiment and remains PREDICTED.

**Differences between modes are results.** S versus P on the same inputs isolates the effect
of the pin. P versus PC isolates the harness's knobs. Neither difference is ever folded into a
single "the package does X" statement.

### 4.2 Verified and unverified model files

- **Verified** means: the file's SHA-256 equals the entry for that path in
  `scripts/model_manifest.json` at `2f62ecd`.
- Every `.pkl`, `.onnx` and tokenizer file in every artifact is listed and hashed by
  `inspect_wheel.py` **from the wheel archive, before installation and without importing
  anything**. Its manifest status is recorded.
- **An unverified pickle or model file is never loaded**, in any mode, unless the owner
  approves that specific file (O-12). Listing and hashing is not loading.

### 4.3 When mode S is blocked

For each `[ml]` cell, before any import, `inspect_wheel.py` determines the artifact's
**default load set**: the files the default scan path would open. It does so by reading the
loader's preference list from that artifact's own source text and applying it to that
artifact's own file list. No code from the artifact runs.

| Default load set | Mode S for that cell |
| --- | --- |
| Every file verified | Runs |
| Any file unverified, or the preference list cannot be determined statically | **BLOCKED.** Not run, not approximated |
| Base install (no `joblib`) | Runs: no pickle can be loaded (E-15); the tripwire confirms it |

A blocked S cell is reported with: the file that blocked it, its hash, and **the questions
left unanswered for that artifact**: its real default verdicts (RQ5), its default coverage and
self-report (RQ4), its default-mode warnings (RQ9), and H5/H8 for that cell. Modes P and PC may
still run for that cell if a verified file can be forced; their results answer "what does this
artifact do with the verified model", never "what does this artifact do by default".

### 4.4 The pickle tripwire

In every mode, `observe.py` installs a Python audit hook before `fie` is imported. On any
attempt to open a file ending in `.pkl`, `.joblib` or `.pickle` whose resolved path and
SHA-256 are not on the verified list, it records the path (relative to the environment) and
**terminates the probe** before the bytes are read by an unpickler.

This is instrumentation, not enforcement by the operating system, and it is a backstop behind
the static check of §4.3, not a replacement. If it fires, the cell is recorded as blocked and
the static inspection is recorded as having been wrong, which is itself a finding.

---

## 5. Network isolation

Revision 1 treated a Python-level socket guard as the isolation. It is not. This section
separates the two and says what may be claimed under each.

### 5.1 Two different things

| | Process-local instrumentation | OS-enforced egress denial |
| --- | --- | --- |
| What it is | Python wrappers around `socket.connect`, `getaddrinfo` and related calls, plus an audit hook, inside the probe process | A rule in the operating system's packet filter that drops outbound traffic from the measurement interpreter |
| What it gives | **Attribution**: which module tried to reach which host, at which phase | **Assurance**: the packet does not leave, whoever sent it |
| What it cannot do | See or stop traffic from native code that does not go through Python's `socket` module (e.g. a C/C++ extension opening its own connection); stop anything in another process | Say which line of code tried |
| Claim it supports | "These attempts were observed at the Python level" | "Outbound connections from the verified measurement executables were denied", within the limits of §5.4. It does **not** support "no data could leave the machine": see DNS, §5.4 |

Both are used. **No statement about machine-level egress is ever based on the instrumentation
alone.**

### 5.2 The proposed OS-level mechanism (Windows 11 Home)

Options considered:

| Option | Verdict |
| --- | --- |
| Windows Sandbox or a Hyper-V VM with networking disabled | Not available on Windows 11 Home |
| WSL2 or Docker with no network | Would measure Linux, not this platform; availability unconfirmed. A useful later addition, not this experiment (O-8) |
| Disabling the network adapter for the measure phase | Strongest and simplest, and it cuts off everything, including the assistant session that would be driving the run. Feasible only as an owner-run, unattended script (Fallback A, §5.5) |
| Setting the firewall's default outbound action to block | Same effect and same problem as above |
| **Windows Defender Firewall outbound block rules scoped to the measurement interpreter's executables** | **Proposed.** Targeted, reversible, built into this edition. Needs administrator rights |

**Design.**

1. **A private interpreter.** On Windows a virtual environment's `python.exe` is a launcher;
   the process that actually opens sockets is the *base* interpreter it starts. Blocking the
   developer's own interpreter would be disruptive and imprecise. So the experiment first makes
   a private copy of an existing local Python 3.10 installation inside the scratch directory
   (a file copy; no download) and creates every measurement environment from that copy.
   Provisioning (`pip download`) is done with a *different* interpreter, so the block never
   has to be lifted for it.
2. **Executable inventory and rules.** After every environment is installed and before any
   rule is added, `exe_inventory.py` lists **every executable file under the scratch
   directory** with its full resolved path and SHA-256: the private interpreter's
   executables, each environment's launchers in `Scripts\`, and every console-script
   launcher that dependencies installed. The owner's script adds one outbound **Block** rule,
   all profiles, all protocols, **per inventory path**, each named `FIE-E51-…`. Block rules
   take precedence over allow rules in Windows Defender Firewall. The inventory is taken
   again after the measure phase; a new or changed executable is a violation.
   A rule existing for a path is not treated as proof that the path is isolated. Proof is the
   control test of §5.3, run through that path.
3. **Who adds them.** Adding firewall rules needs elevation, and the assistant session cannot
   elevate non-interactively. The plan therefore provides two short PowerShell scripts,
   `e51_block.ps1` and `e51_unblock.ps1`, that **the owner runs in an elevated prompt**
   (O-14). The rules stay in place for the whole measure phase and are removed after it.
4. **Measurement executables and the expected process tree.** A *measurement executable* is
   any executable path through which a probe is started. The plan fixes this set, per cell,
   to exactly one entry path: that environment's `Scripts\python.exe`. Starting it creates
   one expected child, the private base interpreter, which is the process image that actually
   runs Python and opens sockets. Both paths are verified (§5.3).
   **Console-script launchers are not measurement executables under the firewall design.**
   `fie.exe` cannot be made to run an independent control test, so its isolation cannot be
   verified rather than inferred. CLI behaviour is therefore measured through
   `Scripts\python.exe -m fie` (a verified path); `fie.exe` is inspected statically (present,
   entry point, hash) and is **executed only under Fallback A** (network off). Under the
   firewall design RQ8 is answered for the module entry point and recorded as not measured
   for the launcher.
   Any process whose image path is not one of the two verified paths for the running cell is
   a violation: the probe records process creation through the audit hook and stops. `fie` is
   not expected to spawn any.

### 5.3 Independent verification, per executable

The block is verified by `egress_selftest.py`: standard library only, no import of `fie`, of
the instrumentation, or of any installed package. It is run with `-I -S -B`, so that no
`site-packages` code and no `.pth` file of an installed distribution executes during the test.

**It is run through every measurement executable, not once.** The private interpreter passing
proves nothing about a cell: a cell is entered through its own launcher path, and the firewall
matches rules by program path. So for each cell the test is started exactly the way the probe
will be started, via that environment's `Scripts\python.exe`, and it records:

- `sys.executable`, and the **actual process image path** obtained from the operating system
  (not from Python's own idea of it), and the parent process's image path;
- that both paths are in the inventory and that a `FIE-E51-*` rule names each;
- the outcome of the connection attempts below.

It is also run directly through the private base interpreter. A cell is **verified** only when
its launcher path and the base interpreter both pass.

| # | Check | When | Run through | Expected |
| --- | --- | --- | --- | --- |
| V1 | **Positive control.** TCP connection to a PyPI address on port 443, by literal IP. Connect and close; nothing is sent | After all environments are installed, before any rule exists | **Each cell's launcher**, and the base interpreter | Succeeds for every path. Proves that this path can reach the network and that the test detects it. A path whose V1 does not succeed has no control and cannot be verified |
| V2 | **Negative control.** The same connection, and a second TCP connection to a different PyPI address | After the rules are added, before the first measurement | **Each cell's launcher**, and the base interpreter | Fails for every path, with a permission or unreachable error within a short timeout |
| V3 | Rule state: `Get-NetFirewallRule -DisplayName "FIE-E51-*"` with each rule's program path, joined to the inventory; firewall profiles' enabled state. Exported to the evidence directory | Before and after | — | One enabled Block rule per inventory path; no inventory path without a rule; firewall enabled on the active profile |
| V4 | V2 repeated | After the last measurement | Each cell's launcher, and the base interpreter | Still fails for every path |
| V5 | After `e51_unblock.ps1`: no `FIE-E51-*` rule remains and V1 succeeds again | Cleanup | Base interpreter | Machine restored |
| V6 | Inventory comparison | After the last measurement | — | Identical to the pre-measurement inventory: no executable appeared or changed |

V1 is the one moment at which measurement executables touch the network: a bare TCP handshake
to PyPI, with no `fie` code loaded and no payload. It is necessary, because a denial with no
prior success is not evidence.

**UDP.** A UDP send to a blocked destination generally returns success to the caller on
Windows even when the packet is dropped, so the control test cannot verify UDP denial by
itself. With dropped-packet logging (O-15) the firewall log is the evidence. Without it, UDP
denial is reported as **configured, not independently verified**.

**Per-cell consequence.** A cell whose V1 fails, whose V2 succeeds, or whose process image is
not as expected is **not measured** under the firewall design. It is recorded as
"isolation not verified" and either left unmeasured or measured under a fallback the owner has
approved (§5.5), with that fallback's weaker claim. No cell inherits another path's result.

If the owner approves it (O-15), dropped-packet logging is enabled for the duration and the
log's entries for the inventory paths are saved as a second, independent record.

**Preconditions checked first; any failure stops the experiment before measurement:** Windows
Defender Firewall is the active filter and is enabled; no third-party security product has
taken over filtering; the private interpreter copy runs; the inventory is complete; V1
succeeds and V2 fails for the base interpreter.

### 5.4 What can and cannot be claimed

With §5.2 in place and V1 to V6 passing for a cell, the report may say, for that cell:

> During the measure phase, outbound TCP connections from the measurement executables of this
> cell were denied by an operating-system firewall rule, verified by a control test run
> through those executables before and after.

It may **not** say "nothing left the machine", "no data could leak" or "no prompt left the
machine". The limits, each stated in the report:

| Limit | Detail |
| --- | --- |
| **DNS is an open data channel under this design** | On Windows a name lookup is normally carried out by the system DNS client service, in a separate process that a per-program rule does not cover. Any code in the measured process can ask the resolver to look up **any name it chooses**, and the name itself can carry data: text encoded into the labels of a query reaches whichever server is authoritative for that domain. So DNS can carry prompt text, file contents or anything else the process can read; it is not limited to revealing which hosts `fie` contacts. The Python-level instrumentation intercepts lookups made through Python's `socket` module only. **Native code** (a compiled extension in a dependency, or a library it loads) can call the resolver directly and is neither seen nor stopped by the instrumentation, and is not stopped by the per-program rule either |
| What the process can read | The probe runs as the logged-in user. Using public benchmark prompts and an allowlisted environment limits what **the experiment hands to** the measured code. It does not limit what that code could read from disk and send through an open channel |
| Other processes | A rule covers the named executable only. A process running any other image is not covered; process creation is recorded and treated as a violation, by instrumentation |
| Code that runs before the instrumentation | `.pth` files and `sitecustomize` hooks of installed distributions execute at interpreter start, before the probe loads its recorder. They are listed by static inspection (§7.4); what they do at run time is constrained by the firewall rule and not by the instrumentation |
| UDP | Denial is configured; it is independently verified only with O-15 (§5.3) |
| Proxies and loopback | Traffic to a proxy on this machine is not outbound and would be forwarded by the proxy process. Preflight records system proxy settings; the probe's environment carries no proxy variable |
| Scope of verification | V2 and V4 test specific destinations through specific paths. They establish that the rules take effect for those paths; they do not enumerate every destination |
| One machine | The claim is about this run on this machine |

**Reduced risk is not a guarantee.** Three things lower the *consequence* of a leak under the
firewall design: every input is public benchmark text; the process environment holds no
credential; its home directory is a scratch directory. None of them makes a leak impossible,
and the report must not present them as isolation. The accurate summary is: *connections are
denied and verified; one data-capable channel, DNS, remains open and is watched only at the
Python level; the data the experiment supplies is public.*

**When the owner-run, network-off procedure (Fallback A) is the preferred option, not a
fallback:**

| Situation | Why |
| --- | --- |
| A statement that no data could leave the machine is wanted | Only removing the network supports it. The firewall design cannot, because of DNS |
| The cell executes code that is not from this repository at a reviewed commit: both **A-pypi** cells, and above all the PyPI `[ml]` cell with its `torch` dependency stack | Unreviewed native code is exactly what can use the channel the instrumentation cannot see |
| An unverified model file is approved for loading under O-12 | Loading it executes arbitrary code |
| The console-script launcher `fie.exe` is to be executed | It is not a verifiable measurement executable (§5.2) |
| Executable-level controls cannot be verified for a cell | §5.3 |
| A third-party filter is in control, or the firewall preconditions fail | The rules may not be the ones enforced |

The recommendation (O-18) is therefore split: the firewall design for cells that run only
repository code at `2f62ecd` with verified model files (A-clean, A-release), and network-off
for the A-pypi cells. The owner may equally choose network-off for the whole measure phase,
which is simpler to reason about and costs only that the run is unattended.

### 5.5 If the OS-level block cannot be established

If the owner declines O-14, a precondition of §5.3 fails, or a cell's executable-level
controls cannot be verified, measurement of the affected cells **stops**. It resumes only in
one of the two forms below, and only if the owner has approved that form (O-16). **The acceptance criterion is weakened
accordingly and the report says so in its first table.**

| | What happens | Claim allowed |
| --- | --- | --- |
| **Fallback A — owner-run, network off** | The owner disables every network adapter (or uses airplane mode) and runs one unattended script that performs the measure phase for the chosen cells; the assistant is not connected while it runs and reads the evidence afterwards. The script first runs the negative control through each cell's launcher and through the base interpreter, records the adapter state and that no interface has a route, and repeats both at the end. V1 (connectivity before) is run while the network is still up | The strongest available here: no interface, so no TCP, UDP or DNS. It is the only form under which DNS is closed. The report records the adapter state the script observed, and the owner's action as the owner's statement. It is preferred, not merely permitted, in the situations listed in §5.4 |
| **Fallback B — instrumentation only** | The measure phase runs with the process-local recorder as the only barrier | Only: "no egress was observed by process-local instrumentation". **Machine-level egress is not excluded.** Mode S is then restricted to the 8 reference prompts and the 40-prompt smoke set, all of which are public benchmark text, with no credential in the environment. That lowers the consequence of a leak of what the experiment supplies; it is not a bound on what could leave, for the reasons in §5.4. This form is not recommended for any A-pypi cell |

No form of the experiment asserts zero machine-level egress on the strength of
instrumentation.

### 5.6 Provisioning: separate, PyPI-only, and shown to be so

| Phase | Interpreter | Network | Activity |
| --- | --- | --- | --- |
| **Provision** | The existing repository interpreter | **PyPI only**: `pypi.org` and `files.pythonhosted.org`, subject to O-2 | Read the package index; download wheels into a local cache; record names, versions, files, SHA-256. Nothing downloaded is executed |
| **Measure** | The private interpreter | **Denied** (§5.2) or absent (Fallback A) | Install from the cache with `--no-index`; run the probe |

"PyPI only" is a claim about where bytes came from. `pip` can be redirected by configuration
the user never sees on the command line: a per-user or global config file, `PIP_*` variables,
an extra index, a mirror, a `--find-links` location, a proxy from the environment or from
Windows settings, a local download cache. So the claim is established by procedure, not by
intention. `provision.py` does the following, and stops at the first failure.

**1. Record what is inherited (before any download).**

- `pip config debug` and `pip config list`: every config file location, whether it exists,
  and every setting in effect. Saved.
- Every `PIP_*`, `*_PROXY`, `NO_PROXY`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`,
  `CURL_CA_BUNDLE` variable in the shell: names recorded, values recorded only as
  "set / not set".
- The system proxy: what Python's standard library reports, and the Windows proxy settings.
- Entries in the `hosts` file for the two PyPI host names.
- `pip --version` and the interpreter's platform tags.

**2. Neutralize it.** Every `pip` invocation in provisioning runs:

- in a **constructed environment** containing none of the variables above, with
  `PIP_CONFIG_FILE` pointing at the null device;
- with `--isolated`, which makes pip ignore environment variables and per-user configuration;
- with `--index-url https://pypi.org/simple` **given explicitly on every command**, and with
  no `--extra-index-url`, `--find-links`, `--trusted-host`, `--cert` or `--proxy`;
- with `--no-cache-dir`, so that nothing is served from a pre-existing local cache of unknown
  origin;
- with `--only-binary :all:`, so that no source distribution is fetched or built;
- with `--disable-pip-version-check`, which removes pip's own call home;
- with `-vv`, the full log saved.

The index read of §6.1 and the digest look-ups below use a small standard-library client with
an explicit `https://pypi.org/…` URL and proxy handling switched off, so they do not depend on
pip's configuration at all.

**3. Refuse conditions that cannot be neutralized.**

| Found | Action |
| --- | --- |
| A system-level proxy is configured (Windows settings or a PAC file) | **Stop.** Traffic would pass through the proxy and "PyPI only" could not be shown. Resume only with an owner decision (O-19) |
| A `hosts` entry redirects either PyPI host name | **Stop** |
| A global pip config file sets an index, extra index, find-links, trusted host or proxy, and `--isolated` plus the null config file do not remove it from `pip config debug`'s effective output | **Stop** |
| TLS verification would have to be disabled for a download to succeed | **Stop** |

**4. Resolve, then download exactly what was resolved.**

- *Resolve*: `pip install --dry-run --ignore-installed --report <file>` for each cell's
  requirement, with the flags above. Nothing is installed. The report lists every
  distribution with its download URL and SHA-256.
- *Check the resolution*: every URL's host must be `files.pythonhosted.org`. Any other host
  stops provisioning.
- *Pin*: write a requirements file from the report, one line per distribution, exact version
  and `--hash=sha256:…`.
- *Download*: `pip download --require-hashes -r <that file>` with the same flags. pip then
  refuses any file whose hash differs from the resolved one.

**5. Verify afterwards, independently of pip.**

| Check | Evidence |
| --- | --- |
| Every host name in the `-vv` logs is one of the two approved hosts | Parsed list of contacted hosts, saved; any other host fails provisioning |
| Every file in the cache has the SHA-256 recorded in the resolution report | Recomputed by `provision.py` |
| Every such SHA-256 equals the digest PyPI's JSON API publishes for that file | Fetched by the standard-library client; a file PyPI does not list fails provisioning |
| The wheels' platform and Python tags fit the private interpreter | Compared with the tags recorded in step 1 |
| The set of files in the cache equals the set in the pinned requirements file | No extra file |

**6. Reproduce.** The pinned, hashed requirements file is the provisioning record. A repeat
run uses it as input and must produce a byte-identical cache. The measure phase installs with
`--no-index --find-links <cache> --require-hashes -r <same file>`, so an environment can only
contain files that passed step 5.

**What this establishes, exactly.** That pip was given one index explicitly, that inherited
configuration was recorded and removed from effect, that every host named in its logs was a
PyPI host, and that every downloaded file is byte-identical to a file PyPI publishes under
that name and version. It does not establish the network path between this machine and PyPI;
a transparent interception that passed TLS validation would be invisible here and would still
have to deliver PyPI's exact bytes to pass the digest check.

Provisioning never contacts GitHub, the Hugging Face hub, a conda channel or any other host.
**If provisioning cannot be shown to have used only the approved PyPI source, it stops and the
report states which check failed. No environment is then built from that cache.**

Wheel builds (§6.2) use `--no-isolation` and contact nothing.

---

## 6. Artifacts and matrix

### 6.1 The current published release is checked first

Revision 1 assumed `1.18.0` is what users have. That rests on metadata saved on 2026-10-07.

**First action of provisioning**, before any artifact is downloaded:

1. Read the PyPI JSON metadata for `fie-sdk` (PyPI only). Save it.
2. Record: the latest version; every release with its upload time; yanked flags; and for
   `1.18.0`, each file's name, size and SHA-256.
3. Compare with the saved metadata: is `1.18.0` still present, unchanged (270,149 bytes), and
   still the latest?

| Outcome | Action |
| --- | --- |
| Latest is `1.18.0`, digests unchanged | Proceed. A-pypi may be described as "the current published release as of the date of the check" |
| **Latest is not `1.18.0`** | **Stop before downloading anything else and ask the owner** (O-17) whether to add the newer release to the matrix. `1.18.0` is then labelled "pinned historical release" everywhere. No version is substituted silently, and the older wheel is not described as current |
| `1.18.0` missing, yanked, or its digest differs from the saved record | Stop and report |

The wheel is then downloaded by exact version with `--no-deps` and its SHA-256 checked against
the digest PyPI publishes.

### 6.2 Artifacts under test

| ID | Artifact | Represents | Obtained by |
| --- | --- | --- | --- |
| **A-pypi-1.18.0** | The PyPI wheel, exact version | The published release (current or historical, per §6.1) | One download from PyPI, digest-checked |
| **A-clean** | Wheel built from `git archive 2f62ecd` | What a clean checkout builds | `python -m build --wheel --no-isolation` in a scratch export; no models present |
| **A-release** | Wheel built from the same export with the model directory populated | What the next release would ship if cut today | Same, with **only verified files** (§4.2) copied from the working tree; the offline equivalent of `download_models.py --strict` |
| A-dist | The stale `dist/` wheel | An unpublished local build | **Static inspection only.** Not installed, not loaded, not deleted |
| *(A-pypi-latest)* | A newer PyPI release, if one exists | The current published release | Only with O-17 |

### 6.3 Matrix

Python 3.10 only (see O-5: other versions would need an interpreter from a source other than
PyPI).

| Cell | Artifact | Extra | Mode S | Mode P | Mode PC | Encoder variants (P, PC) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A-pypi-1.18.0 | base | run | n/a: no model can load, nothing to pin | run | — |
| 2 | A-pypi-1.18.0 | `[ml]` | **run only if §4.3 allows; otherwise BLOCKED** | run if a verified file can be forced | same | as installed. Pre-staging does not apply: this artifact's encoder is the `torch` path (O-4) |
| 3 | A-clean | base | run | n/a | run | — |
| 4 | A-clean | `[ml]` | run (contains no pickle) | n/a: nothing to pin | run | as installed |
| 5 | A-release | base | run | n/a | run | — |
| 6 | A-release | `[ml]` | run (every model file verified by construction) | run | run | as installed; pre-staged |

Every cell above is provisioned in **install mode B** (pre-built local `langdetect` wheel) and is
labelled so. Install mode A is exercised in the separate install-path cells of §6.4. Environments
are created, measured and removed one at a time (§7.6).

"n/a" cells are recorded as not applicable with the reason, not silently omitted. In every
cell the CLI is exercised through `Scripts\python.exe -m fie`; the `fie.exe` launcher is run
only under Fallback A (§5.2). The recommended isolation form differs by artifact: see O-18. Cell 2 also
depends on O-4: the full PyPI `[ml]` dependency set is first resolved by a dry run that
installs nothing, and installed only if the owner approves its size.

### 6.4 Installation modes: source-build path versus pre-built wheel

Stage B found that `fie-sdk` cannot be installed from wheels alone (E-16). The remediation
built a `langdetect` wheel so that resolution could proceed. That wheel carries the same
package files as the source archive, but **installing it is not what a user's pip does**. The
two cases are different experiments and are kept apart. Like the measurement modes of §4, the
install mode is recorded on every environment and every evidence file.

#### Mode A — source-build installation path (fidelity)

*Question answered:* what does pip do when asked to install the published wheel, and does the
resulting environment contain what a user's would?

**Proposed procedure. Feasibility is not yet established; every command below is a proposal
until the checks in "Uncertain" have been run.**

1. *Wheelhouse*, assembled during provisioning and verified before any use:

   | Item | Source | Verified against |
   | --- | --- | --- |
   | The published `fie_sdk-1.18.0-py3-none-any.whl` | PyPI | PyPI digest `be504dc7…` |
   | Every dependency that PyPI publishes as a wheel | PyPI | pinned hash from the resolution, and PyPI's digest |
   | `langdetect-1.0.9.tar.gz`, the **source archive** | PyPI | PyPI digest `cbc1fef8…` |
   | Build tools: `setuptools`, `wheel` (and `packaging`) | PyPI | pinned hashes; already verified in `stageB_remediation` |

   The locally built `langdetect` wheel is **not** in this wheelhouse. Mode A must not be able
   to pick it up.

2. *Environment*: created with plain `python -m venv`, **with** its bundled pip, as a user
   would create one. The bundled pip and setuptools versions are recorded, not chosen.

3. *Install*, offline, with the environment's own pip:

   ```text
   <env>\Scripts\python -m pip install --isolated --no-index --find-links <wheelhouse-A>
       --only-binary :all: --no-binary langdetect
       --require-hashes -r <pins-A.txt> --no-cache-dir --disable-pip-version-check -vv
   ```

   - `--only-binary :all:` stays in force for every package. `--no-binary langdetect` is the
     single, named exception, so pip must take the source archive for that one package and may
     not take a source archive for any other.
   - `<pins-A.txt>` is the resolved set with exact versions and hashes, in which the
     `langdetect` line carries the **source archive's** SHA-256.
   - Build isolation is left at pip's default. Nothing forces `--no-build-isolation` or
     `--use-pep517`: the point is to observe pip's own choice.
   - `--no-index` means pip, and any build environment it creates, can obtain files only from
     the wheelhouse.

4. *The whole of step 3 runs inside the network-off window*, so that the archive's `setup.py`
   executes with no network, and so that success demonstrates the path works offline.

**What pip is expected to do, and why it has to be observed rather than assumed.** How pip
builds a project that has a `setup.py` and no `pyproject.toml` depends on the pip version:

| Installer | Expected path for `langdetect` (PREDICTED) |
| --- | --- |
| An older pip, as bundled with Python 3.10's `venv` | The legacy path: the environment's own pre-installed `setuptools` runs `setup.py` directly, with no isolated build environment. Whether it produces a wheel first or falls back to `setup.py install` depends on whether the `wheel` package is present, which it is not in a fresh environment |
| A current pip | The isolated path: pip creates a temporary build environment, installs `setuptools` into it from the index (here: the wheelhouse), and builds a wheel through the legacy build backend |

Both are "the real path" for some users. The plan therefore records the path taken instead of
asserting one, and O-20 asks which installer to use.

**Evidence required before installation-path fidelity may be claimed** (all of it):

| # | Evidence | From |
| --- | --- | --- |
| F-1 | The artifact installed is the PyPI wheel, by digest | wheelhouse manifest; `pip inspect` |
| F-2 | `langdetect` was installed **from the source archive whose digest equals PyPI's**, not from any wheel | pip `-vv` log: the archive name, the hash check, and the build lines; absence of any `langdetect` wheel in the wheelhouse (file listing with hashes) |
| F-3 | Which build path pip took, and with which `setuptools` (version, and file digest where it came from the wheelhouse; "pre-installed in the environment, version X" where it did not) | pip `-vv` log; `pip list` before and after |
| F-4 | Every other distribution came from a wheel in the wheelhouse whose hash was pinned | pip log; `pip inspect`; comparison with `<pins-A.txt>` |
| F-5 | The install ran with the network off | the isolation record of that window |
| F-6 | The installer's identity: pip version, where it came from (bundled or other) | recorded before the install |
| F-7 | The installed set equals the pinned set: names and versions, nothing missing, nothing extra beyond what the environment had before | `pip inspect` versus pins; `pip check` exit 0 |
| F-8 | `import langdetect` and `import fie` succeed in that environment | probe |

With F-1 to F-8 the report may say: *the published wheel was installed by pip through a
source build of `langdetect` from PyPI's source archive, offline, from a verified wheelhouse.*
It must also state the three ways this still differs from an ordinary user's install: the
index was a local wheelhouse and not PyPI; build-tool versions were those resolved on the
provisioning date; and the installer was the one named in F-6.

**Uncertain, to be established by a feasibility check before mode A is relied on:**

- Whether the existing interpreter's `venv` produces an environment with a working bundled
  pip, and which pip and setuptools versions it bundles.
- Whether `--only-binary :all:` together with `--no-binary langdetect` gives exactly the
  intended selection with that pip.
- Whether `--require-hashes` accepts the pinned source archive and still builds it.
- Whether an isolated build, if pip chooses one, finds its build tools in the wheelhouse under
  `--no-index`. pip does **not** apply `--require-hashes` to build requirements; they are
  covered only by having been verified when the wheelhouse was assembled.
- Whether the legacy path, if pip chooses that, succeeds without the `wheel` package, and what
  it leaves on disk (a directly installed package rather than a wheel-installed one).
- Whether the pinned set, resolved with pip 26.0.1, is accepted unchanged by the bundled pip.

If the source-build path cannot be made to work offline and reproducibly, that is recorded as
a result, mode A is reported as **not achieved**, and the owner is asked how to proceed
(O-23). The pre-built wheel is not substituted under the mode A label.

#### Mode B — pre-built local wheel installation

*What it is:* the environment is installed from wheels only, the `langdetect` wheel being the
one built in the Stage B remediation and declared explicitly:

```text
python -m pip --python <env python> install --isolated --no-index --find-links <cache>
    --find-links <local-wheels\langdetect> --only-binary :all:
    --require-hashes -r <pins-B.txt> --no-deps --no-cache-dir --disable-pip-version-check -vv
```

`<pins-B.txt>` carries the local wheel's SHA-256 (`9acdfda7…`) on the `langdetect` line. No
source is built and no package code runs at install time.

*It must be labelled* "pre-built-local-wheel installation" wherever it appears. It is **not**
a reproduction of the published package's normal installation path and is never described as
one.

| Mode B can support | Mode B cannot support |
| --- | --- |
| Runtime behaviour of the installed `fie` code with this dependency set: imports, component state, coverage, verdicts, policy actions, attempted egress, warnings | Any statement about what `pip install fie-sdk` does: whether it succeeds, what it builds, what it needs at install time, how long it takes |
| Comparisons with `BL-0002` (modes P and PC) | The claim that an environment is "as a user would have it" in its installed metadata (`INSTALLER`, `RECORD`, entry-point scripts, bytecode) |
| Deterministic, repeatable provisioning: the same bytes every time | Anything about the `setuptools` or pip a user would have |

Mode B's runtime conclusions extend to a source-built install **only if** C9 (§8) shows the
installed `langdetect` files are identical in both modes. Until then they are conclusions about
mode B environments.

#### Comparison and recommendation

| | Mode A | Mode B |
| --- | --- | --- |
| Reproduces pip's source-build path | Yes, if it can be made to work offline | No |
| Executes third-party `setup.py` at install | Yes, inside the network-off window | No |
| Deterministic bytes on disk | Not guaranteed (generated metadata, bytecode) | Yes for the wheel; pip's unpacking is deterministic |
| Works with `--only-binary :all:` unmodified | No: one named exception | Yes |
| Risk of not working | Real (see "Uncertain") | Low: already resolved and, for the base set, download-verified |
| Answers RQ14 (what does installation do) | Yes | No |
| Answers RQ4 to RQ9 (what does the installed package do) | Yes, but slower and less repeatable | Yes |

**Recommended smallest valid experiment.** The E51 objective has two parts: what installing
the package does, and what the installed package then does. They need different modes, and
neither needs the full matrix in both:

1. **One install-path cell in mode A**: the published wheel, base install, no `[ml]`. It
   answers RQ14 and H10 for the artifact users actually have, with the smallest dependency set
   and no approval-gated download.
2. **The matrix of §6.3 in mode B**, labelled as such, for all runtime questions.
3. **C9** ties them together: `langdetect`'s installed files in the mode A cell against a mode
   B base environment. If identical, H11 holds and the mode B runtime results are stated to
   hold for source-built installs of the same dependency versions. If not, the differing files
   are reported and the runtime results stay scoped to mode B.

A second mode A cell for the published wheel **with** `[ml]` adds evidence only about whether
the larger set also installs through this path; it is optional and needs that set's separate
approval (O-22). Mode A for the two locally built wheels is not proposed: their dependency
handling is the same pip mechanism on the same `langdetect` requirement.

---

## 7. Setup, commands, evidence

### 7.1 Isolation of the Python environment

1. Environments are created with `python -m venv` from the private interpreter, under the
   scratch directory, with no user site.
2. The probe runs with `python -I -B` from an empty working directory, started through that
   environment's `Scripts\python.exe` and through no other executable.
3. **First check in every probe run:** `fie.__file__` is inside that environment's
   `site-packages` and the repository is not on `sys.path`. Otherwise the cell is void.
4. The probe's process environment is an explicit allowlist: no variable from the developer's
   shell, no `.env`, no credential, no proxy variable. `HOME`, `USERPROFILE`, `TEMP` and (in P
   and PC only) `FIE_FEEDBACK_PATH` point into the cell's scratch directory. In mode S no
   `FIE_*` variable is set, so `fie` uses its default feedback path, which resolves under the
   scratch `USERPROFILE`.

### 7.2 Reproducibility

- `pip freeze` and `pip inspect` saved per environment; the resolved set frozen into a
  `constraints.txt`; the repeat run installs from it.
- Wheels identified by SHA-256. A-clean and A-release are built twice; a byte difference is a
  finding.
- PC is run twice per cell in separate processes; records must be identical. S and P are run
  twice; differences are reported, not required to be zero (S may be non-deterministic by
  design: an unseeded language detector).
- Machine facts recorded: OS build and edition, CPU, Python build, pip version, firewall state.

### 7.3 Command outline

`$X` is the scratch directory. Steps marked **[owner, elevated]** are run by the owner.

```text
# Step 0 — repository environment
python -m pytest tests/ -m "not network" -p no:cacheprovider -q        # expect 942 passed
python -m evals baseline                                                 # expect exit 0 vs BL-0002

# Step 1 — PROVISION (network: PyPI only; every pip call carries the section 5.6 flags)
python provision.py preflight                    # inherited config, proxy, hosts: stop on any refuse condition
python provision.py index fie-sdk                # section 6.1: stop if the latest release is not 1.18.0
PIPFLAGS = --isolated --index-url https://pypi.org/simple --no-cache-dir --only-binary :all:
           --disable-pip-version-check -vv        (constructed environment, PIP_CONFIG_FILE = null device)
pip download fie-sdk==1.18.0 --no-deps $PIPFLAGS -d $X/wheels/pypi
python inspect_wheel.py $X/wheels/pypi/*.whl --manifest scripts/model_manifest.json
pip install --dry-run --ignore-installed --report $X/resolve/<cell>.json $PIPFLAGS <requirement>
python provision.py pin $X/resolve/<cell>.json   # hosts must be files.pythonhosted.org; writes hashed pins
pip download --require-hashes -r $X/pins/<cell>.txt $PIPFLAGS -d $X/cache
python provision.py verify                       # logs' hosts, cache hashes, PyPI digests, tags, file set

# Step 2 — build (no network)
git archive 2f62ecd | tar -x -C $X/src-clean
python -m build --wheel --no-isolation --outdir $X/wheels/clean   $X/src-clean
(copy verified model files into $X/src-release/fie/models)
python -m build --wheel --no-isolation --outdir $X/wheels/release $X/src-release
python inspect_wheel.py $X/wheels/*/*.whl dist/*.whl --manifest scripts/model_manifest.json

# Step 3 — private interpreter and environments (no network)
(copy a local Python 3.10 installation to $X/py310; create $X/env/<cell> from it)
$X/env/<cell>/Scripts/python -m pip install --no-index --find-links $X/cache --require-hashes -r $X/pins/<cell>.txt
python exe_inventory.py $X > inventory-before.json

# Step 4 — positive control, then isolate, then negative control: through EVERY measurement executable
for each cell:  $X/env/<cell>/Scripts/python -I -S -B egress_selftest.py --expect open      # V1
                $X/py310/python              -I -S -B egress_selftest.py --expect open      # V1
[owner, elevated]  e51_block.ps1 inventory-before.json                                      # one rule per inventory path
for each cell:  $X/env/<cell>/Scripts/python -I -S -B egress_selftest.py --expect blocked   # V2
                $X/py310/python              -I -S -B egress_selftest.py --expect blocked   # V2
(V3: export rules joined to the inventory.  A cell that fails V1 or V2 is not measured.)

# Step 5 — MEASURE (verified cells only), per cell and mode
$X/env/<cell>/Scripts/python -I -B probe.py --mode S  --inputs … --out …
$X/env/<cell>/Scripts/python -I -B probe.py --mode P  --inputs … --out …
$X/env/<cell>/Scripts/python -I -B probe.py --mode PC --inputs … --out …
for each cell:  … egress_selftest.py --expect blocked                                        # V4
python exe_inventory.py $X > inventory-after.json ; compare                                  # V6

# Step 6 — restore and compare
[owner, elevated]  e51_unblock.ps1                                       # removes FIE-E51-* rules
$X/py310/python -I -S -B egress_selftest.py --expect open ; Get-NetFirewallRule …           # V5
python compare.py --evidence … --baseline evals/baselines/BL-0002_pair-v6.3b_fie-70de7053
python -m evals baseline                                                 # still exit 0

# Cells assigned to Fallback A (O-18): steps 4-5 are replaced by the owner-run, network-off script.
```

### 7.4 Evidence schema

Every file produced by the probe begins with the same header, so no record can be read out of
context:

```text
mode                  "S" | "P" | "PC"
artifact              "A-pypi-1.18.0" | "A-clean" | "A-release"   + wheel SHA-256
extra                 "base" | "ml"
install_mode          "A-source-build" | "B-prebuilt-local-wheel"
installer             pip version and origin ("bundled with venv" | "repository pip via --python")
langdetect_origin     "built by pip from PyPI source archive <sha256>" | "local wheel <sha256>, built in provisioning"
encoder_variant       "as_installed" | "pre_staged"
env_overrides         every FIE_* and related variable set for this run ([] in mode S)
in_process_changes    every assignment made to the installed fie ([] in modes S and P)
instrumentation       ["socket_recorder", "pickle_tripwire", "subprocess_recorder"]
isolation             "os_firewall_verified" | "owner_network_off" | "instrumentation_only"
isolation_verified    for this cell: launcher path and base path, each with V1, V2, V4 outcome
entry_executable      the path the probe was started through; actual process image; parent image
models_loaded         for each file opened: relative path, SHA-256, manifest status
blocked               null, or {reason, file, sha256, unanswered: [RQ…, H…]}
```

| File | Content |
| --- | --- |
| `wheel.json` | From `inspect_wheel.py`: file list, sizes, SHA-256; each model file's manifest status; `METADATA`; the loader's preference list as read from source; the default load set; console-script entry points |
| `startup_code.json` | Every `.pth` file and `sitecustomize` / `usercustomize` module in the environment, across all installed distributions, with hashes and the lines that import or execute code. This code runs before the instrumentation loads |
| `provisioning.json` | §5.6: inherited pip configuration and proxy state; the exact command lines; resolution reports; pinned hashes; hosts found in the logs; cache-versus-report and report-versus-PyPI digest checks; the verdict |
| `freeze.txt`, `inspect.json` | Installed distributions, versions, size on disk |
| `import.json` | Did `import fie` succeed; exception class if not; seconds; modules imported; `fie.__version__`; `fie.__file__`; optional packages importable |
| `egress_python.jsonl` | **Process-local observations only**: phase, host, port, calling module; and every name passed to a Python-level DNS lookup. No payload. Titled as observations, never as proof of isolation. Lookups made by native code do not appear here |
| `isolation.json` | V1 to V6 results **per executable path and per cell**, both inventories with hashes, the exported rule list joined to the inventory, firewall profile state, proxy settings, whether UDP denial was verified, and (with O-15) the firewall log extract. Under Fallback A: adapter and route state before and after. This file, not the previous one, is the evidence for the isolation claim |
| `state.json` | `warmup()`; `classifier_state()`; `health()` with error text reduced to its state code; `model_identities()`; which of these APIs exist in the artifact |
| `records.jsonl` | Per prompt: verdict, type, confidence, layers fired, scores, `zone`, `decided_by`, `coverage`, `degraded_layers`; or the exception class. Prompt identified by index and SHA-256 |
| `policy.json` | Per record, `decide` under the three presets where `fie.policy` exists; `preflight_check` under the default for the reference prompts (mode S) |
| `cli.json` | Exit code, stdout key list and stderr of `python -m fie detect` / `explain` (mode S); for `fie.exe`: "executed under network-off" with results, or "not executed: launcher isolation not verifiable" |
| `warnings.json` | Warning categories and counts; log records at WARNING and above reduced to fixed prefixes |
| `timing.json` | Install, import, warm-up, first-scan and mean scan times |

Older artifacts lack some APIs (H2). The probe records "absent"; an absent field is a result.

### 7.5 Dependency verification gates

These gates sit between the completed Stage B and any installation. Each must pass, in order,
for each dependency set separately. A gate that fails stops that set; nothing downstream of it
runs, and no other set's result is used in its place.

| Gate | Requirement | Stop if |
| --- | --- | --- |
| **G0 — carried over** | The PyPI release-version check (§6.1), the unmanifested-model rule (§4.2), the no-proxy and no-redirection checks (§5.6) and network-off isolation for measurement all remain in force. G0 is re-checked at the start of Stage C, since time has passed | The latest release is no longer `1.18.0`; a proxy or redirection has appeared |
| **G1 — authorization** | A dependency set is downloaded only after the owner has approved **that set** by name. The base set is approved. The current-tree `[ml]` set (O-21) and the PyPI `[ml]` set (O-22) are separate approvals; one does not imply the other | A set is not approved: its cells are recorded as "not provisioned: awaiting approval", not skipped silently |
| **G2 — re-resolve** | Immediately before download, the set is resolved again with the Stage B flags. The result must equal the pinned set recorded in `stageB_remediation/pins/` (names, versions, hashes) | The resolution has changed (a new upstream release since Stage B). The difference is reported; pins are not updated silently |
| **G3 — download** | `pip download --require-hashes` from the pins, explicit index, inherited configuration neutralized, no cache | pip exits non-zero; a wheel is missing; any host other than the two approved appears in the log |
| **G4 — verify** | Every file in the wheelhouse: SHA-256 equals its pin **and** the digest PyPI publishes for it, size equal, host `files.pythonhosted.org`. The file set equals the pinned set, no extra file | Any digest or size mismatch; any file not in the pins; a resolution/download manifest disagreement |
| **G5 — provenance ledger** | One ledger per wheelhouse lists every file with `origin`: `pypi-wheel`, `pypi-sdist` (the `langdetect` archive, mode A only) or `local-build` (the `langdetect` wheel, mode B only). Locally built artifacts carry the source digest they were built from and are never given a PyPI digest | An artifact of unknown origin; the local wheel present in a mode A wheelhouse; the source archive present in a mode B one |
| **G6 — install** | Installation from the wheelhouse with `--no-index` and `--require-hashes`, in the install mode declared for that environment, recorded in the evidence header | pip exits non-zero; pip reports a source build of anything other than `langdetect` in mode A, or of anything at all in mode B |
| **G7 — post-install proof** | **A successful dry run is not evidence that installation works.** After install: `pip inspect` set equals the pins (F-7); `pip check` exits 0; the interpreter path and version are as expected; `import fie` and each top-level dependency import in a clean subprocess | Any of these fails. The environment is then recorded as "installed but not verified" and is not measured |
| **G8 — model files** | Static listing and hashing of every model file in the installed environment against the manifest (§4.2), before the first import of `fie` | An unverified file is in the default load set: mode S for that cell is blocked (§4.3) |

`--only-binary :all:` is part of G2, G3 and G6 for every set. The single exception is
`--no-binary langdetect` in mode A, stated in the command, recorded in the header, and tested
on its own cell.

### 7.6 Sequential environments and disk space

Free space on the system drive was 32 GB (94% used) at Stage B. `[ml]` environments are
therefore created and removed **one at a time**, never side by side.

**Per cell, in order:** create the environment → install from the wheelhouse (G6) → verify
(G7, G8) → run the modes that apply → write and checksum the evidence → remove that
environment's directory → next cell.

Rules:

- The shared, verified wheelhouse is kept for the whole experiment, so that no cell needs the
  network and a removed environment can be recreated byte-for-byte from the same pins.
- Only directories this experiment created under `C:\Users\ASUS\fie-e51-scratch\env\` are
  removed, each by its full, explicit path, after its evidence has been written and verified.
  Nothing else in the scratch directory, and nothing outside it, is deleted. No automatic
  attempt is made to free disk space.
- Because installs use `--no-index`, the whole create-install-measure-remove sequence can run
  inside the network-off window, driven by the one owner-run script.
- Environment creation inside the offline window means an environment does not yet exist when
  the positive connectivity control is taken. The positive control is therefore taken, before
  going offline, through the base interpreter and through one environment launcher created the
  same way; the negative control is repeated through each environment as it is created. Under
  machine-level isolation this is sufficient: no rule depends on the executable's path.
  **Superseded for Stage C1 by §0.2**: base environments are created before the offline window
  and every launcher gets its own positive control. This rule now applies to `[ml]` cells only,
  and is to be revisited before either `[ml]` set is approved.

**Disk estimate (sequential):**

| Item | Size | Basis |
| --- | --- | --- |
| Wheelhouse, base set | 2 MB | measured |
| Wheelhouse, current-tree `[ml]` | +209 MB | resolved sizes from PyPI metadata |
| Wheelhouse, PyPI `[ml]` | +about 130 MB beyond the above (338 MB alone; most large wheels are shared) | resolved sizes |
| One current-tree `[ml]` environment | about 0.7 GB | the same packages as installed on this machine: about 500 MB for the eight largest |
| One PyPI `[ml]` environment | about 1.5 GB | **estimate**; `torch` and `transformers` are not installed here to measure |
| Base environments | under 20 MB each | wheel sizes |
| Evidence | tens of MB | 2,016 to 3,864 records per comparable run |
| **Peak, sequential** | **about 2 GB** (wheelhouse plus the largest single environment) | |
| For comparison, side by side | about 3.5 GB | not proposed |

Before each `[ml]` install the script records free space and stops if less than 5 GB would
remain. The PyPI `[ml]` figure is the least certain number in this plan.

---

## 8. Datasets, sample sizes, comparisons

| Set | Size | Source | Used in |
| --- | --- | --- | --- |
| Reference prompts | 8 | `tests/test_scan_result_contract.py` | S, P |
| Smoke set | 40 | First 5 of each standard dataset | S, P |
| Standard suites | 2,016 (1,516 attack, 500 benign) | The harness's seven `std.*` suites, exported with SHA-256 | PC |
| Base-install suite | 1,848 | The harness's `lite.std` inputs | PC, base cells |

All inputs are public benchmark text. No new dataset; no sampling choice after seeing results.
S and P are deliberately small: they establish behaviour, state and attempted destinations,
not rates.

| # | Compare | With | Answers | Mode |
| --- | --- | --- | --- | --- |
| C1 | Base cells | `BL-0002` `lite.std.jsonl`, record by record | RQ10, H1 | PC |
| C2 | `[ml]` cells, encoder pre-staged | `BL-0002` `std.*.jsonl` | Does an installed package reproduce the canonical pipeline when its verified files are present? | PC |
| C3 | `[ml]` cells, encoder as installed | C2 | Size of the packaging gap | PC |
| C4 | A-pypi-1.18.0 | A-release | RQ11 | each mode separately |
| C5 | Policy actions per cell | The replay table of PLAN_004 §6.4 | H8; no preset is selected | S for the default; replay otherwise |
| C6 | Coverage self-report | Layer states actually observed | Does each artifact tell the truth about itself? | S |
| C7 | *(new)* S | P, same cell and inputs | What the pin alone changes. Zero difference is expected where the default choice is already the pinned file; any difference is reported | S vs P |
| C8 | *(new)* P | PC, on the 48 shared prompts | What the harness's knobs alone change | P vs PC |

| C9 | *(rev. 4)* Installed `langdetect` files, mode A cell | The same files in a mode B base environment, by path and SHA-256, excluding pip-generated metadata (`RECORD`, `INSTALLER`, `direct_url.json`, `__pycache__`), which are listed separately | H11: do mode B runtime results hold for a source-built install? | install mode A vs B |

**No cross-mode substitution.** A PC result is never used to state what an artifact does by
default, and a blocked S cell is never filled in from P or PC. Likewise a mode B environment
is never used to state what installing the package does, and a mode A failure is never
replaced by a mode B success.

---

## 9. Pass/fail criteria and stop conditions

**E51 is a measurement. It "passes" when it was executed completely and reproducibly. It does
not pass or fail on whether the package behaves well.**

| # | Criterion |
| --- | --- |
| P1 | Every cell and mode of §6.3 was executed, recorded as not applicable with a reason, or recorded as **blocked** with the blocking file and the unanswered questions |
| P2 | In every measured cell the probe verified it imported the installed copy |
| P3 | PC repeat runs are identical; S and P repeat differences are reported |
| P4 | **Isolation, as established, per cell.** Firewall design: V1 and V2/V4 passed **through that cell's launcher and through the base interpreter**, V3 and V6 passed, and the report makes the §5.4 claim with its limits, including that DNS remained an open, data-capable channel. Fallback A: adapter state recorded as down before and after, plus the negative control through each path; this is the only form for which the report may say that no network channel was available. Fallback B: the criterion is reduced to "no egress observed by process-local instrumentation" and the report states that machine-level egress was not excluded. No cell is reported at a level its own executables did not verify |
| P4b | **Provisioning used only the approved source**, as shown by every check of §5.6 step 5. Otherwise provisioning is reported as failed and no measurement follows |
| P5 | No unverified pickle or model file was loaded; the tripwire never fired, or it fired and the cell is recorded as blocked |
| P6 | Every evidence file carries its mode header; no P or PC result is described as as-shipped, default or unmodified |
| P7 | The PyPI release check of §6.1 was performed and its outcome is in the report |
| P8 | The repository is unchanged apart from §3.1; `python -m evals baseline` exits 0 against `BL-0002` afterwards; all `FIE-E51-*` rules are removed and V5 passed |
| P9 | Each hypothesis is marked confirmed, refuted, not testable or blocked, with the mode and evidence file |
| P10 | Every environment's install mode, installer and `langdetect` origin are in its evidence header; no mode B environment is described as reproducing the published installation path |
| P11 | Installation-path fidelity is claimed only with evidence F-1 to F-8 (§6.4); otherwise mode A is reported as not achieved |
| P12 | Gates G0 to G8 (§7.5) passed for every set that was installed; sets awaiting approval are reported as not provisioned |
| P13 | `[ml]` environments existed one at a time; only experiment-created environment directories were removed |

**Stop and report if:**

- the latest PyPI release is not `1.18.0` (until O-17 is answered);
- any gate of §7.5 fails for a set: a missing wheel, an unexpected package source, a resolution that
  no longer equals its pins, a resolution/download disagreement, or a digest mismatch;
- pip builds from source anything other than `langdetect`, or builds anything in mode B;
- the source-build path of mode A cannot be made to work offline (report; O-23);
- an `[ml]` set would have to be downloaded or installed without its own approval;
- less than 5 GB would remain free after an install;
- the probe cannot prove it is measuring the installed copy;
- provisioning would need a source other than PyPI, or a credential;
- a build or install would write inside the repository;
- for any measurement executable, V1 fails (no control) or V2 fails (the block does not hold): that
  cell is not measured, unless a fallback is approved for it;
- a probe would have to be started through an executable that has not passed the controls;
- the executable inventory changes during the measure phase;
- a system proxy, a redirecting `hosts` entry or un-neutralizable pip configuration is found, or any
  provisioning check of §5.6 fails;
- a process is created whose image is not one of the cell's two verified paths;
- the pickle tripwire fires in a cell the static inspection had cleared;
- measuring would require changing `fie/`, packaging, dependencies, models, a workflow or a baseline;
- a cell needs more than about 3 GB of downloads without O-4 approved;
- a secret, token or personal path appears in an evidence file;
- the firewall rules cannot be removed, or V5 fails;
- `python -m evals baseline` no longer exits 0.

---

## 10. Packaging and compatibility risks

The experiment changes nothing public. These are the risks its results are likely to raise.

| # | Risk | Why it matters |
| --- | --- | --- |
| K1 | A version number already means three things (E-1) | Any corrected build needs a new version |
| K2 | `[ml]` changed meaning without a release (E-2) | Switching it changes install size by gigabytes and removes a dependency some users import |
| K3 | Wheel content depends on the builder's disk (E-3 to E-5) | A release cut from a developer machine can ship unreviewed pickles; the `dist/` wheel already contains four the manifest does not list |
| K4 | The encoder arrives by first-use download (E-7) | A default install's coverage depends on reaching GitHub at run time |
| K5 | Unpinned dependencies (E-11) | Scores can drift with no change to `fie` |
| K6 | `requires-python >= 3.9` is untested (E-12), and stays untested in this experiment (O-5) | Recorded as an open gap, not closed |
| K7 | Pickle model files execute code on load (D-008) | Controlled here by §4.2 to §4.4 |

No public API is touched. `compat@1` remains the default.

---

## 11. Security and privacy

| Concern | Control |
| --- | --- |
| Data leaving the machine | Connections: OS-level denial verified through each measurement executable (§5.3). DNS: **open under the firewall design** and capable of carrying data; closed only under the network-off procedure (§5.4). Public benchmark inputs and a credential-free environment reduce the consequence of a leak; they are not a guarantee against one |
| Telemetry | Left enabled in mode S so the attempt is observed; the connection is denied by the firewall rule. Disabled in PC |
| Credentials | Allowlisted environment; `GROQ_API_KEY` and every `.env` key absent; none needed for PyPI |
| Executing downloaded code | A-pypi and its dependencies run only in disposable environments, with egress denied, no credentials, a scratch home directory, and no repository on the path |
| Pickles | Static listing and hashing before install; unverified files never loaded without O-12; S blocked where the default would load one; tripwire as backstop |
| Machine configuration | Firewall rules are named, scoped to executables inside the scratch directory, added and removed by the owner, and verified removed (V5). No default firewall policy, adapter or DNS setting is changed under §5.2 |
| Evidence content | Prompts as index and hash; paths relative to the scratch root; loader error text reduced to state codes; a final scan for the user name, home path, e-mail addresses and token-like strings must find nothing |
| Supply chain | §5.6: inherited pip configuration recorded and neutralized; one explicit index on every command; no cache; hashes pinned from the resolution, enforced by pip and re-checked against PyPI's published digests; `--only-binary :all:` so no `setup.py` runs; start-up code of installed distributions listed |

---

## 12. Cleanup, acceptance, report

**Cleanup.** The owner runs `e51_unblock.ps1`; V5 confirms no `FIE-E51-*` rule remains and
connectivity is as before. Deleting the scratch directory removes every environment, wheel,
export, cache and the private interpreter. Nothing is installed into the `failure-engine`
environment, the base interpreter or the user site. Repository rollback is deleting
`docs/fie_rebuild_2026/experiments/E51/`, `docs/fie_rebuild_2026/evidence/E51/` and the report,
and reverting the master-log edit. `dist/` is not touched. If dropped-packet logging was
enabled (O-15) it is returned to its previous setting.

**Acceptance criteria**

| # | Criterion | Proof |
| --- | --- | --- |
| AC-1 | Step 0 gates pass on `2f62ecd` | 942 tests; `evals baseline` exit 0 |
| AC-2 | Current PyPI release verified before any download | Saved index metadata; §6.1 outcome |
| AC-3 | Every artifact identified | SHA-256, size, file list, metadata, per-file manifest status |
| AC-4 | Every cell × mode executed, n/a, or blocked with reason and unanswered questions | Matrix table |
| AC-5 | Installed-copy check passed in every measured cell | `import.json` |
| AC-6 | Dependencies recorded | `freeze.txt`, `inspect.json`, hashes |
| AC-7 | Component state recorded | `state.json` |
| AC-8 | Verdicts recorded and compared, mode by mode | C1 to C8 |
| AC-9 | Policy actions recorded without changing a default | `policy.json` |
| AC-10 | **Isolation established and verified at the level claimed, for every measurement executable of every measured cell** | `isolation.json`: V1, V2, V4 per launcher path and base path, V3, V6; or the fallback's evidence with its claim. No cell measured without its own verification |
| AC-10b | The report's isolation wording matches the level achieved, and nowhere states or implies that data could not leave under the firewall design | Wording check against §5.4 |
| AC-10c | Provisioning shown to have used only PyPI | `provisioning.json`, every check of §5.6 step 5 passed |
| AC-11 | Python-level egress attempts recorded | `egress_python.jsonl`, labelled as observations |
| AC-12 | No unverified model file loaded | `models_loaded` in every header; tripwire log |
| AC-13 | Modes never conflated | Mode header on every file; report wording check (P6) |
| AC-14 | Repeatable | Second run from the frozen set |
| AC-15 | Evidence is clean | The scan of §11 finds nothing |
| AC-16 | Repository and machine restored | `git status`; `evals baseline` exit 0; V5 |
| AC-17 | Hypotheses resolved | H1 to H11 each confirmed / refuted / not testable / blocked |
| AC-18 | Install modes recorded and never conflated | Header fields on every file; report wording check (P10) |
| AC-19 | Installation-path fidelity either evidenced (F-1 to F-8) or reported as not achieved | Mode A cell's evidence |
| AC-20 | Dependency gates passed per set; dry-run success not used as proof of installation | Gate ledger; G7 output per environment |
| AC-21 | Provenance ledger separates PyPI wheels, the PyPI source archive and the locally built wheel | G5 ledger |
| AC-22 | Sequential execution and scoped cleanup | Free-space log; list of removed paths |

**The execution report must contain:** status and commit state; starting state; the PyPI
release check; the isolation level achieved **per cell**, in its first table, with the exact claim and its
limits, including the status of DNS and of UDP verification; the provisioning verification; the matrix with each cell and mode's outcome; artifact identities and per-file
manifest status; installed dependency sets; per-cell component state, coverage and verdict
comparisons, separated by mode; the Python-level egress table; policy replay; hypothesis
table; blocked cells and the questions they leave open; every statement labelled measured,
simulated or predicted; limitations; findings ranked by user impact; a recommendation for the
follow-up packages; cleanup confirmation including the firewall; final `git status`.

It must not say that the package "works" or "passes" as a whole, must not say "no egress"
or "no data left the machine" for any cell measured under the firewall design or under
instrumentation only, and must not describe a pinned run
as default behaviour.

---

## 13. Follow-up structure and owner decisions

### 13.1 Three packages, in this order

| Proposed | Content | Depends on |
| --- | --- | --- |
| **WP-020** (this plan) | E51: measure and characterise. No behaviour change | WP-003, WP-004 |
| WP-021 (not planned here) | Packaging and default install: wheel contents, encoder delivery, dependency bounds, a new version, a release gate that installs the built wheel and checks coverage | WP-020's results |
| WP-022 (not planned here) | Hermetic default (the roadmap's "WP-005"): telemetry, translation, remote tiebreaker and model download become opt-in and are reported in `coverage` | WP-020's egress evidence; best after WP-021, because "no download by default" and "how the encoder is delivered" are one decision seen from two sides |

### 13.2 Numbering

The roadmap uses WP-001 to WP-019. The master log's WP-004 is runtime policy (D-024), which
displaced the roadmap's WP-004 "install truth" and left its WP-005 "hermetic default"
unscheduled. **Proposed: the next free numbers, WP-020, WP-021, WP-022.** No existing package
is renumbered and no existing plan is touched.

### 13.3 Owner decisions

Decisions O-1 to O-13 are from revision 1 (some reworded); O-14 to O-17 were added in
revision 2; O-18 and O-19 in revision 3; O-20 to O-24 in revision 4. Several have since been
decided by the owner: see §0.1, which governs where it differs from a row below.

| # | Decision | Recommendation | Alternative |
| --- | --- | --- | --- |
| O-1 | Work-package number | **WP-020**; reserve WP-021, WP-022 | Reuse the roadmap's WP-005 slot |
| O-2 | Network during provisioning | **PyPI only** (`pypi.org`, `files.pythonhosted.org`), enforced and verified by the procedure of §5.6, which includes one bare TCP handshake to PyPI from each measurement executable as the positive control (V1) | The owner supplies a wheel cache |
| O-3 | Download the published `fie-sdk==1.18.0` wheel | **Yes**, after the release check of §6.1 | Measure only locally built wheels |
| O-4 | The PyPI `[ml]` cell, which pulls `torch` | **Dry-run report first** (installs nothing); full install only if the owner approves the size shown | Skip the cell |
| O-5 | Python versions | **3.10 only.** The private-interpreter design needs a local installation to copy; 3.9, 3.11 and 3.12 are not on this machine and obtaining them means a conda or python.org download, which is not PyPI. They stay untested and are reported as such | Approve a named second source for interpreters |
| O-6 | Location of the experiment code | **`docs/fie_rebuild_2026/experiments/E51/`** | `scripts/` or `evals/` |
| O-7 | Mode PC applies the harness's in-process settings to the installed `fie` | **Yes**, as a separately labelled mode | S and P only; verdicts then cannot be compared with `BL-0002` |
| O-8 | Platform | **Windows, this machine**, stated as a limitation; Linux later through CI | Add WSL2 or Docker if available and permitted |
| O-9 | Build wheels locally with `--no-isolation` | **Yes** | Isolated builds, which need PyPI for the build backend |
| O-10 | A-release's models come from working-tree files that match the manifest | **Yes** | Run `download_models.py` against GitHub (a second network source) |
| O-11 | Release-version implication | **Decide in WP-021.** Noted: the next published build needs a new version | — |
| O-12 | Loading a pickle or model file that is not in the manifest | **No.** List and hash only; the affected S cell is recorded as blocked | Approve specific named files, after seeing the inspection output, for loading in the isolated environment only |
| O-13 | The stale wheel and sdist in `dist/` | **Leave in place; inspect only** | Delete as part of WP-021 |
| **O-14** | **Firewall rules for the measure phase**: the owner runs `e51_block.ps1` and `e51_unblock.ps1` in an elevated prompt; rules are named `FIE-E51-*` and scoped to executables in the scratch directory | **Yes.** It is the only OS-level mechanism available on this edition that leaves the rest of the machine working | Fallback A or B (O-16) |
| **O-15** | Enable firewall dropped-packet logging for the duration, restore afterwards | **Yes**: a second independent record, and the only way UDP denial is verified | No; rely on V1 to V6, with UDP denial reported as configured but not verified |
| **O-16** | If O-14 is declined, its preconditions fail, or a cell's executable-level controls cannot be verified | **Fallback A** (owner-run with the network off) for the affected cells | Fallback B (instrumentation only, reduced claim and reduced S inputs; not for A-pypi cells); or leave the affected cells unmeasured |
| **O-18** | **Isolation form per artifact**, given that DNS stays open under the firewall design | **Split:** firewall rules for A-clean and A-release (repository code at a reviewed commit, verified models); **owner-run network-off for both A-pypi cells**, and for any run of `fie.exe` | Network-off for the whole measure phase (simplest, strongest); or firewall rules throughout, accepting an open DNS channel while unreviewed published code runs |
| **O-19** | If a system-level proxy is configured on this machine | **Stop and ask.** Options then: the owner disables it for provisioning; or approves it by name as part of the path, with the report saying so | The owner supplies a wheel cache |
| **O-17** | If the latest PyPI release is not `1.18.0` | **Add the latest as its own artifact and keep `1.18.0` as "pinned historical"** | Measure the latest only; or `1.18.0` only, labelled historical |

| **O-20** | **Installer for the mode A cell** | **The environment's own bundled pip**, as a fresh `python -m venv` provides it: that is the installer a user has before upgrading anything. The path it takes is recorded | Also run the cell with the repository's current pip, to observe the isolated-build path; or the current pip only |
| **O-21** | **Download and install the current-tree `[ml]` set** (47 packages, 209 MB download, about 0.7 GB installed), for the two locally built wheels | Approve, sequentially | Decline: cells 4 and 6 are reported as not provisioned |
| **O-22** | **Download and install the PyPI `[ml]` set** (56 packages including `torch` 2.14.1, 338 MB download, about 1.5 GB installed, estimated), for the published wheel. **Separate from O-21** | Approve, sequentially, after O-21's environments have been removed | Decline: cell 2 is reported as not provisioned, and H3 rests on the dry run alone |
| **O-23** | If the source-build path (mode A) cannot be made to work offline | **Stop and ask.** Options then: report mode A as not achieved and keep mode B results scoped to mode B; or allow a mode A install with the network connected, PyPI only, with the weaker isolation stated | — |
| **O-24** | Scope of mode A | **One cell**: published wheel, base install | Add the published wheel with `[ml]` (needs O-22) |

Not decisions for this package: the default policy stays `compat@1`; nothing selects
`strict@1`.

---

## 14. Limitations and unresolved feasibility

**Nothing was executed.** Every statement about install-time behaviour is READ, INSPECTED or
PREDICTED.

**Feasibility not yet established, each checked in preflight before anything is measured:**

| # | Open point | If it fails |
| --- | --- | --- |
| F1 | Whether a copied Python 3.10 installation runs correctly from the scratch directory and can create working virtual environments | Block the existing interpreter's executable for the duration instead (more disruptive, owner decision), or Fallback A |
| F2 | Whether Windows Defender Firewall is the filter actually enforcing rules on this machine (no third-party product in control) | Fallback A |
| F3 | Whether a per-program rule takes effect for each launcher path and for the base interpreter it starts, and whether the control test can read the true process image | Established per cell by V1/V2; a cell that cannot be verified is not measured under this design |
| F4 | DNS queries issued by the system resolver on the program's behalf cannot be blocked per program, and can carry data | Not solvable within the firewall design. Stated limit (§5.4); Fallback A where that is not acceptable (O-18) |
| F8 | UDP denial cannot be verified by the control test alone | Verified only with O-15; otherwise reported as configured, not verified |
| F9 | Whether pip's inherited configuration on this machine can be fully neutralized, and whether a system proxy is present | Checked in provisioning preflight; stop conditions in §5.6 |
| F10 | Whether `pip install --dry-run --report` and `pip download --require-hashes` resolve identically for every cell with the installed pip version | If not, provisioning stops for that cell and the reason is reported |
| F11 | Whether the owner-run network-off script can run the full measure phase unattended and restore connectivity reliably | Dry-run the script on a trivial cell first; the owner restores the adapter manually if it does not |
| F5 | Whether the PyPI wheel's loader preference list can be read statically | That cell's mode S is blocked |
| F6 | Whether the owner is willing to run two elevated commands | Fallback A or B |
| F7 | Whether `1.18.0` is still the latest release | O-17 |

**Added in revision 4.** Mode A's feasibility is open on six points listed in §6.4 (bundled pip present and its version; the `--only-binary` / `--no-binary` combination; `--require-hashes` with a source archive; build tools found offline under an isolated build, which pip does not hash-check; the legacy path without the `wheel` package; pins from one pip accepted by another). Agreement between resolution and hash-required download is shown for the base set only. The PyPI `[ml]` environment's installed size is an estimate. The locally built wheel's hash identifies one build, on one machine, with one `setuptools` version.

**Other limitations.** One machine, one operating system, one Python version. Mode S measures
behaviour when network calls fail; behaviour when a first-use download succeeds is not
measured. The `dist/` versus PyPI comparison rests on size until step 1. Wheel builds may not
be byte-reproducible. The matrix's duration is unknown.

---

## 15. Revision history

| Revision | Change |
| --- | --- |
| 1 | Initial proposal |
| 2 | **Network isolation**: process-local instrumentation separated from OS-enforced denial; per-program Windows Defender Firewall rules on a private interpreter proposed, with positive and negative control tests, stated limits (DNS, other processes, proxies), two fallbacks and correspondingly reduced claims; provisioning kept separate and PyPI-only. **Modes**: S (as-shipped configuration), P (manifest-pinned), PC (manifest-pinned, comparable); `FIE_PAIR_VERSION` removed from anything called as-shipped; unverified model files listed and hashed, never loaded without approval; S blocked where the default would load one, with unanswered questions recorded; pickle tripwire added. **PyPI**: current release verified before any download, with a stop and an owner decision if it is not `1.18.0`. Matrix, evidence schema, comparisons (C7, C8), hypotheses (H9, mode column), pass criteria, acceptance criteria and owner decisions (O-14 to O-17; O-5 and O-12 changed) updated to match. Python versions other than 3.10 dropped from the default matrix because obtaining them needs a non-PyPI source |
| 3 | **DNS**: the claim that a lookup can expose only a host name was wrong and is removed; DNS is described as an open, data-capable channel that native code can use without passing the Python instrumentation; reduced consequence (public inputs, no credentials) is distinguished from a guarantee; network-off is named the preferred form for the A-pypi cells, unverified models and `fie.exe` (O-18). **Executables**: a full executable inventory with one rule per path; positive and negative controls run through every cell's launcher and the base interpreter, reading the true process image; no cell inherits another path's result; console-script launchers are not run under the firewall design; UDP verification limited to the firewall log. **Provisioning**: inherited pip configuration, indexes, mirrors, caches and proxies recorded and neutralized; one explicit index on every command; resolve, pin by hash, download with hashes required; hosts and digests verified independently; stop conditions including a system proxy (O-19). Evidence schema, criteria (P4, P4b, AC-10, AC-10b, AC-10c), stop conditions and feasibility list (F3, F4, F8 to F11) updated |
| 4 | Written after Stages A and B. **Record** of what was executed and of owner decisions that supersede earlier text (§0.1): airplane-mode isolation for the whole measure phase, no private interpreter, scratch directory, per-set `[ml]` approval, the `langdetect` exception. **Installation modes** (§6.4): mode A, the source-build path, proposed with its commands, its eight items of required evidence and its open feasibility points; mode B, the pre-built local wheel, with what it can and cannot support; recommendation of one mode A cell plus the mode B matrix, tied by comparison C9. **Dependency gates** G0 to G8 (§7.5), including that a dry run is not proof of installation. **Sequential environments** and disk estimate (§7.6). New finding E-16, hypotheses H10 and H11, RQ14, criteria P10 to P13, AC-18 to AC-22, decisions O-20 to O-24 |
| 5 | Stage C1 execution instructions only (§0.2, one note in §7.6): existing interpreter, owner-switched airplane mode, all base-cell launchers created before the offline window with a positive control through each and negative controls at the start and end, and the boundary between connected provisioning and offline measurement. No change to hypotheses, matrix, criteria or modes. `[ml]` remains unapproved |

---

## 16. Recommendation

Continue WP-020 as a measurement-only package under the decisions recorded in §0.1.

For the `langdetect` problem: run **one install-path cell in mode A** (published wheel, base
install, source build of `langdetect` from the verified archive, offline, with the
environment's bundled pip) to establish what installation actually does; run the **matrix in
mode B**, labelled as pre-built-local-wheel installation, for every runtime question; and use
comparison C9 to decide whether the mode B runtime results may be stated for source-built
installs. If mode A cannot be made to work offline, report that and ask (O-23); do not relabel.

Approve the two `[ml]` sets separately (O-21, O-22), each downloaded only after its own
approval and each passing gates G1 to G8, with environments created and removed one at a time.

Then decide packaging (WP-021) and the hermetic default (WP-022) from the evidence. E-16, that
every install of the package builds a dependency from source, is already an input to WP-021.

**Implementation of Stage C and beyond is not authorized by this document.**
