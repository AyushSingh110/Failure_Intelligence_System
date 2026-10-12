# PLAN_021 — Packaging and Base-Import Correctness

> **PROPOSED — IMPLEMENTATION NOT AUTHORIZED.**
> Revision 1. This is a plan. Nothing in it has been implemented: no production source, test,
> packaging metadata, dependency, model file, workflow, `dist/` file or baseline has been
> changed, and nothing has been downloaded or installed to write it.

| | |
| --- | --- |
| Proposed work package | **WP-021** (numbering: PLAN_020 §13.2; no existing package is renumbered) |
| Date | 2026-10-12 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `becc2c7` |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053` |
| Follows | WP-020 / E51 Stage C1: [EXECUTION_020_install-truth-stageC1.md](../EXECUTION_REPORTS/EXECUTION_020_install-truth-stageC1.md) |
| Coordinates with | WP-022, hermetic default (not planned here; boundary in §7) |
| Kind of work | A packaging contract: what `pip install fie-sdk` installs, what `import fie` needs, what a wheel may contain, and what a release must prove |
| Not in scope | Detector retraining, thresholds, benchmark changes, policy defaults, the server (`app/`, `engine/`), the remaining E51 cells |
| Owner decisions | W21-1 to W21-14, §9 |

Evidence labels. **MEASURED**: observed in E51's offline, gated runs. **STATIC**: read from a
source file, a wheel archive or saved metadata without running it. **PREDICTED**: implied by
the code or the metadata; not observed.

---

## 0. State verified for this plan

| Check | Result |
| --- | --- |
| Branch, HEAD | `rebuild/wp-001-eval-harness`, `becc2c7` "docs: E51 Stage C1 base-install results…" (the owner's commit of the Stage C1 work) |
| Working tree | Clean before this file was written; nothing staged |
| `CANONICAL` | `BL-0002_pair-v6.3b_fie-70de7053` |
| E51 evidence | Checksums verify: attempt 4 (13 files), attempt 5 (37), attempt 7 (97), three diagnostics (3 each), comparisons (3) |
| Source behind the measured local wheels | `fie/`, `pyproject.toml` and `scripts/model_manifest.json` are identical to `2f62ecd`, the commit A-clean and A-release were built from |

E51's results are used here as they stand: four cells installed, four cells import-blocked,
**zero scan records**. Nothing in this plan treats them as a runtime measurement.

---

## 1. Findings that define the scope

### 1.1 The base-import failure

| # | Finding | Label |
| --- | --- | --- |
| F-1 | `import fie` raises `ModuleNotFoundError: No module named 'starlette'` in a base install: published wheel `1.18.0` in both install modes, and wheels built from `2f62ecd` with and without model files. `python -m fie` fails the same way. 4 cells, 0 scans | MEASURED |
| F-2 | The chain is `fie/__init__.py:12` `from fie import integrations` → `fie/integrations/__init__.py:25` `from fie.integrations import openai, anthropic, fastapi` → `fie/integrations/fastapi.py:28-31`, four `from starlette…` imports at module level | STATIC, and MEASURED as the recorded import chain |
| F-3 | `starlette` is declared nowhere by name. It arrives only through `fastapi`, which is listed under the `server` and `dev` extras. Base dependencies are `requests`, `deep-translator`, `langdetect` | STATIC |
| F-4 | **`starlette` is the only third-party module imported unguarded at module level anywhere in the import closure of `fie`, apart from `requests`**, which is a base dependency. The closure is 26 modules. `fie/integrations/openai.py` and `anthropic.py` import only the standard library at module level | STATIC (AST walk of module-level imports from `fie/__init__.py`) |
| F-5 | Removing the `starlette` requirement from the import path would let `import fie` succeed in a base install | **PREDICTED** from F-4. Function-level imports and import-time side effects are not covered by F-4 |
| F-6 | `import fie` also starts a telemetry thread (`fie/__init__.py:19`, `_ping_telemetry`) unless `FIE_NO_TELEMETRY` is set | STATIC. Belongs to WP-022 (§7) |
| F-7 | The repository's own tests never caught F-1: they run in the development environment, where `starlette` is installed, and import `fie` from the working tree. CI's `package` job builds a wheel and inspects its file list but never installs or imports it. `publish-pypi.yml` checks only that the wheel contains at least one model file | STATIC |
| F-8 | In-repository uses: `tests/test_integration.py` and `tests/test_runtime_policy.py` import `fie.integrations.openai` and `.anthropic` by submodule path. Nothing in the repository imports `fie.integrations.fastapi` or uses `FIEMiddleware`; there is no test for it | STATIC |
| F-9 | The `server` extra lists `langgraph`, `pymongo`, `slowapi`, `uvicorn`, `python-dotenv`: dependencies of the FIE server (`app/`, `engine/`), which the wheel does not contain (`packages = ["fie"]`) | STATIC |

### 1.2 Packaging and artifact integrity

| # | Finding | Label |
| --- | --- | --- |
| P-1 | **Three artifacts carry version `1.18.0`**: the PyPI wheel (`be504dc7…`, 270,149 bytes, uploaded 2026-08-10), the wheel in `dist/` (`e0b415d4…`, 313,989 bytes), and what the tree builds now (A-release `17b23a9e…`, 314,838 bytes; A-clean `87104867…`, 151,936 bytes) | STATIC |
| P-2 | The PyPI wheel lacks `fie/policy.py` and `fie/onnx_encoder.py`, which the tree has. So `1.18.0` on PyPI is not the code at HEAD | STATIC |
| P-3 | **`[ml]` differs.** PyPI `1.18.0`: `joblib`, `pandas`, `scikit-learn`, `sentence-transformers`, `xgboost` (56 distributions, 338 MB resolved). Tree: `scikit-learn`, `xgboost`, `joblib`, `pandas`, `onnxruntime`, `tokenizers` (47, 209 MB), with `sentence-transformers` moved to a new `torch` extra. Same version number | STATIC (wheel metadata; Stage B remediation resolutions) |
| P-4 | **Wheel content follows the builder's disk.** `fie/models/` is git-ignored; `[tool.hatch.build] artifacts` force-includes whatever `fie/models/*.pkl` and `*.json` exist at build time. A clean checkout builds a wheel with no model; this working tree would include every file present | STATIC; A-clean and A-release are the two MEASURED ends |
| P-5 | **The published wheel's `meta_clf.pkl` is not the manifest's.** PyPI and `dist/`: 179,675 bytes, `01be1995…`. Manifest and A-release: 212,131 bytes, `be6673d0…`. The accompanying `meta_clf.json` differs too: published = 11 features, threshold 0.50, 18,601 training rows; manifest = 12 features, threshold 0.41, 2,370 rows | STATIC (archive listing and the JSON side-car; **no pickle was opened**) |
| P-6 | The manifest's `meta_clf` entries changed in commit `08d5474` "fix+measure: meta-classifier read 6 of 11 features as constant zero", after the `v1.18.0` commit. So the published file is most likely the meta-classifier from **before** that fix, not a corrupted file | STATIC for the history; the causal reading is **PREDICTED** |
| P-7 | The `dist/` wheel also carries four pickles the manifest does not list: `pair_intent_classifier_v2.pkl`, `_v3_w3x.pkl`, `_v6_1_contaminated.pkl`, `_v6_noaug.pkl`. The PyPI wheel carries none of them | STATIC |
| P-8 | **Nothing verifies a model file at load time.** `fie/layers/pair.py` calls `joblib.load` on the first preferred file that exists (`:184`, `:334`). `scripts/model_manifest.json` is not shipped in the wheel. The digest is reported after loading (WP-003), not checked before | STATIC |
| P-9 | The classifier loader takes the first existing file from a fixed preference list, `v6_3b` first. Which model an install runs therefore depends on which files that wheel happens to contain | STATIC |
| P-10 | **The encoder is never in the wheel** (`model.onnx`, 90 MB, and `tokenizer.json` live in a subdirectory the `artifacts` patterns do not match). On first use `fie/onnx_encoder.py` downloads them from the GitHub release `models-v1.18.0` unless `FIE_NO_AUTO_DOWNLOAD` is set | STATIC |
| P-11 | If that download fails, an `[ml]` install has no encoder and the classifier layer reports itself unavailable | PREDICTED. No `[ml]` cell has run |
| P-12 | **`pip install fie-sdk` always builds `langdetect` from source.** PyPI has no Python 3 wheel for `langdetect` 1.0.9 | MEASURED (Stage B resolution failure; mode A install) |
| P-13 | With pip 23.0.1 that build is the legacy `setup.py install` path, which pip 23.1 removed. Other pip versions take other paths | MEASURED for 23.0.1; other versions **not measured** |
| P-14 | `langdetect` is imported only inside functions, behind guards, in `fie/multilingual.py`; it is not needed for `import fie` | STATIC |
| P-15 | Every dependency is a lower bound only, in base and in every extra. Model pickles are tied to the library versions that wrote them; the v6.3b metadata records none | STATIC |
| P-16 | `requires-python = ">=3.9"` and classifiers for 3.9 to 3.12. Only 3.10.19 has been exercised, in this repository and in E51 | STATIC |
| P-17 | PyPI also publishes a source distribution, `fie_sdk-1.18.0.tar.gz`, 3,308,210 bytes. **Its contents have not been inspected.** `publish-pypi.yml` runs `python -m build`, which builds both | STATIC for existence; contents unknown |
| P-18 | `publish-pypi.yml` runs on a `v*` tag or by hand, downloads models with `--strict`, builds, checks that at least one model file is present, and uploads. It does not check that the version is unused, that the wheel's contents equal the manifest, or that the wheel imports | STATIC |
| P-19 | A base install cannot load any pickle: the loader imports `joblib` first and `joblib` is not a base dependency | STATIC; consistent with E51 (no model file opened), but E51 never reached a scan |

How the published wheel came to hold a file that `download_models.py --strict` would not have
produced is **not known**. P-6 is the likeliest explanation and is to be confirmed, not assumed
(§4.2).

---

## 2. Import architecture

### 2.1 What the public surface is today

`fie.__all__` includes `"integrations"`. `fie.integrations.__all__` is
`["openai", "anthropic", "fastapi"]`. The documented uses are
`from fie.integrations import openai`, `from fie.integrations import anthropic` and
`from fie.integrations.fastapi import FIEMiddleware`.

In any environment where `import fie` works today, `starlette` is installed. In every
environment where it is not, `import fie` fails. So **no working installation depends on the
current eager import**, and any change below can only turn failures into successes for base
users.

### 2.2 Option A — lazy or guarded integration imports

`import fie` stops importing the FastAPI integration. The integration is imported when asked
for.

| Question | Answer |
| --- | --- |
| Base import | Succeeds without `starlette` (PREDICTED, F-4/F-5) |
| Public API | `fie.integrations`, `fie.integrations.openai`, `fie.integrations.anthropic` unchanged. `from fie.integrations.fastapi import FIEMiddleware` unchanged where `starlette` is installed |
| Extra dependencies for a minimal install | None |
| Optional dependency missing | Nothing happens until the integration is requested |
| Explicit request without the dependency | Must fail at the request, loudly, naming the missing package and the install command. Never a silent stub |
| Risk | Attribute access `fie.integrations.fastapi` after a bare `import fie` changes from "already imported" to "imported now". Star-import and tools that enumerate attributes need a decision (below) |

Two ways to do it:

- **A1, lazy attribute (PEP 562).** `fie/integrations/__init__.py` imports `openai` and
  `anthropic` eagerly, as now (they need nothing), and provides a module-level `__getattr__`
  that imports `fastapi` on first access. `from fie.integrations.fastapi import …` and
  `from fie.integrations import fastapi` both go through the normal import system and behave
  as before.
- **A2, guarded import.** `try: from fie.integrations import fastapi` / `except ImportError`.
  Rejected: it swallows the reason, leaves the attribute simply absent, and would hide a real
  bug inside the module as "not installed".

### 2.3 Option B — declare the dependency

| Variant | Effect | Assessment |
| --- | --- | --- |
| B1: add `starlette` (or `fastapi`) to base `dependencies` | `import fie` works | **Rejected.** It makes a web framework mandatory for a detection SDK, to satisfy an import the SDK does not use. It converts an optional dependency into a mandatory one, widens the supply-chain surface of every install, and leaves the real defect (the eager import) in place for the next integration |
| B2: tell users to install `fie-sdk[server]` | No code change | **Rejected.** `pip install fie-sdk` stays broken, and `server` pulls `langgraph` and `pymongo` for code the wheel does not ship (F-9) |
| B3: a small, correctly named extra for the integration | `pip install "fie-sdk[fastapi]"` brings `starlette` | **Needed, but not sufficient.** It gives Option A's error message something true to say. It does not fix the base import |

### 2.4 Recommendation

**Option A1, with B3.**

1. `fie/integrations/__init__.py`: keep `openai` and `anthropic` eager; load `fastapi` through
   `__getattr__`; provide `__dir__` so it stays discoverable.
2. `fie/integrations/fastapi.py`: wrap the four `starlette` imports. On
   `ModuleNotFoundError`, raise `ModuleNotFoundError` again (same class, `name="starlette"`,
   original chained) with the message
   `fie.integrations.fastapi needs Starlette, which is not installed. Install it with: pip install "fie-sdk[fastapi]"`.
   Any other exception raised while importing the module propagates unchanged.
3. `pyproject.toml`: a new extra `fastapi = ["starlette>=<bound>"]` (W21-3). The middleware
   uses `starlette` only; it does not import `fastapi`.
4. `fie/__init__.py`: unchanged. `from fie import integrations` stays, and `"integrations"`
   stays in `__all__`.

Why this fits the repository: the SDK's layers already follow the same rule. Optional
packages are imported inside functions behind `fie/_degrade.attempt`, and a missing one is
reported as a state (`unavailable_dependency`), not raised at import. The integration package
is the one place that rule was not applied. A1 changes two files and one metadata table, adds
no dependency to a base install, and leaves every documented import statement working.

One compatibility point needs a decision (**W21-2**): `fie.integrations.__all__` lists
`"fastapi"`. With a lazy attribute, `from fie.integrations import *` would trigger the import
and raise in a base install. Proposed: `__all__ = ["openai", "anthropic"]`, with `fastapi`
documented as available by name. Star-import of this package fails in a base install today, so
nothing that works now breaks.

### 2.5 Tests that state the contract

In `tests/`, running in the development environment:

| Test | Asserts |
| --- | --- |
| Import closure | An AST walk from `fie/__init__.py` over module-level imports finds no third-party module outside the declared base dependencies. This is the static form of F-4 and fails the moment someone adds another eager optional import |
| Blocked-module import | In a subprocess with `starlette`, `fastapi`, `openai`, `anthropic`, `joblib`, `sklearn`, `xgboost`, `numpy`, `pandas`, `onnxruntime`, `tokenizers` made unimportable by a `sys.meta_path` finder: `import fie` succeeds, `fie.integrations.openai` and `.anthropic` are present, `fie.__all__` resolves |
| Explicit request, dependency absent | Same subprocess: `from fie.integrations.fastapi import FIEMiddleware` raises `ModuleNotFoundError` with `name == "starlette"` and a message containing `fie-sdk[fastapi]` |
| Explicit request, dependency present | `from fie.integrations.fastapi import FIEMiddleware` works; a minimal Starlette app with the middleware blocks a known injection and passes a benign request |
| A bug is not "not installed" | A non-import error raised inside the integration module is not converted or swallowed |

A blocked-module test is necessary but is **not** the acceptance evidence. The acceptance
evidence is the installed-wheel gate of §5, because F-7 is exactly the case of tests passing
against a tree that does not behave like the artifact.

---

## 3. Scope

### 3.1 In scope

| # | Item | Addresses |
| --- | --- | --- |
| S1 | Base import: Option A1 and the `fastapi` extra | F-1 to F-5 |
| S2 | A release gate that builds the wheel, installs **that file** into a fresh environment and tests it | F-7, P-18 |
| S3 | Deterministic wheel content: the set of model files comes from a tracked list, not from a glob over the builder's disk; the build fails if a listed file is absent or has the wrong digest, or if an unlisted one would be included | P-4, P-7 |
| S4 | A manifest inside the wheel, and verification of a model file's digest **before** it is unpickled | P-5, P-8 |
| S5 | An explicit default model selection instead of first-file-found | P-9 |
| S6 | A new version for the corrected artifact, and a publish step that refuses a version PyPI already has | P-1, P-2, P-3 |
| S7 | Metadata: extras that say what they install; `[ml]` defined once and documented as changed | P-3, F-9 |
| S8 | Truthful state when a model or optional dependency is missing, using the existing coverage contract | P-11, P-19 |
| S9 | `langdetect`: a decision, with evidence, on whether it stays an unconditional dependency | P-12 to P-14 |
| S10 | Dependency bounds and supported Python versions: a decision and a test matrix | P-15, P-16 |
| S11 | Static inspection of the published source distribution | P-17 |

### 3.2 Out of scope

- Whether anything is downloaded, sent or contacted by default: telemetry at import, encoder
  download, translation, the remote tiebreaker. **WP-022.**
- Detector quality. Retraining, thresholds, the meta-classifier's training data.
- The server, deployment, the Hugging Face Space.
- `fie/session_tracker.py:283` (`pickle.loads` of stored session data). Noted as a separate
  security item; not a packaging matter.
- Yanking or deleting anything on PyPI. That is an owner action (W21-6), never a script's.

### 3.3 Files

| File | Change | Phase |
| --- | --- | --- |
| `fie/integrations/__init__.py`, `fie/integrations/fastapi.py` | Lazy import; explicit error | 1 |
| `pyproject.toml` | `fastapi` extra; later: version, extras, bounds, `artifacts` replaced by an explicit list | 1, 3, 4 |
| `tests/test_base_import.py` (new), `tests/test_fastapi_integration.py` (new) | §2.5 | 1 |
| `scripts/release_check.py` (new) | The installed-wheel gate | 2 |
| `.github/workflows/ci.yml`, `publish-pypi.yml` | Run the gate; refuse an existing version | 2, 4 |
| `scripts/model_manifest.json`, a build hook or build script, `fie/models/MANIFEST.json` (generated) | Deterministic content and the in-wheel manifest | 3 |
| `fie/layers/pair.py` | Verify before load; explicit default | 3 |
| `tests/test_model_integrity.py` (new) | Phase 3 | 3 |
| `CHANGELOG.md`, `README.md` | The new version, the `[ml]` change, the `fastapi` extra | 4 |
| `docs/fie_rebuild_2026/EXECUTION_REPORTS/EXECUTION_021_*.md`, master log | Record | each phase |

Workflows, `pyproject.toml` and the loader are all files the owner has kept under explicit
authorization so far. Each phase below is a separate authorization.

---

## 4. Model integrity and wheel content

### 4.1 Rules proposed

1. **A wheel contains exactly the model files the manifest lists for the wheel**, each with
   the manifest's digest. Not fewer (unless the owner chooses a model-free base wheel,
   W21-8), not more, not different.
2. **The wheel carries its own manifest** (`fie/models/MANIFEST.json`: path, size, SHA-256),
   generated at build time from the tracked manifest and checked against the files actually
   packed.
3. **A model file is hashed before it is unpickled.** If it is not in the in-wheel manifest,
   or its digest differs, it is not loaded. The layer reports a state and the reason is
   logged. Unpickling executes code; verifying afterwards is too late.
4. **The default model is named**, not discovered. The in-wheel manifest marks one classifier
   as the default. `FIE_PAIR_VERSION` may select another **listed, verified** file. A file
   that merely exists in the directory is never chosen.
5. The build fails, with the file named, on any violation of rule 1.

Rule 3 needs a decision on how the refusal is reported (**W21-9**). The WP-003 contract has
`unavailable_model` (file absent) and `unavailable_load_failed` (present, failed to load), and
`LAYER_STATES` is a pinned set. Proposed: use `unavailable_load_failed` with a fixed reason
code `integrity`, which needs no contract change. A new state `unavailable_integrity` would be
clearer and is a schema change with its own tests.

### 4.2 Reconciling `meta_clf.pkl`

Not to be done by editing the manifest to match the published file, and not by loading either
file to "see which works".

| Step | Action | Kind |
| --- | --- | --- |
| R1 | Tabulate, per artifact, the digest and size of every model file and JSON side-car: PyPI wheel, `dist/` wheel, A-release, the working tree, the manifest. Mostly done (P-5, P-7); to be completed for all files | Static |
| R2 | Establish which file each candidate is from history: the manifest's change in `08d5474`, the `v1.18.0` and `models-v1.18.0` tags, the release asset's published digest | Static, plus one read of public release metadata |
| R3 | Record the conclusion as a decision: which digest is the intended meta-classifier for the corrected release, and that the published `1.18.0` carries a different one | Owner decision W21-7 |
| R4 | The corrected release ships exactly the manifest's file, enforced by rules 1 to 3 | Phase 3 |
| R5 | The published `1.18.0` is not altered. Its difference is documented in the changelog and in the execution report. Whether it is yanked is W21-6 | Owner |
| R6 | The four unmanifested pickles in `dist/`: `dist/` is not a release artifact and is not published from. Proposed: leave the stale wheel untouched until the owner decides (W21-10); rule 5 makes a recurrence impossible | Owner |

**What is not known and is not assumed:** that the published `meta_clf.pkl` is safe to load;
that it is the pre-fix model (P-6 is a reading of the history, to be confirmed by R2); what a
published `[ml]` install does with it. If R2 cannot identify the file, it stays "unidentified"
and is treated as unverified.

### 4.3 What a base wheel contains

A base install cannot load any pickle (P-19), so today's base wheel carries about 0.2 MB of
model files it cannot use. Two coherent choices (**W21-8**):

- **Keep models in the one wheel.** Simple; `pip install "fie-sdk[ml]"` works offline for
  everything except the encoder. Recommended for this package: one artifact to verify.
- A model-free wheel plus a separate model package. Cleaner separation, two artifacts to keep
  in step, and a larger change than this package needs.

---

## 5. Evaluation and acceptance criteria

### 5.1 The release gate

One script, `scripts/release_check.py`, run in CI on every push and in the publish workflow
before upload. It is given **a wheel file**, never a source tree.

```text
python -m build --wheel --outdir <out> .
python scripts/release_check.py --wheel <out>/fie_sdk-<version>-py3-none-any.whl
```

For each install target it creates a fresh `venv`, installs the wheel file with the target's
extra, and runs the checks from an empty working directory with `python -I`, so that the
repository cannot be imported by accident. It reuses the E51 harness's approach: the installed
copy is proven by `fie.__file__` being inside that environment's `site-packages`.

| Target | Install | Purpose |
| --- | --- | --- |
| T-base | `<wheel>` | The minimal SDK |
| T-fastapi | `<wheel>[fastapi]` | The integration with its declared dependency |
| T-ml | `<wheel>[ml]` | The detector stack. **Subject to W21-12**: it is a 209 MB install |

The gate needs an index to install dependencies from. It does not need, and must not be made
to pass by, a model download at run time or a call to any external service: it runs with
`FIE_NO_AUTO_DOWNLOAD=1` and `FIE_NO_TELEMETRY=1`, and a check that would need the network to
pass is reported as *not established*, not as passed. How the encoder reaches a T-ml
environment is WP-022's answer (§7); until then T-ml asserts the truthful state, not full
coverage.

### 5.2 Criteria

Five separate claims. A pass on one is never reported as a pass on another.

**Level 1 — the package installs.**

| # | Criterion | Command | Expected |
| --- | --- | --- | --- |
| AC-1 | The wheel installs in a fresh environment | `python -m pip install <wheel>` in T-base | exit 0 |
| AC-2 | Dependencies are consistent | `python -m pip check` | exit 0, "No broken requirements found." |
| AC-3 | The installed set is what the metadata declares, and nothing else | `python -I -X utf8 -m pip inspect` compared with the wheel's `Requires-Dist` closure | no missing, no undeclared distribution |
| AC-4 | The installed files equal the wheel's files | hash comparison | 0 differences |

**Level 2 — the SDK imports.**

| # | Criterion | Command | Expected |
| --- | --- | --- | --- |
| AC-5 | **`import fie` succeeds in T-base** | `python -I -c "import fie, sys; assert 'site-packages' in fie.__file__"` | exit 0 |
| AC-6 | Every name in `fie.__all__` resolves | `python -I -c "import fie; [getattr(fie, n) for n in fie.__all__]"` | exit 0 |
| AC-7 | The command line starts | `python -I -m fie --help`; `python -I -m fie detect "What is the capital of France?" --output json` | exit 0; valid JSON |
| AC-8 | No optional package was imported by `import fie` | `sys.modules` after import | none of `starlette`, `fastapi`, `openai`, `anthropic`, `joblib`, `sklearn`, `xgboost`, `onnxruntime` |
| AC-9 | The static import-closure test passes | `pytest tests/test_base_import.py` | pass |

**Level 3 — an optional integration is available when, and only when, declared.**

| # | Criterion | Command | Expected |
| --- | --- | --- | --- |
| AC-10 | In T-base, requesting the integration fails clearly | `python -I -c "from fie.integrations.fastapi import FIEMiddleware"` | `ModuleNotFoundError`, `name == 'starlette'`, message names `fie-sdk[fastapi]` |
| AC-11 | In T-fastapi it works | the same import, then a Starlette test client: one known injection, one benign request | import succeeds; injection answered 400 with `adversarial_input_blocked`; benign request passed through |
| AC-12 | `openai` and `anthropic` wrappers import in T-base without their SDKs, and fail clearly when a client is constructed without them | import; construct | import succeeds; a clear `ImportError` at construction |

**Level 4 — detector layers report what is actually available.**

| # | Criterion | Command | Expected |
| --- | --- | --- | --- |
| AC-13 | In T-base a scan returns, and says what did not run | `scan_prompt(...)`; read `coverage` | a result; `coverage.status == "partial"`; the classifier layer `unavailable_dependency`; **never `full`** |
| AC-14 | In T-base `preflight_check` under `STRICT` does not allow a partially scanned prompt | `preflight_check(prompt, policy=fie.policy.STRICT)` | not `allow` |
| AC-15 | Wheel content equals the manifest | `python scripts/release_check.py --wheel … --contents` | every listed file present with its digest; no unlisted model file; the in-wheel manifest equals the tracked one |
| AC-16 | A tampered or unlisted model file is refused before it is unpickled | T-ml, with one byte of a model file changed, and separately with an extra `.pkl` added | not loaded; the layer reports the agreed state (W21-9); an audit hook confirms the file was hashed and never passed to an unpickler |
| AC-17 | The default model is the named one | T-ml: `model_identities()` | the digest marked default in the manifest |
| AC-18 | In T-ml with no encoder present and downloads disabled, the classifier says so | `coverage` | `unavailable_model` for the classifier layer; not `ok`, not `full` |

**Level 5 — detector quality. Not established by anything above.**

| # | Criterion | Command | Expected |
| --- | --- | --- | --- |
| AC-19 | The working tree's behaviour is unchanged | `python -m pytest tests/ -m "not network" -p no:cacheprovider`; `python -m evals baseline` | all existing tests pass; exit 0, records byte-identical to `BL-0002` |
| AC-20 | Quality of an **installed** package is measured by the evaluation harness against the installed copy, not inferred | E51's `[ml]` cells, or a harness profile for an installed subject (PLAN_020 E-13) | reported separately; **not a WP-021 gate** |

**Release hygiene.**

| # | Criterion | Expected |
| --- | --- | --- |
| AC-21 | The publish workflow refuses a version that PyPI already has | fails before build, naming the version |
| AC-22 | The publish workflow runs the gate on the exact file it uploads | upload step depends on the gate; the uploaded file's digest is logged |
| AC-23 | Two builds of the same commit are byte-identical | equal SHA-256 (already true for A-clean and A-release with `SOURCE_DATE_EPOCH`) |
| AC-24 | The gate fails on the current tree before the fix | AC-5 fails on a wheel built from `becc2c7`, reproducing E51 |

AC-24 matters: a gate that has never been seen to fail on the known defect has not been shown
to detect it.

### 5.3 What may be claimed

| After | May be said | May not be said |
| --- | --- | --- |
| Level 1 | "The wheel installs" | "The package works" |
| Level 2 | "The SDK imports and its CLI starts in a base install" | Anything about detection |
| Level 3 | "The FastAPI integration works when its extra is installed" | "Integrations are covered" beyond what was exercised |
| Level 4 | "The installed package reports its own coverage truthfully and loads only verified models" | "Detection is accurate" |
| Level 5 | Whatever the harness measured, for the subject it measured | Anything carried over from another artifact |

**A clean `import fie` is not evidence that attack detection or hallucination verification
works.** In a base install, by AC-13, the package itself says most of the detector did not run.

---

## 6. Implementation sequence

Each phase is separately authorized, leaves the tree passing its gates, and ends with an
execution report and the owner's manual commit.

| Phase | Content | Gate to leave the phase |
| --- | --- | --- |
| **0. Reproduce** | Write `scripts/release_check.py` (Levels 1 and 2 only) and run it on a wheel built from the unmodified tree | AC-24: AC-5 **fails** with the `starlette` error. No source change |
| **1. Base import** | Option A1; the `fastapi` extra; tests of §2.5 | AC-1 to AC-12, AC-19. `BL-0002` byte-identical |
| **2. Gate in CI** | The gate in `ci.yml`'s `package` job; T-base and T-fastapi | The CI job fails on a deliberately broken branch and passes on the fix |
| **3. Content and integrity** | Explicit model list; in-wheel manifest; verify-before-load; named default; `meta_clf` reconciliation R1 to R4 | AC-13 to AC-18, AC-23, AC-19 |
| **4. Version and release** | New version; changelog; `[ml]` documented; version-exists check and gate in `publish-pypi.yml` | AC-21, AC-22. **No publication**: publishing is a separate owner action |
| **5. Decisions with evidence** | `langdetect` (W21-11), dependency bounds and Python matrix (W21-13), the published sdist (W21-14) | Each closed by a recorded decision, with a test where it changes behaviour |

Phase 1 is small and independent: two source files, one metadata table, two test files. It
could be authorized alone. Phases 3 and 4 touch the loader, the build and the workflows and
deserve their own review.

Stop conditions, every phase: an existing test would have to be edited; `python -m evals
baseline` differs from `BL-0002`; a check can only pass by downloading a model or contacting a
service at run time; a public import statement that works today stops working; a model file of
unknown origin would be packed or loaded.

---

## 7. Boundary with WP-022 (hermetic default)

One question divides them. **WP-021: what is in the artifact, and is it what we say it is?
WP-022: what does the installed package do on its own initiative at run time?**

| Topic | WP-021 | WP-022 |
| --- | --- | --- |
| Encoder | States the fact in the package contract: the encoder is not in the wheel, and without it the classifier reports `unavailable_model` (AC-18). Keeps the release gate from depending on a download | Decides how the encoder is delivered and whether anything is fetched without being asked: in-wheel, a separate package, or an explicit `fie download-models` command; and the default of `FIE_NO_AUTO_DOWNLOAD` |
| Telemetry at import (F-6) | Not changed. The gate sets `FIE_NO_TELEMETRY=1` and says so | Makes it opt-in, or reports it |
| Translation, remote tiebreaker | Not changed | Opt-in and reported in `coverage` |
| Model verification | Verify-before-load for files inside the wheel (rule 3) | Applies the **same** check to anything fetched later. WP-022 must not add a second, weaker path |
| Coverage states | Uses the existing ones | May add states for "network step skipped"; owns that schema change |
| `langdetect` | Whether it is a base dependency (install-time cost) | What happens to a non-English prompt when translation is off |

Handoff from WP-021 to WP-022: (1) `scripts/release_check.py`, which WP-022 extends with
network-off checks; (2) the in-wheel manifest and the verify-before-load function, as the only
approved way to load a model file; (3) the fixed fact that a default `[ml]` install without a
download has no encoder, recorded as an open defect that WP-022 closes.

WP-021 does not make the encoder download happen, default or otherwise, and does not put the
90 MB file in the wheel. If the owner wants the encoder in the wheel, that is WP-022's
decision and comes back here only as one more line in the manifest.

---

## 8. Relationship to E51

- E51 **characterises the artifacts that exist**. WP-021 **changes the packaging contract**.
  Neither rewrites the other. E51's evidence, its blocked outcomes and its zero scan records
  stay as recorded, and stay true of `1.18.0` after WP-021 ships a new version.
- E51 Stage C1 proves nothing about detection quality, in either direction. Its blocked
  questions (default verdicts, coverage self-report, warnings, agreement with `BL-0002`,
  policy actions, egress at scan time) remain open for every artifact.
- **The remaining E51 `[ml]` cells are not run until the owner decides, after reviewing this
  plan.** Rationale: the import failure is MEASURED in all four base cells; neither resolved
  `[ml]` dependency set contains `starlette` or `fastapi` (STATIC: 47 and 56 distributions,
  Stage B remediation). The same blocker is therefore **PREDICTED** for every `[ml]` cell, and
  the 209 MB and 338 MB installs would probably stop at the same line. That is a prediction.
  **No `[ml]` cell has been run, and none is described as a confirmed failure.**
- After WP-021 Phase 1 there is a new artifact for E51 to measure. Proposed then, as a
  separate decision: rerun the base cells on the corrected wheel (which would give E51 its
  first scan records and comparison C1), and only then decide on `[ml]`. The published
  `1.18.0` `[ml]` cell stays a question about the published artifact; a fix does not answer it.

---

## 9. Owner decisions

| # | Decision | Recommendation |
| --- | --- | --- |
| W21-1 | Import architecture | Option A1 with extra B3 (§2.4). Not B1 |
| W21-2 | `fie.integrations.__all__` | `["openai", "anthropic"]`; `fastapi` lazy and documented |
| W21-3 | Name and bound of the integration extra | `fastapi = ["starlette>=…"]`; the bound to be the oldest version the middleware is tested against |
| W21-4 | The `server` extra, which installs dependencies for code not in the wheel (F-9) | Keep in this package, document what it is for; reconsider with the server work |
| W21-5 | The next version number | A new version is mandatory (P-1). Proposed `1.18.1` if only Phase 1 ships; `1.19.0` once `[ml]`'s changed meaning ships as a documented change |
| W21-6 | The published `1.18.0`, which cannot be imported in a base install | Owner's call: leave, or yank on PyPI with a reason once a fixed version exists. No script does this |
| W21-7 | Which `meta_clf.pkl` is the intended one (after R1, R2) | The manifest's, if R2 confirms P-6 |
| W21-8 | Models inside the single wheel, or a model-free wheel | Inside, for now |
| W21-9 | How a refused model file is reported | `unavailable_load_failed` with reason code `integrity`; no schema change |
| W21-10 | The stale `dist/` wheel with four unmanifested pickles | Leave untouched until the owner removes it; it is not a release source |
| W21-11 | `langdetect` as an unconditional dependency, given that every install builds it from source | Decide in Phase 5 with evidence. Options: keep; move to an extra with a guarded fallback (the code already guards it, P-14); replace. Moving it changes multilingual behaviour in a base install and needs the harness |
| W21-12 | Whether the release gate installs `[ml]` (209 MB) on every CI run | On the publish workflow and on a schedule; T-base and T-fastapi on every push |
| W21-13 | Upper bounds or tested ranges for dependencies; which Python versions are claimed | Claim only what CI runs. Decide in Phase 5 |
| W21-14 | Inspect the published source distribution (needs one 3.3 MB download from PyPI) | Yes, static only, in Phase 5 |

---

## 10. Uncertainties

- **F-5 is a prediction.** The static walk covers module-level imports. An import-time side
  effect, or a function called at import, could still fail in a base install. Phase 0 and
  AC-5 are what settle it.
- Whether `import fie` works on Python 3.9, 3.11 or 3.12. Never run.
- What the published `[ml]` install does, including with its `meta_clf.pkl`. Unknown.
- How the published wheel came to contain a file that differs from the manifest (P-6).
- What the published source distribution contains (P-17).
- Whether hatchling can express an explicit, verified file list without a custom build hook.
  If it needs a hook, that hook is new build-time code and is reviewed as such.
- `starlette`'s lower bound for the middleware. Not yet tested against any version but the
  one installed here.
- Whether verify-before-load has a measurable cost per process start (hashing about 0.5 MB of
  pickles is expected to be negligible; the 90 MB encoder, if WP-022 brings it under the same
  check, is not).
- The `langdetect` build path for pip versions other than 23.0.1.

---

## 11. Revision history

| Revision | Change |
| --- | --- |
| 1 | Initial proposal, after E51 Stage C1 |
