# EXECUTION_021, Phase 3, Stage 3.3a — Packaging feasibility experiment (WP-021)

| | |
| --- | --- |
| Work package | WP-021 Phase 3, **Stage 3.3a**: offline sdist and wheel feasibility experiment |
| Status | **COMPLETE.** The full matrix ran: 36 matrix builds and 26 probe builds (62 in all). Two probe builds failed, deliberately and informatively. Nothing implemented |
| Commit status | Not committed by the assistant. Nothing staged, pushed, published or deployed |
| Date | 2026-10-14 |
| Branch / HEAD | `rebuild/wp-001-eval-harness` / `56335e28ae34048b87f5939e27b81f40b3b38dfe`, unchanged |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053`, unchanged |
| Evidence | `docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3a/` (81 files, `SHA256SUMS.txt` verified) |

Labels: **observed** (measured in this run, named in the evidence), **inferred** (reasoned from
observations), **unverified** (not measured).

Every build ran in a temporary directory outside the repository. The source was a `git archive`
of HEAD, built with `python -m build --no-isolation` and offline. No model was deserialized:
files were read as bytes, hashed and copied, and archive members were hashed. No repository file
other than this report and its evidence directory changed.

---

## 1. Preflight (observed)

| Check | Result |
| --- | --- |
| Branch, HEAD | `rebuild/wp-001-eval-harness`, `56335e2` (Stage 3.2 committed) |
| Working tree at start | One untracked file: `EXECUTION_REPORTS/EXECUTION_021_phase3-stage3_3-plan.md` (the 3.3 plan, not yet committed by the owner). Nothing staged |
| Stage 3.2 CI | PR run `38525827576` succeeded (recorded in the 3.3 plan) |
| `main` push run `38526409887` | **Unverified.** It was queued at the last check; not polled, per instructions |
| Tools (installed, not changed) | Python 3.10.19 (Windows 11, `platform.platform()` reports `Windows-10-10.0.26300-SP0`); build 1.4.0; hatchling 1.29.0; pathspec 1.0.4; packaging 25.0; pyproject_hooks 1.2.0 |
| `pyproject.toml` build section | `[tool.hatch.build] artifacts = ["fie/models/*.pkl", "fie/models/*.json"]`; `[tool.hatch.build.targets.wheel] packages = ["fie"]`; hatchling unpinned. The current configuration uses `artifacts`, **not** `force-include` |
| Open decisions | W21-16, W21-19, W21-28, W21-29, W21-31, W21-32, W21-33 are undecided; none was decided here |

## 2. Method

**Offline controls (observed):**

- Every build used `--no-isolation`, so the frontend installs nothing.
- Environment: `PIP_NO_INDEX=1`, HTTP and HTTPS proxies pointed at a closed loopback port, and
  `SOURCE_DATE_EPOCH` removed so the default behaviour is measured.
- A `sitecustomize` audit hook on `PYTHONPATH` logged and refused every `socket.connect` and
  `urllib.Request`, including in the build-backend subprocesses. Its self-test printed
  `BLOCKED network blocked by the stage 3.3a guard`.
- **Zero network attempts** were logged across all builds.

**Exports (observed, `exports.json`):**

- **LF:** `git -c core.autocrlf=false archive HEAD`, 1,106 files. Its only `fie/models/` file is
  `.gitkeep`.
- **CRLF:** a copy of the LF export in which 874 files were converted. A file was converted only
  if `git ls-files --eol` reports it as `i/lf`, it has no eol attribute, it is not under
  `fie/models/`, it has no binary extension, it contains no NUL byte, and it has at least one LF.
  The converted list is recorded. Only those files differ between the exports (asserted). Model
  artifacts and the generated manifest are added afterwards, byte-identical in both variants.

**Scenarios (observed, `scenario_inputs.json`):**

| Scenario | Added to the export |
| --- | --- |
| M0 | Nothing. The export has no model files. Its root-level `models/.gitkeep` and the tracked `models/pair_intent_meta.json` are ordinary tracked files and stay |
| M1 | The 25 tracked-inventory files at their declared paths, **each verified by size and SHA-256 (25/25)**, plus `fie/models/MANIFEST.json` derived from the tracked inventory at HEAD: 5,112 bytes, `fbab28ab…393bf2f5`, identical to the repository's generated file |
| M2 | M1 plus the 14 experiment files and the 5 encoder by-products, all **found (19/19)**, with size and SHA-256 recorded |

**Matrix:** 3 scenarios × {LF, CRLF} × 2 repeats from fresh copies. Each setup built:

1. an sdist from the tree;
2. a wheel straight from the tree;
3. a wheel from the sdist, unpacked fresh.

That is **36 builds, all exit 0**. Builds never changed the source tree (snapshot comparison).

## 3. Results

### 3.1 Builds and determinism (observed, `builds.json`)

| Setup | sdist SHA-256 (prefix), size | direct wheel = wheel from sdist | repeat 1 = repeat 2 |
| --- | --- | --- | --- |
| M0 LF | `b2d709d5…`, 67.2 MB | **yes**, `57ce2d39…` (44 members) | yes |
| M0 CRLF | `b26203ac…` | yes, `67e7ba0f…` | yes |
| M1 LF | `f3af59c6…`, 67.3 MB | **yes**, `fb2a6729…` (63 members) | yes |
| M1 CRLF | `2792b645…` | yes, `0be6530e…` | yes |
| M2 LF | `4591218b…`, 67.4 MB | **yes**, `c7473819…` (77 members) | yes |
| M2 CRLF | `2896c2b0…` | yes, `4b8e5c81…` | yes |

- On this machine, with these tools, **every build is byte-reproducible**. The wheel built from
  the sdist is **byte-identical** to the wheel built directly from the tree.
- All wheel members are timestamped `2020-02-02T00:00:00`, and all sdist members have mtime
  `1580601600` (hatchling's reproducible default).
- **M0 builds succeed without warning or error:** a model-free sdist and wheel are produced
  silently.

### 3.2 `fie/models/` content per scenario (observed, named members)

Identical in the sdist, the direct wheel and the wheel from the sdist, for LF and CRLF.

| Group | Inventory role, `in_wheel` | M0 | M1 | M2 |
| --- | --- | --- | --- | --- |
| 7 pair classifiers + 7 side-cars, meta-classifier + side-car | four SDK roles, true | 0/16 | **16/16** | 16/16 |
| `fie/models/MANIFEST.json` (derived) | — | absent | **present**, 5,112 B, `fbab28ab…` | present |
| `failure_classifier_v4.pkl`, `feature_columns_v4.pkl` | legacy, **false** | absent | **present** | **present** |
| `fie/models/.gitkeep` | — | **present** | **present** | **present** |
| 7 experiment classifiers + 7 side-cars | unlisted | absent | absent (not in the tree) | **present, 14/14** |
| 5 encoder by-products (`minilm-onnx/*`) | unlisted | absent | absent | **absent** |
| `minilm-onnx/model.onnx`, `tokenizer.json` | encoder / tokenizer, false | absent | **absent** (present in the tree) | absent |
| 5 server files (`models/…slim…`, `storage/faiss_…`) | server, false | absent | **absent** (present in the tree) | absent |

**Compared with the proposed allowlist** (16 `in_wheel` paths plus `MANIFEST.json`):

| Scenario | Extra members | Missing |
| --- | --- | --- |
| M0 | `.gitkeep` | 17 |
| M1 | `.gitkeep` and the two legacy pickles | 0 |
| M2 | M1's three extras plus all 14 experiment files | 0 |

Every packed model member in M1 has exactly the size and SHA-256 of its inventory entry; the
member digests are in `builds.json`. M1 reproduces **CI's reported "Files: 63, bundled models:
19"** exactly: 63 members, with 19 `.pkl`/`.json` files plus `.gitkeep` under `fie/models/`. This
supports, but does not prove by name, that the CI wheel contains the same set.

### 3.3 LF compared with CRLF (observed)

| Archive | Same member names | Members whose bytes differ |
| --- | --- | --- |
| Wheel (M1) | yes | **40**: all 38 `.py` files, `dist-info/licenses/LICENSE` and `dist-info/RECORD`. `METADATA` is identical. **No `fie/models/` member differs** |
| sdist (M1) | yes | 847 |

Line endings change every archive hash and every packaged source file, but never the model
content. This is a controlled variant on one machine, not a cross-platform measurement.

### 3.4 Include, exclude and `.gitignore` probes (observed, `probes.json`)

The probes used a fresh LF export with dummy fixtures. These are text files named
`dummy_listed.pkl`, `dummy_extra.pkl`, `dummy_side.json`, `sub/dummy_nested.json`,
`dummy_plain.txt` and `dummy_ignored.log`; they are never deserialized. Each probe changed only
the temporary `pyproject.toml`.

| Probe | Configuration change | Wheel result | sdist result | Wheel from sdist | Conclusion |
| --- | --- | --- | --- | --- | --- |
| P0 | none | all dummy `.pkl`/`.json`, `.gitkeep`, plain file; not the `.log` | same | same | `artifacts` packs every matching ignored file |
| P1 | wheel `exclude = [dummy_extra.pkl, .gitkeep]` | **`dummy_extra.pkl` still packed**; `.gitkeep` removed | unchanged | same as wheel | **A wheel exclude does not override `artifacts`.** It does remove an ordinary file |
| P2 | wheel `exclude = ["fie/models/dummy_*"]` | all dummies still packed | unchanged | same | An exclude glob does not override `artifacts` |
| P3 | build-level `exclude = [dummy_extra.pkl]` | still packed | still packed | still packed | A build-level exclude does not override `artifacts` |
| P4 | sdist `exclude = [dummy_extra.pkl]` | still packed | **still packed** | still packed | An sdist exclude does not override `artifacts` |
| P5 | `artifacts` removed | ignored `.pkl`/`.json` **not packed**; `.gitkeep` and `sub/dummy_nested.json` **still packed** | same | same | Without `artifacts`, `.gitignore` decides. `**/models/*.json` does not cover sub-directories, so a nested JSON is packed as an ordinary file |
| P6 | `artifacts` removed; wheel `force-include` of `dummy_listed.pkl`; wheel `exclude` of the same path | `dummy_listed.pkl` **packed** | not in the sdist | **build fails**: `Forced include not found` | **`force-include` overrides `exclude`.** A wheel-target `force-include` does not put the file into the sdist |
| P7 | as P6 without the exclude | `dummy_listed.pkl` packed | not in the sdist | **build fails** (same error) | Control for P6. A wheel built from such an sdist fails closed when the file is absent |
| G1 | P0's sdist unpacked (no `.git`); files added after unpacking | added `late_ignored.log` **excluded**; added `.pkl`/`.json` packed via `artifacts` | — | — | **`.gitignore` is honoured in an unpacked sdist without `.git`** |
| G2 | as G1, with the sdist's `.gitignore` deleted | `late_ignored.log` **packed** | — | — | The sdist's own `.gitignore` is what excludes ignored files there |

The P6 and P7 wheel-from-sdist failure logs are in `logs/`.

### 3.5 Other observations

- **The sdist is about 67 MB**, against 3.3 MB for the stale August sdist. It carries most of
  the tracked repository, including `brag-output*/` (about 41 MB: MP4 videos and MP3 music
  files), `docs/media/*.mp4`, `evals/baselines/`, `data/meta_clf_trainset.jsonl` and notebooks.
  This bears on W21-29 (narrowing the sdist) and possibly on licensing (bundled third-party music
  files). That is recorded here, not decided.
- With the current configuration the sdist **does** carry everything a wheel build needs:
  `pyproject.toml`, `.gitignore`, `fie/` and whatever `artifacts` matched. That is why the wheel
  built from it is identical to the direct wheel.

## 4. Answers to the feasibility questions

| Question | Answer | Basis |
| --- | --- | --- |
| Can the current build produce an sdist with no models? Direct and sdist wheels? | Yes, all three, exit 0, **without any warning**. The wheels contain only `.gitkeep` under `fie/models/` | Observed (M0) |
| Does the sdist carry what a clean wheel build needs? | Yes, today. A wheel from the sdist is byte-identical to the direct wheel in every scenario | Observed |
| Why does a wheel from an sdist contain models? | Because they are **in the sdist**, put there by `artifacts`, and `artifacts` re-matches them when the wheel is built. Not because of `force-include` (unused) or of any network access | Observed (M1, P0, G1) |
| Does an ignored experiment file change the output when present? | **Yes.** All 14 experiment files go into the sdist and the wheel | Observed (M2) |
| Do exclude rules override the current includes? | **No.** Wheel, build-level and sdist excludes, exact paths or globs, do not remove `artifacts` matches. `force-include` also wins over `exclude` | Observed (P1 to P4, P6) |
| Is the runtime manifest in the wheel, and how is that known? | Yes in M1 and M2 (named member, 5,112 B, `fbab28ab…`); absent in M0 | Observed |
| Exact M0/M1/M2 differences | Section 3.2 | Observed |
| Are repeat builds byte-identical? | Yes, every setup, on this machine | Observed |
| Do LF and CRLF change hashes, members or bytes? | Hashes: yes. Member names: no. Source bytes: yes (38 `.py`, LICENSE, RECORD). Model bytes: no | Observed |
| What is still needed for other operating systems? | The same matrix on Linux (CI) and macOS with the same tool versions, and a comparison of archive hashes, `.py` bytes and `fie/models/` digests | Unverified |
| Gates before 3.3b | Section 6 | — |

## 5. Findings matrix

| # | Finding | Status |
| --- | --- | --- |
| F1 | `artifacts` packs every `fie/models/*.pkl` and `*.json` present, into both sdist and wheel | Observed |
| F2 | No exclude form tested overrides `artifacts` | Observed (hatchling 1.29.0) |
| F3 | `force-include` overrides `exclude` | Observed |
| F4 | A wheel-target `force-include` does not reach the sdist; a wheel from that sdist fails | Observed |
| F5 | `.gitignore` is honoured without `.git` in an unpacked sdist | Observed |
| F6 | Legacy pickles and `.gitkeep` ship today against the allowlist | Observed (M1) |
| F7 | Experiment files ship whenever they are present | Observed (M2) |
| F8 | The ONNX directory, encoder by-products and server files never ship | Observed |
| F9 | A model-free build succeeds silently | Observed (M0) |
| F10 | Same-machine byte reproducibility; direct wheel equals wheel from sdist | Observed |
| F11 | Line endings change archive and source bytes, not model bytes | Observed (one machine) |
| F12 | `**/models/*.json` does not cover sub-directories of `fie/models/` | Observed (P5) |
| F13 | The CI wheel at `56335e2` has the M1 member set | Inferred (counts match) |
| F14 | Cross-platform reproducibility | Unverified |
| F15 | Behaviour with other hatchling versions (unpinned in `[build-system]`) | Unverified |
| F16 | The published 1.18.0 sdist (W21-14) | Unverified |
| F17 | The `main` push run `38526409887` | Unverified |
| — | Blocked parts | None |

## 6. Implications for 3.3b (recommendations, not decisions)

1. **An allowlist cannot be added on top of `artifacts` with `exclude`** (F2). The `artifacts`
   globs must be removed, and the allowed files included explicitly: by `force-include` entries
   or by a build hook that includes exactly the verified files.
2. **The sdist must carry the allowed artifacts itself** if wheels are built from it, as the
   publish workflow does. A wheel-target `force-include` alone breaks sdist → wheel (F4). That
   failure is fail-closed, but it means the sdist needs its own inclusion: W21-28.
3. **The allowlist cannot rely on `exclude` to block extras once `force-include` is used** (F3).
   An extra listed for inclusion would ship regardless. So the inclusion list must be derived only
   from `in_wheel: true` entries, plus a post-build archive check (3.3c), which is what catches a
   mistake.
4. **`.gitkeep` can be removed with a plain wheel `exclude`** (P1): it is an ordinary file, not an
   artifact.
5. **Nested files under `fie/models/`** would be packed as ordinary files (F12). The archive check
   must cover sub-directories.
6. **Silent model-free builds** (F9) make W21-16 a real choice, not a theoretical one.
7. **Line endings:** model content is unaffected (F11). The case for `.gitattributes` (W21-19) is
   reproducible archive hashes across checkouts, not model integrity.
8. **Sdist scope** (W21-29): the 67 MB of non-SDK material, including third-party music files,
   suggests narrowing the sdist deserves an owner decision before the next publication.

**Decision gates before 3.3b:**

- **W21-16:** missing artifacts — fail, or an explicit model-free mode.
- **W21-28:** should the sdist carry the allowed artifacts? Finding F4 makes this binding for the
  publish path.
- **W21-31:** should the packed `MANIFEST.json` be derived at build time?
- **The inclusion mechanism:** explicit `force-include` tables or a custom build hook. Both are
  feasible on this evidence. A hook can verify digests before including.
- **W21-33:** pin hatchling, since F2 to F4 were measured on 1.29.0 only.

W21-19, W21-29 and W21-32 can follow.

## 7. Final integrity check (observed)

| Check | Result |
| --- | --- |
| `git status` | The untracked 3.3 plan (present at start) plus this report and `evidence/WP021/phase3/stage3_3a/`. Nothing else |
| Build outputs inside the repository | None. All trees, archives and logs stayed in the temporary work area |
| Model artifacts, manifests, workflows, `pyproject.toml`, baselines | Unchanged |
| Network, installation, download | None. Zero guarded network attempts; no `pip install`; no download |
| Commit, push, publish, deploy | None |
| Deserialization | None |

## 8. Evidence

`docs/fie_rebuild_2026/evidence/WP021/phase3/stage3_3a/`; verify with `sha256sum -c SHA256SUMS.txt`.

| File | Content |
| --- | --- |
| `exports.json` | Archive digest, file counts, CRLF procedure and the list of 874 converted files |
| `scenario_inputs.json` | 25/25 inventory checks, 19/19 extras with digests, the derived manifest |
| `builds.json` | Versions, guard self-test, every build's command, exit code, output digest and network-attempt log; per-archive `fie/models/` members with sizes and digests; allowlist classification |
| `members/<archive-sha16>.json` | Every member of each of the 12 distinct archives (name, size, SHA-256, timestamp) |
| `probes.json` | Probe configurations (lines added and removed), questions, and member presence per wheel, sdist and wheel from sdist |
| `logs/*.log` | Full output of all 62 builds (36 matrix; 26 probe: 8 probes × wheel, sdist and wheel from sdist, plus G1 and G2), local paths scrubbed, and the empty network-attempt log |
| `scripts/run_experiment.py`, `scripts/package_evidence.py` | The scripts that ran the experiment and packaged this evidence |

**To reproduce:** `python scripts/run_experiment.py <repo> <empty work dir outside the repo>`,
on the same tool versions, then `package_evidence.py`.
