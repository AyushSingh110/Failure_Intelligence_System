# E51 Stage B remediation — `langdetect` source-build exception

Date: 2026-10-11. Work package WP-020. HEAD `2f62ecd`. Nothing here is a measurement of `fie`.

## Why this directory exists

Stage B (see `../stageB/`) stopped because the complete dependency set could not be resolved
under the plan's wheels-only rule. `fie-sdk` requires `langdetect>=1.0.9` in every install.
PyPI publishes `langdetect` 1.0.9 as a source archive and a wheel tagged `py2` only; neither is
a wheel a Python 3 interpreter accepts. The original failure is preserved, unchanged, in
`../stageB/resolve/base.pip.log` and `base.summary.json`. **That original resolution did not
pass, and nothing here changes that record.**

## The exception, as approved by the owner

One narrowly scoped exception to "no source builds": a wheel for `langdetect` 1.0.9 was built
**once, during provisioning**, from the official PyPI source archive, and is supplied to pip as
an explicitly declared local wheel. Every other dependency is still a wheel published on PyPI,
resolved with `--only-binary :all:` from the one approved index.

**The built wheel is not a PyPI-published artifact.** It is recorded everywhere with origin
"locally built from the digest-verified PyPI source archive" and with no PyPI digest.

## What was done

| Step | Evidence |
| --- | --- |
| PyPI release metadata for `langdetect` 1.0.9 read with the standard library (proxy handling off) | `pypi_langdetect_1.0.9.json` |
| Source archive downloaded from `files.pythonhosted.org`; SHA-256 and size equal PyPI's published values | `sdist_langdetect_1.0.9.json` |
| Archive inspected without executing it: 88 members, no links, no unsafe paths, no compiled or binary files; `setup.py` and `setup.cfg` only, no `pyproject.toml` | same file (`build_files` holds both texts) |
| Build tools resolved, hash-pinned and downloaded from PyPI: `setuptools` 84.0.0, `wheel` 0.48.0, `packaging` 26.3 | `resolve/build-tools.*`, `pins/build-tools.txt`, `download/build-tools.*` |
| Build tools installed with `--no-index --require-hashes` into a dedicated build environment (`env/_build` in the scratch directory, Python 3.10.19). This is not an experiment environment | `build/install_build_tools.log` |
| Wheel built twice, into separate output directories, with `pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir` and `SOURCE_DATE_EPOCH=1620374053` (the archive's upload time) | `build/pip_wheel_run1.log`, `build/pip_wheel_run2.log` |
| The two wheels are byte-identical. All 69 payload files equal the corresponding files in the source archive. Tag `py3-none-any`; `Requires-Dist: six` | `build/langdetect_wheel.json` |
| Complete resolutions rerun with the wheel declared: `base`, `ml-head`, `ml-pypi` | `resolve/*.summary.json`, `resolve/*.report.json`, `pins/*.txt` |
| Hash-required download of the complete base set agrees with its resolution: 11 of 11 files | `download/base.summary.json` |

## Identities

| Artifact | SHA-256 | Size |
| --- | --- | --- |
| `langdetect-1.0.9.tar.gz` (PyPI source archive) | `cbc1fef89f8d062739774bd51eda3da3274006b3661d199c2655f6b3f6d605a0` | 981,474 |
| `langdetect-1.0.9-py3-none-any.whl` (locally built) | `9acdfda7ee22ffdaaaf87f993cec0eb186d772fe0f3b93469832d83be6e9c8a4` | 993,364 |

## Limits of this exception

- Building executed the archive's `setup.py` on this machine, in the build environment, with
  the network connected. The file was read first: it reads `README.md` and calls `setup()`.
  pip was given `--no-index`; its log shows no download.
- Build isolation was off by design, so that the build tools are the pinned, hash-verified ones
  and not whatever pip would fetch at build time.
- Reproducibility was shown for this machine, this Python and these tool versions, twice. A
  different `setuptools` version may produce different bytes (the wheel's `WHEEL` file names the
  generator). The hash above identifies this build, not "the" wheel for `langdetect` 1.0.9.
- What a real user's `pip install fie-sdk` does is build this package from source at install
  time with whatever `setuptools` pip fetches. The experiment will install a pre-built wheel
  instead. The `langdetect` code is identical; the install-time behaviour is not, and E51 will
  report the source-build requirement as a finding rather than reproduce it in each cell.
- The `[ml]` sets were resolved only. They were not downloaded, so agreement between their
  resolutions and a hash-required download is not yet shown.

Paths in the logs are redacted (`<HOME>`, `<USER>`).
