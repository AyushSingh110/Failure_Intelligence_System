# Stage 3.0 — commands and evidence sources

Everything here is read-only. No model file was opened by anything but a hash function; no
wheel or source distribution was built; nothing was installed; no script from `scripts/` was
executed; no release asset was downloaded. The two things downloaded were GitHub Actions
artifacts named `release-check` (the gate's own report), into a scratch directory.

## Repository state

```text
git rev-parse HEAD ; git branch --show-current ; git status --short ; git diff --cached --name-only
git fetch origin ; git rev-parse origin/main ; git merge-base --is-ancestor HEAD origin/main
git ls-remote --heads origin
cat evals/baselines/CANONICAL
sha256sum -c SHA256SUMS.txt        (in each of the five WP-021 evidence directories)
git ls-files scripts/release_check.py .github/workflows/ci.yml tests/test_release_check.py
```

## CI runs (`gh`, authenticated as the repository owner)

```text
gh run view <run> --json databaseId,event,headSha,headBranch,status,conclusion,createdAt,updatedAt,workflowName,url,jobs
gh api repos/AyushSingh110/Failure_Intelligence_System/actions/runs/<run>/artifacts
gh run download <run> -n release-check -D <scratch>
gh run view <run> --job <package job id> --log
gh pr list --state all --json number,title,headRefName,state,mergedAt,mergeCommit
```

for `<run>` = `38282217538` and `38282733913`. `ci/ci_runs_summary.json` is derived from those
four sources by a script that re-verifies each downloaded artifact against its own
`SHA256SUMS.txt` and compares the wheel digest printed by `sha256sum` in the job log, the digest
printed by the gate in the job log, and the digest in the uploaded `report.json`.

Kept in `ci/ci_run_<run>/`: the uploaded `report.json` byte for byte, the artifact's own
checksum list, the run metadata and the artifact listing. Not kept: the per-command output
files of the artifacts and the raw job logs (available from GitHub until the artifacts expire
on 2027-01-10; the logs follow GitHub's retention).

## Model inventory and provenance

```text
git tag -l ; git rev-list -n1 v1.18.0 ; git rev-list -n1 models-v1.18.0
git log --format='%H %ad' --date=iso-strict -- scripts/model_manifest.json
git show <commit>:scripts/model_manifest.json          (each commit in that history, and v1.18.0)
git show --stat --format='%h %ad %s' 08d5474
gh release view models-v1.18.0 --repo AyushSingh110/Failure_Intelligence_System --json tagName,publishedAt,createdAt,isDraft,isPrerelease,assets
sha256 of every file under fie/models/ (working tree)
sha256 of every fie/models/ member of the PyPI wheel (the E51 digest-verified copy) and of dist/fie_sdk-1.18.0-py3-none-any.whl, read from the archives
evals/baselines/BL-0002_pair-v6.3b_fie-70de7053/fingerprint.json   (identity.models)
docs/fie_rebuild_2026/evidence/E51/stageC1/provisioning/pypi_fie-sdk_summary.json   (PyPI upload time and digest, saved 2026-10-11)
```

## The seven unmanifested files

```text
git ls-files <path> ; git check-ignore -v <path>
git hash-object <path> ; git cat-file -e <blob id>            (do these exact bytes exist anywhere in the object database?)
git log --all --format='%H %ad %s' --date=short -- <path>
git show --name-status --format= <commit> -- <path> ; git rev-parse <commit>:<path>
git grep -n -e 'pair_intent_classifier_<v>\b' -e 'pair_intent_meta_<v>\b' -e '--tag <v>\b' -- . ':!docs/fie_rebuild_2026'
git grep -n -E 'FIE_PAIR_VERSION' -- . ':!docs/fie_rebuild_2026'
git grep -n -E 'joblib\.load|pickle\.load' -- scripts
the side-car JSON of each file, parsed as JSON
scripts/eval_harmaug.py, scripts/train_pair_harmaug.py, scripts/retrain_pair_v2.py, scripts/retrain_pair_v3.py, fie/layers/pair.py: read as text
```

## Not done

- No comparison of a file's content beyond size and digest.
- The published source distribution was not downloaded.
- The original `meta_clf.pkl` release asset of 2026-08-10 no longer exists and could not be hashed.
- No research script was run to see whether it still works.
