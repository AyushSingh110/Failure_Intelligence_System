# FIE Operations Guide

Runbook for everything that lives *outside* the code: model artifact
distribution, releases, monitoring, error tracking, coverage, and repository
maintenance. Everything here uses free tiers only.

---

## 1. Model artifact distribution (GitHub Release assets)

The trained models (`.pkl` classifiers, FAISS index) are **deliberately not in
git** — they are distributed as assets on a GitHub Release and verified
against pinned SHA-256 checksums in [`scripts/model_manifest.json`](../scripts/model_manifest.json).

### Why

- CI builds (Docker image, PyPI wheel) start from a bare `git checkout` that
  contains no model files. Without this mechanism they would silently ship a
  degraded build that falls back to regex-only detection.
- A fresh `git clone` becomes fully functional with one command.
- Checksums make tampering or corruption impossible to miss.

### One-time setup: publish the models release

From the repo root on the machine that has the model files (verify first):

```bash
# 1. Verify every local artifact matches the manifest
python scripts/download_models.py --check

# 2. Create the release and upload assets.
#    Asset names = manifest path with "/" replaced by "__"
#    (release assets are a flat namespace).
TAG=models-v1.13.0
gh release create "$TAG" --title "Model artifacts v1.13.0" \
  --notes "SHA-256-pinned model artifacts. Consumed by scripts/download_models.py — do not download manually."

python - <<'EOF'
import json, shutil, subprocess, tempfile, os
manifest = json.load(open("scripts/model_manifest.json"))
tag = manifest["release_tag"]
with tempfile.TemporaryDirectory() as tmp:
    for art in manifest["artifacts"]:
        asset = os.path.join(tmp, art["path"].replace("/", "__"))
        shutil.copy(art["path"], asset)
        subprocess.run(["gh", "release", "upload", tag, asset, "--clobber"], check=True)
        print("uploaded", art["path"])
EOF

# 3. Prove the round-trip works: delete one file locally and re-download
python scripts/download_models.py
```

### When models are retrained

1. Train and place the new files at their usual paths.
2. Regenerate the manifest hashes:

   ```bash
   python - <<'EOF'
   import hashlib, json, os
   m = json.load(open("scripts/model_manifest.json"))
   for art in m["artifacts"]:
       art["sha256"] = hashlib.sha256(open(art["path"], "rb").read()).hexdigest()
       art["size"]   = os.path.getsize(art["path"])
   m["release_tag"] = "models-vX.Y.Z"   # bump to match the SDK release
   json.dump(m, open("scripts/model_manifest.json", "w"), indent=2)
   EOF
   ```

3. Create the new release tag and upload (same as one-time setup).
4. Commit the updated manifest — the manifest **is** the deployment pin.

### Where the download runs

| Consumer | Mode | Behavior when release missing |
| --- | --- | --- |
| `Dockerfile` (Cloud Run image) | best-effort | boots with rule-based fallbacks |
| CI `test` / `package` / `health-check` jobs | best-effort | tests run against fallbacks |
| `publish-pypi.yml` | **`--strict`** | **release fails — a wheel without models is never published** |
| A person who cloned the repo | manual: `python scripts/download_models.py` | warns, explains |

---

## 2. Releasing a new SDK version

The single source of truth for the version is **`pyproject.toml`**.
`fie/client.py` (`SDK_VERSION`), `fie/__init__.py` (`__version__`), and the
server's reported version (`config.py`) all resolve it from package metadata.

1. Bump `version` in `pyproject.toml`.
2. Update `CHANGELOG.md` and the README "What's new" section.
3. If models changed, follow §1 first and update `release_tag` in the manifest.
4. Verify locally: `python -m build && unzip -l dist/*.whl | grep fie/models`
5. Tag and push: `git tag v1.14.0 && git push origin v1.14.0`
   — `publish-pypi.yml` builds (strict model check) and publishes.
6. After PyPI propagates, verify the published wheel really bundles models:

   ```bash
   pip download fie-sdk --no-deps -d /tmp/fie-wheel
   python -c "import zipfile,glob; print([n for n in zipfile.ZipFile(glob.glob('/tmp/fie-wheel/*.whl')[0]).namelist() if 'models/' in n])"
   ```

---

## 3. Uptime monitoring (UptimeRobot — free)

1. Create an account at [uptimerobot.com](https://uptimerobot.com) (free: 50 monitors, 5-min interval).
2. Add a monitor:
   - Type: **HTTP(s)** — Keyword
   - URL: `https://failure-intelligence-system-800748790940.asia-south1.run.app/health`
   - Keyword: `healthy` (alerts also fire when the API degrades, not just when it dies)
   - Interval: 5 minutes
3. Add a second keyword monitor for the dashboard at
   `https://failure-intelligence-system.pages.dev`.
4. Set the alert contact to your email.

The `/health` endpoint is excluded from request logging (see `app/main.py`),
so monitor traffic does not pollute the logs. A status page can be published
from UptimeRobot for free (nice public link for the README).

---

## 4. Error tracking (Sentry — free tier)

Sentry is wired into `app/main.py` behind an env var. It is a no-op unless
`SENTRY_DSN` is set, and `send_default_pii=False` is hard-coded so prompts
and user data are never shipped.

1. Create a project at [sentry.io](https://sentry.io) (free: 5k errors/month) → FastAPI.
2. Copy the DSN.
3. Local: add `SENTRY_DSN=...` to `.env`.
   Cloud Run: add `SENTRY_DSN` to the GitHub `secrets` and append it to the
   `--set-env-vars` list in `.github/workflows/ci.yml` (deploy step).
4. Optional tuning: `SENTRY_ENVIRONMENT` (default `production`),
   `SENTRY_TRACES_SAMPLE_RATE` (default `0.1`).

---

## 5. Coverage (Codecov — free for public repos)

CI already produces `coverage.xml` (pytest `--cov`) and uploads it with
`codecov/codecov-action@v4`. The step is `continue-on-error` and can never
fail the build.

1. Sign in at [codecov.io](https://codecov.io) with GitHub and enable the repo.
2. Copy the upload token → add as the `CODECOV_TOKEN` repo secret.
3. Add the badge to README:

   ```markdown
   [![codecov](https://codecov.io/gh/AyushSingh110/Failure_Intelligence_System/graph/badge.svg)](https://codecov.io/gh/AyushSingh110/Failure_Intelligence_System)
   ```

---

## 6. Secrets hygiene on Cloud Run (optional upgrade)

Today secrets are injected via `--set-env-vars` from GitHub Secrets — anyone
with project *read* access in the GCP console can see them. GCP Secret
Manager's free tier (6 active secret versions, 10k accesses/month) covers
the critical ones:

```bash
echo -n "$MONGODB_URI" | gcloud secrets create mongodb-uri --data-file=-
echo -n "$JWT_SECRET_KEY" | gcloud secrets create jwt-secret --data-file=-
gcloud secrets add-iam-policy-binding mongodb-uri \
  --member "serviceAccount:<runtime-sa>" --role roles/secretmanager.secretAccessor
```

Then in the deploy step replace those entries in `--set-env-vars` with:

```
--set-secrets "MONGODB_URI=mongodb-uri:latest,JWT_SECRET_KEY=jwt-secret:latest"
```

---

## 7. Shrinking git history (one-time, do with care)

The pack is ~140 MB because binaries (images, notebooks output, node_modules)
were committed historically. **This rewrites history — every clone/fork must
re-clone afterwards. Do it on a quiet day, after backing up.**

```bash
pip install git-filter-repo
git clone --mirror https://github.com/AyushSingh110/Failure_Intelligence_System.git fie-mirror
cd fie-mirror

# Preview the biggest blobs first:
git filter-repo --analyze && cat filter-repo/analysis/blob-shas-and-paths.txt | head -30

# Strip the known offenders from all history:
git filter-repo \
  --path node_modules --path .DS_Store --path paper --path dist \
  --invert-paths

git push --force --mirror https://github.com/AyushSingh110/Failure_Intelligence_System.git
```

Afterwards: re-clone your working copy, re-protect the `main` branch
(force-push temporarily requires removing protection), and tell any forks.

---

## 8. Hugging Face Hub mirror (optional, good for visibility)

Mirroring models to HF Hub (free) gives them a model card, download stats,
and a citable home next to the Zenodo DOI.

```bash
pip install huggingface_hub
hf auth login
hf repo create AyushSingh110/fie-models --repo-type model
hf upload AyushSingh110/fie-models fie/models/ --include "*.pkl" --include "*.json"
```

Write the model card README with: training data provenance, eval results
(link the README tables), license (Apache-2.0), and intended use
(adversarial prompt detection — not content moderation).

---

## 9. Routine checks

| Cadence | Check |
| --- | --- |
| Weekly | `pip-audit -r requirements.txt` locally (CI runs it advisory-only) |
| Per release | §2 step 6 — verify the published wheel bundles models |
| Per retrain | §1 — regenerate manifest, upload release, commit manifest |
| Monthly | MongoDB Atlas free-tier storage (512 MB cap) — prune old inference records |
| Monthly | Groq/Serper key usage and rotation |
