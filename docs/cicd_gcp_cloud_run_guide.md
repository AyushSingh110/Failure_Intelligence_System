# CI/CD Pipeline on Google Cloud Run — Complete Setup Guide

A reusable step-by-step guide to set up a production-grade CI/CD pipeline using GitHub Actions, Workload Identity Federation, and Google Cloud Run. No service account JSON keys required.

---

## What This Sets Up

Every push to `main` automatically:
1. Scans for hardcoded secrets (gitleaks)
2. Audits dependencies for CVEs (pip-audit)
3. Lints code (ruff)
4. Runs tests
5. Builds a Docker image and pushes to Artifact Registry
6. Deploys to Cloud Run
7. Verifies the live deployment hits `/health` successfully

PRs get full CI but never trigger a deploy.

---

## Prerequisites

- A GitHub repository (public or private)
- A Google Cloud project with billing enabled
- Python app with a `Dockerfile` and `requirements.txt`
- A `/health` endpoint that returns `{"status": "healthy"}` or `{"status": "degraded"}`

---

## Part 1 — Google Cloud Setup

### Step 1 — Enable Required APIs

In GCP Console → **APIs & Services → Enable APIs**, enable:
- `Cloud Run API`
- `Artifact Registry API`
- `IAM Credentials API`
- `Security Token Service API`

### Step 2 — Create Artifact Registry Repository

1. Go to **Artifact Registry → + Create Repository**
2. Fill in:
   - **Name:** `your-app` (e.g. `fie`)
   - **Format:** Docker
   - **Mode:** Standard
   - **Region:** your region (e.g. `asia-south1`)
3. Click **Create**

If a `cloud-run-source-deploy` repository already exists, you can use that instead.

### Step 3 — Create a Dedicated Service Account

Never use the default compute service account for CI/CD.

1. Go to **IAM & Admin → Service Accounts → + Create Service Account**
2. Name: `github-actions-deploy`
3. Click **Create and Continue**
4. Add these roles:
   - `Cloud Run Admin`
   - `Artifact Registry Writer`
   - `Service Account User`
5. Click **Done** (skip the optional Step 3)

### Step 4 — Set Up Workload Identity Federation

This allows GitHub Actions to authenticate to GCP without any stored JSON keys.

#### 4a — Create a Workload Identity Pool

1. Go to **IAM & Admin → Workload Identity Federation → + Create Pool**
2. Fill in:
   - **Name:** `github-actions`
   - **Pool ID:** `github-actions`
3. Click **Continue**

#### 4b — Add a Provider to the Pool

On the same screen (Add a provider):
- **Provider type:** OpenID Connect (OIDC)
- **Provider name:** `github`
- **Provider ID:** `github`
- **Issuer URL:** `https://token.actions.githubusercontent.com`

Click **Continue**.

#### 4c — Configure Provider Attributes

In the attribute mapping section:
| Google | OIDC |
|---|---|
| `google.subject` | `assertion.sub` |
| `attribute.repository` | `assertion.repository` |

Under **Attribute Conditions**, click **Add condition** and enter:
```
assertion.repository == "YOUR_GITHUB_USERNAME/YOUR_REPO_NAME"
```

Click **Save**.

#### 4d — Note the Provider Resource Name

After saving, edit the provider and copy the resource name — it looks like:
```
projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-actions/providers/github
```
This is your `GCP_WORKLOAD_IDENTITY_PROVIDER` secret value.

### Step 5 — Grant WIF Access to the Service Account

1. Go to **IAM & Admin → Service Accounts → `github-actions-deploy`**
2. Click the **Permissions** tab → **Grant Access**
3. **New principals:**
   ```
   principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-actions/attribute.repository/GITHUB_USERNAME/REPO_NAME
   ```
4. **Role:** `Service Account Token Creator`
5. Click **Save**

### Step 6 — Grant actAs Permission on Default Compute Service Account

Cloud Run uses the default compute service account internally. Your deploy SA needs permission to act as it.

1. Go to **IAM & Admin → Service Accounts**
2. Click on `PROJECT_NUMBER-compute@developer.gserviceaccount.com`
3. **Permissions tab → Grant Access**
4. **New principals:** `github-actions-deploy@YOUR_PROJECT_ID.iam.gserviceaccount.com`
5. **Role:** `Service Account User`
6. Click **Save**

---

## Part 2 — GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**.

Add all of these:

| Secret Name | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID (e.g. `my-project-123`, not the number) |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/NUMBER/locations/global/workloadIdentityPools/github-actions/providers/github` |
| `GCP_SERVICE_ACCOUNT` | `github-actions-deploy@YOUR_PROJECT_ID.iam.gserviceaccount.com` |
| `MONGODB_URI` | Your MongoDB Atlas connection string |
| `GROQ_API_KEY` | Your Groq API key |
| `JWT_SECRET_KEY` | Your JWT secret (min 32 chars) |
| `ADMIN_EMAIL` | Your admin email |
| `CORS_ALLOWED_ORIGINS` | Your production frontend URL |

**Important:** Secret names are case-sensitive. Double-check the spelling before saving.

---

## Part 3 — GitHub Actions Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # ── Security: scan for hardcoded secrets ─────────────────────────────
  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Scan for hardcoded secrets (gitleaks)
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  # ── Security: dependency vulnerability audit ──────────────────────────
  dependency-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install pip-audit
        run: pip install pip-audit
      - name: Audit Python dependencies
        run: pip-audit -r requirements.txt || true

  # ── Lint ──────────────────────────────────────────────────────────────
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install ruff
        run: pip install ruff
      - name: Lint
        run: ruff check . --select E,F,W --ignore E501,F401,E402

  # ── Tests ─────────────────────────────────────────────────────────────
  test:
    runs-on: ubuntu-latest
    needs: [lint]
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install dependencies
        run: pip install -r requirements.txt pytest
      - name: Run tests
        run: pytest tests/ -v --tb=short

  # ── Health-check ──────────────────────────────────────────────────────
  health-check:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Start server and check /health
        env:
          JWT_SECRET_KEY: "ci-secret-key-minimum-32-chars-long"
        run: |
          uvicorn app.main:app --host 0.0.0.0 --port 8000 &
          echo "Waiting for server..."
          for i in {1..20}; do
            curl -sf http://localhost:8000/health > /tmp/health.json 2>/dev/null && break
            echo "Attempt $i failed, retrying in 2s..."
            sleep 2
          done
          cat /tmp/health.json
          python -c "
          import json
          data = json.load(open('/tmp/health.json'))
          assert data['status'] in ('healthy', 'degraded'), f'Bad status: {data}'
          print('Health check passed:', data['status'])
          "

  # ── CD: Deploy to Cloud Run (main branch only) ────────────────────────
  deploy:
    runs-on: ubuntu-latest
    needs: [test, health-check]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    permissions:
      contents: read
      id-token: write

    env:
      PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
      REGION: asia-south1                          # change to your region
      SERVICE: your-cloud-run-service-name         # change to your service name
      IMAGE: asia-south1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/your-repo/backend

    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev --quiet

      - name: Build and push Docker image
        run: |
          docker build -t ${{ env.IMAGE }}:${{ github.sha }} -t ${{ env.IMAGE }}:latest .
          docker push ${{ env.IMAGE }}:${{ github.sha }}
          docker push ${{ env.IMAGE }}:latest

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ${{ env.SERVICE }} \
            --image ${{ env.IMAGE }}:${{ github.sha }} \
            --region ${{ env.REGION }} \
            --platform managed \
            --allow-unauthenticated \
            --set-env-vars "KEY=${{ secrets.YOUR_SECRET }}" \
            --quiet

      - name: Verify deployment
        run: |
          URL=$(gcloud run services describe ${{ env.SERVICE }} \
            --region ${{ env.REGION }} \
            --format 'value(status.url)')
          echo "Deployed to: $URL"
          curl -sf "$URL/health"
```

---

## Part 4 — Rollback

If a deployment breaks production, roll back instantly:

```bash
# List recent revisions
gcloud run revisions list --service YOUR_SERVICE --region YOUR_REGION

# Roll back to a specific image SHA
gcloud run deploy YOUR_SERVICE \
  --image asia-south1-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/backend:PREVIOUS_SHA \
  --region YOUR_REGION
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `workload_identity_provider or credentials_json must be specified` | Secret name typo in GitHub | Check exact spelling — `GCP_WORKLOAD_IDENTITY_PROVIDER` |
| `Permission 'iam.serviceAccounts.getAccessToken' denied` | WIF principal not granted Token Creator role on SA | Step 5 above — grant `Service Account Token Creator` to the `principalSet://` principal |
| `Permission 'iam.serviceaccounts.actAs' denied` | Deploy SA can't act as compute SA | Step 6 above — grant `Service Account User` on the default compute SA |
| `Unauthenticated request` on docker push | Docker not configured for Artifact Registry | Make sure `gcloud auth configure-docker` runs before `docker push` |
| `Health check: json.decoder.JSONDecodeError` | Server not ready when curl ran | Use retry loop (20 attempts × 2s) instead of fixed `sleep` |
| `Service account key creation is disabled` | Org policy blocks JSON keys | Use Workload Identity Federation instead (this guide) |

---

## Costs

| Service | Free Tier | Typical Cost |
|---|---|---|
| GitHub Actions | Unlimited for public repos, 2000 min/month free for private | $0 |
| Cloud Run | 2M requests/month + 360K GB-seconds free | $0 for small projects |
| Artifact Registry | 0.5 GB free storage | ~$0.10/GB/month after free tier |
| Workload Identity Federation | Free | $0 |

Set a **$5/month billing alert** in GCP → Billing → Budgets & Alerts as a safety net.
