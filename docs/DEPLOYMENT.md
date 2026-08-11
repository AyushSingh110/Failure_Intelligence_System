# Deployment

How to run FIE in production, and how to get it onto a free tier.

---

## torch has been removed (done)

This was the blocker. It is fixed. Recorded here because the numbers justify the
approach and the verification protocol is reusable.

`torch` existed solely to compute 384-dimensional MiniLM embeddings at two call
sites. It has been replaced with ONNX Runtime.

### Measured results

| | torch (CUDA build) | ONNX Runtime | Change |
| --- | --- | --- | --- |
| Disk | 4,668 MB | ~113 MB | **~41x smaller** |
| Embedding latency | 16.64 ms | 6.80 ms | **2.4x faster** |
| Detector warm-up | 12.6 s | 1.97 s | **6.4x faster** |
| End-to-end scan (median) | 38.2 ms | 24.6 ms | **36% faster** |
| Test suite | 32.2 s | 16.2 s | 2x faster |
| Embedding equivalence | reference | **cosine 1.000000** | identical |

Removing the whole torch stack (`torch`, `torchvision`, `torchaudio`,
`transformers`, `sentence-transformers`) reclaimed **4.5 GB** of disk.

### Why fp32 and not int8

int8 quantisation was tested and **rejected**:

| | Size | Latency | Cosine vs fp32 |
| --- | --- | --- | --- |
| fp32 | 90.3 MB | 6.80 ms | 1.000000 (vs torch) |
| int8 | 22.7 MB | 3.70 ms | **0.879 min / 0.946 mean — FAILED** |

All 27 corpus prompts fell below the 0.999 gate. Faster and smaller, but the
embeddings sit somewhere else in the space — the PAIR classifier's decision
boundary was fitted on the fp32 vectors, so int8 would have silently invalidated
every published benchmark number while still returning confident probabilities.

This is precisely the failure the gate exists to catch. Do not revisit int8
without re-fitting PAIR on int8 embeddings and re-measuring the full suite.

### Reproducing the check

```bash
pip install sentence-transformers            # reference implementation
python scripts/verify_onnx_equivalence.py    # gate: min cosine > 0.999
pytest tests/test_detection_golden.py        # confidences pinned to 4 dp
```

### How it is wired

`fie/onnx_encoder.py` reimplements the sentence-transformers pipeline exactly:
WordPiece tokenisation (256-token window) -> **mean** pooling weighted by the
attention mask -> L2 normalisation. Getting any of those three wrong produces
plausible-looking but subtly wrong vectors.

Both call sites prefer ONNX and fall back to sentence-transformers if it is
installed and the ONNX artifacts are missing:

- `fie/layers/pair.py` -> `_build_embedder()`
- `engine/encoder.py` -> `_build_backend()`

Force the reference path for A/B comparison with `FIE_EMBED_BACKEND=torch`.

### Distribution

The 90 MB fp32 model exceeds PyPI's per-file limit, so it ships as a GitHub
Release asset alongside the `.pkl` classifiers (`scripts/model_manifest.json`).
`fie/onnx_encoder.py` downloads it once on first use if absent — matching the
behaviour users already had, where sentence-transformers pulled ~90 MB from the
HuggingFace hub on first call. Disable with `FIE_NO_AUTO_DOWNLOAD=1` for
air-gapped deployments.

---

## Live deployment (current)

**https://ayush-singh9791-fie.hf.space** — Hugging Face Space, Docker SDK, free CPU tier.

Serves the demo at `/` and the full API at `/api/v1/*` from one HTTPS URL, so
the Cloudflare Pages dashboard needs no separate backend and no CORS gymnastics.

Deploy an update:

```bash
python deploy/huggingface/deploy_space.py --name fie --secrets-only   # config only
# or push the full build — see deploy/huggingface/DEPLOY.md
```

### Platform notes, learned the hard way

Two options were evaluated and rejected on evidence:

**Oracle Cloud Always Free.** Oracle *reclaims* Always Free compute when CPU
(95th percentile), network and memory all sit below 20% for a 7-day period —
precisely the profile of a low-traffic demo backend. Losing the instance without
warning is a worse failure mode than a Space sleeping and waking on request.

**A new Hugging Face Space on the free tier.** Creating Gradio or Docker Spaces
now requires PRO (`HTTP 402`). ZeroGPU is exempt for free accounts but requires a
genuine `@spaces.GPU` function, which a CPU-only guardrail does not have.

What worked: the PRO gate applies to *creating* a Space, not to pushing to one
that already exists. An existing, unused Docker Space was repurposed.

---

## Step 2 — Pick a host

Once the image is ~300 MB rather than ~3 GB, these all become viable at zero cost.

| Platform | Free tier | Cold start | Best for |
| --- | --- | --- | --- |
| **Hugging Face Spaces** | CPU basic, 16 GB RAM, no aggressive sleep | none | **The public demo — do this first** |
| **Fly.io** | small always-on allowance | ~1–2 s | Staging |
| **Render** | free web service, sleeps after 15 min idle | ~30 s from sleep | Backup |
| **Cloudflare Pages** | unlimited static | none | Dashboard (already deployed) |

### Why Hugging Face Spaces

Free, automatic HTTPS, no credit card, and it puts the project in front of
exactly the audience most likely to use and cite it. A Space where someone pastes
a prompt and watches all twelve layer scores light up is shareable in a way a
README table is not.

It also serves the **full API** alongside the demo, so the Cloudflare dashboard
points at the same URL. See [`deploy/huggingface/DEPLOY.md`](../deploy/huggingface/DEPLOY.md).

### A note on always-free VPS offers

Oracle Cloud Always Free was evaluated and rejected. Oracle **reclaims** Always
Free compute when CPU (95th percentile), network and memory all sit below 20%
for a 7-day period — which is precisely the profile of a low-traffic demo
backend. Losing the instance outright is a worse failure mode than a Space
sleeping and waking on request, and it arrives without warning.

---

## Step 3 — Production configuration

### Required environment

```env
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGODB_DB_NAME=fie_database
JWT_SECRET_KEY=<32+ random chars>
ADMIN_EMAIL=you@example.com
GROQ_API_KEY=gsk_...            # hallucination monitoring only
```

### Strongly recommended

```env
# Block when the scanner itself fails, instead of forwarding unscanned prompts.
# See docs/PRODUCTION_ENGINEERING.md §2.
FIE_SCAN_FAILURE_MODE=closed

# Size to your instance. On 1 vCPU, lower it — 12 threads on one core adds
# context-switch overhead without parallelism.
FIE_LAYER_POOL_SIZE=8

CORS_ALLOWED_ORIGINS=https://your-dashboard.pages.dev
SENTRY_DSN=<optional, never ships prompt text>
```

### Health probes

Wire these correctly — conflating them causes restart loops.

| Probe | Endpoint | Why |
| --- | --- | --- |
| Liveness | `GET /health` | Cheap, no network, no model load. A liveness probe that touches slow dependencies times out under load and kills the container. |
| Readiness | `GET /ready` | Returns **503 until warm-up completes**. Prevents traffic reaching an instance whose classifier has not loaded. |
| Diagnosis | `GET /health/deep` | Pings every dependency. On-call and dashboards only — too expensive for a probe. |

Example (Kubernetes-style; adapt for your platform):

```yaml
livenessProbe:
  httpGet: { path: /health, port: 8080 }
  periodSeconds: 10
readinessProbe:
  httpGet: { path: /ready, port: 8080 }
  periodSeconds: 5
  failureThreshold: 30      # allow ~150s for model warm-up
```

`/ready` is what makes zero-downtime rolling deploys work: the old instance keeps
serving until the new one is genuinely warm, not merely running.

---

## Step 4 — Docker

The current `Dockerfile` installs the full `requirements.txt`, including torch.
After the ONNX migration, use a slim runtime image and a multi-stage build:

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080
COPY . .
RUN python scripts/download_models.py && mkdir -p /app/storage
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
```

`build-essential` is only needed to compile native wheels. Once torch is gone,
every remaining dependency ships a manylinux wheel, so it can be dropped —
another ~200 MB.

---

## Step 5 — Frontend

The Cloudflare Pages dashboard currently points at a Cloud Run backend that is
returning 429/503. A visitor's first click fails, which is worse than having no
demo at all.

Two things to do:

1. **Remove the "Live" badge from the README** until a backend is actually up.
2. **Make the dashboard degrade honestly.** Bundle a JSON fixture and have
   `Frontend/src/lib/api.js` fall back to it with a visible
   *"demo data — backend offline"* banner when the API is unreachable. A dead
   backend should produce an obviously-sample dashboard, not a broken one.

---

## Deployment checklist

- [ ] ONNX export verified at `min cosine > 0.999`
- [ ] Full benchmark suite re-run and matching published numbers
- [ ] `pytest tests/test_detection_golden.py` passes
- [ ] `scikit-learn==1.7.2` installed (see PRODUCTION_ENGINEERING.md §11)
- [ ] `FIE_SCAN_FAILURE_MODE=closed`
- [ ] `JWT_SECRET_KEY` is 32+ random chars, not a placeholder
- [ ] `CORS_ALLOWED_ORIGINS` restricted to your dashboard origin
- [ ] Liveness on `/health`, readiness on `/ready`
- [ ] `.env` is not committed (`git check-ignore .env`)
- [ ] Models fetched and SHA-256 verified (`python scripts/download_models.py`)
