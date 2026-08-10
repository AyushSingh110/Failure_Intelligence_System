# FIE backend image.
#
# Multi-stage, torch-free. Since sentence-transformers/torch was replaced with
# ONNX Runtime (see docs/DEPLOYMENT.md), every remaining dependency ships a
# prebuilt wheel — so the runtime stage needs no compiler and no CUDA libraries.
#
#   before: python:3.11-slim + build-essential + torch  -> ~3 GB
#   after:  python:3.11-slim + onnxruntime              -> ~400 MB
#
# That is what makes Oracle Cloud Always Free (and every other free tier) viable.

# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# build-essential lives ONLY here. Anything that needs compiling is compiled in
# this stage and never reaches the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --user --no-warn-script-location -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/home/fie/.local/bin:$PATH \
    PORT=8080 \
    # Block rather than forward unscanned prompts if the scanner itself fails.
    # See docs/PRODUCTION_ENGINEERING.md section 2.
    FIE_SCAN_FAILURE_MODE=closed \
    # Oracle Always Free ARM instances give plenty of cores, but the layer pool
    # is per-request concurrency, not throughput. 8 is a sane default; raise it
    # if you have headroom, lower it to 4 on a 1-vCPU box.
    FIE_LAYER_POOL_SIZE=8

# curl is kept for container-level healthchecks. Nothing else is installed.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin fie

WORKDIR /app

COPY --from=builder /root/.local /home/fie/.local
COPY . .

# Fetch and SHA-256-verify model artifacts (including the 90 MB ONNX encoder).
# Strict: a container that boots without its models would serve confident
# verdicts from a pipeline whose main classifier never loaded.
RUN python scripts/download_models.py --strict \
    && mkdir -p /app/storage \
    && chown -R fie:fie /app /home/fie/.local

USER fie

EXPOSE 8080

# Readiness, not liveness. /ready returns 503 until models are warm, so an
# orchestrator will not route traffic to a cold instance.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/ready" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
