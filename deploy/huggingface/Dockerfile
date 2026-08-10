# Hugging Face Space image (Docker SDK).
#
# Differs from the root Dockerfile in three Space-specific ways:
#   1. Listens on 7860 (the Spaces convention; also declared as app_port in the
#      Space README frontmatter).
#   2. Runs as uid 1000 named "user" — Spaces mount the working tree for that
#      uid, and writing as root produces permission errors at runtime.
#   3. Installs gradio and bakes the model artifacts into the image, so the
#      first visitor never waits on a 90 MB download.
#
# Torch-free, so this builds in a couple of minutes and stays well inside the
# free CPU Basic tier (2 vCPU / 16 GB).

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# gradio is needed only for the Space UI, not by the library or the API server.
RUN pip install --user --no-warn-script-location -r requirements.txt "gradio>=4.44,<6"


FROM python:3.11-slim

# Spaces run the container as uid 1000. Create it before copying anything so
# ownership is right from the start.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    # 2 vCPU on CPU Basic — a 16-thread layer pool would just thrash.
    FIE_LAYER_POOL_SIZE=4 \
    # Public demo: never forward an unscanned prompt if the scanner itself fails.
    FIE_SCAN_FAILURE_MODE=closed \
    # Models are baked in below; no runtime fetch should ever be needed.
    FIE_NO_AUTO_DOWNLOAD=1

WORKDIR $HOME/app

COPY --from=builder --chown=user /root/.local /home/user/.local
COPY --chown=user . .

# Bake model artifacts into the image (SHA-256 verified). Strict: a Space that
# boots without its models would serve confident verdicts from a pipeline whose
# main classifier never loaded.
RUN python scripts/download_models.py --strict && mkdir -p storage

EXPOSE 7860

CMD ["python", "deploy/huggingface/space_app.py"]
