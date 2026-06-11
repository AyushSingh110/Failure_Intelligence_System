FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# System packages kept minimal; build tools help with native Python wheels when needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Model artifacts (.pkl classifiers, FAISS index) are not in git. When the
# image is built from a CI checkout they are missing, so fetch them from the
# pinned GitHub Release (SHA-256 verified by scripts/model_manifest.json).
# Local builds where the files already exist skip the download entirely.
# Best-effort by design: the server boots with rule-based fallbacks if the
# release is unreachable — flip to `--strict` to make missing models fatal.
RUN python scripts/download_models.py

# Ensure runtime-generated files have a writable home inside the container.
RUN mkdir -p /app/storage

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
