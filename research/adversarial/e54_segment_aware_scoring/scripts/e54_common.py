"""
E54 shared helpers. Research-only; never imported by production code.

Hermetic: a socket guard is installed on import (any outbound attempt raises and is
logged); the `fie` package is never imported (the production ONNX encoder module is
loaded from its file after hash checks); every output is exclusive-create.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
LOGS, CACHE, MANIFESTS, RECORDS, RESULTS = (HERE / d for d in ("logs", "cache", "manifests", "records", "results"))
E53 = REPO / "research" / "adversarial" / "e53_frozen_encoder_head_diagnostic"
BL0002 = REPO / "evals" / "baselines" / "BL-0002_pair-v6.3b_fie-70de7053"

for k, v in {"FIE_NO_TELEMETRY": "1", "FIE_NO_AUTO_DOWNLOAD": "1", "HF_HUB_OFFLINE": "1",
             "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}.items():
    os.environ.setdefault(k, v)
sys.dont_write_bytecode = True


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _blocked(*args, **kwargs):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "network_attempts.log", "a", encoding="utf-8") as f:
        f.write(f"{now()} blocked outbound attempt args={str(args)[:200]}\n")
    raise RuntimeError("E54 is hermetic: network access is forbidden")


socket.socket.connect = _blocked          # type: ignore[assignment]
socket.socket.connect_ex = _blocked       # type: ignore[assignment]
socket.create_connection = _blocked       # type: ignore[assignment]
socket.getaddrinfo = _blocked             # type: ignore[assignment]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))         # stdlib-only `evals` loaders


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_hash(t: str) -> str:
    return hashlib.sha1(" ".join(t.lower().split()).encode("utf-8")).hexdigest()


def write_new(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        with open(path, "xb") as f:
            f.write(data)
    else:
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            f.write(data)
    return path


def _json_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def write_json_new(path: Path, obj) -> Path:
    return write_new(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def write_jsonl_new(path: Path, rows) -> Path:
    return write_new(path, "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"- {now()} {msg}\n"
    with open(LOGS / "RUN_LOG.md", "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


PAIR_PREFIX = "Represent this text for security threat classification: "
PINNED = {
    "fie/models/minilm-onnx/model.onnx":           "57eb46cc82cd048d1986b0a4c30d50e4a87fc06d15773075b74441748f8ed2ea",
    "fie/models/minilm-onnx/tokenizer.json":       "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "fie/models/pair_intent_classifier_v6_3b.pkl": "9c682b285a9fa20519da0c764c34b5ed49702c663f64ba8e39c8c1f728702514",
    "fie/onnx_encoder.py":                          None,   # recorded, not pinned
}


def verify_pinned() -> dict:
    out = {}
    for rel, expected in PINNED.items():
        got = sha256_file(REPO / rel)
        if expected and got != expected:
            raise SystemExit(f"STOP: {rel} sha256 {got} != pinned {expected}")
        out[rel] = got
    return out


def verify_e53() -> None:
    """E53 outputs are read, never written; refuse to run if any checksum differs."""
    for line in (E53 / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        if sha256_file(E53 / rel) != digest:
            raise SystemExit(f"STOP: E53 artifact changed: {rel}")


class Scorer:
    """Frozen PAIR v6.3b: production ONNX encoder (one text per call) + shipped head."""

    def __init__(self):
        import joblib
        verify_pinned()
        spec = importlib.util.spec_from_file_location("e54_onnx_encoder", REPO / "fie" / "onnx_encoder.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.enc = mod.OnnxEncoder(model_dir=REPO / "fie" / "models" / "minilm-onnx")
        if not self.enc.available:
            raise SystemExit(f"STOP: ONNX encoder unavailable: {self.enc.status()}")
        # The approved, hash-verified shipped artifact (same safe path as E53).
        self.head = joblib.load(REPO / "fie" / "models" / "pair_intent_classifier_v6_3b.pkl")
        self.calls = 0

    def score(self, text: str) -> float:
        self.calls += 1
        v = self.enc.encode([PAIR_PREFIX + text], normalize_embeddings=True)
        return float(self.head.predict_proba(v)[0, 1])
