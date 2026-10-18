"""
E53 shared helpers. Research-only; never imported by production code.

Hermetic by construction:
  * a socket guard is installed on import: every outbound connection attempt
    raises and is appended to logs/network_attempts.log;
  * the `fie` package is never imported. The production ONNX encoder module is
    loaded from its file, after its model files are hash-checked;
  * every output is written with exclusive-create, never overwritten.
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

HERE = Path(__file__).resolve().parent.parent           # .../e53_frozen_encoder_head_diagnostic
REPO = HERE.parents[2]                                   # repository root
LOGS = HERE / "logs"
CACHE = HERE / "cache"
MANIFESTS = HERE / "manifests"
RESULTS = HERE / "results"

os.environ.setdefault("FIE_NO_TELEMETRY", "1")
os.environ.setdefault("FIE_NO_AUTO_DOWNLOAD", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


# ── network guard ────────────────────────────────────────────────────────────

class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "network_attempts.log", "a", encoding="utf-8") as f:
        f.write(f"{now()} blocked outbound attempt args={str(args)[:200]}\n")
    raise NetworkBlocked("E53 is hermetic: network access is forbidden")


socket.socket.connect = _blocked          # type: ignore[assignment]
socket.socket.connect_ex = _blocked       # type: ignore[assignment]
socket.create_connection = _blocked       # type: ignore[assignment]
socket.getaddrinfo = _blocked             # type: ignore[assignment]

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))         # for the stdlib-only `evals` loaders


# ── small utilities ──────────────────────────────────────────────────────────

def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_new(path: Path, data: str | bytes) -> Path:
    """Exclusive create. Refuses to overwrite anything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if isinstance(data, bytes) else "x"
    kwargs = {} if isinstance(data, bytes) else {"encoding": "utf-8", "newline": "\n"}
    with open(path, mode, **kwargs) as f:
        f.write(data)
    return path


def write_json_new(path: Path, obj) -> Path:
    return write_new(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"- {now()} {msg}\n"
    with open(LOGS / "RUN_LOG.md", "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── pinned artifacts ─────────────────────────────────────────────────────────

PAIR_PREFIX = "Represent this text for security threat classification: "
SEED = 42

PINNED = {
    "fie/models/minilm-onnx/model.onnx":          "57eb46cc82cd048d1986b0a4c30d50e4a87fc06d15773075b74441748f8ed2ea",
    "fie/models/minilm-onnx/tokenizer.json":      "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "fie/models/pair_intent_classifier_v6_3b.pkl": "9c682b285a9fa20519da0c764c34b5ed49702c663f64ba8e39c8c1f728702514",
    "fie/models/pair_intent_meta_v6_3b.json":     "5babb7ec71c8a6e809379a3290c0c16fb0d35d696af0f17d2dd27ccc0b35c27f",
}

TRAIN_FILES = [   # (path, weight) in the v6.3b recipe's order
    ("data/pair_training/train.decontam.jsonl", 1.0),
    ("data/pair_training_v6/dataset_v6.decontam.jsonl", 1.0),
    ("data/pair_training_v6/augmented_v6.decontam.jsonl", 2.0),
    ("data/pair_training_v6/augmented_v6_3_train.jsonl", 2.0),
    ("data/pair_training_v6/augmented_v6_3b_benign_train.jsonl", 2.0),
]
CAL_FILES = ["data/pair_training/val.jsonl", "data/pair_training_v6/augmented_v6_3_test.jsonl"]
EXTRA_EVAL_FILES = ["data/pair_training/test.jsonl", "data/pair_training_v6/augmented_v6_3b_benign_test.jsonl"]

STD_DATASETS = ["xstest_safe", "xstest_unsafe", "orbench_hard", "jailbreakbench",
                "harmbench", "strongreject", "sorrybench"]
ATTACK4 = ["jailbreakbench", "harmbench", "strongreject", "sorrybench"]
BL0002 = REPO / "evals" / "baselines" / "BL-0002_pair-v6.3b_fie-70de7053"


def verify_pinned() -> dict:
    out = {}
    for rel, expected in PINNED.items():
        got = sha256_file(REPO / rel)
        if got != expected:
            raise SystemExit(f"STOP: {rel} sha256 {got} != pinned {expected}")
        out[rel] = got
    return out


# ── training corpus (exact v6.3b recipe) ─────────────────────────────────────

def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def _hash(t: str) -> str:
    return hashlib.sha1(_norm(t).encode("utf-8")).hexdigest()


def _lab(r: dict) -> int:
    return 1 if r.get("label", 0) in (1, "1", "attack", True) else 0


def build_training_corpus() -> list[dict]:
    rows, seen = [], set()
    for rel, weight in TRAIN_FILES:
        for r in read_jsonl(REPO / rel):
            p = (r.get("prompt") or "").strip()
            if not p or _hash(p) in seen:
                continue
            seen.add(_hash(p))
            rows.append({
                "text": p[:512], "label": _lab(r), "weight": weight,
                "source": r.get("source"), "category": r.get("category"),
                "file": rel, "mode": r.get("mode"),
            })
    return rows


# ── encoder (production module, loaded by file path) ─────────────────────────

_ENCODER = None


def encoder():
    global _ENCODER
    if _ENCODER is None:
        verify_pinned()
        spec = importlib.util.spec_from_file_location("e53_onnx_encoder", REPO / "fie" / "onnx_encoder.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        enc = mod.OnnxEncoder(model_dir=REPO / "fie" / "models" / "minilm-onnx")
        if not enc.available:
            raise SystemExit(f"STOP: ONNX encoder unavailable: {enc.status()}")
        _ENCODER = enc
    return _ENCODER


def embed_one(text: str):
    """Exactly what production does: one prefixed text per call, normalised."""
    return encoder().encode([PAIR_PREFIX + text], normalize_embeddings=True)[0]
