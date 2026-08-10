"""
ONNX Runtime sentence encoder — a drop-in replacement for SentenceTransformer.

WHY THIS EXISTS
---------------
`torch` was the single largest dependency in this project by an enormous margin
(4.67 GB for the CUDA build; 1.32 GB even CPU-only), and it existed solely to
compute 384-dimensional MiniLM embeddings at two call sites. That one dependency
was why FIE did not fit on any free hosting tier and why `pip install` failed on
machines with limited disk.

    torch + transformers + sentence-transformers   ~4,760 MB
    onnxruntime + tokenizers + model.onnx            ~113 MB

EQUIVALENCE
-----------
This reimplements exactly what `SentenceTransformer("all-MiniLM-L6-v2").encode(...)`
does, in three steps that must all match or the embeddings drift:

  1. WordPiece tokenisation, truncated to 256 tokens
  2. MEAN pooling over the token axis, weighted by the attention mask so padding
     contributes nothing  (this model uses mean pooling, NOT [CLS])
  3. L2 normalisation, so cosine similarity is a plain dot product

Getting any of these wrong produces embeddings that look plausible but sit
somewhere else in the space — the PAIR classifier's decision boundary was fitted
on the torch vectors, so "close enough" is not good enough. Equivalence is
enforced by scripts/verify_onnx_equivalence.py, which requires cosine similarity
> 0.999 against torch across a mixed attack/benign corpus.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Anchored to the package root so this survives file moves — the same class of
# bug that silently broke meta-classifier loading during the layer split.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _PACKAGE_ROOT / "models" / "minilm-onnx"

# MiniLM-L6-v2 was trained with a 256-token window. Longer inputs are truncated,
# matching sentence-transformers' default for this model.
_MAX_TOKENS = 256
_EMBED_DIM = 384


# Where the exported model lives when it is not bundled. The fp32 ONNX file is
# 90 MB, over PyPI's per-file limit, so it ships as a GitHub Release asset
# instead of inside the wheel — the same channel already used for the .pkl
# classifiers and the FAISS index.
_RELEASE_TAG = "models-v1.18.0"
_RELEASE_BASE = (
    "https://github.com/AyushSingh110/Failure_Intelligence_System/"
    f"releases/download/{_RELEASE_TAG}"
)
_MODEL_FILES = ("model.onnx", "tokenizer.json")

_download_lock = threading.Lock()


def _ensure_model_downloaded(model_dir: Path) -> None:
    """
    Fetch the ONNX artifacts on first use if they are not already present.

    This preserves the behaviour users already had: with sentence-transformers,
    the first call silently downloaded ~90 MB of MiniLM weights from the
    HuggingFace hub. Doing nothing here would have made `pip install fie-sdk[ml]`
    ship a permanently dead PAIR layer — a regression disguised as a size win.

    Never raises: on failure the caller falls back to sentence-transformers, or
    reports the layer unavailable. Opt out with FIE_NO_AUTO_DOWNLOAD=1 for
    air-gapped deployments that pre-stage artifacts.
    """
    import os

    if os.environ.get("FIE_NO_AUTO_DOWNLOAD", "").strip() in ("1", "true", "yes"):
        logger.info("encoder=onnx auto-download disabled by FIE_NO_AUTO_DOWNLOAD")
        return

    with _download_lock:
        # Re-check under the lock: several layers start concurrently on the
        # first scan and would otherwise each pull 90 MB.
        if all((model_dir / f).exists() for f in _MODEL_FILES):
            return

        import urllib.request

        model_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "encoder=onnx model absent, downloading ~90 MB from %s "
            "(one time; set FIE_NO_AUTO_DOWNLOAD=1 to disable)", _RELEASE_TAG,
        )
        for fname in _MODEL_FILES:
            target = model_dir / fname
            if target.exists():
                continue
            tmp = target.with_suffix(target.suffix + ".partial")
            try:
                urllib.request.urlretrieve(f"{_RELEASE_BASE}/{fname}", tmp)
                # Atomic rename: a half-written model.onnx that looks complete
                # would fail in ONNX Runtime with an unhelpful parse error.
                tmp.replace(target)
                logger.info("encoder=onnx downloaded %s", fname)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                logger.warning(
                    "encoder=onnx download failed for %s (%s: %s)",
                    fname, type(exc).__name__, exc,
                )
                return


class OnnxEncoder:
    """
    Thread-safe lazy-loading ONNX sentence encoder.

    Mirrors the SentenceTransformer surface used by FIE (`encode`), so both call
    sites can swap backends without changing their logic.
    """

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._loaded = False
        self._failed = False
        self._reason = ""

    # ── Availability ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        self._load()
        return self._loaded and not self._failed

    def status(self) -> dict:
        """Non-blocking health snapshot. Never triggers a load."""
        if not self._loaded:
            return {"backend": "onnx", "state": "not_loaded", "path": str(self._model_dir)}
        if self._failed:
            return {"backend": "onnx", "state": "failed", "reason": self._reason}
        return {"backend": "onnx", "state": "ready", "path": str(self._model_dir)}

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> bool:
        if self._loaded:
            return not self._failed

        with self._lock:
            if self._loaded:
                return not self._failed
            # Set before attempting, so a failed load is never retried on every
            # call — that would stall every request thread behind a retry storm.
            self._loaded = True
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                model_path = self._model_dir / "model.onnx"
                tok_path = self._model_dir / "tokenizer.json"
                if not model_path.exists() or not tok_path.exists():
                    _ensure_model_downloaded(self._model_dir)
                if not model_path.exists() or not tok_path.exists():
                    raise FileNotFoundError(
                        f"ONNX model not found in {self._model_dir} and could not "
                        "be downloaded. Fetch it with: "
                        "python scripts/download_models.py --strict"
                    )

                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Bound intra-op threads. ORT defaults to one thread per core,
                # which on a shared host oversubscribes badly: FIE already runs
                # 12 detection layers concurrently, and each nested ORT pool
                # would multiply against that.
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1

                self._session = ort.InferenceSession(
                    str(model_path), opts, providers=["CPUExecutionProvider"],
                )
                self._tokenizer = Tokenizer.from_file(str(tok_path))
                self._tokenizer.enable_truncation(max_length=_MAX_TOKENS)
                self._tokenizer.enable_padding()

                self._input_names = {i.name for i in self._session.get_inputs()}
                logger.info(
                    "encoder=onnx status=ready path=%s inputs=%s",
                    self._model_dir, sorted(self._input_names),
                )
            except ImportError as exc:
                self._failed = True
                self._reason = f"missing dependency: {exc}"
                logger.warning(
                    "encoder=onnx status=unavailable reason=%s "
                    "action='pip install onnxruntime tokenizers'", self._reason,
                )
            except Exception as exc:
                self._failed = True
                self._reason = f"{type(exc).__name__}: {exc}"
                logger.warning("encoder=onnx status=unavailable reason=%s", self._reason)

        return not self._failed

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        sentences: str | list[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,   # accepted and ignored: API parity
        convert_to_numpy: bool = True,     # accepted and ignored: always numpy
        batch_size: int = 32,
        **_ignored,
    ) -> np.ndarray:
        """
        Encode text to L2-normalised float32 embeddings.

        Signature mirrors SentenceTransformer.encode so callers need no changes.
        A single string returns shape (384,); a list returns (n, 384) — same
        convention as sentence-transformers.
        """
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)

        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)

        if not self._load():
            raise RuntimeError(f"ONNX encoder unavailable: {self._reason}")

        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            out.append(self._encode_batch(texts[start:start + batch_size], normalize_embeddings))
        vecs = np.vstack(out)

        return vecs[0] if single else vecs

    def _encode_batch(self, texts: list[str], normalize: bool) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention}
        # BERT-family exports usually take token_type_ids too; some do not.
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )

        # (batch, tokens, 384)
        last_hidden = self._session.run(None, feed)[0]

        # ── Mean pooling ──────────────────────────────────────────────────────
        # Weight by the attention mask so padding tokens contribute nothing.
        # all-MiniLM-L6-v2 uses MEAN pooling; taking [CLS] instead would produce
        # a different, subtly wrong embedding space.
        mask = attention[..., None].astype(np.float32)          # (b, t, 1)
        summed = (last_hidden * mask).sum(axis=1)               # (b, 384)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts

        if normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, a_min=1e-12, a_max=None)

        return pooled.astype(np.float32)


# ── Module-level singleton ────────────────────────────────────────────────────

_encoder: OnnxEncoder | None = None
_encoder_lock = threading.Lock()


def get_onnx_encoder(model_dir: Path | str | None = None) -> OnnxEncoder:
    """Return the process-wide ONNX encoder (created once)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is None:
            _encoder = OnnxEncoder(model_dir)
    return _encoder


def onnx_available(model_dir: Path | str | None = None) -> bool:
    """True if the ONNX backend can be used. Safe to call at import time."""
    try:
        return get_onnx_encoder(model_dir).available
    except Exception:
        return False
