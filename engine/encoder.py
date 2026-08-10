from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)

# Settings are resolved lazily so importing this module never triggers env
# validation — matters for CLI/test paths that never encode anything.
_settings = None


def _lazy_settings():
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _build_backend(model_name: str):
    """
    Return (embedder, backend_name), preferring ONNX Runtime over torch.

    Same rationale as fie/layers/pair.py::_build_embedder — ONNX is ~45x smaller
    on disk, ~2.4x faster, and produces embeddings identical to torch (verified
    cosine 1.000000 by scripts/verify_onnx_equivalence.py). Falls back to
    sentence-transformers so existing installs keep working unchanged.

    Set FIE_EMBED_BACKEND=torch to force the reference implementation.
    """
    import os

    if (os.environ.get("FIE_EMBED_BACKEND") or "").strip().lower() != "torch":
        try:
            from fie.onnx_encoder import OnnxEncoder
            enc = OnnxEncoder()
            if enc.available:
                return enc, "onnx"
            logger.info(
                "encoder=onnx status=unavailable reason=%s falling_back=torch",
                enc.status().get("reason"),
            )
        except Exception as exc:
            logger.info(
                "encoder=onnx status=unavailable reason=%s: %s falling_back=torch",
                type(exc).__name__, exc,
            )

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name), "sentence-transformers"


class SentenceEncoder:
    """
    Thread-safe lazy-loading sentence encoder.

    Degradation contract
    --------------------
    This class never raises on the encode path. If the backend cannot load,
    `available` becomes False and `encode*` returns zero vectors, which
    downstream agents treat as "no signal" and use to lower their confidence.
    Callers that need to *know* the encoder is real must check `available`
    (or `status()`) rather than inspecting the returned vectors.
    """

    def __init__(self) -> None:
        self._model   = None
        self._lock    = threading.Lock()
        self._loaded  = False
        self._failed  = False
        self._reason  = ""

    # ── Public API ─────────

    @property
    def available(self) -> bool:
        """True if sentence-transformers loaded successfully."""
        self._get_model()   # trigger lazy load if not yet done
        return self._loaded and not self._failed

    def status(self) -> dict:
        """
        Machine-readable health for /health/deep.

        Does NOT trigger a load — reports what is known so far, so a health
        probe can never be the thing that pays the model-load cost.
        """
        if not self._loaded:
            return {"backend": "transformer", "state": "not_loaded"}
        if self._failed:
            return {"backend": "zero_vector_fallback", "state": "degraded",
                    "reason": self._reason}
        return {"backend": "transformer", "state": "ready"}

    def encode(self, text: str) -> np.ndarray:
        """
        Encodes a single string to a 384-dim L2-normalised float32 vector.
        """
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a list of strings.
        """
        cfg = _lazy_settings()

        if not texts:
            return np.zeros((0, cfg.embedding_dimension), dtype=np.float32)

        model = self._get_model()

        if model is None:
            # Fallback: zero vectors — agents will lower their confidence
            return np.zeros((len(texts), cfg.embedding_dimension), dtype=np.float32)

        try:
            # show_progress_bar=False keeps logs clean in production
            vecs = model.encode(
                texts,
                normalize_embeddings=True,   # L2 normalise  cosine = dot product
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vecs.astype(np.float32)
        except Exception as exc:
            # Per-call failure (bad input, transient OOM) — the model itself is
            # still considered healthy, so we do not flip `_failed` here.
            logger.warning(
                "encoder=encode status=failed n_texts=%d reason=%s: %s",
                len(texts), type(exc).__name__, exc,
            )
            return np.zeros((len(texts), cfg.embedding_dimension), dtype=np.float32)

    # ── Internal

    def _get_model(self):
        """
        Lazy-loads the SentenceTransformer model exactly once.

        Failure is recorded, not raised: `_failed` is set and None is returned
        so `encode_batch` can fall back to zero vectors. The double-checked
        lock guarantees the (expensive) load happens once even under
        concurrent first requests.
        """
        if self._loaded:
            return self._model

        with self._lock:
            if self._loaded:
                return self._model
            # Set before the attempt so a failed load is never retried on every
            # call — a retry storm here would stall every request thread.
            self._loaded = True
            try:
                cfg = _lazy_settings()
                model_name = cfg.embedding_transformer_model
                logger.info("encoder=load status=started model=%s", model_name)
                self._model, backend = _build_backend(model_name)
                probe = self._model.encode(["smoke test"], normalize_embeddings=True)
                logger.info(
                    "encoder=load status=ready backend=%s model=%s dim=%d",
                    backend, model_name, probe.shape[-1],
                )
            except ImportError as exc:
                # Expected in lite/minimal installs — this is a supported
                # configuration, so it is a warning, not an error.
                self._failed = True
                self._reason = "no embedding backend available"
                logger.warning(
                    "encoder=load status=unavailable reason=%s "
                    "action='pip install onnxruntime tokenizers' (or fie-sdk[torch]) "
                    "detail=%s", self._reason, exc,
                )
            except Exception as exc:
                # Unexpected: bad weights, no disk, corrupt cache, OOM.
                # Keep the traceback — this one is genuinely actionable.
                self._failed = True
                self._reason = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "encoder=load status=failed reason=%s", self._reason, exc_info=True
                )

        return self._model


@lru_cache(maxsize=1)
def get_encoder() -> SentenceEncoder:
    """
    Returns the singleton SentenceEncoder.
    """
    return SentenceEncoder()
