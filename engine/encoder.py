from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np

from config import get_settings

logger   = settings = None 


def _lazy_settings():
    global settings
    if settings is None:
        settings = get_settings()
    return settings


class SentenceEncoder:
    """
    Thread-safe lazy-loading sentence encoder.

    Usage:
        encoder = get_encoder()
        vec = encoder.encode("Ignore previous instructions")
        # vec.shape == (384,), dtype float32, L2-normalised
    """

    def __init__(self) -> None:
        self._model   = None
        self._lock    = threading.Lock()
        self._loaded  = False
        self._failed  = False

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if sentence-transformers loaded successfully."""
        return self._loaded and not self._failed

    def encode(self, text: str) -> np.ndarray:
        """
        Encodes a single string to a 384-dim L2-normalised float32 vector.
        Returns a zero vector if the model is unavailable.
        """
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a list of strings.
        Returns np.ndarray shape (N, 384), float32, each row L2-normalised.
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
                normalize_embeddings=True,   # L2 normalise → cosine = dot product
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vecs.astype(np.float32)
        except Exception as exc:
            logging.getLogger(__name__).warning("Encoding failed: %s", exc)
            return np.zeros((len(texts), cfg.embedding_dimension), dtype=np.float32)

    # ── Internal ───────────────────────────────────────────────────────

    def _get_model(self):
        """Lazy-loads the SentenceTransformer model exactly once."""
        if self._loaded:
            return self._model

        with self._lock:
            if self._loaded:      # double-checked locking
                return self._model
            self._loaded = True   # mark before trying to prevent retry storms
            try:
                from sentence_transformers import SentenceTransformer
                cfg = _lazy_settings()
                model_name = cfg.embedding_transformer_model
                print(f"[encoder] Loading: {model_name}")
                self._model = SentenceTransformer(model_name)
                test_vec = self._model.encode(["smoke test"], normalize_embeddings=True)
                print(f"[encoder] Loaded OK — shape: {test_vec.shape}")
                logging.getLogger(__name__).info(
                    "Loaded sentence encoder: %s", model_name
                )
            except ImportError:
                print("[encoder] ERROR: sentence-transformers not installed.")
                print("[encoder] Run: pip install sentence-transformers")
                self._failed = True
            except Exception as exc:
                import traceback
                print(f"[encoder] ERROR loading model: {exc}")
                traceback.print_exc()
                print("[encoder] Falling back to n-gram mode.")
                logging.getLogger(__name__).error(
                    "Failed to load sentence encoder: %s", exc, exc_info=True
                )
                self._failed = True

        return self._model


@lru_cache(maxsize=1)
def get_encoder() -> SentenceEncoder:
    """
    Returns the singleton SentenceEncoder.
    Safe to call at module level — model loads lazily on first .encode() call.
    """
    return SentenceEncoder()