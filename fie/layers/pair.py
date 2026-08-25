"""
Layer 8 — PAIR semantic intent classifier, and the XGBoost meta-classifier.

PAIR (MiniLM embedding + calibrated SVM) is the layer the ablation study
identifies as carrying the common case: on standard benchmarks it alone
matches the full pipeline. v6.3b is the shipped default.

The meta-classifier lives here too because it consumes every layer's score
and is loaded through the same lazy-artifact machinery.

Extracted from fie/adversarial.py. Detection logic is unchanged — see
tests/test_detection_golden.py, which pins the exact confidence values.
"""
from __future__ import annotations

import logging
import json as _json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Model artifact location ───────────────────────────────────────────────────
# Anchored to the `fie` PACKAGE root, not to this file's directory.
#
# This module used to live in fie/adversarial.py, where `Path(__file__).parent`
# was `fie/` and resolved model paths correctly. Moving it to fie/layers/ made
# that expression point at `fie/layers/models/`, which does not exist — the
# meta-classifier silently failed to load and every blended confidence changed.
# Layer scores stayed identical, so only the golden-output test caught it.
#
# Deriving from the package root means these paths survive any future file move.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent   # .../fie
_PKG_MODELS   = _PACKAGE_ROOT / "models"                 # bundled in the wheel
_REPO_MODELS  = _PACKAGE_ROOT.parent / "models"          # source checkout only


def _build_embedder(embed_model: str):
    """
    Construct the sentence embedder, preferring ONNX Runtime over torch.

    ONNX is the default backend because it is strictly better here on every axis
    that matters:

        disk    torch(CUDA) 4,668 MB  ->  onnxruntime + model  ~113 MB   (~45x)
        speed   torch 16.64 ms        ->  onnx 6.80 ms                   (2.4x)
        output  cosine similarity 1.000000 against torch — bit-identical

    Verified by scripts/verify_onnx_equivalence.py, which gates on cosine
    > 0.999 across a mixed attack/benign/multilingual corpus. Because the
    embeddings are identical, the PAIR classifier's fitted decision boundary is
    unaffected and every published benchmark number still holds.

    Falls back to sentence-transformers when the ONNX artifacts are absent, so
    existing installs and the reference implementation both keep working.
    Set FIE_EMBED_BACKEND=torch to force the old path (used for A/B checks).
    """
    import os

    backend = (os.environ.get("FIE_EMBED_BACKEND") or "").strip().lower()

    if backend != "torch":
        try:
            from fie.onnx_encoder import OnnxEncoder
            enc = OnnxEncoder()
            if enc.available:
                logger.info("layer=pair_classifier embedder=onnx status=ready")
                return enc
            logger.info(
                "layer=pair_classifier embedder=onnx status=unavailable reason=%s "
                "falling_back=sentence-transformers", enc.status().get("reason"),
            )
        except Exception as exc:
            logger.info(
                "layer=pair_classifier embedder=onnx status=unavailable reason=%s: %s "
                "falling_back=sentence-transformers", type(exc).__name__, exc,
            )

    from sentence_transformers import SentenceTransformer
    logger.info("layer=pair_classifier embedder=sentence-transformers model=%s", embed_model)
    return SentenceTransformer(embed_model)


def _resolve_models_dir(sentinel: str) -> Path:
    """
    Pick the model directory containing `sentinel`.

    Prefers the models bundled inside the installed package; falls back to the
    repo-root models/ directory for source checkouts. Returns the package path
    when neither exists, so error messages name the location users should fix.
    """
    if (_PKG_MODELS / sentinel).exists():
        return _PKG_MODELS
    if (_REPO_MODELS / sentinel).exists():
        return _REPO_MODELS
    return _PKG_MODELS


#Layer 8: PAIR semantic intent classifier

_pair_clf      = None
_pair_embedder = None
_pair_threshold: float = 0.60
_pair_load_attempted: bool = False
_pair_load_error: str = ""
# PAIR loads two artifacts (an sklearn classifier and a sentence-transformer).
# The lock makes that pair of assignments atomic with respect to other threads:
# without it, a concurrent scan could observe the classifier already assigned
# while the embedder was still loading, pass the readiness check, and then
# dereference None. Layers run in parallel, so this is reachable on any first
# scan — it is not a theoretical race.
_pair_lock = threading.Lock()

# ── Meta-classifier (XGBoost on 12 layer scores) ─────────────────────────────
_meta_clf             = None
_meta_clf_threshold:  float      = 0.50
_meta_clf_features:   list[str]  = []
_meta_clf_attempted:  bool       = False
_meta_clf_lock        = threading.Lock()


def _load_meta_classifier() -> bool:
    global _meta_clf, _meta_clf_threshold, _meta_clf_features, _meta_clf_attempted
    with _meta_clf_lock:
        if _meta_clf_attempted:
            return _meta_clf is not None
        _meta_clf_attempted = True
        try:
            import json as _json2
            import joblib
            _models_dir = _resolve_models_dir("meta_clf.pkl")
            _clf_path   = _models_dir / "meta_clf.pkl"
            _meta_path  = _models_dir / "meta_clf.json"
            if not _clf_path.exists():
                return False
            _meta_clf = joblib.load(_clf_path)
            if _meta_path.exists():
                with open(_meta_path, encoding="utf-8") as f:
                    meta = _json2.load(f)
                _meta_clf_threshold = float(meta.get("threshold", 0.30))
                _meta_clf_features  = meta.get("layer_names", [])
            return True
        except Exception as exc:
            logger.warning(
                "degraded capability=_load_meta_classifier impact='this optional step was skipped' "
                "reason=%s: %s", type(exc).__name__, exc,
            )
            return False


def _run_meta_classifier(layer_scores: dict[str, float]) -> float:
    """Return meta-classifier attack probability (0.0 if unavailable)."""
    if not _load_meta_classifier():
        return 0.0
    try:
        import numpy as _np
        vec = _np.array(
            [[layer_scores.get(f, 0.0) for f in _meta_clf_features]],
            dtype=_np.float32,
        )
        return float(_meta_clf.predict_proba(vec)[0][1])
    except Exception as exc:
        logger.warning(
            "degraded capability=_run_meta_classifier impact='this optional step was skipped' "
            "reason=%s: %s", type(exc).__name__, exc,
        )
        return 0.0


def _load_pair_classifier() -> bool:
    """
    Load the PAIR semantic classifier and its sentence embedder, exactly once.

    Publication is all-or-nothing: both artifacts are built into locals and only
    assigned to module globals once BOTH succeeded. A half-loaded PAIR layer
    used to report ready (the readiness check only looked at the classifier) and
    then raise AttributeError on the embedder for the rest of the process
    lifetime — silently disabling the layer the ablation identifies as carrying
    the common case, while every scan still returned a confident-looking verdict.

    Returns True when the layer is fully usable.
    """
    global _pair_clf, _pair_embedder, _pair_threshold, _pair_load_attempted
    global _pair_load_error

    # Fast path: no lock once loading has settled.
    if _pair_load_attempted:
        return _pair_clf is not None and _pair_embedder is not None

    with _pair_lock:
        if _pair_load_attempted:
            return _pair_clf is not None and _pair_embedder is not None
        return _load_pair_classifier_locked()


def _load_pair_classifier_locked() -> bool:
    """Body of the PAIR load. Caller must hold _pair_lock."""
    global _pair_clf, _pair_embedder, _pair_threshold, _pair_load_attempted
    global _pair_load_error

    _pair_load_attempted = True
    try:
        import json as _json
        import joblib
        # sentence-transformers is NOT imported here. It is now an optional
        # fallback behind ONNX, so importing it eagerly would make torch a hard
        # requirement again — defeating the entire migration. _build_embedder()
        # imports it lazily, only if the ONNX backend is unavailable.

        # Look inside the installed package first (fie/models/), then fall back
        # to the repo root models/ directory for local development.
        _models_dir = _resolve_models_dir("pair_intent_classifier.pkl")

        # Prefer v6 > v5 > v4 > v3 > v2 > v1.
        # v6 = domain-balanced corpus (medical/legal/coding/factual benign +
        #      genuinely-harmful attacks). Fixes the out-of-distribution benign
        #      FPR (medical 71%, legal 67% on v5) caused by Alpaca-only benign data.
        # v5 = sklearn 1.7.2 + NLLB multilingual augmentation. Fixed the
        #      1.6.1->1.7.2 boundary-shift regression but kept Alpaca-only benign.
        # v4 = 3× hard-positive weighting at threshold 0.50.
        #
        # Override with FIE_PAIR_VERSION=v5 (etc.) to force a specific model —
        # used for A/B comparison in the v6 evaluation.
        import os as _os
        _force = (_os.environ.get("FIE_PAIR_VERSION") or "").strip().lower()
        _versions = [
            # v6.3b = SHIPPED DEFAULT (E26). v6.2 corpus + targeted soft-harm/euphemism
            # positives (E24) + safe-but-scary benign negatives (E25), threshold 0.50.
            # Full-pipeline ship-gate PASSED: Pareto win over v6.2 — soft-harm recall
            # +32 pts, over-refusal flat, no clean-recall regression. v6/v6_3 retained
            # below for A/B via FIE_PAIR_VERSION (e.g. FIE_PAIR_VERSION=v6 for v6.2).
            ("v6_3b", _models_dir / "pair_intent_classifier_v6_3b.pkl",
                      _models_dir / "pair_intent_meta_v6_3b.json"),
            ("v6", _models_dir / "pair_intent_classifier_v6.pkl",
                   _models_dir / "pair_intent_meta_v6.json"),
            ("v6_3", _models_dir / "pair_intent_classifier_v6_3.pkl",
                     _models_dir / "pair_intent_meta_v6_3.json"),
            ("v5", _models_dir / "pair_intent_classifier_v5.pkl",
                   _models_dir / "pair_intent_meta_v5.json"),
            ("v4", _models_dir / "pair_intent_classifier_v4.pkl",
                   _models_dir / "pair_intent_meta_v4.json"),
            ("v3", _models_dir / "pair_intent_classifier_v3.pkl",
                   _models_dir / "pair_intent_meta_v3.json"),
            ("v2", _models_dir / "pair_intent_classifier_v2.pkl",
                   _models_dir / "pair_intent_meta_v2.json"),
            ("v1", _models_dir / "pair_intent_classifier.pkl",
                   _models_dir / "pair_intent_meta.json"),
        ]

        clf_path = meta_path = None
        if _force:
            for name, clf, meta in _versions:
                if name == _force and clf.exists():
                    clf_path, meta_path = clf, meta
                    break
        if clf_path is None:
            for name, clf, meta in _versions:
                if clf.exists():
                    clf_path, meta_path = clf, meta
                    break

        if clf_path is None or not clf_path.exists():
            _pair_load_error = (
                f"no PAIR classifier found in {_models_dir} — "
                "run `python scripts/download_models.py`"
            )
            logger.warning("layer=pair_classifier status=unavailable reason=%s",
                           _pair_load_error)
            return False

        # Build into locals. Nothing is published to module state until both
        # artifacts exist, so a failure here leaves the layer cleanly disabled
        # rather than half-initialised.
        local_clf = joblib.load(clf_path)

        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = _json.load(f)
            local_threshold = float(meta.get("threshold", 0.60))
            embed_model = meta.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        else:
            local_threshold = _pair_threshold
            embed_model = "sentence-transformers/all-MiniLM-L6-v2"

        local_embedder = _build_embedder(embed_model)

        # ── Publish (both artifacts are ready) ────────────────────────────────
        _pair_clf       = local_clf
        _pair_embedder  = local_embedder
        _pair_threshold = local_threshold
        _pair_load_error = ""
        # Report the backend actually in use, not the model NAME from meta.json.
        # Logging `embed_model` here printed "embedder=sentence-transformers/..."
        # even when ONNX was serving the embeddings, which reads as though torch
        # were loaded — exactly the thing an operator checks this line to rule out.
        backend = type(local_embedder).__name__
        logger.info(
            "layer=pair_classifier status=ready model=%s threshold=%.2f "
            "backend=%s embed_model=%s",
            clf_path.name, local_threshold, backend, embed_model,
        )
        return True
    except ImportError as exc:
        # Expected in lite installs (`pip install fie-sdk` without [ml]).
        _pair_load_error = f"missing dependency: {exc}"
        logger.warning(
            "layer=pair_classifier status=unavailable reason=%s "
            "action='pip install fie-sdk[ml]'", _pair_load_error,
        )
        return False
    except Exception as exc:
        # Corrupt pickle, sklearn version skew, no HF cache and no network.
        # Keep the traceback: this one is actionable and, per the ablation,
        # this layer carries most of the detection.
        _pair_load_error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "layer=pair_classifier status=failed reason=%s — detection recall "
            "will be materially reduced", _pair_load_error, exc_info=True,
        )
        return False


def _run_pair_classifier(prompt: str) -> tuple[str | None, float, dict]:
    if not _load_pair_classifier():
        return None, 0.0, {}
    try:
        _PREFIX = "Represent this text for security threat classification: "
        # show_progress_bar=False: sentence-transformers writes a tqdm bar to
        # stderr by default, which corrupts structured log output in containers.
        vec  = _pair_embedder.encode(
            [_PREFIX + prompt],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        prob = float(_pair_clf.predict_proba(vec)[0][1])
        if prob >= _pair_threshold:
            return "JAILBREAK_ATTEMPT", round(prob, 4), {
                "pair_probability": round(prob, 4),
                "threshold":        _pair_threshold,
            }
        return None, 0.0, {}
    except Exception as exc:
        # Raised to the layer wrapper so the scan is marked degraded rather than
        # reporting a clean verdict from a layer that never actually scored.
        raise RuntimeError(f"PAIR inference failed: {type(exc).__name__}: {exc}") from exc




def _pair_state() -> dict:
    """
    Read-only snapshot of this layer's loaded artifacts, for health reporting.

    Exists so callers do not import this module's private globals directly.
    Health checks used to reach in for `_pair_clf` / `_pair_embedder`, which
    silently broke the moment the layer moved into its own module — and, worse,
    read the two flags non-atomically, so a health probe racing a first scan
    could report "classifier loaded" for a layer that was not yet usable.

    Deliberately does NOT trigger a load: a health probe must never be the thing
    that pays the model-load cost.
    """
    with _pair_lock:
        return {
            "loaded":    _pair_clf is not None and _pair_embedder is not None,
            "attempted": _pair_load_attempted,
            "threshold": _pair_threshold,
            "error":     _pair_load_error or None,
        }


def _meta_state() -> dict:
    """Read-only snapshot of the meta-classifier, for health reporting."""
    with _meta_clf_lock:
        return {
            "loaded":    _meta_clf is not None,
            "attempted": _meta_clf_attempted,
            "threshold": _meta_clf_threshold,
            "n_features": len(_meta_clf_features),
        }


def _meta_threshold() -> float:
    """
    Current meta-classifier firing threshold.

    Read through a function, never captured into a module-level constant by
    callers: the real value comes from meta_clf.json at load time, so anything
    that snapshots it at import gets the 0.50 placeholder permanently.
    """
    with _meta_clf_lock:
        return _meta_clf_threshold
