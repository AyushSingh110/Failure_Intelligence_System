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
import re
import threading
from pathlib import Path

from fie.layers import _verified_models as _vm

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
#
# Every artifact is selected through the package's own manifest and verified before it is
# deserialized; see fie/layers/_verified_models.py. There is no second directory: the
# repository-root models/ directory is never consulted (WP-021, W21-18).
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent   # .../fie
_PKG_MODELS   = _PACKAGE_ROOT / "models"                 # bundled in the wheel


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


# ── Artifact identity ─────────────────────────────────────────────────────────
# A scan result names the artifacts that produced it: a version taken from the
# artifact's own metadata file and the first 16 hex characters of the SHA-256
# of the bytes that were deserialized. Both are already public (scripts/model_manifest.json
# lists the full hashes). Neither a path nor a file name is ever recorded here.
_DIGEST_HEX: int = 16
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")


def _declared_version(meta: dict) -> str | None:
    """The version an artifact's metadata declares, if it is a plain short token."""
    version = meta.get("version") if isinstance(meta, dict) else None
    if isinstance(version, str) and _VERSION_RE.match(version):
        return version
    return None


def _resolve_models_dir(sentinel: str | None = None) -> Path:
    """
    The directory model artifacts are read from: the package's own models/.

    Always the same directory. `sentinel` is accepted so existing callers keep
    working and is ignored: which files happen to exist where no longer decides
    anything. Which file is loaded is decided by the package manifest alone.
    """
    return _PKG_MODELS


#Layer 8: PAIR semantic intent classifier

_pair_clf      = None
_pair_embedder = None
_pair_threshold: float = 0.60
_pair_load_attempted: bool = False
_pair_load_error: str = ""
# Why the classifier is not usable, as a fixed code: "dependency" (a package it
# needs cannot be imported), "model" (nothing to load: no package manifest, or a
# listed file is absent) or "load_failed" (refused or could not be loaded: not
# listed, altered, unreadable, not deserializable). Empty while usable or
# while a load is still in progress. `_pair_load_error` keeps the free text for
# logs and health(); it never names an absolute location, and never reaches a result.
_pair_unavailable_reason: str = ""
# The finer-grained internal code behind that reason: a _verified_models detail
# ("manifest_missing", "missing", "not_listed", "mismatch", ...), "dependency" or
# "unexpected". For logs, warmup() and diagnostics; never part of a scan result.
_pair_unavailable_detail: str = ""
# {"version", "digest"} of the classifier file that was loaded. Empty otherwise.
_pair_identity: dict = {}
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
_meta_clf_identity:   dict       = {}     # {"version", "digest"} of the loaded file
# Why the meta-classifier is not loaded after an attempt, as a fixed code: a
# _verified_models detail ("manifest_missing", "mismatch", ...), "dependency" or
# "unexpected". For logs and diagnostics only; a scan result just says "unavailable".
_meta_clf_unavailable_detail: str = ""
_meta_clf_lock        = threading.Lock()
# Whether the calling thread's last _run_meta_classifier() call produced a real
# probability. The function returns 0.0 both for "no attack pattern" and for
# "could not run"; the scanner needs to tell them apart to report coverage.
_meta_call = threading.local()


def _load_meta_classifier() -> bool:
    """
    Load the meta-classifier and its side-car, exactly once, through the package manifest.

    Both artifacts are verified before either is used: the side-car's threshold and
    feature list are read only from verified bytes, and only verified bytes are
    deserialized. Nothing is published unless everything succeeded; a missing or
    altered side-car leaves the meta-classifier unavailable rather than running it
    with a default threshold or an empty feature list.
    """
    global _meta_clf, _meta_clf_threshold, _meta_clf_features, _meta_clf_attempted
    global _meta_clf_identity, _meta_clf_unavailable_detail
    with _meta_clf_lock:
        if _meta_clf_attempted:
            return _meta_clf is not None
        _meta_clf_attempted = True
        _meta_clf_unavailable_detail = ""
        try:
            import joblib  # noqa: F401 - without it nothing can be deserialized: report that first
            clf_entry, meta_entry = _vm.select_meta(_vm.package_manifest())
            models_dir = _resolve_models_dir("meta_clf.pkl")
            data = _vm.read_artifact(models_dir, clf_entry)
            meta = _vm.parse_json(meta_entry, _vm.read_artifact(models_dir, meta_entry))
            threshold = _vm.required_threshold(meta_entry, meta)
            features = _vm.required_names(meta_entry, meta, "layer_names")
            version = _declared_version(meta)
            model = _vm.unpickle(clf_entry, data)
        except _vm.LoadRefused as exc:
            _meta_clf_unavailable_detail = exc.detail
            logger.warning(
                "degraded capability=_load_meta_classifier impact='this optional step was skipped' "
                "reason=%s detail=%s artifact=%s: %s", exc.reason, exc.detail, exc.artifact, exc,
                exc_info=exc.detail == "deserialize_failed",
            )
            return False
        except Exception as exc:
            _meta_clf_unavailable_detail = "dependency" if isinstance(exc, ImportError) else "unexpected"
            logger.warning(
                "degraded capability=_load_meta_classifier impact='this optional step was skipped' "
                "reason=%s: %s", type(exc).__name__, exc,
            )
            return False
        _meta_clf_threshold = threshold
        _meta_clf_features  = features
        _meta_clf_identity  = {"version": version, "digest": _vm.digest(data)[:_DIGEST_HEX]}
        _meta_clf           = model
        return True


def _meta_unavailable_detail() -> str | None:
    """Why the meta-classifier is not loaded, as a fixed code; None if loaded or never attempted."""
    if _meta_clf is not None or not _meta_clf_attempted:
        return None
    return _meta_clf_unavailable_detail or None


def _run_meta_classifier(layer_scores: dict[str, float]) -> float:
    """Return meta-classifier attack probability (0.0 if unavailable)."""
    _meta_call.ran = False
    if not _load_meta_classifier():
        return 0.0
    try:
        import numpy as _np
        vec = _np.array(
            [[layer_scores.get(f, 0.0) for f in _meta_clf_features]],
            dtype=_np.float32,
        )
        prob = float(_meta_clf.predict_proba(vec)[0][1])
        _meta_call.ran = True
        return prob
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
    global _pair_load_error, _pair_unavailable_reason, _pair_identity, _pair_unavailable_detail

    _pair_load_attempted = True
    _pair_unavailable_reason = ""
    _pair_unavailable_detail = ""
    _pair_identity = {}
    try:
        import joblib  # noqa: F401 - without it nothing can be deserialized: report that first
        # sentence-transformers is NOT imported here. It is now an optional
        # fallback behind ONNX, so importing it eagerly would make torch a hard
        # requirement again — defeating the entire migration. _build_embedder()
        # imports it lazily, only if the ONNX backend is unavailable.

        # Which classifier: the one the package manifest marks as the default
        # (v6.3b, shipped after the E26 full-pipeline ship gate), or exactly the
        # version FIE_PAIR_VERSION names — a selection key such as v6 (PAIR v6.2)
        # or v5, used for A/B comparison. A version the manifest does not list,
        # or a listed one whose file is missing or altered, leaves the layer
        # unavailable. Another classifier is never loaded in its place.
        import os as _os
        requested = (_os.environ.get("FIE_PAIR_VERSION") or "").strip().lower()
        clf_entry, meta_entry = _vm.select_pair(_vm.package_manifest(), requested or None)
        models_dir = _resolve_models_dir("pair_intent_classifier.pkl")

        # Build into locals. Nothing is published to module state until both
        # artifacts exist, so a failure here leaves the layer cleanly disabled
        # rather than half-initialised. Both files are verified before either
        # is used: the side-car decides the threshold and which encoder the
        # fallback fetches, so none of its fields is read until it has passed.
        data = _vm.read_artifact(models_dir, clf_entry)
        meta = _vm.parse_json(meta_entry, _vm.read_artifact(models_dir, meta_entry))
        local_threshold = _vm.required_threshold(meta_entry, meta)
        embed_model = _vm.required_text(meta_entry, meta, "embed_model")
        local_version = _declared_version(meta)
        # The identity on a scan result describes the bytes that were deserialized.
        local_digest = _vm.digest(data)[:_DIGEST_HEX]
        local_clf = _vm.unpickle(clf_entry, data)

        local_embedder = _build_embedder(embed_model)

        # ── Publish (both artifacts are ready) ────────────────────────────────
        _pair_clf       = local_clf
        _pair_embedder  = local_embedder
        _pair_threshold = local_threshold
        _pair_load_error = ""
        _pair_identity = {"version": local_version, "digest": local_digest}
        # Report the backend actually in use, not the model NAME from meta.json.
        # Logging `embed_model` here printed "embedder=sentence-transformers/..."
        # even when ONNX was serving the embeddings, which reads as though torch
        # were loaded — exactly the thing an operator checks this line to rule out.
        backend = type(local_embedder).__name__
        logger.info(
            "layer=pair_classifier status=ready model=%s threshold=%.2f "
            "backend=%s embed_model=%s",
            _vm.file_name(clf_entry), local_threshold, backend, embed_model,
        )
        return True
    except _vm.LoadRefused as exc:
        # Nothing to load (no manifest, a listed file absent), or something that
        # must not be loaded (unlisted, altered, unreadable, undeserializable).
        # The public state is the reason; the detail code stays internal.
        _pair_load_error = f"{exc.detail}: {exc}"
        _pair_unavailable_reason = exc.reason
        _pair_unavailable_detail = exc.detail
        if exc.reason == "model":
            logger.warning(
                "layer=pair_classifier status=unavailable reason=model detail=%s artifact=%s: %s "
                "action='python scripts/download_models.py'", exc.detail, exc.artifact, exc,
            )
        else:
            logger.error(
                "layer=pair_classifier status=failed reason=load_failed detail=%s artifact=%s: %s "
                "— detection recall will be materially reduced", exc.detail, exc.artifact, exc,
                exc_info=exc.detail == "deserialize_failed",
            )
        return False
    except ImportError as exc:
        # Expected in lite installs (`pip install fie-sdk` without [ml]).
        _pair_load_error = f"missing dependency: {exc}"
        _pair_unavailable_reason = "dependency"
        _pair_unavailable_detail = "dependency"
        logger.warning(
            "layer=pair_classifier status=unavailable reason=%s "
            "action='pip install fie-sdk[ml]'", _pair_load_error,
        )
        return False
    except Exception as exc:
        # Not an integrity outcome: the encoder could not be built (no HF cache
        # and no network), or a defect. Keep the traceback: this one is
        # actionable and, per the ablation, this layer carries most of the detection.
        _pair_load_error = f"{type(exc).__name__}: {exc}"
        _pair_unavailable_reason = "load_failed"
        _pair_unavailable_detail = "unexpected"
        logger.error(
            "layer=pair_classifier status=failed reason=%s — detection recall "
            "will be materially reduced", _pair_load_error, exc_info=True,
        )
        return False


def _pair_unavailable_detail_code() -> str | None:
    """
    The internal code behind an unavailable classifier, or None if it is usable or no load
    has been attempted. One of the _verified_models detail codes, "dependency" or
    "unexpected". Fixed values only: safe to log, never placed in a scan result.
    """
    if _pair_clf is not None and _pair_embedder is not None:
        return None
    if not _pair_load_attempted:
        return None
    return _pair_unavailable_detail or None


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


def _pair_unavailable_state() -> str | None:
    """
    Why the classifier cannot run, as a fixed code, or None when it is usable.

        None                        usable, or no load has been attempted yet
        "unavailable_dependency"    a package it needs cannot be imported
        "unavailable_model"         no classifier file is present
        "unavailable_load_failed"   the file exists and could not be loaded
        "loading"                   another thread is still inside the first load

    Non-blocking by design: it takes no lock, so a scan thread never waits on a
    load that another thread is running. It never triggers a load. The codes are
    the public ones used in scan results; the free-text reason stays in
    `_pair_state()["error"]`.
    """
    if _pair_clf is not None and _pair_embedder is not None:
        return None
    if not _pair_load_attempted:
        return None
    reason = _pair_unavailable_reason
    return f"unavailable_{reason}" if reason else "loading"


def _meta_call_ran() -> bool:
    """True if this thread's last _run_meta_classifier() call returned a real probability."""
    return bool(getattr(_meta_call, "ran", False))


def _meta_call_reset() -> None:
    """Forget this thread's last meta-classifier call (the caller is about to make one)."""
    _meta_call.ran = False


def _model_identities() -> dict[str, dict]:
    """
    What is loaded in this process, per artifact role, as plain values.

    Read-only and non-blocking; never triggers a load. `loaded` is derived from
    the live objects, not from the recorded identity, so a role that is not
    usable can never report a version or a digest. No path and no file name.
    """
    pair_loaded = _pair_clf is not None and _pair_embedder is not None
    meta_loaded = _meta_clf is not None
    embedder = _pair_embedder if pair_loaded else None

    encoder = {"loaded": pair_loaded, "version": None, "digest": None, "threshold": None, "backend": None}
    describe = getattr(embedder, "identity", None)
    if callable(describe):
        try:
            described = describe()
            encoder["backend"] = described.get("backend")
            encoder["digest"] = described.get("digest")
        except Exception as exc:
            logger.warning(
                "degraded capability=encoder_identity impact='scan results omit the encoder digest' "
                "reason=%s", type(exc).__name__,
            )
    elif embedder is not None and type(embedder).__name__ == "SentenceTransformer":
        # The torch fallback loads from the Hugging Face cache, a directory of
        # files rather than one artifact, so there is no single digest to report.
        encoder["backend"] = "sentence-transformers"

    return {
        "pair_classifier": {
            "loaded":    pair_loaded,
            "version":   _pair_identity.get("version") if pair_loaded else None,
            "digest":    _pair_identity.get("digest") if pair_loaded else None,
            "threshold": _pair_threshold if pair_loaded else None,
            "backend":   None,
        },
        "meta_classifier": {
            "loaded":    meta_loaded,
            "version":   _meta_clf_identity.get("version") if meta_loaded else None,
            "digest":    _meta_clf_identity.get("digest") if meta_loaded else None,
            "threshold": _meta_clf_threshold if meta_loaded else None,
            "backend":   None,
        },
        "encoder": encoder,
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
