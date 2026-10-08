"""
The configuration fingerprint: what decides whether two runs may be compared.

Three blocks:

  IDENTITY       what was measured   code revision, fie source-tree hash, model
                                     hashes, dataset content hashes, suite
                                     definitions, harness version
  CONFIGURATION  how                 profile, thresholds, detector constants,
                                     tiebreaker and translation state, seeds,
                                     relevant environment values
  ENVIRONMENT    where               Python, platform, package versions

and four keys derived from them:

  dataset_key   differs -> the runs are INCOMPARABLE
  config_key    differs -> comparable only if the configuration is the change under test
  subject_key   differs -> expected: this is the change being measured
  env_key       differs -> verdicts comparable with a warning; latency is not

Byte-identical deterministic artifacts are promised only when all four match.

Secrets appear by PRESENCE only, never by value. Timestamps, durations, run ids,
host names and absolute paths never enter the fingerprint; they live in run.json.

Standard library only.
"""
from __future__ import annotations

import hashlib
import os
import platform
import re
import sys
from pathlib import Path

from evals import HARNESS_VERSION, SCHEMA_VERSION, canonical

# Values recorded as-is. None of these is a secret or a path.
ENV_VALUE_NAMES = (
    "FIE_PAIR_VERSION", "FIE_NO_TELEMETRY", "FIE_NO_AUTO_DOWNLOAD", "FIE_DISABLE_META",
    "FIE_UNCERTAIN_ALLOW", "FIE_EMBED_BACKEND", "FIE_LAYER_POOL_SIZE", "FIE_LAYER_DEADLINE_S",
    "FIE_ONNX_THREADS", "FIE_SCAN_FAILURE_MODE", "FIE_LITE", "FIE_TELEMETRY",
    "SCAN_THRESHOLD", "FRAMING_DAMPEN_FACTOR", "PREFLIGHT_BLOCK_ENABLED",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONHASHSEED",
)
# Recorded as present / absent only.
ENV_PRESENCE_NAMES = (
    "GROQ_API_KEY", "FIE_API_KEY", "FIE_URL", "REDIS_URL", "LIBRETRANSLATE_URL",
    "MONGODB_URI", "JWT_SECRET_KEY", "SERPER_API_KEY", "SENDGRID_API_KEY",
    "HUGGING_FACE_TOKEN", "HF_TOKEN", "PYPI_TOKEN",
)
THREAD_ENV_NAMES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
PACKAGES = ("numpy", "scikit-learn", "onnxruntime", "tokenizers", "xgboost", "joblib",
            "pandas", "langdetect", "deep-translator", "requests", "urllib3")
# Packages whose version can change a verdict or a confidence. Part of env_key.
NUMERIC_PACKAGES = ("numpy", "scikit-learn", "onnxruntime", "tokenizers", "xgboost")

# Paths whose uncommitted modification would make the measured subject untraceable.
SUBJECT_PATHS = ("fie", "engine", "app", "storage", "scripts/model_manifest.json",
                 "data/overrefusal", "data/benchmark_audit", "pyproject.toml")


# ── source-tree hashes ───────────────────────────────────────────────────────

def tree_sha256(repo_root: str | Path, rel_dir: str, suffixes: tuple[str, ...] = (".py",)) -> tuple[str, int]:
    """
    Hash every matching file under `rel_dir`: sorted POSIX path plus the SHA-256
    of its LF-normalized content. Independent of line endings and of git.
    Returns (hash, file count).
    """
    root = Path(repo_root) / rel_dir
    h = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file() or path.suffix not in suffixes or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        digest = canonical.sha256_bytes(canonical.lf(path.read_bytes()))
        h.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
        count += 1
    return h.hexdigest(), count


def fie_version(repo_root: str | Path) -> str | None:
    try:
        text = (Path(repo_root) / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def subject_identity(repo_root: str | Path) -> dict:
    digest, count = tree_sha256(repo_root, "fie")
    return {"fie_tree_sha256": digest, "fie_py_files": count, "fie_version": fie_version(repo_root)}


def harness_identity(repo_root: str | Path) -> dict:
    digest, count = tree_sha256(repo_root, "evals")
    return {"version": HARNESS_VERSION, "schema_version": SCHEMA_VERSION,
            "tree_sha256": digest, "py_files": count}


# ── worker-side blocks ───────────────────────────────────────────────────────

def _package_versions() -> dict:
    from importlib import metadata
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment_block() -> dict:
    """Where the measurement ran. Computed inside the worker."""
    uname = platform.uname()
    return {
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "platform": {"system": uname.system, "release": uname.release, "machine": uname.machine},
        "cpu": {"model": uname.processor or os.environ.get("PROCESSOR_IDENTIFIER", ""),
                "count": os.cpu_count()},
        "packages": _package_versions(),
        "threads": {n: os.environ.get(n) for n in THREAD_ENV_NAMES},
        "machine_tag": canonical.sha256_bytes(uname.node.encode("utf-8"))[:8],
    }


def profile_configuration(profile_id: str, profile: dict, subject_config: dict,
                          translation: str, langdetect_seed) -> dict:
    """How the measurement ran: the profile plus what the subject reports about itself."""
    return {
        "profile": {"id": profile_id, "sha256": canonical.hash_obj(profile),
                    "title": profile["title"]},
        "pair_version_requested": profile.get("pair_version"),
        "blocked_imports": sorted(profile.get("blocked_imports", [])),
        "translation": {"unavailable": "unavailable (harness stub returns None)",
                        "fixed_benign": "fixed benign sentence (harness stub)"}[translation],
        "langdetect_seed": langdetect_seed,
        "env": {n: os.environ.get(n) for n in ENV_VALUE_NAMES},
        "env_present": {n: bool(os.environ.get(n)) for n in ENV_PRESENCE_NAMES},
        "statistics": {"bootstrap_seed": 42, "bootstrap_resamples": 10000, "confidence_level": 0.95},
        "subject": subject_config,
    }


# ── assembly ─────────────────────────────────────────────────────────────────

def env_key_inputs(environment: dict) -> dict:
    major_minor = ".".join(environment["python"]["version"].split(".")[:2])
    return {
        "python": major_minor,
        "implementation": environment["python"]["implementation"],
        "system": environment["platform"]["system"],
        "machine": environment["platform"]["machine"],
        "packages": {n: environment["packages"].get(n) for n in NUMERIC_PACKAGES},
    }


def build(identity: dict, configuration: dict, environment: dict) -> dict:
    """
    Assemble fingerprint.json.

    identity       {"subject", "git", "models", "datasets", "suites", "fixtures", "harness"}
    configuration  {profile_id: profile_configuration(...)}
    environment    environment_block()
    """
    keys = {
        "dataset_key": canonical.hash_obj({
            "schema_version": SCHEMA_VERSION,
            "datasets": {k: {"content_sha256": v["content_sha256"], "rows": v["rows"]}
                         for k, v in identity["datasets"].items()},
            "suites": identity["suites"],
            "fixtures": identity.get("fixtures", {}),
        }),
        "config_key": canonical.hash_obj(configuration),
        "subject_key": canonical.hash_obj({
            "fie_tree_sha256": identity["subject"]["fie_tree_sha256"],
            "models": {profile: {role: m["sha256"] for role, m in roles.items()}
                       for profile, roles in identity["models"].items()},
        }),
        "env_key": canonical.hash_obj(env_key_inputs(environment)),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "keys": keys,
        "identity": identity,
        "configuration": configuration,
        "environment": environment,
    }


def comparability(a_keys: dict, b_keys: dict) -> dict:
    """How two runs relate, from their keys alone."""
    same = {k: a_keys.get(k) == b_keys.get(k)
            for k in ("dataset_key", "config_key", "subject_key", "env_key")}
    if not same["dataset_key"]:
        status = "incomparable"
        reason = "datasets, suite definitions or schema differ"
    elif not same["config_key"]:
        status = "comparable only as a declared configuration change"
        reason = "configuration differs"
    elif not same["env_key"]:
        status = "comparable with a warning"
        reason = "environment differs: verdicts comparable, byte identity not expected, latency not comparable"
    elif same["subject_key"]:
        status = "identical inputs"
        reason = "all four keys match: deterministic artifacts are expected to be byte-identical"
    else:
        status = "comparable"
        reason = "same datasets, configuration and environment; the subject differs"
    return {"same": same, "status": status, "reason": reason}
