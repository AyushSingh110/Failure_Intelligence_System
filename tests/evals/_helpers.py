"""
Helpers for the evaluation-harness tests.

Two rules these tests follow:

  * The network guard is NEVER installed in the pytest process. It is built on a
    Python audit hook, which cannot be removed once added, so it would break
    every test that runs afterwards. Guard tests run a child interpreter and
    read its JSON output.
  * Child processes get an explicit, minimal environment. No test depends on, or
    can leak, a credential from the developer's shell.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_PASSTHROUGH = ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT",
                "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "LD_LIBRARY_PATH",
                "LANG", "LC_ALL", "SYSTEMDRIVE")


def child_env(tmp_path: Path, extra: dict | None = None) -> dict:
    """A minimal environment for a child interpreter. Nothing secret is inherited."""
    env = {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update({
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONNOUSERSITE": "1",
        "TEMP": str(tmp_path), "TMP": str(tmp_path), "TMPDIR": str(tmp_path),
        "HOME": str(home), "USERPROFILE": str(home),
        "FIE_NO_TELEMETRY": "1",
        "FIE_NO_AUTO_DOWNLOAD": "1",
        "FIE_FEEDBACK_PATH": str(tmp_path / "flagged_events.jsonl"),
    })
    if extra:
        for key, value in extra.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return env


def run_child(code: str, tmp_path: Path, extra_env: dict | None = None,
              timeout: int = 180) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter at the repository root."""
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-c", code],
        cwd=str(REPO_ROOT), env=child_env(tmp_path, extra_env),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )


def child_json(code: str, tmp_path: Path, extra_env: dict | None = None,
               timeout: int = 180) -> dict:
    """Run `code`, which must print one JSON object as its last stdout line."""
    proc = run_child(code, tmp_path, extra_env, timeout)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"child printed nothing. exit={proc.returncode}\nstderr:\n{proc.stderr[-2000:]}"
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        raise AssertionError(
            f"child's last line is not JSON: {lines[-1][:300]!r}\nstderr:\n{proc.stderr[-2000:]}"
        )


MODEL_FILES = (
    "fie/models/pair_intent_classifier_v6_3b.pkl",
    "fie/models/pair_intent_meta_v6_3b.json",
    "fie/models/meta_clf.pkl",
    "fie/models/meta_clf.json",
    "fie/models/minilm-onnx/model.onnx",
    "fie/models/minilm-onnx/tokenizer.json",
)


def models_present() -> bool:
    return all((REPO_ROOT / rel).exists() for rel in MODEL_FILES)
