"""pytest configuration for the evaluation-harness tests. Helpers live in _helpers.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
for p in (str(REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from _helpers import models_present  # noqa: E402


@pytest.fixture
def need_models():
    """Skip, with an accurate reason, when the trained artifacts are absent."""
    if not models_present():
        pytest.skip("model artifacts not present — run: python scripts/download_models.py --strict")
