from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class FIEConfig:
    """Holds all configuration needed to connect to the FIE server."""
    fie_url:  str    # URL of your running FIE FastAPI server
    api_key:  str    # API key which user get after logging


def get_config(
    fie_url:  str | None = None,
    api_key:  str | None = None,
) -> FIEConfig:
    """
    Builds FIEConfig from arguments or environment variables.
    """
    return FIEConfig(
        fie_url = (
            fie_url
            or os.getenv("FIE_URL", "http://localhost:8000")
        ).rstrip("/"),
        api_key = (
            api_key
            or os.getenv("FIE_API_KEY", "")
        ),
    )