from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class FIEConfig:
    """Holds all configuration needed to connect to the FIE server."""
    fie_url:  str    # URL of your running FIE FastAPI server
    api_key:  str    # API key you give to the user


def get_config(
    fie_url:  str | None = None,
    api_key:  str | None = None,
) -> FIEConfig:
    """
    Builds FIEConfig from arguments or environment variables.

    Usage in @monitor:
        @monitor(fie_url="https://your-server.com", api_key="fie-abc123")

    OR set environment variables in .env:
        FIE_URL=https://your-server.com
        FIE_API_KEY=fie-abc123
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