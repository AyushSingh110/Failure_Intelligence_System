"""
fie.integrations.fastapi — FIE middleware for FastAPI / Starlette.

Drop this one line into any FastAPI app to protect every route:

    from fie.integrations.fastapi import FIEMiddleware
    app.add_middleware(FIEMiddleware)

Options:
    block_attacks  bool   Block detected attacks (default True). False → log only.
    threshold      float  Confidence threshold override (default: FIE global threshold).
    exempt_paths   list   URL paths to skip (e.g. ["/health", "/docs"]).
    local_mode     bool   Use local scan_prompt() instead of FIE server (default True).
    fie_url        str    FIE server URL for server mode (e.g. "http://localhost:8000").
    fie_api_key    str    FIE API key for server mode.

Response on blocked request (HTTP 400):
    {"error": "adversarial_input_blocked",
     "attack_type": "PROMPT_INJECTION",
     "confidence": 0.87,
     "message": "Request blocked by FIE adversarial scan."}
"""
from __future__ import annotations

import json
from typing import Callable, Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class FIEMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        block_attacks: bool = True,
        threshold: float | None = None,
        exempt_paths: Sequence[str] = ("/health", "/docs", "/openapi.json", "/redoc"),
        local_mode: bool = True,
        fie_url: str | None = None,
        fie_api_key: str | None = None,
    ) -> None:
        super().__init__(app)
        self.block_attacks  = block_attacks
        self.threshold      = threshold
        self.exempt_paths   = set(exempt_paths)
        self.local_mode     = local_mode
        self.fie_url        = fie_url
        self.fie_api_key    = fie_api_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        prompt = await self._extract_prompt(request)
        if not prompt or len(prompt.strip()) < 10:
            return await call_next(request)

        attack_type, confidence, is_attack = self._scan(prompt)

        if is_attack and self.block_attacks:
            return JSONResponse(
                status_code=400,
                content={
                    "error":       "adversarial_input_blocked",
                    "attack_type": attack_type,
                    "confidence":  confidence,
                    "message":     "Request blocked by FIE adversarial scan.",
                },
            )

        return await call_next(request)

    async def _extract_prompt(self, request: Request) -> str | None:
        try:
            body = await request.body()
            if not body:
                return None
            ct = request.headers.get("content-type", "")
            if "application/json" in ct:
                data = json.loads(body)
                # Common field names across LLM APIs
                for field in ("prompt", "input", "query", "message", "text", "content"):
                    if isinstance(data.get(field), str):
                        return data[field]
                # OpenAI-style messages array
                msgs = data.get("messages")
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    if isinstance(last, dict):
                        content = last.get("content", "")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    return block.get("text", "")
            return None
        except Exception:
            return None

    def _scan(self, prompt: str) -> tuple[str | None, float, bool]:
        try:
            if self.local_mode:
                from fie.adversarial import scan_prompt
                result = scan_prompt(prompt, threshold=self.threshold)
                return result.attack_type, result.confidence, result.is_attack
            else:
                return self._scan_server(prompt)
        except Exception:
            return None, 0.0, False

    def _scan_server(self, prompt: str) -> tuple[str | None, float, bool]:
        if not self.fie_url:
            return None, 0.0, False
        try:
            import requests as _req
            headers = {"Content-Type": "application/json"}
            if self.fie_api_key:
                headers["X-API-Key"] = self.fie_api_key
            resp = _req.post(
                f"{self.fie_url.rstrip('/')}/api/v1/scan",
                json={"prompt": prompt, "threshold": self.threshold},
                headers=headers,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("attack_type"), data.get("confidence", 0.0), data.get("is_attack", False)
        except Exception:
            pass
        return None, 0.0, False
