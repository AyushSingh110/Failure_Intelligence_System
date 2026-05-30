from __future__ import annotations
import os
import platform
import threading

_TELEMETRY_URL = "https://failure-intelligence-system.onrender.com/ping"


def _ping_telemetry(version: str) -> None:
    if os.getenv("FIE_NO_TELEMETRY", "").strip() in ("1", "true", "yes"):
        return

    def _send():
        try:
            import urllib.request, json
            payload = json.dumps({
                "version": version,
                "os": platform.system(),
                "python": platform.python_version(),
            }).encode()
            req = urllib.request.Request(
                _TELEMETRY_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()
