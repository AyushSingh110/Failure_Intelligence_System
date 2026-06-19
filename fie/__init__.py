from fie.monitor import monitor
from fie.client import FIEClient
from fie.config import get_config, FIEConfig
from fie.adversarial import scan_prompt, scan_prompt_async, build_cwd_injection, ScanResult, LayerEvidence
from fie.preflight import preflight_check, GuardedResponse
from fie.output_scanner import scan_output, scan_output_async, OutputScanResult
from fie.stream_guard import stream_guard, astream_guard
from fie import integrations
from fie._telemetry import _ping_telemetry

# Kept in sync with pyproject.toml via package metadata (see fie/client.py).
from fie.client import SDK_VERSION as __version__

# anonymous usage ping — disable with FIE_NO_TELEMETRY=1
_ping_telemetry(__version__)
__all__      = [
    "monitor",
    "FIEClient",
    "get_config",
    "FIEConfig",
    "scan_prompt",
    "scan_prompt_async",
    "build_cwd_injection",
    "ScanResult",
    "LayerEvidence",
    "preflight_check",
    "GuardedResponse",
    "scan_output",
    "scan_output_async",
    "OutputScanResult",
    "stream_guard",
    "astream_guard",
    "integrations",
]
