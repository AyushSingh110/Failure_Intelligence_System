"""
fie.integrations — Drop-in wrappers for popular LLM SDKs and web frameworks.

Usage:
    # OpenAI
    from fie.integrations import openai
    client = openai.Client(api_key="sk-...", fie_url="...", fie_api_key="...")
    response = client.chat.completions.create(model="gpt-4o", messages=[...])

    # Anthropic
    from fie.integrations import anthropic
    client = anthropic.Client(api_key="sk-ant-...", fie_url="...", fie_api_key="...")
    response = client.messages.create(model="claude-3-5-sonnet-20241022", messages=[...])

    # FastAPI / Starlette (one-line middleware — protects every route)
    from fie.integrations.fastapi import FIEMiddleware
    app.add_middleware(FIEMiddleware)

SDK wrappers:
  - Run FIE adversarial scan on the prompt BEFORE the API call (blocks attacks)
  - Send prompt + response to FIE server in background (mode="monitor")
  - Or block and return corrected answer (mode="correct")
  - Never raise if FIE is unreachable — always return the real LLM response
"""
import importlib
import importlib.util

from fie.integrations import openai, anthropic

# `fastapi` is loaded on first use, not here: it needs Starlette, which a base
# install does not have, and `import fie` must not depend on a web framework.
# It is left out of __all__ so that `from fie.integrations import *` cannot
# trigger that import; request it by name, as shown above.
__all__ = ["openai", "anthropic"]

_LAZY = ("fastapi",)


def __getattr__(name: str):
    if name in _LAZY:
        # Raises ModuleNotFoundError (name="starlette") when Starlette is absent;
        # see fie/integrations/fastapi.py. On success the import system binds the
        # submodule on this package, so this runs once.
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = set(globals())
    # Listed only where it can be imported, so tools that walk dir() and read
    # every attribute do not fail in a base install. dir() itself never raises.
    try:
        available = importlib.util.find_spec("starlette") is not None
    except (ImportError, ValueError):
        available = False
    if available:
        names.update(_LAZY)
    return sorted(names)
