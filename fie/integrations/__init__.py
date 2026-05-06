"""
fie.integrations — Drop-in wrappers for popular LLM SDKs.

Usage:
    # OpenAI
    from fie.integrations import openai
    client = openai.Client(api_key="sk-...", fie_url="...", fie_api_key="...")
    response = client.chat.completions.create(model="gpt-4o", messages=[...])

    # Anthropic
    from fie.integrations import anthropic
    client = anthropic.Client(api_key="sk-ant-...", fie_url="...", fie_api_key="...")
    response = client.messages.create(model="claude-3-5-sonnet-20241022", messages=[...])

Both wrappers:
  - Run FIE adversarial scan on the prompt BEFORE the API call (blocks attacks)
  - Send prompt + response to FIE server in background (mode="monitor")
  - Or block and return corrected answer (mode="correct")
  - Never raise if FIE is unreachable — always return the real LLM response
"""
from fie.integrations import openai, anthropic

__all__ = ["openai", "anthropic"]
