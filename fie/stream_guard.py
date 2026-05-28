"""
Streaming output interception (Flaw: No streaming interception).

The output scanner in output_scanner.py is post-hoc — it runs after the full response
is delivered to the user. For streaming LLMs, a jailbroken response reaches the user
chunk-by-chunk before any post-hoc scan can act.

stream_guard() wraps any sync or async generator and intercepts mid-stream:

  1. Buffer the first SCAN_WINDOW_CHARS characters (default 400).
  2. Run scan_output on the buffer.
  3. If CLEAN → yield buffered chunks and pass through all remaining chunks untouched.
  4. If FLAGGED → discard the buffer, yield a refusal message, stop iteration.
     The caller receives the refusal as if the model produced it — no exception raised.

Usage (sync):
    from fie.stream_guard import stream_guard

    raw_stream = openai_client.chat.completions.create(stream=True, ...)
    for chunk in stream_guard(raw_stream, prompt=user_prompt, text_extractor=lambda c: c.choices[0].delta.content or ""):
        print(chunk, end="", flush=True)

Usage (async):
    async for chunk in astream_guard(raw_stream, prompt=user_prompt, text_extractor=...):
        yield chunk

For plain-string generators (text already extracted per chunk):
    for text in stream_guard(my_gen, prompt=prompt):
        print(text, end="")
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Callable, Generator, Optional, TypeVar

from fie.output_scanner import scan_output

logger = logging.getLogger("fie.stream")

T = TypeVar("T")

_REFUSAL = (
    "I'm unable to continue this response. "
    "The content was flagged by the safety layer before delivery was complete."
)

# How many characters to buffer before deciding whether to pass through or block.
# 400 chars is enough to catch policy-echo phrases in the first paragraph while
# keeping the buffer latency negligible (≈ 2–3 chunks from most LLM APIs).
SCAN_WINDOW_CHARS: int = 400


def stream_guard(
    stream:         "Generator[T, None, None]",
    prompt:         str = "",
    session_id:     Optional[str] = None,
    text_extractor: Optional[Callable[[T], str]] = None,
    refusal:        str = _REFUSAL,
    scan_window:    int = SCAN_WINDOW_CHARS,
) -> "Generator[T | str, None, None]":
    """
    Synchronous streaming guard. Wraps a generator and intercepts harmful output.

    Args:
        stream:         Any iterable / generator yielding chunks (str or SDK objects).
        prompt:         Original user prompt (passed to scan_output for context).
        session_id:     Optional session ID for feedback store logging.
        text_extractor: Function to extract text string from each chunk.
                        None → chunk is treated as a plain string.
        refusal:        Message yielded instead of blocked content.
        scan_window:    Characters to buffer before scan decision (default 400).
    """
    _extract = text_extractor or (lambda c: c if isinstance(c, str) else "")

    buffer_chunks: list[T] = []
    buffer_text = ""
    decision_made = False
    blocked = False

    for chunk in stream:
        text = _extract(chunk)

        if not decision_made:
            buffer_chunks.append(chunk)
            buffer_text += text

            if len(buffer_text) >= scan_window:
                result = scan_output(prompt, buffer_text, session_id)
                decision_made = True

                if result.is_flagged:
                    blocked = True
                    logger.warning(
                        "[FIE:stream] INTERCEPTED mid-stream | type=%s | confidence=%.2f | matched=%.80s",
                        result.flag_type, result.confidence,
                        result.evidence.get("matched", ""),
                    )
                    # Record in feedback store
                    try:
                        from fie.feedback_store import record as _fb_record
                        _fb_record(
                            kind="output_flag",
                            flag_type=result.flag_type,
                            confidence=result.confidence,
                            prompt=prompt,
                            matched=result.evidence.get("matched", ""),
                            session_id=session_id,
                        )
                    except Exception:
                        pass
                    yield refusal
                    return
                else:
                    # Safe — flush buffered chunks
                    for buffered in buffer_chunks:
                        yield buffered
                    buffer_chunks = []
                    buffer_text = ""
        else:
            if not blocked:
                yield chunk

    # Stream ended before buffer filled SCAN_WINDOW — scan whatever we have
    if not decision_made and buffer_chunks:
        result = scan_output(prompt, buffer_text, session_id)
        if result.is_flagged:
            logger.warning(
                "[FIE:stream] INTERCEPTED at end-of-stream | type=%s | confidence=%.2f",
                result.flag_type, result.confidence,
            )
            try:
                from fie.feedback_store import record as _fb_record
                _fb_record(
                    kind="output_flag", flag_type=result.flag_type,
                    confidence=result.confidence, prompt=prompt,
                    matched=result.evidence.get("matched", ""), session_id=session_id,
                )
            except Exception:
                pass
            yield refusal
        else:
            for buffered in buffer_chunks:
                yield buffered


async def astream_guard(
    stream:         "AsyncGenerator[T, None]",
    prompt:         str = "",
    session_id:     Optional[str] = None,
    text_extractor: Optional[Callable[[T], str]] = None,
    refusal:        str = _REFUSAL,
    scan_window:    int = SCAN_WINDOW_CHARS,
) -> "AsyncGenerator[T | str, None]":
    """
    Async streaming guard. Same semantics as stream_guard but for async generators.

    Usage:
        async for chunk in astream_guard(client.stream(...), prompt=prompt):
            yield chunk
    """
    _extract = text_extractor or (lambda c: c if isinstance(c, str) else "")

    buffer_chunks: list[T] = []
    buffer_text = ""
    decision_made = False
    blocked = False

    async for chunk in stream:
        text = _extract(chunk)

        if not decision_made:
            buffer_chunks.append(chunk)
            buffer_text += text

            if len(buffer_text) >= scan_window:
                result = scan_output(prompt, buffer_text, session_id)
                decision_made = True

                if result.is_flagged:
                    blocked = True
                    logger.warning(
                        "[FIE:stream] INTERCEPTED mid-stream (async) | type=%s | confidence=%.2f",
                        result.flag_type, result.confidence,
                    )
                    try:
                        from fie.feedback_store import record as _fb_record
                        _fb_record(
                            kind="output_flag", flag_type=result.flag_type,
                            confidence=result.confidence, prompt=prompt,
                            matched=result.evidence.get("matched", ""), session_id=session_id,
                        )
                    except Exception:
                        pass
                    yield refusal
                    return
                else:
                    for buffered in buffer_chunks:
                        yield buffered
                    buffer_chunks = []
                    buffer_text = ""
        else:
            if not blocked:
                yield chunk

    if not decision_made and buffer_chunks:
        result = scan_output(prompt, buffer_text, session_id)
        if result.is_flagged:
            logger.warning(
                "[FIE:stream] INTERCEPTED at end-of-stream (async) | type=%s | confidence=%.2f",
                result.flag_type, result.confidence,
            )
            try:
                from fie.feedback_store import record as _fb_record
                _fb_record(
                    kind="output_flag", flag_type=result.flag_type,
                    confidence=result.confidence, prompt=prompt,
                    matched=result.evidence.get("matched", ""), session_id=session_id,
                )
            except Exception:
                pass
            yield refusal
        else:
            for buffered in buffer_chunks:
                yield buffered
