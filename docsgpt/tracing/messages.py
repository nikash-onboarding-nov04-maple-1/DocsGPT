"""Bounded GenAI message previews; binary files are never copied into telemetry."""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable

from docsgpt.tracing.otel import content_capture_enabled


def _best_effort(function: Callable[..., Any]) -> Callable[..., Any]:
    """Keep malformed provider content from interfering with the agent loop."""
    @functools.wraps(function)
    def record(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception:
            logging.getLogger(__name__).debug("Could not normalize trace content", exc_info=True)
    return record


def _safe_content(value: Any) -> Any:
    """Replace native file/image blocks and data URLs with labels."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[binary file, {len(value)} bytes]"
    if isinstance(value, str):
        return "[inline file omitted]" if value.startswith("data:") else value
    if isinstance(value, list):
        return [_safe_content(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") in ("image_url", "input_image", "file", "input_file", "document", "image"):
            return {"type": "text", "text": "[file content omitted]"}
        return {k: _safe_content(v) for k, v in value.items()}
    return value


@_best_effort
def record_input(span: Any, messages: list[dict]) -> None:
    """Keep redacted, bounded input excerpts only when OTel content capture is on."""
    if not span or not content_capture_enabled():
        return
    converted = []
    for message in messages:
        role = message.get("role", "user")
        content = _safe_content(message.get("content"))
        parts = []
        if content:
            parts.append({"type": "text", "content": content})
        if role == "tool":
            parts = [{"type": "tool_call_response", "id": message.get("tool_call_id"), "response": content}]
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append({"type": "tool_call", "id": call.get("id"), "name": function.get("name"),
                          "arguments": function.get("arguments")})
        if parts:
            converted.append({"role": role, "parts": parts})
    span.preview("input_messages", converted)


@_best_effort
def record_output(span: Any, llm: Any, output: Any, *, stream: bool) -> None:
    """Normalize actual provider replies, merging streamed tool-call fragments."""
    if not span or output is None or not content_capture_enabled():
        return
    from docsgpt.llm.handlers.handler_creator import LLMHandlerCreator

    handler = LLMHandlerCreator.create_handler(getattr(llm, "provider_name", "openai"))
    parts, calls = [], {}
    finish_reason = None
    for chunk in (output if stream and not isinstance(output, str) else [output]):
        # Do not export private reasoning/thought events.
        if isinstance(chunk, dict) and chunk.get("type") == "thought":
            continue
        parsed = handler.parse_response(chunk)
        if parsed.content:
            if parts and parts[-1].get("type") == "text":
                parts[-1]["content"] += parsed.content
            else:
                parts.append({"type": "text", "content": parsed.content})
        finish_reason = parsed.finish_reason or finish_reason
        for call in parsed.tool_calls or []:
            key = call.index if call.index is not None else call.id
            item = calls.setdefault(key, {"type": "tool_call", "id": "", "name": "", "arguments": ""})
            item["id"] = call.id or item["id"]
            item["name"] += call.name or ""
            if isinstance(call.arguments, str):
                item["arguments"] += call.arguments
            else:
                item["arguments"] = call.arguments
    for call in calls.values():
        if isinstance(call["arguments"], str):
            try:
                call["arguments"] = json.loads(call["arguments"])
            except (ValueError, TypeError):
                pass
        parts.append(call)
    if parts:
        message = {"role": "assistant", "parts": _safe_content(parts)}
        if finish_reason:
            message["finish_reason"] = finish_reason
        span.preview("output_messages", [message])
