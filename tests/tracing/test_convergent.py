"""Convergent reuses the trace replay and respects its content boundaries."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from docsgpt import tracing
from docsgpt.core.settings import settings
from docsgpt.llm.deadlines import timed_stream
from docsgpt.tracing import convergent as connection
from docsgpt.tracing.messages import record_input, record_output
from docsgpt.tracing.otel import _preview_attributes


@pytest.fixture(autouse=True)
def capture(monkeypatch):
    monkeypatch.setattr(settings, "TRACES_ENABLED", True)
    monkeypatch.setattr(settings, "TRACES_CAPTURE_CONTENT", True)
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "span_only")


def test_no_destination_preserves_existing_provider(monkeypatch):
    import convergent

    for name in ("CONVERGENT_API_KEY", "CONVERGENT_SPANS_DIR", "CONVERGENT_TRACES_EXPORTER"):
        monkeypatch.delenv(name, raising=False)
    init = Mock()
    monkeypatch.setattr(convergent, "init", init)
    assert connection.configured_provider() is None
    init.assert_not_called()


@pytest.mark.parametrize("release", [None, "deployment-42"])
def test_release_defaults_to_application_version(monkeypatch, release):
    import convergent

    monkeypatch.setenv("CONVERGENT_SPANS_DIR", "/unused")
    monkeypatch.delenv("CONVERGENT_RELEASE", raising=False)
    if release:
        monkeypatch.setenv("CONVERGENT_RELEASE", release)
    init = Mock(return_value=SimpleNamespace(enabled=True))
    provider = object()
    monkeypatch.setattr(convergent, "init", init)
    monkeypatch.setattr(convergent, "tracer_provider", lambda: provider)
    assert connection.configured_provider() is provider
    init.assert_called_once_with(release=release or "1.1.0")


def test_stream_reader_keeps_enclosing_agent_parent():
    trace = tracing.start_trace(source="test")

    def read():
        with tracing.span(tracing.KIND_LLM, "chat model"):
            yield "answer"

    with tracing.activate(trace):
        with tracing.span(tracing.KIND_AGENT, "agent") as agent:
            assert list(timed_stream(read, first_token=5, stall=5)) == ["answer"]
    trace.finish()
    model = next(s for s in trace.spans if s.kind == tracing.KIND_LLM)
    assert model.parent_id == agent.id
    assert model.status == tracing.STATUS_OK


def test_native_files_are_labels_and_secrets_are_redacted():
    trace = tracing.start_trace(source="test")
    with tracing.activate(trace):
        with tracing.span(tracing.KIND_LLM, "chat") as span:
            record_input(span, [{"role": "user", "content": [
                {"type": "text", "text": "Read this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}},
                {"type": "file", "file": {"file_data": "PRIVATE"}},
                {"api_key": "SECRET", "data": b"PRIVATE"},
            ]}])
    preview = json.dumps(span.previews)
    assert "PRIVATE" not in preview
    assert "SECRET" not in preview
    assert "omitted" in preview


def test_streamed_tool_output_is_actual_response():
    trace = tracing.start_trace(source="test")
    # Real OpenAI choice/delta shape, including fragmented arguments.
    def chunk(arguments, name=None, call_id=None):
        call = SimpleNamespace(index=0, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
        return SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[call]), finish_reason=None)

    with tracing.activate(trace):
        with tracing.span(tracing.KIND_LLM, "chat") as span:
            record_output(span, SimpleNamespace(provider_name="openai"), [
                chunk('{"query":', "search", "call-1"), chunk('"guide"}'),
            ], stream=True)
    messages = span.previews["output_messages"]
    assert messages[0]["parts"] == [{
        "type": "tool_call", "id": "call-1", "name": "search", "arguments": {"query": "guide"},
    }]
    exported = _preview_attributes(tracing.KIND_LLM, span.previews)
    assert json.loads(exported["gen_ai.output.messages"]) == messages


def test_content_capture_remains_opt_in(monkeypatch):
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    trace = tracing.start_trace(source="test")
    with tracing.activate(trace):
        with tracing.span(tracing.KIND_LLM, "chat") as span:
            record_input(span, [{"role": "user", "content": "private"}])
            record_output(span, SimpleNamespace(provider_name="openai"), "private", stream=False)
    assert not span.previews


def test_malformed_content_cannot_break_agent():
    span = Mock()
    record_input(span, [None])
    span.preview.assert_not_called()
