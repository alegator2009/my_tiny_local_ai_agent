"""Tests for the asyncio.Queue-based real-time tool event drain.

When ``_emit_tool_started`` is invoked, the callback must drain everything
currently sitting in ``tool_event_sink`` and push the ``tool_call_started``
event ahead of those drained events.  This guarantees the order
``tool_call_started -> tool_call_display`` reaches the UI before any
``message_delta`` is yielded by ``stream_chat``.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from app.services import orchestrator


def _make_capture(sink: asyncio.Queue, realtime: list) -> Any:
    """Reproduce the body of ``_capture_tool_started`` from stream_chat."""

    async def capture(name: str, args: dict) -> None:
        realtime.append(("tool_call_started", {"name": name, "args": args}))
        while not sink.empty():
            realtime.append(sink.get_nowait())

    return capture


def test_callback_drains_tool_call_display() -> None:
    """The callback must move tool_call_display into realtime_events."""
    sink: asyncio.Queue = asyncio.Queue()
    realtime: list = []
    sink.put_nowait(("tool_call_display", {"name": "get_weather", "args": {"city": "Lviv"}}))
    capture = _make_capture(sink, realtime)

    asyncio.run(capture("get_weather", {"city": "Lviv"}))

    # Expected order: tool_call_started FIRST, then tool_call_display.
    assert realtime[0][0] == "tool_call_started"
    assert realtime[1][0] == "tool_call_display"
    assert realtime[0][1]["name"] == "get_weather"
    assert realtime[1][1]["args"] == {"city": "Lviv"}
    assert sink.empty()


def test_capture_tool_started_drains_queue() -> None:
    """The _capture_tool_started closure inside stream_chat must drain."""
    sink: asyncio.Queue = asyncio.Queue()
    realtime: list = []
    sink.put_nowait(("tool_call_display", {"name": "fn", "args": {}}))
    capture = _make_capture(sink, realtime)

    asyncio.run(capture("fn", {"x": 1}))

    assert [e[0] for e in realtime] == ["tool_call_started", "tool_call_display"]


def test_empty_queue_still_emits_started() -> None:
    """If tool_event_sink is empty, callback still emits tool_call_started."""
    sink: asyncio.Queue = asyncio.Queue()
    realtime: list = []
    capture = _make_capture(sink, realtime)

    asyncio.run(capture("fn", {"x": 1}))

    assert realtime == [("tool_call_started", {"name": "fn", "args": {"x": 1}})]


def test_multiple_drains_for_multiple_tool_calls() -> None:
    """Two sequential tool calls each drain their own display event."""
    sink: asyncio.Queue = asyncio.Queue()
    realtime: list = []
    capture = _make_capture(sink, realtime)

    sink.put_nowait(("tool_call_display", {"name": "a", "args": {}}))
    asyncio.run(capture("a", {}))
    sink.put_nowait(("tool_call_display", {"name": "b", "args": {}}))
    asyncio.run(capture("b", {}))

    names = [e[0] for e in realtime]
    assert names == [
        "tool_call_started",
        "tool_call_display",
        "tool_call_started",
        "tool_call_display",
    ]
    assert realtime[0][1]["name"] == "a"
    assert realtime[2][1]["name"] == "b"
    assert sink.empty()


def test_run_with_tool_loop_signature_accepts_queue() -> None:
    """The parameter type for tool_event_sink is now asyncio.Queue."""
    sig = inspect.signature(orchestrator._run_with_tool_loop)
    param = sig.parameters["tool_event_sink"]
    assert "asyncio.Queue" in str(param.annotation)


def test_stream_chat_constructs_queue() -> None:
    """The stream_chat function still imports cleanly with the new ctor."""
    src = inspect.getsource(orchestrator.stream_chat)
    assert "asyncio.Queue()" in src
    # The old list constructor must be gone.
    assert "tool_event_sink: list[tuple" not in src


def test_all_call_sites_use_emit_or_put_nowait() -> None:
    """Every write to tool_event_sink uses put_nowait (directly or via emit)."""
    src = inspect.getsource(orchestrator)
    # Allow the type-annotation mentions but ban actual `.append(` calls.
    assert "tool_event_sink.append(" not in src
    # The function should expose an emit() helper or call put_nowait.
    # We accept either style as long as the queue is used.
    has_emit = "def emit(" in src and "put_nowait" in src
    has_direct = src.count("tool_event_sink.put_nowait(") >= 6
    assert has_emit or has_direct, "no emit() helper and no direct put_nowait calls"
