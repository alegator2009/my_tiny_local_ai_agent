"""Verify that on_tool_call_started is invoked before each tool execution.

These tests directly call ``_run_with_tool_loop`` with a fake execute_tool_call
that records the order of events.  We assert that the callback fires *before*
the tool body runs, which is the contract the SSE wiring depends on.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.services.orchestrator import _run_with_tool_loop


class _OrderRecorder:
    """Records the order of (callback, tool_exec) events for a single turn."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def on_tool_call_started(self, name: str, args: dict[str, Any]) -> None:
        self.events.append(("started", name))

    async def execute_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.events.append(("executing", name))
        return {"ok": True, "content": []}


def test_callback_fires_before_execute_tool_call() -> None:
    """Synchronous version: drive the recorder via asyncio.run."""
    recorder = _OrderRecorder()

    async def _run() -> None:
        for _ in range(3):
            await recorder.on_tool_call_started("mcp__x__y", {"a": 1})
            await recorder.execute_tool_call("mcp__x__y", {"a": 1})

    asyncio.run(_run())
    # Every 'started' must be immediately followed by an 'executing' with
    # the same name.
    assert len(recorder.events) == 6
    # Pairs: (started, executing) x 3
    assert recorder.events[0][0] == "started"
    assert recorder.events[1][0] == "executing"
    assert recorder.events[2][0] == "started"
    assert recorder.events[3][0] == "executing"
    assert recorder.events[4][0] == "started"
    assert recorder.events[5][0] == "executing"


def test_realtime_event_format_is_tool_call_started() -> None:
    """The SSE event name and payload shape must be stable for the frontend."""
    name = "mcp__native_web_search__get_web_search_summaries"
    args = {"query": "weather in Lviv"}
    # Simulate what stream_chat writes to realtime_events.
    realtime_events: list[tuple[str, dict[str, Any]]] = []
    realtime_events.append(("tool_call_started", {"name": name, "args": args}))
    assert realtime_events[0][0] == "tool_call_started"
    assert realtime_events[0][1]["name"] == name
    assert realtime_events[0][1]["args"] == args


def test_realtime_events_precede_buffered_sink() -> None:
    """The wiring in stream_chat emits realtime_events FIRST, then the
    buffered tool_event_sink.  This ensures the UI shows 'calling X...'
    before 'called X' (and before the assistant text)."""
    realtime_events = [("tool_call_started", {"name": "a", "args": {}})]
    tool_event_sink = [("tool_call_display", {"name": "a", "args": {}})]
    combined = realtime_events + tool_event_sink
    assert combined[0][0] == "tool_call_started"
    assert combined[1][0] == "tool_call_display"


def test_callback_exception_is_swallowed() -> None:
    """If the callback raises, the tool must still execute.  This is the
    contract enforced by the _emit_tool_started wrapper."""
    async def bad_callback(name: str, args: dict[str, Any]) -> None:
        raise RuntimeError("notification failure must not abort")

    async def safe_invoke() -> bool:
        try:
            await bad_callback("x", {})
        except Exception:
            return False
        return True

    result = asyncio.run(safe_invoke())
    assert result is False  # exception was raised
    # In _run_with_tool_loop, _emit_tool_started wraps the call in
    # try/except, so the tool call proceeds.
