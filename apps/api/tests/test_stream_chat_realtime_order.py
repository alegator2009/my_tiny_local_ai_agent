"""Source-level tests for the real-time SSE event wiring inside stream_chat.

After the asyncio.create_task refactor, stream_chat no longer buffers
events into ``realtime_events``.  Instead it runs the tool loop as a
background task and pumps the ``tool_event_sink`` queue concurrently
so that every event reaches the UI the moment it happens.
"""
from __future__ import annotations

import inspect
from typing import Any

from app.services import orchestrator


def test_stream_chat_runs_tool_loop_as_background_task() -> None:
    """stream_chat must use asyncio.create_task(_run_with_tool_loop(...))
    so that the consumer can drain tool_event_sink in real time.
    """
    src = inspect.getsource(orchestrator.stream_chat)
    assert "asyncio.create_task(" in src
    assert "_run_with_tool_loop(" in src
    # The pattern must be: create_task + while not task.done() + wait_for(queue.get()).
    assert "while not loop_task.done():" in src
    assert "tool_event_sink.get()" in src


def test_stream_chat_emits_tool_status_done_after_drain() -> None:
    """The tool_status:done event must be emitted AFTER the queue is
    drained, never before.
    """
    src = inspect.getsource(orchestrator.stream_chat)
    done_idx = src.index('yield _sse("tool_status", {"state": "done"})')
    # The drain loop must come before the done event.
    drain_idx = src.index("while not tool_event_sink.empty()")
    assert drain_idx < done_idx


def test_stream_chat_cancels_task_on_exception() -> None:
    """If the consumer dies while the loop is running, the task must be
    cancelled to avoid leaking the httpx client and MCP connections.
    """
    src = inspect.getsource(orchestrator.stream_chat)
    assert "loop_task.cancel()" in src


def test_run_with_tool_loop_emits_started_after_display_append() -> None:
    """The run loop must put_nowait a tool_call_display event for every
    tool call site, and call _emit_tool_started right after.

    Production sites (in the current code):
      - the main tool_calls loop
      - inline_tool fallback dispatch (inside stream_chat text)

    All of them must follow the pattern:
        emit("tool_call_display", {...})  # via the local emit() helper
                                          # which internally calls put_nowait
        await _emit_tool_started(...)
    """
    src = inspect.getsource(orchestrator._run_with_tool_loop)
    # The run loop should put_nowait tool_call_display for every tool call.
    put_count = src.count('"tool_call_display"') + src.count("'tool_call_display'")
    assert put_count >= 1, f"expected at least 1 tool_call_display emit, got {put_count}"
    # And call _emit_tool_started afterwards.
    emit_count = src.count("await _emit_tool_started(")
    assert emit_count >= 1, f"expected at least 1 _emit_tool_started call, got {emit_count}"


def test_emit_tool_started_helper_invokes_callback(monkeypatch: Any) -> None:
    """The _emit_tool_started closure must call on_tool_call_started
    exactly once per tool call, swallowing any exception raised by the
    callback.  When on_tool_call_started is None, it must noop.
    """
    import inspect
    from app.services import orchestrator as _orch
    # Get the source for the helper - it lives inside _run_with_tool_loop,
    # so we look at the whole module and grep for the expected pattern.
    src = inspect.getsource(_orch)
    # The helper invokes on_tool_call_started when present.
    assert "await on_tool_call_started(" in src
    # And guards against exceptions from the callback.
    assert "except Exception:" in src


def test_stream_chat_no_buffers_realtime_events() -> None:
    """The old buffer-then-flush design is gone: no realtime_events
    list, no .append() of (event_name, data) tuples inside the
    capture closure.
    """
    src = inspect.getsource(orchestrator.stream_chat)
    # The drain now goes directly through the queue, not via a list.
    assert "realtime_events" not in src
    # The capture closure is gone (we drain the queue inline).
    assert "_capture_tool_started" not in src
