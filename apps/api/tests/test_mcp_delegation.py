"""Tests for MCPToolRegistry._resolve_delegation (skills-mcp delegation chain).

The skills-mcp wrapper can return a content block whose text starts with
``DELEGATE:{...}`` to indicate that the skill is a thin wrapper around
another MCP tool.  In that case ``_resolve_delegation`` parses the JSON
payload, calls the named tool, and substitutes the result.

We exercise the method directly with a fake ``call_tool`` (a simple
``AsyncMock``) so we don't need a real MCP client or the full registry
to be configured.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.services.mcp import MCPToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(text: str) -> dict[str, Any]:
    """Build a content block matching the MCP ``content[]`` shape."""
    return {"type": "text", "text": text}


def _make_result(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "content": list(blocks)}


def _make_registry(call_tool: AsyncMock) -> MCPToolRegistry:
    """Build a minimal MCPToolRegistry with a stubbed ``call_tool``.

    We construct the registry and immediately replace ``call_tool`` with
    a mock so we don't need to wire up real clients/tools.
    """
    registry = MCPToolRegistry()
    registry.call_tool = call_tool  # type: ignore[method-assign]
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_delegate_passes_target_tool_and_args() -> None:
    """DELEGATE: directive is parsed, target tool called with payload args."""
    call_tool = AsyncMock(return_value=_make_result(_make_block("final answer")))
    registry = _make_registry(call_tool)

    payload = {"tool": "native-web-search", "args": {"query": "weather Dnipro"}}
    raw = _make_result(_make_block("DELEGATE:" + json.dumps(payload)))

    result = asyncio.run(registry._resolve_delegation(raw))

    # call_tool was invoked with the parsed target tool and args.
    call_tool.assert_awaited_once_with("native-web-search", {"query": "weather Dnipro"})
    # The substituted result is the final answer.
    assert result == {"ok": True, "content": [_make_block("final answer")]}


def test_no_delegate_returns_input_unchanged() -> None:
    """When there's no DELEGATE: directive, the result is returned as-is."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    raw = _make_result(_make_block("plain text"), _make_block("more text"))
    result = asyncio.run(registry._resolve_delegation(raw))

    assert result is raw
    call_tool.assert_not_awaited()


def test_invalid_json_in_delegate_returns_input() -> None:
    """Malformed JSON after DELEGATE: must NOT crash - input is returned."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    raw = _make_result(_make_block("DELEGATE:this is not json"))
    result = asyncio.run(registry._resolve_delegation(raw))

    assert result is raw
    call_tool.assert_not_awaited()


def test_delegate_without_tool_field_returns_input() -> None:
    """If the JSON parses but has no ``tool`` field, do not invoke call_tool."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    raw = _make_result(_make_block("DELEGATE:" + json.dumps({"args": {"x": 1}})))
    result = asyncio.run(registry._resolve_delegation(raw))

    assert result is raw
    call_tool.assert_not_awaited()


def test_delegate_default_args_when_missing() -> None:
    """If ``args`` is absent in the payload, default to empty dict."""
    call_tool = AsyncMock(return_value=_make_result(_make_block("ok")))
    registry = _make_registry(call_tool)

    raw = _make_result(_make_block("DELEGATE:" + json.dumps({"tool": "wikipedia"})))
    result = asyncio.run(registry._resolve_delegation(raw))

    call_tool.assert_awaited_once_with("wikipedia", {})
    assert result == {"ok": True, "content": [_make_block("ok")]}


def test_delegate_recursion_depth_limit() -> None:
    """A chain of DELEGATE directives longer than 2 must stop recursing."""
    # Mock always returns a DELEGATE: directive - the resolver must stop
    # calling the tool after depth == 2 to avoid an infinite loop.
    call_tool = AsyncMock(
        side_effect=lambda *_a, **_kw: _make_result(
            _make_block("DELEGATE:" + json.dumps({"tool": "loop", "args": {}}))
        )
    )
    registry = _make_registry(call_tool)

    raw = _make_result(_make_block("DELEGATE:" + json.dumps({"tool": "loop"})))
    result = asyncio.run(registry._resolve_delegation(raw))

    # Walk through the chain:
    #   depth=0: see DELEGATE: in `raw`, call_tool(loop) -> block A
    #   depth=1: see DELEGATE: in block A, call_tool(loop) -> block B
    #   depth=2: see DELEGATE: in block B, return block B (no more recursion)
    assert call_tool.await_count == 2
    # The result is the second DELEGATE block - the resolver must NOT have
    # made a third call_tool invocation.
    assert isinstance(result, dict)
    assert "content" in result
    assert result["content"][0]["text"].startswith("DELEGATE:")


def test_tool_args_returns_payload_as_content() -> None:
    """TOOL_ARGS: directive surfaces the JSON as ``ok/content``."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    payload = {"foo": "bar", "n": 42}
    raw = _make_result(_make_block("TOOL_ARGS:" + json.dumps(payload)))
    result = asyncio.run(registry._resolve_delegation(raw))

    assert result == {"ok": True, "content": payload}
    call_tool.assert_not_awaited()


def test_tool_args_invalid_json_returns_input() -> None:
    """Malformed TOOL_ARGS: must not crash - return input."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    raw = _make_result(_make_block("TOOL_ARGS:not json"))
    result = asyncio.run(registry._resolve_delegation(raw))

    assert result is raw
    call_tool.assert_not_awaited()


def test_non_string_text_blocks_are_skipped() -> None:
    """Content blocks with non-string ``text`` are skipped gracefully."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    raw = {
        "ok": True,
        "content": [
            {"type": "text", "text": None},  # text=None is not a string
            {"type": "text", "text": 12345},  # text=number is not a string
            {"type": "image"},  # no text at all
            _make_block("normal block"),  # a normal text block
        ],
    }
    result = asyncio.run(registry._resolve_delegation(raw))

    assert result is raw
    call_tool.assert_not_awaited()


def test_non_dict_input_returns_input() -> None:
    """If the result is not a dict, return it unchanged."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    # Pass a non-dict result.
    for not_a_dict in (None, "string", 42, ["a", "list"]):
        result = asyncio.run(registry._resolve_delegation(not_a_dict))  # type: ignore[arg-type]
        assert result == not_a_dict

    call_tool.assert_not_awaited()


def test_non_list_content_returns_input() -> None:
    """If ``content`` is not a list, return the result as-is."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    raw = {"ok": True, "content": "not a list"}
    result = asyncio.run(registry._resolve_delegation(raw))

    assert result is raw
    call_tool.assert_not_awaited()


def test_empty_content_list_returns_input() -> None:
    """An empty content list means no directives - return input as-is."""
    call_tool = AsyncMock()
    registry = _make_registry(call_tool)

    raw = {"ok": True, "content": []}
    result = asyncio.run(registry._resolve_delegation(raw))

    assert result is raw
    call_tool.assert_not_awaited()


def test_delegate_only_first_matching_block_triggers() -> None:
    """When multiple text blocks exist, the first DELEGATE: triggers the call.

    The other blocks are not consulted once the directive is found.
    """
    call_tool = AsyncMock(return_value=_make_result(_make_block("DONE")))
    registry = _make_registry(call_tool)

    raw = _make_result(
        _make_block("DELEGATE:" + json.dumps({"tool": "t1", "args": {"a": 1}})),
        _make_block("DELEGATE:" + json.dumps({"tool": "t2", "args": {"a": 2}})),
    )
    result = asyncio.run(registry._resolve_delegation(raw))

    call_tool.assert_awaited_once_with("t1", {"a": 1})
    assert result == {"ok": True, "content": [_make_block("DONE")]}


def test_delegate_chains_two_levels() -> None:
    """A delegating tool that itself returns a DELEGATE: result is followed once."""
    final_block = _make_block("final-final")

    # First call returns DELEGATE:, second call returns the final answer.
    call_tool = AsyncMock(
        side_effect=[
            _make_result(
                _make_block("DELEGATE:" + json.dumps({"tool": "second", "args": {}}))
            ),
            _make_result(final_block),
        ]
    )
    registry = _make_registry(call_tool)

    raw = _make_result(
        _make_block("DELEGATE:" + json.dumps({"tool": "first", "args": {}}))
    )
    result = asyncio.run(registry._resolve_delegation(raw))

    # Both tools were called in order.
    assert call_tool.await_count == 2
    call_tool.await_args_list[0].assert_called_with("first", {})
    call_tool.await_args_list[1].assert_called_with("second", {})
    # The final result is unwrapped from the second delegation.
    assert result == {"ok": True, "content": [final_block]}


def test_orchestrator_invokes_resolve_delegation_in_call_tool() -> None:
    """Source-level check: call_tool calls _resolve_delegation between client and normalize."""
    import inspect

    from app.services import mcp as mcp_module

    src = inspect.getsource(mcp_module.MCPToolRegistry.call_tool)
    # The call must happen after client.call_tool and before _normalize_mcp_result.
    assert "client.call_tool" in src
    assert "_resolve_delegation" in src
    assert "_normalize_mcp_result" in src
    # Ordering: client.call_tool < _resolve_delegation < _normalize_mcp_result.
    pos_client = src.find("client.call_tool")
    pos_delegate = src.find("_resolve_delegation")
    pos_normalize = src.find("_normalize_mcp_result")
    assert pos_client < pos_delegate < pos_normalize, (
        f"order wrong: client={pos_client} delegate={pos_delegate} normalize={pos_normalize}"
    )
