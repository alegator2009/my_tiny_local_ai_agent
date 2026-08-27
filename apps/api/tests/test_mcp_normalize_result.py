"""Tests for ``mcp._normalize_mcp_result``.

MCP servers do not all return the same shape: some wrap the payload in
``{"result": ...}``, some return an ``{"error": ...}`` envelope, and some
return a plain dict with no ``ok`` field. The orchestrator and the tool
validator both expect a uniform ``{"ok": bool, ...}`` shape so a single
``result.get("ok")`` check is enough to decide whether to surface an error
to the model. ``_normalize_mcp_result`` is the boundary that enforces that.
"""

from __future__ import annotations

import pytest

from app.services.mcp import _normalize_mcp_result


def test_non_dict_result_is_wrapped_as_error():
    out = _normalize_mcp_result("not a dict")
    assert out["ok"] is False
    assert "non-dict" in out["error"]


def test_non_dict_list_result_is_wrapped_as_error():
    out = _normalize_mcp_result([1, 2, 3])
    assert out["ok"] is False
    assert "non-dict" in out["error"]


def test_string_error_is_normalised():
    out = _normalize_mcp_result({"error": "boom"})
    assert out["ok"] is False
    assert out["error"] == "boom"


def test_dict_error_extracts_message_and_code():
    out = _normalize_mcp_result(
        {"error": {"message": "bad input", "code": -32600}}
    )
    assert out["ok"] is False
    assert out["error"] == "bad input"
    assert out["error_code"] == -32600
    assert out["raw_error"] == {"message": "bad input", "code": -32600}


def test_dict_error_falls_back_to_code_when_no_message():
    out = _normalize_mcp_result({"error": {"code": 42}})
    assert out["ok"] is False
    assert out["error"] == "42"


def test_dict_error_preserves_stringified_form():
    out = _normalize_mcp_result({"error": {"message": "x"}})
    assert "raw_error" in out


def test_is_error_true_field_is_mirrored_to_ok_false():
    out = _normalize_mcp_result({"isError": True, "content": []})
    assert out["ok"] is False
    assert out["isError"] is True


def test_is_error_false_field_is_mirrored_to_ok_true():
    out = _normalize_mcp_result({"isError": False, "content": [{"text": "hi"}]})
    assert out["ok"] is True


def test_plain_dict_without_ok_gets_ok_true_default():
    out = _normalize_mcp_result({"content": [{"text": "hello"}]})
    assert out["ok"] is True
    assert out["content"] == [{"text": "hello"}]


def test_existing_ok_field_is_preserved():
    out = _normalize_mcp_result({"ok": False, "error": "user said no"})
    assert out["ok"] is False
    assert out["error"] == "user said no"


def test_error_envelope_wins_over_content():
    # If a server returns both ``error`` and ``content`` (which it should
    # not, but better be safe), the error wins.
    out = _normalize_mcp_result(
        {"error": "nope", "content": [{"text": "ignored"}]}
    )
    assert out["ok"] is False
    assert out["error"] == "nope"
    assert "content" not in out
