"""Tests for ``filter_unknown_properties`` / ``coerce_for_send`` / ``sanitize_for_send``.

The MCP server is the source of truth on what is acceptable, so we adapt the
outgoing arguments instead of arguing with it:

* extra keys that the schema rejects (``additionalProperties: false``) are
  dropped before the call;
* scalar values that are technically the wrong type (``"42"`` for an
  integer field) are coerced;
* everything else passes through unchanged.

These helpers are conservative: they never drop a value the model
intended to send and they never lose data when the declared type does not
match (e.g. a non-numeric string is left alone instead of raising).
"""

from __future__ import annotations

import pytest

from app.services.schema_sanitizer import (
    coerce_for_send,
    filter_unknown_properties,
    sanitize_for_send,
)


# ---------------------------------------------------------------------------
# filter_unknown_properties
# ---------------------------------------------------------------------------


def test_no_schema_returns_arguments_unchanged():
    args = {"a": 1, "b": 2}
    assert filter_unknown_properties(args, None) is args or filter_unknown_properties(args, None) == args


def test_non_strict_schema_keeps_all_keys():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        # additionalProperties is implicit True / not set
    }
    out = filter_unknown_properties({"a": "x", "b": 2}, schema)
    assert out == {"a": "x", "b": 2}


def test_strict_schema_drops_unknown_keys():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
    }
    out = filter_unknown_properties({"a": "x", "b": 2, "c": True}, schema)
    assert out == {"a": "x"}


def test_non_dict_arguments_returned_unchanged():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    assert filter_unknown_properties("not a dict", schema) == "not a dict"


def test_non_dict_schema_returns_arguments_unchanged():
    args = {"a": 1, "b": 2}
    assert filter_unknown_properties(args, "not a schema") is args


# ---------------------------------------------------------------------------
# coerce_for_send
# ---------------------------------------------------------------------------


def test_string_field_accepts_non_string_values():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    out = coerce_for_send({"name": 42, "count": "5"}, schema)
    assert out["name"] == "42"
    assert out["count"] == 5


def test_string_field_keeps_string_unchanged():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    out = coerce_for_send({"name": "alice"}, schema)
    assert out["name"] == "alice"


def test_integer_field_keeps_int_and_truncates_float():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
    }
    out = coerce_for_send({"n": 5.0}, schema)
    assert out["n"] == 5
    assert isinstance(out["n"], int)


def test_integer_field_leaves_non_numeric_string_alone():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
    }
    out = coerce_for_send({"n": "abc"}, schema)
    assert out["n"] == "abc"


def test_bool_is_never_coerced_to_int():
    # ``True`` would otherwise match ``isinstance(v, int)``.
    schema = {
        "type": "object",
        "properties": {"flag": {"type": "string"}, "count": {"type": "integer"}},
    }
    out = coerce_for_send({"flag": True, "count": True}, schema)
    assert out["flag"] is True
    assert out["count"] is True


def test_number_field_accepts_int():
    schema = {
        "type": "object",
        "properties": {"score": {"type": "number"}},
    }
    out = coerce_for_send({"score": 5}, schema)
    assert out["score"] == 5


def test_unknown_field_passes_through():
    # The helper only touches fields that are declared in ``properties``;
    # it never inspects extra fields, which the filter step is responsible
    # for removing when the schema is strict.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    out = coerce_for_send({"a": "x", "mystery": 42}, schema)
    assert out == {"a": "x", "mystery": 42}


def test_empty_schema_returns_arguments_unchanged():
    args = {"a": 1, "b": "two"}
    assert coerce_for_send(args, {}) == args
    assert coerce_for_send(args, {"type": "object", "properties": "not a dict"}) == args


# ---------------------------------------------------------------------------
# sanitize_for_send: the convenience wrapper used by MCPToolRegistry
# ---------------------------------------------------------------------------


def test_sanitize_for_send_combines_filter_and_coerce():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    out = sanitize_for_send(
        {"query": 42, "limit": "5", "extra": "drop me"},
        schema,
    )
    assert out == {"query": "42", "limit": 5}


def test_sanitize_for_send_is_idempotent():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "additionalProperties": False,
    }
    once = sanitize_for_send({"a": 1, "b": "2", "x": True}, schema)
    twice = sanitize_for_send(once, schema)
    assert once == twice
