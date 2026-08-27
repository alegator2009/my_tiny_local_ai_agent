"""Tests for :mod:`app.services.schema_sanitizer`.

MCP servers and 3rd-party tool providers ship JSON Schemas in many shapes.
The sanitiser must:

* always return an ``object`` root with a ``properties`` dict;
* drop compound keywords our validator does not implement (``oneOf``,
  ``anyOf``, ``allOf``, ``$ref``, ``$defs``, ``definitions``);
* cap long ``enum`` lists to a configurable maximum;
* cap ``description`` length to keep prompts small;
* never mutate the input.
"""

from __future__ import annotations

import pytest

from app.services.schema_sanitizer import (
    DEFAULT_MAX_DESCRIPTION,
    DEFAULT_MAX_ENUM_SIZE,
    sanitize_schema,
    schema_fingerprint,
)


# ---------------------------------------------------------------------------
# Root shape
# ---------------------------------------------------------------------------


def test_non_dict_input_collapses_to_object_root():
    out = sanitize_schema(None)
    assert out == {"type": "object", "properties": {}}

    out = sanitize_schema("not a schema")
    assert out["type"] == "object"
    assert out["properties"] == {}


def test_missing_type_is_forced_to_object():
    out = sanitize_schema({"properties": {"x": {"type": "string"}}})
    assert out["type"] == "object"
    assert "x" in out["properties"]


def test_non_object_root_is_coerced_to_object():
    # Some servers send ``{"type": "function", ...}`` thinking the function
    # tag is part of the schema. The sanitiser should normalise to object.
    out = sanitize_schema({"type": "function", "properties": {}})
    assert out["type"] == "object"


def test_properties_must_be_dict():
    out = sanitize_schema({"type": "object", "properties": ["bogus"]})
    assert out["properties"] == {}


def test_required_must_be_list_of_strings():
    out = sanitize_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": "a",  # wrong type
        }
    )
    assert out["required"] == []


# ---------------------------------------------------------------------------
# Stripping unsupported keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    ["oneOf", "anyOf", "allOf", "$ref", "$defs", "definitions"],
)
def test_unsupported_compound_keywords_are_stripped(kw):
    out = sanitize_schema(
        {
            "type": "object",
            "properties": {"x": {"type": "string", kw: [{"type": "integer"}]}},
        }
    )
    assert kw not in out["properties"]["x"]


def test_nested_unsupported_keywords_are_stripped():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {
                        "oneOf": [{"type": "string"}, {"type": "integer"}],
                    }
                },
            }
        },
    }
    out = sanitize_schema(schema)
    assert "oneOf" not in out["properties"]["outer"]["properties"]["inner"]


# ---------------------------------------------------------------------------
# Enum truncation
# ---------------------------------------------------------------------------


def test_short_enum_is_kept_intact():
    schema = {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
    }
    out = sanitize_schema(schema)
    assert out["properties"]["color"]["enum"] == ["red", "green", "blue"]


def test_long_enum_is_capped():
    big = [{"type": "string", "enum": [f"v{i}" for i in range(200)]}]
    out = sanitize_schema(big[0])
    enum = out["properties"]  # not the same shape; rework
    # Use a real top-level properties shape:
    out = sanitize_schema(
        {
            "type": "object",
            "properties": {"tag": {"type": "string", "enum": [f"v{i}" for i in range(200)]}},
        }
    )
    tag_enum = out["properties"]["tag"]["enum"]
    assert len(tag_enum) == DEFAULT_MAX_ENUM_SIZE + 1
    assert tag_enum[-1].startswith("... ")


def test_custom_max_enum_size_is_honoured():
    big = [f"v{i}" for i in range(50)]
    out = sanitize_schema(
        {
            "type": "object",
            "properties": {"tag": {"type": "string", "enum": big}},
        },
        max_enum_size=5,
    )
    tag_enum = out["properties"]["tag"]["enum"]
    assert len(tag_enum) == 6
    assert tag_enum[-1] == "... 45 more"


# ---------------------------------------------------------------------------
# Description truncation
# ---------------------------------------------------------------------------


def test_long_description_is_truncated():
    long = "x" * 2000
    out = sanitize_schema(
        {
            "type": "object",
            "properties": {"q": {"type": "string", "description": long}},
        }
    )
    desc = out["properties"]["q"]["description"]
    assert len(desc) <= DEFAULT_MAX_DESCRIPTION
    assert desc.endswith("...")


def test_short_description_is_kept_verbatim():
    out = sanitize_schema(
        {
            "type": "object",
            "properties": {"q": {"type": "string", "description": "short"}},
        }
    )
    assert out["properties"]["q"]["description"] == "short"


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_input_is_not_mutated():
    schema = {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["r", "g", "b"]},
            "nested": {"oneOf": [{"type": "string"}]},
        },
    }
    snapshot = {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["r", "g", "b"]},
            "nested": {"oneOf": [{"type": "string"}]},
        },
    }
    sanitize_schema(schema)
    assert schema == snapshot


# ---------------------------------------------------------------------------
# Validation-friendly output
# ---------------------------------------------------------------------------


def test_sanitised_schema_is_accepted_by_validator():
    # Smoke test: the output of the sanitiser should validate cleanly with
    # our tool validator. The point of sanitising is to keep both happy.
    from app.services.tool_validation import validate_tool_args

    raw = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["query"],
    }
    schema = {
        "type": "function",
        "function": {"name": "search", "parameters": sanitize_schema(raw)},
    }
    errors = validate_tool_args("search", {"query": "hi", "limit": 2}, schema)
    assert errors == []
    errors = validate_tool_args("search", {"limit": 99}, schema)
    assert any(e.path == "query" for e in errors)
    errors = validate_tool_args("search", {"query": "hi", "limit": 99}, schema)
    assert any(e.path == "limit" and "maximum" in e.message for e in errors)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_schema_changes():
    a = schema_fingerprint({"type": "object", "properties": {"x": {"type": "string"}}})
    b = schema_fingerprint({"type": "object", "properties": {"x": {"type": "integer"}}})
    assert a != b


def test_fingerprint_is_stable_for_same_schema():
    a = schema_fingerprint({"type": "object", "properties": {"x": {"type": "string"}}})
    b = schema_fingerprint({"type": "object", "properties": {"x": {"type": "string"}}})
    assert a == b


def test_fingerprint_handles_non_json_values():
    # If json.dumps falls back to repr, we should still get a stable string.
    a = schema_fingerprint({"x": {1, 2, 3}})  # sets are not JSON-serialisable
    b = schema_fingerprint({"x": {1, 2, 3}})
    assert a == b
