"""Tests for JSON-Schema validation of tool-call arguments.

The orchestrator hands the model-emitted ``tool_calls`` arguments to a JSON
Schema validator before the executor runs. These tests pin the behaviour:

* missing required properties are reported;
* type mismatches are reported with the actual vs expected type;
* enum violations are reported with a truncated enum;
* string ``minLength`` / ``maxLength`` and numeric ``minimum`` / ``maximum``
  bounds are enforced;
* additional properties are rejected when ``additionalProperties=false``;
* nested objects and arrays are validated recursively;
* an unknown tool name (no schema) does not block the call;
* a non-dict argument is reported as a single ``expected object`` error.
"""

from __future__ import annotations

import pytest

from app.services.tool_validation import (
    ValidationError,
    find_tool_schema,
    format_validation_errors,
    validate_and_format,
    validate_tool_args,
)


# ---------------------------------------------------------------------------
# Fixtures: representative tool schemas
# ---------------------------------------------------------------------------


WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "recency_days": {"type": "integer", "minimum": 0, "maximum": 365},
                "source": {"type": "string", "enum": ["news", "web", "scholar"]},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


TERMINAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_terminal_command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            "required": ["command"],
        },
    },
}


WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {
                        "overwrite": {"type": "boolean"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["path", "content"],
        },
    },
}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_args_return_no_errors():
    errors = validate_tool_args(
        "web_search",
        {"query": "python typing", "limit": 5, "source": "news"},
        WEB_SEARCH_SCHEMA,
    )
    assert errors == []


def test_unknown_tool_with_no_schema_is_accepted():
    # No schema => no validation, no errors. The caller decides what to do.
    errors = validate_tool_args("dynamic_tool", {"anything": 123}, None)
    assert errors == []


# ---------------------------------------------------------------------------
# Required / type / additionalProperties
# ---------------------------------------------------------------------------


def test_missing_required_property_is_reported():
    errors = validate_tool_args("web_search", {"limit": 5}, WEB_SEARCH_SCHEMA)
    assert any(e.path == "query" and "required" in e.message for e in errors)


def test_type_mismatch_is_reported_with_actual_type():
    errors = validate_tool_args(
        "web_search",
        {"query": 42, "limit": "five"},
        WEB_SEARCH_SCHEMA,
    )
    paths = {(e.path, "type" in e.message or "expected" in e.message) for e in errors}
    assert ("query", True) in paths
    assert ("limit", True) in paths


def test_unknown_top_level_property_is_reported():
    errors = validate_tool_args(
        "web_search",
        {"query": "x", "qurey": "y"},  # typo
        WEB_SEARCH_SCHEMA,
    )
    assert any(e.path == "qurey" and "unknown" in e.message for e in errors)


def test_non_dict_args_yield_single_error():
    errors = validate_tool_args("web_search", "not-a-dict", WEB_SEARCH_SCHEMA)
    assert len(errors) == 1
    assert "expected object" in errors[0].message


# ---------------------------------------------------------------------------
# Enum / string / numeric bounds
# ---------------------------------------------------------------------------


def test_enum_violation_is_reported():
    errors = validate_tool_args(
        "web_search",
        {"query": "x", "source": "telegram"},
        WEB_SEARCH_SCHEMA,
    )
    assert any(e.path == "source" and "enum" in e.message for e in errors)


def test_string_min_length_is_reported():
    errors = validate_tool_args("web_search", {"query": ""}, WEB_SEARCH_SCHEMA)
    assert any(e.path == "query" and "minLength" in e.message for e in errors)


def test_numeric_minimum_is_reported():
    errors = validate_tool_args(
        "run_terminal_command",
        {"command": "ls", "timeout_sec": 0},
        TERMINAL_SCHEMA,
    )
    assert any(e.path == "timeout_sec" and "minimum" in e.message for e in errors)


def test_numeric_maximum_is_reported():
    errors = validate_tool_args(
        "run_terminal_command",
        {"command": "ls", "timeout_sec": 999},
        TERMINAL_SCHEMA,
    )
    assert any(e.path == "timeout_sec" and "maximum" in e.message for e in errors)


# ---------------------------------------------------------------------------
# Nested objects / arrays
# ---------------------------------------------------------------------------


def test_nested_object_validation():
    errors = validate_tool_args(
        "write_file",
        {
            "path": "a.txt",
            "content": "hi",
            "options": {"overwrite": "yes"},  # wrong type
        },
        WRITE_FILE_SCHEMA,
    )
    assert any(e.path == "options.overwrite" for e in errors)


def test_array_items_are_validated_recursively():
    errors = validate_tool_args(
        "write_file",
        {
            "path": "a.txt",
            "content": "hi",
            "options": {"tags": [1, 2, "ok"]},  # ints in a string array
        },
        WRITE_FILE_SCHEMA,
    )
    paths = [e.path for e in errors]
    assert "options.tags[0]" in paths
    assert "options.tags[1]" in paths


# ---------------------------------------------------------------------------
# Schema lookup
# ---------------------------------------------------------------------------


def test_find_tool_schema_exact_match():
    schema = find_tool_schema([WEB_SEARCH_SCHEMA, TERMINAL_SCHEMA], "web_search")
    assert schema is WEB_SEARCH_SCHEMA


def test_find_tool_schema_fuzzy_match_for_mcp_name():
    schema = find_tool_schema(
        [WEB_SEARCH_SCHEMA], "mcp__native_web_search__web_search"
    )
    assert schema is WEB_SEARCH_SCHEMA


def test_find_tool_schema_returns_none_when_missing():
    assert find_tool_schema([WEB_SEARCH_SCHEMA], "unknown") is None
    assert find_tool_schema(None, "anything") is None
    assert find_tool_schema([], "anything") is None


# ---------------------------------------------------------------------------
# Error rendering
# ---------------------------------------------------------------------------


def test_format_validation_errors_includes_tool_name_and_count():
    text = format_validation_errors(
        "web_search",
        [ValidationError("query", "required property is missing")],
        received_args={"limit": 5},
    )
    assert "web_search" in text
    assert "1 error" in text
    assert "query" in text
    assert "limit" in text  # from received_args echo


def test_format_validation_errors_truncates_long_arg_dumps():
    huge = {"x": "y" * 1000}
    text = format_validation_errors(
        "web_search",
        [ValidationError("x", "boom")],
        received_args=huge,
    )
    # Body should be much shorter than 1k chars to keep the prompt small.
    assert len(text) < 600
    assert "..." in text


# ---------------------------------------------------------------------------
# One-shot helper
# ---------------------------------------------------------------------------


def test_validate_and_format_returns_true_for_known_good_args():
    ok, msg, errs = validate_and_format(
        "web_search",
        {"query": "hello"},
        [WEB_SEARCH_SCHEMA],
    )
    assert ok is True
    assert msg == ""
    assert errs == []


def test_validate_and_format_returns_false_with_message_for_bad_args():
    ok, msg, errs = validate_and_format(
        "web_search",
        {"limit": 5},  # missing query
        [WEB_SEARCH_SCHEMA],
    )
    assert ok is False
    assert "web_search" in msg
    assert "Please fix" in msg
    assert any(e.path == "query" for e in errs)


def test_validate_and_format_passes_through_unknown_tools():
    ok, msg, errs = validate_and_format("dynamic_tool", {"foo": 1}, [WEB_SEARCH_SCHEMA])
    assert ok is True
    assert msg == ""
    assert errs == []
