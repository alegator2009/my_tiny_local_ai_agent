"""Tests for the structural tool-result truncation used by the orchestrator.

The previous implementation only sliced the first ``max_chars`` of the
text, which silently dropped tail content and applied a constant cap
regardless of the window budget. The new utility must:

* preserve both the head and the tail of large text outputs;
* collapse top-level JSON arrays when the tool returned many items;
* collapse long log-shaped output by removing whole middle lines;
* honour per-tool overrides and the absolute upper bound;
* expose a marker that names the tool and reports the dropped size.
"""

from __future__ import annotations

import json

import pytest

from app.services.text_truncation import (
    ABSOLUTE_MAX_CHARS,
    DEFAULT_PER_TOOL_BUDGET,
    DEFAULT_TOOL_BUDGET_RATIO,
    TruncationResult,
    apply_truncation,
    compute_tool_budget,
    estimate_chars_from_tokens,
    truncate_tool_text,
)


# ---------------------------------------------------------------------------
# passthrough
# ---------------------------------------------------------------------------


def test_short_text_passes_through_unchanged():
    text = "hello world"
    result = truncate_tool_text(text, max_chars=200, tool_name="anything")
    assert result.truncated is False
    assert result.mode == "passthrough"
    assert result.text == "hello world"
    assert result.original_chars == len(text)
    assert result.kept_chars == len(text)


def test_empty_text_passes_through():
    result = truncate_tool_text("", max_chars=200, tool_name="x")
    assert result.truncated is False
    assert result.text == ""
    assert result.original_chars == 0


# ---------------------------------------------------------------------------
# head + tail
# ---------------------------------------------------------------------------


def test_plain_text_uses_head_tail():
    # ~5000 chars of plain text. Default budget for ``default`` tool is 3200.
    text = "A" * 2500 + "MIDDLE" + "B" * 2500
    result = truncate_tool_text(text, max_chars=3200, tool_name="default")
    assert result.truncated is True
    assert result.mode == "head_tail"
    # We must have shrunk the text and emitted a marker.
    assert result.kept_chars < result.original_chars
    assert "truncated" in result.text
    # The note (or apply_truncation output) names the tool and original size.
    rendered = apply_truncation(result)
    assert "tool=default" in rendered
    assert str(len(text)) in rendered


def test_head_tail_preserves_both_ends():
    text = "START_BEGIN " + "x" * 4000 + " END_FINISH"
    result = truncate_tool_text(text, max_chars=1000, tool_name="default")
    assert result.truncated is True
    rendered = apply_truncation(result)
    assert rendered.startswith("START_BEGIN")
    assert "END_FINISH" in rendered


# ---------------------------------------------------------------------------
# JSON array collapse
# ---------------------------------------------------------------------------


def test_json_array_is_collapsed_with_dropped_count():
    items = [{"i": i, "title": f"item-{i}"} for i in range(200)]
    text = json.dumps(items)
    result = truncate_tool_text(text, max_chars=2000, tool_name="web_search")
    assert result.truncated is True
    assert result.mode == "json_collapse"
    # Body itself mentions how many items were dropped.
    assert "more items truncated" in result.text
    # After apply_truncation, the note (with tool name and original size) is appended.
    rendered = apply_truncation(result)
    assert "tool=web_search" in rendered
    assert str(len(text)) in rendered


def test_small_json_array_is_not_collapsed():
    items = [{"i": i} for i in range(3)]
    text = json.dumps(items)
    result = truncate_tool_text(text, max_chars=200, tool_name="web_search")
    # 3 items, well under the head_tail threshold, no need to collapse.
    assert result.truncated is False


# ---------------------------------------------------------------------------
# log / line collapse
# ---------------------------------------------------------------------------


def test_log_shaped_text_collapses_middle_lines():
    lines = [f"line {i:04d}: some status info" for i in range(200)]
    text = "\n".join(lines)
    result = truncate_tool_text(text, max_chars=1500, tool_name="run_terminal_command")
    assert result.truncated is True
    assert result.mode == "line_collapse"
    # Some early lines should be kept.
    assert "line 0000" in result.text
    # Some late lines should be kept.
    assert "line 0199" in result.text
    # The middle should be gone.
    assert "line 0100" not in result.text


def test_short_text_with_many_lines_is_not_line_collapsed():
    # Below the log-collapse threshold (40 lines) we should not collapse.
    lines = [f"line {i}" for i in range(20)]
    text = "\n".join(lines)
    result = truncate_tool_text(text, max_chars=200, tool_name="run_terminal_command")
    # 20 lines * 6 chars ~= 120 chars, well under 200. Passthrough.
    assert result.truncated is False


# ---------------------------------------------------------------------------
# budget computation
# ---------------------------------------------------------------------------


def test_compute_tool_budget_uses_per_tool_default_when_no_remaining():
    budget = compute_tool_budget(
        remaining_tokens=0,  # disables window-aware path
        tool_name="web_search",
    )
    assert budget == DEFAULT_PER_TOOL_BUDGET["web_search"]


def test_compute_tool_budget_caps_at_absolute_max():
    budget = compute_tool_budget(
        remaining_tokens=10_000_000,  # huge window budget
        tool_name="web_search",
    )
    assert budget <= ABSOLUTE_MAX_CHARS


def test_compute_tool_budget_uses_override_when_smaller():
    budget = compute_tool_budget(
        remaining_tokens=100_000,  # plenty of room
        tool_name="web_search",
        per_tool_overrides={"web_search": 1500},
    )
    assert budget == 1500


def test_compute_tool_budget_uses_window_when_smaller_than_override():
    budget = compute_tool_budget(
        remaining_tokens=100,  # ~400 chars
        tool_name="web_search",
        per_tool_overrides={"web_search": 10_000},
    )
    # Window side is smaller; it wins.
    assert budget <= 500


def test_estimate_chars_from_tokens_zero():
    assert estimate_chars_from_tokens(0) == 0
    assert estimate_chars_from_tokens(-5) == 0


def test_estimate_chars_from_tokens_positive():
    assert estimate_chars_from_tokens(100) == 400
    assert estimate_chars_from_tokens(250) == 1000


# ---------------------------------------------------------------------------
# apply_truncation
# ---------------------------------------------------------------------------


def test_apply_truncation_appends_note_once():
    res = TruncationResult(
        text="hello",
        truncated=True,
        original_chars=1000,
        kept_chars=5,
        mode="head_tail",
        note="[tool result truncated: 1000 -> 5 chars, tool=x]",
    )
    out = apply_truncation(res)
    assert "hello" in out
    assert out.count("truncated") == 1


def test_apply_truncation_passthrough_returns_plain_text():
    res = TruncationResult(
        text="ok",
        truncated=False,
        original_chars=2,
        kept_chars=2,
        mode="passthrough",
        note="should not appear",
    )
    assert apply_truncation(res) == "ok"


# ---------------------------------------------------------------------------
# Realistic combined scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,expected_default",
    [
        ("web_search", 4000),
        ("run_terminal_command", 6000),
        ("filesystem", 3000),
        ("unknown_tool", 3200),
    ],
)
def test_default_per_tool_budgets(tool_name, expected_default):
    budget = compute_tool_budget(
        remaining_tokens=0,  # disables window side, exposes per-tool default
        tool_name=tool_name,
    )
    assert budget == expected_default


def test_budget_ratio_does_not_break_small_budgets():
    # When the window is nearly full, the budget should still be at least
    # ``MIN_SIDE`` so that the model receives *some* content.
    tiny = compute_tool_budget(
        remaining_tokens=1,
        tool_name="web_search",
        tool_budget_ratio=DEFAULT_TOOL_BUDGET_RATIO,
    )
    assert tiny >= 200
