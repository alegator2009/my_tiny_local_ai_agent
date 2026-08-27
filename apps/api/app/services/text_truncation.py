"""Structural tool-result truncation utilities.

These helpers are used by the orchestrator to keep tool outputs inside the
current context window without losing the parts the model is most likely to
need. The previous implementation only sliced the first ``max_chars`` of the
text, which silently dropped tail content (where most "the answer" actually
lives for logs, search results, and command output) and applied the same
constant limit regardless of the available budget.

Three improvements over the old behaviour:

1. **Head + tail preservation.** When the text exceeds the per-tool budget we
   keep the first ``head`` characters, the last ``tail`` characters, and insert
   a clear marker describing how much was removed. The model can still see
   both the request context and the final outcome.
2. **Line-aware log truncation.** Output that looks like a log (many short
   lines, no JSON structure) is collapsed by removing whole middle lines
   instead of breaking a line in half.
3. **Budget-aware sizing.** The orchestrator can call
   :func:`compute_tool_budget` to derive a per-tool character budget from the
   remaining tokens in the active window. This prevents wasting context on a
   fixed 3200-char cap when the window is large, and prevents blowing the
   window when many tool results fire in a single turn.

The functions in this module are intentionally pure (no DB or network
access) so they are trivial to unit-test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


# Default per-tool character budgets. These are the *outer* cap; the actual
# size used in a given turn is also clamped to the remaining window budget.
DEFAULT_PER_TOOL_BUDGET: dict[str, int] = {
    "web_search": 4000,
    "web-search": 4000,
    "run_terminal_command": 6000,
    "terminal": 6000,
    "write_file": 2000,
    "read_file": 3000,
    "filesystem": 3000,
    "default": 3200,
}

# Hard upper bound for a single tool result regardless of window budget.
# Prevents pathological cases (e.g. a 50MB log) from saturating memory.
ABSOLUTE_MAX_CHARS = 24_000

# Minimum head/tail sizes we are willing to produce. If the budget is below
# 2 * MIN_SIDE we just truncate head-only.
MIN_SIDE = 200

# Reserve this fraction of the remaining window for tool results (the rest
# goes to system prompt, recent messages, and the model's own output).
DEFAULT_TOOL_BUDGET_RATIO = 0.30


@dataclass(frozen=True)
class TruncationResult:
    """Outcome of :func:`truncate_tool_text`."""

    text: str
    truncated: bool
    original_chars: int
    kept_chars: int
    mode: str  # "head" | "head_tail" | "line_collapse" | "json_collapse" | "passthrough"
    note: str = ""  # human-readable note appended to the truncated text

    @property
    def dropped_chars(self) -> int:
        return self.original_chars - self.kept_chars


def estimate_chars_from_tokens(tokens: int) -> int:
    """Translate a token budget to a character budget.

    The model's tokenizer is not loaded here, so we use the same rough
    word-based heuristic that ``indexing.estimate_token_count`` uses:
    roughly 1 token ~ 4 characters of English text. This is intentionally
    conservative (a bit too small rather than too large) so the prompt
    assembly never exceeds the window.
    """
    if tokens <= 0:
        return 0
    return max(0, int(tokens * 4))


def compute_tool_budget(
    *,
    remaining_tokens: int,
    tool_name: str,
    tool_budget_ratio: float = DEFAULT_TOOL_BUDGET_RATIO,
    per_tool_overrides: dict[str, int] | None = None,
) -> int:
    """Return the maximum number of characters a single tool result may use.

    The smaller of:
      - ``remaining_tokens * tool_budget_ratio * 4`` (window-aware)
      - per-tool override (e.g. user-configured cap for ``web_search``)
      - :data:`ABSOLUTE_MAX_CHARS`

    When ``remaining_tokens`` is zero or negative the window-aware side is
    disabled and the per-tool default is returned directly. ``MIN_SIDE`` is
    only used as a floor so the model always receives *some* content.
    """
    per_tool = (per_tool_overrides or {}).get(tool_name) or DEFAULT_PER_TOOL_BUDGET.get(
        tool_name, DEFAULT_PER_TOOL_BUDGET["default"]
    )
    if remaining_tokens <= 0:
        return max(MIN_SIDE, min(per_tool, ABSOLUTE_MAX_CHARS))
    window_budget = estimate_chars_from_tokens(
        max(0, int(remaining_tokens * tool_budget_ratio))
    )
    return max(MIN_SIDE, min(window_budget, per_tool, ABSOLUTE_MAX_CHARS))


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_LINE_HEAVY = re.compile(r"\n")
_JSON_OBJECT = re.compile(r"^\s*[\{\[]", re.MULTILINE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _looks_like_log(text: str) -> bool:
    """Return True if the text has the shape of a log/dump.

    Heuristic: many lines, average line length < 200 chars, no JSON object at
    the start. False positives are harmless (we just collapse the middle).
    """
    if not text:
        return False
    lines = text.splitlines()
    if len(lines) < 40:
        return False
    avg_len = sum(len(l) for l in lines) / len(lines)
    if avg_len > 200:
        return False
    if _JSON_OBJECT.match(text):
        return False
    return True


def _collapse_middle_lines(text: str, budget: int) -> str | None:
    """For log-shaped text, drop whole middle lines until under ``budget``.

    Returns ``None`` if the text cannot be reduced enough.
    """
    if not _looks_like_log(text):
        return None
    head_lines = max(5, int(budget * 0.6 / 80))
    tail_lines = max(5, int(budget * 0.4 / 80))
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return None
    kept = lines[:head_lines] + lines[-tail_lines:]
    return "\n".join(kept)


def _collapse_json_array(text: str, budget: int) -> str | None:
    """For JSON arrays, keep the first few items and report the rest as dropped.

    Returns ``None`` if the text is not a top-level JSON array.
    """
    stripped = text.lstrip()
    if not stripped.startswith("["):
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if not isinstance(parsed, list) or len(parsed) < 6:
        return None
    # Greedily keep items until we would exceed the budget.
    kept: list[Any] = []
    used = 2  # for "[]"
    for item in parsed:
        encoded = json.dumps(item, ensure_ascii=False)
        cost = len(encoded) + 2  # comma + space
        if used + cost > budget - 80 and kept:
            break
        kept.append(item)
        used += cost
    dropped = len(parsed) - len(kept)
    body = json.dumps(kept, ensure_ascii=False)
    note = f"\n[... {dropped} more items truncated ...]"
    return body + note


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def truncate_tool_text(
    text: str,
    *,
    max_chars: int,
    tool_name: str = "default",
) -> TruncationResult:
    """Truncate a tool result to fit ``max_chars`` while preserving useful ends.

    The result always includes a clearly delimited marker describing how much
    of the original text was removed and which strategy was used. This lets
    the model reason about completeness (and lets the user inspect transcripts
    to see whether truncation may have lost the answer).
    """
    value = (text or "").strip()
    original = len(value)
    if original == 0:
        return TruncationResult(text="", truncated=False, original_chars=0, kept_chars=0, mode="passthrough")
    if original <= max_chars:
        return TruncationResult(
            text=value,
            truncated=False,
            original_chars=original,
            kept_chars=original,
            mode="passthrough",
        )

    budget = max(MIN_SIDE, max_chars)

    # 1) Try JSON-array collapse first (very common for search/list tools).
    collapsed = _collapse_json_array(value, budget)
    if collapsed is not None and len(collapsed) <= budget + 100:
        return TruncationResult(
            text=collapsed,
            truncated=True,
            original_chars=original,
            kept_chars=len(collapsed),
            mode="json_collapse",
            note=f"[tool result truncated: {original} -> {len(collapsed)} chars, tool={tool_name}]",
        )

    # 2) Try line-aware collapse (logs / dumps).
    collapsed = _collapse_middle_lines(value, budget)
    if collapsed is not None and len(collapsed) <= budget + 100:
        return TruncationResult(
            text=collapsed,
            truncated=True,
            original_chars=original,
            kept_chars=len(collapsed),
            mode="line_collapse",
            note=f"[tool result truncated: {original} -> {len(collapsed)} chars, tool={tool_name}]",
        )

    # 3) Fall back to head + tail.
    head_budget = max(MIN_SIDE, int(budget * 0.6))
    tail_budget = max(MIN_SIDE, budget - head_budget - 80)  # 80 for marker
    head = value[:head_budget].rstrip()
    tail = value[-tail_budget:].lstrip() if tail_budget > 0 else ""
    dropped = original - len(head) - len(tail)
    if tail:
        body = f"{head}\n\n[... {dropped} chars of {original} truncated ...]\n\n{tail}"
    else:
        body = f"{head}\n\n[... {dropped} chars of {original} truncated ...]"
    note = f"[tool result truncated: {original} -> {len(body)} chars, tool={tool_name}, mode=head_tail]"
    # Final safety clamp in case marker pushes us over.
    if len(body) > max_chars + 200:
        body = body[: max_chars + 100].rstrip() + "\n[truncated]"
    return TruncationResult(
        text=body,
        truncated=True,
        original_chars=original,
        kept_chars=len(body),
        mode="head_tail",
        note=note,
    )


def apply_truncation(result: TruncationResult) -> str:
    """Return the truncated text with the note appended on a new line.

    The note is appended even when the result was not truncated, so the model
    always sees a uniform "tool result" envelope that downstream parsers can
    recognise. We still return the bare text for the ``passthrough`` case to
    avoid noise on the common short-result path.
    """
    if not result.truncated:
        return result.text
    if result.note and result.note not in result.text:
        return f"{result.text}\n{result.note}"
    return result.text
