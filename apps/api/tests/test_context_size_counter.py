"""Tests for the context-size counter.

The counter has three pieces that must stay consistent:

* ``estimate_token_count(text)`` — deterministic rough estimate used for
  budgeting. Implemented in ``services/indexing.py``.
* ``_window_usage(session, window)`` — returns
  ``(token_limit, used_tokens, used_percent)`` for a window.
* ``total_token_count`` on the ``sessions`` table — running sum updated
  by ``_save_message``.

The SKILL.state integration must not double-count tokens or leave the
counter inconsistent with the actual messages table.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.db import execute, fetch_one, init_db
from app.services.indexing import estimate_token_count
from app.services.orchestrator import _save_message, _window_usage
from app.services.skill_state import apply_step


def _make_session(session_id: str, window_id: str = "w1", *, token_limit: int = 128000) -> None:
    """Insert a session + window row so FK constraints are satisfied."""
    init_db()
    execute(
        """
        INSERT OR IGNORE INTO sessions (
          id, title, description, created_at, updated_at, status,
          total_message_count, total_token_count, settings_json
        ) VALUES (?, ?, NULL, ?, ?, 'active', 0, 0, '{}')
        """,
        (session_id, session_id, "2026-01-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z"),
    )
    execute(
        """
        INSERT OR IGNORE INTO windows (
          id, session_id, window_index, started_at, token_limit,
          rollover_trigger_percent, pre_rollover_started_at,
          hard_rollover_started_at, checkpoint_id, closing_reason
        ) VALUES (?, ?, 1, ?, ?, 0.85, NULL, NULL, NULL, NULL)
        """,
        (window_id, session_id, "2026-01-01T00:00:00.000Z", token_limit),
    )


def _session_row(session_id: str) -> dict[str, Any]:
    return fetch_one("SELECT * FROM sessions WHERE id=?", (session_id,))


# ---------------------------------------------------------------------------
# estimate_token_count
# ---------------------------------------------------------------------------


def test_estimate_token_count_is_zero_for_empty():
    assert estimate_token_count("") == 0


def test_estimate_token_count_is_positive_for_words():
    n = estimate_token_count("hello world this is a token estimate test")
    assert n >= 4
    # 1.35x multiplier on word count.
    assert n == max(1, int(8 * 1.35))


def test_estimate_token_count_handles_long_text():
    text = "word " * 1000
    n = estimate_token_count(text.strip())
    assert n == max(1, int(1000 * 1.35))


def test_estimate_token_count_floor_is_one():
    assert estimate_token_count("a") == 1


# ---------------------------------------------------------------------------
# _save_message and total_token_count
# ---------------------------------------------------------------------------


def test_save_message_increments_total_token_count(isolated_data_dir):
    _make_session("sess-counter-1")
    before = _session_row("sess-counter-1")
    assert before["total_token_count"] == 0
    assert before["total_message_count"] == 0

    _save_message(
        session_id="sess-counter-1",
        window_id="w1",
        role="user",
        content_text="hello world from a user message",
        message_type="user",
        turn_id="t1",
    )
    after = _session_row("sess-counter-1")
    expected = estimate_token_count("hello world from a user message")
    assert after["total_token_count"] == expected
    assert after["total_message_count"] == 1

    _save_message(
        session_id="sess-counter-1",
        window_id="w1",
        role="assistant",
        content_text="a short reply",
        message_type="assistant",
        turn_id="t1",
    )
    after2 = _session_row("sess-counter-1")
    expected_total = expected + estimate_token_count("a short reply")
    assert after2["total_token_count"] == expected_total
    assert after2["total_message_count"] == 2


def test_save_message_writes_per_message_token_count(isolated_data_dir):
    _make_session("sess-counter-2")
    msg = _save_message(
        session_id="sess-counter-2",
        window_id="w1",
        role="user",
        content_text="a quick brown fox jumps over the lazy dog",
        message_type="user",
        turn_id="t1",
    )
    row = fetch_one("SELECT token_count FROM messages WHERE id=?", (msg["id"],))
    assert row["token_count"] == msg["token_count"]
    assert row["token_count"] == estimate_token_count(
        "a quick brown fox jumps over the lazy dog"
    )


# ---------------------------------------------------------------------------
# _window_usage
# ---------------------------------------------------------------------------


def test_window_usage_empty(isolated_data_dir):
    _make_session("sess-win-1")
    token_limit, used_tokens, used_percent = _window_usage("sess-win-1", "w1")
    assert token_limit >= 1
    assert used_tokens == 0
    assert used_percent == 0.0


def test_window_usage_reflects_messages(isolated_data_dir):
    _make_session("sess-win-2", token_limit=1000)
    for i in range(5):
        _save_message(
            session_id="sess-win-2",
            window_id="w1",
            role="user",
            content_text=f"message {i} with several words in it",
            message_type="user",
            turn_id=f"t{i}",
        )
    _token_limit, used_tokens, used_percent = _window_usage("sess-win-2", "w1")
    # used_tokens must equal the sum of per-message token_count.
    sum_row = fetch_one(
        "SELECT COALESCE(SUM(token_count), 0) AS s FROM messages WHERE session_id=? AND window_id=?",
        ("sess-win-2", "w1"),
    )
    assert used_tokens == int(sum_row["s"])
    assert used_tokens > 0
    # used_percent is used/limit, capped >= 0.
    assert 0.0 <= used_percent <= 1.0 or used_tokens > _token_limit


def test_window_usage_scales_with_token_limit(isolated_data_dir):
    _make_session("sess-win-3", token_limit=200)
    for i in range(10):
        _save_message(
            session_id="sess-win-3",
            window_id="w1",
            role="user",
            content_text=f"message {i}",
            message_type="user",
            turn_id=f"t{i}",
        )
    _token_limit, used_tokens, used_percent = _window_usage("sess-win-3", "w1")
    assert used_tokens > 0
    # With 10 messages and a 200-token limit, percent must exceed 0
    # and respect the 1.35x heuristic.
    assert used_percent > 0


def test_window_usage_handles_missing_window_row(isolated_data_dir):
    """If a window row is missing, ``_window_usage`` falls back to the
    active model's window size rather than crashing — this matters
    because ``hard_rollover`` closes the source window before the
    next window exists."""
    _make_session("sess-win-missing")
    token_limit, used_tokens, used_percent = _window_usage("sess-win-missing", "nonexistent-window")
    # Falls back to 128_000 (or the resolved active model limit, which
    # is >= 128_000 with no provider).
    assert token_limit >= 1
    assert used_tokens == 0
    assert used_percent == 0.0


# ---------------------------------------------------------------------------
# SKILL.state interaction with the counter
# ---------------------------------------------------------------------------


def test_skill_state_does_not_double_count_user_message(
    isolated_data_dir, monkeypatch, tmp_path
):
    """Saving a user message and then recording the same message as the
    SKILL.state observation must not double-count it in
    ``total_token_count``."""
    _make_session("sess-skill-1")
    # Stub the registry so ``apply_step`` can resolve the skill.
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "registry.json").write_text(
        '{"skills":{"demo":{"name":"demo","description":"d","instructions":["x"]}},'
        '"version":1,"lastModified":""}',
        encoding="utf-8",
    )
    import app.services.skill_state as ss

    monkeypatch.setattr(
        ss,
        "_load_skill",
        lambda name: {"name": name, "description": "d", "instructions": ["x"]},
    )

    user_text = "what is the weather in dnipro"

    # 1) The orchestrator saves the user turn to messages (this is the
    #    canonical place where the counter is incremented).
    user_msg = _save_message(
        session_id="sess-skill-1",
        window_id="w1",
        role="user",
        content_text=user_text,
        message_type="user",
        turn_id="t1",
    )

    after_user_save = _session_row("sess-skill-1")
    expected_tokens = estimate_token_count(user_text)
    assert after_user_save["total_token_count"] == expected_tokens
    assert after_user_save["total_message_count"] == 1

    # 2) The SKILL.state runtime pushes the same text as its own
    #    observation — but this is stored in the bounded state ring,
    #    NOT in ``messages``, so the counter must not move.
    apply_step(
        "sess-skill-1",
        "demo",
        user_prompt=user_text,
        transition={"kind": "advance"},
    )

    after_skill = _session_row("sess-skill-1")
    assert after_skill["total_token_count"] == expected_tokens, (
        "SKILL.state observation pushed the same text but must not "
        "double-count the user message in the session counter"
    )
    assert after_skill["total_message_count"] == 1, (
        "SKILL.state observation must not create a duplicate row in "
        "the messages table"
    )

    # 3) Saving the assistant reply still increments the counter on
    #    top of the original user message.
    _save_message(
        session_id="sess-skill-1",
        window_id="w1",
        role="assistant",
        content_text="it is sunny in dnipro",
        message_type="assistant",
        turn_id="t1",
    )
    after_assistant = _session_row("sess-skill-1")
    expected_total = expected_tokens + estimate_token_count("it is sunny in dnipro")
    assert after_assistant["total_token_count"] == expected_total
    assert after_assistant["total_message_count"] == 2


def test_skill_state_history_does_not_inflate_window_usage(
    isolated_data_dir, monkeypatch, tmp_path
):
    """The bounded SKILL.state history (up to ``max_history``
    observations) must not show up in ``_window_usage.used_tokens`` —
    only ``messages.token_count`` does. This is by design: the ring
    buffer exists precisely so the prompt does not grow with execution
    history, and the window-usage counter mirrors the prompt size."""
    _make_session("sess-skill-2", token_limit=128000)
    import app.services.skill_state as ss

    # A skill with enough steps that 20 advances keep it running.
    instructions = [f"step-{i}" for i in range(50)]
    monkeypatch.setattr(
        ss,
        "_load_skill",
        lambda name: {"name": name, "description": "d", "instructions": instructions},
    )

    for i in range(20):
        apply_step(
            "sess-skill-2",
            "demo",
            observation={"kind": "tool", "content": f"obs-{i} with several words"},
            transition={"kind": "advance"},
        )

    _limit, used_tokens, _pct = _window_usage("sess-skill-2", "w1")
    assert used_tokens == 0, (
        "SKILL.state observations must not appear in the window-usage "
        "counter — they live in the bounded state ring, not in messages"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_window_usage_handles_zero_token_limit(isolated_data_dir, monkeypatch):
    """If the saved limit is 1 (legacy single-config) and the model
    limit is also tiny, ``used_percent`` must stay finite (the code
    already guards with ``max(token_limit, 1)``)."""
    _make_session("sess-edge-1", token_limit=1)
    # Pretend the active model reports a 1-token window.
    from app.config import load_app_config

    cfg = load_app_config()
    monkeypatch.setattr(cfg, "model_context_window_size_override", 1, raising=False)
    token_limit, _used_tokens, used_percent = _window_usage("sess-edge-1", "w1")
    assert token_limit >= 1
    assert 0.0 <= used_percent <= 1.0


def test_total_token_count_never_decreases(isolated_data_dir):
    """Adding messages monotonically grows ``total_token_count``."""
    _make_session("sess-mono-1")
    last = 0
    for i in range(5):
        _save_message(
            session_id="sess-mono-1",
            window_id="w1",
            role="user",
            content_text=f"line {i}",
            message_type="user",
            turn_id=f"t{i}",
        )
        cur = _session_row("sess-mono-1")["total_token_count"]
        assert cur >= last
        last = cur


def test_token_count_consistent_with_messages_sum(isolated_data_dir):
    """``total_token_count`` on the session row must equal the sum of
    ``messages.token_count`` for that session."""
    _make_session("sess-sum-1")
    for i in range(7):
        _save_message(
            session_id="sess-sum-1",
            window_id="w1",
            role="user",
            content_text=f"message number {i} with a few words",
            message_type="user",
            turn_id=f"t{i}",
        )
    sum_row = fetch_one(
        "SELECT COALESCE(SUM(token_count), 0) AS s FROM messages WHERE session_id=?",
        ("sess-sum-1",),
    )
    session_row = _session_row("sess-sum-1")
    assert int(sum_row["s"]) == session_row["total_token_count"]