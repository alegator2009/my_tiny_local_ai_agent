"""Tests for the run-time guards added after reviewing the last
session:

* ``SearchBudgetTracker`` caps consecutive empty results and rotates
  the search engine.
* ``validate_terminal_command`` blocks hallucinated utilities like
  ``search`` / ``google`` / ``scrape`` and only allows known CLIs.
* ``plan_skill_delegation`` reads the persisted SKILL.state
  history to pick the right ``prefer_engine`` and returns
  ``exhausted=True`` once ``max_attempts`` empty results have been
  seen.
* ``update_working_set`` reads the latest run's task + progress so
  a background run's state stays visible.
"""
from __future__ import annotations

import json

import pytest

from app.services.search_budget import (
    FALLBACK_ENGINES,
    SearchBudgetTracker,
    exhausted_payload,
    looks_like_empty_search_result,
)
from app.services.tool_validation import validate_terminal_command


# ---------------------------------------------------------------------------
# SearchBudgetTracker
# ---------------------------------------------------------------------------


def test_empty_result_detection():
    assert looks_like_empty_search_result("Search engine: None; 0 result requested/0 obtained")
    assert looks_like_empty_search_result("0 results")
    assert looks_like_empty_search_result("")
    assert looks_like_empty_search_result(None)
    assert not looks_like_empty_search_result("Got 5 results about Dnipro weather")
    assert not looks_like_empty_search_result("Top hit: foo bar")


def test_tracker_first_call_uses_preferred_engine():
    t = SearchBudgetTracker()
    plan = t.plan_next_call("run-1", {"query": "foo"})
    assert plan["prefer_engine"] == "brave"
    assert "engine" in plan
    assert "__search_budget_exhausted" not in plan


def test_tracker_rotates_engine_after_empty():
    t = SearchBudgetTracker()
    t.plan_next_call("run-1", {"query": "a"})
    t.record("run-1", "brave", "0 results")
    plan = t.plan_next_call("run-1", {"query": "a"})
    # First empty should rotate to fallback[0] = duckduckgo
    assert plan["prefer_engine"] == "duckduckgo"
    t.record("run-1", "duckduckgo", "0 results")
    plan = t.plan_next_call("run-1", {"query": "a"})
    assert plan["prefer_engine"] == "bing"


def test_tracker_exhausts_after_max_empty():
    t = SearchBudgetTracker(max_empty=2, fallback_engines=("brave", "ddg"))
    t.plan_next_call("run-1", {"query": "a"})
    t.record("run-1", "brave", "0 results")
    assert not t.snapshot("run-1")["exhausted"]
    t.record("run-1", "ddg", "0 results")
    assert t.snapshot("run-1")["exhausted"]
    plan = t.plan_next_call("run-1", {"query": "a"})
    assert plan["__search_budget_exhausted"] is True


def test_tracker_resets_on_non_empty():
    t = SearchBudgetTracker()
    t.plan_next_call("run-1", {"query": "a"})
    t.record("run-1", "brave", "0 results")
    t.record("run-1", "ddg", "got 5 results about Dnipro")
    summary = t.snapshot("run-1")
    # The non-empty result breaks the streak.
    assert summary["consecutive_empty"] == 0
    assert not summary["exhausted"]


def test_tracker_per_run_isolation():
    t = SearchBudgetTracker()
    t.plan_next_call("run-a", {"query": "x"})
    t.record("run-a", "brave", "0 results")
    t.record("run-a", "brave", "0 results")
    t.record("run-a", "brave", "0 results")
    # run-b has its own budget.
    plan = t.plan_next_call("run-b", {"query": "y"})
    assert "engine" in plan
    assert plan["prefer_engine"] == "brave"


def test_exhausted_payload_shape():
    p = exhausted_payload("weather in dnipro")
    assert p["ok"] is False
    assert p["error_kind"] == "search_exhausted"
    assert "Do NOT call" in p["error"]
    assert p["query"] == "weather in dnipro"


# ---------------------------------------------------------------------------
# validate_terminal_command
# ---------------------------------------------------------------------------


def test_terminal_whitelist_allows_known_clis():
    assert validate_terminal_command("ls -la") is None
    assert validate_terminal_command("grep -r foo .") is None
    assert validate_terminal_command("curl https://example.com") is None
    assert validate_terminal_command("npm test") is None
    assert validate_terminal_command("python -m pytest") is None


def test_terminal_whitelist_strips_path_and_extension():
    assert validate_terminal_command("/usr/bin/grep foo bar") is None
    assert validate_terminal_command("C:\\Tools\\grep.exe foo bar") is None
    assert validate_terminal_command("FOO=bar python -m pytest") is None


def test_terminal_whitelist_rejects_hallucinated_utilities():
    msg = validate_terminal_command("search 'free LLM API'")
    assert msg is not None
    assert "search" in msg.lower()
    assert "web-search" in msg.lower() or "mcp" in msg.lower()

    msg = validate_terminal_command("google 'weather'")
    assert msg is not None
    assert "google" in msg.lower() or "web-search" in msg.lower()

    msg = validate_terminal_command("scrape https://example.com")
    assert msg is not None

    msg = validate_terminal_command("llm 'hi'")
    assert msg is not None


def test_terminal_whitelist_rejects_unknown():
    msg = validate_terminal_command("foobar --baz")
    assert msg is not None
    assert "allow-list" in msg.lower() or "not in" in msg.lower()


def test_terminal_rejects_empty():
    assert validate_terminal_command("") is not None
    assert validate_terminal_command("   ") is not None


# ---------------------------------------------------------------------------
# plan_skill_delegation
# ---------------------------------------------------------------------------


def _stub_registry(tmp_path, monkeypatch, registry_dict):
    """Point the skill registry at a tmp registry.json so tests can
    control the on-disk spec without touching the real one."""
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry_dict), encoding="utf-8")
    # The skill loader resolves the registry via env var first.
    monkeypatch.setenv("SKILLS_REGISTRY_PATH", str(path))
    return path


def test_plan_skill_delegation_picks_first_engine(tmp_path, monkeypatch, isolated_data_dir):
    from app.db import init_db

    init_db()
    _stub_registry(
        tmp_path,
        monkeypatch,
        {
            "skills": {
                "web-search-loop": {
                    "name": "web-search-loop",
                    "description": "Test skill",
                    "instructions": [],
                    "delegates_to": {
                        "tool": "mcp__native_web_search__full_web_search",
                        "args_from": ["query"],
                        "default_args": {"query": "", "prefer_engine": "brave"},
                        "max_attempts": 3,
                    },
                }
            },
            "version": 2,
        },
    )

    from app.services.memory import plan_skill_delegation, start_or_resume_skill

    sid = "test-session-1"
    start_or_resume_skill(sid, "web-search-loop")
    plan = plan_skill_delegation(sid, "web-search-loop", user_args={"query": "free LLM API"})
    assert plan["tool"] == "mcp__native_web_search__full_web_search"
    assert plan["exhausted"] is False
    # The first call has no empty observations yet, so the default
    # ``brave`` is preserved.
    assert plan["args"]["prefer_engine"] == "brave"
    assert plan["args"]["query"] == "free LLM API"


def test_plan_skill_delegation_rotates_engine_after_empty(tmp_path, monkeypatch, isolated_data_dir):
    from app.db import init_db

    init_db()
    _stub_registry(
        tmp_path,
        monkeypatch,
        {
            "skills": {
                "web-search-loop": {
                    "name": "web-search-loop",
                    "description": "Test skill",
                    "instructions": [],
                    "delegates_to": {
                        "tool": "mcp__native_web_search__full_web_search",
                        "args_from": ["query"],
                        "default_args": {"query": "", "prefer_engine": "brave"},
                        "max_attempts": 3,
                    },
                }
            },
            "version": 2,
        },
    )

    from app.services.memory import (
        plan_skill_delegation,
        record_skill_tool_observation,
        start_or_resume_skill,
    )

    sid = "test-session-2"
    start_or_resume_skill(sid, "web-search-loop")
    # First attempt is empty.
    record_skill_tool_observation(
        sid,
        "web-search-loop",
        tool="mcp__native_web_search__full_web_search",
        result_text="0 results, engine: None",
    )
    plan = plan_skill_delegation(sid, "web-search-loop", user_args={"query": "foo"})
    # One empty observation → next engine = fallback[0] = duckduckgo
    assert plan["args"]["prefer_engine"] == FALLBACK_ENGINES[0]


def test_plan_skill_delegation_exhausts_at_max_attempts(tmp_path, monkeypatch, isolated_data_dir):
    from app.db import init_db

    init_db()
    _stub_registry(
        tmp_path,
        monkeypatch,
        {
            "skills": {
                "web-search-loop": {
                    "name": "web-search-loop",
                    "description": "Test skill",
                    "instructions": [],
                    "delegates_to": {
                        "tool": "mcp__native_web_search__full_web_search",
                        "args_from": ["query"],
                        "default_args": {"query": "", "prefer_engine": "brave"},
                        "max_attempts": 2,
                    },
                }
            },
            "version": 2,
        },
    )

    from app.services.memory import (
        plan_skill_delegation,
        record_skill_tool_observation,
        start_or_resume_skill,
    )

    sid = "test-session-3"
    start_or_resume_skill(sid, "web-search-loop")
    for _ in range(2):
        record_skill_tool_observation(
            sid,
            "web-search-loop",
            tool="mcp__native_web_search__full_web_search",
            result_text="0 results, engine: None",
        )
    plan = plan_skill_delegation(sid, "web-search-loop", user_args={"query": "foo"})
    assert plan["exhausted"] is True
    assert plan["attempts"] == 2
    assert plan["max_attempts"] == 2


def test_plan_skill_delegation_no_delegates_to(tmp_path, monkeypatch, isolated_data_dir):
    from app.db import init_db

    init_db()
    _stub_registry(
        tmp_path,
        monkeypatch,
        {
            "skills": {
                "no-delegates": {
                    "name": "no-delegates",
                    "description": "Plain skill",
                    "instructions": [],
                }
            },
            "version": 2,
        },
    )
    from app.services.memory import plan_skill_delegation

    plan = plan_skill_delegation("test-session-4", "no-delegates")
    assert plan["tool"] is None
    assert plan["exhausted"] is False


# ---------------------------------------------------------------------------
# update_working_set: surface the run's task + progress
# ---------------------------------------------------------------------------


def test_update_working_set_surfaces_run_task(isolated_data_dir):
    """When the latest user message has source='run', the working
    set's ``current_objective`` should be the run's ``task_text``,
    not the raw user prompt that queued it. This keeps the model's
    attention on the real objective during a long pipeline."""
    from app.db import execute, init_db
    from app.services.memory import update_working_set
    from app.services.sessions import create_session, create_next_window

    init_db()
    s = create_session(__import__("app").schemas.SessionCreate(title="ws"))
    sid = s["id"]
    win = create_next_window(sid, "init", None)
    wid = win["id"]
    run_id = "run-x"
    execute(
        """
        INSERT INTO runs (id, session_id, task_text, status, created_at, updated_at, progress_json)
        VALUES (?, ?, ?, 'running', '2026-09-01T00:00:00+00:00', '2026-09-01T00:00:00+00:00', ?)
        """,
        (run_id, sid, "Find a free LLM API", '{"phase": "executing", "current_step": 2, "total_steps": 4, "workflow_type": "research"}'),
    )
    execute(
        """
        INSERT INTO messages (
            id, session_id, window_id, turn_id, role, content_text, message_type, timestamp, source
        ) VALUES (?, ?, ?, ?, 'user', ?, 'user', '2026-09-01T00:00:00+00:00', 'run')
        """,
        ("msg-1", sid, wid, "turn-1", "Run a research task"),
    )
    execute(
        """
        INSERT INTO messages (
            id, session_id, window_id, turn_id, role, content_text, message_type, timestamp, source
        ) VALUES (?, ?, ?, ?, 'assistant', ?, 'assistant', '2026-09-01T00:00:01+00:00', 'run')
        """,
        ("msg-2", sid, wid, "turn-1", "Step 2/4 done"),
    )
    ws = update_working_set(sid)
    assert ws["current_objective"] == "Find a free LLM API"
    assert "step 2/4" in ws["current_subtask"].lower()
    assert "step 3/4" in ws["next_suggested_step"].lower()
    assert "executing" in ws["current_subtask"]


def test_update_working_set_falls_back_to_user_text_without_run(isolated_data_dir):
    """When there's no run, the working set should mirror the
    regular ``role='user'`` message — same behaviour as before."""
    from app.db import execute, init_db
    from app.services.memory import update_working_set
    from app.services.sessions import create_session, create_next_window

    init_db()
    s = create_session(__import__("app").schemas.SessionCreate(title="ws2"))
    sid = s["id"]
    win = create_next_window(sid, "init", None)
    wid = win["id"]
    execute(
        """
        INSERT INTO messages (
            id, session_id, window_id, turn_id, role, content_text, message_type, timestamp, source
        ) VALUES (?, ?, ?, ?, 'user', ?, 'user', '2026-09-01T00:00:00+00:00', 'chat')
        """,
        ("msg-1", sid, wid, "turn-1", "Hi there"),
    )
    ws = update_working_set(sid)
    assert ws["current_objective"] == "Hi there"
    assert ws["current_subtask"] == "Respond to latest user intent with context continuity"