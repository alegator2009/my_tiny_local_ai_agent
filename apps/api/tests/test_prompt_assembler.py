from __future__ import annotations

import uuid

from app.config import load_app_config
from app.db import execute, init_db, utcnow_iso
from app.services.prompt import assemble_prompt
from app.storage import ensure_session_dirs, write_memory_file


def test_prompt_contains_precedence_sections(isolated_data_dir):
    init_db()
    session_id = str(uuid.uuid4())
    now = utcnow_iso()
    window_id = str(uuid.uuid4())

    execute(
        """
        INSERT INTO sessions (id, title, created_at, updated_at, status, settings_json)
        VALUES (?, 's', ?, ?, 'active', '{}')
        """,
        (session_id, now, now),
    )
    execute(
        """
        INSERT INTO windows (id, session_id, window_index, started_at, token_limit, rollover_trigger_percent)
        VALUES (?, ?, 1, ?, 128000, 0.92)
        """,
        (window_id, session_id, now),
    )

    execute(
        """
        INSERT INTO messages (id, session_id, window_id, turn_id, role, timestamp, content_text, token_count, message_type, source, is_pinned)
        VALUES (?, ?, ?, 't1', 'user', ?, 'Always answer short', 3, 'user', 'chat', 1)
        """,
        (str(uuid.uuid4()), session_id, window_id, now),
    )

    ensure_session_dirs(session_id)
    write_memory_file(session_id, "durable_facts.json", [{"subject": "assistant", "predicate": "style", "object": "short"}])
    write_memory_file(session_id, "working_set.json", {"current_objective": "build api"})

    cfg = load_app_config()
    prompt = assemble_prompt(session_id, "Implement endpoint", cfg, None, thinking_mode="medium")
    system_content = prompt[0]["content"]

    assert "Pinned user instructions" in system_content
    assert "Durable facts" in system_content
    assert "Working set" in system_content
    assert prompt[-1]["content"] == "Implement endpoint"


def test_prompt_does_not_replay_internal_system_events_after_initial_system_message(isolated_data_dir):
    init_db()
    session_id = str(uuid.uuid4())
    window_id = str(uuid.uuid4())
    now = utcnow_iso()

    execute(
        """
        INSERT INTO sessions (id, title, created_at, updated_at, status, settings_json)
        VALUES (?, 's', ?, ?, 'active', '{}')
        """,
        (session_id, now, now),
    )
    execute(
        """
        INSERT INTO windows (id, session_id, window_index, started_at, token_limit, rollover_trigger_percent)
        VALUES (?, ?, 1, ?, 128000, 0.92)
        """,
        (window_id, session_id, now),
    )
    for role, text, message_type in [
        ("user", "Hello", "user"),
        ("system", "Provider error: stale internal event", "internal_event"),
        ("assistant", "Hi", "assistant"),
    ]:
        execute(
            """
            INSERT INTO messages (id, session_id, window_id, turn_id, role, timestamp, content_text, token_count, message_type, source, is_pinned)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 'chat', 0)
            """,
            (str(uuid.uuid4()), session_id, window_id, str(uuid.uuid4()), role, now, text, message_type),
        )

    ensure_session_dirs(session_id)
    prompt = assemble_prompt(session_id, "Continue", load_app_config(), None, thinking_mode="medium")

    assert prompt[0]["role"] == "system"
    assert all(message["role"] in {"system", "user", "assistant"} for message in prompt)
    assert all(message["role"] != "system" for message in prompt[1:])
    assert all("stale internal event" not in message["content"] for message in prompt)
    assert prompt[-1] == {"role": "user", "content": "Continue"}
