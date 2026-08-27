from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from ..config import load_app_config
from ..db import execute, fetch_all, fetch_one, utcnow_iso
from ..schemas import SessionCreate, SessionUpdate
from ..storage import delete_session_dir, ensure_session_dirs, write_session_json

DEFAULT_THINKING_MODE = "medium"
DEFAULT_MESSAGE_PREFIX_PROMPT = ""
DEFAULT_HIDE_SYSTEM_MESSAGES = False
DEFAULT_RUN_IN_BACKGROUND = False
DEFAULT_FORCE_SEARCH_NEXT = False
DEFAULT_BYPASS_SEARCH_CACHE_NEXT = False
DEFAULT_CONTEXT_MODE = "full"
_VALID_CONTEXT_MODES = {"full", "skill_state"}


@dataclass
class SessionRecord:
    id: str
    title: str
    description: str | None
    created_at: str
    updated_at: str
    status: str
    workspace_path: str | None
    last_window_id: str | None
    total_message_count: int
    total_token_count: int


def row_to_session(row) -> dict:
    settings_json = row["settings_json"] or "{}"
    try:
        settings = json.loads(settings_json)
    except json.JSONDecodeError:
        settings = {}
    thinking_mode = settings.get("thinking_mode", DEFAULT_THINKING_MODE)
    message_prefix_prompt = settings.get("message_prefix_prompt", DEFAULT_MESSAGE_PREFIX_PROMPT)
    if not isinstance(message_prefix_prompt, str):
        message_prefix_prompt = str(message_prefix_prompt or "")
    provider_id = settings.get("provider_id") or None
    model_id = settings.get("model_id") or None
    hide_system_messages = bool(settings.get("hide_system_messages", DEFAULT_HIDE_SYSTEM_MESSAGES))
    run_in_background = bool(settings.get("run_in_background", DEFAULT_RUN_IN_BACKGROUND))
    force_search_next = bool(settings.get("force_search_next", DEFAULT_FORCE_SEARCH_NEXT))
    bypass_search_cache_next = bool(settings.get("bypass_search_cache_next", DEFAULT_BYPASS_SEARCH_CACHE_NEXT))
    context_mode = settings.get("context_mode", DEFAULT_CONTEXT_MODE)
    if context_mode not in _VALID_CONTEXT_MODES:
        context_mode = DEFAULT_CONTEXT_MODE

    return {
        "id": row["id"],
        "title": row["title"],
        "description": row["description"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"],
        "workspace_path": row["workspace_path"],
        "last_window_id": row["last_window_id"],
        "total_message_count": row["total_message_count"],
        "total_token_count": row["total_token_count"],
        "thinking_mode": thinking_mode,
        "message_prefix_prompt": message_prefix_prompt,
        "provider_id": provider_id,
        "model_id": model_id,
        "hide_system_messages": hide_system_messages,
        "run_in_background": run_in_background,
        "force_search_next": force_search_next,
        "bypass_search_cache_next": bypass_search_cache_next,
        "context_mode": context_mode,
    }


def _resolve_window_context_size(
    *, provider_id: str | None, model_id: str | None
) -> int:
    cfg = load_app_config()
    _, model = cfg.resolve_pair(provider_id, model_id)
    if model is not None:
        return max(1, int(model.context_window_size))
    # Fall back to the active model so we always have a sensible size.
    _, active = cfg.active_pair()
    if active is not None:
        return max(1, int(active.context_window_size))
    return 128000


def _create_window(
    session_id: str,
    window_index: int = 1,
    *,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> str:
    cfg = load_app_config()
    now = utcnow_iso()
    window_id = str(uuid.uuid4())
    token_limit = _resolve_window_context_size(provider_id=provider_id, model_id=model_id)
    execute(
        """
        INSERT INTO windows (
          id, session_id, window_index, started_at, token_limit, rollover_trigger_percent
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            window_id,
            session_id,
            window_index,
            now,
            token_limit,
            cfg.rollover_config.hard_rollover_threshold,
        ),
    )
    execute("UPDATE sessions SET last_window_id=?, updated_at=? WHERE id=?", (window_id, now, session_id))
    return window_id


def create_session(payload: SessionCreate) -> dict:
    cfg = load_app_config()
    now = utcnow_iso()
    sid = str(uuid.uuid4())
    # If the user didn't pick a model, inherit the active one so the
    # provider/model picker in the chat UI doesn't have to.
    active_provider, active_model = cfg.active_pair()
    provider_id = payload.provider_id or (active_provider.id if active_provider else None)
    model_id = payload.model_id or (active_model.id if active_model else None)
    settings_json = {
        "thinking_mode": payload.thinking_mode,
        "message_prefix_prompt": payload.message_prefix_prompt,
        "provider_id": provider_id,
        "model_id": model_id,
        "hide_system_messages": payload.hide_system_messages,
        "run_in_background": payload.run_in_background,
        "force_search_next": payload.force_search_next,
        "bypass_search_cache_next": payload.bypass_search_cache_next,
        "context_mode": payload.context_mode or DEFAULT_CONTEXT_MODE,
    }
    execute(
        """
        INSERT INTO sessions (
          id, title, description, created_at, updated_at, status, workspace_path, settings_json
        ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
        """,
        (sid, payload.title, payload.description, now, now, payload.workspace_path, json.dumps(settings_json)),
    )
    wid = _create_window(
        sid,
        window_index=1,
        provider_id=provider_id,
        model_id=model_id,
    )
    session_payload = {
        "id": sid,
        "title": payload.title,
        "description": payload.description,
        "created_at": now,
        "updated_at": now,
        "status": "active",
        "workspace_path": payload.workspace_path,
        "last_window_id": wid,
        "thinking_mode": payload.thinking_mode,
        "message_prefix_prompt": payload.message_prefix_prompt,
        "provider_id": provider_id,
        "model_id": model_id,
        "hide_system_messages": payload.hide_system_messages,
        "run_in_background": payload.run_in_background,
        "force_search_next": payload.force_search_next,
        "bypass_search_cache_next": payload.bypass_search_cache_next,
        "context_mode": payload.context_mode or DEFAULT_CONTEXT_MODE,
    }
    ensure_session_dirs(sid)
    write_session_json(sid, session_payload)
    return get_session(sid)


def list_sessions() -> list[dict]:
    rows = fetch_all("SELECT * FROM sessions WHERE status != 'deleted' ORDER BY updated_at DESC")
    return [row_to_session(r) for r in rows]


def get_session(session_id: str) -> dict:
    row = fetch_one("SELECT * FROM sessions WHERE id=?", (session_id,))
    if row is None:
        raise KeyError("session_not_found")
    return row_to_session(row)


def update_session(session_id: str, payload: SessionUpdate) -> dict:
    current_row = fetch_one("SELECT * FROM sessions WHERE id=?", (session_id,))
    if current_row is None:
        raise KeyError("session_not_found")
    current = row_to_session(current_row)
    current_settings = json.loads(current_row["settings_json"] or "{}")
    updated = {
        "title": payload.title if payload.title is not None else current["title"],
        "description": payload.description if payload.description is not None else current["description"],
        "workspace_path": payload.workspace_path if payload.workspace_path is not None else current["workspace_path"],
        "status": payload.status if payload.status is not None else current["status"],
        "thinking_mode": payload.thinking_mode if payload.thinking_mode is not None else current["thinking_mode"],
        "message_prefix_prompt": (
            payload.message_prefix_prompt
            if payload.message_prefix_prompt is not None
            else current.get("message_prefix_prompt", DEFAULT_MESSAGE_PREFIX_PROMPT)
        ),
        "provider_id": (
            payload.provider_id
            if payload.provider_id is not None
            else current.get("provider_id")
        ),
        "model_id": (
            payload.model_id
            if payload.model_id is not None
            else current.get("model_id")
        ),
        "hide_system_messages": (
            payload.hide_system_messages
            if payload.hide_system_messages is not None
            else current.get("hide_system_messages", DEFAULT_HIDE_SYSTEM_MESSAGES)
        ),
        "run_in_background": (
            payload.run_in_background
            if payload.run_in_background is not None
            else current.get("run_in_background", DEFAULT_RUN_IN_BACKGROUND)
        ),
        "force_search_next": (
            payload.force_search_next
            if payload.force_search_next is not None
            else current.get("force_search_next", DEFAULT_FORCE_SEARCH_NEXT)
        ),
        "bypass_search_cache_next": (
            payload.bypass_search_cache_next
            if payload.bypass_search_cache_next is not None
            else current.get("bypass_search_cache_next", DEFAULT_BYPASS_SEARCH_CACHE_NEXT)
        ),
        "context_mode": (
            payload.context_mode
            if payload.context_mode is not None
            else current.get("context_mode", DEFAULT_CONTEXT_MODE)
        ),
    }
    if updated["context_mode"] not in _VALID_CONTEXT_MODES:
        updated["context_mode"] = DEFAULT_CONTEXT_MODE
    current_settings["thinking_mode"] = updated["thinking_mode"]
    current_settings["message_prefix_prompt"] = updated["message_prefix_prompt"]
    current_settings["provider_id"] = updated["provider_id"]
    current_settings["model_id"] = updated["model_id"]
    current_settings["hide_system_messages"] = updated["hide_system_messages"]
    current_settings["run_in_background"] = updated["run_in_background"]
    current_settings["force_search_next"] = updated["force_search_next"]
    current_settings["bypass_search_cache_next"] = updated["bypass_search_cache_next"]
    current_settings["context_mode"] = updated["context_mode"]
    now = utcnow_iso()
    execute(
        """
        UPDATE sessions
        SET title=?, description=?, workspace_path=?, status=?, settings_json=?, updated_at=?
        WHERE id=?
        """,
        (
            updated["title"],
            updated["description"],
            updated["workspace_path"],
            updated["status"],
            json.dumps(current_settings),
            now,
            session_id,
        ),
    )
    result = get_session(session_id)
    write_session_json(session_id, result)
    return result


def archive_session(session_id: str) -> dict:
    return update_session(session_id, SessionUpdate(status="archived"))


def delete_session(session_id: str) -> None:
    execute("DELETE FROM sessions WHERE id=?", (session_id,))
    delete_session_dir(session_id)


def get_last_window(session_id: str):
    row = fetch_one(
        "SELECT * FROM windows WHERE session_id=? ORDER BY window_index DESC LIMIT 1",
        (session_id,),
    )
    return row


def create_next_window(session_id: str, closing_reason: str, checkpoint_id: str | None) -> dict:
    last = get_last_window(session_id)
    if last is None:
        raise KeyError("window_not_found")
    now = utcnow_iso()
    execute(
        "UPDATE windows SET closed_at=?, closing_reason=?, checkpoint_id=? WHERE id=?",
        (now, closing_reason, checkpoint_id, last["id"]),
    )
    # Reuse the session's preferred provider/model so the new window
    # has the same context size as the user expects.
    provider_id, model_id = _session_provider_model_ids(session_id)
    new_id = _create_window(
        session_id,
        window_index=last["window_index"] + 1,
        provider_id=provider_id,
        model_id=model_id,
    )
    row = fetch_one("SELECT * FROM windows WHERE id=?", (new_id,))
    return dict(row) if row else {"id": new_id}


def _session_provider_model_ids(session_id: str) -> tuple[str | None, str | None]:
    row = fetch_one("SELECT settings_json FROM sessions WHERE id=?", (session_id,))
    if row is None:
        return None, None
    try:
        settings = json.loads(row["settings_json"] or "{}")
    except json.JSONDecodeError:
        settings = {}
    return settings.get("provider_id"), settings.get("model_id")
