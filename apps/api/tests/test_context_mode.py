"""Tests for the context-mode selector (AppConfig + SessionUpdate +
orchestrator wiring). The selector chooses between two prompt
strategies:

* ``"full"`` (default, backward compatible): the orchestrator replays
  the recent chat history, durable facts, working set, recall pack,
  and checkpoint summary in the system prompt.

* ``"skill_state"``: SKILL.state runtime — the model only ever sees
  the (spec, state, observation) bundle from a registered skill plus
  the current user turn. When the user's prompt matches a registered
  skill the orchestrator activates it and rebuilds the prompt.
"""
from __future__ import annotations

import pytest

from app.config import AppConfig
from app.schemas import SessionCreate, SessionUpdate


def test_default_context_mode_is_full():
    cfg = AppConfig()
    assert cfg.context_mode == "full"


def test_context_mode_accepts_full_and_skill_state():
    assert AppConfig(context_mode="full").context_mode == "full"
    assert AppConfig(context_mode="skill_state").context_mode == "skill_state"


def test_context_mode_normalises_legacy_variants():
    """Older configs may have used ``skill-state`` or ``skillstate``;
    we coerce anything that looks like the SKILL.state selector."""
    assert AppConfig(context_mode="skill-state").context_mode == "skill_state"
    assert AppConfig(context_mode="skillstate").context_mode == "skill_state"
    # Unknown values fall back to "full" so the runtime never crashes.
    assert AppConfig(context_mode="garbage").context_mode == "full"
    assert AppConfig(context_mode="").context_mode == "full"


def test_session_create_accepts_context_mode():
    payload = SessionCreate(title="t", context_mode="skill_state")
    assert payload.context_mode == "skill_state"


def test_session_update_accepts_context_mode():
    payload = SessionUpdate(context_mode="skill_state")
    assert payload.context_mode == "skill_state"


def test_session_round_trip_persists_context_mode(isolated_data_dir):
    """The full create → update → fetch cycle must round-trip the
    ``context_mode`` field on the session row."""
    from app.db import execute, init_db
    from app.services.sessions import create_session, get_session, update_session

    init_db()
    created = create_session(SessionCreate(title="cm-test", context_mode="skill_state"))
    sid = created["id"]
    assert get_session(sid)["context_mode"] == "skill_state"

    updated = update_session(sid, SessionUpdate(context_mode="full"))
    assert updated["context_mode"] == "full"

    # The persisted row must reflect the new value (this is what the
    # next chat turn's orchestrator call will read).
    from app.db import fetch_one
    row = fetch_one("SELECT settings_json FROM sessions WHERE id=?", (sid,))
    import json
    settings = json.loads(row["settings_json"])
    assert settings["context_mode"] == "full"


def test_session_round_trip_normalises_invalid_context_mode(isolated_data_dir):
    """If somehow an invalid value sneaks into the JSON, ``get_session``
    must downgrade it to ``"full"`` rather than crashing."""
    import json
    from app.db import execute, init_db
    from app.services.sessions import create_session

    init_db()
    created = create_session(SessionCreate(title="cm-bad"))
    sid = created["id"]

    # Inject an invalid value directly.
    execute(
        "UPDATE sessions SET settings_json=? WHERE id=?",
        (json.dumps({"context_mode": "what is this"}), sid),
    )
    from app.services.sessions import get_session
    assert get_session(sid)["context_mode"] == "full"


def test_settings_endpoint_get_put(isolated_data_dir):
    """The /api/settings/context-mode endpoints accept and return the
    selector value exactly as stored."""
    from app.services.skill_state import _load_skill  # noqa: F401  -- importable smoke test
    # The endpoint is a thin wrapper around load/save_app_config — we
    # exercise the underlying helpers here since routing through the
    # FastAPI test client is already covered elsewhere.
    from app.config import load_app_config, save_app_config

    cfg = load_app_config()
    cfg.context_mode = "skill_state"
    save_app_config(cfg)
    assert load_app_config().context_mode == "skill_state"

    cfg.context_mode = "full"
    save_app_config(cfg)
    assert load_app_config().context_mode == "full"


def test_settings_context_mode_normalises_legacy_via_settings_endpoint(
    isolated_data_dir,
):
    """Saving an unknown value through the settings layer must normalise
    to ``"full"`` so the API never persists a value the runtime cannot
    interpret."""
    from app.config import load_app_config, save_app_config

    cfg = AppConfig(context_mode="bogus")
    save_app_config(cfg)
    assert load_app_config().context_mode == "full"