"""Tests for the SKILL.state server-side runtime."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.skill_state import (
    ALLOWED_KINDS,
    ALLOWED_STATUSES,
    DEFAULT_MAX_HISTORY,
    Observation,
    SkillState,
    TransitionError,
    apply_step,
    build_prompt_bundle,
    list_states,
    load_state,
    reset_state,
    start_or_resume,
    validate_transition,
    _push_observation,
)


def _stub_registry(tmp_path: Path, monkeypatch) -> Path:
    """Create a minimal skills/registry.json and point ``_load_skill``
    at it so the runtime can resolve skill definitions."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    registry_path = skills_dir / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "skills": {
                    "demo": {
                        "name": "demo",
                        "description": "Demo skill for tests",
                        "instructions": ["Step A", "Step B", "Step C"],
                        "whenToUse": "Whenever",
                        "examples": [{"prompt": "p", "action": "a"}],
                    },
                    "other": {
                        "name": "other",
                        "description": "Other skill",
                        "instructions": ["Only step"],
                    },
                },
                "version": 1,
                "lastModified": "2026-01-01T00:00:00.000Z",
            }
        ),
        encoding="utf-8",
    )

    # The runtime resolves the registry path by walking up from
    # ``apps/api/app/services/skill_state.py``. For tests we monkeypatch
    # the helper directly.
    import app.services.skill_state as ss

    monkeypatch.setattr(ss, "_load_skill", lambda name: json.loads(registry_path.read_text(encoding="utf-8"))["skills"].get(name))
    monkeypatch.setattr(ss, "list_skills_in_registry", lambda: list(json.loads(registry_path.read_text(encoding="utf-8"))["skills"].keys()))
    return registry_path


def _seed_session(session_id: str, window_id: str = "w1") -> None:
    """Insert minimal session + window rows so FK constraints are
    satisfied when tests insert ``messages`` rows manually."""
    from app.db import execute, init_db

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
        ) VALUES (?, ?, 1, ?, 128000, 0.85, NULL, NULL, NULL, NULL)
        """,
        (window_id, session_id, "2026-01-01T00:00:00.000Z"),
    )


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def test_create_initial_state_pulls_total_steps(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-1", "demo")
    assert state.skill_name == "demo"
    assert state.status == "running"
    assert state.current_step == 1
    assert state.total_steps == 3
    assert state.max_history == DEFAULT_MAX_HISTORY
    assert state.history == []
    assert state.last_observation is None


def test_user_prompt_seeds_first_observation(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-2", "demo", user_prompt="Hello")
    assert len(state.history) == 1
    assert state.history[0].kind == "user"
    assert state.history[0].content == "Hello"
    assert state.last_observation == state.history[0]


def test_validate_transition_advance(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-3", "demo")
    next_state = validate_transition(state, {"kind": "advance"})
    assert next_state.current_step == 2
    assert next_state.iterations == 1
    assert next_state.status == "running"
    # Original state untouched (referential transparency).
    assert state.current_step == 1


def test_validate_transition_auto_completes(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-4", "demo")
    state = validate_transition(state, {"kind": "advance"})
    state = validate_transition(state, {"kind": "advance"})
    state = validate_transition(state, {"kind": "advance"})
    assert state.status == "completed"
    assert state.current_step == 3


def test_set_variable_merges(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-5", "demo")
    state = validate_transition(state, {"kind": "set-variable", "set": {"city": "Dnipro", "temp": 22}})
    assert state.variables == {"city": "Dnipro", "temp": 22}
    state = validate_transition(state, {"kind": "set-variable", "set": {"temp": 24}})
    assert state.variables == {"city": "Dnipro", "temp": 24}


def test_fail_and_retry(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-6", "demo")
    state = validate_transition(state, {"kind": "fail", "error": "boom"})
    assert state.status == "failed"
    assert state.error == "boom"

    with pytest.raises(TransitionError) as excinfo:
        validate_transition(state, {"kind": "advance"})
    assert excinfo.value.code == "invalid_advance"

    state = validate_transition(state, {"kind": "retry"})
    assert state.status == "running"
    assert state.error is None


def test_unknown_kind_raises(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-7", "demo")
    with pytest.raises(TransitionError) as excinfo:
        validate_transition(state, {"kind": "bogus"})
    assert excinfo.value.code == "unknown_kind"


def test_set_variable_requires_set_object(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-8", "demo")
    with pytest.raises(TransitionError) as excinfo:
        validate_transition(state, {"kind": "set-variable"})
    assert excinfo.value.code == "invalid_payload"


def test_push_observation_bounds_history(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = start_or_resume("sess-9", "demo")
    for i in range(20):
        state = _push_observation(state, {"kind": "tool", "content": f"obs-{i}"})
    assert state.max_history == DEFAULT_MAX_HISTORY
    assert len(state.history) == DEFAULT_MAX_HISTORY
    assert state.history[0].content == f"obs-{20 - DEFAULT_MAX_HISTORY}"
    assert state.last_observation.content == "obs-19"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_apply_step_persists_state(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    state = apply_step(
        "sess-10",
        "demo",
        transition={"kind": "advance"},
        observation={"kind": "tool", "content": "tool output"},
    )
    assert state.current_step == 2
    assert len(state.history) == 1
    assert state.history[0].kind == "tool"

    # Reload from disk and confirm.
    reloaded = load_state("sess-10", "demo")
    assert reloaded is not None
    assert reloaded.current_step == 2
    assert len(reloaded.history) == 1


def test_reset_state_wipes(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    apply_step("sess-11", "demo", transition={"kind": "advance"})
    reset_state("sess-11", "demo")
    state = load_state("sess-11", "demo")
    assert state is not None
    assert state.current_step == 1
    assert state.status == "running"


def test_list_states(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    apply_step("sess-12", "demo")
    apply_step("sess-12", "other")
    states = list_states("sess-12")
    names = {s["skillName"] for s in states}
    # ``list_states`` only returns states that have been persisted, so
    # both must show up here.
    assert "demo" in names
    assert "other" in names


def test_build_prompt_bundle(monkeypatch, tmp_path, isolated_data_dir):
    _stub_registry(tmp_path, monkeypatch)
    apply_step(
        "sess-13",
        "demo",
        user_prompt="Hello",
        observation={"kind": "tool", "content": "result"},
    )
    bundle = build_prompt_bundle("sess-13", "demo")
    assert bundle["spec"]["name"] == "demo"
    assert bundle["spec"]["instructions"] == ["Step A", "Step B", "Step C"]
    assert bundle["state"]["status"] == "running"
    assert bundle["observation"] is not None
    assert bundle["observation"]["kind"] in {"user", "tool"}
    assert len(bundle["history"]) >= 1


# ---------------------------------------------------------------------------
# Prompt assembly (state-aware)
# ---------------------------------------------------------------------------


def test_assemble_prompt_drops_history_when_skill_active(monkeypatch, tmp_path, isolated_data_dir):
    from app.services import prompt as prompt_mod
    from app.config import AppConfig

    _stub_registry(tmp_path, monkeypatch)

    # Seed a few messages so the legacy code path would replay them.
    from app.storage import ensure_session_dirs
    from app.db import execute

    _seed_session("sess-prompt")
    ensure_session_dirs("sess-prompt")
    execute(
        """
        INSERT INTO messages (
          id, session_id, window_id, turn_id, role, timestamp,
          content_text, content_json, token_count, message_type, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("m1", "sess-prompt", "w1", "t1", "user", "2026-01-01T00:00:00.000Z",
         "historic user msg", "{}", 1, "user", "chat"),
    )
    apply_step("sess-prompt", "demo", user_prompt="current msg")

    cfg = AppConfig()
    messages = prompt_mod.assemble_prompt(
        "sess-prompt",
        "current msg",
        cfg,
        recall_pack=None,
        thinking_mode="medium",
        active_skill="demo",
    )
    # The system prompt must contain the SKILL.state bundle ...
    joined = "\n".join(m["content"] for m in messages if m["role"] == "system")
    assert "SKILL.state active skill bundle" in joined
    # ... but the historic user msg must NOT appear in any role.
    for msg in messages:
        assert "historic user msg" not in msg["content"]


def test_assemble_prompt_keeps_legacy_history_when_no_skill(monkeypatch, tmp_path, isolated_data_dir):
    from app.services import prompt as prompt_mod
    from app.config import AppConfig
    from app.db import execute
    from app.storage import ensure_session_dirs

    _seed_session("sess-legacy")
    ensure_session_dirs("sess-legacy")
    execute(
        """
        INSERT INTO messages (
          id, session_id, window_id, turn_id, role, timestamp,
          content_text, content_json, token_count, message_type, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("m2", "sess-legacy", "w1", "t1", "user", "2026-01-01T00:00:00.000Z",
         "old user msg", "{}", 1, "user", "chat"),
    )
    cfg = AppConfig()
    messages = prompt_mod.assemble_prompt(
        "sess-legacy",
        "current msg",
        cfg,
        recall_pack=None,
        thinking_mode="medium",
    )
    # Legacy path: the historic user msg must be replayed.
    joined = "\n".join(m["content"] for m in messages)
    assert "old user msg" in joined
    assert "SKILL.state active skill bundle" not in joined


# ---------------------------------------------------------------------------
# Memory bridge
# ---------------------------------------------------------------------------


def test_memory_bridge_round_trip(monkeypatch, tmp_path, isolated_data_dir):
    from app.services.memory import (
        apply_skill_transition,
        load_skill_state,
        reset_skill_state,
        start_or_resume_skill,
    )

    _stub_registry(tmp_path, monkeypatch)

    start_or_resume_skill("sess-mem", "demo", user_prompt="hi")
    state = apply_skill_transition(
        "sess-mem", "demo",
        transition={"kind": "set-variable", "set": {"city": "Lviv"}},
    )
    assert state["variables"]["city"] == "Lviv"
    assert load_skill_state("sess-mem", "demo")["variables"]["city"] == "Lviv"

    reset_skill_state("sess-mem", "demo")
    assert load_skill_state("sess-mem", "demo")["currentStep"] == 1


def test_transition_error_propagates(monkeypatch, tmp_path, isolated_data_dir):
    from app.services.memory import apply_skill_transition
    from app.services.skill_state import TransitionError

    _stub_registry(tmp_path, monkeypatch)
    apply_skill_transition("sess-err", "demo", transition={"kind": "fail", "error": "boom"})
    with pytest.raises(TransitionError):
        apply_skill_transition("sess-err", "demo", transition={"kind": "advance"})