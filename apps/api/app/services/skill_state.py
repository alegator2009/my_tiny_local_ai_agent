"""skill_state.py — SKILL.state runtime (server side).

Implements the runtime contract from arXiv:2608.26263. A skill execution
sees three inputs only:

    spec         — the immutable skill definition (loaded from the
                   registry on disk).
    state        — the current structured execution state, stored in a
                   SQLite row + on-disk JSON file under
                   ``./data/sessions/<id>/skill_states/<skill>.json``.
    observation  — the latest observation (user input, tool result,
                   auto-search grounding, …).

Intermediate reasoning produced by the model is **discarded** the moment
a validated state update is produced. The chat history is no longer
appended to on every turn — only observations are pushed onto a bounded
ring buffer inside the state.

This module is the Python mirror of ``skills/state.js`` in the Node side
of the project; both share the same field names and transition kinds.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable

from ..db import execute, fetch_one, utcnow_iso
from ..storage import write_memory_file, read_memory_file


SCHEMA_VERSION = 1
ALLOWED_STATUSES = {"running", "completed", "failed"}
ALLOWED_KINDS = {"advance", "set-variable", "complete", "fail", "retry"}
DEFAULT_MAX_HISTORY = 6


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TransitionError(ValueError):
    """Raised when a proposed transition cannot be validated against the
    current state. The HTTP layer surfaces these as 400 responses."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    kind: str
    content: str
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: utcnow_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "content": self.content,
            "meta": dict(self.meta),
            "timestamp": self.timestamp,
        }


@dataclass
class SkillState:
    skill_name: str
    status: str = "running"
    current_step: int = 1
    total_steps: int = 0
    variables: dict[str, Any] = field(default_factory=dict)
    history: list[Observation] = field(default_factory=list)
    last_observation: Observation | None = None
    pending_transition: dict[str, Any] | None = None
    error: str | None = None
    max_history: int = DEFAULT_MAX_HISTORY
    iterations: int = 0
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: utcnow_iso())
    updated_at: str = field(default_factory=lambda: utcnow_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "skillName": self.skill_name,
            "schemaVersion": self.schema_version,
            "status": self.status,
            "currentStep": self.current_step,
            "totalSteps": self.total_steps,
            "variables": dict(self.variables),
            "history": [o.to_dict() for o in self.history],
            "lastObservation": self.last_observation.to_dict() if self.last_observation else None,
            "pendingTransition": self.pending_transition,
            "error": self.error,
            "maxHistory": self.max_history,
            "iterations": self.iterations,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillState":
        if not isinstance(payload, dict):
            raise TransitionError("invalid_state", "state payload must be an object")
        history_raw = payload.get("history") or []
        history: list[Observation] = []
        for item in history_raw:
            if not isinstance(item, dict):
                continue
            history.append(
                Observation(
                    kind=str(item.get("kind") or "note"),
                    content=str(item.get("content") or ""),
                    source=item.get("source"),
                    meta=dict(item.get("meta") or {}),
                    timestamp=str(item.get("timestamp") or utcnow_iso()),
                )
            )
        last_raw = payload.get("lastObservation")
        last_observation = None
        if isinstance(last_raw, dict):
            last_observation = Observation(
                kind=str(last_raw.get("kind") or "note"),
                content=str(last_raw.get("content") or ""),
                source=last_raw.get("source"),
                meta=dict(last_raw.get("meta") or {}),
                timestamp=str(last_raw.get("timestamp") or utcnow_iso()),
            )
        return cls(
            skill_name=str(payload.get("skillName") or ""),
            status=str(payload.get("status") or "running"),
            current_step=int(payload.get("currentStep") or 1),
            total_steps=int(payload.get("totalSteps") or 0),
            variables=dict(payload.get("variables") or {}),
            history=history,
            last_observation=last_observation,
            pending_transition=payload.get("pendingTransition"),
            error=payload.get("error"),
            max_history=int(payload.get("maxHistory") or DEFAULT_MAX_HISTORY),
            iterations=int(payload.get("iterations") or 0),
            schema_version=int(payload.get("schemaVersion") or SCHEMA_VERSION),
            created_at=str(payload.get("createdAt") or utcnow_iso()),
            updated_at=str(payload.get("updatedAt") or utcnow_iso()),
        )


# ---------------------------------------------------------------------------
# Skill registry bridge
# ---------------------------------------------------------------------------


def _skills_registry_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "..",
        "skills",
        "registry.json",
    )


def _load_skill(skill_name: str) -> dict[str, Any] | None:
    """Load a skill definition from the on-disk registry. Kept
    deliberately decoupled from the Node ``registry.js`` so the FastAPI
    process never has to shell out — but the JSON shape on disk is
    shared so both sides stay in sync.

    The registry file is searched in a small set of well-known
    locations:

    1. ``$SKILLS_REGISTRY_PATH`` env override (used in tests).
    2. ``/app/skills/registry.json`` (the Docker layout — the
       ``skills/`` directory is bind-mounted into ``/app/skills`` by
       ``docker-compose.yml``).
    3. ``<repo-root>/skills/registry.json`` computed by walking up from
       this file's location, for local dev (no Docker) and for the
       tests that run from the host checkout.
    """
    override = os.environ.get("SKILLS_REGISTRY_PATH")
    candidates: list[str] = []
    if override:
        candidates.append(override)
    candidates.extend([
        "/app/skills/registry.json",
        "/skills/registry.json",
    ])
    here = os.path.dirname(os.path.abspath(__file__))
    # ``apps/api/app/services/skill_state.py`` → go up to ``apps/api``,
    # then up to ``apps``, and finally up to the repo root, where
    # ``skills/`` lives.
    for levels in (2, 3, 4):
        joined = os.path.normpath(os.path.join(here, *([".."] * levels), "skills", "registry.json"))
        candidates.append(joined)

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return (data.get("skills") or {}).get(skill_name)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Reducer (pure)
# ---------------------------------------------------------------------------


def _push_observation(state: SkillState, observation: dict[str, Any] | Observation | None) -> SkillState:
    """Return a *new* state with the observation appended. The original
    state is left untouched so the reducer stays referentially
    transparent — important for checkpointing."""
    if observation is None:
        return state
    if isinstance(observation, Observation):
        obs = observation
    elif isinstance(observation, dict):
        obs = Observation(
            kind=str(observation.get("kind") or "note"),
            content=str(observation.get("content") or ""),
            source=observation.get("source"),
            meta=dict(observation.get("meta") or {}),
        )
    else:
        return state
    history = list(state.history)
    history.append(obs)
    while len(history) > state.max_history:
        history.pop(0)
    return SkillState(
        skill_name=state.skill_name,
        status=state.status,
        current_step=state.current_step,
        total_steps=state.total_steps,
        variables=dict(state.variables),
        history=history,
        last_observation=obs,
        pending_transition=state.pending_transition,
        error=state.error,
        max_history=state.max_history,
        iterations=state.iterations,
        schema_version=state.schema_version,
        created_at=state.created_at,
        updated_at=utcnow_iso(),
    )


def validate_transition(state: SkillState, transition: dict[str, Any] | None) -> SkillState:
    """Apply a transition to a state and return a NEW state. Mirrors
    ``state.js:validateTransition`` exactly."""
    transition = transition if isinstance(transition, dict) else {}
    kind = transition.get("kind") or "advance"
    if kind not in ALLOWED_KINDS:
        raise TransitionError("unknown_kind", f"unknown transition kind '{kind}'")
    now = utcnow_iso()
    next_state = SkillState(
        skill_name=state.skill_name,
        status=state.status,
        current_step=state.current_step,
        total_steps=state.total_steps,
        variables=dict(state.variables),
        history=list(state.history),
        last_observation=state.last_observation,
        pending_transition={**transition, "validatedAt": now},
        error=state.error,
        max_history=state.max_history,
        iterations=state.iterations,
        schema_version=state.schema_version,
        created_at=state.created_at,
        updated_at=now,
    )
    next_state.iterations = state.iterations + 1

    if kind == "advance":
        if state.status != "running":
            raise TransitionError(
                "invalid_advance", f"cannot advance from status '{state.status}'"
            )
        if state.current_step >= state.total_steps:
            next_state.status = "completed"
            next_state.current_step = state.total_steps
        else:
            next_state.current_step = state.current_step + 1
    elif kind == "set-variable":
        set_payload = transition.get("set")
        if not isinstance(set_payload, dict):
            raise TransitionError("invalid_payload", "set-variable requires 'set' object")
        for k, v in set_payload.items():
            next_state.variables[k] = v
    elif kind == "complete":
        next_state.status = "completed"
        next_state.error = None
    elif kind == "fail":
        next_state.status = "failed"
        next_state.error = str(transition.get("error") or "unspecified")
    elif kind == "retry":
        if state.status != "failed":
            raise TransitionError("invalid_retry", "retry requires failed state")
        next_state.status = "running"
        next_state.error = None

    if next_state.status not in ALLOWED_STATUSES:
        raise TransitionError("invalid_status", f"status '{next_state.status}' not allowed")
    return next_state


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _state_file(session_id: str, skill_name: str) -> str:
    # ``read_memory_file`` / ``write_memory_file`` operate on
    # ``memory/<filename>.json`` under the session dir. Flatten the
    # name so we never create subdirectories that the bootstrap code
    # in ``storage.ensure_session_dirs`` does not pre-create.
    safe = skill_name.replace("/", "_").replace("\\", "_")
    return f"skill_state_{safe}.json"


def load_state(session_id: str, skill_name: str) -> SkillState | None:
    payload = read_memory_file(session_id, _state_file(session_id, skill_name))
    if not payload:
        return None
    try:
        return SkillState.from_dict(payload)
    except TransitionError:
        return None


def save_state(session_id: str, state: SkillState) -> None:
    write_memory_file(session_id, _state_file(session_id, state.skill_name), state.to_dict())


def reset_state(session_id: str, skill_name: str) -> SkillState:
    skill = _load_skill(skill_name)
    total = len((skill or {}).get("instructions") or [])
    state = SkillState(
        skill_name=skill_name,
        status="running",
        current_step=1,
        total_steps=total,
    )
    save_state(session_id, state)
    return state


# ---------------------------------------------------------------------------
# Prompt bundle
# ---------------------------------------------------------------------------


def build_prompt_bundle(session_id: str, skill_name: str, state: SkillState | None = None) -> dict[str, Any]:
    """Build the (spec, state, observation) bundle the model sees.

    The conversation history is **not** included — that is the whole
    point of SKILL.state. The bounded ``history`` ring inside the
    state is the only past the model observes.

    In addition, this bundle now surfaces the session's
    ``durable_facts`` as a top-level ``known_entities`` field. The
    post-mortem of the last session showed that the model can answer
    a follow-up "give me the answer with those services" only if the
    previous services are visible to it; otherwise the SKILL.state
    runtime forgets everything between turns and the model has to
    re-ask.  We feed the facts the foreground chat path
    (see :func:`memory.record_turn_entities`) and the background
    run path (see :func:`memory.record_durable_facts_from_run`) have
    collected so the model can re-use them without growing the
    prompt with the entire conversation history.
    """
    skill = _load_skill(skill_name) or {}
    if state is None:
        state = load_state(session_id, skill_name) or reset_state(session_id, skill_name)

    spec = {
        "name": skill.get("name", skill_name),
        "description": skill.get("description", ""),
        "whenToUse": skill.get("whenToUse", ""),
        "instructions": list(skill.get("instructions") or []),
        "examples": list(skill.get("examples") or []),
    }

    compact_state = {
        "skillName": state.skill_name,
        "status": state.status,
        "currentStep": state.current_step,
        "totalSteps": state.total_steps,
        "variables": dict(state.variables),
        "iterations": state.iterations,
        "error": state.error,
    }

    observation = state.last_observation.to_dict() if state.last_observation else None
    history = [o.to_dict() for o in state.history]

    # Pull the durable_facts the runtime has accumulated for this
    # session.  We deliberately read the file on every bundle build
    # (not the in-memory state) so the facts collected by either
    # the foreground chat path or the background run path are
    # visible.
    known_entities: list[dict[str, Any]] = []
    try:
        from .memory import read_memory_file  # late import to avoid cycle
        facts_raw = read_memory_file(session_id, "durable_facts.json")
    except Exception:
        facts_raw = None
    if isinstance(facts_raw, list):
        # Cap the list so we never blow the window with one giant
        # run's worth of research.  The most recent facts are kept.
        for fact in facts_raw[-30:]:
            if isinstance(fact, dict) and fact.get("claim"):
                known_entities.append(
                    {
                        "claim": str(fact.get("claim") or ""),
                        "source": str(fact.get("source") or "durable_facts"),
                        "confidence": str(fact.get("confidence") or "medium"),
                    }
                )

    return {
        "spec": spec,
        "state": compact_state,
        "observation": observation,
        "history": history,
        "known_entities": known_entities,
    }


def start_or_resume(session_id: str, skill_name: str, *, user_prompt: str | None = None) -> SkillState:
    """Load the existing state for ``(session, skill)`` or build a
    fresh one. If ``user_prompt`` is provided and the state has no
    observations yet it is pushed as the first observation."""
    state = load_state(session_id, skill_name)
    if state is None:
        state = reset_state(session_id, skill_name)
    if user_prompt and not state.history:
        state = _push_observation(state, {"kind": "user", "source": "user", "content": user_prompt})
        save_state(session_id, state)
    return state


def apply_step(
    session_id: str,
    skill_name: str,
    *,
    transition: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    user_prompt: str | None = None,
) -> SkillState:
    """Single helper that wraps the whole reducer pipeline:

        state := load_state(session, skill) or reset_state(...)
        if user_prompt: state := push_observation(state, user_prompt)
        if observation: state := push_observation(state, observation)
        state := validate_transition(state, transition)
        save_state(session, state)

    Returns the new state. Raises ``TransitionError`` on invalid input.
    """
    state = start_or_resume(session_id, skill_name, user_prompt=user_prompt)
    if observation:
        state = _push_observation(state, observation)
    state = validate_transition(state, transition)
    save_state(session_id, state)
    return state


def plan_skill_delegation(
    session_id: str,
    skill_name: str,
    *,
    user_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan the next delegated tool call for a skill that has
    ``delegates_to`` in its spec.

    Returns a dict with the tool name and the args the caller
    should pass to the MCP registry. Handles the special case of
    ``web-search-loop`` (and any other skill that declares
    ``delegates_to.max_attempts``) by:

    * Reading the per-run state to count how many consecutive
      empty-result tool calls we've already seen.
    * Rotating ``prefer_engine`` across the fallback list
      ``(brave → duckduckgo → bing → google)`` on each attempt.
    * Returning ``{"__exhausted": True, "tool": "..."}`` once
      ``max_attempts`` is reached, so the caller can short-circuit
      and surface a "stop searching" error to the model.

    The function is intentionally idempotent: calling it twice in a
    row with the same state returns the same plan, so retries from
    the model don't accidentally skip engines.
    """
    skill = _load_skill(skill_name) or {}
    delegates = skill.get("delegates_to")
    if not isinstance(delegates, dict):
        return {"tool": None, "args": {}, "exhausted": False}

    tool = str(delegates.get("tool") or "")
    default_args = dict(delegates.get("default_args") or {})
    args_from = list(delegates.get("args_from") or [])
    max_attempts = int(delegates.get("max_attempts") or 1)

    # Build the args. User-supplied values win over defaults; only
    # keys listed in ``args_from`` are pulled from the user dict so
    # the skill stays in control of its contract.
    merged: dict[str, Any] = dict(default_args)
    if user_args:
        for k in args_from:
            if k in user_args:
                merged[k] = user_args[k]

    # Engine rotation. We keep a simple counter on the persisted
    # state so the rotation survives process restarts.
    state = load_state(session_id, skill_name)
    if state is None:
        state = start_or_resume(session_id, skill_name)

    # Count how many tool observations we've already received for
    # this skill and look at how many of them were "empty" so we
    # only rotate when the previous attempt really did fail.
    tool_attempts = 0
    empty_attempts = 0
    for obs in state.history:
        if getattr(obs, "kind", None) != "tool":
            continue
        tool_attempts += 1
        meta = getattr(obs, "meta", None) or {}
        if isinstance(meta, dict) and meta.get("empty_result"):
            empty_attempts += 1

    if empty_attempts >= max_attempts:
        return {
            "tool": tool,
            "args": merged,
            "exhausted": True,
            "attempts": empty_attempts,
            "max_attempts": max_attempts,
        }

    # Rotate engine if we have at least one empty result in the
    # history; the counter is per-skill so a fresh start stays on
    # the preferred default.
    if empty_attempts > 0 and isinstance(merged.get("prefer_engine"), str):
        from .search_budget import FALLBACK_ENGINES, PREFERRED_INITIAL_ENGINE

        # Choose the engine based on how many empty attempts we've
        # seen so far. The first empty rotates to ``fallback[0]``
        # (Brave's first replacement, i.e. DuckDuckGo). The second
        # empty rotates to ``fallback[1]`` (Bing), and so on. This
        # matches the in-memory ``SearchBudgetTracker`` exactly so
        # the SKILL.state path and the run-loop path produce the
        # same engine sequence for the same pattern of empty
        # results.
        idx = min(empty_attempts - 1, len(FALLBACK_ENGINES) - 1)
        next_engine = FALLBACK_ENGINES[idx]
        merged["prefer_engine"] = next_engine
        merged["engine"] = next_engine

    return {
        "tool": tool,
        "args": merged,
        "exhausted": False,
        "attempts": empty_attempts,
        "max_attempts": max_attempts,
    }


def record_skill_tool_observation(
    session_id: str,
    skill_name: str,
    *,
    tool: str,
    result_text: str | None,
    is_empty: bool | None = None,
) -> SkillState:
    """Push a tool observation onto the skill's history with the
    ``empty_result`` metadata flag set when the result was empty.

    Used by callers that wire a skill's ``delegates_to.tool`` to a
    real MCP call (e.g. the web-search MCP for ``web-search-loop``).
    The flag is what ``plan_skill_delegation`` reads back to decide
    whether to rotate the engine on the next attempt.
    """
    state = load_state(session_id, skill_name)
    if state is None:
        state = start_or_resume(session_id, skill_name)
    if is_empty is None:
        from .search_budget import looks_like_empty_search_result

        is_empty = looks_like_empty_search_result(result_text)
    state = _push_observation(
        state,
        {
            "kind": "tool",
            "source": tool,
            "content": (result_text or "")[:4000],
            "meta": {
                "tool": tool,
                "empty_result": bool(is_empty),
            },
        },
    )
    save_state(session_id, state)
    return state


def list_states(session_id: str) -> list[dict[str, Any]]:
    """Return a lightweight summary of every persisted state for the
    session. Used by the GET /skill-states endpoint.

    The list of skill names is derived from the on-disk registry via
    the same helper used elsewhere (``_load_skill``), which makes the
    function trivially testable — the caller can swap that helper in
    tests to point at a fixture file.
    """
    summaries: list[dict[str, Any]] = []
    for name in list_skills_in_registry():
        state = load_state(session_id, name)
        if state is None:
            continue
        summaries.append(state.to_dict())
    return summaries


def list_skills_in_registry() -> list[str]:
    """Enumerate every skill name registered on disk. Wraps the same
    candidate-file walk used elsewhere so tests can monkeypatch it."""
    override = os.environ.get("SKILLS_REGISTRY_PATH")
    candidates: list[str] = []
    if override:
        candidates.append(override)
    candidates.extend([
        "/app/skills/registry.json",
        "/skills/registry.json",
    ])
    here = os.path.dirname(os.path.abspath(__file__))
    for levels in (2, 3, 4):
        joined = os.path.normpath(os.path.join(here, *([".."] * levels), "skills", "registry.json"))
        candidates.append(joined)

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return list((data.get("skills") or {}).keys())
        except Exception:
            return []
    return []