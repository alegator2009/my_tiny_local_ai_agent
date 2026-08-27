"""skill_state.py (routes) — HTTP surface for the SKILL.state runtime.

The endpoints here mirror the operations available in
``services.skill_state`` and ``services.memory`` so that the web UI and
external clients can drive the same state machine the orchestrator uses
internally.

Endpoints
---------

* ``GET  /api/skill-state/{session_id}``               — list states.
* ``GET  /api/skill-state/{session_id}/{skill_name}``  — fetch one.
* ``POST /api/skill-state/{session_id}/{skill_name}/start`` —
        initialise or resume. Body: ``{"user_prompt": "..."}``.
* ``POST /api/skill-state/{session_id}/{skill_name}/step``  —
        apply a transition. Body: ``{"transition": {...},
        "observation": {...}, "user_prompt": "..."}``.
* ``POST /api/skill-state/{session_id}/{skill_name}/reset`` —
        wipe the persisted state.
* ``GET  /api/skill-state/{session_id}/{skill_name}/bundle`` —
        build the (spec, state, observation) bundle used by the model.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

class PlanDelegationRequest(BaseModel):
    user_args: dict[str, Any] | None = None


class RecordToolObservationRequest(BaseModel):
    tool: str
    result_text: str | None = None
    is_empty: bool | None = None


from ..services.memory import (
    apply_skill_transition,
    build_skill_prompt_bundle,
    list_skill_states,
    load_skill_state,
    plan_skill_delegation,
    record_skill_tool_observation,
    reset_skill_state,
    start_or_resume_skill,
)
from ..services.skill_state import TransitionError

router = APIRouter(prefix="/api/skill-state", tags=["skill-state"])


class StartRequest(BaseModel):
    user_prompt: str | None = None
    max_history: int | None = Field(default=None, ge=1, le=64)


class StepRequest(BaseModel):
    transition: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    user_prompt: str | None = None


def _wrap_transition_error(exc: TransitionError) -> HTTPException:
    return HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message})


@router.get("/{session_id}")
def list_states(session_id: str) -> dict[str, Any]:
    return {"states": list_skill_states(session_id)}


@router.get("/{session_id}/{skill_name}")
def get_state(session_id: str, skill_name: str) -> dict[str, Any]:
    state = load_skill_state(session_id, skill_name)
    if state is None:
        raise HTTPException(status_code=404, detail="skill state not found")
    return state


@router.post("/{session_id}/{skill_name}/start")
def start_state(session_id: str, skill_name: str, payload: StartRequest | None = None) -> dict[str, Any]:
    payload = payload or StartRequest()
    try:
        state = start_or_resume_skill(
            session_id,
            skill_name,
            user_prompt=payload.user_prompt,
        )
        return state
    except TransitionError as exc:
        raise _wrap_transition_error(exc)


@router.post("/{session_id}/{skill_name}/step")
def step_state(session_id: str, skill_name: str, payload: StepRequest) -> dict[str, Any]:
    try:
        state = apply_skill_transition(
            session_id,
            skill_name,
            transition=payload.transition,
            observation=payload.observation,
            user_prompt=payload.user_prompt,
        )
        return state
    except TransitionError as exc:
        raise _wrap_transition_error(exc)


@router.post("/{session_id}/{skill_name}/reset")
def reset_state(session_id: str, skill_name: str) -> dict[str, Any]:
    return reset_skill_state(session_id, skill_name)


@router.get("/{session_id}/{skill_name}/bundle")
def bundle(session_id: str, skill_name: str) -> dict[str, Any]:
    return build_skill_prompt_bundle(session_id, skill_name)


@router.post("/{session_id}/{skill_name}/plan-delegation")
def plan_delegation(
    session_id: str,
    skill_name: str,
    payload: PlanDelegationRequest | None = None,
) -> dict[str, Any]:
    """Plan the next delegated tool call for a skill with a
    ``delegates_to`` block. Returns ``{tool, args, exhausted,
    attempts, max_attempts}`` so the caller (typically the JS
    ``wrapper.mjs`` MCP server) can call the right tool with the
    right ``prefer_engine`` on the next attempt. Returns
    ``exhausted=True`` when the skill's budget is already
    spent so the wrapper can short-circuit with a "stop
    searching" message instead of calling the engine again."""
    user_args = payload.user_args if payload else None
    return plan_skill_delegation(session_id, skill_name, user_args=user_args)


@router.post("/{session_id}/{skill_name}/record-tool")
def record_tool(
    session_id: str,
    skill_name: str,
    payload: RecordToolObservationRequest,
) -> dict[str, Any]:
    """Push a tool observation onto the skill's history. The
    ``empty_result`` flag (derived from the result text when not
    provided) is what the next ``plan-delegation`` call uses to
    decide whether to rotate the engine."""
    return record_skill_tool_observation(
        session_id,
        skill_name,
        tool=payload.tool,
        result_text=payload.result_text,
        is_empty=payload.is_empty,
    )