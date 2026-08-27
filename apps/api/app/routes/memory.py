from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import PinRequest
from ..services.memory import (
    detect_fact_conflicts,
    create_checkpoint,
    list_claim_sources,
    list_checkpoints,
    list_fact_conflicts,
    list_lint_runs,
    list_memory_table,
    list_retrieval_logs,
    maybe_run_scheduled_wiki_lint,
    memory_snapshot,
    pin_message,
    run_wiki_lint,
)
from ..services.sessions import get_last_window

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/{session_id}/checkpoints")
def checkpoints(session_id: str):
    return list_checkpoints(session_id)


@router.get("/{session_id}/facts")
def facts(session_id: str):
    return list_memory_table(session_id, "facts")


@router.get("/{session_id}/decisions")
def decisions(session_id: str):
    return list_memory_table(session_id, "decisions")


@router.get("/{session_id}/tasks")
def tasks(session_id: str):
    return list_memory_table(session_id, "tasks")


@router.get("/{session_id}/retrieval-logs")
def retrieval_logs(session_id: str):
    return list_retrieval_logs(session_id)


@router.get("/{session_id}/claim-sources")
def claim_sources(session_id: str):
    return list_claim_sources(session_id)


@router.get("/{session_id}/fact-conflicts")
def fact_conflicts(session_id: str):
    return list_fact_conflicts(session_id)


@router.get("/{session_id}/lint-runs")
def lint_runs(session_id: str):
    return list_lint_runs(session_id)


@router.get("/{session_id}/snapshot")
def snapshot(session_id: str):
    return memory_snapshot(session_id)


@router.post("/{session_id}/checkpoints")
def create_manual_checkpoint(session_id: str, reason: str = "manual"):
    window = get_last_window(session_id)
    if not window:
        raise HTTPException(status_code=404, detail="Session window not found")
    return create_checkpoint(session_id, source_window_id=window["id"], reason=reason)


@router.post("/{session_id}/lint")
def lint(session_id: str, reason: str = "manual"):
    try:
        return run_wiki_lint(session_id, reason=reason)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{session_id}/lint/scheduled")
def lint_scheduled(session_id: str):
    try:
        result = maybe_run_scheduled_wiki_lint(session_id)
        return {"ok": True, "run": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{session_id}/fact-conflicts/refresh")
def refresh_fact_conflicts(session_id: str):
    try:
        return detect_fact_conflicts(session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{session_id}/pin")
def pin(session_id: str, payload: PinRequest):
    try:
        pin_message(
            session_id=session_id,
            message_id=payload.message_id,
            pinned=payload.pinned,
            anchor=payload.anchor,
        )
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
