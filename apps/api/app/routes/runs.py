from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import RunArtifactOut, RunCreate, RunEventOut, RunOut
from ..services.runs import cancel_run, create_run, get_run, list_run_artifacts, list_run_events, list_runs

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("/{session_id}", response_model=RunOut)
def start_run(session_id: str, payload: RunCreate):
    try:
        return create_run(session_id=session_id, task_text=payload.content)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{session_id}", response_model=list[RunOut])
def list_runs_endpoint(session_id: str):
    try:
        return list_runs(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/{run_id}", response_model=RunOut)
def get_run_endpoint(session_id: str, run_id: str):
    try:
        return get_run(session_id, run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")


@router.get("/{session_id}/{run_id}/events", response_model=list[RunEventOut])
def get_run_events_endpoint(session_id: str, run_id: str):
    try:
        return list_run_events(session_id, run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")


@router.get("/{session_id}/{run_id}/artifacts", response_model=list[RunArtifactOut])
def get_run_artifacts_endpoint(session_id: str, run_id: str):
    try:
        return list_run_artifacts(session_id, run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")


@router.post("/{session_id}/{run_id}/cancel", response_model=RunOut)
def cancel_run_endpoint(session_id: str, run_id: str):
    try:
        return cancel_run(session_id, run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
