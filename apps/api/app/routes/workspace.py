from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..schemas import WorkspaceEventIn
from ..storage import list_artifacts, list_workspace_files
from ..services.artifacts import resolve_artifact_download
from ..services.workspace import get_workspace_status, log_workspace_event

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


@router.get("/{session_id}/artifacts")
def artifacts(session_id: str):
    return {"items": list_artifacts(session_id)}


@router.get("/{session_id}/artifacts/{artifact_id}/download")
def download_artifact(session_id: str, artifact_id: str):
    try:
        artifact = resolve_artifact_download(session_id=session_id, artifact_id=artifact_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(
        artifact["path"],
        media_type=artifact["media_type"],
        filename=artifact["filename"],
    )


@router.get("/{session_id}/files")
def files(session_id: str):
    return {"items": list_workspace_files(session_id)}


@router.get("/{session_id}/status")
def status(session_id: str):
    return get_workspace_status(session_id)


@router.post("/{session_id}/events")
def add_event(session_id: str, payload: WorkspaceEventIn):
    try:
        return log_workspace_event(
            session_id=session_id,
            event_type=payload.event_type,
            payload_json=payload.payload_json,
            summary_text=payload.summary_text,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
