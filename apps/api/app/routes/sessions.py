from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import ExportImportResponse, ImportSessionRequest, SessionCreate, SessionOut, SessionUpdate
from ..services.export_import import export_session, import_session
from ..services.graph import build_session_graph
from ..services.sessions import (
    archive_session,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    update_session,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut)
def create_session_endpoint(payload: SessionCreate):
    return create_session(payload)


@router.get("", response_model=list[SessionOut])
def list_sessions_endpoint():
    return list_sessions()


@router.get("/{session_id}", response_model=SessionOut)
def get_session_endpoint(session_id: str):
    try:
        return get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.patch("/{session_id}", response_model=SessionOut)
def update_session_endpoint(session_id: str, payload: SessionUpdate):
    try:
        return update_session(session_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/{session_id}")
def delete_session_endpoint(session_id: str):
    delete_session(session_id)
    return {"ok": True}


@router.post("/{session_id}/archive", response_model=SessionOut)
def archive_session_endpoint(session_id: str):
    try:
        return archive_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/{session_id}/export", response_model=ExportImportResponse)
def export_session_endpoint(session_id: str):
    try:
        path = export_session(session_id)
        return {"path": path}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/import", response_model=ExportImportResponse)
def import_session_endpoint(payload: ImportSessionRequest):
    try:
        sid = import_session(payload.archive_path)
        return {"path": sid}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{session_id}/graph")
def get_session_graph_endpoint(session_id: str):
    """Return a graph representation of a session for visualization.

    The response shape is:
    {
      "session_id": str,
      "windows": [...],
      "checkpoints": [...],
      "nodes": [...],
      "edges": [...]
    }
    """
    try:
        return build_session_graph(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
