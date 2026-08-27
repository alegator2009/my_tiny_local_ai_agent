from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas import MessageCreate
from ..services.orchestrator import get_transcript, get_window_state, stream_chat

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/{session_id}/stream")
async def stream_response(session_id: str, payload: MessageCreate):
    try:
        iterator = stream_chat(
            session_id=session_id,
            user_content=payload.content,
            thinking_mode_override=payload.thinking_mode,
            provider_id=payload.provider_id,
            model_id=payload.model_id,
            force_search=payload.force_search,
            bypass_search_cache=payload.bypass_search_cache,
            active_skill=payload.active_skill,
            context_mode_override=payload.context_mode_override,
        )
        return StreamingResponse(iterator, media_type="text/event-stream")
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/transcript")
def get_transcript_endpoint(session_id: str):
    try:
        return get_transcript(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/window-state")
def get_window_state_endpoint(session_id: str):
    try:
        return get_window_state(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")