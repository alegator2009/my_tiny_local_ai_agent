from __future__ import annotations

import json
import uuid
from typing import Any

from ..db import execute, fetch_all, utcnow_iso
from ..storage import append_transcript_event, list_artifacts, list_workspace_files
from .sessions import get_last_window


def log_workspace_event(session_id: str, event_type: str, payload_json: dict[str, Any], summary_text: str) -> dict[str, Any]:
    window = get_last_window(session_id)
    if not window:
        raise KeyError("window_not_found")

    wid = window["id"]
    eid = str(uuid.uuid4())
    ts = utcnow_iso()

    execute(
        """
        INSERT INTO workspace_events (id, session_id, window_id, event_type, timestamp, payload_json, summary_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (eid, session_id, wid, event_type, ts, json.dumps(payload_json, ensure_ascii=False), summary_text),
    )

    append_transcript_event(
        session_id,
        {
            "timestamp": ts,
            "type": "workspace_event",
            "id": eid,
            "window_id": wid,
            "event_type": event_type,
            "payload_json": payload_json,
            "summary_text": summary_text,
        },
    )

    return {
        "id": eid,
        "session_id": session_id,
        "window_id": wid,
        "event_type": event_type,
        "timestamp": ts,
        "payload_json": payload_json,
        "summary_text": summary_text,
    }


def get_workspace_status(session_id: str) -> dict[str, Any]:
    recent_rows = fetch_all(
        "SELECT event_type, timestamp, summary_text FROM workspace_events WHERE session_id=? ORDER BY timestamp DESC LIMIT 20",
        (session_id,),
    )

    active_files = []
    for row in fetch_all(
        "SELECT payload_json FROM workspace_events WHERE session_id=? ORDER BY timestamp DESC LIMIT 30",
        (session_id,),
    ):
        payload = json.loads(row["payload_json"])
        if "file" in payload:
            active_files.append(payload["file"])

    return {
        "active_files": sorted(set(active_files)),
        "recent_events": [dict(r) for r in recent_rows],
        "artifacts": list_artifacts(session_id),
        "workspace_files": list_workspace_files(session_id),
    }
