from __future__ import annotations

import json
import uuid
from typing import Any

from ..db import execute, fetch_all, fetch_one, utcnow_iso
from .vector_store import vector_store


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    # Deterministic rough estimate for budgeting and rollover triggers.
    return max(1, int(len(text.split()) * 1.35))


def detect_chunk_type(text: str, role: str = "assistant") -> str:
    lowered = text.lower()
    if "error" in lowered or "traceback" in lowered:
        return "error"
    todo_hints = (
        "todo", "next", "need to", "we should",
        "pendiente", "necesito", "hay que", "à faire", "preciso",
        "müssen", "dovremmo", "musimy", "moeten", "yapmalıyız",
        "やること", "해야 할 일", "我们需要", "हमें करना", "يجب أن",
        "потрібно",
    )
    decision_hints = (
        "decide", "decided", "decision",
        "decidido", "decidimos", "décidé", "entschieden",
        "abbiamo deciso", "zdecydowaliśmy", "besloten", "karar verdik",
        "決定", "결정", "हमने तय किया", "قررنا",
        "виріш",
    )
    if any(h in lowered for h in todo_hints):
        return "todo"
    if any(h in lowered for h in decision_hints):
        return "decision"
    if "```" in text:
        return "code"
    if role == "user":
        return "requirement"
    return "conversation"


def index_message(
    *,
    message_id: str,
    session_id: str,
    window_id: str,
    role: str,
    text: str,
    timestamp: str,
) -> dict[str, Any]:
    token_count = estimate_token_count(text)
    chunk_type = detect_chunk_type(text, role=role)

    chunk_row = fetch_one("SELECT COALESCE(MAX(chunk_index), 0) AS max_idx FROM chunks WHERE session_id=?", (session_id,))
    chunk_index = int(chunk_row["max_idx"]) + 1 if chunk_row else 1
    chunk_id = str(uuid.uuid4())
    created_at = utcnow_iso()

    emb = vector_store.upsert_chunk(
        chunk_id=chunk_id,
        session_id=session_id,
        window_id=window_id,
        chunk_type=chunk_type,
        text=text,
        created_at=created_at,
    )

    execute(
        """
        INSERT INTO chunks (
          id, session_id, window_id, chunk_index, chunk_type, text, token_count,
          embedding_ref, start_message_id, end_message_id, recency_score, importance_score, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            session_id,
            window_id,
            chunk_index,
            chunk_type,
            text,
            token_count,
            json.dumps({"dims": len(emb)}),
            message_id,
            message_id,
            1.0,
            1.0 if chunk_type in {"decision", "error", "todo"} else 0.5,
            created_at,
        ),
    )

    execute(
        """
        INSERT INTO messages_fts (message_id, session_id, window_id, role, text, timestamp, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (message_id, session_id, window_id, role, text, timestamp, chunk_type),
    )

    execute(
        """
        INSERT INTO chunks_fts (chunk_id, session_id, window_id, chunk_type, text, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, session_id, window_id, chunk_type, text, created_at),
    )

    return {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "token_count": token_count,
    }


def recent_neighbor_chunks(session_id: str, chunk_ids: list[str], prev_count: int, next_count: int) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []

    placeholders = ",".join("?" for _ in chunk_ids)
    selected = fetch_all(
        f"SELECT id, chunk_index FROM chunks WHERE session_id=? AND id IN ({placeholders})",
        (session_id, *chunk_ids),
    )

    wanted_indexes = set()
    for row in selected:
        idx = row["chunk_index"]
        for delta in range(-prev_count, next_count + 1):
            wanted_indexes.add(idx + delta)

    rows = fetch_all(
        """
        SELECT id, chunk_index, chunk_type, text, created_at
        FROM chunks
        WHERE session_id=?
        ORDER BY chunk_index ASC
        """,
        (session_id,),
    )
    return [dict(r) for r in rows if r["chunk_index"] in wanted_indexes]
