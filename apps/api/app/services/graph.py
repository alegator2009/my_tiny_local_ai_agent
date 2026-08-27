from __future__ import annotations

import json
from typing import Any

from ..db import fetch_all


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()} if row is not None else {}


def _safe_json(raw: Any) -> Any:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def build_session_graph(session_id: str) -> dict[str, Any]:
    """Build a graph representation of a session.

    Returns:
        {
          "session_id": str,
          "windows": [{id, index, started_at, closed_at, closing_reason, token_limit, used_tokens}],
          "checkpoints": [{id, source_window_id, checkpoint_index, created_at, summary_text}],
          "nodes": [{id, type, label, sub_label, window_id, turn_id, ts, token_count, meta}],
          "edges": [{id, source, target, kind, label}],
        }
    """
    # 1) Windows (context windows) — these define "frames" of the graph
    window_rows = fetch_all(
        """
        SELECT id, window_index, started_at, closed_at, closing_reason,
               token_limit, rollover_trigger_percent, checkpoint_id
        FROM windows WHERE session_id=?
        ORDER BY window_index ASC
        """,
        (session_id,),
    )
    windows: list[dict[str, Any]] = []
    for r in window_rows:
        windows.append(
            {
                "id": r["id"],
                "index": r["window_index"],
                "started_at": r["started_at"],
                "closed_at": r["closed_at"],
                "closing_reason": r["closing_reason"],
                "token_limit": r["token_limit"],
                "rollover_trigger_percent": r["rollover_trigger_percent"],
                "checkpoint_id": r["checkpoint_id"],
            }
        )

    # 2) Checkpoints (memory summaries between windows)
    checkpoint_rows = fetch_all(
        """
        SELECT id, source_window_id, checkpoint_index, created_at, summary_text,
               working_set_json, decisions_json
        FROM checkpoints WHERE session_id=?
        ORDER BY checkpoint_index ASC
        """,
        (session_id,),
    )
    checkpoints: list[dict[str, Any]] = []
    for r in checkpoint_rows:
        checkpoints.append(
            {
                "id": r["id"],
                "source_window_id": r["source_window_id"],
                "checkpoint_index": r["checkpoint_index"],
                "created_at": r["created_at"],
                "summary_text": r["summary_text"],
                "decisions_count": len(_safe_json(r["decisions_json"]) or []),
            }
        )

    # 3) Messages (chat / tool_call / tool_result)
    msg_rows = fetch_all(
        """
        SELECT id, window_id, turn_id, role, message_type, timestamp,
               content_text, token_count, is_anchor, is_pinned
        FROM messages WHERE session_id=?
        ORDER BY timestamp ASC, id ASC
        """,
        (session_id,),
    )

    # 4) Workspace events (mcp / terminal / file / artifact)
    ws_rows = fetch_all(
        """
        SELECT id, window_id, event_type, timestamp, payload_json, summary_text
        FROM workspace_events WHERE session_id=?
        ORDER BY timestamp ASC, id ASC
        """,
        (session_id,),
    )

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # 4a) Window "frame" nodes
    for w in windows:
        nodes.append(
            {
                "id": f"window:{w['id']}",
                "type": "window",
                "label": f"Window #{w['index']}",
                "sub_label": w["closing_reason"] or "active",
                "window_id": w["id"],
                "turn_id": None,
                "ts": w["started_at"],
                "token_count": 0,
                "meta": {
                    "closed_at": w["closed_at"],
                    "token_limit": w["token_limit"],
                    "index": w["index"],
                },
            }
        )

    # 4b) Checkpoint nodes
    for cp in checkpoints:
        nodes.append(
            {
                "id": f"checkpoint:{cp['id']}",
                "type": "checkpoint",
                "label": f"Checkpoint #{cp['checkpoint_index']}",
                "sub_label": (cp["summary_text"] or "")[:80] or "—",
                "window_id": cp["source_window_id"],
                "turn_id": None,
                "ts": cp["created_at"],
                "token_count": 0,
                "meta": {
                    "decisions_count": cp["decisions_count"],
                    "summary": cp["summary_text"],
                },
            }
        )

    # 4c) Message nodes
    msg_id_to_node: dict[str, str] = {}
    ordered_msg_ids: list[str] = []
    for m in msg_rows:
        kind = m["message_type"] or m["role"] or "message"
        node_id = f"msg:{m['id']}"
        msg_id_to_node[m["id"]] = node_id
        ordered_msg_ids.append(m["id"])
        content_preview = (m["content_text"] or "").replace("\n", " ").strip()
        if len(content_preview) > 90:
            content_preview = content_preview[:87] + "..."
        label_map = {
            "user": "👤 user",
            "assistant": "🤖 assistant",
            "tool_call": "🛠 tool call",
            "tool_result": "📥 tool result",
            "system": "system",
        }
        nodes.append(
            {
                "id": node_id,
                "type": kind,
                "label": label_map.get(kind, kind),
                "sub_label": content_preview or "—",
                "window_id": m["window_id"],
                "turn_id": m["turn_id"],
                "ts": m["timestamp"],
                "token_count": m["token_count"] or 0,
                "meta": {
                    "role": m["role"],
                    "is_anchor": bool(m["is_anchor"]),
                    "is_pinned": bool(m["is_pinned"]),
                    "full_content": m["content_text"] or "",
                },
            }
        )

    # 4d) Workspace event nodes
    ws_id_to_node: dict[str, str] = {}
    ordered_ws_ids: list[str] = []
    for ev in ws_rows:
        node_id = f"ws:{ev['id']}"
        ws_id_to_node[ev["id"]] = node_id
        ordered_ws_ids.append(ev["id"])
        payload = _safe_json(ev["payload_json"])
        tool_name = ""
        if isinstance(payload, dict):
            tool_name = payload.get("tool") or payload.get("command") or ""
        sub = (ev["summary_text"] or tool_name or ev["event_type"] or "").strip()
        if len(sub) > 90:
            sub = sub[:87] + "..."
        label_map = {
            "mcp_tool_call": "📡 mcp call",
            "mcp_tool_result": "📨 mcp result",
            "terminal_command": "💻 terminal",
            "terminal_output": "🖥 terminal out",
            "file_change": "✏️ file change",
            "file_snapshot": "📄 file snap",
            "diff_summary": "🔍 diff",
            "test_result": "✅ test",
            "build_error": "❌ build error",
            "image_created": "🖼 image",
            "artifact_created": "📦 artifact",
        }
        nodes.append(
            {
                "id": node_id,
                "type": ev["event_type"],
                "label": label_map.get(ev["event_type"], ev["event_type"]),
                "sub_label": sub or "—",
                "window_id": ev["window_id"],
                "turn_id": None,
                "ts": ev["timestamp"],
                "token_count": 0,
                "meta": {
                    "summary": ev["summary_text"] or "",
                    "payload": payload,
                },
            }
        )

    # ---- Edges ----
    # 5a) Sequence: messages and ws events in chronological order, scoped per window
    # Group by window
    per_window: dict[str, list[tuple[str, str]]] = {}
    for mid in ordered_msg_ids:
        row = next((r for r in msg_rows if r["id"] == mid), None)
        if not row:
            continue
        per_window.setdefault(row["window_id"], []).append(
            (row["timestamp"], msg_id_to_node[mid])
        )
    for wid in per_window:
        per_window[wid].sort(key=lambda x: x[0])

    for wid in per_window:
        items = per_window[wid]
        prev = None
        for _, nid in items:
            if prev is not None:
                edges.append(
                    {
                        "id": f"seq:{prev}->{nid}",
                        "source": prev,
                        "target": nid,
                        "kind": "sequence",
                        "label": "",
                    }
                )
            prev = nid

    # 5b) Merge workspace events into the same temporal sequence per window
    for ev in ws_rows:
        per_window.setdefault(ev["window_id"], []).append(
            (ev["timestamp"], ws_id_to_node[ev["id"]])
        )
    for wid in per_window:
        per_window[wid].sort(key=lambda x: x[0])
        items = per_window[wid]
        prev = None
        for _, nid in items:
            if prev is not None:
                # Only add if not already a sequence edge (messages are already chained)
                if not any(
                    e["source"] == prev and e["target"] == nid for e in edges
                ):
                    edges.append(
                        {
                            "id": f"seq:{prev}->{nid}",
                            "source": prev,
                            "target": nid,
                            "kind": "sequence",
                            "label": "",
                        }
                    )
            prev = nid

    # 5c) Tool call → tool result (best-effort by tool name in payload)
    open_calls: list[dict[str, Any]] = []
    for ev in ws_rows:
        payload = _safe_json(ev["payload_json"])
        if ev["event_type"] == "mcp_tool_call":
            open_calls.append(
                {
                    "node_id": ws_id_to_node[ev["id"]],
                    "tool": payload.get("tool") if isinstance(payload, dict) else "",
                    "ts": ev["timestamp"],
                }
            )
        elif ev["event_type"] == "mcp_tool_result" and open_calls:
            payload_dict = payload if isinstance(payload, dict) else {}
            tool = payload_dict.get("tool", "")
            # Match by tool name first; fallback to FIFO
            match_idx = None
            for i, c in enumerate(open_calls):
                if tool and c["tool"] == tool:
                    match_idx = i
                    break
            if match_idx is None:
                match_idx = 0
            call = open_calls.pop(match_idx)
            edges.append(
                {
                    "id": f"tool:{call['node_id']}->{ws_id_to_node[ev['id']]}",
                    "source": call["node_id"],
                    "target": ws_id_to_node[ev["id"]],
                    "kind": "tool_io",
                    "label": tool or "",
                }
            )

    # Also: message-level tool_call → tool_result
    open_msg_calls: list[dict[str, Any]] = []
    for m in msg_rows:
        if m["message_type"] == "tool_call":
            # Try to extract tool name from content_text "Tool call <name>: {args}"
            text = m["content_text"] or ""
            tool_name = ""
            if text.startswith("Tool call "):
                rest = text[len("Tool call "):]
                tool_name = rest.split(":", 1)[0].strip()
            open_msg_calls.append(
                {
                    "node_id": msg_id_to_node[m["id"]],
                    "tool": tool_name,
                    "ts": m["timestamp"],
                }
            )
        elif m["message_type"] == "tool_result" and open_msg_calls:
            text = m["content_text"] or ""
            tool_name = ""
            if text.startswith("Tool result "):
                rest = text[len("Tool result "):]
                tool_name = rest.split(":", 1)[0].strip()
            match_idx = None
            for i, c in enumerate(open_msg_calls):
                if tool_name and c["tool"] == tool_name:
                    match_idx = i
                    break
            if match_idx is None:
                match_idx = 0
            call = open_msg_calls.pop(match_idx)
            edges.append(
                {
                    "id": f"tool:{call['node_id']}->{msg_id_to_node[m['id']]}",
                    "source": call["node_id"],
                    "target": msg_id_to_node[m["id"]],
                    "kind": "tool_io",
                    "label": tool_name or "",
                }
            )

    # 5d) Window frame → first message in that window
    for w in windows:
        first = next(
            (nid for _, nid in sorted(per_window.get(w["id"], []), key=lambda x: x[0])),
            None,
        )
        if first:
            edges.append(
                {
                    "id": f"window:{w['id']}->{first}",
                    "source": f"window:{w['id']}",
                    "target": first,
                    "kind": "contains",
                    "label": "contains",
                }
            )
        # Window → checkpoint (if any)
        if w["checkpoint_id"]:
            edges.append(
                {
                    "id": f"window:{w['id']}->checkpoint:{w['checkpoint_id']}",
                    "source": f"window:{w['id']}",
                    "target": f"checkpoint:{w['checkpoint_id']}",
                    "kind": "summarizes",
                    "label": "summarizes",
                }
            )

    # 5e) Checkpoint → first message of the next window (resume)
    for i, w in enumerate(windows):
        if not w["checkpoint_id"]:
            continue
        # Find next window
        if i + 1 < len(windows):
            nxt = windows[i + 1]
            first_next = next(
                (nid for _, nid in sorted(per_window.get(nxt["id"], []), key=lambda x: x[0])),
                None,
            )
            if first_next:
                edges.append(
                    {
                        "id": f"resume:{w['checkpoint_id']}->{first_next}",
                        "source": f"checkpoint:{w['checkpoint_id']}",
                        "target": first_next,
                        "kind": "resumes",
                        "label": "resumes",
                    }
                )

    # 5f) Last msg of window → checkpoint
    for w in windows:
        if not w["checkpoint_id"]:
            continue
        last = per_window.get(w["id"], [])
        if last:
            last_nid = last[-1][1]
            edges.append(
                {
                    "id": f"closes:{last_nid}->checkpoint:{w['checkpoint_id']}",
                    "source": last_nid,
                    "target": f"checkpoint:{w['checkpoint_id']}",
                    "kind": "closes",
                    "label": "closes",
                }
            )

    return {
        "session_id": session_id,
        "windows": windows,
        "checkpoints": checkpoints,
        "nodes": nodes,
        "edges": edges,
    }
