from __future__ import annotations

import uuid

from ..db import execute, fetch_all, fetch_one, utcnow_iso
from ..schemas import MessagePrefixTemplateCreate


def _row_to_template(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "prompt": row["prompt"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_message_prefix_templates() -> list[dict]:
    rows = fetch_all(
        "SELECT id, name, prompt, created_at, updated_at FROM message_prefix_templates ORDER BY updated_at DESC"
    )
    return [_row_to_template(row) for row in rows]


def save_message_prefix_template(payload: MessagePrefixTemplateCreate) -> dict:
    name = payload.name.strip()
    prompt = payload.prompt.strip()
    if not name:
        raise ValueError("template_name_required")
    if not prompt:
        raise ValueError("template_prompt_required")

    now = utcnow_iso()
    existing = fetch_one("SELECT id FROM message_prefix_templates WHERE name=?", (name,))
    if existing is not None:
        template_id = existing["id"]
        execute(
            "UPDATE message_prefix_templates SET prompt=?, updated_at=? WHERE id=?",
            (prompt, now, template_id),
        )
    else:
        template_id = str(uuid.uuid4())
        execute(
            """
            INSERT INTO message_prefix_templates (id, name, prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (template_id, name, prompt, now, now),
        )

    saved = fetch_one(
        "SELECT id, name, prompt, created_at, updated_at FROM message_prefix_templates WHERE id=?",
        (template_id,),
    )
    if saved is None:
        raise RuntimeError("template_not_saved")
    return _row_to_template(saved)


def delete_message_prefix_template(template_id: str) -> None:
    existing = fetch_one("SELECT id FROM message_prefix_templates WHERE id=?", (template_id,))
    if existing is None:
        raise KeyError("template_not_found")
    execute("DELETE FROM message_prefix_templates WHERE id=?", (template_id,))
