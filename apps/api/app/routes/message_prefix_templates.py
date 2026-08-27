from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import MessagePrefixTemplateCreate, MessagePrefixTemplateOut
from ..services.message_prefix_templates import (
    delete_message_prefix_template,
    list_message_prefix_templates,
    save_message_prefix_template,
)

router = APIRouter(prefix="/api/message-prefix-templates", tags=["message-prefix-templates"])


@router.get("", response_model=list[MessagePrefixTemplateOut])
def list_message_prefix_templates_endpoint():
    return list_message_prefix_templates()


@router.post("", response_model=MessagePrefixTemplateOut)
def save_message_prefix_template_endpoint(payload: MessagePrefixTemplateCreate):
    try:
        return save_message_prefix_template(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{template_id}")
def delete_message_prefix_template_endpoint(template_id: str):
    try:
        delete_message_prefix_template(template_id)
        return {"ok": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="Template not found")
