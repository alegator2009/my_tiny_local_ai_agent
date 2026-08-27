from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException

from ..config import ModelEntry, ProviderConfig, load_app_config, save_app_config
from ..db import utcnow_iso
from ..schemas import (
    ActiveSelectionUpdate,
    ModelEntryCreate,
    ModelEntryOut,
    ModelEntryUpdate,
    ProviderCreate,
    ProviderListResponse,
    ProviderOut,
    ProviderUpdate,
)

router = APIRouter(prefix="/api/providers", tags=["providers"])


def _provider_or_404(provider_id: str):
    cfg = load_app_config()
    for p in cfg.providers:
        if p.id == provider_id:
            return cfg, p
    raise HTTPException(status_code=404, detail="provider_not_found")


def _model_or_404(provider, model_id: str):
    for m in provider.models:
        if m.id == model_id:
            return m
    raise HTTPException(status_code=404, detail="model_not_found")


def _stamp(provider) -> str:
    provider.updated_at = utcnow_iso()
    return provider.updated_at


def _ensure_single_default(cfg, provider_id: str, model_id: str) -> None:
    """If ``model_id`` is marked default, clear ``is_default`` on every
    other model of every provider.  We only allow one global default so
    the active pair picker has a stable fallback."""

    for prov in cfg.providers:
        for m in prov.models:
            m.is_default = bool(m.is_default) and (
                prov.id == provider_id and m.id == model_id
            )


def _model_from_payload(payload: ModelEntryCreate) -> ModelEntry:
    raw = payload.model_dump()
    raw.setdefault("id", str(uuid4()))
    raw.setdefault("created_at", "")
    raw.setdefault("updated_at", "")
    return ModelEntry.model_validate(raw)


@router.get("", response_model=ProviderListResponse)
def list_providers():
    cfg = load_app_config()
    return {
        "providers": [p.model_dump() for p in cfg.providers],
        "active_provider_id": cfg.active_provider_id,
        "active_model_id": cfg.active_model_id,
    }


@router.post("", response_model=ProviderOut)
def create_provider(payload: ProviderCreate):
    cfg = load_app_config()
    now = utcnow_iso()
    raw = payload.model_dump()
    raw.setdefault("id", str(uuid4()))
    raw["created_at"] = now
    raw["updated_at"] = now
    new_provider = ProviderConfig.model_validate(raw)
    cfg.providers.append(new_provider)

    # If this is the first provider and no active selection exists,
    # make the first model the active one so the chat picker has
    # something to bind to.
    if cfg.active_provider_id is None and new_provider.models:
        cfg.active_provider_id = new_provider.id
        cfg.active_model_id = (
            new_provider.default_model().id if new_provider.default_model() else None
        )
        if cfg.active_model_id:
            _ensure_single_default(cfg, new_provider.id, cfg.active_model_id)

    save_app_config(cfg)
    return new_provider


@router.get("/{provider_id}", response_model=ProviderOut)
def get_provider(provider_id: str):
    _, provider = _provider_or_404(provider_id)
    return provider


@router.patch("/{provider_id}", response_model=ProviderOut)
def update_provider(provider_id: str, payload: ProviderUpdate):
    cfg, provider = _provider_or_404(provider_id)

    updates = payload.model_dump(exclude_unset=True)
    # Apply field by field so we can keep validators happy.
    for field, value in updates.items():
        if hasattr(provider, field):
            setattr(provider, field, value)
    _stamp(provider)

    # If the user disabled this provider while it's the active one,
    # drop the active selection so the next reload picks something else
    # automatically.
    if not provider.enabled and cfg.active_provider_id == provider.id:
        cfg.active_provider_id = None
        cfg.active_model_id = None

    save_app_config(cfg)
    return provider


@router.delete("/{provider_id}")
def delete_provider(provider_id: str):
    cfg, provider = _provider_or_404(provider_id)
    cfg.providers = [p for p in cfg.providers if p.id != provider_id]
    if cfg.active_provider_id == provider_id:
        cfg.active_provider_id = None
        cfg.active_model_id = None
    save_app_config(cfg)
    return {"ok": True, "deleted_provider_id": provider_id}


@router.post("/{provider_id}/models", response_model=ModelEntryOut)
def add_model(provider_id: str, payload: ModelEntryCreate):
    cfg, provider = _provider_or_404(provider_id)
    new_model = _model_from_payload(payload)
    new_model.created_at = utcnow_iso()
    new_model.updated_at = new_model.created_at
    provider.models.append(new_model)
    if payload.is_default:
        _ensure_single_default(cfg, provider_id, new_model.id)
    if not provider.default_model():
        new_model.is_default = True
        cfg.active_provider_id = cfg.active_provider_id or provider_id
        cfg.active_model_id = cfg.active_model_id or new_model.id
    _stamp(provider)
    save_app_config(cfg)
    return new_model


@router.patch("/{provider_id}/models/{model_id}", response_model=ModelEntryOut)
def update_model(provider_id: str, model_id: str, payload: ModelEntryUpdate):
    cfg, provider = _provider_or_404(provider_id)
    model = _model_or_404(provider, model_id)
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        if hasattr(model, field):
            setattr(model, field, value)
    model.updated_at = utcnow_iso()
    if payload.is_default is True:
        _ensure_single_default(cfg, provider_id, model_id)
    _stamp(provider)
    save_app_config(cfg)
    return model


@router.delete("/{provider_id}/models/{model_id}")
def delete_model(provider_id: str, model_id: str):
    cfg, provider = _provider_or_404(provider_id)
    before = len(provider.models)
    provider.models = [m for m in provider.models if m.id != model_id]
    if len(provider.models) == before:
        raise HTTPException(status_code=404, detail="model_not_found")
    if cfg.active_provider_id == provider_id and cfg.active_model_id == model_id:
        cfg.active_model_id = (
            provider.default_model().id if provider.default_model() else None
        )
    _stamp(provider)
    save_app_config(cfg)
    return {"ok": True, "deleted_model_id": model_id}


@router.post("/{provider_id}/models/{model_id}/activate")
def activate_model(provider_id: str, model_id: str):
    cfg, provider = _provider_or_404(provider_id)
    model = _model_or_404(provider, model_id)
    if not provider.enabled:
        raise HTTPException(status_code=400, detail="provider_disabled")
    if not model.enabled:
        raise HTTPException(status_code=400, detail="model_disabled")
    cfg.active_provider_id = provider_id
    cfg.active_model_id = model_id
    _ensure_single_default(cfg, provider_id, model_id)
    _stamp(provider)
    save_app_config(cfg)
    return {
        "ok": True,
        "active_provider_id": cfg.active_provider_id,
        "active_model_id": cfg.active_model_id,
    }


@router.post("/active")
def set_active(payload: ActiveSelectionUpdate):
    cfg = load_app_config()
    if payload.provider_id is not None:
        provider = next((p for p in cfg.providers if p.id == payload.provider_id), None)
        if provider is None:
            raise HTTPException(status_code=404, detail="provider_not_found")
        cfg.active_provider_id = payload.provider_id
        if payload.model_id:
            model = provider.find_model(payload.model_id)
            if model is None:
                raise HTTPException(status_code=404, detail="model_not_found")
            cfg.active_model_id = payload.model_id
        else:
            default = provider.default_model()
            cfg.active_model_id = default.id if default else None
    save_app_config(cfg)
    return {
        "ok": True,
        "active_provider_id": cfg.active_provider_id,
        "active_model_id": cfg.active_model_id,
    }