from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import AppConfig, effective_mcp_servers, load_app_config, save_app_config
from ..db import utcnow_iso
from ..schemas import (
    MCPDiscoveryResponse,
    ProviderValidationRequest,
    ProviderValidationResponse,
)
from ..services.auto_search import run_auto_search, should_search
from ..services.mcp import MCPToolRegistry

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=AppConfig)
def get_config():
    return load_app_config()


@router.put("", response_model=AppConfig)
def update_config(payload: AppConfig):
    save_app_config(payload)
    return payload


@router.post("/validate-provider", response_model=ProviderValidationResponse)
async def validate_provider(payload: ProviderValidationRequest):
    """Hit ``base_url + endpoint`` (defaults to ``/models`` for an OpenAI
    compatible server) and parse the response.  When the endpoint is the
    models index we return the discovered model identifiers so the
    ``Add Provider`` UI can pre-fill them."""

    endpoint = payload.endpoint or "/models"
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    url = payload.base_url.rstrip("/") + endpoint
    headers: dict[str, str] = {}
    if payload.api_key:
        headers["Authorization"] = f"Bearer {payload.api_key}"
    timeout_sec = max(5, min(int(payload.timeout_sec or 15), 600))

    try:
        async with httpx.AsyncClient(timeout=float(timeout_sec)) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code >= 400:
                return {"ok": False, "detail": f"HTTP {resp.status_code}", "models": []}
            # If we hit a /models endpoint, parse the OpenAI-compatible
            # ``{"data": [{"id": ...}, ...]}`` shape.  Otherwise the call
            # was a liveness check and we have no model list.
            models: list[str] = []
            if endpoint.endswith("/models"):
                try:
                    payload_json = resp.json()
                    data = payload_json.get("data") if isinstance(payload_json, dict) else None
                    if isinstance(data, list):
                        for entry in data:
                            if isinstance(entry, dict):
                                mid = entry.get("id")
                                if isinstance(mid, str) and mid:
                                    models.append(mid)
                except Exception:
                    pass
            return {
                "ok": True,
                "detail": "Provider reachable",
                "models": models,
            }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/mcp/discover", response_model=MCPDiscoveryResponse)
async def discover_mcp_tools():
    cfg = load_app_config()
    if not cfg.mcp_config.enabled:
        return {"ok": True, "tools": [], "errors": [{"server": "mcp", "error": "MCP disabled in settings"}]}
    servers = effective_mcp_servers(cfg)
    if not servers:
        return {"ok": True, "tools": [], "errors": []}

    registry = await MCPToolRegistry.from_server_configs(servers)
    try:
        return {
            "ok": len(registry.discovery_errors) == 0,
            "tools": registry.list_tools_for_api(),
            "errors": registry.discovery_errors,
        }
    finally:
        await registry.close()


class AutoSearchTestRequest(BaseModel):
    query: str
    force: bool = False
    bypass_cache: bool = False


class AutoSearchTestResponse(BaseModel):
    decision: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)


class ContextModeRequest(BaseModel):
    context_mode: str = Field(..., pattern="^(full|skill_state)$")


class ContextModeResponse(BaseModel):
    context_mode: str
    auto: bool


@router.post("/auto-search/test", response_model=AutoSearchTestResponse)
async def test_auto_search(payload: AutoSearchTestRequest):
    """Dry-run the auto-search router for a sample query.

    The response always carries the heuristic decision so the UI can
    explain *why* a search fired (or didn't).  When the decision is to
    search we also execute the router end-to-end — useful as a manual
    smoke test from the Settings page.
    """

    cfg = load_app_config()
    auto_cfg = cfg.mcp_config.auto_search
    decision = should_search(
        payload.query,
        policy=auto_cfg.policy,
        enabled=auto_cfg.enabled,
        force=payload.force,
        freshness_hints=auto_cfg.freshness_hints or None,
        factual_hints=auto_cfg.factual_hints or None,
        opinion_hints=auto_cfg.opinion_hints or None,
    )
    if not decision.should_search:
        return AutoSearchTestResponse(
            decision={
                "should_search": decision.should_search,
                "reason": decision.reason,
                "policy": decision.policy,
                "query": decision.query,
                "normalized_query": decision.normalized_query,
            },
            result={},
        )

    result = await run_auto_search(
        payload.query,
        cfg=cfg,
        force=payload.force,
        bypass_cache=payload.bypass_cache,
    )
    return AutoSearchTestResponse(
        decision={
            "should_search": decision.should_search,
            "reason": decision.reason,
            "policy": decision.policy,
            "query": decision.query,
            "normalized_query": decision.normalized_query,
        },
        result=result.to_dict(),
    )


@router.get("/context-mode", response_model=ContextModeResponse)
def get_context_mode() -> ContextModeResponse:
    """Return the global ``context_mode`` selector. The value is the
    ``full`` (default) or ``skill_state`` literal defined by
    ``AppConfig.context_mode``. New sessions inherit this value unless
    the caller overrides it via ``SessionCreate.context_mode``."""
    cfg = load_app_config()
    return ContextModeResponse(context_mode=cfg.context_mode, auto=True)


@router.put("/context-mode", response_model=ContextModeResponse)
def set_context_mode(payload: ContextModeRequest) -> ContextModeResponse:
    """Persist the global context-mode selector. ``skill_state`` opts
    into the SKILL.state runtime (arXiv:2608.26263): the orchestrator
    swaps the append-only chat history for the (spec, state,
    observation) bundle whenever a registered skill matches the user
    prompt. ``full`` keeps the legacy behaviour — chat history is
    always replayed."""
    cfg = load_app_config()
    cfg.context_mode = payload.context_mode
    save_app_config(cfg)
    return ContextModeResponse(context_mode=cfg.context_mode, auto=True)