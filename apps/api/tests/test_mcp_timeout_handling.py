from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import effective_mcp_servers, load_app_config
from app.main import create_app
from app.services.mcp import MCPError
from app.services.orchestrator import _execute_tool_call, _wants_tools
from app.services.sessions import get_session


def test_effective_native_web_search_timeouts_are_propagated(isolated_data_dir):
    cfg = load_app_config()
    cfg.mcp_config.enabled = True
    cfg.mcp_config.native_web_search_enabled = True
    cfg.mcp_config.native_web_search_timeout_sec = 70

    servers = effective_mcp_servers(cfg)
    native = next((s for s in servers if s.get("name") == "native-web-search"), None)
    assert native is not None

    assert native["request_timeout_sec"] == 90
    assert native["startup_timeout_sec"] == 70
    assert native["env"]["DEFAULT_TIMEOUT"] == "70000"
    assert native["env"]["SEARCH_TIMEOUT_MS"] == "70000"


def test_native_web_search_path_resolves_from_api_cwd(isolated_data_dir):
    cfg = load_app_config()
    cfg.mcp_config.enabled = True
    cfg.mcp_config.native_web_search_enabled = True
    cfg.mcp_config.native_web_search_path = "mcp/web-search-mcp/codex-wrapper.mjs"

    native = next(s for s in effective_mcp_servers(cfg) if s.get("name") == "native-web-search")
    assert Path(native["args"][0]).is_file()


def test_execute_tool_call_returns_error_instead_of_raising_on_mcp_timeout(isolated_data_dir):
    app = create_app()
    client = TestClient(app)

    created = client.post("/api/sessions", json={"title": "MCP Timeout Session", "workspace_path": "/tmp/workspace"})
    assert created.status_code == 200
    session_id = created.json()["id"]
    session_info = get_session(session_id)

    class FakeMCPRegistry:
        def resolve_tool_name(self, requested_name: str) -> str | None:
            return requested_name

        async def call_tool(self, namespaced_name: str, arguments: dict[str, object]) -> dict[str, object]:
            raise MCPError("Timeout waiting for MCP response (native-web-search:tools/call)")

    result = asyncio.run(
        _execute_tool_call(
            fn_name="mcp__native_web_search__full_web_search",
            args={"query": "test"},
            session_id=session_id,
            session_info=session_info,
            mcp_registry=FakeMCPRegistry(),  # type: ignore[arg-type]
        )
    )

    assert result["ok"] is False
    assert result["retryable"] is True
    assert "Timeout waiting for MCP response" in str(result["error"])
