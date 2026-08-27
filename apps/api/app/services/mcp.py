from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from .schema_cache import default_cache as _schema_cache
from .schema_sanitizer import (
    sanitize_for_send as _sanitize_for_send,
    sanitize_schema as _sanitize_schema,
)
from .telemetry import (
    mcp_schema_cache_ops as _mcp_cache_ops,
    mcp_tool_calls_total as _mcp_calls_total,
    mcp_tool_latency_seconds as _mcp_latency,
    mcp_tool_retries_total as _mcp_retries,
)


class MCPError(RuntimeError):
    pass


def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return slug or fallback


def _truncate_name(name: str, max_len: int = 64) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len]


def _is_timeout_error(exc: Exception) -> bool:
    return "Timeout waiting for MCP response" in str(exc)


@dataclass
class MCPTool:
    namespaced_name: str
    server_name: str
    server_slug: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPServerSpec:
    name: str
    transport: str
    message_mode: str
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str | None
    startup_timeout_sec: int
    request_timeout_sec: int


def parse_mcp_server_spec(raw: dict[str, Any]) -> MCPServerSpec:
    name = str(raw.get("name") or "").strip()
    command = str(raw.get("command") or "").strip()
    if not name:
        raise MCPError("MCP server config requires non-empty 'name'")
    if not command:
        raise MCPError(f"MCP server '{name}' requires non-empty 'command'")

    transport = str(raw.get("transport") or "stdio").strip().lower()
    if transport != "stdio":
        raise MCPError(f"MCP server '{name}' has unsupported transport '{transport}', only 'stdio' is supported")
    message_mode = str(raw.get("message_mode") or raw.get("framing") or "line").strip().lower()
    if message_mode not in {"line", "header"}:
        raise MCPError(f"MCP server '{name}' has unsupported message_mode '{message_mode}'")

    raw_args = raw.get("args", [])
    args: list[str]
    if isinstance(raw_args, list):
        args = [str(v) for v in raw_args]
    elif isinstance(raw_args, str):
        args = [raw_args]
    else:
        args = []

    raw_env = raw.get("env", {})
    env: dict[str, str] = {}
    if isinstance(raw_env, dict):
        env = {str(k): str(v) for k, v in raw_env.items()}

    cwd_raw = raw.get("cwd")
    cwd = str(cwd_raw).strip() if isinstance(cwd_raw, str) and cwd_raw.strip() else None

    startup_timeout_sec = int(raw.get("startup_timeout_sec", 12))
    request_timeout_sec = int(raw.get("request_timeout_sec", 25))

    startup_timeout_sec = max(1, min(startup_timeout_sec, 180))
    request_timeout_sec = max(1, min(request_timeout_sec, 600))

    return MCPServerSpec(
        name=name,
        transport=transport,
        message_mode=message_mode,
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        startup_timeout_sec=startup_timeout_sec,
        request_timeout_sec=request_timeout_sec,
    )


class StdioMCPClient:
    def __init__(self, spec: MCPServerSpec):
        self.spec = spec
        self.proc: asyncio.subprocess.Process | None = None
        self._request_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: list[str] = []

    async def start(self) -> None:
        if self.proc is not None:
            return
        env = None
        if self.spec.env:
            env = {**os.environ, **self.spec.env}
        self.proc = await asyncio.create_subprocess_exec(
            self.spec.command,
            *self.spec.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.spec.cwd,
            env=env,
        )
        if self.proc.stdout is None or self.proc.stdin is None:
            raise MCPError(f"Failed to start MCP server '{self.spec.name}'")

        if self.spec.message_mode == "header":
            self._reader_task = asyncio.create_task(self._reader_loop_header())
        else:
            self._reader_task = asyncio.create_task(self._reader_loop_line())
        if self.proc.stderr is not None:
            self._stderr_task = asyncio.create_task(self._stderr_loop())

        try:
            await asyncio.wait_for(self._initialize(), timeout=self.spec.startup_timeout_sec)
        except Exception as exc:
            await self.close()
            raise MCPError(f"MCP initialize failed for '{self.spec.name}': {exc}") from exc

    async def _initialize(self) -> None:
        last_error: Exception | None = None
        for version in ("2024-11-05", "2024-10-07", "2024-09-30"):
            try:
                await self.request(
                    "initialize",
                    {
                        "protocolVersion": version,
                        "capabilities": {},
                        "clientInfo": {"name": "ai-infinite-session", "version": "0.1.0"},
                    },
                )
                await self.notify("notifications/initialized", {})
                return
            except Exception as exc:
                last_error = exc
        raise MCPError(f"initialize failed for all protocol versions: {last_error}")

    async def _stderr_loop(self) -> None:
        if not self.proc or not self.proc.stderr:
            return
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            self._stderr_lines.append(text)
            if len(self._stderr_lines) > 200:
                self._stderr_lines = self._stderr_lines[-200:]

    async def _reader_loop_line(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        stream = self.proc.stdout
        while True:
            try:
                line = await stream.readline()
            except Exception:
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if not text.startswith("{"):
                # Ignore non-protocol stdout noise
                continue
            try:
                message = json.loads(text)
            except Exception:
                continue
            if not isinstance(message, dict):
                continue
            msg_id = message.get("id")
            if isinstance(msg_id, int) and msg_id in self._pending:
                fut = self._pending.pop(msg_id)
                if not fut.done():
                    fut.set_result(message)

    async def _reader_loop_header(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        stream = self.proc.stdout
        while True:
            try:
                header_bytes = await stream.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError:
                break
            except Exception:
                break

            try:
                headers = header_bytes.decode("utf-8", errors="replace").split("\r\n")
                content_length = 0
                for line in headers:
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":", 1)[1].strip())
                        break
                if content_length <= 0:
                    continue
                body = await stream.readexactly(content_length)
                message = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                continue

            if not isinstance(message, dict):
                continue
            msg_id = message.get("id")
            if isinstance(msg_id, int) and msg_id in self._pending:
                fut = self._pending.pop(msg_id)
                if not fut.done():
                    fut.set_result(message)

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            raise MCPError(f"MCP server '{self.spec.name}' is not started")
        if self.spec.message_mode == "line":
            body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.proc.stdin.write(body)
            await self.proc.stdin.drain()
            return
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\nContent-Type: application/json\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + body)
        await self.proc.stdin.drain()

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = self._request_id
        self._request_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
        )
        try:
            response = await asyncio.wait_for(fut, timeout=self.spec.request_timeout_sec)
        except Exception as exc:
            self._pending.pop(req_id, None)
            raise MCPError(f"Timeout waiting for MCP response ({self.spec.name}:{method})") from exc
        if "error" in response:
            err = response.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise MCPError(f"MCP error from '{self.spec.name}' {method}: {msg}")
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self.request("tools/list", {})
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        return result

    def stderr_tail(self) -> list[str]:
        return self._stderr_lines[-20:]

    async def close(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

        if self._reader_task:
            self._reader_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()

        proc = self.proc
        self.proc = None
        if not proc:
            return
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                proc.kill()
                await proc.wait()


def _normalize_mcp_result(result: Any) -> dict[str, Any]:
    """Wrap the raw MCP response in the orchestrator's uniform shape.

    MCP servers return either ``{"result": ...}``, an ``{"error": ...}``
    envelope, or a plain dict. Downstream code (and the tool validator)
    rely on a consistent ``{"ok": bool, ...}`` shape so the error path
    can always be detected with a single ``result.get("ok")`` check.
    """
    if not isinstance(result, dict):
        return {"ok": False, "error": f"MCP returned non-dict: {type(result).__name__}"}
    if "error" in result and isinstance(result["error"], (dict, str)):
        err = result["error"]
        if isinstance(err, dict):
            message = str(err.get("message") or err.get("code") or err)
            code = err.get("code")
        else:
            message = str(err)
            code = None
        out = {"ok": False, "error": message, "raw_error": err}
        if code is not None:
            out["error_code"] = code
        return out
    if "ok" not in result:
        out = dict(result)
        # If the server returned an ``isError`` field, mirror it to ``ok``.
        if "isError" in out:
            out["ok"] = not bool(out["isError"])
        else:
            out.setdefault("ok", True)
        return out
    return result


class MCPToolRegistry:
    def __init__(self):
        self.clients: dict[str, StdioMCPClient] = {}
        self.tools_by_name: dict[str, MCPTool] = {}
        self.discovery_errors: list[dict[str, str]] = []

    @classmethod
    async def from_server_configs(cls, server_configs: list[dict[str, Any]]) -> MCPToolRegistry:
        registry = cls()
        for raw in server_configs:
            try:
                spec = parse_mcp_server_spec(raw)
                client = StdioMCPClient(spec)
                await client.start()
                registry.clients[spec.name] = client
                await registry._register_tools(spec, client)
            except Exception as exc:
                name = str(raw.get("name") or raw.get("command") or "unknown")
                registry.discovery_errors.append({"server": name, "error": str(exc)})
        return registry

    async def _register_tools(self, spec: MCPServerSpec, client: StdioMCPClient) -> None:
        raw_tools = await client.list_tools()
        server_slug = _slugify(spec.name, "server")
        # Drop any cached schemas for this server so a re-registration
        # (e.g. after a restart) does not return stale entries.
        _schema_cache.invalidate(server_slug)
        _mcp_cache_ops.with_labels(op="invalidate").inc()
        used_names = set(self.tools_by_name.keys())

        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                continue
            tool_name = str(raw_tool.get("name") or "").strip()
            if not tool_name:
                continue
            tool_slug = _slugify(tool_name, "tool")
            base_name = _truncate_name(f"mcp__{server_slug}__{tool_slug}")
            namespaced_name = base_name
            suffix = 2
            while namespaced_name in used_names:
                namespaced_name = _truncate_name(f"{base_name}_{suffix}")
                suffix += 1
            used_names.add(namespaced_name)

            description = str(raw_tool.get("description") or "").strip()
            input_schema = raw_tool.get("inputSchema")
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object", "properties": {}}

            self.tools_by_name[namespaced_name] = MCPTool(
                namespaced_name=namespaced_name,
                server_name=spec.name,
                server_slug=server_slug,
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
            )

    def has_tools(self) -> bool:
        return bool(self.tools_by_name)

    def has_tool(self, namespaced_name: str) -> bool:
        return namespaced_name in self.tools_by_name

    def resolve_tool_name(self, requested_name: str) -> str | None:
        if requested_name in self.tools_by_name:
            return requested_name
        if not requested_name:
            return None

        requested_tail = requested_name.removeprefix("mcp__")
        requested_single = requested_tail.replace("__", "_")
        requested_compact = requested_tail.replace("_", "")

        for existing in self.tools_by_name.keys():
            existing_tail = existing.removeprefix("mcp__")
            if requested_single == existing_tail.replace("__", "_"):
                return existing
            if requested_compact and requested_compact == existing_tail.replace("_", ""):
                return existing
        return None

    def tool_schemas(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for tool in self.tools_by_name.values():
            # Sanitise through the LRU cache so repeated turns do not
            # re-normalise the same schema, and so MCP-emitted constructs
            # the local model cannot follow (oneOf/$ref/huge enums) are
            # pruned before the schema reaches the prompt.
            parameters = _schema_cache.get_or_compute(
                server_slug=tool.server_slug,
                tool_name=tool.tool_name,
                raw_schema=tool.input_schema,
            )
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.namespaced_name,
                        "description": (
                            f"MCP server '{tool.server_name}' tool '{tool.tool_name}'. "
                            f"{tool.description}".strip()
                        ),
                        "parameters": parameters,
                    },
                }
            )
        return out

    def prompt_tool_lines(self) -> list[str]:
        lines: list[str] = []
        for tool in self.tools_by_name.values():
            desc = tool.description or "no description"
            lines.append(f"- {tool.namespaced_name}: {desc}")
        return lines

    def list_tools_for_api(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for tool in self.tools_by_name.values():
            items.append(
                {
                    "name": tool.namespaced_name,
                    "server_name": tool.server_name,
                    "mcp_tool_name": tool.tool_name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
        return items


    async def _resolve_delegation(self, result: dict[str, Any], depth: int = 0) -> dict[str, Any]:
        """Recursively resolve DELEGATE:/TOOL_ARGS: directives in MCP results.

        The skills-mcp wrapper can return a content block whose text starts
        with ``DELEGATE:{...}`` to indicate that the skill is a thin wrapper
        around another MCP tool.  In that case we parse the JSON payload,
        call the named tool, and substitute the result.  ``TOOL_ARGS:`` is
        treated as a literal arg bag that should be returned to the
        orchestrator (rarely used).  ``depth`` prevents infinite loops if
        a delegated tool itself returns a DELEGATE directive.
        """
        if depth >= 2:
            return result  # guard against loops
        if not isinstance(result, dict):
            return result
        content = result.get("content")
        if not isinstance(content, list) or not content:
            return result
        # Look for a text block starting with DELEGATE: or TOOL_ARGS:.
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            if text.startswith("DELEGATE:"):
                import json as _json
                try:
                    payload = _json.loads(text[len("DELEGATE:"):])
                except Exception:
                    return result
                target_tool = payload.get("tool")
                target_args = payload.get("args") or {}
                if not target_tool:
                    return result
                # Recurse - this is what makes a delegating skill actually
                # execute the underlying MCP tool.
                delegated = await self.call_tool(target_tool, target_args)
                return await self._resolve_delegation(delegated, depth + 1)
            if text.startswith("TOOL_ARGS:"):
                import json as _json
                try:
                    payload = _json.loads(text[len("TOOL_ARGS:"):])
                except Exception:
                    return result
                # Surface the tool_args as the "content" of the result so
                # downstream code can use it as if the tool had been called.
                return {"ok": True, "content": payload}
        return result

    async def call_tool(self, namespaced_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resolved_name = self.resolve_tool_name(namespaced_name)
        if not resolved_name:
            raise MCPError(f"Unknown MCP tool '{namespaced_name}'")
        tool = self.tools_by_name.get(resolved_name)
        if not tool:
            raise MCPError(f"Unknown MCP tool '{namespaced_name}'")
        client = self.clients.get(tool.server_name)
        if not client:
            raise MCPError(f"MCP server '{tool.server_name}' is unavailable")

        # Pre-filter unknown properties and coerce scalar types so the MCP
        # server sees a payload it is happy with. Uses the *raw* schema
        # (not the sanitised prompt version) because the server is the
        # authority on what is acceptable.
        send_arguments = _sanitize_for_send(arguments, tool.input_schema)
        max_attempts = 3 if tool.server_name == "native-web-search" else 1
        last_error: Exception | None = None
        _mcp_started = time.perf_counter()
        _mcp_outcome = "ok"
        for attempt in range(max_attempts):
            try:
                raw_result = await client.call_tool(tool.tool_name, send_arguments)
                # Resolve DELEGATE: directives recursively (skills-mcp uses
                # this to wrap other MCP tools like native-web-search).
                raw_result = await self._resolve_delegation(raw_result)
                result = _normalize_mcp_result(raw_result)
                _mcp_outcome = "ok" if result.get("ok", True) else "error"
                _mcp_latency.with_labels(server=tool.server_name, tool=tool.tool_name).observe(time.perf_counter() - _mcp_started)
                _mcp_calls_total.with_labels(server=tool.server_name, tool=tool.tool_name, outcome=_mcp_outcome).inc()
                return result
            except Exception as exc:
                last_error = exc
                should_retry = attempt + 1 < max_attempts and _is_timeout_error(exc)
                if not should_retry:
                    _mcp_outcome = "error"
                    _mcp_latency.with_labels(server=tool.server_name, tool=tool.tool_name).observe(time.perf_counter() - _mcp_started)
                    _mcp_calls_total.with_labels(server=tool.server_name, tool=tool.tool_name, outcome="error").inc()
                    if tool.server_name == "native-web-search" and _is_timeout_error(exc):
                        timeout_result = _normalize_mcp_result(
                            {
                                "ok": False,
                                "error": str(exc),
                                "status": "timeout",
                                "results": [],
                                "engine": "None",
                            }
                        )
                        _mcp_outcome = "ok"  # The MCP layer reported a structured timeout; not a hard error.
                        _mcp_latency.with_labels(server=tool.server_name, tool=tool.tool_name).observe(time.perf_counter() - _mcp_started)
                        _mcp_calls_total.with_labels(server=tool.server_name, tool=tool.tool_name, outcome="timeout").inc()
                        return timeout_result
                    raise
                _mcp_retries.with_labels(server=tool.server_name, tool=tool.tool_name).inc()

                # For transient MCP stalls/timeouts, restart native server and retry once.
                try:
                    await client.close()
                except Exception:
                    pass
                try:
                    restarted = StdioMCPClient(client.spec)
                    await restarted.start()
                    self.clients[tool.server_name] = restarted
                    client = restarted
                except Exception as restart_exc:
                    raise MCPError(
                        f"{exc}; retry restart failed for MCP server '{tool.server_name}': {restart_exc}"
                    ) from restart_exc

        _mcp_latency.with_labels(server=tool.server_name, tool=tool.tool_name).observe(time.perf_counter() - _mcp_started)
        _mcp_calls_total.with_labels(server=tool.server_name, tool=tool.tool_name, outcome="error").inc()
        _mcp_outcome = "error"
        if last_error:
            raise last_error
        raise MCPError(f"MCP server '{tool.server_name}' failed to return tool result")

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients.values()), return_exceptions=True)
        self.clients.clear()
