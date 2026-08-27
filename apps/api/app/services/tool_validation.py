"""JSON-Schema validation for tool-call arguments.

The orchestrator asks a local language model to emit ``tool_calls`` whose
arguments must conform to the JSON Schema bundled with each tool. The previous
behaviour was to forward whatever the model produced to ``execute_tool_call``,
which meant that a typo (``"qurey"`` instead of ``"query"``), a wrong type
(``42`` instead of a string), or a missing required field crashed the tool
and produced a confusing ``tool error`` reply in the next turn.

This module validates the arguments *before* the tool is invoked and produces
a short, structured error message that the orchestrator can inject into the
chat as a ``tool`` role message. The model then sees its own mistake and
typically corrects it on the next iteration of the tool loop.

Only the small subset of JSON Schema used by the project's own tools and
MCP servers is supported. Anything more exotic (oneOf, $ref, allOf) is
deliberately not handled: the validation must be cheap enough to run on every
tool call, and the project does not currently generate schemas of that
complexity. Unknown keywords are silently ignored, which matches the spirit
of JSON Schema's "be liberal in what you accept" default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationError:
    """A single argument-level validation failure."""

    path: str  # e.g. "query", "options.limit", "items[0].url"
    message: str  # human-readable, e.g. "expected string, got integer"

    def render(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message


# ---------------------------------------------------------------------------
# Schema lookup
# ---------------------------------------------------------------------------


def _tool_name_from_schema(schema: dict[str, Any]) -> str | None:
    function = schema.get("function") or {}
    name = function.get("name")
    return str(name) if isinstance(name, str) and name else None


def find_tool_schema(
    tools_schema: list[dict[str, Any]] | None,
    fn_name: str,
) -> dict[str, Any] | None:
    """Locate the JSON Schema for ``fn_name`` in the registered tool list.

    Returns ``None`` when the tool is not registered. The orchestrator treats
    that as a non-fatal warning (the tool call still proceeds) so we do not
    block model-emitted tools that come from dynamic MCP servers whose schema
    has not been registered with the active request.
    """
    if not tools_schema or not fn_name:
        return None
    # Exact match first.
    for schema in tools_schema:
        if _tool_name_from_schema(schema) == fn_name:
            return schema
    # Fuzzy match by suffix (e.g. "mcp__native_web_search__get_web_search_summary"
    # should match a schema whose function name is "get_web_search_summary").
    for schema in tools_schema:
        registered = _tool_name_from_schema(schema) or ""
        if fn_name.endswith("__" + registered) or fn_name.endswith("." + registered):
            return schema
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _expected_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# Map JSON Schema "type" to a predicate that returns True for a match.
_TYPE_PREDICATES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _check_type(value: Any, declared: str, path: str, errors: list[ValidationError]) -> None:
    predicate = _TYPE_PREDICATES.get(declared)
    if predicate is None:
        # Unknown type, JSON Schema says we should still validate.
        return
    if not predicate(value):
        errors.append(
            ValidationError(
                path=path,
                message=f"expected {declared}, got {_expected_type(value)}",
            )
        )


def _check_enum(value: Any, enum: list[Any], path: str, errors: list[ValidationError]) -> None:
    if not enum:
        return
    if value not in enum:
        rendered = ", ".join(repr(x) for x in enum[:5])
        if len(enum) > 5:
            rendered += ", ..."
        errors.append(
            ValidationError(
                path=path,
                message=f"value {_expected_type(value)} not in enum [{rendered}]",
            )
        )


def _check_string_bounds(
    value: str,
    schema: dict[str, Any],
    path: str,
    errors: list[ValidationError],
) -> None:
    min_len = schema.get("minLength")
    max_len = schema.get("maxLength")
    if isinstance(min_len, int) and len(value) < min_len:
        errors.append(ValidationError(path=path, message=f"string shorter than minLength={min_len}"))
    if isinstance(max_len, int) and len(value) > max_len:
        errors.append(ValidationError(path=path, message=f"string longer than maxLength={max_len}"))


def _check_number_bounds(
    value: float,
    schema: dict[str, Any],
    path: str,
    errors: list[ValidationError],
) -> None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        if value < minimum:
            errors.append(ValidationError(path=path, message=f"value {value} < minimum={minimum}"))
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        if value > maximum:
            errors.append(ValidationError(path=path, message=f"value {value} > maximum={maximum}"))


def _validate_value(
    value: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[ValidationError],
) -> None:
    declared_type = schema.get("type")
    if isinstance(declared_type, str):
        _check_type(value, declared_type, path, errors)
        # If type failed, skip further checks to avoid noisy cascades.
        if errors and errors[-1].path == path and "expected" in errors[-1].message:
            return
    if "enum" in schema and isinstance(schema["enum"], list):
        _check_enum(value, schema["enum"], path, errors)
    if isinstance(value, str):
        _check_string_bounds(value, schema, path, errors)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _check_number_bounds(float(value), schema, path, errors)
    if declared_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{i}]", errors)
    if declared_type == "object" and isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        for key, sub_schema in schema["properties"].items():
            if key in value and isinstance(sub_schema, dict):
                _validate_value(value[key], sub_schema, f"{path}.{key}" if path else key, errors)
        # additionalProperties: false -> no extra keys.
        if schema.get("additionalProperties") is False:
            allowed = set(schema["properties"].keys())
            for key in value.keys():
                if key not in allowed:
                    errors.append(
                        ValidationError(
                            path=f"{path}.{key}" if path else key,
                            message="unknown property (additionalProperties=false)",
                        )
                    )


def validate_tool_args(
    fn_name: str,
    args: Any,
    schema: dict[str, Any] | None,
) -> list[ValidationError]:
    """Return the list of validation errors for ``args`` against ``schema``.

    A non-dict ``args`` is reported as a single ``expected object`` error.
    A missing schema returns an empty list (caller decides what to do).
    """
    if schema is None:
        return []
    params = schema.get("function", {}).get("parameters") or schema.get("parameters") or {}
    if not isinstance(params, dict):
        return []
    if not isinstance(args, dict):
        return [ValidationError(path="", message=f"expected object, got {_expected_type(args)}")]

    errors: list[ValidationError] = []
    required = params.get("required") or []
    if isinstance(required, list):
        for key in required:
            if key not in args:
                errors.append(ValidationError(path=key, message="required property is missing"))

    properties = params.get("properties") or {}
    if isinstance(properties, dict):
        for key, sub_schema in properties.items():
            if key in args and isinstance(sub_schema, dict):
                _validate_value(args[key], sub_schema, key, errors)

    # Reject unknown top-level keys if additionalProperties is false.
    if params.get("additionalProperties") is False and isinstance(properties, dict):
        allowed = set(properties.keys())
        for key in args.keys():
            if key not in allowed:
                errors.append(ValidationError(path=key, message="unknown property (additionalProperties=false)"))

    return errors


# ---------------------------------------------------------------------------
# Error rendering
# ---------------------------------------------------------------------------


def format_validation_errors(
    fn_name: str,
    errors: list[ValidationError],
    *,
    received_args: Any | None = None,
) -> str:
    """Build a short, model-friendly error message.

    The text is intentionally compact: it is appended to a ``tool`` role
    message that the model will read on the next iteration of the tool loop.
    Verbose Python tracebacks would just burn context and confuse the model.
    """
    lines = [f"Validation failed for tool '{fn_name}'."]
    if received_args is not None:
        try:
            args_repr = json.dumps(received_args, ensure_ascii=False)
        except (TypeError, ValueError):
            args_repr = repr(received_args)
        if len(args_repr) > 400:
            args_repr = args_repr[:400] + "..."
        lines.append(f"Received arguments: {args_repr}")
    if errors:
        lines.append(f"{len(errors)} error(s):")
        for err in errors[:8]:
            lines.append(f"  - {err.render()}")
        if len(errors) > 8:
            lines.append(f"  - ... and {len(errors) - 8} more")
    lines.append("Please fix the arguments and call the tool again.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: one-shot validate-and-format
# ---------------------------------------------------------------------------


def validate_and_format(
    fn_name: str,
    args: Any,
    tools_schema: list[dict[str, Any]] | None,
) -> tuple[bool, str, list[ValidationError]]:
    """Validate ``args`` for ``fn_name`` and return (ok, error_message, errors).

    When the tool has no registered schema, the call is considered valid and
    ``ok`` is True. This preserves the current behaviour for dynamic MCP
    tools whose schemas are not always present in the active request.
    """
    schema = find_tool_schema(tools_schema, fn_name)
    if schema is None:
        return True, "", []
    errors = validate_tool_args(fn_name, args, schema)
    if not errors:
        return True, "", []
    return False, format_validation_errors(fn_name, errors, received_args=args), errors


# ---------------------------------------------------------------------------
# Terminal-command guard (hallucinated utilities)
# ---------------------------------------------------------------------------

# Whitelist of utilities the small local models are known to call
# correctly. Anything outside this list is rejected with a structured
# error pointing the model at the canonical web-search MCP tool, so
# we don't burn turns on shell errors like ``search: command not
# found``.
_ALLOWED_TERMINAL_COMMANDS: frozenset[str] = frozenset(
    {
        # Inspection
        "ls", "dir", "pwd", "echo", "cat", "head", "tail", "wc",
        "find", "tree", "stat", "file", "which", "command",
        # Text processing
        "grep", "rg", "ag", "awk", "sed", "sort", "uniq", "cut",
        "tr", "xargs", "tee", "less", "more",
        # Filesystem
        "cp", "mv", "mkdir", "rm", "touch", "ln", "chmod", "du", "df",
        # Network (read-only, no shell injection)
        "curl", "wget", "ping", "nslookup", "dig", "host", "traceroute",
        # Build / package tooling
        "git", "make", "cmake", "ninja", "go", "cargo", "rustc",
        "npm", "pnpm", "yarn", "node", "python", "python3", "pip",
        "pip3", "pytest", "uv", "poetry", "pipx",
        "docker", "docker-compose", "kubectl", "terraform", "ansible",
        # System
        "ps", "top", "htop", "kill", "uptime", "uname", "whoami",
        "id", "env", "printenv", "date", "cal", "tar", "gzip", "gunzip",
        "zip", "unzip", "jq", "yq", "xq", "base64", "sha256sum", "md5sum",
        # Common text editors (read-only mode)
        "vi", "vim", "nano",
    }
)

# Token-prefixes (commands) that small local models sometimes invent.
# We map each hallucination to the canonical MCP tool the model
# should use instead. When the validator rejects a hallucinated
# command it returns one of these messages so the next iteration
# of the tool loop can self-correct.
_HALLUCINATED_TERMINAL_COMMANDS: dict[str, str] = {
    "search": "Unknown shell command 'search'. Use the web-search MCP tool (mcp__native_web_search__full_web_search or mcp__native_web_search__get_web_search_summaries) instead of spawning a 'search' utility.",
    "google": "Unknown shell command 'google'. Use the web-search MCP tool (mcp__native_web_search__full_web_search) instead of a 'google' CLI.",
    "bing": "Unknown shell command 'bing'. Use the web-search MCP tool (mcp__native_web_search__full_web_search) instead of a 'bing' CLI.",
    "duckduckgo": "Unknown shell command 'duckduckgo'. Use the web-search MCP tool (mcp__native_web_search__full_web_search) instead of a 'duckduckgo' CLI.",
    "ask": "Unknown shell command 'ask'. Use a regular LLM prompt instead of an 'ask' CLI.",
    "chat": "Unknown shell command 'chat'. Use a regular LLM prompt instead of a 'chat' CLI.",
    "llm": "Unknown shell command 'llm'. Use a regular LLM prompt instead of an 'llm' CLI.",
    "web": "Unknown shell command 'web'. Use the web-search MCP tool (mcp__native_web_search__full_web_search) instead of a 'web' CLI.",
    "scrape": "Unknown shell command 'scrape'. Use the web-search MCP tool (mcp__native_web_search__fetch_url) instead of a 'scrape' CLI.",
    "fetch": "Unknown shell command 'fetch'. Use the web-search MCP tool (mcp__native_web_search__fetch_url) instead of a 'fetch' CLI.",
    "download": "Unknown shell command 'download'. Use the web-search MCP tool or curl/wget instead of a 'download' CLI.",
}


def _first_token(command: str) -> str:
    """Return the first whitespace-delimited token of ``command``.

    Honours single/double quotes so ``"git status"`` is parsed as
    ``git``. Pipes and chains (``a; b``, ``a && b``, ``a | b``) are
    deliberately *not* supported: the model is supposed to call
    ``run_terminal_command`` once per logical command. Anything more
    elaborate is left to the shell itself (we only inspect the first
    token)."""
    s = (command or "").lstrip()
    if not s:
        return ""
    if s[0] in {"'", '"'}:
        quote = s[0]
        end = s.find(quote, 1)
        if end > 0:
            return s[1:end].strip()
    # Skip leading env-var assignments (FOO=bar command …).
    head = s
    while True:
        stripped = head.lstrip()
        if "=" in stripped.split(" ", 1)[0]:
            parts = stripped.split(" ", 1)
            if len(parts) == 1:
                return ""
            head = parts[1]
            continue
        break
    parts = head.split(None, 1)
    if not parts:
        return ""
    # Strip the executable path so ``/usr/bin/grep`` and ``grep`` both
    # match the whitelist entry. We split on both ``/`` and ``\`` so
    # Windows-style paths (``C:\Tools\grep.exe``) also resolve to
    # the bare executable.
    token = parts[0]
    for sep in ("/", "\\"):
        if sep in token:
            token = token.rsplit(sep, 1)[-1]
    # On Windows, executables commonly end in ``.exe``; strip it.
    if token.lower().endswith(".exe"):
        token = token[:-4]
    return token.lower()


def validate_terminal_command(command: str) -> str | None:
    """Return ``None`` when the command is allowed, or a short
    human-readable error string when it should be rejected.

    The check is intentionally cheap: extract the first token, look
    it up in the whitelist, and fall back to the hallucination
    dictionary. The model receives the rejection as a ``tool`` role
    message and can self-correct on the next iteration.
    """
    if not command or not command.strip():
        return "empty command"
    token = _first_token(command)
    if not token:
        return "could not parse executable from command"
    if token in _ALLOWED_TERMINAL_COMMANDS:
        return None
    if token in _HALLUCINATED_TERMINAL_COMMANDS:
        return _HALLUCINATED_TERMINAL_COMMANDS[token]
    return (
        f"Refused shell command '{token}': not in the allow-list. "
        "Use a built-in tool (web-search MCP, write_file, etc.) or a "
        "well-known CLI (grep, curl, find, ls, cat, …) instead."
    )
