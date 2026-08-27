"""Normalise and sanitise JSON Schemas before they reach the model.

MCP servers and third-party tool providers emit JSON Schemas in many shapes.
Some are valid JSON Schema but use constructs our local-model validator does
not understand (``oneOf``, ``anyOf``, ``$ref``); others are technically valid
but cause practical problems (enums with hundreds of values blow up the
prompt; missing ``type: object`` on the root confuses the model). This
module rewrites such schemas into a conservative subset the rest of the
orchestrator can rely on.

The sanitiser is intentionally *lossy* with respect to the original schema:
its job is not to round-trip the schema faithfully but to make sure the
version handed to the model and the validator is internally consistent and
cheap to evaluate. Callers that need the full original schema (for example
to send back to an MCP server on a tool call) keep it; sanitisation only
applies to the schema that is shown to the model and the validator.
"""

from __future__ import annotations

import copy
from typing import Any


# Maximum number of ``enum`` values we will surface to the model. Anything
# longer is summarised as ``"... N more"`` so the prompt does not blow up.
DEFAULT_MAX_ENUM_SIZE = 20

# Maximum length of a single description in the sanitised schema. Longer text
# is replaced with the original up to this length plus a trailing note.
DEFAULT_MAX_DESCRIPTION = 500

# Construct keywords our validator does not implement. Stripping them is safe
# for the model's prompt (it is encouraged to follow the simpler remaining
# schema) and the validator just ignores the rest of the schema anyway.
_UNSUPPORTED_KEYWORDS = {"oneOf", "anyOf", "allOf", "$ref", "$defs", "definitions"}


def _truncate_string(value: Any, max_len: int) -> Any:
    if not isinstance(value, str) or len(value) <= max_len:
        return value
    # Reserve room for the trailing ellipsis so the final string respects
    # ``max_len`` *exactly*.
    if max_len <= 3:
        return "." * max_len
    return value[: max_len - 3].rstrip() + "..."


def _truncate_enum(enum: list[Any], max_size: int) -> list[Any]:
    if not isinstance(enum, list) or len(enum) <= max_size:
        return enum
    return list(enum[:max_size]) + [f"... {len(enum) - max_size} more"]


def _ensure_object_root(schema: dict[str, Any]) -> dict[str, Any]:
    """Make sure the root looks like an object schema."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    if schema.get("type") != "object":
        # Be conservative: keep the original description but coerce to object.
        schema = dict(schema)
        schema["type"] = "object"
    schema.setdefault("properties", {})
    if not isinstance(schema.get("properties"), dict):
        schema["properties"] = {}
    return schema


def _sanitize_subschema(
    node: Any,
    *,
    max_enum_size: int,
    max_description: int,
) -> Any:
    """Recursively sanitise a node of the schema tree."""
    if isinstance(node, list):
        return [
            _sanitize_subschema(item, max_enum_size=max_enum_size, max_description=max_description)
            for item in node
        ]
    if not isinstance(node, dict):
        return node

    cleaned = dict(node)
    # Strip unsupported compound keywords. We do this *before* recursing so
    # that any nested ``$ref`` or ``oneOf`` is dropped together.
    for kw in list(cleaned.keys()):
        if kw in _UNSUPPORTED_KEYWORDS:
            cleaned.pop(kw, None)
        elif kw == "enum" and isinstance(cleaned[kw], list):
            cleaned[kw] = _truncate_enum(cleaned[kw], max_enum_size)
        elif kw == "description":
            cleaned[kw] = _truncate_string(cleaned[kw], max_description)
        elif kw == "type" and not isinstance(cleaned[kw], str):
            cleaned.pop(kw, None)

    if isinstance(cleaned.get("properties"), dict):
        cleaned["properties"] = {
            key: _sanitize_subschema(
                sub, max_enum_size=max_enum_size, max_description=max_description
            )
            for key, sub in cleaned["properties"].items()
        }

    if "items" in cleaned:
        cleaned["items"] = _sanitize_subschema(
            cleaned["items"],
            max_enum_size=max_enum_size,
            max_description=max_description,
        )

    return cleaned


def sanitize_schema(
    schema: Any,
    *,
    max_enum_size: int = DEFAULT_MAX_ENUM_SIZE,
    max_description: int = DEFAULT_MAX_DESCRIPTION,
) -> dict[str, Any]:
    """Return a sanitised copy of ``schema`` that is safe to feed to the model.

    The input is not mutated. The output is a fresh dict tree that always:

    * is a JSON object with ``type: object`` and a dict ``properties``;
    * has no compound keywords (``oneOf``/``anyOf``/``allOf``/``$ref``/``$defs``);
    * has ``enum`` lists capped at ``max_enum_size`` entries;
    * has every ``description`` capped at ``max_description`` characters;
    * preserves ``required``, ``additionalProperties``, ``minimum`` /
      ``maximum`` / ``minLength`` / ``maxLength`` untouched.

    A non-dict input (or ``None``) collapses to ``{"type": "object",
    "properties": {}}`` so callers can pass the result straight to the
    validator without further checks.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out = copy.deepcopy(schema)
    out = _ensure_object_root(out)
    out = _sanitize_subschema(
        out, max_enum_size=max_enum_size, max_description=max_description
    )
    # Re-apply the object-root guarantees because _sanitize_subschema may
    # have replaced ``properties`` with a list-typed value.
    out = _ensure_object_root(out)
    if not isinstance(out.get("required"), list):
        out["required"] = [
            k for k in out["required"] if isinstance(k, str)
        ] if isinstance(out.get("required"), list) else []
    return out


def schema_fingerprint(schema: Any) -> str:
    """Return a short, stable string that changes when ``schema`` changes.

    Used by :mod:`schema_cache` to avoid re-sanitising the same schema on
    every turn. The fingerprint is intentionally not cryptographic: it just
    has to be cheap and change when the schema bytes change.
    """
    try:
        import hashlib
        import json
        payload = json.dumps(schema, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = repr(schema)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pre-send: filter unknown properties + coerce types
# ---------------------------------------------------------------------------


def filter_unknown_properties(
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Drop keys that the schema rejects.

    When ``additionalProperties: false`` is set on the (object) schema, MCP
    servers return an error if the model adds an extra field (for example
    a cached ``_retrieval_id`` or a duplicate of an existing field). Stripping
    those here turns a tool error into a successful call without changing
    the semantics of well-formed arguments.

    Unknown keys are still kept when ``additionalProperties`` is not
    explicitly ``False`` (the JSON Schema default is to allow them).
    """
    if not isinstance(arguments, dict) or not isinstance(schema, dict):
        return arguments
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return arguments
    if schema.get("additionalProperties") is not False:
        return arguments
    allowed = set(properties.keys())
    return {k: v for k, v in arguments.items() if k in allowed}


def _coerce_scalar(value: Any, declared: str) -> Any:
    """Best-effort coercion for a single scalar value.

    Returns the original value when coercion would lose information
    (for example a non-numeric string into an integer). The orchestrator
    treats the result as *fallible*: if coercion fails the caller can
    fall back to the original argument.
    """
    if declared == "string":
        if isinstance(value, bool):
            # ``bool`` is a subclass of ``int``; never silently convert.
            return value
        if isinstance(value, str):
            return value
        return str(value)
    if declared == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return value
            try:
                as_float = float(stripped)
            except ValueError:
                return value
            return int(as_float) if as_float.is_integer() else value
        return value
    if declared == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return value
            try:
                return float(stripped)
            except ValueError:
                return value
        return value
    if declared == "boolean":
        if isinstance(value, bool):
            return value
        return value
    return value


def coerce_for_send(
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Coerce top-level argument values to the declared JSON Schema types.

    Some MCP servers reject payloads that are technically correct but
    technically *incompatible* (``"42"`` instead of ``42`` for a field
    declared ``"type": "integer"``). This helper walks ``properties`` and
    applies :func:`_coerce_scalar` to each value whose type does not match
    the declared type. The coercion is conservative: it widens but never
    narrows, and it never drops data the model intended to send.

    A non-dict argument or a non-dict schema returns the arguments
    unchanged so the call path can handle errors downstream.
    """
    if not isinstance(arguments, dict) or not isinstance(schema, dict):
        return arguments
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return arguments
    out: dict[str, Any] = dict(arguments)
    for key, sub_schema in properties.items():
        if key not in out or not isinstance(sub_schema, dict):
            continue
        declared = sub_schema.get("type")
        if not isinstance(declared, str):
            continue
        out[key] = _coerce_scalar(out[key], declared)
    return out


def sanitize_for_send(
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convenience wrapper: filter unknown keys, then coerce.

    Equivalent to ``coerce_for_send(filter_unknown_properties(arguments,
    schema), schema)`` but kept as a single function for the hot path.
    """
    return coerce_for_send(filter_unknown_properties(arguments, schema), schema)
