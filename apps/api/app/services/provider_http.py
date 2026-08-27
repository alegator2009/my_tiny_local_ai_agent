"""Helpers for building the HTTP request to an OpenAI-compatible
provider.  Centralising this keeps ``orchestrator``, ``runs`` and
``evolution`` consistent — they all need the same URL construction and
payload shape, so any tweak to one of them shouldn't drift between
call-sites."""

from __future__ import annotations

from typing import Any

from ..config import ModelEntry, ProviderConfig, load_app_config


class ProviderUnavailable(RuntimeError):
    """Raised when the chat layer tried to call a provider but the
    configuration is missing or incomplete."""


def resolve_provider_model(
    *,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> tuple[ProviderConfig | None, ModelEntry | None]:
    cfg = load_app_config()
    return cfg.resolve_pair(provider_id, model_id)


def build_payload(
    *,
    provider: ProviderConfig,
    model: ModelEntry,
    messages: list[dict[str, Any]],
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Return (url, headers, payload) for an OpenAI-compatible
    chat-completions call."""

    endpoint = provider.endpoint if provider.endpoint.startswith("/") else f"/{provider.endpoint}"
    url = provider.base_url.rstrip("/") + endpoint
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    payload: dict[str, Any] = {
        "model": model.name,
        "messages": messages,
        "stream": stream,
        "temperature": model.temperature,
        "top_p": model.top_p,
        "max_tokens": model.max_output_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    # Per-model extras override defaults last so the UI's per-model
    # ``extra_params_json`` is the source of truth.
    payload.update(model.extra_params_json or {})
    return url, headers, payload


def provider_timeout_seconds(provider: ProviderConfig) -> float:
    raw_value = provider.request_timeout_sec
    try:
        timeout_sec = int(raw_value)
    except Exception:
        timeout_sec = 240
    return float(max(5, min(timeout_sec, 600)))


def ensure_callable(
    *,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> tuple[ProviderConfig, ModelEntry]:
    """Return the provider/model pair or raise if the chat layer cannot
    talk to anyone.  Used by the tool-loop path which has no synthetic
    fallback."""

    provider, model = resolve_provider_model(provider_id=provider_id, model_id=model_id)
    if provider is None or model is None:
        raise ProviderUnavailable("no provider/model configured")
    if not provider.base_url:
        raise ProviderUnavailable("provider base_url is empty")
    return provider, model