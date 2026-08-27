"""Unit tests for the search query rewriter.

The rewriter is a small async helper that asks a dedicated LLM to turn
the user's last message + recent chat context into a single, clean
search query.  These tests cover:

* output sanitisation (``_sanitize_rewriter_output``);
* the fallback chain used by ``rewrite_search_query_safe``;
* the format of the conversation blob fed to the rewriter;
* end-to-end behaviour with a stubbed HTTP transport (so we never hit
  the real provider).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.services.query_rewriter import (
    _format_conversation_for_rewriter,
    _sanitize_rewriter_output,
    rewrite_search_query_safe,
)


# ---------------------------------------------------------------------------
# _sanitize_rewriter_output
# ---------------------------------------------------------------------------


def test_sanitize_strips_think_blocks():
    assert _sanitize_rewriter_output(
        "<think>the user wants X</think>aposolix artist biography"
    ) == "aposolix artist biography"


def test_sanitize_strips_code_fences():
    assert _sanitize_rewriter_output(
        "```\nAposolix Ukrainian electronic artist\n```"
    ) == "Aposolix Ukrainian electronic artist"


def test_sanitize_strips_leading_quotes_and_bullets():
    assert _sanitize_rewriter_output('"- Aposolix discography"') == "Aposolix discography"


def test_sanitize_returns_none_for_explicit_none():
    assert _sanitize_rewriter_output("NONE") is None
    assert _sanitize_rewriter_output("None") is None
    assert _sanitize_rewriter_output("n/a") is None


def test_sanitize_returns_none_for_empty_or_garbage():
    assert _sanitize_rewriter_output("") is None
    assert _sanitize_rewriter_output("   \n\n  ") is None
    assert _sanitize_rewriter_output("---") is None


def test_sanitize_truncates_oversized_output():
    long_query = "word " * 200
    out = _sanitize_rewriter_output(long_query)
    assert out is not None
    assert len(out) <= 200


def test_sanitize_keeps_first_line_when_model_adds_preamble():
    raw = (
        "Here is the search query:\n"
        "Aposolix artist biography\n"
        "\n"
        "Hope that helps!"
    )
    assert _sanitize_rewriter_output(raw) == "Aposolix artist biography"


# ---------------------------------------------------------------------------
# _format_conversation_for_rewriter
# ---------------------------------------------------------------------------


def test_format_includes_recent_messages_and_current():
    blob = _format_conversation_for_rewriter(
        "find that artist",
        recent_user_messages=["Aposolix", "hello world"],
    )
    assert "User: Aposolix" in blob
    assert "User: hello world" in blob
    assert blob.rstrip().endswith("User: find that artist")


def test_format_handles_no_recent_messages():
    blob = _format_conversation_for_rewriter("Aposolix", None)
    assert blob == "User: Aposolix"


def test_format_skips_empty_prior_messages():
    blob = _format_conversation_for_rewriter(
        "find that artist", recent_user_messages=["", "  ", "Aposolix"]
    )
    assert blob.count("User:") == 2
    assert "User: Aposolix" in blob


def test_format_caps_context_length_and_realigns_to_user():
    long_prior = "x" * 6000
    blob = _format_conversation_for_rewriter(
        "find the artist", recent_user_messages=[long_prior]
    )
    # The blob is hard-capped to _MAX_CONTEXT_CHARS and we re-anchor to
    # the start of the next "User:" line so we never emit half a line.
    assert len(blob) <= 4000
    assert blob.startswith("User:")


# ---------------------------------------------------------------------------
# rewrite_search_query_safe — fallback chain
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal stand-in for the provider/model pair returned by
    ``resolve_provider_model``.  Allows tests to control whether the
    LLM-call path is "available" or not."""

    base_url: str | None = "http://stub"
    request_timeout_sec: int = 60

    def __init__(self, available: bool, base_url: str | None = "http://stub") -> None:
        self.available = available
        self.base_url = base_url


@pytest.mark.asyncio
async def test_safe_rewriter_falls_back_to_resolver_when_no_provider(
    monkeypatch,
):
    """When ``resolve_provider_model`` returns ``None`` we must skip the
    LLM call and use the regex fallback so the search still works."""

    from app.services import query_rewriter

    monkeypatch.setattr(
        query_rewriter,
        "resolve_provider_model",
        lambda provider_id=None, model_id=None: (None, None),
    )

    resolved = await rewrite_search_query_safe(
        "the performer",
        recent_user_messages=["Aposolix", "Who is Aposolix?"],
        fallback_resolver=lambda msg, ctx: (
            "Who is Aposolix? | Aposolix | the performer"
        ),
    )
    assert resolved is not None
    assert "Aposolix" in resolved
    assert "the performer" in resolved


@pytest.mark.asyncio
async def test_safe_rewriter_falls_back_to_raw_when_nothing_helps(monkeypatch):
    from app.services import query_rewriter

    monkeypatch.setattr(
        query_rewriter,
        "resolve_provider_model",
        lambda provider_id=None, model_id=None: (None, None),
    )
    resolved = await rewrite_search_query_safe(
        "some completely standalone question",
        recent_user_messages=None,
        fallback_resolver=lambda msg, ctx: msg,
    )
    assert resolved == "some completely standalone question"


@pytest.mark.asyncio
async def test_safe_rewriter_uses_llm_result_when_available(monkeypatch):
    """Happy path: the LLM returns a clean query and we trust it over
    the regex fallback."""

    from app.services import query_rewriter

    class _StubModel:
        pass

    monkeypatch.setattr(
        query_rewriter,
        "resolve_provider_model",
        lambda provider_id=None, model_id=None: (_FakeProvider(True), _StubModel()),
    )

    captured: dict[str, Any] = {}

    def _fake_build_payload(provider, model, messages, stream):  # type: ignore[no-untyped-def]
        captured["messages"] = messages
        return ("http://stub", {"Authorization": "Bearer x"}, {"messages": messages})

    monkeypatch.setattr(query_rewriter, "build_payload", _fake_build_payload)

    # Patch ``httpx.AsyncClient`` directly on the module — both the
    # test scope and the rewriter module share the same httpx import
    # so this rebinds the symbol the rewriter will resolve at call time.
    import httpx as _httpx

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *exc_info) -> None:
            return None

        async def post(self, url, headers=None, json=None, **kwargs):  # type: ignore[no-untyped-def]
            class _Resp:
                status_code = 200

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, Any]:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Aposolix artist biography",
                                }
                            }
                        ]
                    }

            return _Resp()

    monkeypatch.setattr(_httpx, "AsyncClient", _FakeClient)

    resolved = await rewrite_search_query_safe(
        "find that artist",
        recent_user_messages=["Aposolix"],
        fallback_resolver=lambda msg, ctx: "should not be called",
    )
    assert resolved == "Aposolix artist biography"

    # Confirm the rewriter actually received the conversation blob as
    # the user message, *not* the raw "find that artist".
    user_messages = [m for m in captured["messages"] if m["role"] == "user"]
    assert len(user_messages) == 1
    assert "Aposolix" in user_messages[0]["content"]
    assert "find that artist" in user_messages[0]["content"]


@pytest.mark.asyncio
async def test_safe_rewriter_falls_back_when_provider_call_fails(monkeypatch):
    """A network / HTTP failure from the provider must NEVER break the
    chat — the rewriter must hand off to the fallback resolver."""

    from app.services import query_rewriter

    class _StubModel:
        pass

    monkeypatch.setattr(
        query_rewriter,
        "resolve_provider_model",
        lambda provider_id=None, model_id=None: (_FakeProvider(True), _StubModel()),
    )

    monkeypatch.setattr(
        query_rewriter,
        "build_payload",
        lambda *a, **k: ("http://stub", {}, {}),
    )

    class _BrokenClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_BrokenClient":
            return self

        async def __aexit__(self, *exc_info) -> None:
            return None

        async def post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise query_rewriter.httpx.ConnectError("simulated outage")

    monkeypatch.setattr(query_rewriter.httpx, "AsyncClient", _BrokenClient)

    fallback_called: list[Any] = []

    def _fallback(msg, ctx):
        fallback_called.append((msg, ctx))
        return "Aposolix artist"

    resolved = await rewrite_search_query_safe(
        "find that artist",
        recent_user_messages=["Aposolix"],
        fallback_resolver=_fallback,
    )
    assert resolved == "Aposolix artist"
    assert fallback_called, "fallback resolver should have been invoked"


@pytest.mark.asyncio
async def test_safe_rewriter_falls_back_when_llm_returns_none(monkeypatch):
    """If the LLM responds with ``NONE`` (i.e. 'no searchable subject'),
    fall back to the regex resolver so we still try to search rather
    than silently dropping the turn."""

    from app.services import query_rewriter

    class _StubModel:
        pass

    monkeypatch.setattr(
        query_rewriter,
        "resolve_provider_model",
        lambda provider_id=None, model_id=None: (_FakeProvider(True), _StubModel()),
    )

    monkeypatch.setattr(
        query_rewriter,
        "build_payload",
        lambda *a, **k: ("http://stub", {}, {}),
    )

    class _NoneClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_NoneClient":
            return self

        async def __aexit__(self, *exc_info) -> None:
            return None

        async def post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            class _Resp:
                status_code = 200

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, Any]:
                    return {"choices": [{"message": {"content": "NONE"}}]}

            return _Resp()

    monkeypatch.setattr(query_rewriter.httpx, "AsyncClient", _NoneClient)

    resolved = await rewrite_search_query_safe(
        "the performer",
        recent_user_messages=["Aposolix"],
        fallback_resolver=lambda msg, ctx: "Aposolix | the performer",
    )
    assert resolved == "Aposolix | the performer"