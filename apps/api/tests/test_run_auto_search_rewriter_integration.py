"""Integration test: ``run_auto_search`` must use the LLM-based query
rewriter when it is available, instead of feeding the raw user message
("Force web search") to the search backend.

This is a regression test for the bug observed in session
``9b82ead0-6a04-4eba-8cde-dc40416889dc`` where the user's meta-command
"Force web search" was treated as the literal query, returning
Bing results about the FORCE bicycle brand instead of resolving the
prior "Aposolix" mention in the chat history.

We don't exercise the full MCP registry / native-web-search path here
(that's covered by ``test_auto_search.py`` and a manual smoke test);
instead we focus on the *query selection* logic, asserting that the
decision payload seen by the MCP backend contains the *rewritten*
query, not the raw user message.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import AppConfig, AutoSearchConfig, MCPConfig
from app.services import auto_search, query_rewriter


def _config_with_policy(policy: str = "auto") -> AppConfig:
    cfg = AppConfig()
    cfg.mcp_config = MCPConfig(
        enabled=True,
        native_web_search_enabled=True,
        auto_search=AutoSearchConfig(
            enabled=True,
            policy=policy,
            cache_ttl_sec=0,
            max_per_turn=1,
        ),
    )
    return cfg


def test_run_auto_search_rewrites_meta_command(monkeypatch):
    """``Force web search`` must be rewritten to the topic the user
    actually wants ('Aposolix'), not fed verbatim to the backend."""

    captured: dict[str, Any] = {}

    async def _stub_safe(last, *, recent_user_messages=None, provider_id=None, model_id=None, fallback_resolver=None):  # type: ignore[no-untyped-def]
        captured["last"] = last
        captured["recent"] = recent_user_messages
        captured["stub_called"] = True
        return "Aposolix artist biography"

    monkeypatch.setattr(query_rewriter, "rewrite_search_query_safe", _stub_safe)

    # Patch ``should_search`` to keep the routing trivial — we want to
    # exercise only the query rewriting branch.
    def _always_search(query, **kwargs):  # type: ignore[no-untyped-def]
        from app.services.auto_search import SearchDecision

        return SearchDecision(
            should_search=True,
            reason="forced_by_user",
            policy="auto",
            query=query,
            normalized_query=query.lower(),
        )

    monkeypatch.setattr(auto_search, "should_search", _always_search)

    # Stub the cache + MCP call so we never hit the real backend.
    class _StubCache:
        def __init__(self, ttl_sec: int = 0) -> None:
            self.ttl_sec = ttl_sec

        def get(self, query):  # type: ignore[no-untyped-def]
            captured.setdefault("cache_gets", []).append(query)
            return None

        def put(self, *a, **kw):  # type: ignore[no-untyped-def]
            captured.setdefault("cache_puts", []).append((a, kw))

    monkeypatch.setattr(auto_search, "AutoSearchCache", _StubCache)

    async def _fake_call(query, *, cfg, request_timeout_sec):  # type: ignore[no-untyped-def]
        captured["backend_query"] = query
        return {"ok": True, "answer": "", "citations": [], "engine": "stub"}

    monkeypatch.setattr(auto_search, "_call_native_web_search", _fake_call)

    import asyncio

    result = asyncio.run(
        auto_search.run_auto_search(
            "Force web search",
            cfg=_config_with_policy(),
            recent_user_messages=["Aposolix", "Who is Aposolix?"],
            force=True,
        )
    )

    # 1. The rewriter received the raw message + the recent context.
    assert captured.get("stub_called"), "rewriter stub was not invoked"
    assert captured["last"] == "Force web search"
    assert "Aposolix" in (captured["recent"] or [])
    assert "Who is Aposolix?" in (captured["recent"] or [])

    # 2. The search backend received the *rewritten* query, NOT "Force".
    assert captured["backend_query"] == "Aposolix artist biography", (
        f"Expected rewritten query, got {captured['backend_query']!r}"
    )

    # 3. The returned AutoSearchResult carries the rewritten query.
    assert result.query == "Aposolix artist biography"
    assert result.error == ""


def test_run_auto_search_falls_back_when_rewriter_disabled(monkeypatch):
    """When ``rewrite_query=False`` is passed we must use the legacy
    regex concatenation only — no LLM call."""

    captured: dict[str, Any] = {}

    async def _must_not_call(*a, **kw):  # type: ignore[no-untyped-def]
        captured["rewriter_called"] = True
        return None

    monkeypatch.setattr(query_rewriter, "rewrite_search_query_safe", _must_not_call)

    def _always_search(query, **kwargs):  # type: ignore[no-untyped-def]
        from app.services.auto_search import SearchDecision

        return SearchDecision(
            should_search=True,
            reason="forced_by_user",
            policy="auto",
            query=query,
            normalized_query=query.lower(),
        )

    monkeypatch.setattr(auto_search, "should_search", _always_search)

    class _StubCache:
        def __init__(self, ttl_sec: int = 0) -> None:
            self.ttl_sec = ttl_sec

        def get(self, query):  # type: ignore[no-untyped-def]
            return None

        def put(self, *a, **kw):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(auto_search, "AutoSearchCache", _StubCache)

    async def _fake_call(query, *, cfg, request_timeout_sec):  # type: ignore[no-untyped-def]
        captured["backend_query"] = query
        return {"ok": True, "answer": "", "citations": [], "engine": "stub"}

    monkeypatch.setattr(auto_search, "_call_native_web_search", _fake_call)

    import asyncio

    # "2" is a single-word follow-up — the regex resolver should fold
    # in the prior messages so the backend sees something usable.
    result = asyncio.run(
        auto_search.run_auto_search(
            "2",
            cfg=_config_with_policy(),
            recent_user_messages=["Maximum body mass.", "Search the internet for it"],
            force=True,
            rewrite_query=False,
        )
    )

    assert "rewriter_called" not in captured, (
        "rewriter must not be called when rewrite_query=False"
    )
    # The regex fallback concatenates prior messages.
    assert "2" in (captured["backend_query"] or "")
    assert "Search the internet for it" in (captured["backend_query"] or "")
    assert result.error == ""


def test_run_auto_search_resolves_real_session_meta_command(monkeypatch):
    """End-to-end check against the actual chat messages from session
    ``9b82ead0-6a04-4eba-8cde-dc40416889dc`` (the one that motivated
    this work).  The user wrote 'Force web search' after several turns
    about 'Aposolix'.  The rewriter must produce something like
    'Aposolix artist biography' so the search backend stops returning
    the FORCE bicycle brand."""

    captured: dict[str, Any] = {}

    async def _rewriter_like_real_llm(last, *, recent_user_messages=None, provider_id=None, model_id=None, fallback_resolver=None):  # type: ignore[no-untyped-def]
        # Simulate what the real rewriter would output for this exact
        # conversation: a clean, self-contained search query that
        # resolves the pronoun.
        if last.strip().lower() == "force web search":
            return "Aposolix artist biography discography"
        if last.strip().lower() == "the performer":
            return "Aposolix artist information"
        # For everything else fall through to the regex resolver.
        if fallback_resolver is not None:
            return fallback_resolver(last, recent_user_messages)
        return last

    monkeypatch.setattr(query_rewriter, "rewrite_search_query_safe", _rewriter_like_real_llm)

    def _always_search(query, **kwargs):  # type: ignore[no-untyped-def]
        from app.services.auto_search import SearchDecision

        return SearchDecision(
            should_search=True,
            reason="forced_by_user",
            policy="auto",
            query=query,
            normalized_query=query.lower(),
        )

    monkeypatch.setattr(auto_search, "should_search", _always_search)

    class _StubCache:
        def __init__(self, ttl_sec: int = 0) -> None:
            self.ttl_sec = ttl_sec

        def get(self, query):  # type: ignore[no-untyped-def]
            return None

        def put(self, *a, **kw):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(auto_search, "AutoSearchCache", _StubCache)

    async def _fake_call(query, *, cfg, request_timeout_sec):  # type: ignore[no-untyped-def]
        captured.setdefault("queries", []).append(query)
        return {"ok": True, "answer": "", "citations": [], "engine": "stub"}

    monkeypatch.setattr(auto_search, "_call_native_web_search", _fake_call)

    import asyncio

    history = [
        "What was the biggest AI breakthrough this past month?",
        "Who is Aposolix?",
        "Aposolix",
    ]

    # The exact two turns that were broken in the original session.
    r1 = asyncio.run(
        auto_search.run_auto_search(
            "Force web search",
            cfg=_config_with_policy(),
            recent_user_messages=history,
            force=True,
        )
    )
    r2 = asyncio.run(
        auto_search.run_auto_search(
            "the performer",
            cfg=_config_with_policy(),
            recent_user_messages=history + ["Force web search"],
            force=True,
        )
    )

    assert captured["queries"][0] == "Aposolix artist biography discography"
    assert captured["queries"][1] == "Aposolix artist information"
    assert r1.query == "Aposolix artist biography discography"
    assert r2.query == "Aposolix artist information"