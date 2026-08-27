"""Unit tests for the auto-search router.

These tests cover the cheap regex heuristics and the SQLite cache
without ever spinning up the real native-web-search MCP server (which
needs Node, Playwright and outbound network).  The end-to-end happy
path is covered by manual smoke-testing from the Settings page.
"""

from __future__ import annotations

import time

import pytest

from app.config import AutoSearchConfig
from app.db import execute, init_db
from app.services.auto_search import (
    AutoSearchCache,
    build_grounded_block,
    should_search,
)


# ---------------------------------------------------------------------------
# should_search
# ---------------------------------------------------------------------------


def test_should_search_off_policy_never_fires():
    decision = should_search(
        "What's the weather in Lisbon?",
        policy="off",
        enabled=True,
    )
    assert decision.should_search is False
    assert decision.reason == "policy_off"


def test_should_search_always_policy_fires_for_opinions():
    decision = should_search(
        "What do you think about Rust vs Go?",
        policy="always",
        enabled=True,
    )
    assert decision.should_search is True
    assert decision.reason == "policy_always"


def test_should_search_auto_picks_up_freshness_hints():
    decision = should_search(
        "What's the latest news today?",
        policy="auto",
        enabled=True,
    )
    assert decision.should_search is True
    assert decision.reason == "freshness_hint"


def test_should_search_auto_picks_up_factual_hints():
    decision = should_search(
        "Who is the CEO of Anthropic?",
        policy="auto",
        enabled=True,
    )
    assert decision.should_search is True
    assert decision.reason in {"freshness_hint", "factual_hint"}


def test_should_search_auto_picks_up_price_hint():
    decision = should_search(
        "iPhone 16 Pro price in USD",
        policy="auto",
        enabled=True,
    )
    assert decision.should_search is True


def test_should_search_auto_skips_pure_chitchat():
    decision = should_search(
        "Hello!",
        policy="auto",
        enabled=True,
    )
    assert decision.should_search is False
    assert decision.reason in {"opinion_or_chitchat", "no_signal"}


def test_should_search_auto_skips_subjective_opinion():
    decision = should_search(
        "What's your opinion on Rust?",
        policy="auto",
        enabled=True,
    )
    assert decision.should_search is False
    assert decision.reason == "opinion_or_chitchat"


def test_should_search_force_overrides_policy_off():
    decision = should_search(
        "hi",
        policy="off",
        enabled=True,
        force=True,
    )
    assert decision.should_search is True
    assert decision.reason == "forced_by_user"


def test_should_search_empty_query_is_skipped():
    decision = should_search(
        "   ",
        policy="always",
        enabled=True,
    )
    assert decision.should_search is False
    assert decision.reason == "empty_query"


# ---------------------------------------------------------------------------
# AutoSearchCache
# ---------------------------------------------------------------------------


def _sample_citations():
    return [
        {
            "title": "Anthropic — Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Anthropic",
            "description": "Anthropic PBC is an American artificial intelligence startup.",
            "snippet": "Anthropic PBC is an American artificial intelligence startup.",
            "engine": "bing",
        },
    ]


def test_cache_miss_then_hit(isolated_data_dir):
    init_db()
    cache = AutoSearchCache(ttl_sec=60)
    assert cache.get("Who is the CEO of Anthropic?") is None

    cache.put(
        "Who is the CEO of Anthropic?",
        answer="Anthropic's CEO is Dario Amodei.",
        citations=_sample_citations(),
        engine="bing",
    )
    hit1 = cache.get("Who is the CEO of Anthropic?")
    assert hit1 is not None
    assert hit1["cache_hit"] is True
    assert hit1["answer"] == "Anthropic's CEO is Dario Amodei."
    assert hit1["citations"][0]["url"].endswith("/Anthropic")
    assert hit1["hits"] == 1  # one get bumped the counter

    hit2 = cache.get("Who is the CEO of Anthropic?")
    assert hit2 is not None
    assert hit2["hits"] == 2  # a second get bumps it again


def test_cache_expiry(isolated_data_dir):
    init_db()
    cache = AutoSearchCache(ttl_sec=1)
    cache.put(
        "What is the current weather in Madrid?",
        answer="Sunny.",
        citations=_sample_citations(),
        engine="duckduckgo",
    )
    assert cache.get("What is the current weather in Madrid?") is not None
    time.sleep(1.5)
    assert cache.get("What is the current weather in Madrid?") is None


def test_cache_normalises_whitespace_and_case(isolated_data_dir):
    init_db()
    cache = AutoSearchCache(ttl_sec=60)
    cache.put("Hello   World", answer="hi", citations=[], engine="bing")
    hit = cache.get("hello world")
    assert hit is not None
    assert hit["answer"] == "hi"


# ---------------------------------------------------------------------------
# build_grounded_block
# ---------------------------------------------------------------------------


def test_build_grounded_block_renders_citations():
    from app.services.auto_search import AutoSearchResult

    result = AutoSearchResult(
        query="Who is the CEO of Anthropic?",
        normalized_query="who is the ceo of anthropic?",
        answer="Dario Amodei is the CEO of Anthropic.",
        citations=_sample_citations(),
        engine="bing",
        source="auto_search",
        cache_hit=False,
        took_ms=120,
    )
    block = build_grounded_block(result)
    assert "Grounded web search results" in block
    assert "engine: bing" in block
    assert "[1]" in block
    assert "Anthropic" in block
    assert "en.wikipedia.org" in block


def test_build_grounded_block_empty_when_error_and_no_citations():
    from app.services.auto_search import AutoSearchResult

    result = AutoSearchResult(
        query="x",
        normalized_query="x",
        answer="",
        citations=[],
        engine="",
        source="auto_search",
        cache_hit=False,
        took_ms=0,
        error="search_timeout",
    )
    assert build_grounded_block(result) == ""


# ---------------------------------------------------------------------------
# AutoSearchConfig validation
# ---------------------------------------------------------------------------


def test_auto_search_config_normalises_policy():
    cfg = AutoSearchConfig(policy="ALWAYS", enabled=True)
    assert cfg.policy == "always"
    cfg = AutoSearchConfig(policy="garbage", enabled=True)
    assert cfg.policy == "auto"


def test_auto_search_config_clamps_negative_values():
    cfg = AutoSearchConfig(
        enabled=True,
        max_chars=-10,
        max_per_turn=-2,
        cache_ttl_sec=-100,
    )
    assert cfg.max_chars == 0
    assert cfg.max_per_turn == 0
    assert cfg.cache_ttl_sec == 0
