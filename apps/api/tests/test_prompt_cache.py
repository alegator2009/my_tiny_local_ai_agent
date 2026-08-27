"""Tests for the prompt prefix cache."""

from __future__ import annotations

import time

import pytest

from app.services.prompt_cache import (
    DEFAULT_CAPACITY,
    DEFAULT_TTL_SECONDS,
    PromptCache,
    fingerprint_config,
    fingerprint_tool_set,
    make_cache_key,
)


# --- helpers ----------------------------------------------------------------


class _FakeCfg:
    """Minimal stand-in for AppConfig with only the fields used by
    fingerprint_config."""

    def __init__(self, system_prompt: str = "You are a helper.", session_memory_profile: str = "default") -> None:
        self.system_prompt = system_prompt
        self.session_memory_profile = session_memory_profile


@pytest.fixture
def cache() -> PromptCache:
    return PromptCache(capacity=4, ttl_seconds=60.0)


# --- fingerprinting ---------------------------------------------------------


def test_fingerprint_config_is_stable() -> None:
    a = fingerprint_config(_FakeCfg("hello", "default"))
    b = fingerprint_config(_FakeCfg("hello", "default"))
    assert a == b


def test_fingerprint_config_changes_with_system_prompt() -> None:
    a = fingerprint_config(_FakeCfg("hello", "default"))
    b = fingerprint_config(_FakeCfg("different", "default"))
    assert a != b


def test_fingerprint_config_changes_with_profile() -> None:
    a = fingerprint_config(_FakeCfg("hello", "default"))
    b = fingerprint_config(_FakeCfg("hello", "verbose"))
    assert a != b


def test_fingerprint_config_ignores_unrelated_fields() -> None:
    """Only system_prompt and session_memory_profile are part of the
    fingerprint.  Other attributes on the config object must not affect it."""
    a = fingerprint_config(_FakeCfg())
    b = fingerprint_config(_FakeCfg())
    # Mutate unrelated attributes -- shouldn't change fingerprint.
    cfg = _FakeCfg()
    cfg.system_prompt = "You are a helper."
    cfg.session_memory_profile = "default"
    fp1 = fingerprint_config(cfg)
    cfg.unrelated_attribute = "ignored"
    fp2 = fingerprint_config(cfg)
    assert fp1 == fp2 == a == b


def test_fingerprint_tool_set_none_returns_none_token() -> None:
    assert fingerprint_tool_set(None) == "none"
    assert fingerprint_tool_set([]) == "none"


def test_fingerprint_tool_set_is_stable_for_same_lines() -> None:
    a = fingerprint_tool_set(["read_file: read a file", "write_file: write a file"])
    b = fingerprint_tool_set(["read_file: read a file", "write_file: write a file"])
    assert a == b


def test_fingerprint_tool_set_order_matters() -> None:
    a = fingerprint_tool_set(["read_file", "write_file"])
    b = fingerprint_tool_set(["write_file", "read_file"])
    assert a != b


def test_fingerprint_tool_set_strips_trailing_whitespace() -> None:
    a = fingerprint_tool_set(["read_file  "])
    b = fingerprint_tool_set(["read_file"])
    assert a == b


def test_make_cache_key_returns_tuple() -> None:
    key = make_cache_key(config_hash="abc", thinking_mode="medium", tool_set_hash="xyz")
    assert key == ("abc", "medium", "xyz")
    assert isinstance(key, tuple)


# --- basic get/put ---------------------------------------------------------


def test_get_returns_none_on_miss(cache: PromptCache) -> None:
    assert cache.get(("cfg", "medium", "tools")) is None


def test_put_then_get_returns_value(cache: PromptCache) -> None:
    cache.put(("cfg", "medium", "tools"), "hello world")
    assert cache.get(("cfg", "medium", "tools")) == "hello world"


def test_put_overwrites_existing_value(cache: PromptCache) -> None:
    cache.put(("cfg", "medium", "tools"), "first")
    cache.put(("cfg", "medium", "tools"), "second")
    assert cache.get(("cfg", "medium", "tools")) == "second"


# --- LRU eviction ----------------------------------------------------------


def test_evicts_oldest_when_capacity_exceeded(cache: PromptCache) -> None:
    for i in range(cache.capacity):
        cache.put((f"k{i}", "medium", "t"), f"v{i}")
    # Now exceed capacity
    cache.put(("overflow", "medium", "t"), "v_overflow")
    # First key should have been evicted
    assert cache.get(("k0", "medium", "t")) is None
    assert cache.get(("overflow", "medium", "t")) == "v_overflow"


def test_get_promotes_to_most_recently_used(cache: PromptCache) -> None:
    for i in range(cache.capacity):
        cache.put((f"k{i}", "medium", "t"), f"v{i}")
    # Touch k0 -> it should not be evicted next.
    assert cache.get(("k0", "medium", "t")) == "v0"
    cache.put(("overflow", "medium", "t"), "v_overflow")
    # k1 is now the LRU and should be gone
    assert cache.get(("k1", "medium", "t")) is None
    assert cache.get(("k0", "medium", "t")) == "v0"


# --- TTL expiry -------------------------------------------------------------


def test_expired_entry_returns_none_and_increments_expirations() -> None:
    cache = PromptCache(capacity=4, ttl_seconds=0.05)
    cache.put(("k", "m", "t"), "v")
    time.sleep(0.1)
    assert cache.get(("k", "m", "t")) is None
    assert cache.stats()["expirations"] == 1


def test_unexpired_entry_is_returned(cache: PromptCache) -> None:
    cache.put(("k", "m", "t"), "v")
    # No sleep -- should still be present.
    assert cache.get(("k", "m", "t")) == "v"


# --- stats ------------------------------------------------------------------


def test_stats_track_hits_and_misses(cache: PromptCache) -> None:
    cache.put(("k", "m", "t"), "v")
    cache.get(("k", "m", "t"))  # hit
    cache.get(("missing", "m", "t"))  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_stats_track_evictions(cache: PromptCache) -> None:
    for i in range(cache.capacity + 2):
        cache.put((f"k{i}", "m", "t"), f"v{i}")
    assert cache.stats()["evictions"] >= 2


def test_reset_stats_preserves_entries(cache: PromptCache) -> None:
    cache.put(("k", "m", "t"), "v")
    cache.get(("k", "m", "t"))
    cache.reset_stats()
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["size"] == 1
    assert cache.get(("k", "m", "t")) == "v"


# --- invalidate -------------------------------------------------------------


def test_invalidate_clears_all_entries(cache: PromptCache) -> None:
    cache.put(("a", "m", "t"), "1")
    cache.put(("b", "m", "t"), "2")
    cache.invalidate()
    assert cache.get(("a", "m", "t")) is None
    assert cache.get(("b", "m", "t")) is None
    assert cache.stats()["size"] == 0


# --- construction validation ----------------------------------------------


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        PromptCache(capacity=0)
    with pytest.raises(ValueError):
        PromptCache(capacity=-1)


def test_ttl_must_be_positive() -> None:
    with pytest.raises(ValueError):
        PromptCache(ttl_seconds=0)
    with pytest.raises(ValueError):
        PromptCache(ttl_seconds=-1.0)


def test_default_capacity_is_64() -> None:
    assert DEFAULT_CAPACITY == 64


def test_default_ttl_is_five_minutes() -> None:
    assert DEFAULT_TTL_SECONDS == 300.0


# --- thread safety ---------------------------------------------------------


def test_get_and_put_are_thread_safe() -> None:
    import threading
    cache = PromptCache(capacity=128, ttl_seconds=60.0)

    def writer(idx: int) -> None:
        for i in range(200):
            cache.put((f"k{idx}-{i}", "m", "t"), f"v{idx}-{i}")

    def reader(idx: int) -> None:
        for i in range(200):
            cache.get((f"k{idx}-{i}", "m", "t"))

    threads = []
    for i in range(4):
        threads.append(threading.Thread(target=writer, args=(i,)))
        threads.append(threading.Thread(target=reader, args=(i,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = cache.stats()
    # At least one operation must have hit or missed.  Evictions are
    # possible because the test only writes 800 entries into a cache of
    # capacity 128.
    assert stats["hits"] + stats["misses"] > 0
    assert stats["size"] <= 128
