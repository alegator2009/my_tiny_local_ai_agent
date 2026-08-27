"""Tests for :class:`app.services.schema_cache.SchemaCache`.

The cache is a tiny LRU; we only need to verify:

* same input -> same output, with a hit counter incrementing;
* different input -> new entry, miss counter incrementing;
* capacity is respected (oldest entry evicted);
* invalidation by server slug works;
* thread-safety: concurrent calls do not corrupt the cache.
"""

from __future__ import annotations

import threading

import pytest

from app.services.schema_cache import SchemaCache


def test_first_call_misses_and_computes():
    cache = SchemaCache(capacity=4)
    raw = {"type": "object", "properties": {"x": {"type": "string"}}}
    out = cache.get_or_compute(server_slug="srv", tool_name="x", raw_schema=raw)
    assert out["type"] == "object"
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_second_call_with_same_input_hits_cache():
    cache = SchemaCache(capacity=4)
    raw = {"type": "object", "properties": {"x": {"type": "string"}}}
    cache.get_or_compute(server_slug="srv", tool_name="x", raw_schema=raw)
    cache.get_or_compute(server_slug="srv", tool_name="x", raw_schema=raw)
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_different_input_triggers_new_compute():
    cache = SchemaCache(capacity=4)
    a = {"type": "object", "properties": {"x": {"type": "string"}}}
    b = {"type": "object", "properties": {"x": {"type": "integer"}}}
    cache.get_or_compute(server_slug="srv", tool_name="x", raw_schema=a)
    cache.get_or_compute(server_slug="srv", tool_name="x", raw_schema=b)
    stats = cache.stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 0


def test_different_tool_name_is_a_separate_entry():
    cache = SchemaCache(capacity=4)
    raw = {"type": "object", "properties": {"x": {"type": "string"}}}
    cache.get_or_compute(server_slug="srv", tool_name="a", raw_schema=raw)
    cache.get_or_compute(server_slug="srv", tool_name="b", raw_schema=raw)
    stats = cache.stats()
    assert stats["misses"] == 2
    assert stats["size"] == 2


def test_different_server_slug_is_a_separate_entry():
    cache = SchemaCache(capacity=4)
    raw = {"type": "object", "properties": {"x": {"type": "string"}}}
    cache.get_or_compute(server_slug="srv1", tool_name="x", raw_schema=raw)
    cache.get_or_compute(server_slug="srv2", tool_name="x", raw_schema=raw)
    assert cache.stats()["size"] == 2


def test_capacity_is_respected_lru_eviction():
    cache = SchemaCache(capacity=2)
    for i in range(5):
        cache.get_or_compute(
            server_slug="srv",
            tool_name=f"t{i}",
            raw_schema={"i": i},
        )
    assert cache.stats()["size"] == 2
    # The most recent two should still be there.
    cache.get_or_compute(server_slug="srv", tool_name="t3", raw_schema={"i": 3})
    cache.get_or_compute(server_slug="srv", tool_name="t4", raw_schema={"i": 4})
    assert cache.stats()["hits"] == 2


def test_invalidate_by_server_slug_drops_only_that_server():
    cache = SchemaCache(capacity=8)
    raw = {"type": "object", "properties": {}}
    cache.get_or_compute(server_slug="srv1", tool_name="x", raw_schema=raw)
    cache.get_or_compute(server_slug="srv2", tool_name="x", raw_schema=raw)
    removed = cache.invalidate("srv1")
    assert removed == 1
    assert cache.stats()["size"] == 1
    # A second call for the same srv1 input is now a miss.
    cache.get_or_compute(server_slug="srv1", tool_name="x", raw_schema=raw)
    assert cache.stats()["misses"] >= 2  # 1 from registration + 1 after invalidate


def test_invalidate_with_no_arg_clears_everything():
    cache = SchemaCache(capacity=4)
    cache.get_or_compute(server_slug="s1", tool_name="a", raw_schema={"x": 1})
    cache.get_or_compute(server_slug="s2", tool_name="b", raw_schema={"x": 2})
    removed = cache.invalidate()
    assert removed == 2
    assert cache.stats()["size"] == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        SchemaCache(capacity=0)
    with pytest.raises(ValueError):
        SchemaCache(capacity=-1)


def test_concurrent_calls_do_not_corrupt_cache():
    cache = SchemaCache(capacity=16)
    raw = {"type": "object", "properties": {"x": {"type": "string"}}}
    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(50):
                cache.get_or_compute(
                    server_slug="srv", tool_name="x", raw_schema=raw
                )
        except Exception as exc:  # pragma: no cover - test only
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # All calls should resolve to a single entry: 1 miss + many hits.
    stats = cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] >= 50 * 8 - 1


def test_cache_returns_sanitised_output():
    # The output must be safe to feed directly to the validator.
    from app.services.tool_validation import validate_tool_args

    cache = SchemaCache(capacity=4)
    raw = {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["r", "g", "b"] * 50},
            "nested": {"oneOf": [{"type": "string"}]},
        },
        "required": ["color"],
    }
    out = cache.get_or_compute(
        server_slug="srv", tool_name="t", raw_schema=raw
    )
    # Strip compound keyword.
    assert "oneOf" not in out["properties"]["nested"]
    # Enum is capped.
    assert len(out["properties"]["color"]["enum"]) <= 21  # 20 + "... N more"
    # And validation works.
    errors = validate_tool_args(
        "t",
        {"color": "r"},
        {"type": "function", "function": {"name": "t", "parameters": out}},
    )
    assert errors == []
