"""Integration tests for assemble_static_prefix_cached."""

from __future__ import annotations

import pytest

from app.config import AppConfig
from app.services.prompt import (
    assemble_static_prefix,
    assemble_static_prefix_cached,
)
from app.services.prompt_cache import PromptCache, fingerprint_config, fingerprint_tool_set


@pytest.fixture
def fresh_cache() -> PromptCache:
    return PromptCache(capacity=8, ttl_seconds=60.0)


def test_cached_prefix_matches_direct_assembly(fresh_cache: PromptCache) -> None:
    cfg = AppConfig()
    direct = assemble_static_prefix(
        cfg,
        thinking_mode="medium",
        terminal_tool_enabled=False,
        tool_instruction_lines=["read_file: read a file"],
    )
    cached = assemble_static_prefix_cached(
        cfg,
        thinking_mode="medium",
        terminal_tool_enabled=False,
        tool_instruction_lines=["read_file: read a file"],
        cache=fresh_cache,
    )
    assert cached == direct


def test_cached_prefix_hits_after_first_call(fresh_cache: PromptCache) -> None:
    cfg = AppConfig()
    args = dict(
        thinking_mode="high",
        terminal_tool_enabled=True,
        tool_instruction_lines=["write_file: write"],
    )
    # First call: miss
    assemble_static_prefix_cached(cfg, cache=fresh_cache, **args)
    # Second call: hit
    assemble_static_prefix_cached(cfg, cache=fresh_cache, **args)
    stats = fresh_cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1


def test_cache_miss_when_thinking_mode_changes(fresh_cache: PromptCache) -> None:
    cfg = AppConfig()
    assemble_static_prefix_cached(
        cfg, thinking_mode="low", tool_instruction_lines=None, cache=fresh_cache
    )
    assemble_static_prefix_cached(
        cfg, thinking_mode="high", tool_instruction_lines=None, cache=fresh_cache
    )
    stats = fresh_cache.stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 0


def test_cache_miss_when_tool_set_changes(fresh_cache: PromptCache) -> None:
    cfg = AppConfig()
    assemble_static_prefix_cached(
        cfg, thinking_mode="medium", tool_instruction_lines=["a: 1"], cache=fresh_cache
    )
    assemble_static_prefix_cached(
        cfg, thinking_mode="medium", tool_instruction_lines=["a: 2"], cache=fresh_cache
    )
    assert fresh_cache.stats()["misses"] == 2


def test_cache_miss_when_config_system_prompt_changes(fresh_cache: PromptCache) -> None:
    cfg_a = AppConfig(system_prompt="Hello")
    cfg_b = AppConfig(system_prompt="Different")
    assemble_static_prefix_cached(
        cfg_a, thinking_mode="medium", tool_instruction_lines=None, cache=fresh_cache
    )
    assemble_static_prefix_cached(
        cfg_b, thinking_mode="medium", tool_instruction_lines=None, cache=fresh_cache
    )
    assert fresh_cache.stats()["misses"] == 2


def test_cache_uses_fingerprint_config_helpers() -> None:
    cfg = AppConfig(system_prompt="hi", session_memory_profile="verbose")
    fp = fingerprint_config(cfg)
    assert isinstance(fp, str) and len(fp) == 16
    fp2 = fingerprint_tool_set(["a", "b"])
    assert isinstance(fp2, str) and len(fp2) == 16
