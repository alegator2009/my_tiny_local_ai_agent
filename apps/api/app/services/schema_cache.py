"""Memoisation of sanitised MCP/tool schemas.

Calling :func:`schema_sanitizer.sanitize_schema` on every tool registration
is cheap, but calling it again on *every* orchestrator turn is wasteful when
the underlying schema has not changed. ``SchemaCache`` stores the last
sanitised result keyed by ``(server_slug, tool_name, fingerprint)`` so that
hot reads are an O(1) dict lookup.

The cache is deliberately simple: a fixed-capacity LRU, no eviction policy
fanfare, no persistence. The MCP server list is small (typically a handful
of servers with a few dozen tools), the orchestrator runs in a single
process, and tools are registered once at startup. The cache mostly exists
to make ``tool_schemas()`` cheap to call repeatedly from the request loop.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from .schema_sanitizer import sanitize_schema, schema_fingerprint


DEFAULT_CACHE_CAPACITY = 256


class SchemaCache:
    """Thread-safe LRU cache for sanitised schemas."""

    def __init__(self, capacity: int = DEFAULT_CACHE_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._entries: "OrderedDict[tuple[str, str, str], dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_compute(
        self,
        *,
        server_slug: str,
        tool_name: str,
        raw_schema: Any,
    ) -> dict[str, Any]:
        """Return the sanitised schema, computing it on cache miss."""
        fingerprint = schema_fingerprint(raw_schema)
        key = (server_slug, tool_name, fingerprint)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return cached
        # Compute outside the lock so the call doesn't serialise other readers.
        sanitised = sanitize_schema(raw_schema)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                # Another thread beat us to it; reuse theirs.
                self._entries.move_to_end(key)
                self._hits += 1
                return existing
            self._entries[key] = sanitised
            self._entries.move_to_end(key)
            self._misses += 1
            self._evict_if_needed()
        return sanitised

    def invalidate(self, server_slug: str | None = None) -> int:
        """Drop cached entries. Returns the number of entries removed.

        When ``server_slug`` is ``None`` the entire cache is cleared; this is
        what callers should use when an MCP server is restarted.
        """
        with self._lock:
            if server_slug is None:
                removed = len(self._entries)
                self._entries.clear()
                return removed
            kept = OrderedDict()
            removed = 0
            for key, value in self._entries.items():
                if key[0] == server_slug:
                    removed += 1
                else:
                    kept[key] = value
            self._entries = kept
            return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._entries),
                "capacity": self._capacity,
                "hits": self._hits,
                "misses": self._misses,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)


# Module-level singleton used by the orchestrator. Tests should instantiate
# their own ``SchemaCache`` to stay isolated.
default_cache = SchemaCache()
