"""Cache for the static portion of the assembled system prompt.

The system prompt that ``prompt.assemble_prompt`` produces has two layers:

1. A *static* prefix that depends on (a) the :class:`AppConfig` shape,
   (b) the thinking mode, and (c) the set of tool instruction lines that
   are passed in.  This prefix is identical for every turn in a session
   as long as the configuration and the tool set do not change.
2. A *per-turn* suffix that is composed of pinned messages, durable facts,
   the working set, retrieved chunks, the checkpoint summary, and the
   last 12 conversational messages.

Only the first layer is eligible for caching.  Re-assembling it for every
turn is wasteful when the model is invoked many times in quick succession
(streaming responses, tool loops, retries) and the configuration has not
changed.

The cache is keyed on a structural fingerprint so that any change to the
underlying config or tool set produces a fresh entry automatically.  Each
entry expires after ``ttl_seconds`` to ensure that drift in DB-derived
state (``pinned_text``, ``durable_facts`` etc.) does not keep a stale
prefix alive forever; the prefix is cheap to recompute, so a short TTL
(e.g. 5 minutes) is the right default.

The implementation is intentionally simple:

* An LRU keyed by the tuple ``(config_hash, thinking_mode, tool_set_hash)``.
* Each entry stores ``(prefix_text, expires_at)``.
* Reads acquire a short-lived lock; mutations (insert / evict) acquire
  the same lock to keep the LRU consistent.
* A small stats surface (hits / misses / evictions) is exposed for
  telemetry.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Iterable

from .telemetry import prompt_cache_ops as _prompt_cache_ops


DEFAULT_TTL_SECONDS = 300.0  # 5 minutes
DEFAULT_CAPACITY = 64


def _stable_hash(payload: Any) -> str:
    """Return a short, stable hash of a JSON-friendly payload.

    The payload must already be normalised (e.g. ``sorted`` keys, no
    mutable containers) so that two semantically equal inputs produce
    the same hash.
    """
    if not isinstance(payload, str):
        payload = repr(sorted(payload.items()) if isinstance(payload, dict) else list(payload))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def fingerprint_config(cfg: Any) -> str:
    """Compute a fingerprint of the parts of ``AppConfig`` that affect the
    static prompt prefix.

    Only fields that are actually rendered into the prefix are included;
    the model endpoint, API key and storage backend do not influence the
    prompt text and are intentionally excluded.
    """
    payload = {
        "system_prompt": getattr(cfg, "system_prompt", ""),
        "session_memory_profile": getattr(cfg, "session_memory_profile", ""),
    }
    return _stable_hash(payload)


def fingerprint_tool_set(tool_instruction_lines: Iterable[str] | None) -> str:
    """Hash the tool instruction lines.  The order is part of the input
    (the prompt concatenates them with newlines) so the list is hashed
    verbatim, but we strip trailing whitespace to keep cosmetic changes
    from invalidating the cache."""
    if not tool_instruction_lines:
        return "none"
    cleaned = [line.strip() for line in tool_instruction_lines if line and line.strip()]
    return _stable_hash(cleaned)


def make_cache_key(
    *,
    config_hash: str,
    thinking_mode: str,
    tool_set_hash: str,
) -> tuple[str, str, str]:
    """Return the (immutable) tuple key used by :class:`PromptCache`."""
    return (config_hash, str(thinking_mode), tool_set_hash)


class PromptCache:
    """Thread-safe LRU+TTL cache for assembled prompt prefixes."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._capacity = int(capacity)
        self._ttl = float(ttl_seconds)
        self._entries: OrderedDict[tuple[str, str, str], tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def get(self, key: tuple[str, str, str]) -> str | None:
        """Return the cached prefix, or ``None`` if missing/expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                _prompt_cache_ops.with_labels(op="miss").inc()
                return None
            value, expires_at = entry
            if expires_at <= now:
                # Expired: drop and report a miss.
                del self._entries[key]
                self._expirations += 1
                self._misses += 1
                _prompt_cache_ops.with_labels(op="expired").inc()
                return None
            # Mark as recently used.
            self._entries.move_to_end(key)
            self._hits += 1
            _prompt_cache_ops.with_labels(op="hit").inc()
            return value

    def put(self, key: tuple[str, str, str], value: str) -> None:
        """Store a prefix.  Evicts the oldest entry if at capacity."""
        expires_at = time.monotonic() + self._ttl
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self._entries[key] = (value, expires_at)
                return
            while len(self._entries) >= self._capacity:
                self._entries.popitem(last=False)
                self._evictions += 1
                _prompt_cache_ops.with_labels(op="evict").inc()
            self._entries[key] = (value, expires_at)

    def invalidate(self) -> None:
        """Drop all entries.  Used when the config is hot-reloaded."""
        with self._lock:
            if self._entries:
                _prompt_cache_ops.with_labels(op="invalidate").inc()
            self._entries.clear()

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "size": len(self._entries),
                "capacity": self._capacity,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0


# Process-wide singleton.  The orchestrator imports this directly; tests
# instantiate their own :class:`PromptCache` to keep assertions hermetic.
default_cache = PromptCache()
