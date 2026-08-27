"""Lightweight in-process telemetry for the orchestrator and MCP layer.

The collector is intentionally dependency-free so it can be imported in unit
tests without spinning up a tracing backend.  Metrics are kept in memory and
exposed through a small JSON snapshot via :func:`collect_snapshot` (or the
``/metrics`` HTTP route wired up in ``app.main``).

Two primitives are supported:

* :class:`Counter` — monotonically increasing integer with optional labels.
* :class:`Histogram` — fixed-bucket count distribution with optional labels.

Counters and histograms are stored on a singleton :class:`TelemetryRegistry`
under stable string keys.  All public methods are thread-safe (a single
``threading.Lock`` guards mutations; reads of the snapshot are also taken
under the lock to keep the returned view consistent).
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

# A reasonable default bucket layout for latencies measured in milliseconds.
# The buckets are inclusive upper bounds; +Inf is implicit (overflow bucket).
DEFAULT_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
    30000.0,
    60000.0,
)

# Cap on label cardinality to bound memory usage if a label key ever goes
# wild (e.g. a free-form error message).  When the cap is exceeded the
# additional series is silently dropped and recorded in ``dropped_series``.
MAX_LABEL_COMBINATIONS = 512


@dataclass
class _CounterSeries:
    value: int = 0
    labels: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {"value": self.value, "labels": dict(self.labels)}


@dataclass
class _HistogramSeries:
    count: int = 0
    sum: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    buckets: dict[float, int] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def snapshot(self, bucket_bounds: tuple[float, ...]) -> dict[str, Any]:
        bucket_snapshot: list[dict[str, Any]] = []
        for bound in bucket_bounds:
            bucket_snapshot.append(
                {"le": bound, "count": int(self.buckets.get(bound, 0))}
            )
        # The +Inf bucket is always present and equals the total count.
        bucket_snapshot.append({"le": "+Inf", "count": int(self.count)})
        return {
            "count": int(self.count),
            "sum": float(self.sum),
            "min": None if self.count == 0 else float(self.min),
            "max": None if self.count == 0 else float(self.max),
            "buckets": bucket_snapshot,
            "labels": dict(self.labels),
        }


class Counter:
    """A labelled monotonic counter.

    Use :meth:`inc` to add to the value.  The combination of metric name and
    label values forms a unique series; duplicate series share state.
    """

    __slots__ = ("_registry", "_name", "_label_keys", "_labels")

    def __init__(
        self,
        registry: "TelemetryRegistry",
        name: str,
        label_keys: tuple[str, ...] = (),
        labels: dict[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._name = name
        self._label_keys = tuple(label_keys)
        self._labels = dict(labels or {})

    @property
    def name(self) -> str:
        return self._name

    def inc(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("Counter.inc only accepts non-negative amounts")
        self._registry._add_counter(self._name, self._label_keys, self._labels, int(amount))

    def with_labels(self, **labels: str) -> "Counter":
        merged: dict[str, str] = dict(self._labels)
        for key, value in labels.items():
            if key not in self._label_keys:
                raise KeyError(
                    f"label '{key}' was not declared for counter '{self._name}'"
                )
            merged[key] = str(value)
        return Counter(self._registry, self._name, self._label_keys, merged)


class Histogram:
    """A labelled histogram with fixed bucket bounds.

    Use :meth:`observe` to record a single sample.  Like :class:`Counter`,
    label combinations are bounded by ``MAX_LABEL_COMBINATIONS`` to avoid
    unbounded memory growth in the face of high-cardinality labels.
    """

    __slots__ = (
        "_registry",
        "_name",
        "_label_keys",
        "_labels",
        "_buckets",
    )

    def __init__(
        self,
        registry: "TelemetryRegistry",
        name: str,
        *,
        label_keys: tuple[str, ...] = (),
        labels: dict[str, str] | None = None,
        buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_MS,
    ) -> None:
        self._registry = registry
        self._name = name
        self._label_keys = tuple(label_keys)
        self._labels = dict(labels or {})
        self._buckets = tuple(sorted(buckets))

    @property
    def name(self) -> str:
        return self._name

    @property
    def buckets(self) -> tuple[float, ...]:
        return self._buckets

    def observe(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Histogram.observe requires a number, got {type(value).__name__}")
        if math.isnan(value):
            return  # NaN samples are silently skipped
        self._registry._add_histogram(
            self._name,
            self._label_keys,
            self._labels,
            float(value),
            self._buckets,
        )

    def with_labels(self, **labels: str) -> "Histogram":
        merged: dict[str, str] = dict(self._labels)
        for key, value in labels.items():
            if key not in self._label_keys:
                raise KeyError(
                    f"label '{key}' was not declared for histogram '{self._name}'"
                )
            merged[key] = str(value)
        return Histogram(
            self._registry,
            self._name,
            label_keys=self._label_keys,
            labels=merged,
            buckets=self._buckets,
        )


class TelemetryRegistry:
    """In-process metrics registry.  Use the :data:`default_registry` instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, frozenset], _CounterSeries] = {}
        self._histograms: dict[tuple[str, frozenset], _HistogramSeries] = {}
        self._counter_decl: dict[str, tuple[str, ...]] = {}
        self._histogram_decl: dict[str, dict[str, Any]] = {}
        self._dropped_series: int = 0
        self._started_at: float = time.time()

    # -- declaration helpers -------------------------------------------------
    def counter(
        self, name: str, label_keys: Iterable[str] = ()
    ) -> Counter:
        keys = tuple(label_keys)
        with self._lock:
            existing = self._counter_decl.get(name)
            if existing is not None and existing != keys:
                raise ValueError(
                    f"counter '{name}' already declared with labels {existing!r}, "
                    f"cannot redeclare with {keys!r}"
                )
            self._counter_decl.setdefault(name, keys)
        return Counter(self, name, keys)

    def histogram(
        self,
        name: str,
        *,
        label_keys: Iterable[str] = (),
        buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_MS,
    ) -> Histogram:
        keys = tuple(label_keys)
        with self._lock:
            existing = self._histogram_decl.get(name)
            if existing is not None and (
                existing.get("label_keys") != keys
                or tuple(existing.get("buckets", ())) != tuple(buckets)
            ):
                raise ValueError(
                    f"histogram '{name}' already declared with different shape"
                )
            self._histogram_decl.setdefault(
                name, {"label_keys": keys, "buckets": tuple(buckets)}
            )
        return Histogram(self, name, label_keys=keys, buckets=buckets)

    # -- internal mutation ---------------------------------------------------
    def _add_counter(
        self,
        name: str,
        declared_keys: tuple[str, ...],
        labels: dict[str, str],
        amount: int,
    ) -> None:
        key = self._series_key(name, declared_keys, labels)
        with self._lock:
            if key not in self._counters and len(self._counters) >= MAX_LABEL_COMBINATIONS:
                self._dropped_series += 1
                return
            series = self._counters.get(key)
            if series is None:
                series = _CounterSeries(labels=dict(labels))
                self._counters[key] = series
            series.value += amount

    def _add_histogram(
        self,
        name: str,
        declared_keys: tuple[str, ...],
        labels: dict[str, str],
        value: float,
        buckets: tuple[float, ...],
    ) -> None:
        key = self._series_key(name, declared_keys, labels)
        with self._lock:
            if key not in self._histograms and len(self._histograms) >= MAX_LABEL_COMBINATIONS:
                self._dropped_series += 1
                return
            series = self._histograms.get(key)
            if series is None:
                series = _HistogramSeries(
                    labels=dict(labels),
                    buckets={bound: 0 for bound in buckets},
                )
                self._histograms[key] = series
            series.count += 1
            series.sum += value
            if value < series.min:
                series.min = value
            if value > series.max:
                series.max = value
            for bound in buckets:
                if value <= bound:
                    series.buckets[bound] = series.buckets.get(bound, 0) + 1

    @staticmethod
    def _series_key(
        name: str, declared_keys: tuple[str, ...], labels: dict[str, str]
    ) -> tuple[str, frozenset]:
        # Use a deterministic, hashable encoding of (name, label set).
        items = tuple(sorted((k, str(labels.get(k, ""))) for k in declared_keys))
        return (name, frozenset(items))

    # -- snapshot / reset ----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (_, _), series in self._counters.items():
                # The series key encodes the name; look it up via declaration.
                # We rebuild name lookup from the declaration table.
                pass

            # We can group by (name) by relying on the fact that the counter
            # name is the first element of the key.  Sort for determinism.
            counters_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (metric_name, _), series in self._counters.items():
                counters_grouped[metric_name].append(series.snapshot())

            histograms_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (metric_name, _), series in self._histograms.items():
                decl = self._histogram_decl.get(metric_name, {})
                buckets = tuple(decl.get("buckets", DEFAULT_LATENCY_BUCKETS_MS))
                histograms_grouped[metric_name].append(series.snapshot(buckets))

            # Stable order for assertions and for human readability.
            counters_out = {
                name: sorted(series_list, key=lambda s: tuple(sorted(s["labels"].items())))
                for name, series_list in sorted(counters_grouped.items())
            }
            histograms_out = {
                name: sorted(series_list, key=lambda s: tuple(sorted(s["labels"].items())))
                for name, series_list in sorted(histograms_grouped.items())
            }

            return {
                "uptime_seconds": max(0.0, time.time() - self._started_at),
                "dropped_series": int(self._dropped_series),
                "counters": counters_out,
                "histograms": histograms_out,
                "declared": {
                    "counters": {
                        name: list(keys)
                        for name, keys in sorted(self._counter_decl.items())
                    },
                    "histograms": {
                        name: {
                            "labels": list(decl["label_keys"]),
                            "buckets": list(decl["buckets"]),
                        }
                        for name, decl in sorted(self._histogram_decl.items())
                    },
                },
            }

    def reset(self) -> None:
        """Wipe all series and declarations.  Intended for tests only."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._counter_decl.clear()
            self._histogram_decl.clear()
            self._dropped_series = 0
            self._started_at = time.time()


# A single process-wide registry.  Tests that need isolation should instantiate
# their own ``TelemetryRegistry`` rather than mutating this one.
default_registry = TelemetryRegistry()


# -- convenience handles for common metrics ---------------------------------
# These are eagerly created so callers can ``inc()``/``observe()`` without
# worrying about declaration order, but the registry is happy to accept
# declarations at any time.
orchestrator_turns_total = default_registry.counter(
    "orchestrator_turns_total", label_keys=("outcome",)
)
orchestrator_tool_calls_total = default_registry.counter(
    "orchestrator_tool_calls_total", label_keys=("tool", "outcome")
)
orchestrator_validation_failures_total = default_registry.counter(
    "orchestrator_validation_failures_total", label_keys=("tool",)
)
orchestrator_truncations_total = default_registry.counter(
    "orchestrator_truncations_total", label_keys=("tool",)
)
orchestrator_turn_duration_seconds = default_registry.histogram(
    "orchestrator_turn_duration_seconds",
    label_keys=("outcome",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)
orchestrator_tool_loop_iterations = default_registry.histogram(
    "orchestrator_tool_loop_iterations",
    label_keys=("outcome",),
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20),
)

mcp_tool_calls_total = default_registry.counter(
    "mcp_tool_calls_total", label_keys=("server", "tool", "outcome")
)
mcp_tool_latency_seconds = default_registry.histogram(
    "mcp_tool_latency_seconds",
    label_keys=("server", "tool"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
mcp_tool_retries_total = default_registry.counter(
    "mcp_tool_retries_total", label_keys=("server", "tool")
)
mcp_schema_cache_ops = default_registry.counter(
    "mcp_schema_cache_ops", label_keys=("op",)
)

prompt_cache_ops = default_registry.counter(
    "prompt_cache_ops", label_keys=("op",)
)  # op in {"hit", "miss", "expired", "evict", "invalidate"}
