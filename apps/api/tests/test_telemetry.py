"""Tests for the in-process telemetry registry."""

from __future__ import annotations

import math
import threading

import pytest

from app.services import telemetry
from app.services.telemetry import (
    Counter,
    DEFAULT_LATENCY_BUCKETS_MS,
    Histogram,
    TelemetryRegistry,
)


@pytest.fixture
def registry() -> TelemetryRegistry:
    return TelemetryRegistry()


# --- Counter -----------------------------------------------------------------


def test_counter_starts_at_zero(registry: TelemetryRegistry) -> None:
    counter = registry.counter("test_total")
    counter.inc()  # Snapshot only emits series that have been touched.
    snap = registry.snapshot()
    assert snap["counters"]["test_total"][0]["value"] == 1
    assert snap["declared"]["counters"]["test_total"] == []


def test_counter_inc_increments(registry: TelemetryRegistry) -> None:
    counter = registry.counter("hits_total")
    counter.inc()
    counter.inc()
    counter.inc(5)
    snap = registry.snapshot()
    series = snap["counters"]["hits_total"]
    assert series[0]["value"] == 7
    assert series[0]["labels"] == {}


def test_counter_rejects_negative(registry: TelemetryRegistry) -> None:
    counter = registry.counter("neg_total")
    with pytest.raises(ValueError):
        counter.inc(-1)


def test_counter_with_labels_creates_separate_series(registry: TelemetryRegistry) -> None:
    counter = registry.counter("events_total", label_keys=("kind",))
    counter.with_labels(kind="tool").inc()
    counter.with_labels(kind="tool").inc(2)
    counter.with_labels(kind="llm").inc(4)
    snap = registry.snapshot()
    by_kind = {s["labels"]["kind"]: s["value"] for s in snap["counters"]["events_total"]}
    assert by_kind == {"tool": 3, "llm": 4}


def test_counter_with_labels_rejects_undeclared_key(registry: TelemetryRegistry) -> None:
    counter = registry.counter("x_total", label_keys=("a",))
    with pytest.raises(KeyError):
        counter.with_labels(b="oops")


def test_counter_redeclaration_with_different_labels_raises(registry: TelemetryRegistry) -> None:
    registry.counter("dup_total", label_keys=("a",))
    with pytest.raises(ValueError):
        registry.counter("dup_total", label_keys=("b",))


# --- Histogram ---------------------------------------------------------------


def test_histogram_observe_records_count_sum_min_max(registry: TelemetryRegistry) -> None:
    h = registry.histogram("lat", buckets=(1.0, 5.0, 10.0))
    h.observe(0.5)
    h.observe(3.0)
    h.observe(11.0)
    snap = registry.snapshot()
    series = snap["histograms"]["lat"][0]
    assert series["count"] == 3
    assert series["sum"] == pytest.approx(14.5)
    assert series["min"] == 0.5
    assert series["max"] == 11.0


def test_histogram_buckets_count_inclusive_upper_bound(registry: TelemetryRegistry) -> None:
    h = registry.histogram("lat", buckets=(1.0, 5.0, 10.0))
    h.observe(1.0)  # bucket le=1.0
    h.observe(5.0)  # bucket le=5.0
    h.observe(10.0)  # bucket le=10.0
    h.observe(11.0)  # overflow -> only +Inf
    snap = registry.snapshot()
    series = snap["histograms"]["lat"][0]
    counts = {b["le"]: b["count"] for b in series["buckets"]}
    assert counts[1.0] == 1
    assert counts[5.0] == 2
    assert counts[10.0] == 3
    assert counts["+Inf"] == 4


def test_histogram_with_labels_share_buckets(registry: TelemetryRegistry) -> None:
    h = registry.histogram("lat", label_keys=("server",), buckets=(1.0, 10.0))
    h.with_labels(server="a").observe(0.5)
    h.with_labels(server="a").observe(5.0)
    h.with_labels(server="b").observe(9.0)
    snap = registry.snapshot()
    series_list = snap["histograms"]["lat"]
    by_server = {s["labels"]["server"]: s for s in series_list}
    assert by_server["a"]["count"] == 2
    assert by_server["b"]["count"] == 1
    assert by_server["a"]["sum"] == pytest.approx(5.5)


def test_histogram_rejects_non_number(registry: TelemetryRegistry) -> None:
    h = registry.histogram("lat")
    with pytest.raises(TypeError):
        h.observe("nope")  # type: ignore[arg-type]


def test_histogram_skips_nan(registry: TelemetryRegistry) -> None:
    h = registry.histogram("lat", buckets=(1.0,))
    h.observe(math.nan)
    h.observe(0.5)
    snap = registry.snapshot()
    assert snap["histograms"]["lat"][0]["count"] == 1


def test_histogram_redeclaration_with_different_buckets_raises(registry: TelemetryRegistry) -> None:
    registry.histogram("h", buckets=(1.0,))
    with pytest.raises(ValueError):
        registry.histogram("h", buckets=(1.0, 2.0))


# --- Snapshot / drop ---------------------------------------------------------


def test_snapshot_groups_by_metric_name(registry: TelemetryRegistry) -> None:
    a = registry.counter("a_total", label_keys=("k",))
    b = registry.counter("b_total", label_keys=("k",))
    a.with_labels(k="x").inc()
    b.with_labels(k="y").inc()
    snap = registry.snapshot()
    assert "a_total" in snap["counters"]
    assert "b_total" in snap["counters"]


def test_snapshot_is_deterministic_ordering(registry: TelemetryRegistry) -> None:
    h = registry.histogram("lat", label_keys=("server",), buckets=(1.0,))
    h.with_labels(server="z").observe(0.5)
    h.with_labels(server="a").observe(0.5)
    h.with_labels(server="m").observe(0.5)
    snap = registry.snapshot()
    order = [s["labels"]["server"] for s in snap["histograms"]["lat"]]
    assert order == ["a", "m", "z"]


def test_dropped_series_counter_increments(registry: TelemetryRegistry) -> None:
    # Use a private counter path to overflow MAX_LABEL_COMBINATIONS cheaply.
    for i in range(telemetry.MAX_LABEL_COMBINATIONS + 5):
        registry._add_counter(
            "drop_total", ("k",), {"k": str(i)}, 1
        )
    snap = registry.snapshot()
    assert snap["dropped_series"] >= 5
    # Only MAX_LABEL_COMBINATIONS distinct series should be stored.
    assert len(snap["counters"]["drop_total"]) == telemetry.MAX_LABEL_COMBINATIONS


def test_reset_clears_state(registry: TelemetryRegistry) -> None:
    c = registry.counter("x_total")
    c.inc(5)
    registry.reset()
    snap = registry.snapshot()
    assert "x_total" not in snap["counters"]
    assert "x_total" not in snap["declared"]["counters"]


def test_uptime_grows_over_time(registry: TelemetryRegistry) -> None:
    snap1 = registry.snapshot()
    # Sleep briefly; threads can be slow, so a tiny sleep is enough.
    import time
    time.sleep(0.01)
    snap2 = registry.snapshot()
    assert snap2["uptime_seconds"] >= snap1["uptime_seconds"]


# --- Thread safety -----------------------------------------------------------


def test_counter_is_thread_safe(registry: TelemetryRegistry) -> None:
    c = registry.counter("thread_total", label_keys=("worker",))
    n_threads = 8
    per_thread = 500

    def worker(idx: int) -> None:
        local = c.with_labels(worker=str(idx))
        for _ in range(per_thread):
            local.inc()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = registry.snapshot()
    series = snap["counters"]["thread_total"]
    total = sum(s["value"] for s in series)
    assert total == n_threads * per_thread
    assert len(series) == n_threads


def test_histogram_is_thread_safe(registry: TelemetryRegistry) -> None:
    h = registry.histogram("thread_lat", buckets=(1.0, 10.0))

    def worker() -> None:
        for _ in range(1000):
            h.observe(0.5)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = registry.snapshot()
    series = snap["histograms"]["thread_lat"][0]
    assert series["count"] == 4000
    assert series["sum"] == pytest.approx(2000.0)


# --- Default registry handles -----------------------------------------------


def test_default_registry_handles_exist() -> None:
    assert isinstance(telemetry.orchestrator_turns_total, Counter)
    assert isinstance(telemetry.orchestrator_turn_duration_seconds, Histogram)
    assert isinstance(telemetry.mcp_tool_calls_total, Counter)
    assert isinstance(telemetry.mcp_tool_latency_seconds, Histogram)
    assert telemetry.orchestrator_turns_total.name == "orchestrator_turns_total"
    assert telemetry.orchestrator_turn_duration_seconds.buckets == (
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    )


def test_default_buckets_are_ms_layout() -> None:
    assert DEFAULT_LATENCY_BUCKETS_MS[0] == 1.0
    assert DEFAULT_LATENCY_BUCKETS_MS[-1] == 60000.0
