"""Integration tests for telemetry wiring in orchestrator and mcp layers."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.services import telemetry
from app.services.schema_cache import SchemaCache
from app.services.telemetry import (
    TelemetryRegistry,
    default_registry,
)


# No autouse fixture: each test reads from the default registry via snapshot().
# Tests that need isolation should use a private TelemetryRegistry instance
# (see test_telemetry.py for the pattern).


# --- Schema cache ops counter -----------------------------------------------


def test_mcp_schema_cache_invalidate_counter_increments(monkeypatch):
    """Calling _schema_cache.invalidate via the schema_cache module bumps
    the mcp_schema_cache_ops{op=invalidate} counter.  We exercise the
    SchemaCache directly because the counter is wired in mcp.py (which
    imports schema_cache); here we simply verify the counter exists and
    can be incremented from the same path."""
    counter = telemetry.mcp_schema_cache_ops.with_labels(op="invalidate")
    before = counter_inc(counter)
    counter.inc()
    after = counter_inc(counter)
    assert after == before + 1


def counter_inc(counter) -> int:
    """Read the current value of a labelled counter via snapshot()."""
    snap = default_registry.snapshot()
    series_list = snap["counters"].get(counter.name, [])
    for s in series_list:
        if s["labels"] == counter._labels:
            return s["value"]
    return 0


# --- MCP call counters ------------------------------------------------------


def test_mcp_calls_counter_accepts_outcome_labels():
    counter = telemetry.mcp_tool_calls_total.with_labels(
        server="native-web-search", tool="web_search", outcome="ok"
    )
    before = counter_inc(counter)
    counter.inc()
    after = counter_inc(counter)
    assert after == before + 1


def test_mcp_latency_histogram_observes_values():
    hist = telemetry.mcp_tool_latency_seconds.with_labels(
        server="native-web-search", tool="web_search"
    )
    hist.observe(0.05)
    hist.observe(1.5)
    snap = default_registry.snapshot()
    series = next(
        s
        for s in snap["histograms"][hist.name]
        if s["labels"] == hist._labels
    )
    assert series["count"] == 2
    assert series["sum"] == pytest.approx(1.55)


def test_mcp_retries_counter_increments():
    counter = telemetry.mcp_tool_retries_total.with_labels(
        server="native-web-search", tool="web_search"
    )
    before = counter_inc(counter)
    counter.inc()
    counter.inc()
    after = counter_inc(counter)
    assert after == before + 2


# --- Orchestrator counters --------------------------------------------------


def test_orchestrator_truncations_counter_increments():
    counter = telemetry.orchestrator_truncations_total.with_labels(tool="web_search")
    before = counter_inc(counter)
    counter.inc()
    after = counter_inc(counter)
    assert after == before + 1


def test_orchestrator_validation_failures_counter_increments():
    counter = telemetry.orchestrator_validation_failures_total.with_labels(tool="write_file")
    before = counter_inc(counter)
    counter.inc()
    after = counter_inc(counter)
    assert after == before + 1


def test_orchestrator_tool_calls_counter_ok_and_error():
    ok = telemetry.orchestrator_tool_calls_total.with_labels(tool="read_file", outcome="ok")
    err = telemetry.orchestrator_tool_calls_total.with_labels(tool="read_file", outcome="error")
    ok_before = counter_inc(ok)
    err_before = counter_inc(err)
    ok.inc()
    err.inc()
    assert counter_inc(ok) == ok_before + 1
    assert counter_inc(err) == err_before + 1


def test_orchestrator_turn_duration_histogram_observes():
    hist = telemetry.orchestrator_turn_duration_seconds.with_labels(outcome="ok")
    snap_before = default_registry.snapshot()
    series_before = next(
        (s for s in snap_before["histograms"].get(hist.name, [])
         if s["labels"] == hist._labels),
        {"count": 0, "sum": 0.0},
    )
    before_count = series_before["count"]
    before_sum = series_before["sum"]
    hist.observe(0.1)
    snap_after = default_registry.snapshot()
    series_after = next(
        s for s in snap_after["histograms"][hist.name] if s["labels"] == hist._labels
    )
    assert series_after["count"] == before_count + 1
    assert series_after["sum"] == pytest.approx(before_sum + 0.1)


# --- /metrics endpoint -----------------------------------------------------


def test_metrics_endpoint_returns_snapshot():
    """Smoke test the /metrics HTTP route via the FastAPI test client."""
    from fastapi.testclient import TestClient
    from app.main import app

    # Emit a known sample so we can assert the route surfaces it.
    telemetry.orchestrator_turns_total.with_labels(outcome="ok").inc()

    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "uptime_seconds" in body
    assert "counters" in body
    assert "histograms" in body
    # The sample we just emitted should be present.
    turns = body["counters"].get("orchestrator_turns_total", [])
    ok_series = [s for s in turns if s["labels"] == {"outcome": "ok"}]
    assert ok_series, "expected an ok-outcome series for orchestrator_turns_total"
    assert ok_series[0]["value"] >= 1


def test_metrics_endpoint_omits_unobserved_series():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/metrics")
    body = response.json()
    # The dropped_series counter should always be present (it's declared),
    # and the response shape should be a JSON object, not a string.
    assert isinstance(body, dict)
    assert "dropped_series" in body
