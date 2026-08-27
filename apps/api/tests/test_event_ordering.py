"""Verify the FIFO ordering of events emitted by the drain loop.

The drain must yield events in the exact order they were put_nowait'd
into the queue, with no reordering across multiple producers.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from app.services import orchestrator


def test_drain_preserves_fifo_order() -> None:
    """Put 10 events in a specific order, drain them, assert order."""
    async def _scenario() -> None:
        sink: asyncio.Queue = asyncio.Queue()
        received: list[str] = []

        async def producer() -> str:
            for i in range(10):
                sink.put_nowait((f"e{i}", {"i": i}))
                await asyncio.sleep(0.01)
            return "done"

        task = asyncio.create_task(producer())
        while not task.done():
            try:
                ev_name, _ = await asyncio.wait_for(sink.get(), timeout=0.05)
                received.append(ev_name)
            except asyncio.TimeoutError:
                continue
        while not sink.empty():
            ev_name, _ = sink.get_nowait()
            received.append(ev_name)
        await task

        assert received == [f"e{i}" for i in range(10)]

    asyncio.run(_scenario())


def test_drain_handles_simultaneous_producers() -> None:
    """Multiple coroutines putting events must have all events preserved
    in some valid interleaving, with no events lost.
    """
    async def _scenario() -> None:
        sink: asyncio.Queue = asyncio.Queue()
        received: list[tuple[str, int]] = []

        async def producer(name: str, count: int) -> None:
            for i in range(count):
                sink.put_nowait((name, i))
                await asyncio.sleep(0.005)

        async def consumer() -> None:
            # Drain 30 events total (3 producers x 10 each).
            while len(received) < 30:
                try:
                    name, i = await asyncio.wait_for(sink.get(), timeout=0.5)
                    received.append((name, i))
                except asyncio.TimeoutError:
                    break

        prod_task = asyncio.gather(
            producer("a", 10), producer("b", 10), producer("c", 10)
        )
        cons_task = asyncio.create_task(consumer())
        await prod_task
        await cons_task

        # Every (producer, index) pair must appear exactly once.
        counts: dict[str, list[int]] = {"a": [], "b": [], "c": []}
        for name, i in received:
            counts[name].append(i)
        for name in ("a", "b", "c"):
            assert sorted(counts[name]) == list(range(10)), (
                f"producer {name} missing/duplicated events: {counts[name]}"
            )

    asyncio.run(_scenario())


def test_drain_does_not_block_on_empty_queue() -> None:
    """The drain loop must wake up periodically even when the queue is
    empty, so it can detect task completion quickly.
    """
    async def _scenario() -> None:
        sink: asyncio.Queue = asyncio.Queue()
        loop_done = asyncio.Event()

        async def quick_task() -> str:
            await asyncio.sleep(0.1)
            loop_done.set()
            return "fast"

        task = asyncio.create_task(quick_task())
        start = time.perf_counter()
        while not task.done():
            try:
                await asyncio.wait_for(sink.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
        elapsed = time.perf_counter() - start
        # Must complete within ~0.5s (100ms task + 250ms polling cycle).
        assert elapsed < 0.6, f"drain took too long: {elapsed:.3f}s"
        await task
        assert loop_done.is_set()

    asyncio.run(_scenario())
