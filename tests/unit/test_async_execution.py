from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import pytest
from modeldeck.async_execution import iterate_in_isolated_thread, run_in_isolated_thread

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_blocking_operations_use_distinct_isolated_threads() -> None:
    event_loop_thread = threading.current_thread()
    operation_threads: list[threading.Thread] = []

    def record(value: int) -> int:
        operation_threads.append(threading.current_thread())
        return value

    first = await asyncio.wait_for(run_in_isolated_thread(record, 1), timeout=1)
    second = await asyncio.wait_for(run_in_isolated_thread(record, 2), timeout=1)

    assert (first, second) == (1, 2)
    assert operation_threads[0] is not event_loop_thread
    assert operation_threads[1] is not event_loop_thread
    assert operation_threads[0] is not operation_threads[1]


@pytest.mark.asyncio
async def test_isolated_thread_propagates_operation_errors() -> None:
    def fail() -> None:
        raise ValueError("fixture failure")

    with pytest.raises(ValueError, match="fixture failure"):
        await asyncio.wait_for(run_in_isolated_thread(fail), timeout=1)


@pytest.mark.asyncio
async def test_blocking_iteration_uses_one_isolated_producer_thread() -> None:
    event_loop_thread = threading.current_thread()
    producer_threads: list[threading.Thread] = []

    def values():
        producer_threads.append(threading.current_thread())
        yield 1
        yield 2

    result = [item async for item in iterate_in_isolated_thread(values)]

    assert result == [1, 2]
    assert len(producer_threads) == 1
    assert producer_threads[0] is not event_loop_thread


def test_runtime_code_does_not_use_the_reusable_asyncio_executor() -> None:
    offenders: list[str] = []
    source_root = REPOSITORY_ROOT / "backend" / "modeldeck"
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_thread"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
            for node in ast.walk(tree)
        ):
            offenders.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert offenders == []
