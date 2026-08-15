from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import cast

THREAD_POLL_SECONDS = 0.005


async def run_in_isolated_thread[**P, R](
    operation: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Run one blocking operation without reusing an asyncio executor thread."""
    outcome: list[tuple[bool, object]] = []

    def run() -> None:
        try:
            value = operation(*args, **kwargs)
        except BaseException as error:
            outcome.append((False, error))
        else:
            outcome.append((True, value))

    operation_name = getattr(operation, "__name__", "operation").replace("_", "-")
    thread = threading.Thread(
        target=run,
        name=f"modeldeck-{operation_name}",
        daemon=True,
    )
    thread.start()
    while thread.is_alive():  # noqa: ASYNC110 - polling avoids cross-thread wake notifications.
        await asyncio.sleep(THREAD_POLL_SECONDS)
    thread.join()
    if not outcome:
        raise RuntimeError("The isolated operation thread exited without a result")
    succeeded, value = outcome[0]
    if not succeeded:
        raise cast(BaseException, value)
    return cast(R, value)


async def iterate_in_isolated_thread[T](
    iterator_factory: Callable[[], Iterator[T]],
) -> AsyncIterator[T]:
    """Iterate blocking model output without cross-thread event-loop notifications."""
    events: queue.SimpleQueue[tuple[bool | None, object | None]] = queue.SimpleQueue()

    def produce() -> None:
        try:
            for item in iterator_factory():
                events.put((True, item))
        except BaseException as error:
            events.put((False, error))
        else:
            events.put((None, None))

    thread = threading.Thread(target=produce, name="modeldeck-stream-producer", daemon=True)
    thread.start()
    while True:
        try:
            succeeded, value = events.get_nowait()
        except queue.Empty:
            await asyncio.sleep(THREAD_POLL_SECONDS)
            continue
        if succeeded is None:
            return
        if not succeeded:
            raise cast(BaseException, value)
        yield cast(T, value)
