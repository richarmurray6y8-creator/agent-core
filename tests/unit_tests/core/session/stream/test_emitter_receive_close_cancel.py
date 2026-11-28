# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Timed stream queue operations preserve caller cancellation."""

import asyncio

import pytest

from openjiuwen.core.session.stream.emitter import AsyncStreamQueue


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["receive", "close"])
async def test_cancellation_propagates_when_timed_operation_finishes(operation: str) -> None:
    queue = AsyncStreamQueue()
    if operation == "close":
        await queue.send("chunk")

    async def run_operation() -> None:
        await getattr(queue, operation)(timeout=1)
        await asyncio.sleep(0)

    task = asyncio.create_task(run_operation())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    if operation == "receive":
        queue._stream_queue.put_nowait("chunk")
    else:
        assert queue._stream_queue.get_nowait() == "chunk"
        queue._stream_queue.task_done()
    # The queue waiter completes just before the caller is cancelled.
    asyncio.get_running_loop().call_soon(task.cancel)

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_receive_times_out_on_empty_queue_and_can_retry() -> None:
    queue = AsyncStreamQueue()
    with pytest.raises(TimeoutError):
        await queue.receive(timeout=0.01)

    await queue.send("chunk")
    assert await queue.receive(timeout=1) == "chunk"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 0, -1])
async def test_receive_without_timeout_waits_for_a_chunk(timeout: float | None) -> None:
    queue = AsyncStreamQueue()
    receive_task = asyncio.create_task(queue.receive(timeout=timeout))
    await asyncio.sleep(0)
    assert not receive_task.done()

    await queue.send("chunk")
    assert await asyncio.wait_for(receive_task, timeout=1) == "chunk"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 0, -1])
async def test_close_without_timeout_waits_for_delivery(timeout: float | None) -> None:
    queue = AsyncStreamQueue(maxsize=1)
    await queue.send("chunk")
    close_task = asyncio.create_task(queue.close(timeout=timeout))
    await asyncio.sleep(0)
    assert queue.is_closed
    assert not close_task.done()

    assert queue._stream_queue.get_nowait() == "chunk"
    queue._stream_queue.task_done()
    await asyncio.wait_for(close_task, timeout=1)


@pytest.mark.asyncio
async def test_close_clears_full_queue_on_timeout() -> None:
    queue = AsyncStreamQueue(maxsize=1)
    await queue.send("chunk")
    await queue.close(timeout=0.01)

    assert queue.is_closed
    assert queue._stream_queue.empty()
    await asyncio.wait_for(queue._stream_queue.join(), timeout=1)
    await queue.close()
    with pytest.raises(RuntimeError, match="StreamQueue is already closed"):
        await queue.receive()
