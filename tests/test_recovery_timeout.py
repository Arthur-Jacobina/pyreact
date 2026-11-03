"""timeout(seconds): cancels the inner call cleanly, per attempt."""

import asyncio

import pytest

from pyreact.core import EventBus, HookContext
from pyreact.core.harness import Harness, through
from pyreact.core.recovery import BoundaryTimeout, retry, timeout


def test_slow_call_is_cancelled_and_raises_boundary_timeout():
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, ["recovery.timeout"])
    trace = []

    async def slow(b):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            trace.append("cancelled")
            raise
        finally:
            trace.append("cleanup")

    seen = {}

    async def spy(b, call_next):
        try:
            return await call_next(b)
        finally:
            seen["timed_out"] = b.attrs.get("timed_out")

    h = Harness([spy, timeout(0.05)])
    with pytest.raises(BoundaryTimeout) as exc:
        asyncio.run(through("llm", "qa", {}, slow, harness=h))
    assert isinstance(exc.value, TimeoutError) and exc.value.seconds == 0.05
    assert "llm 'qa' timed out" in str(exc.value)
    assert trace == ["cancelled", "cleanup"] and seen["timed_out"] is True
    assert got and got[-1].payload["seconds"] == 0.05


def test_fast_call_and_inner_timeout_error_pass_through():
    h = Harness([timeout(1.0)])

    async def fast(b):
        return "ok"

    async def own_timeout(b):
        raise TimeoutError("the API said so")

    assert asyncio.run(through("tool", "t", {}, fast, harness=h)) == "ok"
    with pytest.raises(TimeoutError) as exc:
        asyncio.run(through("tool", "t", {}, own_timeout, harness=h))
    assert not isinstance(exc.value, BoundaryTimeout)


def test_per_attempt_under_retry_and_callable_seconds():
    runs = []

    async def slow_then_fast(b):
        runs.append(b.attempt)
        if b.attempt == 1:
            await asyncio.sleep(5)
        return f"attempt {b.attempt}"

    h = Harness([retry(max_attempts=2, backoff=lambda n: 0),
                 timeout(lambda b: 0.05 if b.name == "slow" else None)])
    assert asyncio.run(through("tool", "slow", {}, slow_then_fast, harness=h)) == "attempt 2"
    assert runs == [1, 2]


def test_outer_cancellation_is_not_turned_into_a_timeout():
    async def slow(b):
        await asyncio.sleep(5)

    h = Harness([timeout(10)])

    async def main():
        task = asyncio.create_task(through("tool", "t", {}, slow, harness=h))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())


def test_llm_call_with_a_slow_async_lm_times_out():
    import time

    import dspy

    from integrations import invoke_module

    class SlowLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="fake/slow", cache=False)
            self.cancelled = 0

        def forward(self, *a, **kw):  # pragma: no cover - async path only
            raise AssertionError

        async def aforward(self, prompt=None, messages=None, **kw):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise

    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, ["llm.error"])
    lm = SlowLM()
    t0 = time.perf_counter()
    with pytest.raises(BoundaryTimeout):
        asyncio.run(invoke_module(dspy.Predict("question -> answer"), {"question": "q"},
                                  lm=lm, harness=Harness([timeout(0.1)])))
    assert time.perf_counter() - t0 < 2
    assert lm.cancelled == 1
    assert "BoundaryTimeout" in got[-1].payload["error"]
