"""Retry middleware (pyreact.core.recovery.retry)."""

import asyncio
from types import SimpleNamespace

import dspy
import pytest

from integrations import invoke_module
from pyreact.core import EventBus, HookContext
from pyreact.core.harness import Blocked, Harness, through
from pyreact.core.recovery import exponential, retry, retry_after_of


def _events():
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, ["recovery.*"])
    return got


class Sleeps:
    def __init__(self):
        self.delays = []

    async def __call__(self, d):
        self.delays.append(d)


def _flaky(fails, exc_factory=lambda n: ConnectionError(f"boom {n}")):
    seen = []

    async def fn(b):
        seen.append(b.attempt)
        if len(seen) <= fails:
            raise exc_factory(len(seen))
        return f"ok after {len(seen)}"

    return fn, seen


def test_retry_bumps_attempt_backs_off_and_publishes():
    events = _events()
    sleeps = Sleeps()
    h = Harness([retry(max_attempts=3, backoff=exponential(0.1, 1.0, jitter=False),
                       sleep=sleeps)])
    fn, seen = _flaky(2)
    out = asyncio.run(through("tool", "weather", {"city": "x"}, fn, harness=h))
    assert out == "ok after 3"
    assert seen == [1, 2, 3]
    assert sleeps.delays == [0.1, 0.2]
    retries = [e for e in events if e.kind == "recovery.retry"]
    assert [e.payload["attempt"] for e in retries] == [1, 2]
    assert retries[0].payload["kind"] == "tool" and retries[0].payload["name"] == "weather"
    assert "ConnectionError" in retries[0].payload["error"]


def test_retry_gives_up_after_max_attempts():
    events = _events()
    h = Harness([retry(max_attempts=2, backoff=lambda n: 0, sleep=Sleeps())])
    fn, seen = _flaky(5)
    with pytest.raises(ConnectionError):
        asyncio.run(through("llm", "m", {}, fn, harness=h))
    assert seen == [1, 2]
    assert [e.kind for e in events] == ["recovery.retry", "recovery.exhausted"]
    assert events[-1].payload["attempts"] == 2


def test_retry_on_types_and_predicate_and_never_blocked():
    h = Harness([retry(max_attempts=3, retry_on=(TimeoutError,), sleep=Sleeps())])
    fn, seen = _flaky(1)  # ConnectionError: not in retry_on
    with pytest.raises(ConnectionError):
        asyncio.run(through("tool", "t", {}, fn, harness=h))
    assert seen == [1]

    h = Harness([retry(max_attempts=3, retry_on=lambda e: "transient" in str(e),
                       sleep=Sleeps())])
    fn, seen = _flaky(1, lambda n: RuntimeError("transient glitch"))
    assert asyncio.run(through("tool", "t", {}, fn, harness=h)) == "ok after 2"

    h = Harness([retry(max_attempts=3, sleep=Sleeps())])
    fn, seen = _flaky(1, lambda n: Blocked("no"))
    with pytest.raises(Blocked):
        asyncio.run(through("tool", "t", {}, fn, harness=h))
    assert seen == [1]

    class Fatal(Exception):
        retryable = False

    fn, seen = _flaky(1, lambda n: Fatal("x"))
    with pytest.raises(Fatal):
        asyncio.run(through("tool", "t", {}, fn, harness=h))
    assert seen == [1]


def test_kinds_filter():
    h = Harness([retry(max_attempts=3, kinds=["llm"], sleep=Sleeps())])
    fn, seen = _flaky(1)
    with pytest.raises(ConnectionError):
        asyncio.run(through("tool", "t", {}, fn, harness=h))
    assert seen == [1]


def test_retry_after_hint_raises_the_delay():
    class RateLimited(Exception):
        def __init__(self):
            super().__init__("429")
            self.response = SimpleNamespace(headers={"retry-after": "2"})

    assert retry_after_of(RateLimited()) == 2.0
    exc = Exception("x")
    exc.retry_after = 0.75
    assert retry_after_of(exc) == 0.75
    exc2 = Exception("y")
    exc2.headers = {"retry-after-ms": "250"}
    assert retry_after_of(exc2) == 0.25
    assert retry_after_of(Exception("none")) is None

    sleeps = Sleeps()
    h = Harness([retry(max_attempts=2, backoff=lambda n: 0.01, sleep=sleeps,
                       max_retry_after=1.5)])
    fn, _ = _flaky(1, lambda n: RateLimited())
    asyncio.run(through("llm", "m", {}, fn, harness=h))
    assert sleeps.delays == [1.5]  # hint (2s) capped by max_retry_after


def test_cancellation_stops_retrying():
    calls = []

    async def always_fails(b):
        calls.append(b.attempt)
        raise ConnectionError("down")

    h = Harness([retry(max_attempts=10, backoff=lambda n: 0.2)])

    async def main():
        task = asyncio.create_task(through("tool", "t", {}, always_fails, harness=h))
        await asyncio.sleep(0.05)  # first attempt failed, now in backoff
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.3)  # no further attempt after the cancel

    asyncio.run(main())
    assert calls == [1]


def test_exponential_jitter_bounds():
    det = exponential(1.0, 5.0, jitter=False)
    assert [det(n) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 5.0]
    full = exponential(1.0, 5.0, jitter=True)
    assert all(0 <= full(3) <= 4.0 for _ in range(50))
    half = exponential(1.0, 5.0, jitter=0.5)
    assert all(2.0 <= half(3) <= 4.0 for _ in range(50))


class FlakyLM(dspy.BaseLM):
    def __init__(self, fails):
        super().__init__(model="fake/flaky", cache=False)
        self.fails, self.calls = fails, 0

    def forward(self, prompt=None, messages=None, **kw):
        self.calls += 1
        if self.calls <= self.fails:
            raise ConnectionError("overloaded")
        msg = SimpleNamespace(content="[[ ## answer ## ]]\nhi\n\n[[ ## completed ## ]]",
                              tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage={}, model=self.model)

    async def aforward(self, prompt=None, messages=None, **kw):
        return self.forward(prompt, messages, **kw)


def test_llm_boundary_reruns_the_module():
    lm = FlakyLM(fails=2)  # attempt 1: the ChatAdapter call and its JSONAdapter fallback
    attempts = []

    async def spy(b, call_next):
        try:
            return await call_next(b)
        finally:
            attempts.append(b.attempt)

    h = Harness([spy, retry(max_attempts=3, backoff=lambda n: 0)])
    mod = dspy.Predict("question -> answer")
    result, usage = asyncio.run(invoke_module(mod, {"question": "q"}, lm=lm, harness=h))
    assert result.answer == "hi"
    assert lm.calls == 3
    assert attempts == [2]


def test_interrupted_turn_stops_retrying():
    """turn_policy="interrupt": a new input cancels the turn's tasks, which
    ends the retry loop in its backoff sleep."""
    from pyreact.boot import bootstrap
    from pyreact.core import component, hooks, use_turn
    from pyreact.core.harness import HarnessProvider

    calls = []

    async def down(b):
        calls.append((b.inputs["q"], b.attempt))
        raise ConnectionError("down")

    @component
    def Worker():
        turn = use_turn()

        async def work():
            await through("tool", "fetch", {"q": turn.text}, down)

        hooks.use_effect(lambda: work() if turn and turn.text == "slow" else None,
                         [turn and turn.id])
        return []

    @component
    def App():
        return [HarnessProvider(key="h", middleware=[
            retry(max_attempts=50, backoff=lambda n: 0.3)], children=[Worker(key="w")])]

    app = bootstrap(App, turn_policy="interrupt")
    try:
        t1 = app.invoke("slow")
        import time

        time.sleep(0.15)
        t2 = app.invoke("next", wait=True, timeout=5)
        assert app.wait_turn(t1, timeout=5)["status"] == "interrupted"
        assert app.wait_turn(t2, timeout=5)["status"] == "done"
        time.sleep(0.4)
    finally:
        app.shutdown()
    assert calls == [("slow", 1)]
