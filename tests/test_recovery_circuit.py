"""circuit_breaker: closed -> open -> half-open -> closed / open."""

import asyncio

import pytest

from pyreact.core import EventBus, HookContext
from pyreact.core.harness import Blocked, Harness, through
from pyreact.core.recovery import CircuitOpen, circuit_breaker, fallback, retry


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _events():
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, ["recovery.circuit"])
    return got


class Backend:
    def __init__(self):
        self.up, self.calls = False, 0

    async def __call__(self, b):
        self.calls += 1
        if not self.up:
            raise ConnectionError("down")
        return "ok"


def _call(h, fn, name="svc", **attrs):
    return asyncio.run(through("tool", name, {}, fn, harness=h, **attrs))


def test_opens_fails_fast_half_opens_and_closes():
    events = _events()
    clock, svc = Clock(), Backend()
    cb = circuit_breaker(failure_threshold=2, reset_timeout=10, clock=clock)
    h = Harness([cb])
    for _ in range(2):
        with pytest.raises(ConnectionError):
            _call(h, svc)
    assert cb.state("svc") == "open" and svc.calls == 2
    with pytest.raises(CircuitOpen) as exc:  # fail fast: backend not called
        _call(h, svc)
    assert svc.calls == 2 and exc.value.retry_after == pytest.approx(10)
    assert exc.value.retryable is False

    clock.t = 10.5  # reset timeout elapsed: one trial
    assert cb.state("svc") == "half_open"
    with pytest.raises(ConnectionError):
        _call(h, svc)  # trial fails -> open again
    assert cb.state("svc") == "open" and svc.calls == 3

    clock.t = 21
    svc.up = True
    assert _call(h, svc) == "ok"  # trial succeeds -> closed
    assert cb.state("svc") == "closed"
    states = [(e.payload["prev"], e.payload["state"]) for e in events]
    assert states == [("closed", "open"), ("open", "half_open"), ("half_open", "open"),
                      ("open", "half_open"), ("half_open", "closed")]


def test_keys_are_independent_and_success_resets_the_count():
    clock, svc = Clock(), Backend()
    cb = circuit_breaker(failure_threshold=2, reset_timeout=10, clock=clock)
    h = Harness([cb])
    with pytest.raises(ConnectionError):
        _call(h, svc, name="a")
    svc.up = True
    _call(h, svc, name="a")  # success: count back to 0
    svc.up = False
    with pytest.raises(ConnectionError):
        _call(h, svc, name="a")
    assert cb.state("a") == "closed"
    for _ in range(2):
        with pytest.raises(ConnectionError):
            _call(h, svc, name="b")
    assert cb.state("b") == "open" and cb.state("a") == "closed"
    # default key: the llm model attr wins over the name
    with pytest.raises(ConnectionError):
        _call(h, svc, name="b", model="m1")
    assert cb.state("m1") == "closed"
    cb.reset()
    assert cb.state("b") == "closed"


def test_blocked_and_cancel_do_not_count_and_fallback_when_open():
    clock = Clock()
    cb = circuit_breaker(failure_threshold=1, reset_timeout=10, clock=clock,
                         fallback=lambda b, e: f"degraded ({type(e).__name__})")
    h = Harness([cb])

    async def refused(b):
        raise Blocked("no")

    with pytest.raises(Blocked):
        _call(h, refused)
    assert cb.state("svc") == "closed"
    svc = Backend()
    with pytest.raises(ConnectionError):
        _call(h, svc)
    assert _call(h, svc) == "degraded (CircuitOpen)" and svc.calls == 1


def test_one_trial_at_a_time_in_half_open():
    clock = Clock()
    cb = circuit_breaker(failure_threshold=1, reset_timeout=1, clock=clock)
    h = Harness([cb])
    gate = {"n": 0}

    async def slow_fail(b):
        gate["n"] += 1
        if gate["n"] == 1:
            raise ConnectionError("x")
        await asyncio.sleep(0.05)
        return "ok"

    async def main():
        with pytest.raises(ConnectionError):
            await through("tool", "svc", {}, slow_fail, harness=h)
        clock.t = 2
        results = await asyncio.gather(
            through("tool", "svc", {}, slow_fail, harness=h),
            through("tool", "svc", {}, slow_fail, harness=h),
            return_exceptions=True)
        return results

    results = asyncio.run(main())
    assert results[0] == "ok" and isinstance(results[1], CircuitOpen)
    assert cb.state("svc") == "closed"


def test_retry_does_not_retry_an_open_circuit_but_fallback_substitutes():
    clock, svc = Clock(), Backend()
    cb = circuit_breaker(failure_threshold=2, reset_timeout=10, clock=clock)
    h = Harness([fallback(lambda b, e: f"fallback after {type(e).__name__}"),
                 retry(max_attempts=5, backoff=lambda n: 0), cb])
    # attempts 1-2 fail and open the circuit; attempt 3 is rejected and not retried
    assert _call(h, svc) == "fallback after CircuitOpen"
    assert svc.calls == 2
