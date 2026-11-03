# recovery.py -----------------------------------------------
"""Error recovery as harness middleware (generic: any boundary kind).

Each factory returns a ``(b, call_next)`` middleware (see
:mod:`pyreact.core.harness`) to put in ``HarnessProvider(middleware=[...])``
or ``install(...)``::

    from pyreact.core.recovery import exponential, retry

    HarnessProvider(middleware=[
        retry(max_attempts=3, backoff=exponential(0.5, 8.0), kinds=("llm", "tool")),
    ], children=[...])

Every recovery action is published as a ``recovery.*`` event on the bus of
the session doing the call (and the process-global bus), with the boundary's
``turn_id`` and component path:

    recovery.retry      {kind, name, boundary_id, attempt, delay, error, hint}
    recovery.exhausted  {kind, name, boundary_id, attempts, error}
    recovery.fallback   {kind, name, boundary_id, to, error}
    recovery.circuit    {kind, name, boundary_id, key, state, prev, failures, error}
    recovery.timeout    {kind, name, boundary_id, attempt, seconds}

DSPy-aware recovery (switching models) lives in :mod:`integrations.recovery`.
"""

from __future__ import annotations

import asyncio
import email.utils
import inspect
import random
import threading
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Type, Union

from .harness import Blocked, Boundary, Middleware, Next, middleware

__all__ = [
    "Backoff",
    "BoundaryTimeout",
    "CircuitOpen",
    "circuit_breaker",
    "RetryOn",
    "exponential",
    "fallback",
    "should_fall_back",
    "is_retryable",
    "publish_recovery",
    "retry",
    "retry_after_of",
    "timeout",
]

Backoff = Callable[[int], float]  # retry number (1, 2, ...) -> seconds
RetryOn = Union[Type[BaseException], Tuple[Type[BaseException], ...],
                Callable[[BaseException], bool]]

DEFAULT_KINDS: Tuple[str, ...] = ("llm", "tool")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def publish_recovery(kind: str, payload: dict, b: Optional[Boundary] = None) -> None:
    """Publish a ``recovery.*`` event on the calling session's bus and the
    process-global bus (once when they are the same). Never raises."""
    try:
        from .events import EventBus
        from .hook import HookContext
        from . import runtime

        glob = HookContext.get_service("event_bus", EventBus)
        sess = None
        ctx = runtime.current_component()
        services = runtime.services_of(ctx) if ctx is not None else None
        if services is not None:
            sess = services.get("event_bus")
        turn_id = b.turn_id if b is not None else None
        source = (b.component if b is not None else None) or "harness"
        if sess is not None and sess is not glob and hasattr(sess, "publish"):
            sess.publish(kind, payload, turn_id=turn_id, source=source)
        glob.publish(kind, payload, turn_id=turn_id, source=source)
    except Exception:  # observability must never break a call
        pass


def _brief(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 300 else text[:297] + "..."


def _base(b: Boundary) -> dict:
    return {"kind": b.kind, "name": b.name, "boundary_id": b.id}


def is_retryable(exc: BaseException) -> bool:
    """Default ``retry_on``: any ``Exception`` except guard refusals
    (:class:`Blocked`) and errors that say so with ``exc.retryable = False``
    (e.g. an open circuit). ``CancelledError`` is never retried."""
    if not isinstance(exc, Exception) or isinstance(exc, Blocked):
        return False
    return getattr(exc, "retryable", True) is not False


def should_fall_back(exc: BaseException) -> bool:
    """Default ``on`` of the fallbacks: any ``Exception`` except guard
    refusals (:class:`Blocked`). Unlike :func:`is_retryable` it includes
    non-retryable errors such as an open circuit: that is when a substitute
    is most useful."""
    return isinstance(exc, Exception) and not isinstance(exc, Blocked)


def _predicate(on: Optional[RetryOn],
               default: Callable[[BaseException], bool] = is_retryable,
               ) -> Callable[[BaseException], bool]:
    if on is None:
        return default
    if isinstance(on, type) or isinstance(on, tuple):
        types = on

        def by_type(exc: BaseException) -> bool:
            return isinstance(exc, types) and default(exc)

        return by_type
    if callable(on):
        return on
    raise TypeError(f"retry_on must be exception type(s) or a predicate, got {on!r}")


def _seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:  # HTTP-date
        when = email.utils.parsedate_to_datetime(text)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _header(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    try:
        value = headers.get(name)
        if value is None:
            value = headers.get(name.title())
        return value
    except Exception:
        return None


def retry_after_of(exc: BaseException) -> Optional[float]:
    """A server's "retry after" hint carried by ``exc``, in seconds.

    Looks at ``exc.retry_after`` (seconds), then the ``Retry-After`` /
    ``retry-after-ms`` headers of ``exc.headers`` or ``exc.response.headers``
    (litellm / openai / httpx errors). ``None`` when there is no hint.
    """
    direct = _seconds(getattr(exc, "retry_after", None))
    if direct is not None:
        return direct
    for holder in (exc, getattr(exc, "response", None)):
        headers = getattr(holder, "headers", None)
        ms = _seconds(_header(headers, "retry-after-ms"))
        if ms is not None:
            return ms / 1000.0
        s = _seconds(_header(headers, "retry-after"))
        if s is not None:
            return s
    return None


def exponential(base: float = 0.5, max: float = 30.0, jitter: Union[bool, float] = True,
                *, factor: float = 2.0, rng: Optional[random.Random] = None) -> Backoff:
    """Exponential backoff: ``min(max, base * factor ** (n - 1))`` seconds before
    retry ``n``. ``jitter=True`` ("full jitter") picks uniformly in
    ``[0, delay]``; a float ``j`` in ``[0, 1]`` keeps ``(1 - j)`` of the delay
    and randomizes the rest; ``False`` is deterministic."""
    cap = max
    pick = rng or random.Random()

    def delay(n: int) -> float:
        d = min(cap, base * (factor ** (n - 1)))
        if jitter is True:
            return pick.uniform(0, d)
        if jitter:
            j = float(jitter)
            return d * (1 - j) + pick.uniform(0, d * j)
        return d

    return delay


def _cancelling() -> bool:
    task = asyncio.current_task()
    return bool(task is not None and getattr(task, "cancelling", lambda: 0)())


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------
def retry(
    max_attempts: int = 3,
    backoff: Optional[Backoff] = None,
    retry_on: Optional[RetryOn] = None,
    kinds: Optional[Iterable[str]] = DEFAULT_KINDS,
    *,
    respect_retry_after: bool = True,
    max_retry_after: float = 60.0,
    name: str = "retry",
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> Middleware:
    """Re-run a failing boundary call (the module / tool runs again).

    - ``max_attempts``: total runs, including the first;
    - ``backoff``: ``fn(n) -> seconds`` before retry ``n`` (default
      :func:`exponential` ``(0.5, 30)`` with jitter);
    - ``retry_on``: exception type(s) or ``predicate(exc) -> bool`` (default
      :func:`is_retryable`: never ``Blocked``, ``CancelledError`` or errors
      with ``retryable = False``);
    - a ``Retry-After`` hint carried by the error (:func:`retry_after_of`)
      raises the delay to it (capped at ``max_retry_after``);
    - cancellation wins: an interrupted turn cancels the task, which stops
      the backoff sleep and never starts another attempt.

    Bumps ``b.attempt`` before each re-run, records ``b.attrs["retries"]``
    and publishes ``recovery.retry`` / ``recovery.exhausted``.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    wait = backoff if backoff is not None else exponential(0.5, 30.0)
    should = _predicate(retry_on)

    @middleware(kinds=kinds, name=name)
    async def _retry(b: Boundary, call_next: Next) -> Any:
        runs = 0
        while True:
            runs += 1
            try:
                return await call_next(b)
            except Exception as exc:
                if not should(exc) or _cancelling():
                    raise
                if runs >= max_attempts:
                    if max_attempts > 1:
                        publish_recovery("recovery.exhausted", {
                            **_base(b), "attempts": runs, "error": _brief(exc)}, b)
                    raise
                delay = max(0.0, float(wait(runs)))
                hint = retry_after_of(exc) if respect_retry_after else None
                if hint is not None:
                    delay = max(delay, min(hint, max_retry_after))
                b.attrs["retries"] = b.attrs.get("retries", 0) + 1
                publish_recovery("recovery.retry", {
                    **_base(b), "attempt": b.attempt, "next_attempt": b.attempt + 1,
                    "delay": round(delay, 4), "hint": hint, "error": _brief(exc)}, b)
                await sleep(delay)  # cancellation raises here: no more attempts
                b.attempt += 1

    return _retry


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------
def fallback(
    fn: Callable[[Boundary, BaseException], Any],
    on: Optional[RetryOn] = None,
    kinds: Optional[Iterable[str]] = None,
    *,
    name: str = "fallback",
) -> Middleware:
    """When the call fails, return ``fn(b, exc)`` (sync or async) instead.

    The substitute must be what the owner expects: a ``dspy.Prediction`` for
    ``llm`` boundaries, the tool's result type for ``tool``. ``on``: exception
    type(s) or ``predicate(exc)`` (default :func:`should_fall_back`: every
    ``Exception`` but :class:`Blocked`). Sets ``b.attrs["fallback"] = True``
    and publishes ``recovery.fallback``.
    """
    should = _predicate(on, should_fall_back)

    @middleware(kinds=kinds, name=name)
    async def _fallback(b: Boundary, call_next: Next) -> Any:
        try:
            return await call_next(b)
        except Exception as exc:
            if not should(exc) or _cancelling():
                raise
            b.attrs["fallback"] = True
            publish_recovery("recovery.fallback", {
                **_base(b), "to": getattr(fn, "__name__", "substitute"),
                "error": _brief(exc)}, b)
            out = fn(b, exc)
            if inspect.isawaitable(out):
                out = await out
            return out

    return _fallback


# ---------------------------------------------------------------------------
# circuit breaker
# ---------------------------------------------------------------------------
class CircuitOpen(Exception):
    """Raised instead of running a call whose circuit is open (fail fast).

    ``retryable = False``: :func:`retry` does not retry it, while
    :func:`fallback` / ``fallback_model`` do substitute it. ``retry_after``
    is the time left before the circuit lets a trial call through."""

    retryable = False

    def __init__(self, key: Any, retry_after: float = 0.0):
        super().__init__(f"circuit {key!r} is open (retry in {retry_after:.1f}s)")
        self.key = key
        self.retry_after = retry_after


def _default_key(b: Boundary) -> Any:
    return b.attrs.get("model") or b.name


def circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 30.0,
    key: Callable[[Boundary], Any] = _default_key,
    kinds: Optional[Iterable[str]] = DEFAULT_KINDS,
    *,
    failure_on: Optional[RetryOn] = None,
    fallback: Optional[Callable[[Boundary, BaseException], Any]] = None,
    clock: Callable[[], float] = time.monotonic,
    name: str = "circuit_breaker",
) -> Middleware:
    """Stop calling something that keeps failing; probe it again later.

    One circuit per ``key(b)`` (default: the ``llm`` model, else the
    boundary name — e.g. the tool):

    - **closed**: calls run; ``failure_threshold`` consecutive failures
      (``failure_on``: type(s) or predicate, default every ``Exception`` but
      ``Blocked``; cancellations never count) open it;
    - **open**: calls fail fast with :class:`CircuitOpen` — or return
      ``fallback(b, exc)`` when given — without running; after
      ``reset_timeout`` seconds the next call is a trial (**half-open**);
    - **half-open**: one trial at a time (others fail fast); success closes
      the circuit, failure re-opens it.

    Transitions publish ``recovery.circuit``; ``b.attrs["circuit"]`` is the
    state the call met. The returned middleware has ``.state(key)`` and
    ``.reset(key=None)`` for inspection and tests. Put it *inside*
    ``retry`` (each attempt counts) and ``fallback_model`` (an open circuit
    falls back immediately): ``[fallback_model, retry, circuit_breaker,
    timeout]``.
    """
    if failure_threshold < 1:
        raise ValueError("failure_threshold must be >= 1")
    counts = _predicate(failure_on, should_fall_back)
    circuits: dict = {}
    lock = threading.Lock()

    def _get(k: Any) -> dict:
        c = circuits.get(k)
        if c is None:
            c = circuits[k] = {"state": "closed", "failures": 0, "opened_at": 0.0,
                               "trial": False}
        return c

    def _move(b: Boundary, k: Any, c: dict, state: str, error: Any = None) -> None:
        prev, c["state"] = c["state"], state
        if state == "open":
            c["opened_at"] = clock()
        if state == "closed":
            c["failures"] = 0
        publish_recovery("recovery.circuit", {
            **_base(b), "key": str(k), "state": state, "prev": prev,
            "failures": c["failures"], "error": _brief(error) if error else None}, b)

    async def _reject(b: Boundary, k: Any, remaining: float) -> Any:
        exc = CircuitOpen(k, remaining)
        if fallback is None:
            raise exc
        b.attrs["fallback"] = True
        out = fallback(b, exc)
        if inspect.isawaitable(out):
            out = await out
        return out

    @middleware(kinds=kinds, name=name)
    async def _breaker(b: Boundary, call_next: Next) -> Any:
        k = key(b)
        reject: Optional[float] = None
        is_trial = False
        with lock:
            c = _get(k)
            if c["state"] == "open":
                elapsed = clock() - c["opened_at"]
                if elapsed >= reset_timeout:
                    _move(b, k, c, "half_open")
                else:
                    reject = reset_timeout - elapsed
            if reject is None and c["state"] == "half_open":
                if c["trial"]:
                    reject = 0.0  # a trial is in flight
                else:
                    c["trial"] = is_trial = True
            b.attrs["circuit"] = "open" if reject is not None else c["state"]
        if reject is not None:
            return await _reject(b, k, reject)
        try:
            out = await call_next(b)
        except BaseException as exc:
            with lock:
                if is_trial:
                    c["trial"] = False
                if isinstance(exc, Exception) and counts(exc):
                    c["failures"] += 1
                    if c["state"] == "half_open" or (
                            c["state"] == "closed" and c["failures"] >= failure_threshold):
                        _move(b, k, c, "open", exc)
            raise
        with lock:
            if is_trial:
                c["trial"] = False
            if c["state"] == "half_open":
                _move(b, k, c, "closed")
            else:
                c["failures"] = 0
        return out

    def state(k: Any) -> str:
        with lock:
            c = circuits.get(k)
            if c is None:
                return "closed"
            if c["state"] == "open" and clock() - c["opened_at"] >= reset_timeout:
                return "half_open"
            return c["state"]

    def reset(k: Any = None) -> None:
        with lock:
            if k is None:
                circuits.clear()
            else:
                circuits.pop(k, None)

    _breaker.state = state  # type: ignore[attr-defined]
    _breaker.reset = reset  # type: ignore[attr-defined]
    return _breaker


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------
class BoundaryTimeout(TimeoutError):
    """A boundary call took longer than its :func:`timeout` (retryable)."""

    def __init__(self, b: Boundary, seconds: float):
        super().__init__(f"{b.kind} {b.name!r} timed out after {seconds:g}s")
        self.boundary = b
        self.seconds = seconds


def timeout(
    seconds: Union[float, Callable[[Boundary], Optional[float]]],
    kinds: Optional[Iterable[str]] = DEFAULT_KINDS,
    *,
    name: str = "timeout",
) -> Middleware:
    """Bound each run of the call to ``seconds`` (a number, or
    ``fn(b) -> seconds | None``; ``None`` = no limit for that call).

    On expiry the inner call is cancelled (``CancelledError`` inside it, so
    ``finally`` blocks and async LM/tool clients clean up) and
    :class:`BoundaryTimeout` (a ``TimeoutError``) is raised; a
    ``TimeoutError`` raised by the call itself passes through unchanged.
    Place it innermost to bound each attempt of an outer :func:`retry`.
    Work running in a worker thread (sync tools, sync LMs via
    ``asyncio.to_thread``) cannot be interrupted: the boundary returns on
    time, the thread finishes in the background.

    Sets ``b.attrs["timed_out"] = True`` and publishes ``recovery.timeout``.
    """

    @middleware(kinds=kinds, name=name)
    async def _timeout(b: Boundary, call_next: Next) -> Any:
        limit = seconds(b) if callable(seconds) else seconds
        if limit is None:
            return await call_next(b)
        cm = asyncio.timeout(float(limit))
        try:
            async with cm:
                return await call_next(b)
        except TimeoutError:
            if not cm.expired():
                raise
            b.attrs["timed_out"] = True
            publish_recovery("recovery.timeout", {
                **_base(b), "attempt": b.attempt, "seconds": float(limit)}, b)
            raise BoundaryTimeout(b, float(limit)) from None

    return _timeout
