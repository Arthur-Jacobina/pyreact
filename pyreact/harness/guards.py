"""Guards: decide *before* a boundary call runs whether it may run.

A guard is harness middleware (``(b, call_next)``) that either lets the call
through, rewrites its inputs (redaction), or refuses it by raising
:class:`~pyreact.core.harness.Blocked` -- so it plugs into
``HarnessProvider(middleware=[...])``, ``install(...)`` or the component
hooks ``use_guard`` alike. Every refusal publishes a ``guard.blocked`` event::

    {"guard", "reason", "kind", "name", "boundary_id", "call_id",
     "component", "attempt"}

Write your own with :func:`guard` (a predicate over the boundary), or use a
built-in: :func:`tool_allowlist`, :func:`max_calls`, :func:`budget`,
:func:`rate_limit`, :func:`input_policy`, :func:`require_confirmation`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import re
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Callable, Deque, Dict, Hashable, Iterable, Mapping, Optional, Tuple, Union

from pyreact.core.harness import Blocked, Boundary, Next

from ._events import emit

__all__ = [
    "GuardMiddleware",
    "PII",
    "block",
    "budget",
    "guard",
    "input_policy",
    "max_calls",
    "rate_limit",
    "require_confirmation",
    "tool_allowlist",
]

Kinds = Union[str, Iterable[str], None]


def _kinds(kinds: Kinds) -> Optional[Tuple[str, ...]]:
    if kinds is None:
        return None
    return (kinds,) if isinstance(kinds, str) else tuple(kinds)


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def block(b: Boundary, by: str, reason: str) -> None:
    """Refuse ``b``: publish ``guard.blocked`` and raise :class:`Blocked`."""
    emit("guard.blocked", b, guard=by, reason=reason)
    raise Blocked(reason, boundary=b, by=by)


class GuardMiddleware:
    """Base of the built-in guards: a middleware object with ``kinds`` (the
    harness kind filter) and ``harness_name`` (shown in ``Blocked`` messages
    and events). Subclasses implement :meth:`check` (and may override
    ``__call__`` to look at the result)."""

    __pyreact_harness__ = True  # recognized by use_guard / use_verify
    default_kinds: Optional[Tuple[str, ...]] = None

    def __init__(self, *, kinds: Kinds = None, name: Optional[str] = None) -> None:
        self.kinds = _kinds(kinds) if kinds is not None else self.default_kinds
        self.harness_name = name or type(self).__name__

    @property
    def __name__(self) -> str:  # so _name(mw) / logs read naturally
        return self.harness_name

    async def check(self, b: Boundary) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    async def __call__(self, b: Boundary, call_next: Next) -> Any:
        await self.check(b)
        return await call_next(b)

    def block(self, b: Boundary, reason: str) -> None:
        block(b, self.harness_name, reason)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.harness_name!r} kinds={self.kinds}>"


# ---------------------------------------------------------------------------
# guard(fn): your own
# ---------------------------------------------------------------------------
def _reason(result: Any, name: str) -> Optional[str]:
    if result is None or result is True:
        return None
    if result is False:
        return f"refused by {name}"
    if isinstance(result, str):
        return result or f"refused by {name}"
    raise TypeError(
        f"guard {name!r} returned {type(result).__name__}; a guard returns None/True (allow), "
        "False or a reason string (block)"
    )


class _FnGuard(GuardMiddleware):
    def __init__(self, fn: Callable[[Boundary], Any], kinds: Kinds, name: str) -> None:
        super().__init__(kinds=kinds, name=name)
        self.fn = fn

    async def check(self, b: Boundary) -> None:
        reason = _reason(await _resolve(self.fn(b)), self.harness_name)
        if reason is not None:
            self.block(b, reason)


def guard(fn: Optional[Callable[[Boundary], Any]] = None, *, kinds: Kinds = None,
          name: Optional[str] = None):
    """Turn a predicate over the boundary into a guard middleware.

    ``fn(b)`` (sync or async) returns ``None`` / ``True`` to allow the call,
    ``False`` or a reason string to block it (``Blocked`` + ``guard.blocked``).
    ``kinds`` restricts it (``"tool"``, ``["llm", "tool.*"]``); ``name``
    defaults to the function name. Usable as a decorator::

        @guard(kinds="tool")
        def small_orders(b):
            if b.name == "place_order" and b.inputs.get("quantity", 0) > 10:
                return "orders above 10 units need a human"
    """

    def make(f: Callable[[Boundary], Any]) -> GuardMiddleware:
        return _FnGuard(f, kinds, name or getattr(f, "__name__", "guard"))

    return make(fn) if fn is not None else make


# ---------------------------------------------------------------------------
# Per-turn bookkeeping (bounded: old turns are forgotten)
# ---------------------------------------------------------------------------
class _PerTurn:
    def __init__(self, factory: Callable[[], Any], max_turns: int = 1024) -> None:
        self.factory, self.max_turns = factory, max_turns
        self._data: "OrderedDict[Hashable, Any]" = OrderedDict()
        self.lock = threading.Lock()

    def get(self, turn_id: Hashable) -> Any:
        """The value for ``turn_id`` (call with ``lock`` held)."""
        if turn_id not in self._data:
            self._data[turn_id] = self.factory()
            while len(self._data) > self.max_turns:
                self._data.popitem(last=False)
        else:
            self._data.move_to_end(turn_id)
        return self._data[turn_id]

    def peek(self, turn_id: Hashable) -> Any:
        with self.lock:
            return self._data.get(turn_id)


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------
class tool_allowlist(GuardMiddleware):  # noqa: N801 - reads as a function
    """Only let the agent call some tools.

    ``allow``: tool names / glob patterns (``"search_*"``); ``None`` allows
    every tool not denied. ``deny`` wins over ``allow``. The reason lists the
    allowed tools, so a ReAct agent (which receives ``"Error: ..."`` as the
    tool result) can pick another one.
    """

    default_kinds = ("tool",)

    def __init__(self, allow: Optional[Iterable[str]] = None, deny: Iterable[str] = (), *,
                 kinds: Kinds = None, name: str = "tool_allowlist") -> None:
        super().__init__(kinds=kinds, name=name)
        self.allow = None if allow is None else tuple(allow)
        self.deny = tuple(deny)

    def allows(self, tool: str) -> bool:
        if any(fnmatch.fnmatchcase(tool, p) for p in self.deny):
            return False
        return self.allow is None or any(fnmatch.fnmatchcase(tool, p) for p in self.allow)

    async def check(self, b: Boundary) -> None:
        if self.allows(b.name):
            return
        if any(fnmatch.fnmatchcase(b.name, p) for p in self.deny):
            self.block(b, f"tool {b.name!r} is denied")
        self.block(b, f"tool {b.name!r} is not allowed; allowed: {', '.join(self.allow or ())}")


class max_calls(GuardMiddleware):  # noqa: N801
    """At most ``per_turn`` calls of ``kind`` (a kind or list of kinds,
    default ``"llm"``) per turn; further ones are blocked. Calls outside any
    turn share one bucket. Retries of one call (``b.attempt`` > 1) are not
    counted again."""

    def __init__(self, per_turn: int, *, kind: Kinds = "llm", name: str = "max_calls") -> None:
        super().__init__(kinds=kind, name=name)
        if per_turn < 0:
            raise ValueError("per_turn must be >= 0")
        self.per_turn = per_turn
        self._counts = _PerTurn(lambda: {"n": 0, "ids": set()})

    def count(self, turn_id: Optional[str]) -> int:
        c = self._counts.peek(turn_id)
        return 0 if c is None else c["n"]

    async def check(self, b: Boundary) -> None:
        with self._counts.lock:
            c = self._counts.get(b.turn_id)
            if b.id in c["ids"]:  # a retry of a call already admitted
                return
            full = c["n"] >= self.per_turn
            if not full:
                c["n"] += 1
                c["ids"].add(b.id)
        if full:
            self.block(b, f"limit of {self.per_turn} {'/'.join(self.kinds or ('*',))} "
                          "call(s) per turn reached")


class budget(GuardMiddleware):  # noqa: N801
    """Per-turn token / cost budget.

    After each call it adds the call's ``usage.total_tokens`` and ``cost``
    attrs (what the ``llm`` boundary reports; counted once even across
    retries) to the turn's total; once the total reaches a limit, further
    calls *in that turn* are blocked. A call already running is never cut
    short (a ReAct loop is one ``llm`` call). ``kinds`` defaults to ``llm``;
    add ``"tool"`` to also stop tool calls once over budget.
    :meth:`spent` reads a turn's totals.
    """

    default_kinds = ("llm",)

    def __init__(self, max_tokens_per_turn: Optional[int] = None,
                 max_cost_per_turn: Optional[float] = None, *,
                 kinds: Kinds = None, name: str = "budget") -> None:
        super().__init__(kinds=kinds, name=name)
        if max_tokens_per_turn is None and max_cost_per_turn is None:
            raise ValueError("budget needs max_tokens_per_turn and/or max_cost_per_turn")
        self.max_tokens = max_tokens_per_turn
        self.max_cost = max_cost_per_turn
        self._spent = _PerTurn(lambda: {"tokens": 0, "cost": 0.0, "seen": {}})

    def spent(self, turn_id: Optional[str]) -> Dict[str, float]:
        """``{"tokens": int, "cost": float}`` used so far in ``turn_id``."""
        s = self._spent.peek(turn_id)
        return {"tokens": 0, "cost": 0.0} if s is None else {"tokens": s["tokens"],
                                                            "cost": s["cost"]}

    def _over(self, s: Dict[str, Any]) -> Optional[str]:
        if self.max_tokens is not None and s["tokens"] >= self.max_tokens:
            return (f"turn token budget exhausted: {s['tokens']} of {self.max_tokens} "
                    "tokens used")
        if self.max_cost is not None and s["cost"] >= self.max_cost:
            return f"turn cost budget exhausted: ${s['cost']:.4f} of ${self.max_cost:.4f} used"
        return None

    def _account(self, b: Boundary) -> None:
        usage = b.attrs.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)
        cost = float(b.attrs.get("cost") or 0.0)
        with self._spent.lock:
            s = self._spent.get(b.turn_id)
            # attrs are cumulative per boundary: add only the delta since we
            # last looked (a retry inside this guard re-reports the total).
            t0, c0 = s["seen"].get(b.id, (0, 0.0))
            s["tokens"] += max(0, tokens - t0)
            s["cost"] += max(0.0, cost - c0)
            s["seen"][b.id] = (tokens, cost)
            if len(s["seen"]) > 4096:
                s["seen"].clear()

    async def __call__(self, b: Boundary, call_next: Next) -> Any:
        with self._spent.lock:
            reason = self._over(self._spent.get(b.turn_id))
        if reason is not None:
            self.block(b, reason)
        try:
            return await call_next(b)
        finally:
            self._account(b)


class rate_limit(GuardMiddleware):  # noqa: N801
    """At most ``calls`` per ``per_seconds`` (sliding window).

    Over the limit the call is blocked, or with ``wait=True`` delayed until
    a slot frees up. ``key(b)`` splits the window (e.g. per session:
    ``key=lambda b: b.session_id``); default: one window for this guard.
    """

    default_kinds = ("llm",)

    def __init__(self, calls: int, per_seconds: float, *, kinds: Kinds = None,
                 key: Optional[Callable[[Boundary], Hashable]] = None, wait: bool = False,
                 name: str = "rate_limit") -> None:
        super().__init__(kinds=kinds, name=name)
        if calls < 1 or per_seconds <= 0:
            raise ValueError("rate_limit needs calls >= 1 and per_seconds > 0")
        self.calls, self.per, self.wait, self.key = calls, float(per_seconds), wait, key
        self._windows: Dict[Hashable, Deque[float]] = {}
        self._lock = threading.Lock()
        self.clock = time.monotonic

    def _admit(self, k: Hashable) -> float:
        """0 when admitted (and recorded), else seconds until a slot frees."""
        now = self.clock()
        with self._lock:
            w = self._windows.setdefault(k, deque())
            while w and now - w[0] >= self.per:
                w.popleft()
            if len(w) < self.calls:
                w.append(now)
                return 0.0
            return self.per - (now - w[0])

    async def check(self, b: Boundary) -> None:
        k = self.key(b) if self.key is not None else None
        while True:
            delay = self._admit(k)
            if delay <= 0:
                return
            if not self.wait:
                self.block(b, f"rate limit of {self.calls} call(s) per {self.per:g}s reached "
                              f"(retry in {delay:.1f}s)")
            await asyncio.sleep(delay)


PII: Dict[str, "re.Pattern[str]"] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<![\w+])(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),
}
"""Common PII patterns for :func:`input_policy` (``input_policy(PII, redact=True)``).
Deliberately simple; swap in a real detector for production."""

Policy = Union[str, "re.Pattern[str]", Mapping[str, Any], Callable[[str], Any]]


class input_policy(GuardMiddleware):  # noqa: N801
    """Check (or redact) the string inputs of a call.

    ``policy``:

    - a regex (str or compiled): a match blocks the call, or with
      ``redact=True`` is replaced by ``replacement`` (default ``[REDACTED]``);
    - a dict ``{label: regex}`` (e.g. :data:`PII`): same, the reason names
      the label (``input 'q' contains email``) and redaction writes
      ``[EMAIL]`` unless ``replacement`` is given;
    - a callable: ``fn(text) -> None/False`` (ok) / ``True`` / reason str
      (block); with ``redact=True``, ``fn(text) -> new text``.

    Strings nested in lists/dicts are checked too. ``fields`` restricts the
    inputs looked at. Redaction replaces ``b.inputs`` (the module or tool
    gets the redacted text) and publishes ``guard.redacted``
    ``{guard, fields, count}``. Note: ``llm.start`` is published before the
    harness runs, so it still previews the original inputs.
    """

    default_kinds = ("llm", "tool")

    def __init__(self, policy: Policy, *, redact: bool = False, kinds: Kinds = None,
                 fields: Optional[Iterable[str]] = None, replacement: Optional[str] = None,
                 name: str = "input_policy") -> None:
        super().__init__(kinds=kinds, name=name)
        self.redact, self.replacement = redact, replacement
        self.fields = None if fields is None else set(fields)
        self.fn: Optional[Callable[[str], Any]] = None
        self.patterns: Dict[str, "re.Pattern[str]"] = {}
        if callable(policy) and not isinstance(policy, re.Pattern):
            self.fn = policy
        elif isinstance(policy, Mapping):
            self.patterns = {k: re.compile(v) if isinstance(v, str) else v
                             for k, v in policy.items()}
        else:
            self.patterns = {"": re.compile(policy) if isinstance(policy, str) else policy}

    # -- one string ---------------------------------------------------------
    def _violation(self, text: str) -> Optional[str]:
        if self.fn is not None:
            out = self.fn(text)
            if out is None or out is False:
                return None
            return out if isinstance(out, str) and out else "matches the policy"
        for label, rx in self.patterns.items():
            if rx.search(text):
                return f"contains {label}" if label else "matches the policy"
        return None

    def _redacted(self, text: str) -> Tuple[str, int]:
        if self.fn is not None:
            new = self.fn(text)
            new = text if new is None else str(new)
            return new, int(new != text)
        n = 0
        for label, rx in self.patterns.items():
            repl = self.replacement or (f"[{label.upper()}]" if label else "[REDACTED]")
            text, k = rx.subn(repl, text)
            n += k
        return text, n

    def _walk(self, value: Any, fn: Callable[[str], Any]) -> Any:
        if isinstance(value, str):
            return fn(value)
        if isinstance(value, dict):
            return {k: self._walk(v, fn) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._walk(v, fn) for v in value)
        return value

    async def check(self, b: Boundary) -> None:
        names = [k for k in b.inputs if self.fields is None or k in self.fields]
        if not self.redact:
            for k in names:
                found = []
                self._walk(b.inputs[k], lambda s: found.append(self._violation(s)))
                reason = next((r for r in found if r), None)
                if reason:
                    self.block(b, f"input {k!r} {reason}")
            return
        new, hits = dict(b.inputs), {}
        for k in names:
            count = [0]

            def red(s: str, count=count) -> str:
                out, n = self._redacted(s)
                count[0] += n
                return out

            new[k] = self._walk(b.inputs[k], red)
            if count[0]:
                hits[k] = count[0]
        if hits:
            b.inputs = new
            emit("guard.redacted", b, guard=self.harness_name, fields=sorted(hits),
                 count=sum(hits.values()))


class require_confirmation(GuardMiddleware):  # noqa: N801
    """Human-in-the-loop: ask before the call runs.

    ``ask(b)`` (sync -- run in a worker thread -- or async) returns ``True``
    to proceed, ``False`` / ``None`` or a reason string to refuse. ``when(b)``
    limits which calls need confirmation (default: all of ``kinds``, default
    ``"tool"``). ``timeout`` seconds without an answer refuses the call.
    Approvals publish ``guard.confirmed``; refusals ``guard.blocked``.

    In a UI, ``ask`` typically creates a future, shows the request (an event
    or a message) and awaits the user's answer.
    """

    default_kinds = ("tool",)

    def __init__(self, kinds: Kinds = None, ask: Callable[[Boundary], Any] = None, *,
                 when: Optional[Callable[[Boundary], bool]] = None,
                 timeout: Optional[float] = None, name: str = "require_confirmation") -> None:
        super().__init__(kinds=kinds, name=name)
        if ask is None:
            raise TypeError("require_confirmation needs ask=callable(boundary) -> bool | reason")
        self.ask, self.when, self.timeout = ask, when, timeout

    async def check(self, b: Boundary) -> None:
        if self.when is not None and not self.when(b):
            return
        if inspect.iscoroutinefunction(self.ask):
            pending = self.ask(b)
        else:
            pending = asyncio.to_thread(self.ask, b)

        async def answer():
            return await _resolve(await pending)

        try:
            out = await asyncio.wait_for(answer(), self.timeout)
        except asyncio.TimeoutError:
            self.block(b, f"no confirmation within {self.timeout:g}s")
        if out is True:
            emit("guard.confirmed", b, guard=self.harness_name)
            return
        self.block(b, out if isinstance(out, str) and out else "not confirmed")
