# harness.py ------------------------------------------------
"""The harness: one composable pipeline around every boundary call.

A *boundary* is a call that leaves the render tree: a turn, an LLM call, a
tool call, a sandbox exec, ... Harness *middleware* wraps those calls — like
ASGI middleware or HTTP interceptors — to add guards, verification, error
recovery and telemetry without touching component code::

    async def timing(b: Boundary, call_next):
        t0 = time.perf_counter()
        try:
            return await call_next(b)
        finally:
            b.attrs["latency"] = time.perf_counter() - t0

    HarnessProvider(middleware=[timing], children=[App()])   # scoped to a subtree
    install(timing)                                          # process-wide (e.g. telemetry libs)

Order: process-wide middleware is outermost, then providers from the root
down, then the innermost provider's list in order. A middleware may
short-circuit (return without calling ``call_next``), call it again (retry,
with ``b.attempt`` bumped), transform ``b.inputs`` / the result, or raise
(:class:`Blocked` for guards).

Code that *owns* a boundary runs it through :func:`through`::

    result = await through("tool", tool.name, args, lambda b: run_tool(**b.inputs))

Conventional kinds: ``turn``, ``llm``, ``tool``, ``sandbox.exec``; any
dotted name works. Only async boundaries are supported.
"""

from __future__ import annotations

import itertools
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

from .core import component, hooks
from .provider import create_context

__all__ = [
    "Blocked",
    "Boundary",
    "Harness",
    "HarnessProvider",
    "Middleware",
    "Next",
    "current_boundary",
    "install",
    "middleware",
    "resolve_harness",
    "through",
    "uninstall",
    "use_harness",
]

_ids = itertools.count(1)

# The boundary whose call is running (set for the whole middleware chain).
# A ContextVar: it follows tasks spawned inside the call and the worker
# threads of ``asyncio.to_thread``, so nested boundaries find their parent.
_current: ContextVar[Optional["Boundary"]] = ContextVar("pyreact_boundary", default=None)
# turn id -> its running "turn" boundary: the parent of boundaries started by
# that turn's work outside the turn's own context (effects -> spawned tasks).
_turns: Dict[str, "Boundary"] = {}


def current_boundary() -> Optional["Boundary"]:
    """The boundary being run in this context (``None`` outside any).

    Inside a middleware or the owner's call this is the boundary itself;
    inside a tool run by an LLM call it is the tool's; in a worker thread
    started with ``asyncio.to_thread`` from a boundary, that boundary.
    """
    return _current.get()


class Blocked(Exception):
    """Raised by a guard to refuse a boundary call (the call never runs).

    The harness fills ``boundary`` when the guard didn't, so the message
    always says what was blocked: ``"llm 'QA' blocked by toxicity: <reason>"``.
    """

    def __init__(self, reason: str, *, boundary: "Boundary" = None, by: str = None):
        super().__init__(reason)
        self.reason = reason
        self.boundary = boundary
        self.by = by

    def __str__(self) -> str:
        b = self.boundary
        what = f"{b.kind} {b.name!r} blocked" if b is not None else "blocked"
        return f"{what}{f' by {self.by}' if self.by else ''}: {self.reason}"


@dataclass
class Boundary:
    """One boundary call flowing through the harness.

    ``attrs`` is a free-form, mutable annotation dict shared by every
    middleware and the owner of the call (model, usage, cost, verdicts...).
    """

    kind: str
    name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    component: Optional[str] = None  # component path, when called from one
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    id: str = field(default_factory=lambda: f"b{next(_ids)}")
    parent_id: Optional[str] = None  # enclosing boundary (turn -> llm -> tool)


Next = Callable[[Boundary], Awaitable[Any]]
Middleware = Callable[[Boundary, Next], Awaitable[Any]]


def middleware(*, kinds: Optional[Iterable[str]] = None, name: Optional[str] = None):
    """Decorator: restrict a middleware to some kinds (exact or ``"prefix.*"``)."""

    def deco(fn: Middleware) -> Middleware:
        fn.kinds = tuple(kinds) if kinds is not None else None  # type: ignore[attr-defined]
        if name:
            fn.harness_name = name  # type: ignore[attr-defined]
        return fn

    return deco


def _applies(mw: Any, kind: str) -> bool:
    kinds = getattr(mw, "kinds", None)
    if not kinds:
        return True
    return any(k == kind or k == "*" or (k.endswith(".*") and kind.startswith(k[:-1]))
               for k in kinds)


class Harness:
    """An ordered middleware list (outermost first). Immutable composition:
    :meth:`extend` returns a new Harness."""

    def __init__(self, middleware: Sequence[Middleware] = ()) -> None:
        self.middleware: List[Middleware] = list(middleware)

    def extend(self, more: Sequence[Middleware]) -> "Harness":
        return Harness([*self.middleware, *more])

    async def run(self, b: Boundary, fn: Next) -> Any:
        token = _current.set(b)
        is_turn = b.kind == "turn" and b.turn_id is not None
        if is_turn:
            _turns[b.turn_id] = b
        try:
            return await self._run(b, fn)
        finally:
            if is_turn and _turns.get(b.turn_id) is b:
                del _turns[b.turn_id]
            _current.reset(token)

    async def _run(self, b: Boundary, fn: Next) -> Any:
        chain = [mw for mw in self.middleware if _applies(mw, b.kind)]
        if not chain:  # fast path: an empty pipeline is a plain call
            return await fn(b)

        async def call(i: int, bnd: Boundary) -> Any:
            if i == len(chain):
                try:
                    return await fn(bnd)
                except Blocked as exc:  # raised by the owner itself
                    exc._harness_seen = True
                    raise

            async def call_next(nb: Boundary) -> Any:
                return await call(i + 1, nb)

            try:
                return await chain[i](bnd, call_next)
            except Blocked as exc:
                if not getattr(exc, "_harness_seen", False):
                    exc._harness_seen = True  # first frame = the guard that raised
                    if exc.by is None:
                        exc.by = _name(chain[i])
                raise

        try:
            return await call(0, b)
        except Blocked as exc:
            if exc.boundary is None:
                exc.boundary = b
            raise

    def __repr__(self) -> str:
        return f"<Harness {[_name(m) for m in self.middleware]}>"


def _name(mw: Any) -> str:
    return getattr(mw, "harness_name", None) or getattr(mw, "__name__", type(mw).__name__)


# ---------------------------------------------------------------------------
# Process-wide middleware (outermost)
# ---------------------------------------------------------------------------
_global: List[Middleware] = []
_global_lock = threading.Lock()


def install(mw: Middleware) -> Callable[[], None]:
    """Add process-wide middleware (outermost). Returns an uninstall function."""
    with _global_lock:
        _global.append(mw)
    return lambda: uninstall(mw)


def uninstall(mw: Middleware) -> None:
    with _global_lock:
        try:
            _global.remove(mw)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Scoped middleware (context)
# ---------------------------------------------------------------------------
HarnessContext = create_context(default=None, name="Harness")


@component(transparent=True)
def HarnessProvider(*, middleware: Sequence[Middleware] = (), children=None):
    """Wrap a subtree with middleware. Nested providers compose: the outer
    provider's middleware wraps the inner one's."""
    parent = hooks.use_context(HarnessContext)
    base = parent if parent is not None else Harness()
    harness = hooks.use_memo(lambda: base.extend(middleware), [id(base), tuple(middleware)])
    return [HarnessContext(key="harness", value=harness, children=children or [])]


def use_harness() -> Harness:
    """The harness in effect for this component (process-wide + providers).
    Call during render; the returned object is safe to use from tasks."""
    return resolve_harness(hooks.use_context(HarnessContext))


def resolve_harness(scoped: Optional[Harness] = None) -> Harness:
    with _global_lock:
        outer = list(_global)
    return Harness([*outer, *(scoped.middleware if scoped is not None else [])])


def _current_component() -> Any:
    from .core import _context_stack
    from . import runtime

    return _context_stack.get() or runtime.current_owner.get()


def _scoped_from(ctx: Any) -> Optional[Harness]:
    if ctx is None:
        return None
    try:
        return ctx._lookup_context(HarnessContext)
    except Exception:
        return None


async def through(
    kind: str,
    name: str,
    inputs: Optional[Dict[str, Any]],
    fn: Next,
    *,
    ctx: Any = None,
    harness: Optional[Harness] = None,
    parent_id: Optional[str] = None,
    **attrs: Any,
) -> Any:
    """Run ``fn(boundary)`` through the harness and return its result.

    ``ctx`` (a HookContext) defaults to the component being rendered or the
    owner of the current task; it scopes provider middleware and fills
    ``component`` / ``session_id`` / ``turn_id``. Pass ``harness`` to use an
    already-resolved one (e.g. captured with :func:`use_harness`).

    ``parent_id`` defaults to the boundary running in this context
    (:func:`current_boundary`), else the running ``"turn"`` boundary of the
    call's turn: turn -> llm -> tool nest without any plumbing.
    """
    from . import runtime

    ctx = ctx if ctx is not None else _current_component()
    if harness is None:
        harness = resolve_harness(_scoped_from(ctx))
    session = (getattr(ctx, "session_services", None) or {}).get("session") if ctx else None
    turn_id = runtime.current_turn_id(ctx) if ctx is not None else runtime.current_turn_id()
    if parent_id is None:
        parent = _current.get()
        if parent is None and turn_id is not None and kind != "turn":
            parent = _turns.get(turn_id)
        parent_id = parent.id if parent is not None else None
    b = Boundary(
        kind=kind,
        name=name,
        inputs=dict(inputs or {}),
        component=getattr(ctx, "path", None) if ctx is not None else None,
        session_id=getattr(session, "id", None),
        turn_id=turn_id,
        attrs=dict(attrs),
        parent_id=parent_id,
    )
    return await harness.run(b, fn)
