"""Component-scoped guards and verifiers: ``use_guard``, ``use_verify``,
``use_middleware``.

``HarnessProvider`` scopes middleware to the subtree it *wraps*; these hooks
scope it to the component that *calls* them (and its descendants), with the
component's lifecycle: registered on first render, unregistered on unmount::

    @component
    def Shop():
        use_guard(lambda b: b.inputs.get("quantity", 0) <= 10 or "max 10 units",
                  kinds="tool")
        use_verify(schema(Order), on_fail="retry_with_feedback")
        agent = use_agent("request -> order", tools=[...])
        ...

A hook's middleware applies to every boundary whose ``component`` is this
component or below it, in the same tree (session). Order: all hook middleware runs
in the *innermost* layer (after process-wide middleware and providers,
closest to the call), ancestors' hooks outside descendants', and within one
component in hook order.

Stateful built-ins (``budget``, ``max_calls``, ``rate_limit``, a ``verify``
object) are created once: the instance passed on the first render is kept
(pass ``deps=[...]`` to replace it when they change). Plain functions
(a guard predicate, a check, a middleware function) are always called in
their latest version, so they may close over props and state.
"""

from __future__ import annotations

import itertools
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from pyreact.core.core import _context_stack, hooks
from pyreact.core.harness import (
    Boundary,
    Harness,
    Middleware,
    Next,
    _applies,
    _current_component,
    _name,
    install,
)

from .guards import guard
from .verify import _check_name, verify

__all__ = ["use_guard", "use_middleware", "use_verify", "scoped_middleware"]

_seq = itertools.count(1)


@dataclass
class _Entry:
    mw: Middleware
    path: str
    session_id: Optional[str]
    seq: int
    root: Optional[Callable[[], Any]] = None  # weakref to the tree's root
    depth: int = field(init=False)

    def __post_init__(self) -> None:
        self.depth = self.path.count("/")

    def covers(self, b: Boundary, root: Any = None) -> bool:
        """``b`` was made in this component's subtree: same tree (the calling
        component's root when known, else the session id) and a component
        path at or below ours."""
        comp = b.component
        if comp is None:
            return False
        mine = self.root() if self.root is not None else None
        if root is not None and mine is not None:
            if root is not mine:
                return False
        elif b.session_id != self.session_id:
            return False
        return comp == self.path or comp.startswith(self.path + "/")


_registry: List[_Entry] = []
_lock = threading.Lock()
_uninstall: Optional[Callable[[], None]] = None


def _calling_root() -> Any:
    ctx = _current_component()  # what through() scoped this boundary with
    return getattr(ctx, "_root", None) if ctx is not None else None


async def _component_hooks(b: Boundary, call_next: Next) -> Any:
    root = _calling_root()
    with _lock:
        chain = [e for e in _registry if e.covers(b, root) and _applies(e.mw, b.kind)]
    if not chain:
        return await call_next(b)
    chain.sort(key=lambda e: (e.depth, e.seq))
    return await Harness([e.mw for e in chain])._run(b, call_next)


_component_hooks.harness_name = "component-hooks"  # type: ignore[attr-defined]


def _register(mw: Middleware, ctx: Any, seq: int) -> _Entry:
    global _uninstall
    session = (getattr(ctx, "session_services", None) or {}).get("session")
    root = getattr(ctx, "_root", None)
    ref = weakref.ref(root) if root is not None else None
    entry = _Entry(mw, ctx.path, getattr(session, "id", None), seq, ref)
    with _lock:
        # forget hooks of trees that were dropped without unmounting
        _registry[:] = [e for e in _registry if e.root is None or e.root() is not None]
        _registry.append(entry)
        if _uninstall is None:  # first hook mounted: join the pipeline
            _uninstall = install(_component_hooks, innermost=True)
    return entry


def _unregister(entry: Optional[_Entry]) -> None:
    global _uninstall
    if entry is None:
        return
    with _lock:
        try:
            _registry.remove(entry)
        except ValueError:
            return
        if not _registry and _uninstall is not None:
            _uninstall()  # keep the empty-pipeline fast path when no hook is mounted
            _uninstall = None


def scoped_middleware() -> List[Middleware]:
    """Middleware currently registered by component hooks (debugging aid)."""
    with _lock:
        return [e.mw for e in _registry]


def _is_harness_obj(x: Any) -> bool:
    return bool(getattr(x, "__pyreact_harness__", False))


def _use_scoped(make: Callable[[dict], Middleware], latest: Any,
                deps: Optional[Sequence[Any]]) -> Middleware:
    """Register ``make(holder)`` for this component; ``holder["latest"]`` is
    the argument of the current render."""
    ctx = _context_stack.get()
    if ctx is None:
        raise RuntimeError("use_guard / use_verify / use_middleware must be called during render")
    holder = hooks.use_memo(lambda: {"latest": None, "entry": None, "seq": next(_seq)}, [])
    holder["latest"] = latest
    mw = hooks.use_memo(lambda: make(holder), list(deps) if deps is not None else [])
    entry = holder["entry"]
    if entry is None or entry.mw is not mw:
        # registered during render (not in an effect) so calls fired by the
        # effects of this very commit -- children's included -- see it
        _unregister(entry)
        holder["entry"] = _register(mw, ctx, holder["seq"])

    def mount():
        if holder["entry"] is None:  # remounted after an unmount
            holder["entry"] = _register(mw, ctx, holder["seq"])

        def unmount():
            _unregister(holder["entry"])
            holder["entry"] = None

        return unmount

    hooks.use_effect(mount, [])
    return mw


def use_middleware(mw: Middleware, *, deps: Optional[Sequence[Any]] = None) -> Middleware:
    """Register any middleware ``(b, call_next)`` for this component's subtree.
    Returns the registered middleware (e.g. to read ``budget.spent(...)``)."""

    def make(h: dict) -> Middleware:
        first = h["latest"]
        if _is_harness_obj(first):
            return first

        async def latest(b: Boundary, call_next: Next) -> Any:
            return await h["latest"](b, call_next)

        latest.kinds = getattr(first, "kinds", None)  # type: ignore[attr-defined]
        latest.harness_name = _name(first)  # type: ignore[attr-defined]
        return latest

    return _use_scoped(make, mw, deps)


def use_guard(fn: Any, *, kinds: Any = None, name: Optional[str] = None,
              deps: Optional[Sequence[Any]] = None) -> Middleware:
    """A guard for this component's subtree.

    ``fn``: a predicate ``fn(b) -> None/True | False | reason`` (see
    :func:`pyreact.harness.guard`; ``kinds`` / ``name`` apply to it) or a
    guard middleware such as ``tool_allowlist(...)`` / ``budget(...)``.
    """
    if _is_harness_obj(fn):
        if kinds is not None or name is not None:
            raise TypeError("kinds/name are set on the guard object itself, not on use_guard")
        return _use_scoped(lambda h: h["latest"], fn, deps)
    nm = name or getattr(fn, "__name__", "guard").replace("<lambda>", "use_guard")
    return _use_scoped(lambda h: guard(lambda b: h["latest"](b), kinds=kinds, name=nm),
                       fn, deps)


def use_verify(check: Any, *, deps: Optional[Sequence[Any]] = None, **options: Any) -> Middleware:
    """A verifier for this component's subtree.

    ``check``: a check ``(b, result)`` or list of checks plus
    :func:`pyreact.harness.verify` options (``on_fail``, ``kinds``,
    ``max_attempts``, ...), or a ``verify(...)`` object.
    """
    if _is_harness_obj(check):
        if options:
            raise TypeError("options are set on the verify object itself, not on use_verify")
        return _use_scoped(lambda h: h["latest"], check, deps)
    checks = list(check) if isinstance(check, (list, tuple)) else [check]

    def make(h: dict) -> Middleware:
        def proxy(i: int, c: Any) -> Callable:
            def run(b: Boundary, result: Any):
                cur = h["latest"]
                cur = cur[i] if isinstance(cur, (list, tuple)) else cur
                return cur(b, result)

            run.check_name = _check_name(c)  # type: ignore[attr-defined]
            if hasattr(c, "threshold"):
                run.threshold = c.threshold  # type: ignore[attr-defined]
            return run

        return verify([proxy(i, c) for i, c in enumerate(checks)], **options)

    return _use_scoped(make, check, [len(checks), *(deps or [])])
