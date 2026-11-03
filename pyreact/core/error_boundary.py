# error_boundary.py -----------------------------------------
"""``ErrorBoundary``: React error boundaries for agent trees.

Without a boundary a failing component keeps its last good subtree and the
error is reported (``error`` event, turn status ``"error"``). Inside an
``ErrorBoundary`` the error is *handled*: the boundary unmounts its children
and renders ``fallback`` instead, calls ``on_error``, and re-mounts the
children when it resets::

    ErrorBoundary(
        fallback=lambda error: Message(text=f"(chart unavailable: {error})"),
        on_error=lambda error, info: log.warning("%s failed: %s", info["component"], error),
        reset_on="turn",
        children=[Chart(key="chart")],
    )

Caught: errors raised while rendering a descendant, in its effects (sync, or
the task of an async effect) and in tasks it spawned (``runtime.spawn``) —
not errors of the boundary itself (they go to the next boundary up), not
cleanup errors, and not errors a hook turns into data (``use_call`` puts a
failed LLM call in ``call.error``). An error raised while the fallback is
shown goes to the next boundary up (or the default handling).
"""

from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable, List, Optional, Sequence, Union

from . import runtime
from .core import VNode, _context_stack, component, hooks

__all__ = ["ErrorBoundary"]

Fallback = Union[None, VNode, Sequence[VNode], Callable[..., Any]]
ResetOn = Union[str, Callable[[BaseException, dict], bool]]


def _call_fallback(fn: Callable[..., Any], error: BaseException, info: dict,
                   reset: Callable[[], None]) -> Any:
    offered = {"error": error, "info": info, "reset": reset}
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return fn(error)
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return fn(**offered)
    named = {p.name: offered[p.name] for p in params
             if p.name in offered and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)}
    if named:
        return fn(**named)
    return fn(error)


def _as_vnodes(out: Any, tag: str) -> List[VNode]:
    items = out if isinstance(out, (list, tuple)) else [out]
    nodes = [n for n in items if isinstance(n, VNode)]
    for i, n in enumerate(nodes):
        if n.key is None:
            # Never reconcile the fallback with the children it replaces, and
            # (the boundary being transparent) keep the paths of two
            # boundaries' fallbacks under one parent apart.
            n.key = f"{tag}.fallback{i}"
    return nodes


def _publish(ctx: Any, boundary_path: str, exc: BaseException, failed: Any, phase: str,
             turn_id: Optional[str]) -> None:
    try:
        bus = ctx.session_services.get("event_bus") if ctx is not None else None
        if bus is None:
            from .events import EventBus
            from .hook import HookContext

            bus = HookContext.get_service("event_bus", EventBus)
        bus.publish("error.caught", {
            "phase": phase,
            "component": getattr(failed, "path", None),
            "boundary": boundary_path,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": "".join(traceback.format_exception(type(exc), exc,
                                                            exc.__traceback__)),
        }, turn_id=turn_id, source=boundary_path)
    except Exception:
        pass


@component(transparent=True)
def ErrorBoundary(
    fallback: Fallback = None,
    on_error: Optional[Callable[[BaseException, dict], Any]] = None,
    reset_on: ResetOn = "turn",
    reset_keys: Optional[Sequence[Any]] = None,
    children=None,
):
    """Render ``children``; after an error in them, render ``fallback``.

    - ``fallback``: a VNode (or list), or a function / component called with
      the parameters it declares among ``error``, ``info``, ``reset`` (a
      plain one-argument function gets the error); ``None`` renders nothing.
    - ``on_error(error, info)``: called once per caught error; ``info`` is
      ``{"component", "phase", "turn_id", "boundary"}``.
    - ``reset_on`` — when to re-mount the children:
      ``"turn"`` (default): on the next turn (re-mounted children see it via
      ``use_turn()``); ``"props"``: when ``reset_keys`` change;
      a callable ``fn(error, info) -> bool`` checked on each render of the
      boundary. ``reset()`` (passed to the fallback) resets at any time.

    Like providers, the boundary is *transparent*: it adds no segment to
    its descendants' paths (``App/Chat[chat]`` with or without a boundary
    around it), so wrapping a subtree later does not orphan its durable
    state. Fallback nodes without a key get ``"<boundary key>.fallback<i>"``.

    A caught error is handled: it publishes ``error.caught`` (with the
    traceback) instead of ``error``, is not printed, and does not make the
    turn end as ``"error"``.
    """
    from .turns import use_turn

    if reset_on not in ("turn", "props") and not callable(reset_on):
        raise ValueError(f"reset_on must be 'turn', 'props' or a callable, got {reset_on!r}")
    ctx = _context_stack.get()
    keys = tuple(reset_keys) if reset_keys is not None else None
    record, set_record = hooks.use_state(None)
    box = hooks.use_memo(lambda: {"record": None, "shown": False}, [])
    turn = use_turn() if reset_on == "turn" else None
    path = ctx.path
    tag = str(ctx.key) if ctx.key is not None else f"#{ctx.index}"

    def reset() -> None:
        box["record"] = None
        set_record(None)

    def catch(exc: BaseException, failed: Any, phase: str) -> bool:
        if box["record"] is not None:
            # More errors of the subtree being replaced (or of its unmounted
            # leftovers) are absorbed; an error of the fallback goes up.
            return not box["shown"] or not getattr(failed, "_mounted", True)
        tid = runtime.current_turn_id(failed)
        info = {"component": getattr(failed, "path", None), "phase": phase,
                "turn_id": tid, "boundary": path}
        tm = ctx.session_services.get("turns")
        latest = getattr(getattr(tm, "latest", None), "id", None)
        rec = {"error": exc, "info": info, "keys": keys, "turns": {tid, latest}}
        box["record"] = rec
        set_record(rec)
        _publish(ctx, path, exc, failed, phase, tid)
        if on_error is not None:
            try:
                on_error(exc, info)
            except Exception as cb_exc:
                runtime.report_error(cb_exc, ctx, "on_error")
        return True

    ctx._catch_error = catch

    rec = box["record"] if box["record"] is not None else record
    if rec is not None:
        info = rec["info"]
        if reset_on == "turn":
            # a turn that started after the error (never the one it happened in)
            stale = turn is not None and turn.id not in rec["turns"]
        elif reset_on == "props":
            stale = keys != rec["keys"]
        else:
            stale = bool(reset_on(rec["error"], info))
        if stale:
            box["record"] = rec = None
            if record is not None:
                set_record(None)
    box["shown"] = rec is not None
    if rec is None:
        return list(children or [])
    if fallback is None:
        return []
    if isinstance(fallback, (VNode, list, tuple)):
        return _as_vnodes(fallback, tag)
    return _as_vnodes(_call_fallback(fallback, rec["error"], rec["info"], reset), tag)
