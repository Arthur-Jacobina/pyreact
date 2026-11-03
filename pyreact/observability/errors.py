"""Route component errors (``runtime.report_error``) to ``error`` events."""

from __future__ import annotations

import sys
import threading
import traceback
from typing import Any, Callable, Optional

from pyreact.core import runtime
from pyreact.core.events import EventBus

from ._base import component_path

_guard = threading.local()


def _print_stderr(exc: BaseException, ctx: Any, phase: str) -> None:
    name = getattr(ctx, "name", None) or "?"
    print(f"[pyreact] error during {phase} of <{name}>:", file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def error_payload(exc: BaseException, ctx: Any, phase: str) -> dict:
    return {
        "phase": phase,
        "component": component_path(ctx),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def session_event_bus(ctx: Any = None) -> Optional[EventBus]:
    """EventBus of the session owning ``ctx`` (default: the current
    component), or ``None`` when the error is not attributable to one."""
    target = ctx if ctx is not None else runtime.current_component()
    services = runtime.services_of(target)
    bus = services.get("event_bus") if services else None
    return bus if bus is not None and hasattr(bus, "publish") else None


def _publish_once(
    bus: Optional[EventBus], exc: BaseException, ctx: Any, phase: str
) -> None:
    """Publish an ``error`` event on ``bus`` unless this report already did
    (several chained handlers may resolve to the same bus)."""
    if bus is None or getattr(_guard, "active", False):
        return  # error raised by an error listener: don't loop
    published = getattr(_guard, "published", None)
    if published is not None:
        if id(bus) in published:
            return
        published.add(id(bus))
    _guard.active = True
    try:
        bus.publish(
            "error",
            error_payload(exc, ctx, phase),
            turn_id=runtime.current_turn_id(ctx),
            source=component_path(ctx) or phase,
        )
    except Exception:
        pass
    finally:
        _guard.active = False


def _build_handler(
    resolve: Callable[[Any], Optional[EventBus]],
    *,
    stderr: bool,
    chain: Optional[Callable[[BaseException, Any, str], None]],
) -> Callable[[BaseException, Any, str], None]:
    def handler(exc: BaseException, ctx: Any, phase: str) -> None:
        outer = getattr(_guard, "published", None) is None
        if outer:
            _guard.published = set()
        try:
            if chain is not None:
                try:
                    chain(exc, ctx, phase)
                except Exception:
                    if stderr:
                        _print_stderr(exc, ctx, phase)
            elif stderr:
                _print_stderr(exc, ctx, phase)
            _publish_once(resolve(ctx), exc, ctx, phase)
        finally:
            if outer:
                _guard.published = None

    return handler


def make_error_handler(
    bus: EventBus,
    *,
    stderr: bool = True,
    chain: Optional[Callable[[BaseException, Any, str], None]] = None,
) -> Callable[[BaseException, Any, str], None]:
    """Build a ``runtime`` error handler that publishes ``error`` events on ``bus``.

    The error is also passed to ``chain`` (a previously installed handler) if
    given, else printed to stderr when ``stderr`` is true. Errors raised while
    publishing (e.g. a failing ``error`` listener) never recurse.
    """
    return _build_handler(lambda _ctx: bus, stderr=stderr, chain=chain)


def make_session_error_handler(
    *,
    stderr: bool = True,
    chain: Optional[Callable[[BaseException, Any, str], None]] = None,
) -> Callable[[BaseException, Any, str], None]:
    """Like :func:`make_error_handler`, but publishes each error on the
    EventBus of the session owning the failing component (see
    :func:`session_event_bus`). Unattributable errors are only printed."""
    return _build_handler(session_event_bus, stderr=stderr, chain=chain)


def install_error_events(bus: EventBus, *, stderr: bool = True) -> Callable[[], None]:
    """Install :func:`make_error_handler` on the runtime; returns an uninstaller
    that restores the previously installed handler."""
    previous = runtime._error_handler
    handler = make_error_handler(bus, stderr=stderr, chain=previous)
    runtime.set_error_handler(handler)

    def uninstall() -> None:
        if runtime._error_handler is handler:
            runtime.set_error_handler(previous)

    return uninstall


_session_lock = threading.Lock()
_session_installs = {"count": 0, "handler": None, "previous": None}


def install_session_error_events(*, stderr: bool = True) -> Callable[[], None]:
    """Install (reference-counted, once per process) the session-aware error
    handler; every AppRunner calls this. Returns an idempotent uninstaller."""
    with _session_lock:
        st = _session_installs
        if st["count"] == 0:
            previous = runtime._error_handler
            st["previous"] = previous
            st["handler"] = make_session_error_handler(stderr=stderr, chain=previous)
            runtime.set_error_handler(st["handler"])
        st["count"] += 1
    done = {"v": False}

    def uninstall() -> None:
        with _session_lock:
            if done["v"]:
                return
            done["v"] = True
            st["count"] = max(0, st["count"] - 1)
            if st["count"] == 0:
                if runtime._error_handler is st["handler"]:
                    runtime.set_error_handler(st["previous"])
                st["handler"] = st["previous"] = None

    return uninstall
