"""Route component errors (``runtime.report_error``) to ``error`` events."""

from __future__ import annotations

import sys
import threading
import traceback
from typing import Any, Callable, Optional

from pyreact.core import runtime
from pyreact.core.events import EventBus

from ._base import component_path, current_turn_id

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


def make_error_handler(
    bus: EventBus,
    *,
    stderr: bool = True,
    chain: Optional[Callable[[BaseException, Any, str], None]] = None,
) -> Callable[[BaseException, Any, str], None]:
    """Build a ``runtime`` error handler that publishes ``error`` events.

    The error is also passed to ``chain`` (a previously installed handler) if
    given, else printed to stderr when ``stderr`` is true. Errors raised while
    publishing (e.g. a failing ``error`` listener) never recurse.
    """

    def handler(exc: BaseException, ctx: Any, phase: str) -> None:
        if chain is not None:
            try:
                chain(exc, ctx, phase)
            except Exception:
                if stderr:
                    _print_stderr(exc, ctx, phase)
        elif stderr:
            _print_stderr(exc, ctx, phase)
        if getattr(_guard, "active", False):
            return  # error raised by an error listener: don't loop
        _guard.active = True
        try:
            bus.publish(
                "error",
                error_payload(exc, ctx, phase),
                turn_id=current_turn_id(),
                source=component_path(ctx) or phase,
            )
        except Exception:
            pass
        finally:
            _guard.active = False

    return handler


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
