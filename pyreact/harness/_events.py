"""Publishing ``guard.*`` / ``verify.*`` events for a boundary.

Same routing as the LLM/tool events of the DSPy layer: the calling
component's *session* bus (what ``app.events`` / ``:events`` observe) and the
process-global bus (once when they are the same object). Observability never
breaks a call: publishing errors are swallowed.
"""

from __future__ import annotations

from typing import Any, Dict

from pyreact.core.events import EventBus
from pyreact.core.harness import Boundary
from pyreact.core.hook import HookContext


def boundary_info(b: Boundary) -> Dict[str, Any]:
    """The fields every guard/verify event carries about its boundary."""
    return {
        "kind": b.kind,
        "name": b.name,
        "boundary_id": b.id,
        "call_id": b.attrs.get("call_id"),
        "component": b.component,
        "attempt": b.attempt,
    }


def publish(kind: str, payload: Dict[str, Any], *, turn_id=None, source=None) -> None:
    try:
        from pyreact.core import runtime

        glob = HookContext.get_service("event_bus", EventBus)
        sess = None
        ctx = runtime.current_component()
        if ctx is not None and hasattr(ctx, "get_service"):
            sess = ctx.get_service("event_bus", EventBus)
        if sess is not None and sess is not glob:
            sess.publish(kind, payload, turn_id=turn_id, source=source)
        glob.publish(kind, payload, turn_id=turn_id, source=source)
    except Exception:
        pass


def emit(kind: str, b: Boundary, **payload: Any) -> None:
    """Publish ``kind`` about boundary ``b`` (``turn_id`` / ``source`` from it)."""
    publish(kind, {**boundary_info(b), **payload}, turn_id=b.turn_id, source=b.component)
