from typing import Literal

import pyreact.core.core as _core
from pyreact.core.core import component, hooks
from pyreact.core.events import EventBus


@component
def Log(text: str, trigger: Literal["change", "mount"] = "change"):
    """Print ``text`` and publish it as a ``log`` event ({"text"})."""
    from pyreact.observability import component_path, current_turn_id

    bus = hooks.get_service("event_bus", EventBus)
    ctx = _core._context_stack.get()
    deps = [] if trigger == "mount" else [text]

    def _log():
        print(text)
        bus.publish(
            "log", {"text": text}, turn_id=current_turn_id(), source=component_path(ctx)
        )

    hooks.use_effect(_log, deps)
    return []
