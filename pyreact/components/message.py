"""``Message``: an agent's chat output, published as a typed ``message`` event."""

from __future__ import annotations

import json
import time
from typing import Literal, Optional

import pyreact.core.core as _core
from pyreact.core.core import component, hooks
from pyreact.core.events import EventBus
from pyreact.core.message_buffer import MessageBuffer

LEGACY_PREFIX = "__MESSAGE__:"

# Compatibility bridge: also write ``__MESSAGE__:{json}`` lines to the stdout
# MessageBuffer so transports that still parse that prefix (the web UI) keep
# working. Disable globally with ``set_legacy_message_bridge(False)`` or per
# message with ``Message(..., legacy=False)``.
_legacy_bridge = True


def set_legacy_message_bridge(enabled: bool) -> None:
    global _legacy_bridge
    _legacy_bridge = bool(enabled)


def legacy_message_bridge_enabled() -> bool:
    return _legacy_bridge


def legacy_line(payload: dict) -> str:
    data = {"type": "message", **payload}
    return f"{LEGACY_PREFIX}{json.dumps(data)}\n"


def parse_legacy_line(text: str) -> Optional[dict]:
    """Inverse of :func:`legacy_line`; ``None`` if ``text`` is not a message line."""
    if not text.startswith(LEGACY_PREFIX):
        return None
    try:
        return json.loads(text[len(LEGACY_PREFIX):])
    except ValueError:
        return None


@component
def Message(
    text: str,
    trigger: Literal["change", "mount"] = "change",
    sender: Literal["user", "system", "assistant"] = "user",
    message_type: Literal["chat", "info", "warning", "error"] = "chat",
    legacy: Optional[bool] = None,
):
    """Chat message. On commit (mount, or whenever ``text`` changes when
    ``trigger="change"``) publishes::

        message  {"text", "sender", "message_type", "timestamp"}

    on the app ``EventBus`` (source = component path, turn_id = current turn
    when known). ``legacy`` overrides the global legacy-bridge flag.
    """
    from pyreact.observability._base import component_path, current_turn_id

    bus = hooks.get_service("event_bus", EventBus)
    console = hooks.get_service("console_buffer", MessageBuffer)
    ctx = _core._context_stack.get()
    deps = [] if trigger == "mount" else [text]

    def _send_message():
        payload = {
            "text": text,
            "sender": sender,
            "message_type": message_type,
            "timestamp": time.time(),
        }
        bus.publish(
            "message",
            payload,
            turn_id=current_turn_id(),
            source=component_path(ctx),
        )
        if _legacy_bridge if legacy is None else legacy:
            console.append(legacy_line(payload))

    hooks.use_effect(_send_message, deps)
    return []
