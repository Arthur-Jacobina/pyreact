"""``Message``: an agent's chat output, published as a typed ``message`` event."""

from __future__ import annotations

import json
import time
from typing import Literal, Optional

from pyreact.core.core import component, hooks
from pyreact.core.io_hooks import use_emit
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
    trigger: Literal["change", "mount", "turn"] = "change",
    sender: Literal["user", "system", "assistant"] = "user",
    message_type: Literal["chat", "info", "warning", "error"] = "chat",
    legacy: Optional[bool] = None,
):
    """Chat message. On commit publishes::

        message  {"text", "sender", "message_type", "timestamp"}

    on the app ``EventBus`` with :func:`~pyreact.core.io_hooks.use_emit`
    (source = component path, turn_id = current turn when known). When it
    publishes depends on ``trigger``:

    - ``"change"`` (default): on mount and whenever ``text`` changes;
    - ``"mount"``: once;
    - ``"turn"``: on mount and whenever ``text`` *or the current turn*
      changes, i.e. once per ``(turn, text)``: the same reply to two inputs
      is published twice.

    ``legacy`` overrides the global legacy-bridge flag.
    """
    from pyreact.observability._base import current_turn_id

    emit = use_emit()
    console = hooks.get_service("console_buffer", MessageBuffer)
    if trigger == "mount":
        deps = []
    elif trigger == "turn":
        deps = [text, current_turn_id()]
    else:
        deps = [text]

    def _send_message():
        payload = {
            "text": text,
            "sender": sender,
            "message_type": message_type,
            "timestamp": time.time(),
        }
        emit("message", payload)
        if _legacy_bridge if legacy is None else legacy:
            console.append(legacy_line(payload))

    hooks.use_effect(_send_message, deps)
    return []
