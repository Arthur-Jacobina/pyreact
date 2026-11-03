import inspect

from pyreact.core.core import component, hooks
from pyreact.core.turns import TurnManager
from pyreact.input.bus import InputBus, Event


def _accepts_two_args(fn) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    positional = 0
    for p in sig.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            return True
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            positional += 1
    return positional >= 2


@component
def Keystroke(on_submit=None):
    """Calls ``on_submit(text)`` or ``on_submit(text, turn)`` on each submit.

    ``turn`` is the :class:`~pyreact.core.turns.Turn` the input started (or
    ``None`` if the input was emitted on the bus outside the turn system).
    """
    state, set_state = hooks.use_state({"text": "", "submit_ver": 0, "turn_id": None})
    bus: InputBus = hooks.get_service("input_bus", InputBus)
    turns: TurnManager = hooks.get_service("turns", TurnManager)

    # Bus handler
    def _handle(ev: Event):
        t = ev.get("type")
        v = ev.get("value", "") or ""
        if t == "submit":
            tid = ev.get("turn_id")
            set_state(
                lambda s: {"text": v, "submit_ver": s["submit_ver"] + 1, "turn_id": tid}
            )

    handler = hooks.use_callback(_handle, deps=[])

    def _bus_effect():
        return bus.subscribe(handler)

    hooks.use_effect(_bus_effect, [handler])

    def _on_submit_effect():
        # on_submit: triggers only after at least one submit
        if on_submit is None or state["submit_ver"] == 0:
            return
        if _accepts_two_args(on_submit):
            on_submit(state["text"], turns.get(state["turn_id"]))
        else:
            on_submit(state["text"])

    hooks.use_effect(_on_submit_effect, [state["submit_ver"]])

    return []
