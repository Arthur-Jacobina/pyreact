import inspect

from pyreact.core.core import component, hooks
from pyreact.core.io_hooks import use_input
from pyreact.core.turns import TurnManager, get_turn_manager


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
    Built on :func:`~pyreact.core.io_hooks.use_input`.
    """
    ev = use_input(types=("submit",))
    turns: TurnManager = get_turn_manager()
    latest = hooks.use_ref(on_submit)
    latest.current = on_submit

    def _on_submit_effect():
        # on_submit: triggers only after at least one submit
        cb = latest.current
        if cb is None or ev is None:
            return
        text = ev.get("value", "") or ""
        if _accepts_two_args(cb):
            cb(text, turns.get(ev.get("turn_id")))
        else:
            cb(text)

    hooks.use_effect(_on_submit_effect, [ev["seq"] if ev is not None else 0])

    return []
