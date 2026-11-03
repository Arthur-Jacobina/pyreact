"""Hooks for a component's session I/O: raw input.

Instead of looking up the session's ``InputBus`` with
``hooks.get_service("input_bus", ...)`` and subscribing by hand, a component
asks for the latest input with :func:`use_input` (cf. a controlled input's
``onChange`` in React)::

    @component
    def Echo():
        ev = use_input(types=("submit",))         # rerenders on every submit
        return [Message(key=f"m{ev['seq']}", text=ev["value"])] if ev else []

To run work once per input prefer :func:`~pyreact.core.turn_hooks.use_on_turn`
(turns are the identity of an input); ``use_input`` is the lower-level view
of what the transports emit (``text`` / ``submit`` / ``key`` events).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

from pyreact.input.bus import Event as InputEvent
from pyreact.input.bus import InputBus

from .core import hooks

__all__ = ["use_input"]

INPUT_BUS = "input_bus"


def _kinds(kinds: Union[str, Iterable[str], None]) -> Optional[Tuple[str, ...]]:
    if kinds is None:
        return None
    if isinstance(kinds, str):
        return (kinds,)
    return tuple(kinds)


def use_input(types: Union[str, Iterable[str], None] = None) -> Optional[InputEvent]:
    """The latest raw input event of this component's session; rerenders on
    every new one.

    ``types`` filters on the event's ``"type"`` (``"submit"``,
    ``("text", "submit")``...; default: every input). Returns ``None`` until
    an input arrives after the component mounted (inputs are not replayed;
    :func:`~pyreact.core.turns.use_turn` gives the in-flight turn instead).

    The returned dict is a copy of the bus event (``type``, ``value``,
    ``source``, ``ts``, ``turn_id`` when the input started a turn) plus
    ``seq``: how many matching inputs this component has seen, so two equal
    inputs are still two values (key effects on ``ev["seq"]``).
    """
    bus: InputBus = hooks.get_service(INPUT_BUS, InputBus)
    wanted = _kinds(types)
    latest, set_latest = hooks.use_state(None)

    def _subscribe():
        def _on_input(ev: InputEvent) -> None:
            if wanted is not None and ev.get("type") not in wanted:
                return
            set_latest(lambda prev: {**ev, "seq": (prev["seq"] if prev else 0) + 1})

        return bus.subscribe(_on_input)

    hooks.use_effect(_subscribe, [id(bus), wanted])
    return latest
