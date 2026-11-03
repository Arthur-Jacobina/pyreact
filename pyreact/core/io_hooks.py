"""Hooks for a component's session I/O: raw input and app events.

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

Events on the session ``EventBus`` (``pyreact.core.events``) go through
:func:`use_event` (subscribe in an effect, cf. ``addEventListener`` in a
``useEffect``) and :func:`use_emit` (publish, attributed to the component and
the current turn) instead of ``hooks.get_service("event_bus", ...)``::

    @component
    def Log(text):
        emit = use_emit()
        hooks.use_effect(lambda: emit("log", {"text": text}), [text])
        return []

    @component
    def Counter():
        n, set_n = hooks.use_state(0)
        use_event("log", lambda ev: set_n(lambda v: v + 1))
        ...

Per ADR 0001 events are for devtools and cross-cutting wiring, not for
rendering: output is components (``Message``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from pyreact.input.bus import Event as InputEvent
from pyreact.input.bus import InputBus

from . import runtime
from .core import hooks, use_path
from .events import Event, EventBus

__all__ = ["use_input", "use_event", "use_emit", "Emit"]

INPUT_BUS = "input_bus"
EVENT_BUS = "event_bus"

_AUTO: Any = object()

Emit = Callable[..., Event]


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


def use_event(kinds: Union[str, Iterable[str], None], fn: Callable[[Event], Any]) -> None:
    """Call ``fn(event)`` for every event of ``kinds`` on this component's
    session ``EventBus`` while it is mounted.

    ``kinds``: an exact kind (``"message"``), a ``"prefix.*"`` pattern, a
    list of them, or ``None`` for every event. The subscription is made in
    an effect (after commit) and removed by its cleanup (unmount, or when
    ``kinds`` change). ``fn`` may change every render: the listener always
    calls the latest one (through a ref), so it sees the latest props/state
    without resubscribing. It runs synchronously on the publisher's thread;
    update state (``set_state``) rather than doing slow work there.
    """
    bus: EventBus = hooks.get_service(EVENT_BUS, EventBus)
    patterns = _kinds(kinds)
    latest = hooks.use_ref(fn)
    latest.current = fn

    def _subscribe():
        return bus.subscribe(lambda ev: latest.current(ev), patterns)

    hooks.use_effect(_subscribe, [id(bus), patterns])


def use_emit() -> Emit:
    """A stable ``emit(kind, payload=None, *, turn_id=..., source=...)`` that
    publishes on this component's session ``EventBus`` and returns the
    :class:`~pyreact.core.events.Event`.

    By default the event's ``source`` is the component's path
    (:func:`~pyreact.core.core.use_path`) and its ``turn_id`` the turn the
    calling code belongs to at emit time (a task's turn, else the turn the
    component's session is processing); pass either explicitly to
    override. Call ``emit`` from effects, callbacks and tasks, not during
    render (rendering must not have side effects).
    """
    bus: EventBus = hooks.get_service(EVENT_BUS, EventBus)
    path = use_path()
    ctx = runtime.current_component()

    def emit(
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        turn_id: Optional[str] = _AUTO,
        source: Optional[str] = _AUTO,
    ) -> Event:
        return bus.publish(
            kind,
            payload,
            turn_id=runtime.current_turn_id(ctx) if turn_id is _AUTO else turn_id,
            source=path if source is _AUTO else source,
        )

    return hooks.use_callback(emit, [id(bus), path])
