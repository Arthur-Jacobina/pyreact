"""Shared helpers for observability sinks."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Tuple

from pyreact.core.events import Event, EventBus, _matches


def component_path(ctx: Any) -> Optional[str]:
    """``Root/App/StoreHome[store]/SupervisorAgent[supervisor]`` for a HookContext."""
    if ctx is None:
        return None
    path = getattr(ctx, "path", None)  # HookContext.path (pyreact.persist)
    if isinstance(path, str) and path:
        return path
    parts = []
    node = ctx
    depth = 0
    while node is not None and depth < 64:
        name = getattr(node, "name", None) or type(node).__name__
        key = getattr(node, "key", None)
        parts.append(f"{name}[{key}]" if key is not None else str(name))
        node = getattr(node, "parent", None)
        depth += 1
    return "/".join(reversed(parts))


def current_turn_id() -> Optional[str]:
    """The turn the calling code belongs to (see ``pyreact.core.turns``)."""
    from pyreact.core import runtime

    return runtime.current_turn_id()


def get_event_bus(ctx: Any = None) -> EventBus:
    """The ``"event_bus"`` service of ``ctx``'s session (default: the current
    component's), else the global registry's."""
    if ctx is None:
        from pyreact.core import runtime

        ctx = runtime.current_component()
    if ctx is not None and hasattr(ctx, "get_service"):
        return ctx.get_service("event_bus", EventBus)
    from pyreact.core.hook import HookContext

    return HookContext.get_service("event_bus", EventBus)


def payload_get(payload: Any, *keys: str, default: Any = None) -> Any:
    """First present, non-None key of ``payload`` (tolerates non-dicts)."""
    if not isinstance(payload, dict):
        return default
    for k in keys:
        v = payload.get(k)
        if v is not None:
            return v
    return default


class Sink:
    """Base class: a callable event listener with an optional kind filter.

    ``kinds`` accepts exact kinds, ``"prefix.*"`` or ``"*"``; ``None`` = all.
    """

    def __init__(self, kinds: Optional[Iterable[str]] = None) -> None:
        self.kinds: Optional[Tuple[str, ...]] = (
            tuple(kinds) if kinds is not None else None
        )
        self._unsubs: list = []

    def accepts(self, ev: Event) -> bool:
        return _matches(ev.kind, self.kinds)

    def __call__(self, ev: Event) -> None:
        if self.accepts(ev):
            self.handle(ev)

    def handle(self, ev: Event) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def attach(self, source: Any, **kw: Any) -> Callable[[], None]:
        """Subscribe to an :class:`EventBus` or :class:`EventLog`.

        Extra keyword args are forwarded (e.g. ``replay=True`` for EventLog).
        Returns an unsubscribe function (also called by :meth:`detach`).
        """
        unsub = source.subscribe(self, **kw)
        self._unsubs.append(unsub)
        return unsub

    def detach(self) -> None:
        for u in self._unsubs:
            try:
                u()
            except Exception:
                pass
        self._unsubs.clear()
