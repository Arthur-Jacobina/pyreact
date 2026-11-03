# events.py -------------------------------------------------
"""Typed application event bus (the "DOM events" of the agent tree).

Everything observable that happens in an app flows through here as an
:class:`Event`: user input, chat messages, LLM calls/usage, tool calls, turn
boundaries, errors. Transports (terminal, web) and sinks (logs, traces,
optimizer datasets) subscribe; components publish.

Conventional kinds (dot-namespaced; extend freely):
    input.text, input.submit        user input (payload: {"value": str})
    turn.start, turn.end            turn lifecycle
    message                         chat message (payload: {"text", "sender", "message_type"})
    llm.start, llm.end, llm.error   LM/module call lifecycle (payload: usage, cost, latency)
    tool.start, tool.end            tool call lifecycle
    error                           component error (payload: {"phase", "component", "error"})
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_ids = itertools.count(1)


@dataclass(frozen=True)
class Event:
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    turn_id: Optional[str] = None
    source: Optional[str] = None  # component path, "web", "term", ...
    ts: float = field(default_factory=time.time)
    id: int = field(default_factory=lambda: next(_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "payload": self.payload,
            "turn_id": self.turn_id,
            "source": self.source,
            "ts": self.ts,
        }


Listener = Callable[[Event], None]


def _matches(kind: str, patterns: Optional[Tuple[str, ...]]) -> bool:
    if patterns is None:
        return True
    for p in patterns:
        if p == kind or (p.endswith(".*") and kind.startswith(p[:-1])) or p == "*":
            return True
    return False


class EventBus:
    """Thread-safe pub/sub. Listener errors are reported, never propagated."""

    def __init__(self) -> None:
        self._subs: List[Tuple[Listener, Optional[Tuple[str, ...]]]] = []
        self._lock = threading.Lock()

    def subscribe(
        self, fn: Listener, kinds: Optional[Iterable[str]] = None
    ) -> Callable[[], None]:
        """Subscribe ``fn``; ``kinds`` filters by exact kind or ``"prefix.*"``."""
        entry = (fn, tuple(kinds) if kinds is not None else None)
        with self._lock:
            self._subs.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subs.remove(entry)
                except ValueError:
                    pass

        return unsubscribe

    def publish(
        self,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        turn_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Event:
        ev = Event(
            kind=kind, payload=dict(payload or {}), turn_id=turn_id, source=source
        )
        self.emit(ev)
        return ev

    def emit(self, ev: Event) -> None:
        with self._lock:
            subs = list(self._subs)
        for fn, kinds in subs:
            if not _matches(ev.kind, kinds):
                continue
            try:
                fn(ev)
            except Exception as exc:  # never let a sink break publishers
                from .runtime import report_error

                report_error(exc, None, "event-listener")
