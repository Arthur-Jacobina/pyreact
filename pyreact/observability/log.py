"""In-memory ring buffer of recent events with replay-on-subscribe."""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, Iterable, List, Optional, Tuple

from pyreact.core.events import Event, EventBus, _matches


class EventLog:
    """Keeps the last ``maxlen`` events of a bus and re-broadcasts them.

    Subscribing through the log (instead of the bus) with ``replay=True``
    delivers already-recorded events first and then live ones, with no gap
    and no duplicates -- so a sink attached after the app booted still sees
    e.g. the welcome message published during the first render.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._events: Deque[Event] = deque(maxlen=maxlen)
        self._subs: List[
            Tuple[Callable[[Event], None], Optional[Tuple[str, ...]]]
        ] = []
        self._lock = threading.RLock()
        self._unsub_bus: Optional[Callable[[], None]] = None

    # -- wiring --------------------------------------------------------------
    def attach(self, bus: EventBus) -> "EventLog":
        self.detach()
        self._unsub_bus = bus.subscribe(self.record)
        return self

    def detach(self) -> None:
        if self._unsub_bus is not None:
            self._unsub_bus()
            self._unsub_bus = None

    def record(self, ev: Event) -> None:
        # Deliver under the lock so replay+subscribe is atomic w.r.t. new events.
        with self._lock:
            self._events.append(ev)
            for fn, kinds in list(self._subs):
                if not _matches(ev.kind, kinds):
                    continue
                try:
                    fn(ev)
                except Exception as exc:
                    from pyreact.core.runtime import report_error

                    report_error(exc, None, "event-listener")

    __call__ = record

    def subscribe(
        self,
        fn: Callable[[Event], None],
        kinds: Optional[Iterable[str]] = None,
        *,
        replay: bool = False,
    ) -> Callable[[], None]:
        """Like :meth:`EventBus.subscribe`; ``replay=True`` first replays history."""
        entry = (fn, tuple(kinds) if kinds is not None else None)
        with self._lock:
            if replay:
                for ev in list(self._events):
                    if _matches(ev.kind, entry[1]):
                        try:
                            fn(ev)
                        except Exception:
                            pass
            self._subs.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subs.remove(entry)
                except ValueError:
                    pass

        return unsubscribe

    # -- queries -------------------------------------------------------------
    def recent(
        self, n: Optional[int] = 20, kinds: Optional[Iterable[str]] = None
    ) -> List[Event]:
        pats = tuple(kinds) if kinds is not None else None
        with self._lock:
            evs = [e for e in self._events if _matches(e.kind, pats)]
        return evs if n is None else evs[-n:]

    def __len__(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
