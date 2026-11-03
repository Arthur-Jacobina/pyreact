"""Observability around the typed :class:`~pyreact.core.events.EventBus`.

Sinks are callables ``sink(event)`` with an optional kind filter; attach them
to an ``EventBus`` (``sink.attach(bus)``) or to an :class:`EventLog`
(``sink.attach(log, replay=True)`` to also receive already-recorded events).

    from pyreact.observability import JSONLSink, TurnTimeline
    sink = JSONLSink("logs/events.jsonl"); sink.attach(app.events, replay=True)
    print(app.timeline.render())

``AppRunner`` wires an :class:`Observer` (EventLog + UsageAggregator +
TurnTimeline) and the ``error``-event handler automatically.
"""

from __future__ import annotations

from typing import Optional

from pyreact.core.events import EventBus

from ._base import Sink, component_path, current_turn_id, get_event_bus
from .console import ConsoleSink, format_event, format_message
from .errors import error_payload, install_error_events, make_error_handler
from .jsonl import JSONLSink, read_jsonl
from .log import EventLog
from .timeline import Turn, TurnTimeline
from .usage import UsageAggregator, UsageStats, estimate_tokens, normalize_usage


class Observer:
    """Default per-app bundle: a replayable :class:`EventLog` plus a
    :class:`UsageAggregator` and :class:`TurnTimeline` fed from it."""

    def __init__(self, maxlen: int = 2000) -> None:
        self.log = EventLog(maxlen=maxlen)
        self.usage = UsageAggregator()
        self.timeline = TurnTimeline()
        self.bus: Optional[EventBus] = None

    def attach(self, bus: EventBus) -> "Observer":
        self.bus = bus
        self.log.attach(bus)
        self.usage.attach(self.log, replay=True)
        self.timeline.attach(self.log, replay=True)
        return self

    def detach(self) -> None:
        self.usage.detach()
        self.timeline.detach()
        self.log.detach()


__all__ = [
    "ConsoleSink",
    "EventLog",
    "JSONLSink",
    "Observer",
    "Sink",
    "Turn",
    "TurnTimeline",
    "UsageAggregator",
    "UsageStats",
    "component_path",
    "current_turn_id",
    "error_payload",
    "estimate_tokens",
    "format_event",
    "format_message",
    "get_event_bus",
    "install_error_events",
    "make_error_handler",
    "normalize_usage",
    "read_jsonl",
]
