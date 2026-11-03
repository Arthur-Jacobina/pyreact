"""EventBus -> OpenTelemetry bridge for things that are not boundaries.

Boundaries (turns, LLM/tool calls) become spans through the harness
middleware. Other app events (``message``, ``error``, ``feedback``,
``optimize.*``, your own kinds) are attached to the trace they belong to:

    bridge = EventBridge(tracer_provider, kinds=("message", "error", "feedback"))
    bridge.attach(app.event_bus)

Target: the current span (the publisher's context), else the live ``turn``
span of ``event.turn_id``, else (``orphans="span"``) a standalone
``pyreact.event <kind>`` span; ``orphans="drop"`` skips it.

``mode="event"`` (default) adds a span *event* named after the kind;
``mode="span"`` opens a zero-length child span ``pyreact.event <kind>``
instead -- for backends that do not display span events (Langfuse shows
them as ``event`` observations).

Payload scalars become ``pyreact.event.<key>`` attributes; content-like keys
(``text``, ``inputs``, ``outputs``, ...) only with ``capture_content=True``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from opentelemetry import trace
from opentelemetry.trace import Span

from pyreact.core.events import Event, EventBus, _matches

from . import attributes as A
from .tracing import _jsonable, _trace_names, turn_span

__all__ = ["DEFAULT_KINDS", "CONTENT_KEYS", "EventBridge"]

DEFAULT_KINDS = ("message", "error", "feedback", "optimize.*")
CONTENT_KEYS = frozenset({
    "text", "content", "message", "messages", "prompt", "completion", "inputs", "outputs",
    "question", "answer", "expected", "result", "args",
})

EventAttributes = Callable[[Event, Optional[Span], Optional[str]], Optional[Mapping[str, Any]]]


class EventBridge:
    def __init__(
        self,
        tracer_provider: Any = None,
        *,
        kinds: Optional[Iterable[str]] = DEFAULT_KINDS,
        mode: str = "event",
        orphans: str = "span",
        capture_content: bool = False,
        max_value_length: int = 2048,
        extra_attributes: Optional[EventAttributes] = None,
    ) -> None:
        if mode not in ("event", "span"):
            raise ValueError(f"mode must be 'event' or 'span', got {mode!r}")
        if orphans not in ("span", "drop"):
            raise ValueError(f"orphans must be 'span' or 'drop', got {orphans!r}")
        from . import __version__

        self.tracer = trace.get_tracer("pyreact_otel.bridge", __version__,
                                       tracer_provider=tracer_provider)
        self.kinds = tuple(kinds) if kinds is not None else None
        self.mode = mode
        self.orphans = orphans
        self.capture_content = capture_content
        self.max_value_length = max_value_length
        self.extra_attributes = extra_attributes
        self._unsubs: List[Callable[[], None]] = []

    # -- wiring -------------------------------------------------------------
    def attach(self, bus: EventBus) -> Callable[[], None]:
        unsub = bus.subscribe(self, kinds=self.kinds)
        self._unsubs.append(unsub)
        return unsub

    def detach(self) -> None:
        for u in self._unsubs:
            try:
                u()
            except Exception:
                pass
        self._unsubs.clear()

    # -- translation ----------------------------------------------------------
    def attributes(self, ev: Event) -> Dict[str, Any]:
        out: Dict[str, Any] = {"pyreact.event.kind": ev.kind, "pyreact.event.id": ev.id}
        if ev.source:
            out["pyreact.event.source"] = ev.source
        if ev.turn_id:
            out[A.TURN] = ev.turn_id
        for k, v in (ev.payload or {}).items():
            if v is None or (k in CONTENT_KEYS and not self.capture_content):
                continue
            if isinstance(v, (bool, int, float)):
                out[f"pyreact.event.{k}"] = v
                continue
            s = v if isinstance(v, str) else json.dumps(_jsonable(v), ensure_ascii=False, default=repr)
            if len(s) > self.max_value_length:
                s = s[: self.max_value_length] + f"...[truncated {len(s) - self.max_value_length} chars]"
            out[f"pyreact.event.{k}"] = s
        return out

    def _target(self, ev: Event) -> Optional[Span]:
        cur = trace.get_current_span()
        if cur.is_recording():
            return cur
        t = turn_span(ev.turn_id)
        if t is not None and t.is_recording():
            return t
        return None

    def __call__(self, ev: Event) -> None:
        if not _matches(ev.kind, self.kinds):
            return
        try:
            self._handle(ev)
        except Exception:  # telemetry must never break a publisher
            pass

    def _handle(self, ev: Event) -> None:
        target = self._target(ev)
        if target is None and self.orphans == "drop":
            return
        attrs = self.attributes(ev)
        root_name = None
        if target is not None:
            root_name = _trace_names.get(target.get_span_context().trace_id)
        if self.extra_attributes is not None:
            attrs.update(self.extra_attributes(ev, target, root_name) or {})
        ts = int(ev.ts * 1e9)
        if target is not None and self.mode == "event":
            target.add_event(ev.kind, attributes=attrs, timestamp=ts)
            return
        ctx = trace.set_span_in_context(target) if target is not None else None
        span = self.tracer.start_span(f"pyreact.event {ev.kind}", context=ctx,
                                      attributes=attrs, start_time=ts)
        if self.mode == "event":  # orphan in event mode: carry it as a span event too
            span.add_event(ev.kind, attributes=attrs, timestamp=ts)
        span.end(end_time=ts)
