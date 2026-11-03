"""Group events by turn and render a compact per-turn summary."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from pyreact.core.events import Event

from ._base import Sink, payload_get
from .usage import NO_TURN, TurnTracker, UsageStats, normalize_usage
from .console import colorize, shorten


@dataclass
class Turn:
    id: str
    start: Optional[float] = None
    end: Optional[float] = None
    input: Optional[str] = None
    events: List[Event] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    llm_pending: int = 0
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tools_pending: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)

    @property
    def first_ts(self) -> Optional[float]:
        if self.start is not None:
            return self.start
        return self.events[0].ts if self.events else None

    @property
    def last_ts(self) -> Optional[float]:
        if self.end is not None:
            return self.end
        return self.events[-1].ts if self.events else None

    @property
    def closed(self) -> bool:
        return self.end is not None

    @property
    def duration(self) -> float:
        a, b = self.first_ts, self.last_ts
        return (b - a) if a is not None and b is not None else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "input": self.input,
            "messages": self.messages,
            "llm_calls": self.llm_calls,
            "tools": self.tools,
            "errors": self.errors,
            "usage": self.usage.to_dict(),
            "events": len(self.events),
        }


def _name(payload: Any, *keys: str) -> str:
    return str(payload_get(payload, *keys, default="?"))


class TurnTimeline(Sink):
    """Groups events by ``turn_id`` (events without one go to the open turn,
    else to the ``"-"`` bucket) and renders per-turn summaries.

    Tolerates missing fields: understands ``turn.start/end``, ``message``,
    ``llm.start/end/error``, ``tool.start/end/error`` and ``error``; other
    kinds are kept in ``Turn.events`` only.
    """

    def __init__(self, kinds: Optional[Iterable[str]] = None, *, max_turns: int = 200) -> None:
        super().__init__(kinds)
        self.turns: "OrderedDict[str, Turn]" = OrderedDict()
        self.max_turns = max_turns
        self._tracker = TurnTracker()
        self._lock = threading.Lock()

    def _get(self, tid: str) -> Turn:
        t = self.turns.get(tid)
        if t is None:
            t = self.turns[tid] = Turn(id=tid)
            while len(self.turns) > self.max_turns:
                self.turns.popitem(last=False)
        return t

    def handle(self, ev: Event) -> None:
        with self._lock:
            tid = self._tracker.observe(ev) or NO_TURN
            t = self._get(tid)
            t.events.append(ev)
            p = ev.payload if isinstance(ev.payload, dict) else {}
            k = ev.kind
            if k == "turn.start":
                t.start = ev.ts
                t.input = payload_get(p, "input", "text", "value", "message", "query")
            elif k == "turn.end":
                t.end = ev.ts
            elif k == "input.submit" and t.input is None:
                t.input = payload_get(p, "value", "text")
            elif k == "message":
                t.messages.append(
                    {
                        "ts": ev.ts,
                        "sender": p.get("sender", "?"),
                        "message_type": p.get("message_type", "chat"),
                        "text": p.get("text", ""),
                    }
                )
            elif k == "llm.start":
                t.llm_pending += 1
            elif k in ("llm.end", "llm.error"):
                t.llm_pending = max(0, t.llm_pending - 1)
                u = normalize_usage(p.get("usage"))
                if k == "llm.end":
                    t.usage.add(p)
                else:
                    t.usage.errors += 1
                t.llm_calls.append(
                    {
                        "ts": ev.ts,
                        "name": payload_get(p, "name", "module", "signature", "component")
                        or ev.source
                        or "?",
                        "model": payload_get(p, "model", "lm"),
                        "latency": payload_get(p, "latency", "duration"),
                        "tokens": u["total_tokens"] if u else None,
                        "estimated": bool(u and u["estimated"]),
                        "error": payload_get(p, "error") if k == "llm.error" else None,
                    }
                )
            elif k == "tool.start":
                t.tools_pending += 1
            elif k in ("tool.end", "tool.error"):
                t.tools_pending = max(0, t.tools_pending - 1)
                err = payload_get(p, "error")
                t.tools.append(
                    {
                        "ts": ev.ts,
                        "name": _name(p, "name", "tool"),
                        "ok": k == "tool.end" and not err,
                        "latency": payload_get(p, "latency", "duration"),
                        "error": err,
                    }
                )
            elif k == "error":
                t.errors.append(
                    {
                        "ts": ev.ts,
                        "phase": p.get("phase"),
                        "component": p.get("component") or ev.source,
                        "error": p.get("error"),
                    }
                )

    # ------------------------------------------------------------------ query
    def get(self, turn_id: str) -> Optional[Turn]:
        return self.turns.get(turn_id)

    def last(self, n: int = 1) -> List[Turn]:
        with self._lock:
            return list(self.turns.values())[-n:]

    # ----------------------------------------------------------------- render
    def render_turn(self, t: Turn, *, color: bool = False, width: int = 100) -> str:
        c = (lambda s, code: colorize(s, code)) if color else (lambda s, code: s)
        t0 = t.first_ts or time.time()
        title = "(no turn)" if t.id == NO_TURN else f"turn {t.id}"
        status = "" if t.closed or t.id == NO_TURN else c(" (open)", "33")
        counts = (
            f"{len(t.messages)} msg, {len(t.llm_calls)} llm"
            + (f" (+{t.llm_pending} pending)" if t.llm_pending else "")
            + f", {len(t.tools)} tool"
            + (f" (+{t.tools_pending} pending)" if t.tools_pending else "")
            + f", {len(t.errors)} err"
        )
        head = f"{c(title, '1;36')}{status}  {t.duration:.1f}s  [{counts}]"
        if t.input:
            head += f"  {c(repr(shorten(t.input, 60)), '33')}"
        lines = [head]
        rows = []
        for m in t.messages:
            label = f"[{str(m['sender']).upper()}]"
            if m["message_type"] not in ("chat", None):
                label += f"({m['message_type']})"
            rows.append((m["ts"], f"{c('msg ', '32')} {label} {shorten(m['text'], width)}"))
        for call in t.llm_calls:
            bits = [str(call["name"])]
            if call["model"]:
                bits.append(f"model={call['model']}")
            if call["latency"] is not None:
                bits.append(f"{float(call['latency']):.1f}s")
            if call["tokens"] is not None:
                bits.append(f"{call['tokens']} tok" + ("~est" if call["estimated"] else ""))
            else:
                bits.append("usage n/a")
            if call["error"]:
                bits.append(c(f"ERROR {shorten(call['error'], 60)}", "31"))
            rows.append((call["ts"], f"{c('llm ', '35')} " + " ".join(bits)))
        for tool in t.tools:
            bits = [str(tool["name"]), "ok" if tool["ok"] else c("FAILED", "31")]
            if tool["latency"] is not None:
                bits.append(f"{float(tool['latency']):.2f}s")
            if tool["error"]:
                bits.append(shorten(tool["error"], 60))
            rows.append((tool["ts"], f"{c('tool', '34')} " + " ".join(bits)))
        for e in t.errors:
            rows.append(
                (e["ts"], c(f"err  {e['phase']} <{e['component']}>: {shorten(e['error'], width)}", "31"))
            )
        rows.sort(key=lambda r: r[0])
        for ts, text in rows:
            lines.append(f"  {c(f'+{ts - t0:5.1f}s', '90')} {text}")
        if t.usage.calls or t.usage.errors:
            lines.append(f"  {c('usage', '90')} {t.usage.summary()}")
        return "\n".join(lines)

    def render(self, last: Optional[int] = None, *, color: bool = False, include_untracked: bool = True) -> str:
        with self._lock:
            turns = [t for t in self.turns.values() if include_untracked or t.id != NO_TURN]
            if last is not None:
                turns = turns[-last:]
            if not turns:
                return "(no turns recorded)"
            return "\n\n".join(self.render_turn(t, color=color) for t in turns)

    def to_dict(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [t.to_dict() for t in self.turns.values()]
