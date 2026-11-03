"""Token / cost / latency accounting from ``llm.end`` events."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional

from pyreact.core.events import Event

from ._base import Sink, payload_get

NO_TURN = "-"


class TurnTracker:
    """Resolves the turn an event belongs to.

    Uses ``Event.turn_id`` (or ``payload["turn_id"]``); events without one are
    attributed to the most recently started, not yet ended turn.
    """

    def __init__(self) -> None:
        self.open: Optional[str] = None

    def observe(self, ev: Event) -> Optional[str]:
        tid = ev.turn_id or payload_get(ev.payload, "turn_id")
        tid = str(tid) if tid is not None else None
        if ev.kind == "turn.start":
            self.open = tid
            return tid
        resolved = tid or self.open
        if ev.kind == "turn.end" and self.open is not None and self.open == resolved:
            self.open = None
        return resolved


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_usage(usage: Any) -> Optional[Dict[str, Any]]:
    """Map OpenAI/Anthropic/DSPy-style usage dicts to
    ``{"prompt_tokens", "completion_tokens", "total_tokens", "estimated"}``.

    Accepts nested per-model dicts (``{"openai/gpt-4o": {...}}``, as returned by
    ``dspy`` usage tracking) and sums them. Returns ``None`` when no token
    counts are present.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    nested = [v for v in usage.values() if isinstance(v, dict)]
    if nested and not any(
        k in usage for k in ("prompt_tokens", "input_tokens", "total_tokens")
    ):
        parts = [normalize_usage(v) for v in nested]
        parts = [p for p in parts if p]
        if not parts:
            return None
        return {
            "prompt_tokens": sum(p["prompt_tokens"] for p in parts),
            "completion_tokens": sum(p["completion_tokens"] for p in parts),
            "total_tokens": sum(p["total_tokens"] for p in parts),
            "estimated": any(p["estimated"] for p in parts),
        }
    prompt = _num(payload_get(usage, "prompt_tokens", "input_tokens"))
    completion = _num(payload_get(usage, "completion_tokens", "output_tokens"))
    total = _num(usage.get("total_tokens"))
    if prompt is None and completion is None and total is None:
        return None
    prompt = prompt or 0
    completion = completion or 0
    if total is None:
        total = prompt + completion
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "estimated": bool(usage.get("estimated")),
    }


@dataclass
class UsageStats:
    calls: int = 0
    errors: int = 0
    calls_with_usage: int = 0
    estimated_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None  # None until some call reports a cost
    latency_total: float = 0.0
    latency_max: float = 0.0

    @property
    def calls_without_usage(self) -> int:
        return self.calls - self.calls_with_usage

    @property
    def latency_avg(self) -> float:
        return self.latency_total / self.calls if self.calls else 0.0

    def add(self, payload: Dict[str, Any]) -> None:
        self.calls += 1
        u = normalize_usage(payload_get(payload, "usage"))
        if u is not None:
            self.calls_with_usage += 1
            self.estimated_calls += int(u["estimated"])
            self.prompt_tokens += u["prompt_tokens"]
            self.completion_tokens += u["completion_tokens"]
            self.total_tokens += u["total_tokens"]
        cost = _num(payload_get(payload, "cost"))
        if cost is not None:
            self.cost = (self.cost or 0.0) + cost
        lat = _num(payload_get(payload, "latency", "duration"))
        if lat is not None:
            self.latency_total += lat
            self.latency_max = max(self.latency_max, lat)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["calls_without_usage"] = self.calls_without_usage
        d["latency_avg"] = self.latency_avg
        return d

    def summary(self) -> str:
        if not self.calls and not self.errors:
            return "no llm calls"
        parts = [f"{self.calls} call{'s' if self.calls != 1 else ''}"]
        if self.calls_with_usage:
            tok = f"{self.total_tokens} tok ({self.prompt_tokens} in / {self.completion_tokens} out)"
            if self.estimated_calls:
                tok += " ~est" if self.estimated_calls == self.calls_with_usage else " (partly ~est)"
            parts.append(tok)
        if self.calls_without_usage:
            parts.append(f"{self.calls_without_usage} without usage")
        parts.append(f"cost ${self.cost:.4f}" if self.cost is not None else "cost n/a")
        if self.calls:
            parts.append(f"latency {self.latency_total:.1f}s (avg {self.latency_avg:.1f}s, max {self.latency_max:.1f}s)")
        if self.errors:
            parts.append(f"{self.errors} error{'s' if self.errors != 1 else ''}")
        return ", ".join(parts)


class UsageAggregator(Sink):
    """Aggregates ``llm.end`` payloads ``{"usage", "cost", "latency"}`` per turn
    and per session. Missing/empty usage is counted (``calls_without_usage``)
    rather than failing; ``llm.error`` events count as errors."""

    def __init__(self, kinds: Optional[Iterable[str]] = None) -> None:
        super().__init__(kinds)
        self.session = UsageStats()
        self.turns: "OrderedDict[str, UsageStats]" = OrderedDict()
        self._tracker = TurnTracker()
        self._lock = threading.Lock()

    def _turn(self, tid: Optional[str]) -> UsageStats:
        key = tid or NO_TURN
        st = self.turns.get(key)
        if st is None:
            st = self.turns[key] = UsageStats()
        return st

    def handle(self, ev: Event) -> None:
        with self._lock:
            tid = self._tracker.observe(ev)
            if ev.kind == "llm.end":
                self.session.add(ev.payload)
                self._turn(tid).add(ev.payload)
            elif ev.kind == "llm.error":
                self.session.errors += 1
                self._turn(tid).errors += 1

    def turn(self, turn_id: str) -> UsageStats:
        return self.turns.get(turn_id, UsageStats())

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session": self.session.to_dict(),
                "turns": {k: v.to_dict() for k, v in self.turns.items()},
            }

    def render(self, last: Optional[int] = 10) -> str:
        with self._lock:
            lines = [f"session: {self.session.summary()}"]
            items = list(self.turns.items())
            if last is not None:
                items = items[-last:]
            for tid, st in items:
                lines.append(f"  turn {tid}: {st.summary()}")
        return "\n".join(lines)


def estimate_tokens(text: Any) -> int:
    """Rough char-based token estimate (~4 chars/token). Approximate only."""
    if not text:
        return 0
    return max(1, (len(str(text)) + 3) // 4)
