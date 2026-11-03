"""Pretty, colored, kind-filterable console rendering of events."""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, Iterable, Optional, TextIO

from pyreact.core.events import Event, _matches

from ._base import Sink, payload_get

RESET = "\x1b[0m"

SENDER_COLORS = {"user": "34", "system": "90", "assistant": "32"}
TYPE_COLORS = {"chat": "", "info": "36", "warning": "33", "error": "31"}


def colorize(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}{RESET}" if code else text


def shorten(text: Any, width: int = 100) -> str:
    s = str(text if text is not None else "").replace("\n", " ⏎ ")
    return s if len(s) <= width else s[: max(1, width - 1)] + "…"


def _compact(v: Any, width: int = 80) -> str:
    try:
        s = json.dumps(v, default=str, ensure_ascii=False)
    except Exception:
        s = repr(v)
    return shorten(s, width)


def format_message(payload: dict, *, color: bool = True) -> str:
    sender = str(payload.get("sender") or "?")
    mtype = str(payload.get("message_type") or "chat")
    text = f"[{sender.upper()}] {payload.get('text', '')}"
    if not color:
        return text
    codes = ";".join(
        c for c in (SENDER_COLORS.get(sender, ""), TYPE_COLORS.get(mtype, "")) if c
    )
    return colorize(text, codes)


def format_event(ev: Event, *, color: bool = True, verbose: bool = False) -> str:
    """One-line (errors: multi-line when ``verbose``) rendering of ``ev``."""
    c = colorize if color else (lambda s, code: s)
    p = ev.payload if isinstance(ev.payload, dict) else {}
    k = ev.kind
    if k == "message":
        return format_message(p, color=color)
    if k == "error":
        head = c(
            f"[ERROR] {p.get('phase', '?')} <{p.get('component') or ev.source or '?'}>: {p.get('error', '')}",
            "31",
        )
        tb = p.get("traceback")
        return head + ("\n" + c(str(tb).rstrip(), "90") if verbose and tb else "")
    if k in ("turn.start", "turn.end"):
        label = "start" if k == "turn.start" else "end"
        tid = ev.turn_id or p.get("turn_id") or "?"
        extra = payload_get(p, "input", "text", "value")
        tail = f" {shorten(extra, 60)!r}" if extra and k == "turn.start" else ""
        return c(f"── turn {tid} {label}{tail} ──", "90")
    if k.startswith("llm."):
        name = payload_get(p, "name", "module", "signature", "component") or ev.source or "?"
        model = payload_get(p, "model", "lm")
        bits = [f"llm.{k[4:]}", str(name)]
        if model:
            bits.append(f"model={model}")
        lat = payload_get(p, "latency", "duration")
        if lat is not None:
            try:
                bits.append(f"{float(lat):.1f}s")
            except (TypeError, ValueError):
                pass
        from .usage import normalize_usage

        u = normalize_usage(p.get("usage"))
        if u:
            bits.append(f"{u['total_tokens']} tok" + ("~est" if u["estimated"] else ""))
        if p.get("cost") is not None:
            bits.append(f"${p['cost']}")
        if p.get("error"):
            bits.append(f"error={shorten(p['error'], 80)}")
        return c("· " + " ".join(bits), "31" if k == "llm.error" else "35")
    if k.startswith("tool."):
        name = payload_get(p, "name", "tool") or "?"
        bits = [k, str(name)]
        if k == "tool.start" and "args" in p:
            bits.append(_compact(p["args"], 60))
        if "result" in p:
            bits.append("→ " + _compact(p["result"], 60))
        if p.get("error"):
            bits.append(f"error={shorten(p['error'], 80)}")
        return c("· " + " ".join(bits), "31" if p.get("error") else "34")
    if k.startswith("trace."):
        names = p.get("rendered") or []
        return c(f"· {k} root={p.get('root')} rendered={len(names)} reasons={p.get('reasons', [])}", "90")
    return c(f"· {k} {_compact(p)}", "90")


class ConsoleSink(Sink):
    """Prints events to a stream. ``kinds`` selects what is shown
    (default: chat messages and errors); ``exclude`` hides kinds."""

    DEFAULT_KINDS = ("message", "error")

    def __init__(
        self,
        kinds: Optional[Iterable[str]] = DEFAULT_KINDS,
        *,
        exclude: Optional[Iterable[str]] = None,
        stream: Optional[TextIO] = None,
        color: Optional[bool] = None,
        verbose: bool = False,
        timestamps: bool = False,
    ) -> None:
        super().__init__(kinds)
        self.exclude = tuple(exclude) if exclude else None
        self._stream = stream
        self._color = color
        self.verbose = verbose
        self.timestamps = timestamps
        self._lock = threading.Lock()

    @property
    def stream(self) -> TextIO:
        return self._stream or sys.stdout

    @property
    def color(self) -> bool:
        if self._color is not None:
            return self._color
        try:
            return self.stream.isatty()
        except Exception:
            return False

    def accepts(self, ev: Event) -> bool:
        if self.exclude and _matches(ev.kind, self.exclude):
            return False
        return super().accepts(ev)

    def format(self, ev: Event) -> str:
        line = format_event(ev, color=self.color, verbose=self.verbose)
        if self.timestamps:
            import time

            line = f"{time.strftime('%H:%M:%S', time.localtime(ev.ts))} {line}"
        return line

    def handle(self, ev: Event) -> None:
        text = self.format(ev)
        with self._lock:
            try:
                self.stream.write(text + "\n")
                self.stream.flush()
            except Exception:
                pass
