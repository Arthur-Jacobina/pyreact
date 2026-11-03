"""Per-session WebSocket transport.

Every WebSocket connection is attached to one :class:`~pyreact.boot.session.Session`
of the shared :class:`~pyreact.boot.app_runner.AppRunner`:

- The client sends ``{"t": "hello", "session": <id or null>, "path", "query",
  "fragment"}`` first. A known, open id is resumed; otherwise a session is
  opened (with the given id if it is well-formed, else a fresh one). The
  server answers ``{"channel": "session", "type": "session", "data": {"id",
  "resumed"}}`` and the client stores the id (``sessionStorage``: per tab,
  survives reloads).
- Inputs (``submit``/``nav``/``debug``) go to that session only.
- Outputs are forwarded only to the sockets attached to the session:
    * session EventBus events of kinds :data:`FORWARDED_EVENT_KINDS` as
      ``{"channel": "events", "event": ev.to_dict()}``;
    * legacy channels: ``ui/nav``, ``chat/message`` (``__MESSAGE__:`` lines)
      and ``logs/stdout``. Stdout is attributed to a session when written
      from its render/effects/tasks; unattributable output (server logs) goes
      to every socket.
- Sessions with no socket attached are closed after ``ttl`` seconds idle.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Callable, Dict, Optional, Set
from urllib.parse import parse_qs

from pyreact.boot.app_runner import AppRunner
from pyreact.boot.session import Session, current_session
from pyreact.core.events import Event

from .ansi import ansi_to_html

FORWARDED_EVENT_KINDS = ("message", "turn.*", "llm.*", "tool.*", "error", "nav")

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def _dumps(msg: dict) -> str:
    return json.dumps(msg, default=str)


class Connection:
    """One WebSocket: an outbox drained by a sender task."""

    def __init__(self, send_text: Callable, *, maxsize: int = 1024) -> None:
        self._send_text = send_text
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.session: Optional[Session] = None
        self.dropped = 0

    def put(self, msg: dict) -> None:
        try:
            self.queue.put_nowait(_dumps(msg))
        except asyncio.QueueFull:
            self.dropped += 1  # slow client: drop rather than block the app

    async def pump(self) -> None:
        while True:
            text = await self.queue.get()
            try:
                await self._send_text(text)
            except Exception:
                return


class SessionHub:
    """Routes traffic between WebSocket connections and runner sessions.

    All methods except :meth:`publish_threadsafe` and :meth:`on_console` must
    be called on the server's event loop.
    """

    def __init__(
        self,
        runner: AppRunner,
        *,
        loop: asyncio.AbstractEventLoop,
        ttl: Optional[float] = 900.0,
        reap_interval: Optional[float] = None,
        event_kinds=FORWARDED_EVENT_KINDS,
    ) -> None:
        self.runner = runner
        self.loop = loop
        self.ttl = ttl
        self.reap_interval = reap_interval or (
            max(0.05, min(30.0, ttl / 2)) if ttl else 30.0
        )
        self.event_kinds = tuple(event_kinds)
        self._conns: Dict[str, Set[Connection]] = {}
        self._all: Set[Connection] = set()
        self._unsubs: Dict[str, Callable[[], None]] = {}

    # ------------------------------------------------------------ routing
    def publish(self, session_id: Optional[str], msg: dict) -> None:
        """Send ``msg`` to a session's sockets (``None`` → every socket)."""
        targets = self._all if session_id is None else self._conns.get(session_id, ())
        for conn in list(targets):
            conn.put(msg)

    def publish_threadsafe(self, session_id: Optional[str], msg: dict) -> None:
        try:
            self.loop.call_soon_threadsafe(self.publish, session_id, msg)
        except RuntimeError:
            pass  # server loop closed

    def connected(self, session_id: str) -> bool:
        return bool(self._conns.get(session_id))

    def on_console(self, text: str) -> None:
        """MessageBuffer listener (any thread): forward stdout / legacy chat."""
        sess = current_session()
        sid = sess.id if sess is not None else None
        if text.startswith("__MESSAGE__:"):
            raw = text[len("__MESSAGE__:") :]
            try:
                data = json.loads(raw)
            except ValueError:
                data = {
                    "type": "message",
                    "text": raw,
                    "sender": "system",
                    "message_type": "error",
                }
            msg = {"channel": "chat", "type": "message", "data": data}
        else:
            msg = {"channel": "logs", "type": "stdout", "data": ansi_to_html(text)}
        self.publish_threadsafe(sid, msg)

    def _forward_event(self, session_id: str, ev: Event) -> None:
        # Runs on whichever thread published the event.
        self.publish_threadsafe(session_id, {"channel": "events", "event": ev.to_dict()})
        if ev.kind == "nav":
            path = (ev.payload or {}).get("path", "/")
            self.publish_threadsafe(
                session_id, {"channel": "ui", "type": "nav", "data": path}
            )

    # -------------------------------------------------------- attach/detach
    def _normalize_id(self, requested) -> Optional[str]:
        if (
            isinstance(requested, str)
            and requested != self.runner.default_session_id
            and _SESSION_ID_RE.match(requested)
        ):
            return requested
        return None

    async def attach(self, conn: Connection, requested_id=None) -> Session:
        sid = self._normalize_id(requested_id)
        existing = self.runner.get_session(sid) if sid else None
        resumed = existing is not None and not existing.closed
        session = await self.runner.aopen_session(sid)
        conn.session = session
        self._conns.setdefault(session.id, set()).add(conn)
        self._all.add(conn)
        if session.id not in self._unsubs:
            sid_ = session.id
            self._unsubs[sid_] = session.event_bus.subscribe(
                lambda ev: self._forward_event(sid_, ev), self.event_kinds
            )
        conn.put(
            {
                "channel": "session",
                "type": "session",
                "data": {"id": session.id, "resumed": resumed},
            }
        )
        return session

    def detach(self, conn: Connection) -> None:
        self._all.discard(conn)
        sess = conn.session
        if sess is None:
            return
        conns = self._conns.get(sess.id)
        if conns is not None:
            conns.discard(conn)
            if not conns:
                self._conns.pop(sess.id, None)
        sess.touch()  # TTL counts from the last disconnect

    def _forget(self, session_id: str) -> None:
        unsub = self._unsubs.pop(session_id, None)
        if unsub is not None:
            unsub()

    async def reap_once(self) -> list:
        """Close sessions idle > ttl with no socket attached."""
        if not self.ttl:
            return []
        closed = await self.runner.aclose_idle_sessions(
            self.ttl, keep=lambda s: self.connected(s.id)
        )
        for sid in closed:
            self._forget(sid)
        # Also drop forwarders of sessions closed by other means.
        for sid in list(self._unsubs):
            sess = self.runner.get_session(sid)
            if sess is None or sess.closed:
                self._forget(sid)
        return closed

    async def reap_forever(self) -> None:
        while True:
            await asyncio.sleep(self.reap_interval)
            try:
                await self.reap_once()
            except Exception:
                pass

    def close(self) -> None:
        for sid in list(self._unsubs):
            self._forget(sid)

    # ---------------------------------------------------------- inputs
    async def handle(self, conn: Connection, msg: dict) -> None:
        sess = conn.session
        if sess is None or sess.closed:
            return
        t = msg.get("t")

        if t in ("hello", "nav"):
            path = msg.get("path", "/") or "/"
            raw_query = msg.get("query", "") or ""
            raw_fragment = msg.get("fragment", "") or ""
            query = {}
            if raw_query:
                q = raw_query[1:] if raw_query.startswith("?") else raw_query
                query = {k: (v[0] if v else "") for k, v in parse_qs(q).items()}
            fragment = raw_fragment[1:] if raw_fragment.startswith("#") else raw_fragment
            await sess.anav(path, query=query, fragment=fragment)
            return

        if t == "submit":
            value = msg.get("v", "")
            if not isinstance(value, str):
                return
            self.publish(
                sess.id,
                {
                    "channel": "chat",
                    "type": "message",
                    "data": {
                        "type": "message",
                        "text": value,
                        "sender": "user",
                        "message_type": "chat",
                        "timestamp": time.time(),
                    },
                },
            )
            # Never block the server's event loop waiting on the app.
            await sess.ainvoke(value, source="web")
            return

        if t == "debug":
            what = msg.get("what")
            if what == "tree":
                await asyncio.get_running_loop().run_in_executor(
                    None, sess.print_vnode_tree
                )
            elif what == "trace":
                await asyncio.get_running_loop().run_in_executor(
                    None, self.runner.print_render_trace
                )
            return

        if t == "ping":
            sess.touch()
