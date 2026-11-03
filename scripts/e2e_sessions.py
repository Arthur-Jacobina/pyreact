"""End-to-end check of per-session web transport (no LLM needed).

Starts a real uvicorn server on a free port, connects two WebSocket clients,
and verifies that inputs/outputs never cross sessions and that a session can
be resumed by id.

    uv run python scripts/e2e_sessions.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402
import websockets  # noqa: E402

from pyreact.boot import AppRunner  # noqa: E402
from pyreact.core import EventBus, component, hooks  # noqa: E402
from pyreact.input.bus import InputBus  # noqa: E402
from pyreact.web.server import create_fastapi_app  # noqa: E402


@component
def EchoApp():
    bus = hooks.get_service("input_bus", InputBus)
    events = hooks.get_service("event_bus", EventBus)
    count, set_count = hooks.use_state(0)

    def on_event(ev):
        if ev.get("type") != "submit":
            return
        set_count(lambda c: c + 1)
        events.publish(
            "message",
            {"text": f"echo:{ev['value']}", "sender": "assistant", "message_type": "chat"},
        )

    hooks.use_effect(lambda: bus.subscribe(on_event), [])
    hooks.use_effect(
        lambda: count and events.publish("turn.end", {"count": count}), [count]
    )
    return []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _recv_until(ws, pred, timeout=5.0):
    got = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = json.loads(await asyncio.wait_for(ws.recv(), deadline - time.time()))
        got.append(msg)
        if pred(msg):
            return got
    raise AssertionError(f"timeout; got {got}")


def _event(kind, **payload):
    def pred(m):
        ev = m.get("event") or {}
        return ev.get("kind") == kind and all(
            ev.get("payload", {}).get(k) == v for k, v in payload.items()
        )

    return pred


def _is_session(m):
    return m.get("channel") == "session"


async def _client_checks(url: str) -> None:
    async with websockets.connect(url) as a, websockets.connect(url) as b:
        await a.send(json.dumps({"t": "hello", "session": None, "path": "/"}))
        await b.send(json.dumps({"t": "hello", "session": None, "path": "/"}))
        ida = (await _recv_until(a, _is_session))[-1]["data"]["id"]
        idb = (await _recv_until(b, _is_session))[-1]["data"]["id"]
        assert ida != idb, "sockets must get distinct sessions"

        await a.send(json.dumps({"t": "submit", "v": "from-a"}))
        got_a = await _recv_until(a, _event("turn.end", count=1))
        assert any(_event("message", text="echo:from-a")(m) for m in got_a)

        await b.send(json.dumps({"t": "submit", "v": "from-b"}))
        got_b = await _recv_until(b, _event("turn.end", count=1))
        assert "from-a" not in json.dumps(got_b), "session A leaked into B"
        print(f"ok: isolated sessions {ida[:8]} / {idb[:8]}")

    async with websockets.connect(url) as c:
        await c.send(json.dumps({"t": "hello", "session": ida, "path": "/"}))
        info = (await _recv_until(c, _is_session))[-1]["data"]
        assert info == {"id": ida, "resumed": True}, info
        await c.send(json.dumps({"t": "submit", "v": "again"}))
        await _recv_until(c, _event("turn.end", count=2))
        print("ok: resumed session kept its state")


def main() -> int:
    port = _free_port()
    runner = AppRunner(EchoApp, default_session=False)
    app, _ = create_fastapi_app(runner, capture_stdout=False)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    try:
        asyncio.run(_client_checks(f"ws://127.0.0.1:{port}/ws"))
    finally:
        server.should_exit = True
        th.join(timeout=10)
    print("e2e sessions: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
