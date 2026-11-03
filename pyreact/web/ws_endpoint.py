from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Iterable

from fastapi import FastAPI, WebSocket
from starlette.endpoints import WebSocketEndpoint
from .broadcast import InMemoryBroadcast


class ChannelName(StrEnum):
    NAV = "nav"
    STDOUT = "stdout"
    MSG = "message"
    INPUT = "input"


def register_ws_routes(
    app: FastAPI,
    *,
    broadcast: InMemoryBroadcast,
    channels_to_forward: Iterable[str],
    input_channel: str,
) -> None:
    """
    Register a WebSocket route that bridges pub/sub channels to the socket.
    - broadcast: object with publish(channel, message) and subscribe(channel) async context manager
    - channels_to_forward: channel names to forward from pub/sub to this socket
    - input_channel: channel name to publish all incoming client messages
    """

    class AppWS(WebSocketEndpoint):
        encoding = "text"

        async def on_connect(self, ws: WebSocket):
            await ws.accept()

            self._forward_tasks = [
                asyncio.create_task(self._forward(ws, channel))
                for channel in channels_to_forward
            ]

        async def on_receive(self, ws: WebSocket, data: str):
            try:
                await broadcast.publish(input_channel, json.loads(data))
            except Exception:
                pass

        async def on_disconnect(self, ws: WebSocket, close_code: int):
            for t in getattr(self, "_forward_tasks", []):
                try:
                    t.cancel()
                except Exception:
                    pass

        async def _forward(self, ws: WebSocket, channel: str):
            try:
                async with broadcast.subscribe(channel) as subscriber:
                    async for event in subscriber:
                        try:
                            await ws.send_text(event.message)
                        except Exception:
                            break
            except Exception:
                return

    app.add_websocket_route("/ws", AppWS)


def register_session_ws_route(app: FastAPI, hub, *, path: str = "/ws") -> None:
    """Register the per-session WebSocket route (see :mod:`pyreact.web.sessions`).

    The first client message should be ``{"t": "hello", "session": id|null,
    "last_event": id|null, ...}``; any other first message attaches a fresh
    session and is then handled normally.
    """
    from starlette.websockets import WebSocketDisconnect

    from .sessions import Connection

    async def _recv_json(ws: WebSocket) -> dict:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if isinstance(msg, dict):
                return msg

    async def endpoint(ws: WebSocket) -> None:
        await ws.accept()
        conn = Connection(ws.send_text)
        sender = asyncio.create_task(conn.pump())
        try:
            first = await _recv_json(ws)
            hello = first.get("t") == "hello"
            await hub.attach(
                conn,
                first.get("session") if hello else None,
                last_event=first.get("last_event") if hello else None,
            )
            await hub.handle(conn, first)
            while True:
                msg = await _recv_json(ws)
                try:
                    await hub.handle(conn, msg)
                except Exception:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            hub.detach(conn)
            sender.cancel()

    app.add_api_websocket_route(path, endpoint)
