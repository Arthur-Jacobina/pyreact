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
