from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager


from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from urllib.parse import parse_qs
from pyreact.boot.app_runner import AppRunner
from pyreact.core.hook import HookContext
from .console import enable_web_print, disable_web_print
from .broadcast import InMemoryBroadcast
from .ansi import ansi_to_html
from .ws_endpoint import (
    register_ws_routes,
    ChannelName,
)
from .input_consumer import InputConsumer
from .templates import BASE_HTML


def create_fastapi_app(runner: AppRunner):
    """Create the FastAPI app with lifecycle via ``lifespan``.

    When ``app_runner`` is provided, the server bridges HTML/nav updates from the
    runner and forwards WebSocket inputs via runner.invoke/nav/print helpers.
    """
    app = FastAPI()
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server_loop = asyncio.get_running_loop()
        broadcast = InMemoryBroadcast()
        enable_web_print()

        async def _broadcast_stdout(text: str) -> None:
            if text.startswith("__MESSAGE__:"):
                message_json = text[12:]  # Remove "__MESSAGE__:"
                message_data = json.loads(message_json)

                await broadcast.publish(
                    ChannelName.MSG,
                    {"channel": "chat", "type": "message", "data": message_data},
                )
                return

            await broadcast.publish(
                ChannelName.STDOUT,
                {"channel": "logs", "type": "stdout", "data": ansi_to_html(text)},
            )

        async def publish_nav(path: str) -> None:
            await broadcast.publish(
                ChannelName.NAV, {"channel": "ui", "type": "nav", "data": path}
            )

        async def publish_console(text: str) -> None:
            await _broadcast_stdout(text)

        runner.attach_web_bridge(
            on_nav=publish_nav,
            on_console=publish_console,
            target_loop=server_loop,
        )

        async def _handle_input_message(msg: dict) -> None:
            t = msg.get("t")

            if t in ("hello", "nav"):
                path = msg.get("path", "/")
                raw_query = msg.get("query", "") or ""
                raw_fragment = msg.get("fragment", "") or ""
                query = {}

                if len(raw_query) > 0:
                    q = raw_query[1:] if raw_query.startswith("?") else raw_query
                    parsed_q = parse_qs(q)

                    query = {
                        k: (v[0] if isinstance(v, list) and v else "")
                        for k, v in parsed_q.items()
                    }

                fragment = (
                    raw_fragment[1:] if raw_fragment.startswith("#") else raw_fragment
                )

                runner.nav(path, query=query, fragment=fragment)
                return

            if t == "submit":
                value = msg.get("v", "")
                await broadcast.publish(
                    ChannelName.MSG,
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
                runner.invoke(value, wait=True)
                return

            if t == "debug":
                what = msg.get("what")
                if what == "tree":
                    runner.print_vnode_tree()
                elif what == "trace":
                    runner.print_render_trace()

        # Register WebSocket routes
        register_ws_routes(
            app,
            broadcast=broadcast,
            channels_to_forward=[ChannelName.NAV, ChannelName.STDOUT, ChannelName.MSG],
            input_channel=ChannelName.INPUT,
        )

        input_consumer = InputConsumer(
            broadcast=broadcast,
            input_channel=ChannelName.INPUT,
            handle_message=_handle_input_message,
        )

        input_task = asyncio.create_task(input_consumer.run())

        try:
            yield
        finally:
            runner.shutdown()
            input_task.cancel()
            HookContext._services.clear()
            disable_web_print()

    app.router.lifespan_context = lifespan

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204, media_type="image/x-icon")

    @app.get("/{full_path:path}")
    async def index(_request: Request):
        return HTMLResponse(BASE_HTML)

    return app, None
