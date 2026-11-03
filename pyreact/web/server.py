from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pyreact.boot.app_runner import AppRunner
from pyreact.core.hook import HookContext
from pyreact.core.message_buffer import MessageBuffer

from .console import disable_web_print, enable_web_print
from .sessions import FORWARDED_EVENT_KINDS, SessionHub
from .templates import BASE_HTML
from .ws_endpoint import register_session_ws_route


def create_fastapi_app(
    runner: AppRunner,
    *,
    session_ttl: Optional[float] = 900.0,
    reap_interval: Optional[float] = None,
    event_kinds=FORWARDED_EVENT_KINDS,
    capture_stdout: bool = True,
):
    """Create the FastAPI app serving ``runner`` over a per-session WebSocket.

    Each WebSocket connection gets (or resumes) its own session on ``runner``
    (see :mod:`pyreact.web.sessions`); inputs and outputs never cross sessions.

    Args:
        runner: the AppRunner hosting the sessions. Its default session (if
            any) is not exposed over the web.
        session_ttl: seconds a session with no socket attached is kept before
            being closed (``None``/``0`` disables reaping).
        reap_interval: how often to check for idle sessions (default
            ``min(30, ttl/2)``).
        event_kinds: EventBus kinds forwarded to the browser.
        capture_stdout: redirect stdout/stderr into the web log channel.

    Returns ``(app, None)`` for backward compatibility; the hub is available as
    ``app.state.session_hub`` while the app is running.
    """
    app = FastAPI()
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub = SessionHub(
            runner,
            loop=asyncio.get_running_loop(),
            ttl=session_ttl,
            reap_interval=reap_interval,
            event_kinds=event_kinds,
        )
        app.state.session_hub = hub
        if capture_stdout:
            enable_web_print()
        console = HookContext.get_service("console_buffer", MessageBuffer)
        console.subscribe(hub.on_console)
        reaper = asyncio.create_task(hub.reap_forever())
        try:
            yield
        finally:
            reaper.cancel()
            console.unsubscribe(hub.on_console)
            hub.close()
            runner.shutdown()
            if capture_stdout:
                disable_web_print()

    app.router.lifespan_context = lifespan

    class _HubProxy:
        """Resolve the hub lazily (it is created in the lifespan)."""

        def __getattr__(self, name):
            return getattr(app.state.session_hub, name)

    register_session_ws_route(app, _HubProxy())

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204, media_type="image/x-icon")

    @app.get("/{full_path:path}")
    async def index(_request: Request):
        return HTMLResponse(BASE_HTML)

    return app, None
