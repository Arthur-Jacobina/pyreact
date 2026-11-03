def make_web_app(
    app_component_fn,
    *,
    session_ttl=900.0,
    turn_policy: str = "queue",
    fps: int = 20,
    capture_stdout: bool = True,
    **server_opts,
):
    """Build the web app for ``app_component_fn`` without serving it.

    Returns ``(asgi_app, runner)``: a FastAPI app (one session per WebSocket
    / browser tab, see :mod:`pyreact.web.sessions`) and the
    :class:`~pyreact.boot.app_runner.AppRunner` hosting the sessions (no
    default session). Serve ``asgi_app`` with any ASGI server, or test it
    in-process::

        from starlette.testclient import TestClient

        app, runner = make_web_app(App, capture_stdout=False)
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            ws.send_json({"t": "hello", "session": None, "path": "/"})
            ws.send_json({"t": "submit", "v": "hi"})
            ...

    The app's lifespan shuts the runner down when the server stops; if you
    never run the app, call ``runner.shutdown()`` yourself.

    ``session_ttl``: seconds a disconnected tab's session is kept for resume
    (``None``/``0``: never reaped). ``turn_policy`` / ``fps`` go to the
    runner; other keyword arguments (``reap_interval``, ``event_kinds``) to
    :func:`pyreact.web.server.create_fastapi_app`.
    """
    from pyreact.boot.app_runner import AppRunner
    from pyreact.web.server import create_fastapi_app

    runner = AppRunner(
        app_component_fn, fps=fps, turn_policy=turn_policy, default_session=False
    )
    try:
        asgi_app, _ = create_fastapi_app(
            runner, session_ttl=session_ttl, capture_stdout=capture_stdout, **server_opts
        )
    except BaseException:
        runner.shutdown()
        raise
    asgi_app.state.runner = runner
    return asgi_app, runner


def run_web(
    app_component_fn,
    *,
    host="127.0.0.1",
    port=8000,
    reload=False,
    session_ttl=900.0,
    **uvicorn_kwargs,
):
    """Serve ``app_component_fn`` over HTTP/WebSocket; one session per tab.

    ``session_ttl``: seconds a disconnected tab's session is kept for resume.
    Same app as :func:`make_web_app` (use that to test or mount it yourself).
    """
    import uvicorn

    asgi_app, _runner = make_web_app(app_component_fn, session_ttl=session_ttl)
    uvicorn.run(asgi_app, host=host, port=port, reload=reload, **uvicorn_kwargs)
