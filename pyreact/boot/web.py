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
    """
    import uvicorn
    from pyreact.boot.bootstrap import bootstrap
    from pyreact.web.server import create_fastapi_app

    app = bootstrap(app_component_fn, default_session=False)
    fastapi_app, _ = create_fastapi_app(app, session_ttl=session_ttl)
    uvicorn.run(fastapi_app, host=host, port=port, reload=reload, **uvicorn_kwargs)
