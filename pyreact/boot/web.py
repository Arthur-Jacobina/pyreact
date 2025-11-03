def run_web(
    app_component_fn, *, host="127.0.0.1", port=8000, reload=False, **uvicorn_kwargs
):
    import uvicorn
    from pyreact.boot.bootstrap import bootstrap
    from pyreact.web.server import create_fastapi_app

    app = bootstrap(app_component_fn)
    fastapi_app, _ = create_fastapi_app(app)
    uvicorn.run(fastapi_app, host=host, port=port, reload=reload, **uvicorn_kwargs)
