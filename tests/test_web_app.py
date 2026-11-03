"""make_web_app: the ASGI app run_web serves, testable in-process."""

from starlette.testclient import TestClient

from pyreact.boot import make_web_app, run_web
from pyreact.core import EventBus, component, hooks, use_on_turn


@component
def Echo():
    events = hooks.get_service("event_bus", EventBus)
    use_on_turn(lambda t: events.publish("message", {"text": f"echo:{t.text}",
                                                     "sender": "assistant"}))
    return []


def test_make_web_app_returns_asgi_app_and_runner():
    app, runner = make_web_app(Echo, capture_stdout=False, session_ttl=None)
    assert app.state.runner is runner and runner.is_running
    with TestClient(app) as client:
        assert "<html" in client.get("/").text.lower()
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"t": "hello", "session": None, "path": "/"})
            sid = ws.receive_json()["data"]["id"]
            assert [s.id for s in runner.sessions()] == [sid]  # no default session
            ws.send_json({"t": "submit", "v": "hi"})
            while True:
                ev = ws.receive_json().get("event") or {}
                if ev.get("kind") == "message":
                    assert ev["payload"]["text"] == "echo:hi"
                    break
    assert not runner.is_running  # the lifespan stopped it


def test_run_web_serves_make_web_app(monkeypatch):
    import uvicorn

    served = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(app=app, **kw))
    run_web(Echo, port=8123, session_ttl=5)
    app = served["app"]
    assert served["port"] == 8123 and app.state.runner.is_running
    app.state.runner.shutdown()
