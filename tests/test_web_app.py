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


# ------------------------------------------------- replay on attach
@component
def Welcome():
    from pyreact.components.message import Message

    events = hooks.get_service("event_bus", EventBus)
    use_on_turn(lambda t: events.publish("message", {"text": f"echo:{t.text}",
                                                     "sender": "assistant"}))
    return [Message(key="hi", text="welcome!", sender="assistant", trigger="mount")]


def _messages_until(ws, text):
    """(id, text) of message events up to the one whose text is ``text``."""
    got = []
    while True:
        m = ws.receive_json()
        ev = m.get("event") or {}
        if ev.get("kind") == "message":
            got.append((ev["id"], ev["payload"]["text"]))
            if ev["payload"]["text"] == text:
                return got


def _hello(ws, sid=None, last_event=None):
    ws.send_json({"t": "hello", "session": sid, "path": "/", "last_event": last_event})
    return ws.receive_json()["data"]


def test_mount_time_messages_reach_the_first_socket():
    app, _runner = make_web_app(Welcome, capture_stdout=False, session_ttl=None)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello(ws)
        ws.send_json({"t": "submit", "v": "a"})
        # (read up to the echo so a regression fails instead of blocking)
        assert [t for _i, t in _messages_until(ws, "echo:a")] == ["welcome!", "echo:a"]


def test_reconnect_replays_only_missed_events_and_reload_replays_all():
    app, _runner = make_web_app(Welcome, capture_stdout=False, session_ttl=None)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            sid = _hello(ws)["id"]
            ws.send_json({"t": "submit", "v": "a"})
            seen = _messages_until(ws, "echo:a")
        last = seen[-1][0]
        with client.websocket_connect("/ws") as ws:  # reconnect: nothing missed
            assert _hello(ws, sid, last_event=last)["resumed"] is True
            ws.send_json({"t": "submit", "v": "b"})
            assert [t for _i, t in _messages_until(ws, "echo:b")] == ["echo:b"]
        with client.websocket_connect("/ws") as ws:  # reload: page is empty again
            _hello(ws, sid)
            ws.send_json({"t": "submit", "v": "c"})
            assert [t for _i, t in _messages_until(ws, "echo:c")] == [
                "welcome!", "echo:a", "echo:b", "echo:c"]
