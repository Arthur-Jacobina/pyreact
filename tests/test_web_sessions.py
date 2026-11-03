import time

import pytest

from pyreact.boot import AppRunner, bootstrap
from pyreact.core import EventBus, component, hooks
from pyreact.input.bus import InputBus


def _on_submit(fn):
    bus = hooks.get_service("input_bus", InputBus)
    hooks.use_effect(
        lambda: bus.subscribe(lambda e: e["type"] == "submit" and fn(e["value"])), []
    )


@pytest.fixture
def runners():
    made = []

    def _make(fn, **kw):
        r = bootstrap(fn, **kw)
        made.append(r)
        return r

    yield _make
    for r in made:
        r.shutdown()


# ---------------------------------------------------------------- web
def _web_app(**kw):
    from pyreact.web.server import create_fastapi_app

    @component
    def App():
        events = hooks.get_service("event_bus", EventBus)
        count, set_count = hooks.use_state(0)

        def on_submit(v):
            set_count(lambda c: c + 1)
            print(f"log:{v}")
            events.publish(
                "message",
                {"text": f"echo:{v}", "sender": "assistant", "message_type": "chat"},
            )

        _on_submit(on_submit)
        hooks.use_effect(
            lambda: count and events.publish("turn.end", {"count": count}), [count]
        )
        return []

    runner = AppRunner(App, default_session=False)
    app, _ = create_fastapi_app(runner, **kw)
    return app, runner


def _recv_until(ws, pred, limit=200):
    got = []
    for _ in range(limit):
        msg = ws.receive_json()
        got.append(msg)
        if pred(msg):
            return got
    raise AssertionError(f"predicate never matched: {got}")


def _is_event(kind, **payload):
    def pred(m):
        ev = m.get("event") or {}
        return (
            m.get("channel") == "events"
            and ev.get("kind") == kind
            and all(ev["payload"].get(k) == v for k, v in payload.items())
        )

    return pred


def test_web_sockets_are_isolated_per_session():
    from starlette.testclient import TestClient

    app, runner = _web_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws1, client.websocket_connect(
            "/ws"
        ) as ws2:
            ws1.send_json({"t": "hello", "session": None, "path": "/"})
            ws2.send_json({"t": "hello", "session": None, "path": "/"})
            s1 = _recv_until(ws1, lambda m: m.get("channel") == "session")[-1]
            s2 = _recv_until(ws2, lambda m: m.get("channel") == "session")[-1]
            id1, id2 = s1["data"]["id"], s2["data"]["id"]
            assert id1 != id2 and not s1["data"]["resumed"]

            ws1.send_json({"t": "submit", "v": "one"})
            got1 = _recv_until(ws1, _is_event("turn.end", count=1))
            assert any(_is_event("message", text="echo:one")(m) for m in got1)
            assert any(
                m.get("channel") == "logs" and "log:one" in m["data"] for m in got1
            )

            ws2.send_json({"t": "submit", "v": "two"})
            got2 = _recv_until(ws2, _is_event("turn.end", count=1))
            assert any(_is_event("message", text="echo:two")(m) for m in got2)
            assert "echo:one" not in repr(got2) and "log:one" not in repr(got2)

        # Resume session 1 from a new socket: state survived the disconnect.
        with client.websocket_connect("/ws") as ws3:
            ws3.send_json({"t": "hello", "session": id1, "path": "/"})
            s3 = _recv_until(ws3, lambda m: m.get("channel") == "session")[-1]
            assert s3["data"] == {"id": id1, "resumed": True}
            ws3.send_json({"t": "submit", "v": "three"})
            _recv_until(ws3, _is_event("turn.end", count=2))
    assert not runner.is_running


def test_web_idle_sessions_are_reaped():
    from starlette.testclient import TestClient

    app, runner = _web_app(session_ttl=0.2, reap_interval=0.05)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"t": "hello", "session": None, "path": "/"})
            sid = _recv_until(ws, lambda m: m.get("channel") == "session")[-1][
                "data"
            ]["id"]
            time.sleep(0.4)  # connected sessions are never reaped
            assert runner.get_session(sid) is not None
        deadline = time.time() + 5
        while runner.get_session(sid) is not None and time.time() < deadline:
            time.sleep(0.05)
        assert runner.get_session(sid) is None


