"""Session lifecycle hooks (AppRunner.on_session) and the participant registry."""

import pytest

from pyreact.boot import AppRunner
from pyreact.boot.stateful import Participants
from pyreact.core import EventBus, component, hooks


@component
def App():
    return []


@pytest.fixture
def runner():
    made = []

    def _make(fn=App, **kw):
        r = AppRunner(fn, **kw)
        made.append(r)
        return r

    yield _make
    for r in made:
        r.shutdown()


def test_open_hook_runs_before_first_render_and_close_after_unmount(runner):
    log = []

    @component
    def Probe():
        sess = hooks.get_service("session", lambda: None)
        log.append(("render", sess.id, hooks.get_service("seed", lambda: None)))

        def eff():
            return lambda: log.append(("unmount", sess.id, sess.close_reason))

        hooks.use_effect(eff, [])
        return []

    app = runner(Probe, default_session=False)

    def opened(s):
        log.append(("open", s.id, s.root is None))
        s.services["seed"] = "seeded"  # visible to the first render

    def closed(s):
        log.append(("close", s.id, s.close_reason))

    unhook = app.on_session(open=opened, close=closed)
    s = app.open_session("a")
    assert log[:2] == [("open", "a", True), ("render", "a", "seeded")]
    app.close_session("a", reason="closed")
    assert log[-2:] == [("unmount", "a", "closed"), ("close", "a", "closed")]

    unhook()
    app.open_session("b")
    assert ("open", "b", True) not in log
    assert s.closed


def test_lifecycle_events_on_the_session_bus(runner):
    app = runner(default_session=False)
    kinds = []
    app.on_session(open=lambda s: s.event_bus.subscribe(
        lambda ev: kinds.append((ev.kind, ev.payload.get("session_id"))), ["session.*"]))
    app.open_session("x")
    app.close_session("x")
    assert kinds == [("session.opened", "x"), ("session.closed", "x")]


def test_replay_calls_open_for_sessions_already_open(runner):
    app = runner()  # default session opened in the constructor
    app.open_session("y")
    seen = []
    app.on_session(open=lambda s: seen.append(s.id), replay=True)
    assert sorted(seen) == ["default", "y"]


def test_failing_hook_is_reported_not_fatal(runner, capsys):
    app = runner(default_session=False)

    def boom(_s):
        raise RuntimeError("hook exploded")

    app.on_session(open=boom, close=boom)
    s = app.open_session("z")
    assert not s.closed
    assert app.close_session("z")
    assert "hook exploded" in capsys.readouterr().err


class _P:
    name = "p"

    def __init__(self, session=None):
        self.session = session

    async def suspend(self, session):
        return {}

    async def resume(self, session, fragment):
        return None


def test_participants_factories_are_added_to_every_session(runner):
    app = runner(participants=[_P, lambda: _NoArg()], default_session=True)
    sess = app.default_session
    assert sess.participants.names() == ["p", "noarg"]
    assert sess.participants.get("p").session is sess
    other = app.open_session("o")
    assert other.participants.get("p").session is other


class _NoArg(_P):
    name = "noarg"


def test_participants_registry_validates():
    ps = Participants()
    p = ps.add(_P())
    assert "p" in ps and len(ps) == 1 and list(ps) == [p]
    with pytest.raises(ValueError):
        ps.add(_P())
    ps.add(_P(), replace=True)
    with pytest.raises(TypeError):
        ps.add(object())

    class NoResume:
        name = "x"

        async def suspend(self, s):
            return {}

    with pytest.raises(TypeError):
        ps.add(NoResume())
    assert ps.remove("p") and not ps.remove("p")


def test_otel_attach_app_follows_sessions_opened_later(runner):
    pyreact_otel = pytest.importorskip("pyreact_otel")
    h = pyreact_otel.setup("memory", service_name="t", bridge_global_bus=False,
                           events=("message",))
    try:
        app = runner(default_session=False)
        h.attach_app(app)
        later = app.open_session("later")  # opened after attach_app
        later.event_bus.publish("message", {"text": "hi", "sender": "x"})
        app.close_session("later")
        later.event_bus.publish("message", {"text": "after close", "sender": "x"})
        h.force_flush()
        spans = h.span_exporter.get_finished_spans()
        kinds = [s.attributes.get("pyreact.event.kind") for s in spans]
        assert kinds.count("message") == 1
    finally:
        h.shutdown()
    # shutdown removed the hook: no error, nothing attached
    app.open_session("after")
    assert isinstance(app.get_session("after").event_bus, EventBus)
