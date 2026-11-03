import warnings

import pytest

from pyreact.boot import bootstrap
from pyreact.core import EventBus, component, hooks
from pyreact.input.bus import InputBus
from pyreact_durable import (
    Durable,
    DurableStateError,
    SessionLog,
    use_conversation,
    use_durable,
    use_persistent_reducer,
    use_persistent_state,
)
from pyreact_durable import react as durable_react


@pytest.fixture
def run_app():
    apps = []

    def _start(fn):
        app = bootstrap(fn)
        apps.append(app)
        return app

    yield _start
    for app in apps:
        app.shutdown()


def _on_submit(fn):
    bus = hooks.get_service("input_bus", InputBus)
    hooks.use_effect(lambda: bus.subscribe(lambda e: e["type"] == "submit" and fn(e["value"])),
                     [])


def _counter_app(log, seen, errors=None):
    @component
    def Counter():
        n, set_n = use_persistent_state(0, key="n")
        total, dispatch = use_persistent_reducer(
            lambda s, a: {"sum": s["sum"] + a}, None, key="total",
            init_fn=lambda _: {"sum": 0},
        )

        def handle(text):
            if text == "bad":
                try:
                    set_n(object())
                except DurableStateError as exc:
                    errors.append(str(exc))
                return
            set_n(lambda v: v + 1)
            dispatch(int(text))

        _on_submit(handle)
        seen.append((n, total))
        return []

    @component
    def App():
        return [Durable(key="durable", log=log, children=[Counter(key="counter")])]

    return App


def test_persistent_hooks_restore_across_two_runners(tmp_path, run_app):
    path = str(tmp_path / "app.jsonl")
    seen1, seen2 = [], []
    app = run_app(_counter_app(path, seen1))
    for v in ("2", "3", "5"):
        app.invoke(v, wait=True, timeout=5)
    assert seen1[-1] == (3, {"sum": 10})
    app.shutdown()  # unmount closes the log it opened (lock released)

    log = SessionLog.open(path, readonly=True)
    assert log.state_at() == {
        "App/Counter[counter]": {"n": 3, "total": {"sum": 10}}
    }
    kinds = [e["kind"] for e in log.events()]
    assert kinds.count("turn.start") == 3 and kinds.count("turn.end") == 3

    app2 = run_app(_counter_app(path, seen2))
    app2.wait_idle(timeout=5)
    assert seen2[0] == (3, {"sum": 10})  # restored on first render
    app2.invoke("1", wait=True, timeout=5)
    assert seen2[-1] == (4, {"sum": 11})


def test_json_error_names_path_and_key(tmp_path, run_app):
    errors, seen = [], []
    app = run_app(_counter_app(str(tmp_path / "e.jsonl"), seen, errors))
    app.invoke("bad", wait=True, timeout=5)
    assert len(errors) == 1
    assert "'n'" in errors[0] and "App/Counter[counter]" in errors[0]
    assert "JSON-serializable" in errors[0]


def test_no_provider_falls_back_to_use_state_and_warns_once(run_app):
    durable_react._WARNED.clear()
    seen = []

    @component
    def Lone():
        n, set_n = use_persistent_state(10, key="n")
        _on_submit(lambda _t: set_n(lambda v: v + 1))
        seen.append((n, use_durable()))
        return []

    @component
    def App():
        return [Lone(key="a"), Lone(key="b")]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        app = run_app(App)
        app.invoke("x", wait=True, timeout=5)
        app.invoke("y", wait=True, timeout=5)
    msgs = [str(x.message) for x in w if "Durable provider" in str(x.message)]
    assert len(msgs) == 1
    assert seen[-1] == (12, None)


def test_checkout_from_component_reloads_branch_state(tmp_path, run_app):
    path = str(tmp_path / "b.jsonl")
    seen = []

    @component
    def Chat():
        conv = use_conversation()
        log = use_durable()

        def handle(text):
            if text.startswith("/label "):
                log.label(text.split()[1])
            elif text.startswith("/fork "):
                log.checkout(text.split()[1])
            else:
                conv.extend([{"role": "user", "content": text},
                             {"role": "assistant", "content": text.upper()}])

        _on_submit(handle)
        seen.append([m["content"] for m in conv.messages])
        return []

    @component
    def App():
        return [Durable(key="d", log=path, children=[Chat(key="chat")])]

    app = run_app(App)
    for t in ("one", "/label base", "two", "/label main", "/fork base", "three"):
        app.invoke(t, wait=True, timeout=5)
    assert seen[-1] == ["one", "ONE", "three", "THREE"]
    app.invoke("/fork main", wait=True, timeout=5)
    assert seen[-1] == ["one", "ONE", "two", "TWO"]
    app.shutdown()

    with SessionLog.open(path, readonly=True) as log:
        # labels taken mid-turn mark that moment: later turn events branch off it
        assert log.branch_points()
        assert log.labels["main"] in {e["id"] for e in log.path_to()}
        hist = log.get_state("App/Chat[chat]", "history")
        assert [m["content"] for m in hist] == ["one", "ONE", "two", "TWO"]


def test_log_object_and_factory_per_session(tmp_path):
    from pyreact.boot import AppRunner

    seen = {}

    @component
    def Who():
        n, set_n = use_persistent_state(0, key="n")
        _on_submit(lambda _t: set_n(lambda v: v + 1))
        sess = hooks.get_service("session", lambda: None)
        seen[sess.id] = n
        return []

    @component
    def App():
        return [Durable(key="d", log_factory=lambda sid: tmp_path / f"{sid}.jsonl",
                        children=[Who(key="w")])]

    runner = AppRunner(App, default_session=False)
    try:
        a, b = runner.open_session("alice"), runner.open_session("bob")
        a.invoke("x", wait=True, timeout=5)
        a.invoke("x", wait=True, timeout=5)
        b.invoke("x", wait=True, timeout=5)
        assert seen == {"alice": 2, "bob": 1}
    finally:
        runner.shutdown()
    for sid, n in (("alice", 2), ("bob", 1)):
        with SessionLog.open(tmp_path / f"{sid}.jsonl", readonly=True) as log:
            assert log.get_state("App/Who[w]", "n") == n

    # a caller-owned SessionLog is not closed on unmount
    own = SessionLog.open(tmp_path / "own.jsonl")

    @component
    def App2():
        return [Durable(key="d", log=own, journal=None, children=[Who(key="w")])]

    r2 = AppRunner(App2)
    r2.invoke("x", wait=True, timeout=5)
    r2.shutdown()
    assert not own.closed and own.get_state("App2/Who[w]", "n") == 1
    assert not own.events()  # journal disabled
    own.close()


def test_journal_sanitizes_payloads(tmp_path, run_app):
    path = str(tmp_path / "j.jsonl")

    @component
    def Pub():
        bus = hooks.get_service("event_bus", EventBus)
        hooks.use_effect(lambda: bus.publish("tool.end", {"obj": object(), "n": 1}), [])
        return []

    @component
    def App():
        return [Durable(key="d", log=path, children=[Pub(key="p")])]

    app = run_app(App)
    app.wait_idle(timeout=5)
    app.shutdown()
    ev = SessionLog.open(path, readonly=True).events("tool.*")
    assert ev and ev[0]["payload"]["n"] == 1 and "object" in ev[0]["payload"]["obj"]


def test_conversation_as_dspy_history():
    import dspy

    from pyreact_durable.react import Conversation

    conv = Conversation(
        [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
         {"role": "system", "content": "note"},
         {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
         {"role": "user", "content": "dangling"}],
        lambda _v: None,
    )
    h = conv.as_dspy_history("question", "answer")
    assert isinstance(h, dspy.History)
    assert h.messages == [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    assert conv.as_chat_messages()[2] == {"role": "system", "content": "note"}
    assert len(conv) == 6


def test_providers_do_not_change_state_paths(tmp_path, run_app):
    """Wrapping the tree in more providers later keeps durable state."""
    from pyreact.core import HarnessProvider, create_context
    from pyreact.tools import ToolProvider

    Extra = create_context(default=None, name="Extra")
    path = str(tmp_path / "wrap.jsonl")
    seen = []

    @component
    def Who():
        n, set_n = use_persistent_state(0, key="n")
        _on_submit(lambda _t: set_n(lambda v: v + 1))
        seen.append((hooks.path, n))
        return []

    @component
    def App():
        return [Durable(key="d", log=path, children=[Who(key="w")])]

    app = run_app(App)
    app.invoke("x", wait=True, timeout=5)
    app.shutdown()
    assert seen[-1] == ("App/Who[w]", 1)

    @component
    def App():  # noqa: F811 - same root name, more (unkeyed) providers in between
        return [Durable(key="d", log=path, children=[
            HarnessProvider(children=[
                Extra(value=1, children=[ToolProvider(tools=[], children=[Who(key="w")])]),
            ]),
        ])]

    seen.clear()
    app = run_app(App)
    app.wait_idle(timeout=5)
    assert seen[-1] == ("App/Who[w]", 1)


def test_state_stored_under_legacy_full_path_is_restored(tmp_path, run_app):
    """Logs written before providers became transparent keep working."""
    path = tmp_path / "old.jsonl"
    with SessionLog.open(path) as log:
        log.set_state("App/Durable[d]/Who[w]", "n", 41)
    seen = []

    @component
    def Who():
        n, set_n = use_persistent_state(0, key="n")
        _on_submit(lambda _t: set_n(lambda v: v + 1))
        seen.append((hooks.path, hooks.full_path, n))
        return []

    @component
    def App():
        return [Durable(key="d", log=str(path), journal=None, children=[Who(key="w")])]

    app = run_app(App)
    app.invoke("x", wait=True, timeout=5)
    app.shutdown()
    assert seen[0] == ("App/Who[w]", "App/Durable[d]/Who[w]", 41)
    assert seen[-1][2] == 42
    with SessionLog.open(path, readonly=True) as log:  # migrated on first write
        assert log.get_state("App/Who[w]", "n") == 42
