"""Turn model: ids, lifecycle events, queue/interrupt policies, use_turn."""

import asyncio
import threading
import time

import pytest

from integrations.dspy_integration import DSPyProvider
from integrations.use_dspy import use_dspy_call
from pyreact.boot import bootstrap
from pyreact.components.keystroke import Keystroke
from pyreact.core import component, hooks, runtime, use_turn
from pyreact.core.turns import TurnManager
from pyreact.input.bus import InputBus


@pytest.fixture
def run_app():
    apps = []

    def _start(fn, **kw):
        app = bootstrap(fn, **kw)
        apps.append(app)
        return app

    yield _start
    for app in apps:
        app.shutdown()


def _record_turn_events(app):
    evs, lock = [], threading.Lock()
    bus = app.event_bus  # the default session's bus

    def _on(ev):
        with lock:
            evs.append(ev)

    bus.subscribe(_on, ["turn.*"])
    return evs


def _slow_agent(log, delay=0.2, policy=None):
    """Component that 'answers' every turn after ``delay`` seconds."""

    @component
    def Agent():
        turn = use_turn(policy=policy)

        async def answer():
            if turn is None:
                return
            log.append(("start", turn.text, runtime.current_turn_id()))
            await asyncio.sleep(delay)
            log.append(("answer", turn.text, runtime.current_turn_id()))

        hooks.use_effect(answer, [turn])
        return []

    return Agent


def test_invoke_returns_turn_id_and_publishes_lifecycle(run_app):
    seen = []

    @component
    def App():
        bus = hooks.get_service("input_bus", InputBus)
        hooks.use_effect(lambda: bus.subscribe(lambda e: seen.append(dict(e))), [])
        return []

    app = run_app(App)
    evs = _record_turn_events(app)
    tid = app.invoke("hello", wait=True, timeout=5)
    assert tid == "t1"
    assert [e["type"] for e in seen] == ["text", "submit"]
    assert all(e["turn_id"] == tid for e in seen)
    assert [(e.kind, e.turn_id) for e in evs] == [("turn.start", tid), ("turn.end", tid)]
    assert evs[0].payload["text"] == "hello"
    assert evs[1].payload["status"] == "done" and evs[1].payload["duration"] >= 0
    assert app.turns.result(tid)["status"] == "done"


def test_queue_policy_answers_burst_in_order(run_app):
    log = []
    app = run_app(_slow_agent(log))
    evs = _record_turn_events(app)
    t1 = app.invoke("q1")
    t2 = app.invoke("q2")
    assert app.wait_turn(t2, timeout=5)["status"] == "done"
    assert [x for x in log if x[0] == "answer"] == [("answer", "q1", t1), ("answer", "q2", t2)]
    kinds = [(e.kind, e.turn_id) for e in evs]
    # turn 2 only starts once turn 1 has ended
    assert kinds == [
        ("turn.start", t1),
        ("turn.end", t1),
        ("turn.start", t2),
        ("turn.end", t2),
    ]
    # turn 1 ended only after its async work finished
    assert evs[1].payload["duration"] >= 0.2


def test_same_input_twice_is_two_turns(run_app):
    log = []
    app = run_app(_slow_agent(log, delay=0.01))
    a = app.invoke("same", wait=True, timeout=5)
    b = app.invoke("same", wait=True, timeout=5)
    assert a != b
    assert [x for x in log if x[0] == "answer"] == [("answer", "same", a), ("answer", "same", b)]


def test_interrupt_policy_cancels_current_turn(run_app):
    log = []
    app = run_app(_slow_agent(log, delay=0.5), turn_policy="interrupt")
    evs = _record_turn_events(app)
    t1 = app.invoke("q1")
    time.sleep(0.1)  # let turn 1 start its work
    t2 = app.invoke("q2", wait=True, timeout=5)
    assert app.turns.result(t1)["status"] == "interrupted"
    assert app.turns.result(t2)["status"] == "done"
    assert [x[1] for x in log if x[0] == "answer"] == ["q2"]
    ends = {e.turn_id: e.payload["status"] for e in evs if e.kind == "turn.end"}
    assert ends == {t1: "interrupted", t2: "done"}
    # turn 1 ended well before its 0.5s of work would have finished
    assert app.turns.result(t1)["duration"] < 0.4


def test_component_can_override_policy(run_app):
    """App policy 'queue', but this component's work gives way to new input."""
    log = []
    app = run_app(_slow_agent(log, delay=0.5, policy="interrupt"))
    t1 = app.invoke("q1")
    time.sleep(0.1)
    t2 = app.invoke("q2", wait=True, timeout=5)
    assert app.turns.result(t1)["status"] == "interrupted"
    assert [x[1] for x in log if x[0] == "answer"] == ["q2"]
    assert t2


def test_component_queue_policy_shields_work_from_interrupt(run_app):
    log = []
    app = run_app(_slow_agent(log, delay=0.3, policy="queue"), turn_policy="interrupt")
    t1 = app.invoke("q1")
    time.sleep(0.05)
    app.invoke("q2", wait=True, timeout=5)
    assert app.turns.result(t1)["status"] == "done"
    assert [x[1] for x in log if x[0] == "answer"] == ["q1", "q2"]


def test_task_error_marks_turn_error(run_app):
    errors = []
    runtime.set_error_handler(lambda exc, ctx, phase: errors.append(phase))

    @component
    def App():
        turn = use_turn()

        async def boom():
            if turn is not None:
                await asyncio.sleep(0.01)
                raise ValueError("x")

        hooks.use_effect(boom, [turn])
        return []

    app = run_app(App)
    tid = app.invoke("go", wait=True, timeout=5)
    assert app.turns.result(tid)["status"] == "error"
    assert errors == ["task"]
    tid2 = app.invoke("go", wait=True, timeout=5)
    assert app.turns.result(tid2)["status"] == "error"


def test_untagged_background_task_does_not_hold_turn(run_app):
    @component
    def App():
        turn = use_turn()

        def start_bg():
            if turn is not None:
                runtime.spawn(asyncio.sleep(30), turn=None, name="bg")

        hooks.use_effect(start_bg, [turn])
        return []

    app = run_app(App)
    app.invoke("go", wait=True, timeout=2)


def test_spawned_tasks_inherit_turn():
    seen = {}

    async def child():
        seen["child"] = runtime.current_turn_id()

    async def parent():
        seen["parent"] = runtime.current_turn_id()
        await runtime.spawn(child())

    async def main():
        runtime.set_active_turn("tA")
        t = runtime.spawn(parent())
        runtime.set_active_turn("tB")  # later turns don't affect running tasks
        await t
        assert runtime.task_turn(t) is None  # bookkeeping cleaned up
        runtime.set_active_turn(None)

    asyncio.run(main())
    assert seen == {"parent": "tA", "child": "tA"}


def test_keystroke_on_submit_one_and_two_args(run_app):
    one, two = [], []

    @component
    def App():
        return [
            Keystroke(key="a", on_submit=lambda text: one.append(text)),
            Keystroke(key="b", on_submit=lambda text, turn: two.append((text, turn))),
        ]

    app = run_app(App)
    tid = app.invoke("hi", wait=True, timeout=5)
    app.invoke("hi", wait=True, timeout=5)
    assert one == ["hi", "hi"]
    assert two[0][0] == "hi" and two[0][1].id == tid and two[0][1].index == 1
    assert two[1][1].index == 2


def test_use_dspy_call_is_attributed_to_its_turn(run_app):
    answers = []

    class FakeModule:
        async def acall(self, message):
            await asyncio.sleep(0.2)
            return type("Pred", (), {"answer": f"re:{message}"})()

    mod = FakeModule()

    @component
    def Agent():
        turn = use_turn()
        call, result, _loading, _err = use_dspy_call(mod, lm=object())
        hooks.use_effect(lambda: turn and call(message=turn.text), [turn])
        hooks.use_effect(
            lambda: result[0] is not None
            and answers.append((result[0].answer, runtime.current_turn_id())),
            [result],
        )
        return []

    @component
    def App():
        return [DSPyProvider(key="d", lm=object(), children=[Agent(key="a")])]

    app = run_app(App)
    t1 = app.invoke("a")
    t2 = app.invoke("b", wait=True, timeout=5)
    # wait=True on turn 2 implies turn 1's LLM call completed and was answered
    assert answers == [("re:a", t1), ("re:b", t2)]


def test_turn_manager_requires_valid_policy():
    with pytest.raises(ValueError):
        TurnManager().configure(policy="nope")
