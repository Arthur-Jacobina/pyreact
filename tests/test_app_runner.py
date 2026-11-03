import asyncio
import threading
import time

import pytest

from pyreact.boot import bootstrap
from pyreact.core import MessageBuffer, component, hooks, runtime
from pyreact.input.bus import InputBus


def _submit_listener(setter):
    bus = hooks.get_service("input_bus", InputBus)
    hooks.use_effect(
        lambda: bus.subscribe(lambda e: e["type"] == "submit" and setter(e["value"])),
        [],
    )


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


def test_input_sent_right_after_bootstrap_is_not_lost(run_app):
    got = []

    @component
    def App():
        v, setv = hooks.use_state(None)
        _submit_listener(setv)
        hooks.use_effect(lambda: v and got.append(v), [v])
        return []

    app = run_app(App)
    app.invoke("first", wait=True, timeout=5)
    assert got == ["first"]


def test_invoke_wait_waits_for_async_work(run_app):
    done = []

    @component
    def App():
        v, setv = hooks.use_state(None)
        _submit_listener(setv)

        async def work():
            if v:
                await asyncio.sleep(0.3)
                done.append(v)

        hooks.use_effect(work, [v])
        return []

    app = run_app(App)
    t0 = time.time()
    app.invoke("x", wait=True, timeout=5)
    assert done == ["x"] and time.time() - t0 >= 0.3


def test_slow_async_effect_does_not_block_other_components(run_app):
    fast = []

    @component
    def Slow():
        async def work():
            await asyncio.sleep(1.0)

        hooks.use_effect(work, [])
        return []

    @component
    def Fast():
        v, setv = hooks.use_state(None)
        _submit_listener(setv)
        hooks.use_effect(lambda: v and fast.append(time.time()), [v])
        return []

    @component
    def App():
        return [Slow(key="s"), Fast(key="f")]

    app = run_app(App)
    t0 = time.time()
    app.invoke("go")
    while not fast and time.time() - t0 < 2:
        time.sleep(0.01)
    assert fast and fast[0] - t0 < 0.5


def test_runner_survives_component_error(run_app):
    handled, errors = [], []
    runtime.set_error_handler(lambda exc, ctx, phase: errors.append(str(exc)))

    @component
    def App():
        v, setv = hooks.use_state(None)
        _submit_listener(setv)
        if v == "boom":
            raise ValueError("bug")
        hooks.use_effect(lambda: v and handled.append(v), [v])
        return []

    app = run_app(App)
    app.invoke("boom", wait=True, timeout=5)
    app.invoke("after", wait=True, timeout=5)
    assert app.is_running
    assert errors == ["bug"] and handled == ["after"]


def test_invoke_on_stopped_runner_raises_instead_of_hanging(run_app):
    @component
    def App():
        return []

    app = run_app(App)
    app.shutdown()
    with pytest.raises(RuntimeError):
        app.invoke("x", wait=True)


def test_set_state_from_worker_thread(run_app):
    hs, seen = {}, []

    @component
    def App():
        v, setv = hooks.use_state(0)
        hs["set"] = setv
        seen.append(v)
        return []

    app = run_app(App)
    th = threading.Thread(target=lambda: hs["set"](7))
    th.start()
    th.join()
    app.wait_idle(timeout=5)
    assert seen[-1] == 7


def test_message_buffer_is_bounded():
    buf = MessageBuffer()
    old = buf.max_chars
    try:
        buf.clear()
        buf.max_chars = 100
        for _ in range(50):
            buf.append("x" * 10)
        assert buf.length() <= 100
        assert buf.dump() == "x" * buf.length()
    finally:
        buf.max_chars = old
        buf.clear()


def test_boot_star_import():
    ns = {}
    exec("from pyreact.boot import *", ns)
    assert "bootstrap" in ns
