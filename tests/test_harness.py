import asyncio

import pytest

from pyreact.boot import bootstrap
from pyreact.core import component, hooks
from pyreact.core.harness import (
    Blocked,
    HarnessProvider,
    install,
    middleware,
    through,
    use_harness,
)


def _tag(name, log):
    async def mw(b, call_next):
        log.append(f"{name}>")
        try:
            return await call_next(b)
        finally:
            log.append(f"<{name}")

    mw.harness_name = name
    return mw


def test_order_global_then_providers_outer_to_inner():
    log, seen = [], {}
    uninstall = install(_tag("global", log))

    @component
    def Leaf():
        h = use_harness()

        async def run():
            seen["result"] = await through("tool", "t", {"x": 1}, lambda b: _done(log, b),
                                           harness=h)

        hooks.use_effect(lambda: run(), [])
        return []

    @component
    def App():
        return [HarnessProvider(key="outer", middleware=[_tag("outer", log)], children=[
            HarnessProvider(key="inner", middleware=[_tag("inner", log)], children=[
                Leaf(key="leaf")
            ])
        ])]

    app = bootstrap(App)
    try:
        app.wait_idle(timeout=5)
    finally:
        app.shutdown()
        uninstall()
    assert log == ["global>", "outer>", "inner>", "call", "<inner", "<outer", "<global"]
    assert seen["result"] == "ok:1"


async def _done(log, b):
    log.append("call")
    return f"ok:{b.inputs['x']}"


def test_kind_filter_short_circuit_retry_and_block():
    calls = []

    @middleware(kinds=["llm"])
    async def cache(b, call_next):
        if b.inputs.get("q") == "cached":
            return "from-cache"  # short-circuit
        return await call_next(b)

    async def retry_once(b, call_next):
        try:
            return await call_next(b)
        except RuntimeError:
            b.attempt += 1
            return await call_next(b)

    @middleware(kinds=["tool.*"])
    async def guard(b, call_next):
        if b.name == "rm":
            raise Blocked("rm is not allowed", boundary=b, by="guard")
        return await call_next(b)

    async def flaky(b):
        calls.append(b.attempt)
        if b.attempt == 1:
            raise RuntimeError("transient")
        return "done"

    from pyreact.core.harness import Harness

    h = Harness([cache, retry_once, guard])

    async def main():
        r1 = await through("llm", "m", {"q": "cached"}, flaky, harness=h)
        r2 = await through("llm", "m", {"q": "x"}, flaky, harness=h)
        with pytest.raises(Blocked) as exc:
            await through("tool.exec", "rm", {}, flaky, harness=h)
        return r1, r2, exc.value

    r1, r2, blocked = asyncio.run(main())
    assert r1 == "from-cache"
    assert r2 == "done" and calls == [1, 2]  # retried with attempt bumped
    assert blocked.by == "guard" and blocked.boundary.name == "rm"


def test_boundary_carries_component_session_and_turn():
    seen = {}

    async def spy(b, call_next):
        seen.update(component=b.component, session=b.session_id, turn=b.turn_id, kind=b.kind)
        return await call_next(b)

    uninstall = install(spy)

    @component
    def Agent():
        from pyreact.core import use_turn

        turn = use_turn()

        async def act():
            await through("llm", "answer", {"q": turn.text}, lambda b: _noop())

        hooks.use_effect(lambda: act() if turn else None, [turn and turn.id])
        return []

    app = bootstrap(Agent, default_session=False)
    try:
        s = app.open_session("sess-1")
        s.invoke("hello", wait=True, timeout=5)
    finally:
        app.shutdown()
        uninstall()
    assert seen["kind"] == "llm"
    assert seen["component"] == "Agent"
    assert seen["session"] == "sess-1"
    assert seen["turn"] and seen["turn"].endswith("t1")


async def _noop():
    return None
