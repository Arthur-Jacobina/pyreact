"""Activity(mode="visible"|"hidden") and effect cleanup reasons."""

import asyncio

from pyreact.boot import AppRunner
from pyreact.core import Activity, HookContext, component, hooks, runtime, use_cleanup_reason, use_turn
from pyreact.core.activity import hide_tree, reveal_tree
from pyreact.core.runtime import schedule_rerender


def _run(coro):
    return asyncio.run(coro)


async def _mount(fn):
    root = HookContext(fn.__name__, fn)
    schedule_rerender(root)
    await runtime.run_until_idle()
    return root


def test_cleanup_reasons_unmount_and_deps_and_one_arg_cleanups():
    log, setters = [], {}

    @component
    def Child():
        n, set_n = hooks.use_state(0)
        setters["n"] = set_n

        def fx():
            log.append(("run", n))
            return lambda: log.append(("zero-arg", use_cleanup_reason()))

        def fx2():
            def cleanup(reason):
                log.append(("one-arg", reason))
            return cleanup

        hooks.use_effect(fx, [n])
        hooks.use_effect(fx2, [n])
        return []

    @component
    def App():
        show, set_show = hooks.use_state(True)
        setters["show"] = set_show
        return [Child(key="c")] if show else []

    async def main():
        await _mount(App)
        setters["n"](1)
        await runtime.run_until_idle()
        setters["show"](False)
        await runtime.run_until_idle()

    _run(main())
    assert log == [
        ("run", 0),
        ("zero-arg", "deps"), ("run", 1), ("one-arg", "deps"),
        ("zero-arg", "unmount"), ("one-arg", "unmount"),
    ]
    assert use_cleanup_reason() is None


def test_async_effect_cleanup_gets_the_reason():
    log = []

    @component
    def App():
        async def fx():
            await asyncio.sleep(0)
            return lambda reason: log.append(reason)

        hooks.use_effect(fx, [])
        return []

    async def main():
        root = await _mount(App)
        root.unmount()

    _run(main())
    assert log == ["unmount"]


def _activity_app(log, setters):
    @component
    def Panel(label):
        n, set_n = hooks.use_state(0)
        setters[label] = set_n
        log.append(("render", label, n))

        def fx():
            log.append(("effect", label, n))
            return lambda: log.append(("cleanup", label, use_cleanup_reason()))

        hooks.use_effect(fx, [n])
        return []

    @component
    def App():
        mode, set_mode = hooks.use_state("visible")
        setters["mode"] = set_mode
        return [
            Activity(key="act", mode=mode, children=[Panel(key="p", label="inner")]),
            Panel(key="q", label="outer"),
        ]

    return App


def test_hidden_keeps_state_cleans_effects_and_visible_reruns_them():
    log, setters = [], {}
    App = _activity_app(log, setters)

    async def main():
        root = await _mount(App)
        inner = root.children[0].children[0]
        assert inner.path == "App/Panel[p]"  # Activity is transparent
        setters["inner"](5)
        await runtime.run_until_idle()
        log.clear()
        setters["mode"]("hidden")
        await runtime.run_until_idle()
        assert inner.is_hidden
        assert ("cleanup", "inner", "hidden") in log
        assert not any(e[0] == "cleanup" and e[1] == "outer" for e in log)
        log.clear()
        # state changes while hidden re-render but run no effect
        setters["inner"](6)
        await runtime.run_until_idle()
        assert log == [("render", "inner", 6)]
        log.clear()
        setters["mode"]("visible")
        await runtime.run_until_idle()
        assert not inner.is_hidden
        assert ("effect", "inner", 6) in log
        assert hooks_state(inner) == 6

    _run(main())


def hooks_state(ctx):
    return ctx.hooks[0]


def test_activity_rejects_unknown_mode():
    @component
    def App():
        return [Activity(key="a", mode="invisible", children=[])]

    errors = []
    runtime.set_error_handler(lambda exc, ctx, phase: errors.append(exc))
    _run(_mount(App))
    assert errors and "mode" in str(errors[0])


def test_mounted_hidden_never_runs_effects_until_revealed():
    log, setters = [], {}

    @component
    def Leaf():
        hooks.use_effect(lambda: log.append("effect"), [])
        return []

    @component
    def App():
        mode, set_mode = hooks.use_state("hidden")
        setters["mode"] = set_mode
        return [Activity(key="a", mode=mode, children=[Leaf(key="l")])]

    async def main():
        await _mount(App)
        assert log == []
        setters["mode"]("visible")
        await runtime.run_until_idle()
        assert log == ["effect"]

    _run(main())


def test_hide_tree_suspend_reason_and_reveal():
    log = []

    @component
    def App():
        hooks.use_effect(lambda: (log.append("run"),
                                  lambda: log.append(use_cleanup_reason()))[1], [])
        return []

    async def main():
        root = await _mount(App)
        tasks = hide_tree(root, "suspend")
        assert tasks == [] and log == ["run", "suspend"]
        root.unmount()  # cleanups already ran: nothing more
        assert log == ["run", "suspend"]

        root2 = await _mount(App)
        hide_tree(root2, "suspend")
        reveal_tree(root2)
        await runtime.run_until_idle()
        assert log[-3:] == ["run", "suspend", "run"]

    _run(main())


def test_hidden_subtree_gets_no_turns():
    seen = []

    @component
    def Agent():
        turn = use_turn()
        if turn is not None:
            seen.append(turn.text)
        return []

    @component
    def App():
        mode = hooks.get_service("mode", lambda: {"v": "visible"})["v"]
        return [Activity(key="a", mode=mode, children=[Agent(key="agent")])]

    app = AppRunner(App, default_session=False)
    try:
        s = app.open_session("s")
        s.invoke("one", wait=True, timeout=5)
        s.services["mode"]["v"] = "hidden"

        async def rerender():
            schedule_rerender(s.root)

        app._submit(rerender()).result()
        s.wait_idle(timeout=5)
        s.invoke("two", wait=True, timeout=5)
        s.services["mode"]["v"] = "visible"
        app._submit(rerender()).result()
        s.wait_idle(timeout=5)
        assert "two" not in seen  # started while hidden: not delivered
        s.invoke("three", wait=True, timeout=5)
        assert seen[-1] == "three"
    finally:
        app.shutdown()
