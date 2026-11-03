"""use_dehydrate: registration, sync/async hydration, Hydrating fallback."""

import asyncio
import warnings

from pyreact.core import (
    HookContext,
    Hydrating,
    HydrationWarning,
    component,
    hooks,
    runtime,
    use_dehydrate,
)
from pyreact.core.dehydrate import begin_hydration
from pyreact.core.runtime import schedule_rerender


async def _mount(fn, slices=None):
    root = HookContext(fn.__name__, fn)
    state = begin_hydration(root, slices) if slices is not None else None
    schedule_rerender(root)
    await runtime.run_until_idle()
    return root, state


def _counter(seen):
    @component
    def Counter(hydrate=None):
        start = use_dehydrate("count", dehydrate=lambda: count, hydrate=hydrate, initial=0)
        count, _ = hooks.use_state(start)
        seen.append(count)
        return []

    return Counter


def test_fresh_tree_returns_initial_and_registers_by_path():
    seen = []
    Counter = _counter(seen)

    @component
    def App():
        return [Counter(key="c")]

    root, _ = asyncio.run(_mount(App))
    c = root.children[0]
    assert seen == [0]
    assert c.path == "App/Counter[c]" and list(c._dehydrators) == ["count"]
    assert c._dehydrators["count"].dehydrate() == 0
    root.unmount()
    assert "_dehydrators" not in c.__dict__  # registrations follow the lifecycle


def test_sync_hydration_during_first_render():
    seen = []
    Counter = _counter(seen)

    @component
    def App():
        return [Counter(key="a"), Counter(key="b", hydrate=lambda s: s * 10)]

    root, state = asyncio.run(_mount(App, {"App/Counter[a]": {"count": 3},
                                           "App/Counter[b]": {"count": 4}}))
    assert seen == [3, 40]
    assert state.claimed == {("App/Counter[a]", "count"), ("App/Counter[b]", "count")}
    assert state.mismatches == []


def test_missing_slice_and_failing_hydrate_are_mismatches_not_errors():
    seen = []
    Counter = _counter(seen)

    def boom(_):
        raise ValueError("bad slice")

    @component
    def App():
        return [Counter(key="a"), Counter(key="b", hydrate=boom)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        root, state = asyncio.run(_mount(App, {"App/Counter[b]": {"count": 1}}))
    assert seen == [0, 0]
    kinds = [(m["kind"], m["path"], m["key"]) for m in state.mismatches]
    assert kinds == [("missing", "App/Counter[a]", "count"), ("error", "App/Counter[b]", "count")]
    assert "bad slice" in state.mismatches[1]["error"]
    assert sum(issubclass(w.category, HydrationWarning) for w in caught) == 2


def test_async_hydrate_renders_nothing_then_mounts_with_the_value():
    log = []

    @component
    def Leaf(v):
        hooks.use_effect(lambda: log.append(("leaf effect", v)), [])
        return []

    @component
    def Slow():
        async def hydrate(s):
            await asyncio.sleep(0.01)
            return s + 1

        v = use_dehydrate("v", dehydrate=lambda: v, hydrate=hydrate, initial=-1)
        hooks.use_effect(lambda: log.append(("effect", v)), [])
        log.append(("render", v))
        return [Leaf(key="leaf", v=v)]

    @component
    def App():
        return [Slow(key="s")]

    async def main():
        root = HookContext("App", App)
        begin_hydration(root, {"App/Slow[s]": {"v": 1}})
        schedule_rerender(root)
        await asyncio.sleep(0)
        await runtime.run_renders()
        slow = root.children[0]
        assert slow.children == [] and log == [("render", -1)]
        await runtime.run_until_idle()
        return root

    root = asyncio.run(main())
    assert log == [("render", -1), ("render", 2), ("effect", 2), ("leaf effect", 2)]
    assert root.children[0]._dehydrators["v"].dehydrate() == 2


def test_hydrating_boundary_shows_fallback_and_holds_siblings():
    log = []

    @component
    def Spinner():
        hooks.use_effect(lambda: (log.append("spinner"),
                                  lambda: log.append("spinner gone"))[1], [])
        return []

    @component
    def Sibling():
        hooks.use_effect(lambda: log.append("sibling effect"), [])
        return []

    gate = {}

    @component
    def Slow():
        async def hydrate(s):
            await gate["ev"].wait()
            return s

        v = use_dehydrate("v", dehydrate=lambda: v, hydrate=hydrate, initial=None)
        hooks.use_effect(lambda: log.append(("slow effect", v)), [])
        return []

    @component
    def App():
        return [Hydrating(key="h", fallback=Spinner(), children=[
            Slow(key="slow"), Sibling(key="sib")])]

    async def main():
        gate["ev"] = asyncio.Event()
        root = HookContext("App", App)
        begin_hydration(root, {"App/Slow[slow]": {"v": "x"}})
        schedule_rerender(root)
        for _ in range(5):
            await asyncio.sleep(0)
            await runtime.run_renders()
        assert log == ["spinner"]
        sib = [n for n in root._descendants() if n.name == "Sibling"][0]
        assert sib.path == "App/Sibling[sib]"  # Hydrating is transparent
        gate["ev"].set()
        await runtime.run_until_idle()

    asyncio.run(main())
    assert log[0] == "spinner"
    assert set(log[1:]) == {"spinner gone", ("slow effect", "x"), "sibling effect"}


def test_hooks_proxy_and_duplicate_keys():
    errors = []
    runtime.set_error_handler(lambda exc, ctx, phase: errors.append(exc))

    @component
    def Dup():
        hooks.use_dehydrate("k", dehydrate=lambda: 1)
        hooks.use_dehydrate("k", dehydrate=lambda: 2)
        return []

    asyncio.run(_mount(Dup))
    assert errors and "used twice" in str(errors[0])
