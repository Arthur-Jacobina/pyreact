import asyncio

from pyreact.core import HookContext, component, create_context, hooks, runtime
from pyreact.core.runtime import run_renders, schedule_rerender


async def _flush():
    for _ in range(3):
        await asyncio.sleep(0)
        await run_renders()


async def _mount(fn):
    root = HookContext(fn.__name__, fn)
    schedule_rerender(root)
    await _flush()
    return root


def test_contexts_with_same_name_are_independent():
    a = create_context(name="Same")
    b = create_context(name="Same")
    assert a is not b


def test_provider_value_does_not_leak_to_siblings():
    Ctx = create_context(name="Leak", default="DEFAULT")
    seen = {}

    @component
    def Reader(tag):
        seen[tag] = hooks.use_context(Ctx)
        return []

    @component
    def App():
        return [
            Ctx(key="p", value="INSIDE", children=[Reader(key="in", tag="inside")]),
            Reader(key="out", tag="outside"),
        ]

    asyncio.run(_mount(App))
    assert seen == {"inside": "INSIDE", "outside": "DEFAULT"}


def test_isolated_child_rerender_sees_current_provider_value():
    Ctx = create_context(name="Env", default=None)
    seen, setters = [], {}

    @component
    def Child():
        n, setn = hooks.use_state(0)
        setters["child"] = setn
        seen.append((n, hooks.use_context(Ctx)))
        return []

    @component
    def Prov():
        v, setv = hooks.use_state("v1")
        setters["prov"] = setv
        return [Ctx(key="p", value=v, children=[Child(key="c")])]

    async def main():
        await _mount(Prov)
        setters["prov"]("v2")
        await _flush()
        setters["child"](1)  # rerenders Child alone
        await _flush()

    asyncio.run(main())
    assert seen[-1] == (1, "v2")


def test_context_set_without_provider_rerenders_subscribers():
    Ctx = create_context(name="Global", default="a")
    seen = []

    @component
    def Reader():
        seen.append(hooks.use_context(Ctx))
        return []

    async def main():
        root = await _mount(Reader)  # keep a strong ref (subscribers are weak)
        Ctx.set("b")
        await _flush()
        return root

    asyncio.run(main())
    assert seen[-1] == "b"


def test_effect_without_deps_runs_every_render():
    runs, hs = [0], {}

    @component
    def E():
        n, setn = hooks.use_state(0)
        hs["set"] = setn
        hooks.use_effect(lambda: runs.__setitem__(0, runs[0] + 1), None)
        return []

    async def main():
        await _mount(E)
        for i in range(3):
            hs["set"](i + 1)
            await _flush()

    asyncio.run(main())
    assert runs[0] == 4


def test_render_error_is_isolated_to_the_failing_component():
    errors, ok = [], []
    runtime.set_error_handler(lambda exc, ctx, phase: errors.append((ctx.name, phase)))

    @component
    def Bad():
        raise ValueError("boom")

    @component
    def Good():
        hooks.use_effect(lambda: ok.append(True), [])
        return []

    @component
    def App():
        return [Bad(key="bad"), Good(key="good")]

    asyncio.run(_mount(App))
    assert errors == [("Bad", "render")]
    assert ok == [True]


def test_effect_error_does_not_stop_other_effects():
    errors, ran = [], []
    runtime.set_error_handler(lambda exc, ctx, phase: errors.append(phase))

    @component
    def App():
        hooks.use_effect(lambda: 1 / 0, [])
        hooks.use_effect(lambda: ran.append(1), [])
        return []

    asyncio.run(_mount(App))
    assert errors == ["effect"] and ran == [1]


def test_failed_render_keeps_previous_children_mounted():
    cleanups, hs = [], {}
    runtime.set_error_handler(lambda *a: None)

    @component
    def Child():
        hooks.use_effect(lambda: (lambda: cleanups.append("child")), [])
        return []

    @component
    def Parent():
        n, setn = hooks.use_state(0)
        hs["set"] = setn
        if n == 1:
            raise RuntimeError("transient")
        return [Child(key="c")]

    async def main():
        root = await _mount(Parent)
        hs["set"](1)
        await _flush()
        return root

    root = asyncio.run(main())
    assert cleanups == []  # child was not leaked nor unmounted
    assert [c.name for c in root.children] == ["Child"]


def test_async_effect_is_cancelled_when_deps_change():
    events, hs = [], {}

    @component
    def App():
        v, setv = hooks.use_state("a")
        hs["set"] = setv

        async def work():
            try:
                await asyncio.sleep(0.2)
                events.append(("done", v))
            except asyncio.CancelledError:
                events.append(("cancelled", v))
                raise

        hooks.use_effect(work, [v])
        return []

    async def main():
        await _mount(App)
        await asyncio.sleep(0.05)
        hs["set"]("b")
        await runtime.run_until_idle()

    asyncio.run(main())
    assert events == [("cancelled", "a"), ("done", "b")]


def test_providers_are_transparent_in_paths():
    Ctx = create_context(name="P")
    seen = {}

    @component(transparent=True)
    def MyProvider(children=None):
        seen["provider"] = (hooks.path, hooks.full_path)
        return list(children or [])

    @component
    def Leaf(tag):
        seen[tag] = (hooks.path, hooks.full_path)
        return []

    @component
    def Box(children=None):
        return list(children or [])

    @component
    def App():
        return [
            Ctx(value=1, children=[MyProvider(key="mine", children=[Leaf(key="a", tag="a")])]),
            Box(key="box", children=[Ctx(value=2, children=[Leaf(tag="b")])]),
        ]

    asyncio.run(_mount(App))
    assert seen["a"] == ("App/Leaf[a]", "App/_Provider[#0]/MyProvider[mine]/Leaf[a]")
    # a provider keeps a path of its own; only its descendants skip it
    assert seen["provider"] == ("App/MyProvider[mine]", "App/_Provider[#0]/MyProvider[mine]")
    # ordinary components still contribute a segment
    assert seen["b"][0] == "App/Box[box]/Leaf[#0]"


def test_use_ref_is_stable_and_does_not_rerender():
    from pyreact.core import Ref

    refs, renders = [], []
    bump = {}

    @component
    def App():
        ref = hooks.use_ref(0)
        n, set_n = hooks.use_state(0)
        bump["set"] = set_n
        bump["ref"] = ref
        refs.append(ref)
        renders.append(ref.current)
        return []

    async def main():
        await _mount(App)
        bump["ref"].current = 41  # mutation alone: no rerender
        await _flush()
        assert len(renders) == 1
        bump["set"](1)
        await _flush()

    asyncio.run(main())
    assert isinstance(refs[0], Ref) and refs[0] is refs[1]
    assert renders == [0, 41]  # initial ignored after mount
