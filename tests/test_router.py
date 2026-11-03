import asyncio

from pyreact.core import HookContext, component, hooks
from pyreact.core.runtime import run_renders, schedule_rerender
from pyreact.router import NavService, Route, Router, use_query_params, use_route_params


async def _flush():
    for _ in range(5):
        await asyncio.sleep(0)
        await run_renders()


def test_router_query_and_params_do_not_tear():
    renders, setters = [], {}

    @component
    def Probe():
        n, setn = hooks.use_state(0)
        setters["probe"] = setn
        renders.append((use_route_params().get("id"), use_query_params().get("q")))
        return []

    @component
    def App():
        return [
            Router(
                children=[
                    Route(key="items", path="/items/:id", children=[Probe(key="p")])
                ]
            )
        ]

    async def main():
        root = HookContext("App", App)
        nav = root.get_service("nav_service", NavService)
        nav.current = "/items/1?q=1"
        schedule_rerender(root)
        await _flush()
        assert renders[-1] == ("1", "1")
        # Probe rerenders in isolation *before* the Router processes the nav.
        setters["probe"](1)
        nav.navigate("/items/2", query={"q": "2"})
        await _flush()
        return root

    asyncio.run(main())
    assert renders[-1] == ("2", "2")
    assert all(pid == q for pid, q in renders), renders
