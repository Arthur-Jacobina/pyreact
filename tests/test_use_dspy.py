"""Regression: the supervisor routing in examples/tools.py lagged one turn."""
import asyncio

from integrations.dspy_integration import DSPyProvider
from integrations.use_dspy import use_dspy_call
from pyreact.core import HookContext, component, hooks, runtime
from pyreact.core.runtime import schedule_rerender


class FakeModule:
    """Async module whose answer is derived from the input message."""

    def __init__(self):
        self.calls = []

    async def acall(self, message):
        self.calls.append(message)
        await asyncio.sleep(0.01)
        return type("Pred", (), {"name": f"route-for:{message}"})()


async def _settle():
    await runtime.run_until_idle()


def test_result_effect_fires_once_per_call_with_the_new_answer():
    mod = FakeModule()
    routed, hs = [], {}

    @component
    def Supervisor(message, turn):
        call, result, loading, error = use_dspy_call(mod, lm=object())
        hooks.use_effect(lambda: message and call(message=message), [message, turn])

        def _emit():
            if result[0] is not None:
                routed.append((message, result[0].name, result[1]))

        hooks.use_effect(_emit, [result])
        return []

    @component
    def Home():
        (turn, msg), set_msg = hooks.use_state((0, ""))
        hs["send"] = lambda m: set_msg(lambda p: (p[0] + 1, m))
        return [Supervisor(key="s", message=msg, turn=turn)]

    @component
    def Root():
        return [DSPyProvider(key="d", lm=object(), children=[Home(key="h")])]

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await _settle()
        for m in ["q1", "q2", "q2"]:
            hs["send"](m)
            await _settle()

    asyncio.run(main())
    assert mod.calls == ["q1", "q2", "q2"]
    assert routed == [
        ("q1", "route-for:q1", 1),
        ("q2", "route-for:q2", 2),
        ("q2", "route-for:q2", 3),
    ]


def test_superseded_call_is_cancelled():
    started, finished = [], []

    class Slow:
        async def acall(self, message):
            started.append(message)
            await asyncio.sleep(0.2)
            finished.append(message)
            return message

    hs = {}

    @component
    def Agent():
        call, result, loading, error = use_dspy_call(Slow(), lm=object())
        hs["call"] = call
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=object(), children=[Agent(key="a")])]

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await _settle()
        hs["call"](message="old")
        await asyncio.sleep(0.05)
        hs["call"](message="new")
        await _settle()

    asyncio.run(main())
    assert started == ["old", "new"] and finished == ["new"]
