"""Call.inputs / Call.tag: tie a result to the request that started it."""

import asyncio

import dspy
from dspy.utils.dummies import DummyLM

from integrations import DSPyProvider, use_call, use_module
from pyreact.core import HookContext, component, runtime
from pyreact.core.runtime import schedule_rerender


def _app(signature, lm, script):
    snaps, hs = [], {}

    @component
    def Agent():
        call = use_call(use_module(signature))
        hs["call"] = call
        snaps.append((call.loading, call.ver, call.inputs, call.tag,
                      getattr(call.result, "answer", None)))
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[Agent(key="a")])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await runtime.run_until_idle()
        await script(hs)

    asyncio.run(main())
    return snaps


class SlowLM(DummyLM):
    async def aforward(self, *a, **kw):
        await asyncio.sleep(0.05)
        return self.forward(*a, **kw)


def test_inputs_and_tag_follow_the_in_flight_then_settled_call():
    lm = SlowLM([{"answer": "one"}, {"answer": "two"}])
    sent = []
    orig = lm.forward

    def forward(*a, **kw):
        sent.append(str(kw.get("messages") or a))
        return orig(*a, **kw)

    lm.forward = forward

    async def script(hs):
        hs["call"].run(question="q1", tag="t1")
        await asyncio.sleep(0.01)
        await runtime.run_renders()
        await runtime.run_until_idle()
        hs["call"].run(question="q2", tag="t2")
        await asyncio.sleep(0.01)
        await runtime.run_renders()
        await runtime.run_until_idle()

    snaps = _app("question -> answer", lm, script)
    assert snaps[0] == (False, 0, None, None, None)
    assert (True, 0, {"question": "q1"}, "t1", None) in snaps  # in flight
    assert (False, 1, {"question": "q1"}, "t1", "one") in snaps  # settled
    assert (True, 1, {"question": "q2"}, "t2", "one") in snaps  # old result, new request
    assert snaps[-1] == (False, 2, {"question": "q2"}, "t2", "two")
    assert all("t1" not in s and "t2" not in s for s in sent)  # tag never reaches the LM


def test_tag_is_an_input_when_the_signature_declares_it():
    class Sig(dspy.Signature):
        tag: str = dspy.InputField()
        answer: str = dspy.OutputField()

    async def script(hs):
        hs["call"].run(tag="python")
        await runtime.run_until_idle()

    snaps = _app(Sig, DummyLM([{"answer": "ok"}]), script)
    assert snaps[-1] == (False, 1, {"tag": "python"}, None, "ok")


def test_error_keeps_the_identity_of_the_failed_call():
    class Boom(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="boom", cache=False)

        def forward(self, *a, **kw):
            raise RuntimeError("boom")

    seen = {}

    async def script(hs):
        hs["call"].run(question="q", tag=("turn", 3))
        await runtime.run_until_idle()
        seen["call"] = hs["call"]

    _app("question -> answer", Boom(), script)
    c = seen["call"]
    assert c.error is not None and c.tag == ("turn", 3) and c.inputs == {"question": "q"}
