"""Tool calls flow through the harness (``"tool"`` boundaries). No network."""

import asyncio

from dspy.utils.dummies import DummyLM

from integrations import DSPyProvider, call_tool, use_agent
from pyreact.core import EventBus, HookContext, component, runtime
from pyreact.core.harness import Blocked, Harness, HarnessProvider, middleware
from pyreact.core.runtime import schedule_rerender
from pyreact.tools import ToolProvider, make_tool


def lookup_price(sku: str) -> str:
    """Look up a price.
    sku: the sku
    """
    return f"price({sku})=10"


def _script(tool_args):
    return DummyLM([
        {"next_thought": "look", "next_tool_name": "lookup_price", "next_tool_args": tool_args},
        {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "ok", "answer": "10"},
    ])


def _run_agent(mws, lm):
    hs = {}
    events = []
    HookContext.get_service("event_bus", EventBus).subscribe(events.append, ["llm.*", "tool.*"])

    @component
    def Agent():
        hs["call"] = use_agent("question -> answer", tools=["lookup_price"], max_iters=3)
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[
            ToolProvider(key="t", tools=[lookup_price], children=[
                HarnessProvider(key="h", middleware=mws, children=[Agent(key="a")])
            ])
        ])]

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await runtime.run_until_idle()
        hs["call"].run(question="price of A1?")
        await runtime.run_until_idle()

    asyncio.run(main())
    return hs["call"], events


def test_react_tool_call_is_a_tool_boundary_inside_the_llm_call():
    seen = []

    async def spy(b, call_next):
        out = await call_next(b)
        seen.append((b.kind, b.name, dict(b.inputs), dict(b.attrs)))
        return out

    call, _ = _run_agent([spy], _script({"sku": "A1"}))
    assert call.error is None and call.result.answer == "10"
    tools = [s for s in seen if s[0] == "tool"]
    llms = [s for s in seen if s[0] == "llm"]
    assert len(tools) == 1 and len(llms) == 1
    _, name, inputs, attrs = tools[0]
    assert name == "lookup_price" and inputs == {"sku": "A1"}
    assert attrs["call_id"] == llms[0][3]["call_id"] == call.call_id  # parent LLM call
    assert attrs["result"] == "price(A1)=10" and attrs["latency"] >= 0


def test_tool_middleware_can_rewrite_args_and_results():
    @middleware(kinds=["tool"])
    async def normalize(b, call_next):
        b.inputs = {**b.inputs, "sku": b.inputs["sku"].upper()}
        return (await call_next(b)) + " (verified)"

    call, events = _run_agent([normalize], _script({"sku": "a1"}))
    end = next(e for e in events if e.kind == "tool.end").payload
    assert end["result"] == "price(A1)=10 (verified)" and end["error"] is None
    assert "price(A1)=10 (verified)" in str(call.result.trajectory)


def test_blocked_tool_returns_an_error_string_to_the_agent():
    @middleware(kinds=["tool"], name="no-prices")
    async def guard(b, call_next):
        raise Blocked("prices are confidential")

    call, events = _run_agent([guard], _script({"sku": "A1"}))
    assert call.error is None and call.result.answer == "10"  # the agent went on
    msg = "Error: tool 'lookup_price' blocked by no-prices: prices are confidential"
    assert msg in str(call.result.trajectory)
    kinds = [e.kind for e in events]
    assert kinds.index("tool.start") < kinds.index("tool.blocked") < kinds.index("tool.end")
    blocked = next(e for e in events if e.kind == "tool.blocked").payload
    assert blocked["by"] == "no-prices" and blocked["reason"] == "prices are confidential"
    end = next(e for e in events if e.kind == "tool.end").payload
    assert end["error"].startswith("blocked: ")


def test_call_tool_binds_positional_args_and_retries():
    runs = []

    def add(a: int, b: int = 1) -> int:
        runs.append((a, b))
        if len(runs) == 1:
            raise RuntimeError("flaky")
        return a + b

    async def retry(bnd, call_next):
        try:
            return await call_next(bnd)
        except RuntimeError:
            bnd.attempt += 1
            return await call_next(bnd)

    inputs = []

    async def spy(bnd, call_next):
        inputs.append(dict(bnd.inputs))
        return await call_next(bnd)

    h = Harness([spy, retry])
    out = asyncio.run(call_tool(make_tool(add), (2,), {"b": 3}, harness=h))
    assert out == 5 and runs == [(2, 3), (2, 3)] and inputs == [{"a": 2, "b": 3}]
