"""Boundary nesting: parent_id links turn -> llm -> tool (current_boundary())."""

import asyncio
import threading

from dspy.utils.dummies import DummyLM

from integrations import DSPyProvider, use_agent
from pyreact.boot import bootstrap
from pyreact.core import component, hooks, use_turn
from pyreact.core.harness import Harness, current_boundary, install, through
from pyreact.tools import ToolProvider


def test_turn_llm_tool_nesting_and_current_boundary_in_worker_threads():
    spans, inside = [], {}
    lock = threading.Lock()

    async def recorder(b, call_next):
        assert current_boundary() is b
        try:
            return await call_next(b)
        finally:
            with lock:
                spans.append((b.kind, b.name, b.id, b.parent_id, b.turn_id))

    def lookup_price(sku: str) -> str:
        """Look up a price.
        sku: the sku
        """
        cur = current_boundary()  # sync tool: runs in a worker thread
        inside["tool"] = (cur.kind, cur.name, threading.current_thread() is not threading.main_thread())
        return "10"

    lm = DummyLM([
        {"next_thought": "look", "next_tool_name": "lookup_price", "next_tool_args": {"sku": "A1"}},
        {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "ok", "answer": "10"},
    ])

    @component
    def Agent():
        turn = use_turn()
        agent = use_agent("question -> answer", tools=["lookup_price"], max_iters=3)
        hooks.use_effect(lambda: agent.run(question=turn.text) if turn else None,
                         [turn and turn.id])
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[
            ToolProvider(key="t", tools=[lookup_price], children=[Agent(key="a")])
        ])]

    un = install(recorder)
    app = bootstrap(Root)
    try:
        tid = app.invoke("price of A1?", wait=True, timeout=5)
    finally:
        app.shutdown()
        un()
    by_kind = {s[0]: s for s in spans}
    turn, llm, tool = by_kind["turn"], by_kind["llm"], by_kind["tool"]
    assert turn[3] is None and turn[4] == tid
    assert llm[3] == turn[2] and llm[4] == tid  # spawned from an effect: via the turn registry
    assert tool[3] == llm[2] and tool[4] == tid  # contextvar inside the LLM call
    assert inside["tool"] == ("tool", "lookup_price", True)
    assert current_boundary() is None


def test_nested_through_in_middleware_and_tasks_and_explicit_parent():
    seen = {}

    async def leaf(b):
        seen.setdefault("leafs", []).append((b.kind, b.parent_id))
        # tasks spawned inside inherit the boundary
        seen["task"] = await asyncio.create_task(_cur())
        seen["thread"] = await asyncio.to_thread(lambda: current_boundary().id)
        return b.id

    async def _cur():
        return current_boundary().id

    async def judge(b, call_next):  # e.g. a verifier calling another LLM
        out = await call_next(b)
        if b.kind == "llm":
            await through("llm.judge", "judge", {"answer": out}, leaf, harness=h)
        return out

    h = Harness([judge])

    async def main():
        outer = await through("llm", "answer", {}, leaf, harness=h)
        explicit = await through("tool", "t", {}, leaf, harness=Harness(), parent_id="custom")
        return outer, explicit

    outer, explicit = asyncio.run(main())
    assert seen["leafs"] == [("llm", None), ("llm.judge", outer), ("tool", "custom")]
    assert seen["task"] == seen["thread"] == explicit
