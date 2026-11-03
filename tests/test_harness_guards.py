"""Guard primitives (pyreact.harness.guards). No network."""

import asyncio

import pytest
from dspy.utils.dummies import DummyLM

from integrations import DSPyProvider, use_agent
from pyreact.core import EventBus, HookContext, component, runtime
from pyreact.core.harness import Blocked, Boundary, Harness, HarnessProvider
from pyreact.core.runtime import schedule_rerender
from pyreact.harness import (
    PII,
    budget,
    guard,
    input_policy,
    max_calls,
    rate_limit,
    require_confirmation,
    tool_allowlist,
)
from pyreact.tools import ToolProvider


def _events(kinds=("guard.*",)):
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, kinds)
    return got


def _run(h, kind="llm", name="m", inputs=None, turn_id="t1", fn=None, **attrs):
    async def ok(b):
        return dict(b.inputs)

    b = Boundary(kind=kind, name=name, inputs=dict(inputs or {}), turn_id=turn_id, attrs=attrs)
    return asyncio.run(h.run(b, fn or ok))


def test_guard_predicate_allows_blocks_and_names_the_guard():
    ev = _events()

    @guard(kinds="tool")
    def small_orders(b):
        if b.inputs.get("quantity", 0) > 10:
            return "orders above 10 units need a human"

    no = guard(lambda b: False, name="never")
    h = Harness([small_orders])
    assert _run(h, "tool", "place_order", {"quantity": 2}) == {"quantity": 2}
    assert _run(h, "llm", "x", {"quantity": 99}) == {"quantity": 99}  # kind filter
    with pytest.raises(Blocked) as exc:
        _run(h, "tool", "place_order", {"quantity": 50})
    assert exc.value.by == "small_orders"
    assert str(exc.value) == "tool 'place_order' blocked by small_orders: orders above 10 units need a human"
    with pytest.raises(Blocked, match="refused by never"):
        _run(Harness([no]))
    assert [e.payload["guard"] for e in ev] == ["small_orders", "never"]
    p = ev[0].payload
    assert p["reason"] == "orders above 10 units need a human" and p["kind"] == "tool"
    assert p["name"] == "place_order" and p["boundary_id"].startswith("b")
    assert ev[0].turn_id == "t1"


def test_async_guard_and_bad_return_type():
    async def slow(b):
        await asyncio.sleep(0)
        return b.name != "bad"

    h = Harness([guard(slow)])
    assert _run(h, name="good") == {}
    with pytest.raises(Blocked):
        _run(h, name="bad")
    with pytest.raises(TypeError, match="returns None/True"):
        _run(Harness([guard(lambda b: 3, name="weird")]))


def test_tool_allowlist_globs_and_deny_wins():
    g = tool_allowlist(allow=["search_*", "lookup_price"], deny=["search_admin"])
    h = Harness([g])
    assert _run(h, "tool", "search_docs") == {}
    assert _run(h, "llm", "anything") == {}  # tools only
    with pytest.raises(Blocked, match="'search_admin' is denied"):
        _run(h, "tool", "search_admin")
    with pytest.raises(Blocked, match="not allowed; allowed: search_\\*, lookup_price"):
        _run(h, "tool", "place_order")


def test_max_calls_per_turn_ignores_retries_and_resets_per_turn():
    g = max_calls(per_turn=2, kind="llm")

    async def retry(b, call_next):
        await call_next(b)
        b.attempt += 1
        return await call_next(b)

    h = Harness([retry, g])
    _run(h, turn_id="t1")
    _run(h, turn_id="t1")  # each call retried once: still 2 calls
    with pytest.raises(Blocked, match="limit of 2 llm call\\(s\\) per turn"):
        _run(h, turn_id="t1")
    assert g.count("t1") == 2
    _run(h, turn_id="t2")
    assert g.count("t2") == 1


def test_budget_accounts_usage_after_calls_and_blocks_the_rest_of_the_turn():
    ev = _events()
    g = budget(max_tokens_per_turn=100, max_cost_per_turn=1.0)

    def spend(tokens, cost=0.0, attempts=1):
        async def fn(b):
            # like invoke_module: usage attrs are cumulative across attempts
            n = b.attrs.get("_n", 0) + 1
            b.attrs["_n"] = n
            b.attrs["usage"] = {"total_tokens": tokens * n}
            b.attrs["cost"] = cost * n
            return "ok"

        return fn

    async def retry(b, call_next):  # inside the budget: re-reports cumulative usage
        await call_next(b)
        return await call_next(b)

    h = Harness([g])
    _run(h, fn=spend(60))
    assert g.spent("t1") == {"tokens": 60, "cost": 0.0}
    _run(h, fn=spend(60))  # allowed: the budget was not exhausted *before* it
    assert g.spent("t1")["tokens"] == 120
    with pytest.raises(Blocked, match="turn token budget exhausted: 120 of 100"):
        _run(h, fn=spend(1))
    _run(h, turn_id="t2", fn=spend(10))  # a new turn has a new budget
    # counted once across retries inside the guard (deltas, not totals)
    h2 = Harness([g, retry])
    _run(h2, turn_id="t3", fn=spend(30))
    assert g.spent("t3")["tokens"] == 60
    _run(Harness([g]), turn_id="t4", fn=spend(0, cost=2.0))
    with pytest.raises(Blocked, match="turn cost budget exhausted"):
        _run(Harness([g]), turn_id="t4")
    assert [e.payload["guard"] for e in ev] == ["budget", "budget"]
    with pytest.raises(ValueError):
        budget()


def test_rate_limit_blocks_or_waits():
    g = rate_limit(2, per_seconds=10)
    now = [0.0]
    g.clock = lambda: now[0]
    h = Harness([g])
    _run(h)
    _run(h)
    with pytest.raises(Blocked, match="rate limit of 2 call\\(s\\) per 10s reached"):
        _run(h)
    now[0] = 10.5  # the window slid
    _run(h)

    w = rate_limit(1, per_seconds=0.05, wait=True)
    h = Harness([w])

    async def two():
        async def ok(b):
            return 1

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await h.run(Boundary(kind="llm", name="a"), ok)
        await h.run(Boundary(kind="llm", name="b"), ok)
        return loop.time() - t0

    assert asyncio.run(two()) >= 0.04  # second call waited for a slot


def test_input_policy_blocks_or_redacts():
    ev = _events()
    h = Harness([input_policy(PII)])
    with pytest.raises(Blocked, match="input 'q' contains email"):
        _run(h, inputs={"q": "mail me at ann@example.com"})
    assert _run(h, inputs={"q": "hello"}) == {"q": "hello"}

    h = Harness([input_policy(PII, redact=True)])
    caller = {"q": "ann@example.com or 555-123-4567", "n": 3, "notes": ["ssn 123-45-6789"]}
    got = _run(h, inputs=caller)
    assert got == {"q": "[EMAIL] or [PHONE]", "n": 3, "notes": ["ssn [SSN]"]}
    assert caller["q"].startswith("ann@")  # the caller's dict is untouched
    red = [e for e in ev if e.kind == "guard.redacted"][0].payload
    assert red["fields"] == ["notes", "q"] and red["count"] == 3

    h = Harness([input_policy(r"secret-\d+", redact=True, fields=["a"])])
    assert _run(h, inputs={"a": "x secret-1", "b": "secret-2"}) == {"a": "x [REDACTED]",
                                                                    "b": "secret-2"}
    h = Harness([input_policy(lambda s: "no shouting" if s.isupper() else None)])
    with pytest.raises(Blocked, match="input 'q' no shouting"):
        _run(h, inputs={"q": "HELLO"})
    h = Harness([input_policy(str.lower, redact=True)])
    assert _run(h, inputs={"q": "HeLLo"}) == {"q": "hello"}


def test_require_confirmation_sync_async_when_and_timeout():
    ev = _events()
    asked = []

    def ask(b):
        asked.append(b.name)
        return b.inputs.get("amount", 0) < 100 or "amount too large for auto-approval"

    g = require_confirmation("tool", ask=ask, when=lambda b: b.name != "read_only")
    h = Harness([g])
    assert _run(h, "tool", "pay", {"amount": 5}) == {"amount": 5}
    assert _run(h, "tool", "read_only") == {}
    with pytest.raises(Blocked, match="amount too large"):
        _run(h, "tool", "pay", {"amount": 500})
    assert asked == ["pay", "pay"]
    assert [e.kind for e in ev] == ["guard.confirmed", "guard.blocked"]

    async def never(b):
        await asyncio.sleep(10)

    h = Harness([require_confirmation("tool", ask=never, timeout=0.01)])
    with pytest.raises(Blocked, match="no confirmation within 0.01s"):
        _run(h, "tool", "pay")
    h = Harness([require_confirmation("tool", ask=lambda b: False)])
    with pytest.raises(Blocked, match="not confirmed"):
        _run(h, "tool", "pay")


def lookup_price(sku: str) -> str:
    """Look up a price.
    sku: the sku
    """
    return "10"


def place_order(sku: str) -> str:
    """Place an order.
    sku: the sku
    """
    return "ordered"


def test_tool_allowlist_inside_a_react_agent_lets_the_agent_adapt():
    lm = DummyLM([
        {"next_thought": "order", "next_tool_name": "place_order", "next_tool_args": {"sku": "A"}},
        {"next_thought": "look", "next_tool_name": "lookup_price", "next_tool_args": {"sku": "A"}},
        {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "ok", "answer": "10"},
    ])
    ev = _events(["guard.*", "tool.*"])
    hs = {}

    @component
    def Agent():
        hs["call"] = use_agent("question -> answer", tools=["lookup_price", "place_order"],
                               max_iters=4)
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[
            ToolProvider(key="t", tools=[lookup_price, place_order], children=[
                HarnessProvider(key="h", middleware=[tool_allowlist(allow=["lookup_*"])],
                                children=[Agent(key="a")])
            ])
        ])]

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await runtime.run_until_idle()
        hs["call"].run(question="buy A")
        await runtime.run_until_idle()

    asyncio.run(main())
    call = hs["call"]
    assert call.error is None and call.result.answer == "10"
    traj = call.result.trajectory
    assert "not allowed; allowed: lookup_*" in traj["observation_0"]
    assert traj["observation_1"] == "10"
    kinds = [e.kind for e in ev]
    assert "guard.blocked" in kinds and "tool.blocked" in kinds
    blocked = [e for e in ev if e.kind == "guard.blocked"][0]
    assert blocked.payload["call_id"] is None or isinstance(blocked.payload["call_id"], str)
    assert blocked.payload["component"].endswith("/Agent[a]")
