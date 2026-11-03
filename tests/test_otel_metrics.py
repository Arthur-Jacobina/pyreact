import asyncio

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402

from pyreact.core.harness import Blocked, Harness, middleware, through  # noqa: E402
from pyreact_otel import otel_middleware  # noqa: E402


def points(reader):
    """{metric name: [(attributes, point)]}"""
    out = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out.setdefault(m.name, []).extend(
                    (dict(p.attributes), p) for p in m.data.data_points)
    return out


async def llm(b):
    b.attrs.update(model="openai/gpt-4o-mini", cost=0.002, module_key="supervisor",
                   usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    return "ok"


def test_boundary_metrics_values():
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])

    @middleware(kinds=["tool"])
    async def guard(b, call_next):
        if b.name == "rm":
            raise Blocked("no", by="guard")
        return await call_next(b)

    h = Harness([otel_middleware(meter_provider=mp), guard])

    async def boom(b):
        raise ValueError("x")

    async def turn(b):
        await through("llm", "m", {}, llm, harness=h)
        await through("llm", "m", {}, llm, harness=h)
        await through("tool", "lookup", {}, lambda b: asyncio.sleep(0), harness=h)
        with pytest.raises(Blocked):
            await through("tool", "rm", {}, lambda b: asyncio.sleep(0), harness=h)
        with pytest.raises(ValueError):
            await through("llm", "bad", {}, boom, harness=h)

    asyncio.run(through("turn", "App", {}, turn, harness=h))
    m = points(reader)

    tokens = {a["gen_ai.token.type"]: p for a, p in m["gen_ai.client.token.usage"]}
    assert tokens["input"].sum == 200 and tokens["input"].count == 2
    assert tokens["output"].sum == 40
    a_in = next(a for a, _ in m["gen_ai.client.token.usage"] if a["gen_ai.token.type"] == "input")
    assert a_in["gen_ai.request.model"] == "openai/gpt-4o-mini"
    assert a_in["gen_ai.provider.name"] == "openai" and a_in["gen_ai.operation.name"] == "chat"

    (cost_attrs, cost), = m["pyreact.llm.cost"]
    assert cost.value == pytest.approx(0.004) and cost_attrs["pyreact.module_key"] == "supervisor"

    ops = {(a["gen_ai.operation.name"], a.get("error.type")): p.count
           for a, p in m["gen_ai.client.operation.duration"]}
    assert ops[("chat", None)] == 2 and ops[("chat", "ValueError")] == 1
    assert ops[("execute_tool", None)] == 1 and ops[("execute_tool", "Blocked")] == 1

    (turn_attrs, turn_p), = m["pyreact.turn.duration"]
    assert turn_p.count == 1 and turn_p.sum > 0

    errors = {(a["pyreact.boundary.kind"], a["error.type"]): p.value for a, p in m["pyreact.boundary.errors"]}
    assert errors == {("tool", "Blocked"): 1, ("llm", "ValueError"): 1}
    kinds = {a["pyreact.boundary.kind"] for a, _ in m["pyreact.boundary.duration"]}
    assert kinds == {"turn", "llm", "tool"}
    mp.shutdown()


def test_no_meter_provider_no_metrics_recorder():
    # tracing-only middleware still works without any metrics
    assert asyncio.run(through("llm", "m", {}, llm, harness=Harness([otel_middleware()]))) == "ok"
