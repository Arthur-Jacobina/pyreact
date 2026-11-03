import asyncio
import contextvars

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from pyreact.core.harness import Blocked, Harness, install, middleware, through  # noqa: E402
from pyreact_otel import otel_middleware, provider_of  # noqa: E402


@pytest.fixture
def otel():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def harness(*inner, **kw):
        return Harness([otel_middleware(provider, **kw), *inner])

    harness.exporter = exporter
    harness.provider = provider
    yield harness
    provider.shutdown()


def by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


USAGE = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}


async def fake_llm(b):
    b.attrs.update(model="openai/gpt-4o-mini", usage=USAGE, cost=0.0012, call_id="llm-7",
                   lm_calls=[{"messages": [{"role": "user", "content": "secret prompt"}],
                              "completion": "secret answer"}])
    return {"answer": "secret answer"}


def test_span_tree_turn_llm_tool_across_tasks_and_threads(otel):
    h = otel()
    seen = {}

    async def tool(b):
        # asyncio.to_thread copies the context: the tool span is current there
        seen["thread_span"] = await asyncio.to_thread(trace.get_current_span)
        return 42

    async def llm(b):
        await fake_llm(b)
        return await through("tool", "lookup_price", {"sku": "X"}, tool, harness=h)

    async def turn(b):
        # a child task inherits the OTel context of the turn span
        return await asyncio.create_task(through("llm", "supervisor", {"q": "hi"}, llm, harness=h))

    async def main():
        return await through("turn", "StoreHome", {"text": "hi"}, turn, harness=h)

    assert asyncio.run(main()) == 42
    spans = by_name(otel.exporter)
    t, c, x = spans["pyreact.turn"], spans["chat openai/gpt-4o-mini"], spans["execute_tool lookup_price"]
    assert t.parent is None
    assert c.parent.span_id == t.context.span_id and x.parent.span_id == c.context.span_id
    assert len({s.context.trace_id for s in (t, c, x)}) == 1
    assert seen["thread_span"].get_span_context().span_id == x.context.span_id
    assert t.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert x.attributes["gen_ai.operation.name"] == "execute_tool"
    assert x.attributes["gen_ai.tool.name"] == "lookup_price"
    assert t.kind == trace.SpanKind.INTERNAL and c.kind == trace.SpanKind.CLIENT


def test_llm_genai_attributes(otel):
    h = otel()
    asyncio.run(through("llm", "supervisor", {"q": "hi"}, fake_llm, harness=h,
                        module_key="supervisor", arch="cot"))
    (s,) = otel.exporter.get_finished_spans()
    a = s.attributes
    assert s.name == "chat openai/gpt-4o-mini"  # renamed once the model is known
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == a["gen_ai.system"] == "openai"
    assert a["gen_ai.request.model"] == "openai/gpt-4o-mini"
    assert a["gen_ai.usage.input_tokens"] == 120 and a["gen_ai.usage.output_tokens"] == 30
    assert a["pyreact.usage.total_tokens"] == 150 and a["pyreact.cost"] == pytest.approx(0.0012)
    assert a["pyreact.call_id"] == "llm-7" and a["pyreact.lm_calls"] == 1
    assert a["pyreact.module_key"] == "supervisor" and a["pyreact.arch"] == "cot"
    assert a["pyreact.attempt"] == 1 and a["pyreact.boundary.kind"] == "llm"
    assert s.status.status_code == StatusCode.UNSET


def test_content_capture_off_by_default_and_size_limited(otel):
    asyncio.run(through("llm", "m", {"q": "secret input"}, fake_llm, harness=otel()))
    (s,) = otel.exporter.get_finished_spans()
    dumped = repr(dict(s.attributes))
    assert "secret" not in dumped
    assert not any(k in s.attributes for k in ("gen_ai.input.messages", "pyreact.inputs"))

    otel.exporter.clear()
    h = otel(capture_content=True, max_content_length=200)

    async def big_tool(b):
        return "x" * 1000

    async def main():
        await through("llm", "m", {"q": "secret input"}, fake_llm, harness=h)
        await through("tool", "t", {"a": 1}, big_tool, harness=h)

    asyncio.run(main())
    spans = by_name(otel.exporter)
    a = spans["chat openai/gpt-4o-mini"].attributes
    assert "secret prompt" in a["gen_ai.input.messages"]
    assert '"role": "assistant"' in a["gen_ai.output.messages"]
    assert "secret input" in a["pyreact.inputs"]
    t = spans["execute_tool t"].attributes
    assert t["gen_ai.tool.call.arguments"] == '{"a": 1}'
    assert t["gen_ai.tool.call.result"] == "x" * 200 + "...[truncated 800 chars]"


def test_error_and_blocked_status(otel):
    @middleware(kinds=["tool"])
    async def guard(b, call_next):
        if b.name == "rm":
            raise Blocked("rm is not allowed", boundary=b, by="guard")
        return await call_next(b)

    h = otel(guard)

    async def boom(b):
        raise ValueError("bad input")

    async def main():
        with pytest.raises(ValueError):
            await through("llm", "m", {}, boom, harness=h)
        with pytest.raises(Blocked):
            await through("tool", "rm", {}, boom, harness=h)

    asyncio.run(main())
    spans = by_name(otel.exporter)
    err, blk = spans["chat m"], spans["execute_tool rm"]
    assert err.status.status_code == StatusCode.ERROR and "bad input" in err.status.description
    assert err.attributes["error.type"] == "ValueError"
    assert [e.name for e in err.events] == ["exception"]
    assert blk.status.status_code == StatusCode.ERROR
    assert blk.attributes["pyreact.blocked.reason"] == "rm is not allowed"
    assert blk.attributes["pyreact.blocked.by"] == "guard"
    assert blk.attributes["error.type"] == "Blocked"
    assert blk.events == ()  # a refusal is not an exception


def test_cancelled_is_not_an_error(otel):
    h = otel()

    async def slow(b):
        await asyncio.sleep(10)

    async def main():
        task = asyncio.create_task(through("llm", "m", {}, slow, harness=h))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    (s,) = otel.exporter.get_finished_spans()
    assert s.attributes["pyreact.cancelled"] is True
    assert s.status.status_code == StatusCode.UNSET


def test_retry_attempt_is_recorded_on_one_span(otel):
    async def retry(b, call_next):
        try:
            return await call_next(b)
        except RuntimeError:
            b.attempt += 1
            return await call_next(b)

    async def flaky(b):
        if b.attempt == 1:
            raise RuntimeError("transient")
        return "ok"

    assert asyncio.run(through("llm", "m", {}, flaky, harness=otel(retry))) == "ok"
    (s,) = otel.exporter.get_finished_spans()
    assert s.attributes["pyreact.attempt"] == 2 and s.status.status_code == StatusCode.UNSET


def test_fallback_parents_when_context_does_not_flow(otel):
    """A call in a task created *outside* the turn's context still nests under
    the turn (same session/turn), or under an explicit ``parent_id``."""
    h = otel()

    async def main():
        started, release = asyncio.Event(), asyncio.Event()

        async def turn(b):
            started.set()
            await release.wait()

        async def outsider():
            await started.wait()
            from pyreact.core.harness import Boundary

            orphan = Boundary(kind="llm", name="effect", session_id="s1", turn_id="s1:t1")
            await h.run(orphan, fake_llm)
            child = Boundary(kind="tool", name="explicit", parent_id=turn_b.id)
            await h.run(child, lambda b: asyncio.sleep(0))
            release.set()

        from pyreact.core.harness import Boundary

        turn_b = Boundary(kind="turn", name="App", session_id="s1", turn_id="s1:t1")
        # empty context: no OTel span flows into the outsider task
        other = asyncio.create_task(outsider(), context=contextvars.Context())
        await h.run(turn_b, turn)
        await other

    asyncio.run(main())
    spans = by_name(otel.exporter)
    t = spans["pyreact.turn"]
    assert spans["chat openai/gpt-4o-mini"].parent.span_id == t.context.span_id
    assert spans["execute_tool explicit"].parent.span_id == t.context.span_id
    assert t.attributes["pyreact.session_id"] == "s1" and t.attributes["session.id"] == "s1"


def test_enrichers_add_vendor_attributes(otel):
    seen = []

    def enrich(info):
        seen.append(info)
        return {"vendor.kind": info.boundary.kind, "vendor.blocked": info.blocked}

    asyncio.run(through("tool", "t", {}, lambda b: asyncio.sleep(0), harness=otel(enrichers=[enrich])))
    (s,) = otel.exporter.get_finished_spans()
    assert s.attributes["vendor.kind"] == "tool" and s.attributes["vendor.blocked"] is False
    assert seen[0].attributes["gen_ai.tool.name"] == "t" and seen[0].content == {}


def test_sandbox_and_custom_kinds(otel):
    async def main():
        await through("sandbox.exec", "python", {}, lambda b: asyncio.sleep(0), harness=otel())
        await through("retrieval", "docs", {}, lambda b: asyncio.sleep(0), harness=otel())

    asyncio.run(main())
    assert set(by_name(otel.exporter)) == {"sandbox.exec python", "retrieval docs"}


def test_provider_of():
    assert provider_of("openai/gpt-4o") == "openai"
    assert provider_of("gpt-4o-mini") == "openai"
    assert provider_of("anthropic/claude-sonnet-4") == "anthropic"
    assert provider_of("claude-cli/haiku") == "anthropic"
    assert provider_of("bedrock/anthropic.claude-3") == "aws.bedrock"
    assert provider_of(None) == "_OTHER"
    assert provider_of("whatever", "my-vendor") == "my-vendor"


def test_installed_process_wide_with_real_session_and_turn():
    """Through the real runtime: component / session / turn land on the span."""
    from pyreact.boot import bootstrap
    from pyreact.core import component, hooks, use_turn

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    uninstall = install(otel_middleware(provider))

    @component
    def Agent():
        turn = use_turn()

        async def act():
            await through("llm", "answer", {"q": turn.text}, fake_llm)

        hooks.use_effect(lambda: act() if turn else None, [turn and turn.id])
        return []

    app = bootstrap(Agent, default_session=False)
    try:
        app.open_session("sess-1").invoke("hello", wait=True, timeout=5)
    finally:
        app.shutdown()
        uninstall()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    s, t = spans["chat openai/gpt-4o-mini"], spans["pyreact.turn"]
    assert s.attributes["pyreact.component"] == "Agent"
    assert s.attributes["pyreact.session_id"] == "sess-1"
    assert s.attributes["pyreact.turn_id"].endswith("t1")
    # the runtime runs each turn as a "turn" boundary: the root of the trace
    assert t.parent is None and s.parent.span_id == t.context.span_id
    assert t.attributes["pyreact.turn.status"] == "done"
    assert t.attributes["pyreact.turn.index"] == 1
    assert t.attributes["pyreact.session_id"] == "sess-1"


def test_enrichers_see_the_trace_root_name(otel):
    seen = {}

    def enrich(info):
        seen[info.boundary.kind] = (info.trace_name, info.is_root)

    h = otel(enrichers=[enrich])

    async def turn(b):
        await through("llm", "m", {}, fake_llm, harness=h)

    asyncio.run(through("turn", "StoreHome", {}, turn, harness=h))
    assert seen == {"turn": ("pyreact.turn StoreHome", True),
                    "llm": ("pyreact.turn StoreHome", False)}


def test_turn_status_error_marks_the_span(otel):
    names = []
    h = otel(enrichers=[lambda info: names.append(info.trace_name)])

    async def turn(b):  # what pyreact.core.turns does: errors in spawned work
        b.attrs.update(status="error", duration=0.1)
        return "error"

    asyncio.run(through("turn", "turn", {"text": "x"}, turn, harness=h,
                        component="Root", index=3, source="web"))
    (t,) = otel.exporter.get_finished_spans()
    assert t.status.status_code == StatusCode.ERROR and t.attributes["error.type"] == "TurnError"
    assert t.attributes["pyreact.turn.status"] == "error"
    assert t.attributes["pyreact.turn.index"] == 3 and t.attributes["pyreact.turn.source"] == "web"
    assert names == ["pyreact.turn"]  # name "turn" + no component -> plain
