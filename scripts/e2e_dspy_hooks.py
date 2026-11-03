"""E2E: DSPy hooks layer against a real LM (``claude -p`` via ClaudeCLILM).

Proves:
  1. ``use_agent`` runs a ReAct agent whose tools come from ToolProvider;
     a deliberately slow *sync* tool (time.sleep(2)) runs in a worker thread,
     so the render loop stays responsive: a sibling component reacts to
     input ("ping N") while the tool is sleeping.
  2. ``use_stream`` falls back gracefully on an LM that doesn't stream.
  3. Every LM/tool call is visible as llm.* / tool.* events on the EventBus.

    uv run python scripts/e2e_dspy_hooks.py

~5-8 LLM calls total (3-15s each).
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy  # noqa: E402

from integrations import DSPyProvider, use_agent, use_predict, use_stream  # noqa: E402
from integrations.claude_cli_lm import CALLS, ClaudeCLILM  # noqa: E402
from pyreact.boot import bootstrap  # noqa: E402
from pyreact.core import EventBus, HookContext, component, hooks  # noqa: E402
from pyreact.input.bus import InputBus  # noqa: E402
from pyreact.tools import ToolProvider  # noqa: E402

TOOL_SLEEP = 2.0
PRICES = {"SHIRT-RED": 29.9, "HAT-WOOL": 19.5}


def lookup_price(sku: str) -> float:
    """Return the unit price for a product SKU (slow backend).
    sku: the product SKU, e.g. "SHIRT-RED"
    """
    time.sleep(TOOL_SLEEP)  # blocking I/O stand-in: must not freeze the app
    return PRICES.get(sku.upper(), 0.0)


def check_stock(sku: str) -> int:
    """Return the number of units in stock for a SKU.
    sku: the product SKU
    """
    return {"SHIRT-RED": 12, "HAT-WOOL": 40}.get(sku.upper(), 0)


class StoreSig(dspy.Signature):
    """Answer the customer's question using the tools. Be brief."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _on_submit(handler):
    bus = hooks.get_service("input_bus", InputBus)
    cb = hooks.use_callback(lambda e: e["type"] == "submit" and handler(e["value"]), [])
    hooks.use_effect(lambda: bus.subscribe(cb), [cb])


@component
def StoreAgent():
    q, set_q = hooks.use_state(None)
    _on_submit(lambda v: v.startswith("ask:") and set_q(v[4:].strip()))
    agent = use_agent(StoreSig, tools=["lookup_price", "check_stock"], max_iters=4)
    hooks.use_effect(lambda: q and agent.run(question=q), [q])

    def _done():
        if agent.ver:
            bus = HookContext.get_service("event_bus", EventBus)
            bus.publish("app.answer", {"answer": getattr(agent.result, "answer", None),
                                       "error": repr(agent.error) if agent.error else None,
                                       "usage": agent.usage})

    hooks.use_effect(_done, [agent.ver])
    return []


@component
def Echo():
    """Unrelated component: must keep reacting while the agent's tool sleeps."""
    n, set_n = hooks.use_state(None)
    _on_submit(lambda v: v.startswith("ping") and set_n(v))

    def _echo():
        if n:
            hooks_bus = HookContext.get_service("event_bus", EventBus)
            hooks_bus.publish("app.echo", {"ping": n})

    hooks.use_effect(_echo, [n])
    return []


@component
def Streamer():
    q, set_q = hooks.use_state(None)
    _on_submit(lambda v: v.startswith("stream:") and set_q(v[7:].strip()))
    mod = use_predict("question -> answer")
    s = use_stream(mod, listen=["answer"])
    hooks.use_effect(lambda: q and s.run(question=q), [q])

    def _done():
        if s.ver:
            HookContext.get_service("event_bus", EventBus).publish(
                "app.stream_done", {"text": s.text, "error": repr(s.error) if s.error else None})

    hooks.use_effect(_done, [s.ver])
    return []


LM = ClaudeCLILM("haiku")


@component
def App():
    return [
        DSPyProvider(key="dspy", lm=LM, children=[
            ToolProvider(key="tools", tools=[lookup_price, check_stock], children=[
                StoreAgent(key="agent"),
                Streamer(key="streamer"),
            ]),
            Echo(key="echo"),
        ])
    ]


def main() -> int:
    events, lock = [], threading.Lock()
    seen = {k: threading.Event() for k in ("tool.start", "app.answer", "app.stream_done")}
    t0 = time.perf_counter()

    def on_event(ev):
        with lock:
            events.append((time.perf_counter() - t0, ev))
        if ev.kind == "tool.start" and ev.payload.get("tool") == "lookup_price":
            seen["tool.start"].set()
        if ev.kind in seen and ev.kind != "tool.start":
            seen[ev.kind].set()

    HookContext.get_service("event_bus", EventBus).subscribe(on_event)
    app = bootstrap(App)
    ok = True
    try:
        app.invoke("ask: What is the price of SHIRT-RED, and is it in stock?")
        if not seen["tool.start"].wait(120):
            print("FAIL: agent never called lookup_price")
            return 1
        # The slow tool is now sleeping in a worker thread. Poke the app.
        ping_lat = []
        for i in range(5):
            sent = time.perf_counter()
            app.invoke(f"ping {i}")
            deadline = sent + 1.0
            while time.perf_counter() < deadline:
                with lock:
                    hit = [t for t, e in events if e.kind == "app.echo" and e.payload["ping"] == f"ping {i}"]
                if hit:
                    ping_lat.append(hit[0] - (sent - t0))
                    break
                time.sleep(0.005)
            else:
                ping_lat.append(None)
            time.sleep(0.2)

        if not seen["app.answer"].wait(180):
            print("FAIL: agent did not finish")
            return 1
        app.invoke("stream: Name one primary color. One word.")
        if not seen["app.stream_done"].wait(120):
            print("FAIL: stream did not finish")
            return 1
        app.wait_idle(timeout=30)
    finally:
        app.shutdown()

    # ---------------------------------------------------------------- report
    print("\n=== event timeline (s since start) ===")
    for t, ev in events:
        p = ev.payload
        if ev.kind == "llm.start":
            d = f"{p['module']} model={p['model']} call={p['call_id']}"
        elif ev.kind in ("llm.end", "llm.error"):
            u = p.get("usage") or {}
            d = f"{p['module']} call={p['call_id']} latency={p['latency']:.2f}s lm_calls={u.get('lm_calls')}"
            if ev.kind == "llm.error":
                d += f" error={p['error']}"
        elif ev.kind == "tool.start":
            d = f"{p['tool']}({p['args']}) call={p['call_id']}"
        elif ev.kind == "tool.end":
            d = f"{p['tool']} {p['duration']:.2f}s -> {p['result']!r} err={p['error']}"
        elif ev.kind == "app.echo":
            d = p["ping"]
        elif ev.kind == "llm.chunk":
            continue
        else:
            d = str(p)[:160]
        print(f"{t:7.2f}  {ev.kind:<16} {d}")

    tool_start = next(t for t, e in events if e.kind == "tool.start" and e.payload["tool"] == "lookup_price")
    tool_end = next((t for t, e in events if e.kind == "tool.end" and e.payload["tool"] == "lookup_price"), None)
    echoes_in_window = [t for t, e in events if e.kind == "app.echo" and tool_end and tool_start < t < tool_end]
    answer = next(e.payload for _, e in events if e.kind == "app.answer")
    stream = next(e.payload for _, e in events if e.kind == "app.stream_done")

    print("\n=== checks ===")
    print(f"ping latencies (s): {[round(x, 3) if x is not None else None for x in ping_lat]}")
    print(f"echoes handled while lookup_price slept ({tool_start:.2f}-{tool_end:.2f}s): {len(echoes_in_window)}")
    print(f"agent answer: {answer['answer']!r} error={answer['error']}")
    print(f"stream (non-streaming LM fallback) text: {stream['text']!r} error={stream['error']}")
    print(f"claude -p calls: {len(CALLS)} (total {sum(c['t'] for c in CALLS):.1f}s)")

    if any(x is None or x > 0.3 for x in ping_lat):
        print("FAIL: render loop was not responsive during the tool call")
        ok = False
    if len(echoes_in_window) < 3:
        print("FAIL: expected pings to be handled while the sync tool slept")
        ok = False
    if answer["error"] or not answer["answer"]:
        ok = False
        print("FAIL: agent errored")
    if stream["error"] or not stream["text"]:
        ok = False
        print("FAIL: stream errored")
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
