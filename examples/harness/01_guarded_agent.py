"""01 - A guarded, verified tool agent (and the trace of every decision).

A store clerk (a ReAct agent with three tools) answers an order request.
Around it, without touching the agent:

- ``HarnessProvider`` (the whole shop) -- guards, before each call:
    ``input_policy(PII, redact=True)``  emails/phones never reach the LM
    ``tool_allowlist(allow=[...])``     the clerk may look things up, not order
    ``budget(max_tokens_per_turn=...)`` once a turn has spent its tokens,
                                        later LLM calls in that turn are blocked
- ``use_verify`` (the clerk component only) -- verification, after each call:
    ``schema({...})`` + ``assert_that(...)``; on failure the call is re-run
    with the verifier's feedback as an extra ``feedback`` input
    (``retry_with_feedback``); still failing -> flagged, and published as a
    ``feedback`` event for the learning loop (``to_feedback=True``).

After the clerk answers, the shop asks for a one-line summary: a second LLM
call in the same turn, which the budget blocks (the clerk spent it).

    uv run python -m examples.harness.01_guarded_agent
    uv run python -m examples.harness.01_guarded_agent "price of 2 silk scarves?"

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
At most 10 LM calls (2 attempts x (4 ReAct steps + extract)); usually 5-8.
"""

from __future__ import annotations

import sys
from typing import Annotated, Literal

import dspy
from pydantic import Field

from examples._devtools import show_messages, trace
from examples._lm import example_lm
from integrations import DSPyProvider, use_agent, use_agent_call
from integrations.harness import (
    PII,
    assert_that,
    budget,
    input_policy,
    schema,
    tool_allowlist,
    use_verify,
)
from pyreact.boot import create_root
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_on_turn
from pyreact.core.harness import HarnessProvider
from pyreact.tools import ToolProvider

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": []}

REQUEST = ("Hi! I'd like 3 wool hats, please place the order and email the receipt to "
           "ann@example.com. How much will it be?")

CATALOG = {"HAT-WOOL": 25.0, "SCARF-SILK": 40.0, "GLOVES-LEATHER": 55.0}
STOCK = {"HAT-WOOL": 12, "SCARF-SILK": 0, "GLOVES-LEATHER": 4}


def lookup_price(sku: str) -> str:
    """Unit price of a product, in USD.
    sku: one of HAT-WOOL, SCARF-SILK, GLOVES-LEATHER
    """
    return f"{CATALOG[sku]:.2f}" if sku in CATALOG else f"unknown sku {sku!r}"


def check_stock(sku: str) -> str:
    """Units in stock.
    sku: one of HAT-WOOL, SCARF-SILK, GLOVES-LEATHER
    """
    return str(STOCK.get(sku, 0))


def place_order(sku: str, quantity: int) -> str:
    """Place an order (charges the customer).
    sku: the product
    quantity: units to order
    """
    return f"ordered {quantity} x {sku}"  # never runs: not on the allowlist


class OrderQuote(dspy.Signature):
    """You are a store clerk. Handle the customer's request using the tools."""

    request: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="reply to the customer")
    sku: str = dspy.OutputField(desc="the product's SKU")
    total: float = dspy.OutputField(desc="total price in USD")


# -- the harness ---------------------------------------------------------------
BUDGET = budget(max_tokens_per_turn=1500)
GUARDS = [
    input_policy(PII, redact=True),
    tool_allowlist(allow=["lookup_price", "check_stock"]),
    BUDGET,
]
CHECKS = [
    schema({"sku": Literal["HAT-WOOL", "SCARF-SILK", "GLOVES-LEATHER"],
            "total": Annotated[float, Field(ge=0)]}),
    assert_that(lambda p: str(p.answer).rstrip().endswith("Anything else?"),
                "End `answer` with the exact sentence 'Anything else?'.", name="sign-off"),
]


@component
def Clerk(on_answer):
    use_verify(CHECKS, on_fail="retry_with_feedback", max_attempts=2, on_exhausted="flag",
               to_feedback=True, name="quote-check")
    agent = use_agent(OrderQuote, tools=["lookup_price", "check_stock", "place_order"],
                      max_iters=4, name="clerk")
    use_on_turn(lambda turn: agent.run(request=turn.text))

    def settled():
        if agent.ver and not agent.loading:
            on_answer(agent)

    hooks.use_effect(settled, [agent.ver])
    return []


@component
def Shop():
    summary = use_agent_call("answer -> one_line_summary", name="summary")
    quote, set_quote = hooks.use_state(None)  # (n, result, error): the clerk's last answer

    def on_answer(agent):  # a callback prop: the clerk reports, the shop decides
        set_quote(lambda q: ((q[0] + 1) if q else 1, agent.result, agent.error))

    def summarize():
        if quote is not None and quote[2] is None:
            summary.run(answer=quote[1].answer)  # 2nd LLM call this turn: over budget

    hooks.use_effect(summarize, [quote and quote[0]])

    out = [Clerk(key="clerk", on_answer=on_answer)]
    if quote is not None:
        n, r, error = quote
        text = (f"clerk error: {error}" if error is not None
                else f"clerk: {r.answer}\n       sku={r.sku} total={r.total}")
        out.append(Message(key=f"quote-{n}", text=text, sender="assistant"))
    if summary.ver:
        text = (f"summary: not run ({type(summary.error).__name__})"
                if summary.error is not None
                else f"summary: {summary.result.one_line_summary}")
        out.append(Message(key=f"summary-{summary.ver}", text=text, sender="system",
                           message_type="info"))
    return out


LM = example_lm()


@component
def Root():
    return [DSPyProvider(key="dspy", lm=LM, children=[
        ToolProvider(key="tools", tools=[lookup_price, check_stock, place_order], children=[
            HarnessProvider(key="harness", middleware=GUARDS, children=[Shop(key="shop")]),
        ]),
    ])]


# -- the trace -------------------------------------------------------------------
def _short(v, n=110):
    s = " ".join(str(v).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def decision(ev):
    """Devtools: one line per harness decision (``None``: not shown)."""
    p, k = ev.payload, ev.kind
    if k == "llm.start":
        return f"  llm     {p['module']}: start"
    if k in ("llm.end", "llm.error"):
        u = p.get("usage") or {}
        end = "end" if k == "llm.end" else "error"
        return (f"  llm     {p['module']}: {end} ({u.get('lm_calls', 0)} LM calls, "
                f"{u.get('total_tokens', 0)} tokens)")
    if k == "tool.start":
        return f"  tool    {p['tool']}({_short(p['args'], 60)})"
    if k == "tool.end" and not p.get("error"):
        return f"  tool    {p['tool']} -> {_short(p['result'], 60)}"
    if k == "guard.redacted":
        return (f"  GUARD   {p['guard']}: redacted {p['count']} item(s) in {p['fields']} "
                f"of {p['kind']} {p['name']!r}")
    if k == "guard.blocked":
        return f"  GUARD   {p['guard']}: BLOCKED {p['kind']} {p['name']!r}: {p['reason']}"
    if k == "verify.passed":
        return f"  VERIFY  {p['verifier']}: passed (score {p['score']:g})"
    if k == "verify.failed":
        return (f"  VERIFY  {p['verifier']}: FAILED (score {p['score']:g}) -> {p['action']}: "
                f"{_short(p['feedback'])}")
    if k == "feedback":
        return f"  LEARN   feedback event for {p['target']}: score {p['score']:g}"
    return None


KINDS = ["llm.start", "llm.end", "llm.error", "tool.*", "guard.*", "verify.*", "feedback"]


def main(request: str) -> None:
    with create_root(Root, session_id="shop") as root:
        trace(root, KINDS, decision)  # the trace of every decision (devtools)
        show_messages(root)           # what the shop says (Message components)
        print(f"> {request}")
        tid = root.send(request, wait=True, timeout=600)
    spent = BUDGET.spent(tid)
    print(f"\nturn budget: {spent['tokens']} of {BUDGET.max_tokens} tokens spent")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or REQUEST)
