"""Multi-agent store: a supervisor routes each message to a tool-using specialist.

What it teaches:

- tools are plain functions (type hints + docstring -> JSON Schema), shared
  once through ``ToolProvider`` and picked *by name* by each agent;
  ``use_tool`` adds a component-local tool (like ``use_callback``)
- a classifier agent (``use_module(..., arch="cot")`` + ``use_call``) and
  ReAct specialists (``use_agent``) are just components; "dispatch" is
  conditional rendering
- every input is a turn (``use_turn``); the app publishes its own events
  (``agent.route``, ``agent.answer``) next to the built-in ``llm.*`` /
  ``tool.*`` ones -- ``scripts/e2e_turns.py`` and the learning examples read them

Run from the repo root::

    uv run python -m examples.tools                 # terminal
    uv run python -m examples.main_web --app tools  # browser, http://127.0.0.1:8000

    > quais produtos vocês vendem?
    > quanto custa o frete do SHOE-RUN-42 para o BR?
    > :events agent  :usage  :turns  :q

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
~3-5 LM calls per question. The store speaks Portuguese: its routing data
(``SPECIALISTS``) is the dataset the learning examples optimize.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import dspy

from examples._devtools import run_repl
from examples._lm import example_models
from integrations import DSPyProvider, use_agent, use_call, use_module
from integrations.learn import use_learned
from pyreact.boot import create_root
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_emit, use_turn
from pyreact.tools import ToolProvider, use_tool

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"stdin": "quais produtos vocês vendem?\nquanto custa o frete do HAT-WOOL?\n:q\n"}


# --- Tools: plain functions; type hints + docstrings drive the JSON Schema. ---
CATALOG = {
    "shirt": ["SHIRT-RED", "SHIRT-BLUE"],
    "shoe": ["SHOE-RUN-42"],
    "hat": ["HAT-WOOL"],
}


def list_products() -> List[str]:
    """List every product SKU available in the catalog."""
    return [sku for skus in CATALOG.values() for sku in skus]


def search_products(query: str) -> List[str]:
    """Search the product catalog by keyword, returning matching SKUs.
    query: keywords to search for
    """
    hits: List[str] = []
    for word, skus in CATALOG.items():
        if word in query.lower():
            hits += skus
    return hits


def lookup_price(sku: str) -> float:
    """Return the unit price for a product SKU.
    sku: the product SKU, e.g. "SHIRT-RED"
    """
    prices = {"SHIRT-RED": 29.9, "SHIRT-BLUE": 29.9, "SHOE-RUN-42": 89.0, "HAT-WOOL": 19.5}
    return prices.get(sku, 0.0)


def estimate_shipping(sku: str, country: str) -> float:
    """Estimate shipping cost for a SKU to a destination country.
    sku: the product SKU
    country: ISO country code, e.g. "BR"
    """
    base = 5.0 if country.upper() == "BR" else 15.0
    return round(base + (0.5 if sku.startswith("SHOE") else 0.0), 2)


def check_stock(sku: str) -> int:
    """Return the number of units in stock for a SKU.
    sku: the product SKU
    """
    stock = {"SHIRT-RED": 12, "SHIRT-BLUE": 0, "SHOE-RUN-42": 3, "HAT-WOOL": 40}
    return stock.get(sku, 0)


def place_order(sku: str, quantity: int) -> str:
    """Place an order for a SKU and return a confirmation id.
    sku: the product SKU
    quantity: number of units to order
    """
    return f"ORDER-{sku}-{quantity}"


# --- Specialists: each is a tool-using agent over its own toolset. ---
# (Also the routing dataset of examples/optimize_router.py and examples/learn/.)
SPECIALISTS = [
    {
        "name": "catalog",
        "description": "Quais produtos existem, listar o catálogo, buscar produtos e consultar preços.",
        "utterances": [
            "quais produtos estão disponíveis",
            "liste os produtos",
            "o que vocês vendem",
            "qual o preço da camisa",
        ],
    },
    {
        "name": "shipping",
        "description": "Custo de frete e elegibilidade a frete grátis.",
        "utterances": [
            "quanto custa o frete",
            "tenho frete grátis",
            "qual o valor da entrega para o BR",
        ],
    },
    {
        "name": "order",
        "description": "Verificar estoque de um produto específico e finalizar pedidos.",
        "utterances": [
            "tem em estoque",
            "quero comprar 2 unidades",
            "fazer um pedido",
        ],
    },
]


class CatalogSig(dspy.Signature):
    """Help the customer find products and prices using the tools."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ShippingSig(dspy.Signature):
    """Answer shipping questions using the tools."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class OrderSig(dspy.Signature):
    """Handle stock checks and orders using the tools."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class RouteSpecialistSig(dspy.Signature):
    """Choose the best specialist for the message.

    Instructions:
    - Return ONLY the exact name of one of the provided specialists.
    - Consider each specialist's description and example utterances.
    - "quais produtos / listar / catálogo / preço" -> catalog.
    - "estoque de um item / comprar / pedido" -> order.
    - "frete / entrega / frete grátis" -> shipping.
    """

    message: str = dspy.InputField()
    specialists: List[dict] = dspy.InputField(
        description="Available specialists with description and utterances"
    )
    name: str = dspy.OutputField()


def use_publish(kind: str, payload: Optional[dict], deps: list) -> None:
    """Custom hook: publish an app event when ``deps`` change (``None`` payload
    = nothing to say). The event is attributed to the current turn."""
    emit = use_emit()

    def _pub():
        if payload is not None:
            emit(kind, payload, source="store")

    hooks.use_effect(_pub, deps)


def use_specialist(name: str, signature, tools: list, question: str, turn: int) -> list:
    """Custom hook shared by the specialists: a ReAct agent over ``tools`` that
    answers ``question`` once per ``turn``, publishes ``agent.answer`` once per
    settled call and returns what to render."""
    agent = use_agent(signature, tools=tools, model="reasoning")

    def ask():
        if question.strip():
            agent.run(question=question)

    hooks.use_effect(ask, [question, turn])

    settled = agent.ver > 0 and not agent.loading
    use_publish("agent.answer", {
        "specialist": name,
        "question": question,
        "answer": getattr(agent.result, "answer", None),
        "error": str(agent.error) if agent.error else None,
    } if settled else None, [agent.ver])

    if agent.loading:
        return [Message(key="loading", text="Pensando...", sender="system", message_type="info")]
    if agent.error is not None:
        return [Message(key=f"error-{agent.ver}", text=f"Erro: {agent.error}", sender="system",
                        message_type="error")]
    if agent.result is None:
        return []
    return [Message(key=f"answer-{agent.ver}", text=str(agent.result.answer), sender="assistant")]


@component
def CatalogAgent(question: str, turn: int = 0):
    tools = ["list_products", "search_products", "lookup_price"]  # ToolProvider names
    return use_specialist("catalog", CatalogSig, tools, question, turn)


@component
def ShippingAgent(question: str, turn: int = 0):
    # A local tool (cf. use_callback) alongside the provider tools.
    free_shipping = use_tool(
        lambda total: total >= 100,
        name="qualifies_for_free_shipping",
        description="True if an order total qualifies for free shipping.",
        deps=[],
    )
    tools = ["estimate_shipping", free_shipping]
    return use_specialist("shipping", ShippingSig, tools, question, turn)


@component
def OrderAgent(question: str, turn: int = 0):
    return use_specialist("order", OrderSig, ["check_stock", "place_order"], question, turn)


SPECIALIST_COMPONENTS = {
    "catalog": CatalogAgent,
    "shipping": ShippingAgent,
    "order": OrderAgent,
}


@component
def SupervisorAgent(*, message: str, turn: int, on_route: Callable[[str, str, int], None]):
    """Classifier agent (no tools): picks a specialist for ``message``.

    ``name="supervisor"`` gives the module a stable key -- the handle that
    optimizers and ``:optimize supervisor`` use. ``program=``: the learned
    program a ``LearnedProgram(key="supervisor", src=...)`` above provides
    (``None`` without one: the baseline); a new ``src`` is a re-render.
    """
    router = use_module(RouteSpecialistSig, arch="cot", name="supervisor",
                        program=use_learned("supervisor"))
    call = use_call(router, model="fast")

    def route():
        if message.strip():
            call.run(message=message, specialists=SPECIALISTS)

    hooks.use_effect(route, [message, turn])

    name = getattr(call.result, "name", None)
    use_publish("agent.route", {"specialist": name, "message": message} if name else None,
                [call.ver])

    def _emit():  # call.ver bumps once per settled call
        if name:
            on_route(name, message, call.ver)

    hooks.use_effect(_emit, [call.ver])

    if call.loading:
        return [Message(key="routing", text="Encaminhando para um especialista...",
                        sender="system", message_type="info")]
    if call.error is not None:
        return [Message(key=f"route-error-{call.ver}", text=f"Erro: {call.error}",
                        sender="system", message_type="error")]
    return []


@component
def StoreHome():
    # Each input is a Turn (id, index, text). A repeated identical message is a
    # new turn, so turn.index re-triggers the supervisor. Turns queue by
    # default: a burst of questions is answered in order.
    turn = use_turn()
    active, set_active = hooks.use_state(None)  # (specialist, question, ver)

    def on_route(name: str, message: str, ver: int):
        set_active((name.strip().strip("'\"").lower(), message, ver))

    children = [
        Message(key="welcome", text="Loja: pergunte sobre produtos, preços, frete ou pedidos.",
                sender="assistant", message_type="info"),
        SupervisorAgent(key="supervisor", message=turn.text if turn else "",
                        turn=turn.index if turn else 0, on_route=on_route),
    ]
    if active is not None and active[0] not in SPECIALIST_COMPONENTS:
        children.append(Message(key=f"unrouted-{active[2]}", sender="system", message_type="warning",
                                text=f"Nenhum especialista chamado {active[0]!r}."))
    elif active is not None:
        name, question, ver = active
        Specialist = SPECIALIST_COMPONENTS[name]  # dispatch = conditional rendering
        children += [
            Message(key=f"dispatch-{ver}", text=f"→ especialista: {name}", sender="system",
                    message_type="info"),
            Specialist(key=f"specialist-{name}", question=question, turn=ver),
        ]
    return children


@component
def App():
    # Shared tools injected once; specialists pick the ones they need by name.
    tools = [list_products, search_products, lookup_price, estimate_shipping,
             check_stock, place_order]
    return [ToolProvider(key="tools", tools=tools, children=[StoreHome(key="store")])]


@component
def Root(models=None):
    # models: {"default", "fast", "reasoning"} -> DSPyProvider (context).
    lms = hooks.use_memo(lambda: models or example_models(), [id(models)])
    return [DSPyProvider(key="dspy", models=lms, children=[App(key="app")])]


def main() -> None:
    with create_root(Root, session_id="store") as root:
        # Messages and errors are rendered from typed events (ConsoleSink);
        # here LLM and tool activity too. :events, :usage, :turns for more.
        run_repl(root, prompt="> ",
                 console_kinds=("message", "error", "llm.end", "llm.error", "tool.*"))


if __name__ == "__main__":
    main()
