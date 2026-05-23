from typing import Callable, List
from integrations.dspy_integration import DSPyProvider
from integrations.dspy_agent import use_dspy_agent
from integrations.use_dspy import use_dspy_call
from integrations.dspy_integration import use_dspy_module
from .message import Message
from pyreact.components.keystroke import Keystroke
from pyreact.core.core import component, hooks
from pyreact.tools import ToolProvider, use_tool
import dspy
import os
import dotenv


dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


def _render_result(result, loading, error):
    # Shared loading/error/result rendering for the specialist agents.
    if loading:
        return [Message(key="loading", text="Pensando...", sender="system", message_type="info")]
    if error:
        return [Message(key="error", text=f"Erro: {error}", sender="system", message_type="error")]
    if result[0] is None:
        return []
    return [
        Message(
            key="answer",
            text=f"{getattr(result[0], 'answer', None)}",
            sender="assistant",
            message_type="chat",
        )
    ]


@component
def CatalogAgent(question: str):
    call, result, loading, error = use_dspy_agent(
        CatalogSig,
        tools=["list_products", "search_products", "lookup_price"],
        model="reasoning",
    )
    hooks.use_effect(lambda: call(question=question) if question.strip() else None, [question])
    return _render_result(result, loading, error)


@component
def ShippingAgent(question: str):
    # A local tool (cf. use_callback) alongside the provider tools.
    free_shipping = use_tool(
        lambda total: total >= 100,
        name="qualifies_for_free_shipping",
        description="True if an order total qualifies for free shipping.",
        deps=[],
    )
    call, result, loading, error = use_dspy_agent(
        ShippingSig, tools=["estimate_shipping", free_shipping], model="reasoning"
    )
    hooks.use_effect(lambda: call(question=question) if question.strip() else None, [question])
    return _render_result(result, loading, error)


@component
def OrderAgent(question: str):
    call, result, loading, error = use_dspy_agent(
        OrderSig, tools=["check_stock", "place_order"], model="reasoning"
    )
    hooks.use_effect(lambda: call(question=question) if question.strip() else None, [question])
    return _render_result(result, loading, error)


SPECIALIST_COMPONENTS = {
    "catalog": CatalogAgent,
    "shipping": ShippingAgent,
    "order": OrderAgent,
}


@component
def SupervisorAgent(*, message: str, on_route: Callable[[str, str, int], None]):
    # Classifier agent (no tools) that picks a specialist, like RouterAgent.
    choose_mod = use_dspy_module(RouteSpecialistSig, dspy.ChainOfThought, name="supervisor")
    call_llm, llm_result, llm_loading, _error = use_dspy_call(choose_mod, model="fast")

    def _decide():
        if not message.strip():
            return
        call_llm(message=message, specialists=SPECIALISTS)

    hooks.use_effect(_decide, [message])

    def _emit():
        if llm_result is None or llm_result[0] is None:
            return
        name = getattr(llm_result[0], "name", None)
        on_route(name, message, llm_result[1])

    hooks.use_effect(_emit, [llm_result])

    if llm_loading:
        return [Message(key="routing", text="Encaminhando para um especialista...", sender="system")]
    return []


@component
def StoreHome():
    last_message, set_last_message = hooks.use_state("")
    active, set_active = hooks.use_state(None)  # (name, question)
    handled_ver, set_handled_ver = hooks.use_state(-1)

    def on_enter(line: str):
        if line.strip():
            set_last_message(line)

    def on_route(name: str, message: str, ver: int):
        if ver == handled_ver or name not in SPECIALIST_COMPONENTS:
            return
        set_handled_ver(ver)
        set_active((name, message))

    children = [
        Message(
            key="welcome",
            text="Loja: pergunte sobre produtos, preços, frete ou pedidos.",
            sender="assistant",
            message_type="info",
        ),
        Keystroke(key="store_input", on_submit=on_enter),
        SupervisorAgent(key="supervisor", message=last_message, on_route=on_route),
    ]

    if active is not None:
        name, question = active
        Specialist = SPECIALIST_COMPONENTS[name]
        children.append(
            Message(
                key="dispatch",
                text=f"→ especialista: {name}",
                sender="system",
                message_type="info",
            )
        )
        children.append(Specialist(key=f"specialist-{name}", question=question))

    return children


@component
def App():
    # Shared tools injected once; specialists pick the ones they need by name.
    return [
        ToolProvider(
            key="tools",
            tools=[
                list_products,
                search_products,
                lookup_price,
                estimate_shipping,
                check_stock,
                place_order,
            ],
            children=[StoreHome(key="store")],
        )
    ]


@component
def Root(models=None):
    if models is None:
        lm_default = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY)
        lm_fast = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
        models = {
            "default": lm_default,
            "fast": lm_fast,
            "reasoning": lm_default,
        }
    return [DSPyProvider(key="dspy", models=models, children=[App(key="app")])]
