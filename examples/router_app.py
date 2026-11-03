"""Routing: pages are routes, and an LLM turns free text into a navigation.

What it teaches:

- ``Router`` / ``Route`` (``:params``, ``/*`` catch-all, catalog metadata:
  ``name``, ``description``, ``utterances``, ``default_params``)
- pages read the location from router context: ``use_route_params``,
  ``use_query_params``, ``use_location``, ``use_routes_catalog``
- ``use_navigate`` from outside the Router: a ``Navigator`` agent maps each
  turn ("show me reviews of the wool hat") to ``path + params + query`` with
  ``use_agent_call(..., arch="cot")`` and navigates

Run from the repo root::

    uv run python -m examples.router_app                     # terminal
    uv run python -m examples.main_web --app router_app      # browser

    > show me the reviews of the running shoe
    > what is this app about?
    > :nav /products/HAT-WOOL?tab=specs      (navigate by hand)
    > :route   :events nav   :q

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
1 LM call per input.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Literal

import dspy

from examples._lm import example_models
from integrations import DSPyProvider, use_agent_call
from pyreact.boot import bootstrap, read_terminal_and_invoke
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_turn
from pyreact.router import (
    Route,
    Router,
    use_location,
    use_navigate,
    use_query_params,
    use_route_params,
    use_routes_catalog,
)

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"stdin": "reviews of the running shoe\nabout this app\n:nav /products/HAT-WOOL?tab=specs\n"
                  ":route\n:nav /nowhere\n:q\n"}

PRODUCTS = {
    "SHOE-RUN-42": {"name": "running shoe", "overview": "Light trail shoe, size 42.",
                    "reviews": "4.6/5 - 'great grip'", "specs": "280 g, 8 mm drop"},
    "HAT-WOOL": {"name": "wool hat", "overview": "Merino beanie.",
                 "reviews": "4.8/5 - 'warm, no itch'", "specs": "100% merino, one size"},
}

# The route table: rendered as <Route>s *and* shown to the Navigator agent.
ROUTES: List[Dict] = [
    {"path": "/", "name": "home", "description": "Start page: what you can do here.",
     "utterances": ["home", "start over", "go back"]},
    {"path": "/products/:sku", "name": "product",
     "description": f"A product page. sku is one of {sorted(PRODUCTS)}; "
                    "query tab = overview | reviews | specs.",
     "utterances": ["show the wool hat", "reviews of the shoe", "specs of SHOE-RUN-42"],
     "default_params": {"sku": "SHOE-RUN-42"}},
    {"path": "/about", "name": "about", "description": "About this demo app.",
     "utterances": ["about", "what is this"]},
]
PATHS = tuple(r["path"] for r in ROUTES)


class NavigateSig(dspy.Signature):
    """Pick the page the user wants. Return one route path exactly as listed,
    values for its :params, and query parameters the user asked for."""

    message: str = dspy.InputField()
    routes: List[dict] = dspy.InputField(desc="path, description, example utterances")
    current_url: str = dspy.InputField()
    path: Literal[PATHS] = dspy.OutputField()  # type: ignore[valid-type]
    params: Dict[str, str] = dspy.OutputField(desc='e.g. {"sku": "HAT-WOOL"}; {} if none')
    query: Dict[str, str] = dspy.OutputField(desc='e.g. {"tab": "reviews"}; {} if none')


# --- Pages: plain views of the location (router context), no input handling. ---
@component
def Home():
    catalog = use_routes_catalog()  # every declared <Route>, with its metadata
    pages = ", ".join(f"{r['name']} ({r['path']})" for r in catalog if r.get("description"))
    came_from = use_query_params().get("from")
    hello = f"Home (back from {came_from}). " if came_from else "Home. "
    return [Message(key="home", text=f"{hello}Pages: {pages}. Ask for one in plain words.",
                    sender="assistant", message_type="info")]


@component
def Product():
    sku = use_route_params().get("sku", "")
    tab = use_query_params().get("tab", "overview")
    product = PRODUCTS.get(sku)
    if product is None:
        return [Message(key="product", text=f"No product {sku!r}.", sender="assistant",
                        message_type="warning")]
    body = product.get(tab, product["overview"])
    return [Message(key="product", text=f"[{product['name']} / {tab}] {body}",
                    sender="assistant")]


@component
def About():
    url = use_location()["url"]  # {"url", "path", "query", "fragment"}
    return [Message(key="about", sender="assistant",
                    text=f"About ({url}): a pyreact demo where routes are pages and an "
                         "LLM picks the route.")]


@component
def NotFound():
    path = use_location()["path"]
    navigate = use_navigate()
    # Redirect after commit (effects run post-render), like <Navigate to="/" />.
    hooks.use_effect(lambda: navigate("/", query={"from": path}), [path])
    return [Message(key="404", text=f"404: {path} -> redirecting home", sender="system",
                    message_type="warning")]


PAGES = {"home": Home, "product": Product, "about": About}


# --- The agent: one input (turn) -> one navigation. ---
@component
def Navigator():
    turn = use_turn()
    navigate = use_navigate()  # works outside <Router> (NavService)
    location = use_location()
    call = use_agent_call(NavigateSig, arch="cot", model="fast")

    def decide():
        if turn is not None and turn.text.strip():
            call.run(message=turn.text, current_url=location["url"],
                     routes=[{k: r[k] for k in ("path", "description", "utterances")}
                             for r in ROUTES])

    hooks.use_effect(decide, [turn and turn.id])

    def go():  # once per settled call
        r = call.result
        if r is None or r.path not in PATHS:
            return
        defaults = next(x for x in ROUTES if x["path"] == r.path).get("default_params", {})
        navigate(r.path, params={**defaults, **(r.params or {})}, query=r.query or None)

    hooks.use_effect(go, [call.ver])

    if call.loading:
        return [Message(key="thinking", text="choosing a page...", sender="system",
                        message_type="info")]
    if call.error is not None:
        return [Message(key=f"err-{call.ver}", text=f"navigator failed: {call.error}",
                        sender="system", message_type="error")]
    if call.result is not None:
        r = call.result
        return [Message(key=f"why-{call.ver}", sender="system", message_type="info",
                        text=f"-> {r.path} {r.params or ''} {r.query or ''} ({r.reasoning[:100]}...)")]
    return []


@component
def App():
    routes = [
        Route(key=r["name"], **r, children=[PAGES[r["name"]](key=r["name"])]) for r in ROUTES
    ]
    routes.append(Route(key="404", path="/*", name="not-found", children=[NotFound(key="404")]))
    return [Navigator(key="navigator"), Router(key="router", children=routes)]


@component
def Root(models=None):
    lms = hooks.use_memo(lambda: models or example_models(), [id(models)])
    return [DSPyProvider(key="dspy", models=lms, children=[App(key="app")])]


async def main() -> None:
    app = bootstrap(Root)
    try:
        await read_terminal_and_invoke(app, prompt="> ")  # messages render via ConsoleSink
    finally:
        app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
