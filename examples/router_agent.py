from typing import Callable, List, Dict, Optional, Any, TypedDict
import dspy

from message import Message
from pyreact.core.core import component, hooks
from pyreact.router import use_route, use_routes_catalog
from integrations.use_dspy import use_dspy_call
from integrations.dspy_integration import use_dspy_module


class Path(TypedDict):
    path: str
    description: str
    utterances: List[str]
    params: Optional[Dict[str, Any]]


class RouteSelectionSig(dspy.Signature):
    """Choose the best route for the message.

    Instructions:
    - Return ONLY the exact path of one of the provided routes.
    - If nothing is appropriate, return None.
    - Consider description and utterances.
    - Utterances are the examples of how the user can ask for the route.
    """

    message: str = dspy.InputField()
    possible_routes: List[Path] = dspy.InputField(description="List of possible routes")
    path: str = dspy.OutputField()


def _is_parametrized(path: str) -> bool:
    return ":" in path or path.endswith("*")


@component
def RouterAgent(*, message: str, on_navigate: Callable[[str, int], None]):
    catalog = use_routes_catalog() or []

    choose_mod = use_dspy_module(
        RouteSelectionSig, dspy.ChainOfThought, name="router-agent"
    )
    call_llm, llm_result, llm_loading, _llm_error = use_dspy_call(
        choose_mod, model="fast"
    )

    def _routes_mapper(routes: List[Dict[str, Any]]) -> List[Path]:
        catalog: List[str] = []
        for route in routes:
            catalog.append(
                {
                    "path": route.get("path") or "",
                    "description": route.get("description") or "",
                    "utterances": route.get("utterances") or [],
                    "params": route.get("params"),
                }
            )

        return catalog

    def _effect_decide():
        if not isinstance(message, str) or not message.strip():
            return
        call_llm(message=message, possible_routes=_routes_mapper(catalog))

    hooks.use_effect(_effect_decide, [message])

    def _effect_llm_nav():
        if llm_result is None:
            return

        mod_res, ver = llm_result
        if mod_res is None:
            return

        path_raw = getattr(mod_res, "path", None)
        reasoning = getattr(mod_res, "reasoning", None)
        print(f"Reasoning: {reasoning}")
        on_navigate(path_raw, ver)

    hooks.use_effect(_effect_llm_nav, [llm_result])

    if llm_loading:
        return [Message(text="Router agent, verificando rota...", sender="system")]

    return []
