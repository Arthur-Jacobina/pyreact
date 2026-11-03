# route.py
from typing import Dict, List, Optional, Any
from pyreact.core.core import component, hooks
from pyreact.core.provider import create_context
from .router import use_route, use_location  # read router context
from .match import compile_route_pattern

RouteParamsContext = create_context(default={}, name="RouteParams")


def use_route_params():
    return hooks.use_context(RouteParamsContext)


def use_query_params() -> Dict[str, str]:
    """Query parameters of the current location.

    Read from router context (same render as :func:`use_route_params`), so a
    component never sees new query params with stale route params.
    """
    return dict(use_location()["query"])


def use_fragment() -> str:
    """Fragment (``#...``) of the current location, from router context."""
    return use_location()["fragment"]


def use_navigate():
    """Hook that returns an enhanced navigate function with parameter support"""
    _, navigate = use_route()
    return navigate


@component
def Route(
    path: str,
    *,
    children,
    exact: bool = True,
    name: Optional[str] = None,
    description: Optional[str] = None,
    utterances: Optional[List[str]] = None,
    default_params: Optional[Dict[str, Any]] = None,
):
    """Routing rules:
    - If ``path`` starts with ``'^'`` → interpreted as REGEX (``re.match``).
    - If ``path`` ends with ``'/*'`` → prefix (e.g., ``'/users/*'`` matches ``'/users'`` and ``'/users/42'``).
    - ``':param'`` becomes a named group (e.g., ``'/users/:id'`` → ``params['id']``).
    - ``'*'`` at the end becomes ``splat`` (e.g., ``'/files/*'`` → ``params['splat']``).
    - When matched, inject params into :class:`RouteParamsContext`.

    Optional metadata props used by the Router's route catalog:
    - ``name``: friendly name (e.g., "home", "about").
    - ``description``: description for ranking/help.
    - ``utterances``: common expressions to map free text.
    - ``default_params``: default parameters if the route is parameterized.
    """
    current_full, _ = use_route()  # e.g. '/about?search=test'
    # Use only the path portion for route matching (without query params)
    current = current_full.split("?")[0]

    # Router now selects the single matching Route VNode; this component
    # only needs to inject params and render children when matched.

    def build_matcher():
        rx = compile_route_pattern(path, exact)

        def match(s: str):
            m = rx.match(s)
            return (m is not None, m.groupdict() if m else {})

        return match

    matcher = hooks.use_memo(build_matcher, [path, exact])

    ok, params = matcher(current)

    # If this route doesn't match, return empty
    if not ok:
        return []

    # pass params to children (if any)
    return [RouteParamsContext(value=params, children=children)]
