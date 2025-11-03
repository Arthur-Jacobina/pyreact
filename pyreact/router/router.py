from typing import (
    Dict,
    Optional,
    Union,
)
from urllib.parse import urlencode, urlparse, urlunparse
import warnings
from pyreact.core.core import component, hooks
from pyreact.core.provider import create_context
from .nav_service import (
    NavService,
    NavigateFn,
    NavigateOptions,
    ParamValue,
    QueryValue,
)
from .match import matches as route_matches


RouteContext = create_context(default="/", name="Route")
RoutesCatalogContext = create_context(default=[], name="RoutesCatalog")


def use_routes_catalog():
    return hooks.use_context(RoutesCatalogContext)


def _build_url(
    path: str,
    params: Optional[Dict[str, ParamValue]] = None,
    query: Optional[Dict[str, QueryValue]] = None,  # noqa: F821
    fragment: str = "",
) -> str:
    """Build a URL with path parameters and query string"""
    # Replace path parameters like :id with actual values
    if params:
        for key, value in params.items():
            path = path.replace(f":{key}", str(value))

    # Build query string
    query_string = ""
    if query:
        # Filter out None values and convert everything to strings
        clean_query = {k: str(v) for k, v in query.items() if v is not None}
        if clean_query:
            query_string = urlencode(clean_query)

    # Combine everything
    parsed = urlparse(path)
    result = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            query_string,
            fragment,
        )
    )

    return result


def use_route():
    # Subscribe to current path changes via RouteContext
    current = hooks.use_context(RouteContext)
    # Navigate comes from the NavService (set by Router on mount)
    navsvc = hooks.get_service("nav_service", NavService)

    def _navigate_live(*args, **kwargs) -> NavigateFn:
        svc_nav: NavigateFn = getattr(navsvc, "navigate", None)
        if callable(svc_nav):
            return svc_nav(*args, **kwargs)
        warnings.warn("navigate called before Router mounted")
        return None

    return current, _navigate_live


@component
def Router(*, initial=None, children):
    navsvc = hooks.get_service("nav_service", NavService)
    if initial is None:
        initial = navsvc.current

    # Force Router to rerender on nav changes so selected Route updates
    _version, _set_version = hooks.use_state(0)

    def _on_nav(_):
        _set_version(lambda v: v + 1)

    _nav_handler = hooks.use_memo(lambda: _on_nav, [])

    def _effect_subscribe():
        navsvc.subs.append(_nav_handler)

        def _cleanup():
            try:
                navsvc.subs.remove(_nav_handler)
            except ValueError:
                pass

        return _cleanup

    hooks.use_effect(_effect_subscribe, [_nav_handler, navsvc])

    # Ensure RouteContext reflects the current initial path on first mount
    hooks.use_effect(lambda: (RouteContext.set(navsvc.current), None)[1], [navsvc])

    # Ensure NavService has a stable navigate function
    def _make_navigate(svc) -> NavigateFn:
        def navigate(
            new_path: Union[str, NavigateOptions],  # noqa: F821
            params: Optional[Dict[str, ParamValue]] = None,
            query: Optional[Dict[str, QueryValue]] = None,
            fragment: str = "",
        ) -> None:
            """
            Navigate to a new path with optional parameters and query string.

            Args:
                new_path: Either a string path or a dict with 'path', 'params', 'query', 'fragment'
                params: Path parameters to replace in the URL (e.g., {'id': 123})
                query: Query string parameters (e.g., {'search': 'hello', 'page': 1})
                fragment: URL fragment/hash
            """
            if isinstance(new_path, dict):
                path_str = new_path.get("path", "/")
                params = new_path.get("params", params)
                query = new_path.get("query", query)
                fragment = new_path.get("fragment", fragment)
            else:
                path_str = new_path

            final_url = _build_url(path_str, params, query, fragment)
            # Update RouteContext and notify subscribers
            RouteContext.set(final_url)
            # Commit and notify NavService subscribers (e.g., Router, server)
            svc.commit(final_url)

        return navigate

    navigate_fn: NavigateFn = hooks.use_memo(lambda: _make_navigate(navsvc), [navsvc])
    navsvc.navigate = navigate_fn

    # Use the CURRENT path from the service
    current_path_only = navsvc.current

    current_path_for_match = current_path_only.split("?")[0]

    selected_child = None
    for ch in children:
        # VNode has attribute 'component_fn' and 'props'
        component_fn = getattr(ch, "component_fn", None)
        props = getattr(ch, "props", None)
        if component_fn is None or props is None:
            continue
        if getattr(component_fn, "__name__", "") != "Route":
            continue
        path_pattern = props.get("path", "/")
        exact = props.get("exact", True)
        if route_matches(path_pattern, current_path_for_match, exact):
            selected_child = ch
            break

    selected_children = [] if selected_child is None else [selected_child]

    # Build a catalog of all declared routes to expose via context
    routes_catalog = []
    for ch in children:
        component_fn = getattr(ch, "component_fn", None)
        props = getattr(ch, "props", None)
        if component_fn is None or props is None:
            continue
        if getattr(component_fn, "__name__", "") != "Route":
            continue
        path_pattern = props.get("path", "/")
        exact = props.get("exact", True)
        # Optional human-friendly name if provided alongside Route props
        name = props.get("name", props.get("title", props.get("key", path_pattern)))
        description = props.get("description")
        utterances = props.get("utterances") or []
        default_params = props.get("default_params")
        routes_catalog.append(
            {
                "path": path_pattern,
                "exact": exact,
                "name": name,
                "description": description,
                "utterances": utterances,
                "params": default_params,
            }
        )

    return [
        RouteContext(
            value=current_path_only,
            children=[
                RoutesCatalogContext(value=routes_catalog, children=selected_children)
            ],
        )
    ]
