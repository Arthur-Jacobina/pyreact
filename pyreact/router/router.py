from typing import (
    Dict,
    Optional,
    Union,
)
import warnings
from pyreact.core.core import component, hooks
from pyreact.core.provider import create_context
from .nav_service import (
    NavService,
    NavigateFn,
    NavigateOptions,
    ParamValue,
    QueryValue,
    build_url,
    parse_location,
)
from .match import matches as route_matches


RouteContext = create_context(default="/", name="Route")
RoutesCatalogContext = create_context(default=[], name="RoutesCatalog")
# {"url", "path", "query", "fragment"} of the location the Router rendered.
LocationContext = create_context(default=None, name="Location")


def use_routes_catalog():
    return hooks.use_context(RoutesCatalogContext)


def _build_url(
    path: str,
    params: Optional[Dict[str, ParamValue]] = None,
    query: Optional[Dict[str, QueryValue]] = None,
    fragment: str = "",
) -> str:
    """Build a URL with path parameters and query string"""
    return build_url(path, params, query, fragment)


def use_location() -> Dict[str, object]:
    """Current location ``{"url", "path", "query", "fragment"}``.

    Inside a :func:`Router` this comes from router context, i.e. from the same
    render pass that produced the route params, so query/fragment and params
    are always consistent (no tearing during navigation). Outside a Router it
    falls back to the session's NavService.
    """
    loc = hooks.use_context(LocationContext)
    if loc is None:
        navsvc = hooks.get_service("nav_service", NavService)
        loc = parse_location(navsvc.current)
    return loc


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
            # Commit and notify NavService subscribers (this Router, server).
            # The Router rerenders and provides the new location to its
            # subtree via context; we deliberately do not touch the global
            # RouteContext value, which is shared by every session.
            svc.commit(final_url)

        return navigate

    navigate_fn: NavigateFn = hooks.use_memo(lambda: _make_navigate(navsvc), [navsvc])
    navsvc.navigate = navigate_fn

    # Snapshot the location ONCE per render; everything below (route
    # selection, params, query, fragment) derives from this snapshot.
    current_path_only = navsvc.current
    location = hooks.use_memo(
        lambda: parse_location(current_path_only), [current_path_only]
    )

    current_path_for_match = location["path"]

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
                LocationContext(
                    value=location,
                    children=[
                        RoutesCatalogContext(
                            value=routes_catalog, children=selected_children
                        )
                    ],
                )
            ],
        )
    ]
