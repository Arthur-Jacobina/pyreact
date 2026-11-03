from dataclasses import dataclass, field
from typing import Callable, NotRequired, Optional, Dict, Union, TypeAlias, TypedDict
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs

ParamValue: TypeAlias = Union[str, int, float, bool]
QueryValue: TypeAlias = Union[str, int, float, bool]


class NavigateOptions(TypedDict):
    path: str
    params: NotRequired[Dict[str, ParamValue]]
    query: NotRequired[Dict[str, QueryValue]]
    fragment: NotRequired[str]


NavigateFn: TypeAlias = Callable[
    [
        Union[str, NavigateOptions],
        Optional[Dict[str, ParamValue]],
        Optional[Dict[str, QueryValue]],
        str,
    ],
    None,
]


def build_url(
    path: str,
    params: Optional[Dict[str, ParamValue]] = None,
    query: Optional[Dict[str, QueryValue]] = None,
    fragment: str = "",
) -> str:
    """Build a URL from a path pattern, path params, query dict and fragment."""
    if params:
        for key, value in params.items():
            path = path.replace(f":{key}", str(value))

    parsed = urlparse(path)
    # A query/fragment already in ``path`` ("/x?tab=a#top", e.g. ``:nav``) is
    # kept; the ``query`` dict / ``fragment`` arguments override it.
    merged = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
    if query:
        merged.update({k: str(v) for k, v in query.items() if v is not None})
    query_string = urlencode(merged) if merged else ""

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            query_string,
            fragment or parsed.fragment,
        )
    )


def parse_location(url: str) -> Dict[str, object]:
    """Split ``url`` into ``{"url", "path", "query", "fragment"}``."""
    parsed = urlparse(url or "/")
    query = {k: (v[0] if v else "") for k, v in parse_qs(parsed.query).items()}
    return {
        "url": url or "/",
        "path": parsed.path or "/",
        "query": query,
        "fragment": parsed.fragment,
    }


@dataclass
class NavService:
    """Per-session navigation state (``hooks.get_service("nav_service", ...)``).

    ``navigate`` is installed by a mounted :func:`Router`; ``go`` works with or
    without one (before a Router mounts it just commits the URL so the Router
    picks it up as its initial location).
    """

    subs: list[Callable[[str], None]] = field(default_factory=list)
    navigate: Optional["NavigateFn"] = None
    current: str = "/"

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def get_query_params(self) -> Dict[str, str]:
        """Get query parameters from current URL"""
        parsed = urlparse(self.current)
        query_params = {}

        for key, values in parse_qs(parsed.query).items():
            query_params[key] = values[0] if values else ""

        return query_params

    def get_fragment(self) -> str:
        """Get fragment/hash from current URL"""
        parsed = urlparse(self.current)
        return parsed.fragment

    def get_path(self) -> str:
        """Get path portion of current URL (without query or fragment)"""
        parsed = urlparse(self.current)
        return parsed.path

    def go(
        self,
        path: str,
        params: Optional[Dict[str, ParamValue]] = None,
        query: Optional[Dict[str, QueryValue]] = None,
        fragment: str = "",
    ) -> None:
        """Navigate via the Router if mounted, else commit the URL directly."""
        if callable(self.navigate):
            self.navigate(path, params, query, fragment)
        else:
            self.commit(build_url(path, params, query, fragment))

    def commit(self, final_url: str) -> None:
        """Set current URL and notify subscribers."""
        self.current = final_url
        for fn in list(self.subs):
            try:
                fn(final_url)
            except Exception:
                pass
