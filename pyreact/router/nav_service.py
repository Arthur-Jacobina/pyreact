from dataclasses import dataclass, field
from typing import Callable, NotRequired, Optional, Dict, Union, TypeAlias, TypedDict
from urllib.parse import urlparse, parse_qs

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


@dataclass
class NavService:
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

    def commit(self, final_url: str) -> None:
        """Set current URL and notify subscribers."""
        self.current = final_url
        for fn in list(self.subs):
            try:
                fn(final_url)
            except Exception:
                pass
