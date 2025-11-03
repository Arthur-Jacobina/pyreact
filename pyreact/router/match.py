import re
from typing import Dict, Tuple


def _to_regex(pat: str, exact_local: bool):
    tokens = []
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == ":":
            j = i + 1
            while j < len(pat) and (pat[j].isalnum() or pat[j] == "_"):
                j += 1
            name = pat[i + 1 : j] or "param"
            tokens.append(f"(?P<{name}>[^/]+)")
            i = j
        elif c == "*" and i == len(pat) - 1:
            tokens.append("(?P<splat>.*)")
            i += 1
        else:
            tokens.append(re.escape(c))
            i += 1
    body = "".join(tokens)
    anchor = "^" + body + ("$" if exact_local else "")
    return re.compile(anchor)


def compile_route_pattern(path_pattern: str, exact: bool):
    """Compile a route path pattern into a regex pattern.

    Rules:
    - If pattern starts with '^' → treat as explicit regex (re.compile as-is)
    - If pattern ends with '/*' → prefix match (non-exact)
    - ':param' tokens become named groups
    - trailing '*' becomes 'splat' named group
    - exact flag controls end anchor
    """
    if path_pattern.startswith("^"):
        return re.compile(path_pattern)
    if path_pattern.endswith("/*"):
        return _to_regex(path_pattern[:-1], exact_local=False)
    return _to_regex(path_pattern, exact_local=exact)


def match_path(
    path_pattern: str, pathname: str, exact: bool
) -> Tuple[bool, Dict[str, str]]:
    """Match a pathname against a route pattern and return (ok, params)."""
    rx = compile_route_pattern(path_pattern, exact)
    m = rx.match(pathname)
    return (m is not None, m.groupdict() if m else {})


def matches(path_pattern: str, pathname: str, exact: bool) -> bool:
    """Convenience function that only returns a boolean match result."""
    ok, _ = match_path(path_pattern, pathname, exact)
    return ok
