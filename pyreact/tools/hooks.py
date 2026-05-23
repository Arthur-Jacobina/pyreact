# hooks.py ----------------------------------------------------
from typing import Callable, Dict, Iterable, List, Optional, Union

from pyreact.core.core import component, hooks
from pyreact.core.provider import create_context

from .tool import Tool, make_tool

# Context carrying the tool registry ({name -> Tool}); None until provided.
ToolContext = create_context(default=None, name="Tools")

ToolLike = Union[Tool, Callable]


def _registry_from(tools):
    registry = {}
    for item in tools or []:
        tool = make_tool(item)  # wraps plain callables
        registry[tool.name] = tool
    return registry


@component
def ToolProvider(*, tools: Optional[Iterable[ToolLike]] = None, children=None):
    # Inject a registry of tools into the subtree (mirrors DSPyProvider).
    registry = _registry_from(tools)
    return [ToolContext(value=registry, children=children or [])]


def use_tools() -> Dict[str, Tool]:
    """Provided tool registry, or {} when no ToolProvider is mounted."""
    registry = hooks.use_context(ToolContext)
    return registry or {}


def use_tool(
    fn: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    deps: Optional[list] = None,
) -> Tool:
    """Declare a Tool local to the component, memoized on deps (cf. use_callback)."""
    deps_key = tuple(deps) if deps is not None else None
    return hooks.use_memo(
        lambda: make_tool(fn, name=name, description=description),
        deps=[name, description, deps_key],
    )


def resolve_tools(selected, registry: Dict[str, Tool]) -> List[Tool]:
    """Resolve Tools / callables / registry-name strings into Tool objects."""
    resolved = []
    for item in selected or []:
        if isinstance(item, str):
            if item not in registry:
                raise KeyError(
                    f"Tool {item!r} not in ToolProvider registry "
                    f"(available: {sorted(registry)})."
                )
            resolved.append(registry[item])
        else:
            resolved.append(make_tool(item))
    return resolved
