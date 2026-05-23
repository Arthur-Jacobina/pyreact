# tool.py ----------------------------------------------------
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .schema import derive_parameters, summary_from_docstring


# Provider-agnostic description of a callable an agent may invoke.
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema ('object')
    fn: Callable  # sync or async

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def make_tool(
    fn: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Tool:
    """Build a Tool from a callable, deriving any metadata not passed explicitly."""
    if isinstance(fn, Tool):
        return fn
    return Tool(
        name=name or getattr(fn, "__name__", "tool"),
        description=description or summary_from_docstring(fn),
        parameters=parameters if parameters is not None else derive_parameters(fn),
        fn=fn,
    )
