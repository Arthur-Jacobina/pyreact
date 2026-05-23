# dspy_agent.py ----------------------------------------------------
from collections.abc import Callable
from typing import Any, Iterable, Optional, Union

import dspy

from pyreact.core.core import hooks
from pyreact.tools import Tool, resolve_tools, use_tools

from .use_dspy import use_dspy_call


def to_dspy_tool(tool: Tool) -> dspy.Tool:
    # Pass the core-derived JSON Schema as DSPy 'args' so DSPy uses our schema
    # instead of re-introspecting the callable.
    properties = tool.parameters.get("properties", {})
    arg_desc = {
        name: prop["description"]
        for name, prop in properties.items()
        if isinstance(prop, dict) and prop.get("description")
    }
    return dspy.Tool(
        tool.fn,
        name=tool.name,
        desc=tool.description,
        args=properties or None,
        arg_desc=arg_desc or None,
    )


def use_dspy_agent(
    signature,  # class dspy.Signature (or string "q -> a")
    *,
    tools: Optional[Iterable[Union[Tool, Callable, str]]] = None,
    max_iters: int = 20,
    model: Optional[str] = None,
    lm: Optional[Any] = None,
    name: Optional[str] = None,  # optional, to distinguish instances
):
    """
    ReAct agent that can call tools.
    'tools' accepts Tool instances, callables, or names resolved against the
    surrounding ToolProvider. Returns (run, result, loading, error) like use_dspy_call.
    """
    registry = use_tools()
    resolved = resolve_tools(tools, registry)

    sig_key = signature if isinstance(signature, str) else signature.__name__
    tools_key = tuple(t.name for t in resolved)

    # Rebuild ReAct only when signature, tool set, limits, or name change.
    react = hooks.use_memo(
        lambda: dspy.ReAct(
            signature, tools=[to_dspy_tool(t) for t in resolved], max_iters=max_iters
        ),
        deps=[sig_key, tools_key, max_iters, name],
    )

    return use_dspy_call(react, model=model, lm=lm)
