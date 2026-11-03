# dspy_agent.py ----------------------------------------------------
"""Backward-compatible tuple API over :func:`integrations.hooks.use_agent`."""

from collections.abc import Callable
from typing import Any, Iterable, Optional, Union

from pyreact.tools import Tool

from .hooks import to_dspy_tool, use_agent

__all__ = ["to_dspy_tool", "use_dspy_agent"]


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
    call = use_agent(signature, tools=tools, max_iters=max_iters, model=model, lm=lm, name=name)
    return call.run, (call.result, call.ver), call.loading, call.error
