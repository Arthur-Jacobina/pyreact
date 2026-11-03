# use_dspy.py ----------------------------------------------------
"""Backward-compatible tuple API over :func:`integrations.hooks.use_call`."""

from collections.abc import Callable
from typing import Any, Optional

from .hooks import use_call


def use_dspy_call(
    module,
    *,
    model: Optional[str] = None,
    lm: Optional[Any] = None,
    policy: str = "replace",
) -> tuple[Callable[..., None], tuple, bool, Optional[Exception]]:
    """
    Policy: replace (the last call wins; the previous in-flight call is cancelled).
    Returns (run, result, loading, error) where ``result`` is ``(prediction, ver)``.
    ``ver`` increases only when a call settles (success or error), so effects
    that depend on ``result`` fire exactly once per completed call.
    """
    call = use_call(module, model=model, lm=lm, policy=policy)
    return call.run, (call.result, call.ver), call.loading, call.error
