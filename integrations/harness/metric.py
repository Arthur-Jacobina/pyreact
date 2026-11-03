"""``check_metric``: a verification check as an optimizer metric."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Optional

from pyreact.core.harness import Boundary
from pyreact.harness.verify import _check_name, _normalize


def check_metric(check: Callable[[Boundary, Any], Any], *, kind: str = "llm",
                 name: Optional[str] = None):
    """Turn a ``verify`` check ``(b, result)`` into a metric
    ``(example, pred) -> (score, feedback)`` for ``integrations.learn.optimize``.

    Closes the loop: the verifier that rejected answers in production
    (``verify(check, to_feedback=True)``) grades the candidates GEPA proposes,
    and its feedback text is what GEPA reflects on. The check sees a
    boundary of ``kind`` whose inputs are the example's inputs.
    """
    label = name or _check_name(check)

    def metric(example: Any, pred: Any):
        inputs = dict(example.inputs()) if hasattr(example, "inputs") else {}
        b = Boundary(kind=kind, name=label, inputs=inputs)
        out = check(b, pred)
        if inspect.isawaitable(out):
            out = asyncio.run(_await(out))
        score, feedback = _normalize(out, label)
        return score, feedback or ("Passes the check." if score >= 1 else f"Fails {label}.")

    metric.__name__ = f"check_metric({label})"
    return metric


async def _await(aw: Any) -> Any:
    return await aw
