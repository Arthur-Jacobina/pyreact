"""DSPy-aware error recovery: switch models on ``llm`` boundaries.

The generic middleware (``retry``, ``fallback``, ``circuit_breaker``,
``timeout``) lives in :mod:`pyreact.core.recovery` and works on any boundary
kind; it is re-exported here so one import covers an agent's whole stack::

    from integrations.recovery import fallback_model, retry, exponential

    DSPyProvider(models={"default": big, "fast": small}, children=[
        HarnessProvider(middleware=[
            fallback_model(chain=["default", "fast"]),
            retry(max_attempts=3, backoff=exponential(0.5, 8.0)),
        ], children=[Agent()]),
    ])

:func:`fallback_model` relies on the model-switching hook point of
:func:`integrations.hooks.invoke_module`: an ``llm`` boundary started with a
``DSPyProvider(models=...)`` registry carries ``b.attrs["model_key"]`` and
``b.attrs["model_names"]``, and setting ``model_key`` before ``call_next``
selects which registry LM that run uses.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from pyreact.core.harness import Boundary, Middleware, Next, middleware
from pyreact.core.recovery import (  # noqa: F401  (re-exports)
    BoundaryTimeout,
    CircuitOpen,
    RetryOn,
    _base,
    _brief,
    _cancelling,
    _predicate,
    circuit_breaker,
    exponential,
    fallback,
    is_retryable,
    publish_recovery,
    retry,
    retry_after_of,
    should_fall_back,
    timeout,
)

__all__ = [
    "BoundaryTimeout",
    "CircuitOpen",
    "circuit_breaker",
    "exponential",
    "fallback",
    "fallback_model",
    "is_retryable",
    "retry",
    "retry_after_of",
    "should_fall_back",
    "timeout",
]


def _order(current: Optional[str], chain: Sequence[str]) -> List[Optional[str]]:
    """Models to try, in order: the call's own model first, then the rest of
    the chain (from the current key on when it is part of the chain)."""
    chain = list(chain)
    if current in chain:
        return chain[chain.index(current):]
    return [current, *[k for k in chain if k != current]]


def fallback_model(
    chain: Sequence[str] = ("default", "fast"),
    on: Optional[RetryOn] = None,
    kinds: Optional[Iterable[str]] = ("llm",),
    *,
    name: str = "fallback_model",
) -> Middleware:
    """On failure, re-run the ``llm`` call with the next model of ``chain``
    (``DSPyProvider(models=...)`` registry keys, in preference order).

    The call starts with its own model (``use_call(model=...)``, else
    ``"default"``); a call whose model is in the chain only degrades to the
    keys after it. Keys missing from the provider's registry are skipped; a
    call without a registry (explicit ``lm=`` outside a provider) is passed
    through untouched. ``on``: exception type(s) or ``predicate(exc)``
    (default: every ``Exception`` but ``Blocked`` — including an open
    circuit, so ``circuit_breaker`` placed *inside* it fails fast to the
    next model).

    Each switch bumps ``b.attempt``, sets ``b.attrs["model_key"]`` /
    ``["model"]`` / ``["fallback"]`` and publishes ``recovery.fallback``
    ``{from, to, error}``. When every model failed, the last error is raised.
    """
    should = _predicate(on, should_fall_back)

    @middleware(kinds=kinds, name=name)
    async def _fallback_model(b: Boundary, call_next: Next) -> Any:
        names = b.attrs.get("model_names") or {}
        if not names:
            return await call_next(b)
        order = [k for i, k in enumerate(_order(b.attrs.get("model_key"), chain))
                 if i == 0 or k in names]
        last: Optional[BaseException] = None
        for i, key in enumerate(order):
            if i > 0:
                prev = b.attrs.get("model_key")
                b.attrs["model_key"] = key
                b.attrs["model"] = names[key]
                b.attrs["fallback"] = True
                b.attempt += 1
                publish_recovery("recovery.fallback", {
                    **_base(b), "from": prev, "to": key, "model": names[key],
                    "error": _brief(last)}, b)
            try:
                return await call_next(b)
            except Exception as exc:
                if not should(exc) or _cancelling() or i == len(order) - 1:
                    raise
                last = exc
        raise RuntimeError("fallback_model: empty chain")  # pragma: no cover

    return _fallback_model
