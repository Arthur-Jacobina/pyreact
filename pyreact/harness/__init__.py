"""Guards and verification for the harness (``pyreact.core.harness``).

Everything here is ordinary harness middleware, so it composes with
``HarnessProvider(middleware=[...])`` (a subtree), ``install(...)`` (the
process) and your own middleware:

- **guards** run *before* a call and may refuse it (``Blocked``) or rewrite
  its inputs: :func:`guard`, :func:`tool_allowlist`, :func:`max_calls`,
  :func:`budget`, :func:`rate_limit`, :func:`input_policy`,
  :func:`require_confirmation`.

This package is framework-generic (no DSPy); the DSPy-aware pieces live in
``integrations.harness``.
"""

from .guards import (
    PII,
    GuardMiddleware,
    block,
    budget,
    guard,
    input_policy,
    max_calls,
    rate_limit,
    require_confirmation,
    tool_allowlist,
)

__all__ = [
    "PII",
    "GuardMiddleware",
    "block",
    "budget",
    "guard",
    "input_policy",
    "max_calls",
    "rate_limit",
    "require_confirmation",
    "tool_allowlist",
]
