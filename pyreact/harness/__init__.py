"""Guards and verification for the harness (``pyreact.core.harness``).

Everything here is ordinary harness middleware, so it composes with
``HarnessProvider(middleware=[...])`` (a subtree), ``install(...)`` (the
process) and your own middleware:

- **guards** run *before* a call and may refuse it (``Blocked``) or rewrite
  its inputs: :func:`guard`, :func:`tool_allowlist`, :func:`max_calls`,
  :func:`budget`, :func:`rate_limit`, :func:`input_policy`,
  :func:`require_confirmation`;
- **verifiers** run *after* a call and check its result, then raise, retry
  (optionally with the feedback injected) or flag: :func:`verify` with the
  checks :func:`schema`, :func:`assert_that`, :func:`grounded_in`;
- **hooks** scope either to the calling component and its subtree, with its
  lifecycle: :func:`use_guard`, :func:`use_verify`, :func:`use_middleware`.

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
from .hooks import scoped_middleware, use_guard, use_middleware, use_verify
from .verify import (
    Verdict,
    VerificationFailed,
    as_mapping,
    assert_that,
    grounded_in,
    schema,
    verify,
)

__all__ = [
    "PII",
    "GuardMiddleware",
    "Verdict",
    "VerificationFailed",
    "as_mapping",
    "assert_that",
    "block",
    "budget",
    "grounded_in",
    "guard",
    "input_policy",
    "max_calls",
    "rate_limit",
    "require_confirmation",
    "schema",
    "scoped_middleware",
    "tool_allowlist",
    "use_guard",
    "use_middleware",
    "use_verify",
    "verify",
]
