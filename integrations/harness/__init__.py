"""Guards and verification for DSPy apps.

Re-exports everything in :mod:`pyreact.harness` (guards, ``verify`` and the
generic checks) plus the DSPy-aware pieces:

- :func:`llm_judge`: an LM-graded check (the GEPA judge metric);
- :func:`attempt_adapter`: how ``llm`` boundaries condition a verifier's
  re-run on ``rollout_id`` / ``feedback`` (applied by ``invoke_module``).

Import from here in DSPy apps: ``from integrations.harness import verify,
schema, llm_judge, budget``.
"""

from pyreact.harness import *  # noqa: F401,F403
from pyreact.harness import __all__ as _generic

from .adapter import FEEDBACK_FIELD, attempt_adapter
from .judge import llm_judge

__all__ = [*_generic, "FEEDBACK_FIELD", "attempt_adapter", "llm_judge"]
