"""pyreact_durable — durable agent memory at the border of the flow.

- :mod:`pyreact_durable.log`   pure storage (no pyreact / dspy imports)
- :mod:`pyreact_durable.react` ``Durable`` provider + persistent hooks
- :mod:`pyreact_durable.lm`    ``durable_lm`` record/replay at the LM boundary

Only the storage layer is imported eagerly, so ``from pyreact_durable import
SessionLog`` works without loading pyreact or dspy.
"""

from .log import (
    FORMAT_VERSION,
    SessionLocked,
    SessionLog,
    SessionLogError,
    register_migration,
)

_REACT = {
    "DEFAULT_JOURNAL", "Durable", "DurableHandle", "DurableStateError", "Conversation",
    "use_durable", "use_durable_lm", "use_persistent_state", "use_persistent_reducer",
    "use_conversation",
}
_LM = {"durable_lm", "ReplayMissError", "DurableLM", "llm_entries"}

__all__ = [
    "FORMAT_VERSION", "SessionLocked", "SessionLog", "SessionLogError",
    "register_migration", *sorted(_REACT), *sorted(_LM),
]


def __getattr__(name):
    if name in _REACT:
        from . import react

        return getattr(react, name)
    if name in _LM:
        from . import lm

        return getattr(lm, name)
    raise AttributeError(f"module 'pyreact_durable' has no attribute {name!r}")
