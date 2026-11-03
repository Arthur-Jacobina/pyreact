"""pyreact_durable — durable agent memory at the border of the flow.

- :mod:`pyreact_durable.log`   pure storage (no pyreact / dspy imports)
- :mod:`pyreact_durable.react` ``Durable`` provider + persistent hooks
- :mod:`pyreact_durable.lm`    ``durable_lm`` record/replay at the LM boundary
- :mod:`pyreact_durable.turns` ``DurableTurnStore``: resume turns cut short by a crash
- :mod:`pyreact_durable.stateful` ``DurableParticipant``: deprecated (``Durable``
  dehydrates its log reference itself, so ``root.suspend()`` carries it)

``render_tree(log)`` / ``log.render()`` draw the entry tree as text. Only the
storage layer (and its text rendering) is imported eagerly, so ``from pyreact_durable import
SessionLog`` works without loading pyreact or dspy.
"""

from .log import (
    FORMAT_VERSION,
    SessionLocked,
    SessionLog,
    SessionLogError,
    archive_log,
    delete_log,
    log_files,
    register_migration,
    safe_session_id,
)
from .render import render_tree
from .stateful import DurableParticipant

_REACT = {
    "DEFAULT_JOURNAL", "Durable", "DurableHandle", "DurableStateError", "Conversation",
    "use_durable", "use_durable_lm", "use_persistent_state", "use_persistent_reducer",
    "use_conversation",
}
_LM = {"durable_lm", "ReplayMissError", "DurableLM", "llm_entries"}

__all__ = [
    "FORMAT_VERSION", "SessionLocked", "SessionLog", "SessionLogError",
    "register_migration", "render_tree", "safe_session_id", "log_files",
    "archive_log", "delete_log", "DurableTurnStore", "DurableParticipant", *sorted(_REACT), *sorted(_LM),
]


def __getattr__(name):
    if name in _REACT:
        from . import react

        return getattr(react, name)
    if name in _LM:
        from . import lm

        return getattr(lm, name)
    if name == "DurableTurnStore":
        from .turns import DurableTurnStore

        return DurableTurnStore
    raise AttributeError(f"module 'pyreact_durable' has no attribute {name!r}")
