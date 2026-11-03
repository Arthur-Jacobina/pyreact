"""``Activity``: hide a subtree without losing its state (cf. React ``<Activity>``).

::

    Activity(mode="visible" if panel_open else "hidden", children=[Coder(key="coder")])

- ``mode="hidden"``: the subtree stays mounted and keeps its hook state, but
  its effects are cleaned up (cleanup reason ``"hidden"``, see
  :func:`~pyreact.core.hook.use_cleanup_reason`) and do not run while hidden.
  Subscriptions made in effects (``use_turn``, ``use_on_turn``, input
  listeners) go away with them, so **no new turns are delivered** to it.
  It still re-renders on state changes (so it is up to date when shown).
- ``mode="visible"``: every effect of the subtree runs again (like a mount).
  A turn that started while the subtree was hidden is not delivered on reveal.

``Activity`` is *transparent*: it adds no segment to its descendants' paths,
so wrapping a subtree in it later does not change their addresses (durable
state, dehydrated slices).

``root.suspend()`` hides the whole tree the same way with the reason
``"suspend"`` (:func:`hide_tree`), so an effect can pause / snapshot what it
owns instead of destroying it before the tree is dehydrated.
"""

from __future__ import annotations

from typing import Any, List

from .core import _context_stack, component

__all__ = ["Activity", "ACTIVITY_MODES", "hide_tree", "reveal_tree"]

ACTIVITY_MODES = ("visible", "hidden")


@component(transparent=True)
def Activity(mode: str = "visible", children=None):
    """Render ``children``; with ``mode="hidden"`` keep them mounted but
    inactive (see the module docs)."""
    if mode not in ACTIVITY_MODES:
        raise ValueError(f"Activity mode must be one of {ACTIVITY_MODES}, got {mode!r}")
    ctx = _context_stack.get()
    ctx._hides_children = mode == "hidden"
    ctx._hide_reason = "hidden"
    return list(children or [])


def hide_tree(root: Any, reason: str = "suspend") -> List[Any]:
    """Hide a whole tree now, its root included: run every effect cleanup
    with ``reason`` (state is kept) and stop running effects. Returns the
    tasks of async cleanups (await them before dehydrating). Must run on the
    render loop."""
    root._self_hidden = True
    root._hide_reason = reason
    root._apply_visibility()
    tasks, root._hide_tasks = list(getattr(root, "_hide_tasks", None) or []), []
    return tasks


def reveal_tree(root: Any) -> None:
    """Undo :func:`hide_tree` (e.g. a suspend that failed): the effects of
    the tree run again at its next commit (a re-render is scheduled)."""
    from .runtime import schedule_rerender

    root._self_hidden = False
    root._hide_reason = "hidden"
    if getattr(root, "_mounted", False):
        schedule_rerender(root, reason="reveal")
