"""Deprecated: ``CompiledModulesParticipant`` (kept for one release).

Learned programs are state of the tree (ADR 0001,
``docs/adr/0001-look-like-react.md``) and dehydrate themselves through the
core's ``use_dehydrate``, so ``root.suspend()`` / ``hydrate_root(...)`` carry
them with no participant:

- ``LearnedProgram`` / ``LearnedPrograms`` providers: slice ``"programs"``
  (``{name: path | {"state": ...} | None}``, see
  ``integrations.learn.programs``);
- the ``DSPyProvider`` env's compiled / hot-swapped-by-key modules
  (``env.compiled``): slice ``"compiled"``::

      "slices": {"App/DSPyProvider[dspy]": {"compiled": [
          {"key": ["ReAct", "QA", "coder"], "key_str": "ReAct:QA:coder", "arch": "react",
           "path": "artifacts/coder_gepa.json"}]}}          # or "state": {...} inline

  hydrated before the first render and applied when ``use_module`` /
  ``use_agent`` build the module for the same key; a result file that is
  gone is a ``hydration.mismatch`` (plus ``session.resume.degraded``,
  ``participant: "dspy.compiled"``) and that module starts from its
  signature.

The legacy ``Session.suspend`` manifest carries the same slices, so this
participant's fragment is empty (``dir=`` is ignored: programs without a
result file are inlined). It still reads the ``modules`` fragment of a
manifest written by an older pyreact.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, Optional

__all__ = ["CompiledModulesParticipant"]


class CompiledModulesParticipant:
    """Deprecated: ``CompiledModulesParticipant(session=None, dir=None)``."""

    name = "dspy.compiled"
    version = "2"

    def __init__(self, session: Any = None, *, dir: Optional[os.PathLike] = None) -> None:
        warnings.warn(
            "CompiledModulesParticipant is deprecated: DSPyProvider and LearnedProgram "
            "dehydrate learned programs themselves (use_dehydrate); suspend with "
            "root.suspend() and resume with hydrate_root(...), no participants needed "
            "(see docs/adr/0001-look-like-react.md)", DeprecationWarning, stacklevel=2)
        self.session = session
        self.dir = dir

    async def suspend(self, session: Any) -> Dict[str, Any]:
        return {}  # the learned modules ride in the manifest's "slices"

    async def resume(self, session: Any, fragment: Dict[str, Any]) -> None:
        from .hooks import RESTORE_SERVICE

        modules = fragment.get("modules") or []
        restore = session.services.setdefault(RESTORE_SERVICE, {}) if modules else {}
        for e in modules:  # a manifest from an older pyreact
            key = tuple(e.get("key") or ())
            state = e.get("state")
            if state is None and e.get("path"):
                try:
                    with open(e["path"], encoding="utf-8") as f:
                        data = json.load(f)
                    state = data["state"] if isinstance(data, dict) and "state" in data else data
                except Exception as exc:
                    session.degrade(self.name, "result missing" if isinstance(
                        exc, FileNotFoundError) else "result unreadable",
                        key=e.get("key_str"), path=e["path"], error=repr(exc))
                    continue
            if key and state is not None:
                restore[key] = state

    def __repr__(self) -> str:
        return "<CompiledModulesParticipant (deprecated)>"
