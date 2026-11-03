"""Checkpoint participant: learned / compiled DSPy programs survive a suspend.

A session's ``DSPyProvider`` envs hold the programs an optimizer compiled or
``OptimizationResult.hot_swap`` swapped in (``env.compiled``). Without this,
a restart rebuilds every module from its signature and the optimized prompt
is gone. ``CompiledModulesParticipant`` (name ``"dspy.compiled"``) records
each one in the suspend manifest::

    "dspy.compiled": {"modules": [
        {"key": ["ReAct", "Coder", "App/Agent[agent]"], "key_str": "ReAct:Coder:App/Agent[agent]",
         "provider": "App/DSPyProvider[dspy]", "arch": "react",
         "path": "artifacts/coder_gepa.json"}]}          # or "state": {...} inline

- ``path``: the ``integrations.learn`` result file the program came from
  (``OptimizationResult.save`` / ``load`` then ``hot_swap``), or one written
  now under ``dir`` (``<dir>/<session>/<key>.json``, same format) for
  programs that have no file;
- ``state``: the program's ``dump_state()`` inline, when there is neither.

On resume the states are loaded **before the first render** and applied
when ``use_module`` / ``use_agent`` build the module for the same key (the
module is then in ``env.compiled`` again, so the next suspend keeps it and an
optimizer does not recompile it). A result file that is gone or unreadable
degrades (``session.resume.degraded``, ``participant: "dspy.compiled"``) and
that module starts from its signature.

    app = AppRunner(App, participants=[DurableParticipant,
                                       lambda s: CompiledModulesParticipant(s, dir="runs/learned")])
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

__all__ = ["CompiledModulesParticipant"]


def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


class CompiledModulesParticipant:
    """``CompiledModulesParticipant(session=None, dir=None)``."""

    name = "dspy.compiled"
    version = "1"

    def __init__(self, session: Any = None, *, dir: Optional[os.PathLike] = None) -> None:
        self.session = session
        self.dir = Path(dir) if dir is not None else None

    def _save(self, session: Any, key: Any, program: Any) -> Optional[str]:
        if self.dir is None:
            return None
        from integrations.feedback import key_str
        from integrations.learn.result import OptimizationResult

        from .architectures import arch_of

        path = (self.dir / quote(str(session.id), safe="")
                / f"{quote(key_str(key), safe='')}.json")
        try:
            res = OptimizationResult(program=program,
                                     optimizer=getattr(program, "_pyreact_optimizer", "unknown"),
                                     arch=arch_of(program), module_key=key)
            return res.save(str(path))
        except Exception:
            return None

    async def suspend(self, session: Any) -> Dict[str, Any]:
        from integrations.feedback import key_str
        from integrations.learn.hmr import find_envs

        from .architectures import arch_of

        modules: List[Dict[str, Any]] = []
        seen = set()
        for provider, env in find_envs(session):
            for key, program in list(env.compiled.items()):
                if key in seen or not isinstance(key, tuple):
                    continue
                seen.add(key)
                entry: Dict[str, Any] = {"key": list(key), "key_str": key_str(key),
                                         "provider": provider, "arch": arch_of(program)}
                path = getattr(program, "_pyreact_result_path", None)
                if not (path and os.path.exists(path)):
                    path = self._save(session, key, program)
                if path:
                    entry["path"] = str(path)
                else:
                    entry["state"] = _plain(program.dump_state())
                modules.append(entry)
        return {"modules": modules}

    async def resume(self, session: Any, fragment: Dict[str, Any]) -> None:
        from .hooks import RESTORE_SERVICE

        restore = session.services.setdefault(RESTORE_SERVICE, {})
        for e in fragment.get("modules") or []:
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
        return f"<CompiledModulesParticipant dir={self.dir}>"
