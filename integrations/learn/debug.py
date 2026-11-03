"""Debugging programs and calls.

``inspect_program(program)`` -- what a program *is*: architecture, signature,
per-predictor instructions and demos.

``explain(call_id)`` / ``explain(last=True)`` -- what a call *did*: the exact
prompt messages the LM received and its raw completions (every LM call made
inside that module call: CoT = 1, RLM = one per REPL step + sub-LM calls).
Recorded by ``invoke_module`` in a small in-memory buffer (last 64 calls);
with ``set_debug(True)`` they are also attached to ``llm.end`` events
(``lm_calls``), so ``explain(call_id, events=...)`` works from a JSONL trace.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional

from integrations import telemetry as tm
from integrations.telemetry import set_debug

from .result import format_demo, predictor_view

__all__ = ["inspect_program", "explain", "Explanation", "Inspection", "set_debug"]


class Inspection(dict):
    """``dict`` with a readable ``str()``."""

    def __str__(self) -> str:
        lines = [f"{self['arch']}  {self['signature']}"]
        if self.get("key"):
            lines[0] += f"   key={self['key']}"
        if self.get("compiled"):
            lines.append("compiled: yes")
        for name, p in self["predictors"].items():
            lines.append(f"[{name}]  fields: {', '.join(p['fields'])}")
            lines.append("  instructions:")
            lines += ["    " + ln for ln in (textwrap.fill(p["instructions"] or "(none)", 96)
                                         .splitlines())]
            lines.append(f"  demos: {len(p['demos'])}")
            for i, d in enumerate(p["demos"][:5], 1):
                lines.append(f"    {i}. {format_demo(d)}")
            if len(p["demos"]) > 5:
                lines.append(f"    … {len(p['demos']) - 5} more")
        return "\n".join(lines)

    __repr__ = __str__


def inspect_program(program: Any) -> Inspection:
    """Architecture, signature, per-predictor instructions and demos."""
    from integrations.architectures import arch_of, signature_of, signature_text
    from integrations.feedback import key_str

    key = getattr(program, "_pyreact_key", None)
    return Inspection(
        arch=arch_of(program),
        signature=signature_text(signature_of(program)),
        key=None if key is None else key_str(key),
        compiled=bool(getattr(program, "_compiled", False)),
        predictors=predictor_view(program),
    )


class Explanation(dict):
    """A recorded call: ``str()`` renders prompts and completions."""

    def __init__(self, *a, max_chars: int = 4000, **kw):
        super().__init__(*a, **kw)
        self.max_chars = max_chars

    def _clip(self, text: Any) -> str:
        s = "" if text is None else str(text)
        if len(s) > self.max_chars:
            return s[: self.max_chars] + f"\n… [{len(s) - self.max_chars} more chars]"
        return s

    def __str__(self) -> str:
        lat = self.get("latency")
        head = [
            f"call {self['call_id']}  module={self.get('module')}  arch={self.get('arch')}",
            f"  key={self.get('module_key')}  component={self.get('component')}",
            f"  model={self.get('model')}  turn={self.get('turn_id')}"
            + (f"  latency={lat:.2f}s" if isinstance(lat, (int, float)) else ""),
            f"  inputs:  {textwrap.shorten(str(self.get('inputs')), 300, placeholder=' …')}",
            f"  outputs: {textwrap.shorten(str(self.get('outputs')), 300, placeholder=' …')}",
        ]
        if self.get("error"):
            head.append(f"  error: {self['error']}")
        calls: List[Dict[str, Any]] = self.get("lm_calls") or []
        head.append(f"  {len(calls)} LM call(s)")
        for i, c in enumerate(calls, 1):
            head.append(f"\n--- LM call {i}/{len(calls)}  ({c.get('model')}) ---")
            for m in c.get("messages") or []:
                head.append(f">>> {str(m.get('role', '?')).upper()}")
                head.append(self._clip(m.get("content")))
            head.append("<<< COMPLETION")
            head.append(self._clip(c.get("completion")))
        return "\n".join(head)

    __repr__ = __str__


def explain(
    call_id: Optional[str] = None,
    *,
    last: bool = True,
    events: Any = None,
    max_chars: int = 4000,
) -> Explanation:
    """Prompt messages + raw completions of a call (default: the last call).

    Looks in the in-memory call buffer first, then (if given) in ``events``
    (any source :func:`integrations.learn.iter_events` accepts) for an
    ``llm.end``/``llm.error`` carrying ``lm_calls`` (``set_debug(True)``).
    """
    if call_id is None and not last:
        raise ValueError("pass a call_id or last=True")
    rec = tm.get_call(call_id)
    if rec is None and events is not None:
        from .datasets import iter_events

        found = None
        for ev in iter_events(events):
            p = ev.get("payload") or {}
            if ev.get("kind") in ("llm.end", "llm.error") and (call_id is None
                                                               or p.get("call_id") == call_id):
                found = p
        if found is not None:
            rec = {**found, "lm_calls": found.get("lm_calls") or []}
    if rec is None:
        known = [c["call_id"] for c in tm.recorded_calls()][-8:]
        raise LookupError(
            f"call {call_id!r} is not in the explain buffer (keeps the last "
            f"{tm.CALL_BUFFER_SIZE} calls of this process; recent: {known}). For traces from "
            "other processes enable set_debug(True) so llm.end events carry the prompts, "
            "then explain(call_id, events=...)."
        )
    return Explanation(rec, max_chars=max_chars)
