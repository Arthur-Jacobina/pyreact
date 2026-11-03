"""Text rendering of a session log (stdlib only).

``render_tree(log)`` is what ``python -m pyreact_durable show`` prints; it is
also ``SessionLog.render()`` and ``pyreact_durable.render_tree``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:  # pragma: no cover
    from .log import SessionLog

__all__ = ["render_tree", "summarize"]


def _brief(value: Any, n: int = 60) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=repr)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_path(path: str, n: int = 40) -> str:
    """``App/A[a]/B[b]/Chat[chat]`` -> ``…/Chat[chat]`` when long."""
    return path if len(path) <= n else "…/" + path.rsplit("/", 1)[-1]


def summarize(e: dict, width: int = 60) -> str:
    t = e.get("type")
    if t == "state":
        what = "deleted" if e.get("deleted") else _brief(e.get("value"), width)
        return f"{_short_path(str(e.get('path')))}#{e.get('key')} = {what}"
    if t == "event":
        p = e.get("payload") or {}
        text = p.get("text") or p.get("value") or p.get("error")
        detail = _brief(text, width) if text is not None else _brief(p, width)
        return f"{e.get('kind')} {detail}"
    if t == "llm":
        choices = (e.get("response") or {}).get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content") or choices[0].get("text")
        return (f"{e.get('model')} {str(e.get('key'))[:8]}#{e.get('occurrence')} → "
                f"{_brief(content, width)}")
    if t == "checkpoint":
        n = sum(len(v) for v in (e.get("state") or {}).values())
        s = f" — {_brief(e['summary'], width)}" if e.get("summary") else ""
        return f"{n} value(s){s}"
    if t == "custom":
        return f"{e.get('customType')} {_brief(e.get('data'), width)}"
    return _brief({k: v for k, v in e.items() if k not in ("type", "id", "parentId",
                                                          "timestamp")}, width)


def render_tree(log: "SessionLog", width: int = 60) -> List[str]:
    """One line per entry: tree shape, labels, ``⑂`` branch points, ``*`` leaf."""
    h = log.header
    st = log.stats()
    out = [f"session {h.get('id')}" + (f" \"{log.name}\"" if log.name else "")
           + f"  v{h.get('version')}  {st['entries']} entries  "
           f"{st['branch_points']} branch point(s)  leaf={log.leaf}"]
    if h.get("parentSession"):
        out.append(f"  forked from {h['parentSession']} @ {h.get('forkedFrom')}")
    if log.recovered:
        out.append(f"  ! ignored {len(log.recovered)} garbled line(s)")
    roots = log.tree()
    stack: List[tuple] = []
    for i, r in reversed(list(enumerate(roots))):
        if len(roots) > 1:
            last = i == len(roots) - 1
            stack.append((r, "└─ " if last else "├─ ", "   " if last else "│  "))
        else:
            stack.append((r, "", ""))
    if log.leaf is None:
        out.append("  * (leaf at root: next entry starts a new root)")
    while stack:
        node, first, lead = stack.pop()
        e = node["entry"]
        mark = "*" if node["is_leaf"] else ("·" if node["on_path"] else " ")
        kids = node["children"]
        extra = ""
        if node["labels"]:
            extra += "  [" + ", ".join(node["labels"]) + "]"
        if len(kids) > 1:
            extra += f"  ⑂{len(kids)}"
        if node["is_leaf"]:
            extra += "  <- leaf"
        out.append(f"{mark} {first}{e['id']} {e.get('type', '?'):<10} "
                   f"{summarize(e, width)}{extra}")
        if len(kids) == 1:
            stack.append((kids[0], lead, lead))
        else:
            for i in reversed(range(len(kids))):
                last = i == len(kids) - 1
                stack.append((kids[i], lead + ("└─ " if last else "├─ "),
                              lead + ("   " if last else "│  ")))
    return out
