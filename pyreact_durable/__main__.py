"""``python -m pyreact_durable`` — inspect and steer session logs.

    show <file>                         tree: labels, branch points, leaf (*)
    state <file> [--at REF]             folded state as JSON
    events <file> [--kind K ...] [--at REF]
    label <file> <name> [--at REF]
    checkout <file> <REF>               move the leaf (survives restarts)
    fork <file> <new_file> [--at REF]   new file with parentSession
    checkpoint <file> [--summary S]

REF is a label, an entry id or a unique id prefix (``@leaf``/``@root`` too).
Read commands open the file read-only (fine while an app is writing it);
write commands take the single-writer lock.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

from .log import SessionLog, SessionLogError


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


def render_tree(log: SessionLog, width: int = 60) -> List[str]:
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


def _ro(path: str) -> SessionLog:
    return SessionLog.open(path, readonly=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyreact_durable",
                                 description="Inspect and steer pyreact_durable session logs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show", help="render the entry tree")
    p.add_argument("file")
    p.add_argument("--width", type=int, default=60)
    p = sub.add_parser("state", help="state at the leaf or --at")
    p.add_argument("file")
    p.add_argument("--at")
    p = sub.add_parser("events", help="journaled events on the path")
    p.add_argument("file")
    p.add_argument("--kind", action="append")
    p.add_argument("--at")
    p = sub.add_parser("label", help="bookmark an entry")
    p.add_argument("file")
    p.add_argument("name")
    p.add_argument("--at")
    p = sub.add_parser("checkout", help="move the leaf")
    p.add_argument("file")
    p.add_argument("ref")
    p = sub.add_parser("fork", help="copy the path into a new session file")
    p.add_argument("file")
    p.add_argument("new_file")
    p.add_argument("--at")
    p = sub.add_parser("checkpoint", help="append a state snapshot")
    p.add_argument("file")
    p.add_argument("--summary")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "show":
            with _ro(args.file) as log:
                print("\n".join(render_tree(log, args.width)))
        elif args.cmd == "state":
            with _ro(args.file) as log:
                print(json.dumps(log.state_at(args.at), indent=2, ensure_ascii=False))
        elif args.cmd == "events":
            with _ro(args.file) as log:
                for e in log.events(args.kind, entry_id=args.at):
                    tid = f" [{e['turn_id']}]" if e.get("turn_id") else ""
                    print(f"{e['timestamp']} {e['id']} {e['kind']}{tid} "
                          f"{_brief(e.get('payload'), 100)}")
        elif args.cmd == "label":
            with SessionLog.open(args.file, create=False) as log:
                e = log.label(args.name, args.at)
                print(f"labelled {e['targetId']} as {args.name!r}")
        elif args.cmd == "checkout":
            with SessionLog.open(args.file, create=False) as log:
                print(f"leaf -> {log.checkout(args.ref)}")
        elif args.cmd == "fork":
            with SessionLog.open(args.file, readonly=True) as log:
                with log.fork_file(args.new_file, at=args.at) as new:
                    print(f"forked {len(new.entries)} entries into {args.new_file} "
                          f"(session {new.id}, leaf {new.leaf})")
        elif args.cmd == "checkpoint":
            with SessionLog.open(args.file, create=False) as log:
                print(f"checkpoint {log.checkpoint(args.summary)['id']}")
    except (SessionLogError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
