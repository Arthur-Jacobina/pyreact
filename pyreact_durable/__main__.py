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
from typing import List, Optional

from .log import SessionLog, SessionLogError
from .render import _brief, render_tree, summarize  # noqa: F401  (re-exported; public: pyreact_durable.render_tree)


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
