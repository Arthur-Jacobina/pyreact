"""02 - Branches as a feature, on the SQLite backend.

A shopping list kept with ``use_persistent_state`` in ``state.sqlite`` (the
suffix picks the backend: ``.sqlite``/``.db`` -> SQLite, anything else ->
JSONL; same entries, same API). The component also grabs the log itself with
``use_durable()`` and turns its tree operations into commands:

    add <item>          append to the list (one ``state`` entry)
    /label <name>       bookmark the current position
    /checkout <label>   jump to a bookmark (or an entry id)
    /fork <label>       copy history up to <label> into a NEW file
    /tree               print the session tree

What happens on ``/checkout base``: the log writes a ``leaf`` entry pointing
at ``base``. ``Durable`` notices the leaf moved and every persistent hook
reloads its value from the state *at that entry*, so the list goes back to
what it was when you labelled it. Nothing is deleted: the newer items are
still in the file, on the branch you left (``/checkout`` back to reach them).
The next ``add`` starts a new branch from ``base`` (``⑂2`` in ``/tree``). The
leaf entry is persisted too, so a restart resumes on the branch you were on.

``/fork <label>`` is the other way to branch: ``fork_file`` writes a new
session file (``<name>-<label>.sqlite``, header ``parentSession`` -> this file)
with only the root->label path; run with ``--log`` on it to continue there.

    uv run python -m examples.durable.02_branches_sqlite
    > add milk / /label base / add eggs / /checkout base / add bread / /tree / :q
    uv run python -m examples.durable.02_branches_sqlite     # same branch, same list

    uv run python -m pyreact_durable show runs/durable/state.sqlite
    uv run python -m pyreact_durable state runs/durable/state.sqlite --at base

No LLM.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pyreact.boot import bootstrap, read_terminal_and_invoke
from pyreact.components.keystroke import Keystroke
from pyreact.core import component, hooks, use_emit
from pyreact_durable import Durable, SessionLogError, use_durable, use_persistent_state

# Offline smoke run (tests/test_examples_smoke.py).
SMOKE = {"args": ["--log", "{tmp}/state.sqlite"],
         "stdin": "add milk\n/label base\nadd eggs\n/label more\n/checkout base\n"
                  "add bread\n/tree\n/fork more\n/checkout more\n/checkout nope\n:q\n"}


@component
def ShoppingList():
    items, set_items = use_persistent_state([], key="items")
    log = use_durable()
    emit = use_emit()

    def say(text: str) -> None:
        emit("message", {"sender": "assistant", "text": text})

    def on_submit(text: str) -> None:
        cmd, _, arg = text.strip().partition(" ")
        arg = arg.strip()
        try:
            if cmd == "add" and arg:
                set_items(lambda xs: [*xs, arg])
            elif cmd == "/label" and arg:
                log.label(arg)
                say(f"labelled the current position {arg!r}")
            elif cmd == "/checkout" and arg:
                say(f"leaf -> {log.checkout(arg)}")  # hooks reload; the effect below prints
            elif cmd == "/fork" and arg:
                path = log.path.with_name(f"{log.path.stem}-{arg}{log.path.suffix}")
                with log.fork_file(path, at=arg) as new:
                    say(f"forked {len(new.entries)} entries into {path} (--log {path})")
            elif cmd == "/tree":
                say(log.render(width=40))
            else:
                say("add <item> | /label <name> | /checkout <label> | /fork <label> | /tree")
        except SessionLogError as exc:  # unknown label, file already exists, ...
            say(f"error: {exc}")

    hooks.use_effect(lambda: say(f"list: {items or '(empty)'}"), [items])
    return [Keystroke(key="keys", on_submit=on_submit)]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/durable/state.sqlite")
    log = ap.parse_args().log
    Path(log).parent.mkdir(parents=True, exist_ok=True)

    @component
    def App():
        return [Durable(key="durable", log=log, children=[ShoppingList(key="list")])]

    app = bootstrap(App)
    print(f"durable list in {log}; :q to quit")
    try:
        # "/label" etc. are app input: terminal commands use ":" (":q", ":help").
        await read_terminal_and_invoke(app, prompt="> ")
    finally:
        app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
