"""01 - Hello durable: a counter that survives restarts.

Two hooks, both ``use_state``-like, both restored from a session log:

- ``use_persistent_reducer(reducer, 0, key="count")``: the count, changed by
  actions (``+``, ``-``, ``reset``); the *state* is saved after each action.
- ``use_persistent_state(0, key="presses")``: how many actions you typed.

The only other durable thing is ``Durable(log=...)`` around the tree: it opens
the log on mount, hands it to the hooks below, and closes it on shutdown.
Nothing else in the app knows about persistence.

    uv run python -m examples.durable.01_counter        # type + + - then :q
    uv run python -m examples.durable.01_counter        # again: same count

Look at what was stored (each set_* call is one ``state`` entry):

    uv run python -m pyreact_durable state runs/durable/counter.jsonl
    uv run python -m pyreact_durable show runs/durable/counter.jsonl

No LLM. Delete the file to start from zero.
"""

from __future__ import annotations

import argparse

from examples._devtools import run_repl
from pyreact.boot import create_root
from pyreact.components.keystroke import Keystroke
from pyreact.components.message import Message
from pyreact.core import component
from pyreact_durable import Durable, use_persistent_reducer, use_persistent_state

# Offline smoke run (tests/test_examples_smoke.py).
SMOKE = {"args": ["--log", "{tmp}/counter.jsonl"], "stdin": "+\n+\n-\nreset\n+\n:q\n"}


def reducer(count: int, action: str) -> int:
    return {"+": count + 1, "-": count - 1, "reset": 0}.get(action, count)


@component
def Counter():
    count, dispatch = use_persistent_reducer(reducer, 0, key="count")
    presses, set_presses = use_persistent_state(0, key="presses")

    def on_submit(text: str) -> None:
        action = text.strip()
        if action in ("+", "-", "reset"):
            dispatch(action)
            set_presses(lambda n: n + 1)

    return [
        Keystroke(key="keys", on_submit=on_submit),
        # published on mount (the restored values) and whenever the text changes
        Message(key="count", text=f"count={count}  (presses so far: {presses})",
                sender="assistant"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/durable/counter.jsonl")
    log = ap.parse_args().log

    @component
    def App():  # address: App/Counter[counter] (providers add no path segment)
        return [Durable(key="durable", log=log, children=[Counter(key="counter")])]

    print(f"state lives in {log}  (inspect: python -m pyreact_durable state {log})")
    print("type +, - or reset; :q to quit")
    with create_root(App, session_id="counter") as root:
        run_repl(root, prompt="> ")


if __name__ == "__main__":
    main()
