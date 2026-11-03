"""Run an example app in the terminal (kept for old docs and muscle memory).

Same as ``python -m examples.<app>``: ``create_root(Root)`` +
``run_repl(root)`` (``examples/_devtools.py``); each line you type is a turn, ``:help`` lists
the built-in commands (``:events``, ``:usage``, ``:turns``, ``:route``, ``:nav``).

    uv run python -m examples.main_terminal                     # router_app
    uv run python -m examples.main_terminal --app qa_guardrail  # or: tools

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
"""

from __future__ import annotations

import argparse
import importlib

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": ["--app", "qa_guardrail"], "stdin": "what is 2+2?\n:q\n"}

APPS = ("router_app", "qa_guardrail", "tools")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app", choices=APPS, default="router_app")
    app_module = importlib.import_module(f"examples.{ap.parse_args().app}")
    app_module.main()
