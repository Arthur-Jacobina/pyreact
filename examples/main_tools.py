"""Alias of ``python -m examples.tools``: the multi-agent store with tools.

    uv run python -m examples.main_tools

See ``examples/tools.py`` for what it teaches.
LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
"""

import asyncio

from examples.tools import main

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"stdin": "tem o SHIRT-RED em estoque?\n:q\n"}

if __name__ == "__main__":
    asyncio.run(main())
