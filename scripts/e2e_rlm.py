"""End-to-end RLM run over the repo source with the ``claude -p`` CLI LM.

Small on purpose: max 6 REPL iterations + 6 sub-LM calls (+1 extract) per
question, i.e. <= ~13 LLM calls.

    uv run python scripts/e2e_rlm.py
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from examples.rlm_agent import ask  # noqa: E402
from integrations import claude_cli_lm  # noqa: E402

QUESTION = (
    "In pyreact/core/runtime.py, which function schedules component rerenders, and "
    "which module-level variable prevents the same component from being enqueued "
    "twice? Answer with the two exact identifiers."
)


def main():
    t0 = time.time()
    out = ask(QUESTION, timeout=900)
    answers = out.get("answers", [])
    q, a, err = answers[-1] if answers else (QUESTION, None, "no answer")
    ends = [e.payload for e in out.get("events", []) if e.kind == "llm.end"]
    report = {
        "context_chars": out["context_chars"],
        "answer": a,
        "error": None if err is None else str(err)[:500],
        "llm_end": ends[-1] if ends else None,
        "events": [e.kind for e in out.get("events", [])],
        "cli_llm_calls": len(claude_cli_lm.CALLS),
        "seconds": round(time.time() - t0, 1),
        "mentions_expected": bool(a) and "schedule_rerender" in a and "_enqueued" in a,
    }
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    with open(os.path.join(ROOT, "artifacts", "rlm_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
