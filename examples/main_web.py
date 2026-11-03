"""Serve an example app in the browser: one session per tab, same components.

What it teaches: an app is transport-agnostic. ``run_web(Root)`` serves the
same ``Root`` the terminal runs; each browser tab gets its own session
(state, turns, router location, event bus), and ``message`` / ``turn.*`` /
``llm.*`` / ``tool.*`` events stream to the page over a WebSocket.

Run from the repo root::

    uv run python -m examples.main_web                      # router_app
    uv run python -m examples.main_web --app qa_guardrail   # or: tools
    # open http://127.0.0.1:8000  (try http://127.0.0.1:8000/products/HAT-WOOL?tab=specs)

    uv run python -m examples.main_web --check              # self-test, no server

``--check`` builds the same ASGI app ``run_web`` serves with
``make_web_app(Root)`` and drives it through starlette's ``TestClient`` (hello ->
submit -> ``turn.end``) and exits 0 if an assistant/system message came back.

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
"""

from __future__ import annotations

import argparse
import importlib
import sys

from pyreact.boot import make_web_app, run_web

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": ["--check", "--app", "router_app"], "timeout": 120}

APPS = {  # --app -> (module, a first message for --check)
    "router_app": ("examples.router_app", "show me the wool hat specs"),
    "qa_guardrail": ("examples.qa_guardrail", "why is the sky blue?"),
    "tools": ("examples.tools", "quais produtos vocês vendem?"),
}


def check(root, text: str, timeout: float = 180) -> int:
    """One WebSocket session: hello, submit ``text``, collect events until turn.end."""
    import time

    from starlette.testclient import TestClient

    app, _runner = make_web_app(root, capture_stdout=False)  # what run_web serves
    messages, deadline = [], time.time() + timeout
    with TestClient(app) as client:
        assert "<html" in client.get("/").text.lower()
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"t": "hello", "session": None, "path": "/"})
            ws.send_json({"t": "submit", "v": text})
            while time.time() < deadline:
                msg = ws.receive_json()
                ev = msg.get("event") or {}
                if ev.get("kind") == "message":
                    messages.append(ev["payload"])
                    print(f"  [{ev['payload']['sender']}] {ev['payload']['text']}")
                if ev.get("kind") == "turn.end":
                    print(f"  turn.end status={ev['payload'].get('status')}")
                    break
    replied = [m for m in messages if m["sender"] in ("assistant", "system")]
    print("web check:", "OK" if replied else "FAILED (no reply)")
    return 0 if replied else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app", choices=sorted(APPS), default="router_app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--check", action="store_true", help="self-test with TestClient and exit")
    args = ap.parse_args()
    module, first = APPS[args.app]
    root = importlib.import_module(module).Root
    if args.check:
        return check(root, first)
    run_web(root, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
