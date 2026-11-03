"""03 - Web app: every browser tab is a session with its own durable log.

``run_web`` mounts the app once per WebSocket session (one per tab; the tab
keeps its session id in ``sessionStorage``, so a reload asks for the same id).
``Durable(log_factory=...)`` turns that id into a file::

    Durable(log_factory=lambda session_id: f"runs/web/{session_id}.sqlite", ...)

The factory is called with the ``"session"`` service's id when the session
mounts (escaped with ``safe_session_id``, so it is always safe in a path), and
the log is closed when the session closes (tab gone for ``session_ttl`` s, or
server stop). Logs of sessions closed by the TTL are kept by default; pass
``Durable(..., on_close="archive" | "delete" | fn)`` to clean them up (server
shutdown never triggers it). Inside, ``use_conversation()`` is the persisted transcript, so:

- two tabs never see each other's conversation (two files);
- reloading a tab, or restarting the server and reloading, restores it: on
  every (re)connect the browser sends ``hello`` -> a ``nav`` event, and the
  component re-sends the transcript so the page can draw it again.

    uv run python -m examples.durable.03_web_sessions          # http://127.0.0.1:8000
    uv run python -m examples.durable.03_web_sessions --check  # self-test, no browser

    ls runs/web/
    uv run python -m pyreact_durable show runs/web/<session id>.sqlite

``--check`` drives ``make_web_app(App)`` (the app ``run_web`` serves) with
starlette's TestClient: two sockets -> two
sessions with separate histories; then a *new server* and a socket asking for
the first id -> its history is back (and the LM sees it); log files on disk.

LM: examples._lm (claude -p by default; PYREACT_EXAMPLE_LM=openai|offline).
``--check`` makes 3 LM calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dspy

from examples._lm import OFFLINE, example_lm
from integrations import DSPyProvider, use_agent_call
from pyreact.boot import make_web_app, run_web
from pyreact.core import EventBus, component, hooks, use_on_turn
from pyreact_durable import Durable, use_conversation

# Offline smoke run (tests/test_examples_smoke.py).
SMOKE = {"args": ["--check", "--logs", "{tmp}/web"], "timeout": 120}


class ChatSig(dspy.Signature):
    """Friendly, concise assistant. Use the conversation history. One sentence."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


@component
def Chat():
    conv = use_conversation()  # this session's transcript, restored from its file
    events = hooks.get_service("event_bus", EventBus)
    call = use_agent_call(ChatSig)
    latest = hooks.use_ref(conv)  # read by the long-lived nav listener below
    latest.current = conv

    def say(text: str, sender: str = "assistant") -> None:
        events.publish("message", {"text": text, "sender": sender, "message_type": "chat"})

    def redraw_on_connect():
        def on_nav(_ev):
            msgs = latest.current.messages
            for m in msgs:
                say(m["content"], m["role"])
            say(f"({len(msgs)} message(s) restored)", "system")

        return events.subscribe(on_nav, ["nav"])

    hooks.use_effect(redraw_on_connect, [])

    def ask(turn):  # once per input (a repeated question is a new turn)
        if turn.text.strip():
            call.run(question=turn.text.strip(), history=conv.as_dspy_history())

    use_on_turn(ask)

    def remember():
        if call.result is not None:  # call.inputs: what the settled call was asked
            conv.extend([{"role": "user", "content": call.inputs["question"]},  # one entry per pair
                         {"role": "assistant", "content": call.result.answer}])
            say(call.result.answer)
        elif call.error is not None:
            say(f"LM error: {call.error}", "system")

    hooks.use_effect(remember, [call.ver])
    return []


def make_app(logs: str, lm):
    @component
    def App():
        return [Durable(key="durable", log_factory=lambda sid: f"{logs}/{sid}.sqlite", children=[
            DSPyProvider(key="dspy", lm=lm, children=[Chat(key="chat")]),
        ])]

    return App


# ----------------------------------------------------------------- --check
def check(logs: str, lm) -> None:
    from contextlib import ExitStack

    from starlette.testclient import TestClient

    def server():  # the ASGI app run_web serves; its lifespan stops the runner
        return TestClient(make_web_app(make_app(logs, lm))[0])

    def recv_until(ws, sender, text=None):
        got = []
        while True:
            m = ws.receive_json()
            p = (m.get("event") or {}).get("payload") or {}
            if m.get("channel") == "events" and p.get("sender"):
                got.append(f"{p['sender']}: {p['text']}")
                if p["sender"] == sender and (text is None or text in p["text"]):
                    return got

    def connect(tabs, client, sid=None):  # what app.js does when a tab opens
        ws = tabs.enter_context(client.websocket_connect("/ws"))
        ws.send_json({"t": "hello", "session": sid, "path": "/"})
        hello = ws.receive_json()["data"]  # {"id", "resumed"}
        return ws, hello, recv_until(ws, "system", "restored")

    with server() as client, ExitStack() as tabs:  # first server: two tabs
        (ws1, s1, _), (ws2, s2, _) = connect(tabs, client), connect(tabs, client)
        for ws, fact in ((ws1, "My name is Ada."), (ws2, "My name is Bob.")):
            ws.send_json({"t": "submit", "v": fact})
            print(f"[{fact}] ->", recv_until(ws, "assistant")[-1])
    files = sorted(p.name for p in Path(logs).glob("*.sqlite"))
    print("logs on disk:", files)
    assert {f"{s1['id']}.sqlite", f"{s2['id']}.sqlite"} <= set(files)

    with server() as client, ExitStack() as tabs:  # restart; tab 1 reloads its id
        ws, hello, restored = connect(tabs, client, s1["id"])
        print("tab 1 after restart:", hello, restored)
        assert hello == {"id": s1["id"], "resumed": False}  # a new session object...
        assert any("Ada" in m for m in restored) and not any("Bob" in m for m in restored)
        ws.send_json({"t": "submit", "v": "What is my name?"})
        answer = recv_until(ws, "assistant")[-1]
        print("[What is my name?] ->", answer)
        assert OFFLINE or "Ada" in answer  # ...with the old memory
    print("check ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="runs/web")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.check:
        check(args.logs, example_lm())
    else:
        run_web(make_app(args.logs, example_lm()), port=args.port)


if __name__ == "__main__":
    main()
