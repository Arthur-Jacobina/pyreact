"""End-to-end check of the turn model on the multi-agent store (real LLM).

Uses the `claude -p` CLI as the LM (no API key needed):

    uv run python -m scripts.e2e_turns

Scenarios:
  (a) queue policy: 2 questions 0.3s apart -> 2 answers, in order, correctly routed
  (b) the same question twice -> 2 answers
  (c) interrupt policy: 2nd question interrupts the 1st -> 1st ends 'interrupted',
      2nd is answered
"""

import sys
import threading
import time

from examples.tools import Root
from integrations import claude_cli_lm
from integrations.claude_cli_lm import ClaudeCLILM
from pyreact.boot import bootstrap
from pyreact.core import component

TIMEOUT = 300
T0 = time.time()
TIMELINE = []
_lock = threading.Lock()


def log(line: str) -> None:
    line = f"[{time.time() - T0:7.2f}s] {line}"
    with _lock:
        TIMELINE.append(line)
    print(line, flush=True)


def start_app(policy: str):
    # Each app's default session owns its services (bus, turns, ...).
    lm = ClaudeCLILM("haiku")
    models = {"default": lm, "fast": lm, "reasoning": lm}

    @component
    def Boot():
        return [Root(key="root", models=models)]

    app = bootstrap(Boot, turn_policy=policy)
    answers, ends = {}, {}
    bus = app.event_bus

    def on_event(ev):
        p = ev.payload
        if ev.kind == "turn.start":
            log(f"{ev.turn_id} turn.start   text={p['text']!r}")
        elif ev.kind == "turn.end":
            ends[ev.turn_id] = p["status"]
            log(f"{ev.turn_id} turn.end     status={p['status']} duration={p['duration']:.2f}s")
        elif ev.kind == "agent.route":
            log(f"{ev.turn_id} agent.route  -> {p['specialist']}")
        elif ev.kind == "agent.answer":
            answers.setdefault(ev.turn_id, []).append(p)
            text = (p["answer"] or p["error"] or "").replace("\n", " ")
            log(f"{ev.turn_id} agent.answer [{p['specialist']}] {text[:110]}")

    bus.subscribe(on_event, ["turn.*", "agent.*"])
    return app, answers, ends


def check(cond: bool, what: str, failures: list) -> None:
    log(("PASS " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list = []

    # (a) burst under the default queue policy -----------------------------
    log("=== (a) queue: burst of 2 questions 0.3s apart ===")
    app, answers, ends = start_app("queue")
    try:
        q1 = "Quais produtos vocês vendem?"
        q2 = "Quanto custa o frete do SHOE-RUN-42 para o BR?"
        t1 = app.invoke(q1)
        time.sleep(0.3)  # the burst itself, not a sync
        t2 = app.invoke(q2)
        app.wait_turn(t2, timeout=TIMEOUT)
        check(ends.get(t1) == "done" and ends.get(t2) == "done", "both turns done", failures)
        check(len(answers.get(t1, [])) == 1 and len(answers.get(t2, [])) == 1,
              "exactly one answer per turn", failures)
        a1, a2 = answers.get(t1, [{}])[0], answers.get(t2, [{}])[0]
        check(a1.get("question") == q1 and a2.get("question") == q2,
              "answers in order, each for its own question", failures)
        check(a1.get("specialist") == "catalog" and a2.get("specialist") == "shipping",
              "routed catalog -> shipping", failures)
    finally:
        app.shutdown()

    # (b) the same question twice ------------------------------------------
    log("=== (b) queue: same question twice ===")
    app, answers, ends = start_app("queue")
    try:
        q = "Tem o SHIRT-RED em estoque?"
        t1 = app.invoke(q, wait=True, timeout=TIMEOUT)
        t2 = app.invoke(q, wait=True, timeout=TIMEOUT)
        check(t1 != t2, "distinct turn ids", failures)
        check(len(answers.get(t1, [])) == 1 and len(answers.get(t2, [])) == 1,
              "two answers (one per turn)", failures)
        check(all(a["specialist"] == "order" for t in (t1, t2) for a in answers.get(t, [])),
              "both routed to order", failures)
    finally:
        app.shutdown()

    # (c) interrupt policy -------------------------------------------------
    log("=== (c) interrupt: 2nd question interrupts the 1st ===")
    app, answers, ends = start_app("interrupt")
    try:
        q1 = "Quais produtos vocês vendem?"
        q2 = "Quanto custa o frete do HAT-WOOL para o BR?"
        t1 = app.invoke(q1)
        time.sleep(0.3)
        t2 = app.invoke(q2, wait=True, timeout=TIMEOUT)
        app.wait_turn(t1, timeout=5)
        check(ends.get(t1) == "interrupted", "first turn interrupted", failures)
        check(not answers.get(t1), "no answer for the interrupted turn", failures)
        check(ends.get(t2) == "done" and len(answers.get(t2, [])) == 1
              and answers[t2][0]["specialist"] == "shipping",
              "second turn answered by shipping", failures)
    finally:
        app.shutdown()

    calls = claude_cli_lm.CALLS
    log(f"LLM calls completed: {len(calls)} (cancelled calls are not counted)")
    log("RESULT: " + ("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
