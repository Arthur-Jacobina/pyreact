"""End-to-end check of durable agent memory against a real LM (`claude -p`).

Drives ``examples/durable_chat.py`` in child processes and kills them with
SIGKILL (no graceful shutdown):

  (a) tell a fact, kill -9, restart -> the agent remembers it
  (b) label, fork to a branch, ask something there, checkout back
      -> original branch intact (also after another kill -9 + restart)
  (c) crash right after the LM answer is journaled but before the state
      update; re-run the turn with durable_lm(mode="replay")
      -> identical answer, 0 model calls

    uv run python scripts/e2e_durable.py [--keep]

~6 real LM calls (haiku). The child protocol: lines on stdin are chat input;
the child prints ``@@{json}`` lines (messages, per-turn summaries).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TURN_TIMEOUT = 240


# --------------------------------------------------------------------- child
def child(log_path: str, mode: str, crash_after_llm: bool) -> None:
    from examples.durable_chat import start
    from integrations import claude_cli_lm
    from pyreact_durable import SessionLog

    def emit(obj):
        print("@@" + json.dumps(obj), flush=True)

    log = SessionLog.open(log_path)
    if crash_after_llm:
        # Die the instant the LM response is on disk, before the transcript
        # (a later state entry) is written: the classic lost-update window.
        def _die(entry, _moved):
            if entry["type"] == "llm":
                emit({"crashing_after": entry["id"]})
                os.kill(os.getpid(), signal.SIGKILL)

        log.subscribe(_die)

    app = start(log, claude_cli_lm.ClaudeCLILM("haiku"), lm_mode=mode)
    app.event_bus.subscribe(
        lambda ev: emit({"msg": ev.payload.get("text"), "sender": ev.payload.get("sender")}),
        ["message"],
    )
    emit({"ready": True, "leaf": log.leaf})
    for line in iter(sys.stdin.readline, ""):
        line = line.rstrip("\n")
        app.invoke(line, wait=True, timeout=TURN_TIMEOUT)
        emit({"done": line, "calls": len(claude_cli_lm.CALLS)})
    app.shutdown()
    log.close()


# -------------------------------------------------------------------- parent
class Child:
    def __init__(self, log_path: str, mode: str = "replay_or_record", crash_after_llm=False):
        argv = [sys.executable, __file__, "--child", log_path, "--mode", mode]
        if crash_after_llm:
            argv.append("--crash-after-llm")
        self.seen_calls = 0
        self.p = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.read_until(lambda o: o.get("ready"))

    def read_until(self, pred):
        got = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                return got + [{"eof": True}]
            if not line.startswith("@@"):
                continue
            obj = json.loads(line[2:])
            got.append(obj)
            if pred(obj):
                return got

    def say(self, text: str) -> dict:
        t0 = time.time()
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()
        got = self.read_until(lambda o: "done" in o)
        msgs = [o["msg"] for o in got if "msg" in o]
        done = next((o for o in got if "done" in o), {})
        total = done.get("calls")
        delta = None if total is None else total - self.seen_calls
        if total is not None:
            self.seen_calls = total
        return {"msgs": msgs, "last": msgs[-1] if msgs else "", "calls": delta,
                "eof": any(o.get("eof") for o in got), "secs": time.time() - t0,
                "raw": got}

    def kill9(self) -> None:
        os.kill(self.p.pid, signal.SIGKILL)
        self.p.wait(10)


RESULTS: list = []
T0 = time.time()


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{time.time() - T0:6.1f}s] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def short(s: str, n: int = 70) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def parent(keep: bool) -> int:
    from pyreact_durable import SessionLog
    from pyreact_durable.__main__ import render_tree

    tmp = Path(tempfile.mkdtemp(prefix="e2e_durable_"))
    log_path = str(tmp / "chat.jsonl")
    calls = 0
    # stable component address of the chat agent (its persistent state key)
    chat_path = ("App/DurableChatApp[app]/Durable[durable]/LMLayer[lm]/DSPyProvider[dspy]/"
                 "_Provider[#0]/DurableChat[chat]")

    # (a) remember across kill -9 ---------------------------------------
    c = Child(log_path)
    r = c.say("Please remember this: my favorite color is teal. Just acknowledge briefly.")
    calls += r["calls"]
    check("a.tell", bool(r["last"]), short(r["last"]))
    c.kill9()
    c = Child(log_path)
    r = c.say("What is my favorite color? Answer with just the color.")
    calls += r["calls"]
    check("a.remembers after kill -9", "teal" in r["last"].lower(), short(r["last"]))

    # (b) label / fork / checkout ----------------------------------------
    c.say("/label base")
    r = c.say("Also remember: my dog's name is Rex. Acknowledge briefly.")
    calls += r["calls"]
    c.say("/label main")
    c.say("/fork base")
    r = c.say("What is my dog's name? If I never told you, reply exactly UNKNOWN.")
    calls += r["calls"]
    check("b.fork forgets the later fact", "rex" not in r["last"].lower(), short(r["last"]))
    c.say("/label alt")
    c.say("/checkout main")
    h = c.say("/history")["last"]
    check("b.checkout main restores branch",
          "Rex" in h and "never told you" not in h, f"{h.count(chr(10)) + 1} lines")
    c.kill9()
    c = Child(log_path)
    h = c.say("/history")["last"]
    check("b.branch survives kill -9", "Rex" in h and "never told you" not in h)
    c.kill9()
    with SessionLog.open(log_path, readonly=True) as log:
        alt = log.get_state(chat_path, "history", [], entry_id="alt")
        main = log.get_state(chat_path, "history", [], entry_id="main")
        check("b.both branches in one file",
              any("never told you" in m["content"] for m in alt)
              and not any("never told you" in m["content"] for m in main)
              and len(log.branch_points()) >= 1,
              f"alt={len(alt)} msgs, main={len(main)} msgs, "
              f"{len(log.branch_points())} branch point(s)")

    # (c) crash after the LM answer, replay ------------------------------
    q = "In one short sentence, suggest a name for a teal-colored boat."
    c = Child(log_path, crash_after_llm=True)
    r = c.say(q)
    calls += 1  # the call that happened before the crash
    check("c.crashed after journaling the LM call", r["eof"])
    with SessionLog.open(log_path, readonly=True) as log:
        last_llm = [e for e in log.path_to() if e["type"] == "llm"][-1]
        raw = last_llm["response"]["choices"][0]["message"]["content"]
        hist_len = len(log.get_state(chat_path, "history", []))
    c = Child(log_path, mode="replay")
    r = c.say(q)
    calls += r["calls"]
    check("c.replay: 0 model calls", r["calls"] == 0, f"calls={r['calls']}")
    check("c.replay: identical answer", bool(r["last"]) and r["last"] in raw,
          short(r["last"]))
    c.kill9()
    with SessionLog.open(log_path, readonly=True) as log:
        hist = log.get_state(chat_path, "history", [])
        check("c.turn completed once", len(hist) == hist_len + 2,
              f"history {hist_len} -> {len(hist)}")
        tree = render_tree(log, width=50)

    # report ---------------------------------------------------------------
    ok = all(r[1] for r in RESULTS)
    print("\n== durable e2e report ==")
    for name, good, detail in RESULTS:
        print(f"  {'PASS' if good else 'FAIL'}  {name:<42} {detail}")
    print(f"  real LM calls: {calls}   wall: {time.time() - T0:.0f}s   log: {log_path}")
    # the tree's skeleton: labels, branch points, LM calls and the leaf
    keep_ln = [ln for ln in tree[1:] if any(t in ln for t in ("  [", "⑂", "<- leaf", " llm "))]
    print("\n".join("  " + ln for ln in tree[:1] + keep_ln))
    print(f"\n{'ALL PASS' if ok else 'FAILURES'} ({sum(r[1] for r in RESULTS)}/{len(RESULTS)})")
    if not keep:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--mode", default="replay_or_record")
    ap.add_argument("--crash-after-llm", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.mode, a.crash_after_llm)
    else:
        sys.exit(parent(a.keep))
