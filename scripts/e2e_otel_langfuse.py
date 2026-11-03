"""End-to-end: pyreact -> pyreact_otel -> Langfuse (real LLM, real Langfuse).

Runs the tool-calling store app (``examples.tools``) with
``pyreact_otel.setup(exporter="langfuse")`` and then checks, through the
Langfuse public API, that the traces and observations arrived.

    docker compose -f deploy/langfuse/docker-compose.yml up -d     # see deploy/langfuse/README.md
    PYREACT_EXAMPLE_LM=claude uv run python -m scripts.e2e_otel_langfuse [--max-calls 10]

Langfuse settings: ``LANGFUSE_HOST`` / ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` (default: the local stack's dev keys).

The runtime runs every turn, LLM call and tool call through the harness, so
the store app is traced as is: one trace per question (``turn`` root ->
``llm`` generations -> ``tool`` calls) in one Langfuse session.
``--synthetic`` adds a turn that drives ``through()`` directly (no LLM) to
show ``sandbox.exec``, a ``Blocked`` tool (WARNING) and a failing one (ERROR).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3100")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-pyreact-local")  # local dev stack defaults
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-pyreact-local")

import pyreact_otel  # noqa: E402
from integrations.claude_cli_lm import CALLS, ClaudeCLILM  # noqa: E402
from pyreact.core.harness import Blocked, Boundary, resolve_harness, through  # noqa: E402

QUESTIONS = [
    "Quanto custa a camisa vermelha (SHIRT-RED)?",
    "Quanto custa o frete do SHOE-RUN-42 para o BR?",
]


# ---------------------------------------------------------------------------
# The store app, traced by the real harness wiring
# ---------------------------------------------------------------------------
class BudgetLM(ClaudeCLILM):
    """``claude -p`` dev LM with a hard call budget (the e2e must stay cheap)."""

    def __init__(self, *a, max_calls: int = 10, **kw):
        super().__init__(*a, **kw)
        self.max_calls = max_calls

    async def aforward(self, prompt=None, messages=None, **kwargs):
        if len(CALLS) >= self.max_calls:
            raise RuntimeError(f"e2e LLM call budget ({self.max_calls}) exhausted")
        return await super().aforward(prompt=prompt, messages=messages, **kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):
        if len(CALLS) >= self.max_calls:
            raise RuntimeError(f"e2e LLM call budget ({self.max_calls}) exhausted")
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def run_store(session_id: str, max_calls: int, model: str, timeout: float) -> None:
    import examples.tools as store
    from pyreact.boot import bootstrap
    from pyreact.core.core import component

    lms = {k: BudgetLM(model, max_calls=max_calls) for k in ("default", "fast", "reasoning")}

    @component
    def Boot():
        return [store.Root(key="root", models=lms)]

    answers = {}
    app = bootstrap(Boot, default_session=False)
    try:
        session = app.open_session(session_id)
        session.event_bus.subscribe(
            lambda ev: answers.__setitem__(ev.turn_id, ev.payload.get("text"))
            if ev.payload.get("sender") == "assistant" else None, kinds=("message",))
        pyreact_otel.active().attach_bus(session.event_bus)  # message/error/feedback events
        for q in QUESTIONS:
            print(f"\n>>> {q}")
            try:
                tid = session.invoke(q, wait=True, timeout=timeout)
                print(f"<<< {answers.get(tid, '(no answer)')}")
            except TimeoutError as exc:
                print(f"[e2e] {exc}", file=sys.stderr)
    finally:
        app.shutdown()
    print(f"\n[e2e] LLM calls: {len(CALLS)} (budget {max_calls})")


# ---------------------------------------------------------------------------
# Synthetic turn: through() calls directly, no LLM
# ---------------------------------------------------------------------------
async def synthetic_turn(session_id: str) -> None:
    async def fake_llm(b):
        b.attrs.update(model="openai/gpt-4o-mini", cost=0.00042,
                       usage={"prompt_tokens": 321, "completion_tokens": 54, "total_tokens": 375},
                       lm_calls=[{"messages": [{"role": "user", "content": "preço do HAT-WOOL?"}],
                                  "completion": "R$ 19,50"}])
        return await through("tool", "lookup_price", {"sku": "HAT-WOOL"}, lambda t: _value(19.5))

    async def guard_blocked(b):
        raise Blocked("place_order needs human approval", boundary=b, by="approval-guard")

    async def failing(b):
        raise TimeoutError("sandbox timed out after 5s")

    async def body(b):
        await through("llm", "supervisor", {"message": "preço do HAT-WOOL?"}, fake_llm,
                      module_key="supervisor", arch="cot", call_id="synthetic-1")
        await through("sandbox.exec", "python", {"code": "print(1 + 1)"}, lambda s: _value("2\n"))
        for name, fn in (("place_order", guard_blocked), ("check_stock", failing)):
            try:
                await through("tool", name, {"sku": "HAT-WOOL"}, fn)
            except (Blocked, TimeoutError):
                pass
        return "ok"

    b = Boundary(kind="turn", name="SyntheticTurn", inputs={"text": "preço do HAT-WOOL?"},
                 session_id=session_id, turn_id=f"{session_id}:t1", attrs={"user_id": "e2e-user"})
    await resolve_harness().run(b, body)


async def _value(v):
    return v


# ---------------------------------------------------------------------------
# Langfuse public API (v4: observations v2)
# ---------------------------------------------------------------------------
def _get(path: str, params: dict) -> dict:
    host = os.environ["LANGFUSE_HOST"].rstrip("/")
    auth = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()
    req = urllib.request.Request(f"{host}{path}?{urllib.parse.urlencode(params)}",
                                 headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def observations(session_id: str, since: datetime) -> list:
    rows, cursor = [], None
    while True:
        params = {"sessionId": session_id, "limit": 1000,
                  "fromStartTime": since.isoformat().replace("+00:00", "Z"),
                  "toStartTime": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
                  "fields": "core,basic,usage,model,trace_context"}
        if cursor:
            params["cursor"] = cursor
        page = _get("/api/public/v2/observations", params)
        rows += page.get("data", [])
        cursor = (page.get("meta") or {}).get("cursor")
        if not cursor or not page.get("data"):
            return rows


def wait_for(session_id: str, since: datetime, minimum: int, timeout: float = 300) -> list:
    deadline, rows, last = time.time() + timeout, [], -1
    while time.time() < deadline:
        try:
            rows = observations(session_id, since)
        except (OSError, ValueError) as exc:  # cold/restarting langfuse-web: retry
            print(f"[e2e] read API not ready ({type(exc).__name__}: {exc}); retrying", file=sys.stderr)
            time.sleep(5)
            continue
        if len(rows) >= minimum and len(rows) == last:  # arrived and stable
            return rows
        last = len(rows)
        time.sleep(3)
    return rows


def report(label: str, rows: list) -> dict:
    by_type, traces, tokens, cost = {}, {}, {"input": 0, "output": 0, "total": 0}, 0.0
    for r in rows:
        by_type[r.get("type")] = by_type.get(r.get("type"), 0) + 1
        traces.setdefault(r.get("traceId"), r.get("traceName"))
        for k in tokens:
            v = (r.get("usageDetails") or {}).get(k)
            if isinstance(v, (int, float)):
                tokens[k] += v
        if isinstance(r.get("totalCost"), (int, float)):
            cost += r["totalCost"]
    print(f"\n[{label}] session={rows[0].get('sessionId') if rows else None} "
          f"observations={len(rows)} traces={len(traces)}")
    for tid, name in traces.items():
        print(f"  trace {tid}  name={name!r}")
    print(f"  types: {dict(sorted(by_type.items(), key=lambda kv: str(kv[0])))}")
    print(f"  tokens: {tokens}  cost: {cost:.6f}")
    for r in sorted(rows, key=lambda r: r.get("startTime") or ""):
        lvl = r.get("level")
        extra = f" model={r.get('providedModelName') or r.get('model')}" if r.get("type") == "GENERATION" else ""
        print(f"    {r.get('type'):<10} {r.get('name')!s:<40} level={lvl}{extra}"
              + (f"  status={r.get('statusMessage')!r}" if r.get("statusMessage") else ""))
    return {"types": by_type, "traces": traces, "tokens": tokens}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--skip-app", action="store_true", help="don't run the store app (no LLM)")
    ap.add_argument("--synthetic", action="store_true",
                    help="also drive through() directly: sandbox.exec, a Blocked and a failing tool")
    args = ap.parse_args()

    run = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    since = datetime.now(timezone.utc) - timedelta(minutes=1)
    otel = pyreact_otel.setup(exporter="langfuse", service_name="pyreact-e2e",
                              capture_content=True, environment="e2e", tags=["pyreact", "e2e"])
    print(f"[e2e] exporting to {otel.endpoint}")

    app_session = f"e2e-store-{run}"
    direct_session = f"e2e-direct-{run}"
    try:
        if args.synthetic:
            asyncio.run(synthetic_turn(direct_session))
        if not args.skip_app:
            run_store(app_session, args.max_calls, args.model, args.timeout)
    finally:
        otel.shutdown()  # flushes

    ok = True
    if not args.skip_app:
        store_rep = report("store app", wait_for(app_session, since, minimum=3))
        want = {"AGENT", "GENERATION", "TOOL"}
        if not want <= set(store_rep["types"]) or len(store_rep["traces"]) < len(QUESTIONS):
            print(f"[e2e] FAIL: want {want} and one trace per question")
            ok = False
    if args.synthetic:
        direct = report("synthetic through()", wait_for(direct_session, since, minimum=6))
        want = {"AGENT", "GENERATION", "TOOL", "SPAN"}
        if not want <= set(direct["types"]):
            print(f"[e2e] FAIL: synthetic session lacks {want - set(direct['types'])}")
            ok = False
    print("\n[e2e] OK" if ok else "\n[e2e] FAILED")
    print(f"[e2e] UI: {os.environ['LANGFUSE_HOST']}/project/pyreact-dev/sessions")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
