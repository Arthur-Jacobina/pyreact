"""End-to-end: pyreact -> pyreact_otel -> a generic OTLP endpoint (OTel Collector).

The vendor-neutral twin of ``scripts/e2e_otel_langfuse.py``: runs the store
app (``examples.tools``, real LLM, <= 10 calls) with
``pyreact_otel.setup(exporter="otlp")`` against the collector of
``deploy/otel-collector`` and checks what it wrote (``out/otlp.jsonl``):
the span tree per turn (turn -> llm -> tool), attributes and metrics.

    docker compose -f deploy/otel-collector/docker-compose.yml up -d
    PYREACT_EXAMPLE_LM=claude uv run python -m scripts.e2e_otel_collector
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyreact_otel
from scripts.e2e_otel_langfuse import QUESTIONS, run_store

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "otel-collector" / "out" / "otlp.jsonl"


def _attrs(kvs) -> dict:
    out = {}
    for kv in kvs or []:
        v = kv.get("value", {})
        out[kv["key"]] = next(iter(v.values()), None) if v else None
    return out


def read_export(offset: int):
    spans, metrics = [], defaultdict(float)
    with open(OUT, encoding="utf-8") as fh:
        fh.seek(offset)
        for line in fh:
            req = json.loads(line)
            for rs in req.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for s in ss.get("spans", []):
                        spans.append({**s, "attrs": _attrs(s.get("attributes"))})
            for rm in req.get("resourceMetrics", []):
                for sm in rm.get("scopeMetrics", []):
                    for m in sm.get("metrics", []):
                        data = m.get("histogram") or m.get("sum") or {}
                        for p in data.get("dataPoints", []):
                            a = _attrs(p.get("attributes"))
                            key = m["name"] + (f"[{a['gen_ai.token.type']}]" if "gen_ai.token.type" in a else "")
                            val = p.get("sum", p.get("asDouble", p.get("asInt", 0)))
                            metrics[key] = max(metrics[key], float(val or 0))  # cumulative
    return spans, dict(metrics)


def print_tree(spans: list) -> None:
    by_id = {s["spanId"]: s for s in spans}
    kids = defaultdict(list)
    for s in spans:
        kids[s.get("parentSpanId") or ""].append(s)

    def show(s, depth):
        a = s["attrs"]
        extra = ""
        if "gen_ai.request.model" in a:
            extra = (f"  model={a['gen_ai.request.model']} in={a.get('gen_ai.usage.input_tokens')}"
                     f" out={a.get('gen_ai.usage.output_tokens')}")
        if "pyreact.turn.status" in a:
            extra = f"  status={a['pyreact.turn.status']} turn={a.get('pyreact.turn_id')}"
        err = " [ERROR]" if (s.get("status") or {}).get("code") == 2 else ""
        print(f"  {'  ' * depth}{s['name']}{extra}{err}")
        for k in sorted(kids[s["spanId"]], key=lambda x: x.get("startTimeUnixNano", "0")):
            show(k, depth + 1)

    roots = [s for s in spans if not s.get("parentSpanId") or s["parentSpanId"] not in by_id]
    for r in sorted(roots, key=lambda x: x.get("startTimeUnixNano", "0")):
        show(r, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4320"))
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    offset = OUT.stat().st_size if OUT.exists() else 0
    session_id = "e2e-otlp-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    pyreact_otel.setup(exporter="otlp", endpoint=args.endpoint, service_name="pyreact-e2e",
                       metric_export_interval=5)
    try:
        run_store(session_id, args.max_calls, args.model, args.timeout)
    finally:
        pyreact_otel.active().shutdown()  # flush spans + final metric export
    time.sleep(2)  # file exporter flush

    spans, metrics = read_export(offset)
    mine = [s for s in spans if s["attrs"].get("pyreact.session_id") == session_id
            or s["attrs"].get("pyreact.event.kind")]
    print(f"\n[collector] session={session_id} spans={len(mine)}")
    print_tree(mine)
    print("\n[collector] metrics (largest cumulative data point per name):")
    for k in sorted(metrics):
        print(f"  {k} = {metrics[k]:g}")

    turns = [s for s in mine if s["name"] == "pyreact.turn"]
    llm = [s for s in mine if s["attrs"].get("gen_ai.operation.name") == "chat"]
    tools = [s for s in mine if s["attrs"].get("gen_ai.operation.name") == "execute_tool"]
    ids = {s["spanId"]: s for s in mine}
    nested = all(ids.get(s.get("parentSpanId"), {}).get("name") == "pyreact.turn" for s in llm) and all(
        ids.get(s.get("parentSpanId"), {}).get("attrs", {}).get("gen_ai.operation.name") == "chat"
        for s in tools)
    ok = (len(turns) == len(QUESTIONS) and llm and tools and nested
          and metrics.get("gen_ai.client.token.usage[input]", 0) > 0)
    print(f"\n[collector] turns={len(turns)} llm={len(llm)} tools={len(tools)} nested={nested}")
    print("[collector] OK" if ok else "[collector] FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
