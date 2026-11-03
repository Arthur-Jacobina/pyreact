# OTel Collector (generic OTLP check)

One `otel/opentelemetry-collector-contrib:0.145.0` container (~50 MB RAM)
receiving OTLP/HTTP on `127.0.0.1:4320`. It logs a summary of what it
receives (`debug` exporter) and writes every export request as OTLP/JSON to
`out/otlp.jsonl` (git-ignored). Every non-Langfuse vendor takes this path:
`exporter="otlp"` plus an endpoint and headers.

```bash
docker compose -f deploy/otel-collector/docker-compose.yml up -d
PYREACT_EXAMPLE_LM=claude uv run python -m scripts.e2e_otel_collector   # store app, <=10 LLM calls
docker compose -f deploy/otel-collector/docker-compose.yml logs -f      # debug exporter
docker compose -f deploy/otel-collector/docker-compose.yml down
```

The script prints the span tree of each turn (`pyreact.turn` → `chat <model>`
→ `execute_tool <name>`) and the cumulative metrics, then checks the nesting
and token usage.
