# Local Langfuse (for pyreact_otel)

A self-hosted Langfuse **v4.73.0** for developing and checking
`pyreact_otel` end to end. It is the [official compose file](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml)
with every image pinned, only the web UI published (on `127.0.0.1:3100`),
and headless init so a project and API keys exist on first boot.

It is heavy: Postgres, ClickHouse, Redis, MinIO, langfuse-web and
langfuse-worker. Budget **~2.5 GB RAM** at idle (ClickHouse ~0.7 GB,
web ~0.7 GB, worker ~0.5 GB), more under load, ~2 GB of images, and 4+ GB of
Docker memory overall (check with `docker info | grep "Total Memory"`).

| Service          | Image                                              |
|------------------|----------------------------------------------------|
| langfuse-web     | `docker.langfuse.com/langfuse/langfuse:4.73.0`     |
| langfuse-worker  | `docker.langfuse.com/langfuse/langfuse-worker:4.73.0` |
| clickhouse       | `clickhouse/clickhouse-server:26.3.62.3`           |
| postgres         | `postgres:17.6`                                    |
| redis            | `redis:7.4.6`                                      |
| minio            | `cgr.dev/chainguard/minio@sha256:3c60a2a8…` (Chainguard only tags `latest`) |

## Up

```bash
docker compose -f deploy/langfuse/docker-compose.yml up -d
# first boot runs migrations: ~1-2 min until
curl -s localhost:3100/api/public/health        # {"status":"OK",...}
```

UI: <http://localhost:3100>, login `dev@pyreact.local` / `pyreact-dev-password`,
project `pyreact-dev`. API keys (created by `LANGFUSE_INIT_*`):

```bash
export LANGFUSE_HOST=http://localhost:3100
export LANGFUSE_PUBLIC_KEY=pk-lf-pyreact-local
export LANGFUSE_SECRET_KEY=sk-lf-pyreact-local
```

These are **local-dev defaults**. To change them, copy `.env.example` to
`.env` (git-ignored). Never expose this stack
on a network.

## Check pyreact_otel against it

```bash
PYREACT_EXAMPLE_LM=claude uv run python -m scripts.e2e_otel_langfuse            # store app, <=10 LLM calls
uv run python -m scripts.e2e_otel_langfuse --skip-app --synthetic                # no LLM at all
```

The script runs the tool-calling store app (`examples.tools`) with
`pyreact_otel.setup(exporter="langfuse")`, then reads the observations back
through the public API (`GET /api/public/v2/observations?sessionId=...`;
v4 has no v1 traces endpoint) and prints trace names, session id,
observation types and token usage.

Note: the first OTLP request after a (re)start of langfuse-web can take
several seconds; the Langfuse exporter uses a 30 s timeout for that reason.

## Down

```bash
docker compose -f deploy/langfuse/docker-compose.yml down      # stop, keep data
docker compose -f deploy/langfuse/docker-compose.yml down -v   # stop and delete all data (volumes)
```
