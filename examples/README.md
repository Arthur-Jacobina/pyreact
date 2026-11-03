# Examples

Every example is a short, runnable program that teaches one idea. Run them
from the repository root with `uv run python -m examples.<name>`.

**LM switch.** All examples get their model from `examples/_lm.py`:

| `PYREACT_EXAMPLE_LM` | Model | Notes |
|---|---|---|
| `claude` (default) | the `claude -p` CLI (haiku) | no API key needed |
| `openai` (default when `OPENAI_API_KEY` is set) | `gpt-4o-mini` | streams token by token |
| `offline` | `OfflineLM`, a deterministic fake | runs in seconds, answers are placeholders |

`uv run pytest -m examples` runs every example offline (each declares a
`SMOKE` spec) and checks it leaves the repository untouched.

## Start here: the React analogy

| Read | Example | Idea |
|---|---|---|
| 1 | `message` | An agent's output is an *event* (`Message`), not a print |
| 2 | `log` | Write your own output component: an effect that publishes an event |
| 3 | `qa_guardrail` | Composition with `children`: a checker agent decides whether its children run |
| 4 | `router_app` | Pages are routes; an LLM turns free text into a navigation |
| 5 | `tools` | Multi-agent store: a supervisor routes each turn to a tool-using specialist |
| 6 | `stream_chat` | Streaming multi-turn chat with `dspy.History` |
| 7 | `rlm_agent` | An RLM answers questions over the whole repo source (~194K chars) |
| — | `main_web --app router_app\|qa_guardrail\|tools` | Any app in the browser, one session per tab (`--check` self-tests) |

## Durable memory (`examples/durable/`, library: [`pyreact_durable`](../pyreact_durable/README.md))

| Example | Idea | LLM |
|---|---|---|
| `durable.01_counter` | `use_persistent_state` / `use_persistent_reducer`: run it twice, the count is still there | no |
| `durable.02_branches_sqlite` | SQLite backend; branches as a feature (`/label`, `/checkout`, `/fork`, `/tree`) | no |
| `durable.03_web_sessions` | Each browser tab has its own durable log (`Durable(log_factory=...)`) | yes |
| `durable.04_llm_chat` | Everything: conversation, branches, `durable_lm` replay after a crash | yes |

Inspect any log while the app runs: `uv run python -m pyreact_durable show <file>`.

## Harness: failure handling (`examples/harness/`)

| Example | Idea | LLM |
|---|---|---|
| `harness.02_resilient_agent` | Flaky tool + flaky/slow model, simulated deterministically: `retry` with backoff and `Retry-After`, `timeout`, `circuit_breaker` opening and closing, `fallback_model`, an `ErrorBoundary` fallback message, then `kill -9` mid-turn and `resume_pending()` with `durable_lm` replay | yes (6 calls) |

See [Error recovery](../README.md#error-recovery) in the main README.

## Sandboxes (`examples/sandbox/`, library: [`pyreact_sandbox`](../pyreact_sandbox/README.md))

| Example | Idea | LLM |
|---|---|---|
| `sandbox.01_coding_agent` | A ReAct agent writes and runs code in a Docker sandbox (`sandbox_tools`); `--resume` restores the sandbox after a restart (`--driver local` = no Docker, not isolated) | yes |

## Learning: architectures & optimizers (`examples/learn/`)

| Example | Idea |
|---|---|
| `learn.01_architectures` | The architecture is a prop: `predict`, `cot`, `rlm` on the same signature |
| `learn.02_optimizers` | Optimizers are build steps: bootstrap vs mipro vs gepa on one dataset and metric |
| `learn.03_live_feedback_loop` | Rate answers (`:feedback`), `:optimize`, hot-swap into the running app |
| `learn.04_custom_architecture_and_optimizer` | Register your own architecture and optimizer |
| `learn.05_durable_learning_loop` | The whole loop across restarts: serve → journal → learn with GEPA → serve better |
| `optimize_router` | The GEPA run behind `artifacts/router_gepa.json`, plus a live hot-swap demo |

## Guards & verification (`examples/harness/`)

| Example | Idea | LLM |
|---|---|---|
| `harness.01_guarded_agent` | A tool agent wrapped by guards (PII redaction, tool allowlist, per-turn token budget) and a schema/assert verifier that retries with feedback; prints every decision | yes (≤ 10 calls) |

## Paths through the examples

- **New to pyreact:** `message` → `qa_guardrail` → `tools` → `main_web --app tools`.
- **Need memory:** `durable.01_counter` → `durable.02_branches_sqlite` → `durable.04_llm_chat`.
- **Going to production:** `harness.02_resilient_agent` (recovery), then `durable.04_llm_chat`.
- **Want better prompts:** `learn.01` → `learn.02` → `learn.03` → `learn.05`.
- **Need guardrails:** `qa_guardrail` (a guard as a component) → `harness.01_guarded_agent` (guards and verifiers as middleware).
