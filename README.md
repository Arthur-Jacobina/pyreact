# Pyreact

Pyreact is a Python multi‑agent framework with React‑style components and hooks. It coordinates agents via a router and input bus, runs on web and terminal, and integrates with DSPy for LLM‑powered agents.

## React ↔ Multi‑agent analogy (how it works)

### 1) Component ↔ Agent
- In React, a component is a function that declares UI and behavior.
- In Pyreact, an agent is a component function decorated with `component` (see `pyreact/core/core.py`). Calling it normally returns a `VNode`; the runtime renders it inside a `HookContext`.
- The `HookContext` (in `pyreact/core/hook.py`) is the agent’s execution context: it stores state, effects, children, and service references.

Key pieces:
- `component(fn)`: wraps your function so it becomes a declarative node (`VNode`) until rendered.
- `HookContext.render()`: executes the function body, reconciles children, and queues effects.
- `HookContext.run_effects()`: runs effect cleanups then new effects after commit.

### 2) Props ↔ Agent inputs
- React passes data via props; Pyreact does the same. Props are stored on `VNode.props` and copied into child `HookContext` during reconciliation (see `HookContext.render`).

### 3) State/hooks ↔ Agent internal memory
- React `useState`/`useReducer` manage local state and rerender on change.
- Pyreact mirrors this in `HookContext`:
  - `use_state(initial)`: returns `[state, set_state]`; calls `schedule_rerender` when value changes.
  - `use_reducer(reducer, initial, init_fn=None, deps=None)`: returns `[state, dispatch]`; supports lazy init and dependency‑based re‑init.
  - `use_memo(factory, deps)`, `use_callback(fn, deps)`: memoization primitives.

### 4) Effects ↔ Agent side‑effects and async work
- React `useEffect` runs after commit when dependencies change.
- Pyreact `use_effect(effect_fn, deps)`: schedules the effect; on commit, `run_effects()` executes cleanups first, then runs the effect (awaiting if it’s async), storing a new cleanup if returned.

### 5) Context/Provider ↔ Shared agent state/services
- React Context shares values across the tree with Providers.
- Pyreact `create_context` (in `pyreact/core/provider.py`) returns a context object with:
  - `Context(value=..., children=[...])`: provider component.
  - `get()`/`set(value)`: read/update. `set` notifies subscribed `HookContext`s via `schedule_rerender`.
- Agents read with `hooks.use_context(Context)`. Subscriptions are tracked so unmount cleans them up.

### 6) Reconciler/scheduler ↔ Agent commit loop
- React batches state updates and commits.
- Pyreact scheduling (in `pyreact/core/runtime.py`):
  - `schedule_rerender(ctx, reason)`: enqueues a context once and signals the loop.
  - `run_renders()`: drains the queue; for each context calls `render()` then `run_effects()`; finally marks idle with `get_render_idle().set()`.

### 7) Events ↔ Environment inputs
- React gets DOM events; agents often react to external inputs.
- Pyreact uses `InputBus` (in `pyreact/input/bus.py`):
  - `subscribe(fn)`: register a listener; returns `unsubscribe()`.
  - `emit(ev)`: deliver `{type, value, source, ts}` to all listeners.
- The `AppRunner` (in `pyreact/boot/app_runner.py`) bridges terminal and web inputs into the `InputBus` and advances the render loop.

### 8) Routing/orchestration ↔ Agent coordination
- React apps route to components; multi‑agent systems route tasks/messages.
- Pyreact Router (in `pyreact/router/router.py`) and `NavService` coordinate which agent subtree is active. Hooks like `use_route`, `use_navigate`, `use_query_params` expose navigation state and helpers.
- Agents can drive navigation imperatively (e.g., after an effect completes) by calling `navigate(...)` from `use_navigate`.

### 9) Turns ↔ Conversation steps
- Every submitted input (`AppRunner.invoke`, terminal, web) becomes a `Turn(id, index, text, source, ts)`; `invoke`/`ainvoke` return the turn id and InputBus events carry `turn_id`.
- The `TurnManager` service (`"turns"`, `pyreact/core/turns.py`) publishes `turn.start` / `turn.end` (`{"status": "done"|"interrupted"|"error", "duration"}`) on the `EventBus`. A turn ends when there are no queued renders and no pending tasks spawned during it (`runtime.spawn(..., turn=...)` tags tasks; default: the current turn; `turn=None` opts out).
- Policy: `bootstrap(App, turn_policy="queue")` (default; inputs are answered one at a time, in order) or `"interrupt"` (a new input cancels the in-flight turn). A component overrides it for its own (and its descendants') work with `use_turn(policy="interrupt"|"queue")`.
- `use_turn()` (from `pyreact.core`) returns the latest turn and rerenders on each new one, so a repeated identical message still triggers effects keyed on the turn. `Keystroke(on_submit=fn)` calls `fn(text)` or `fn(text, turn)`.
- `app.invoke(text, wait=True)` blocks until that turn ends; `app.wait_turn(turn_id)` waits for an earlier one.

### 10) Durable memory ↔ Persistence at the border
- Agent memory that survives restarts lives in a separate library, [`pyreact_durable`](pyreact_durable/README.md); `pyreact` never imports it.
- `Durable(log="chat.jsonl", children=[...])` provides a pi-style append-only JSONL session log (a tree: branches, labels, checkpoints) and journals `EventBus` events into it.
- `use_persistent_state(initial, key=...)`, `use_persistent_reducer`, `use_conversation()` are `use_state`-like hooks restored from the log; `use_durable()` exposes it for `label` / `checkout` / `fork_file`.
- `durable_lm(lm, log)` journals LM calls so re-running a turn after a crash replays the answer instead of calling the model again.
- Inspect files with `python -m pyreact_durable show chat.jsonl`; try `uv run python -m examples.durable_chat`.

### Lifecycle overview
1. A root agent (component) is wrapped in a `HookContext` by the runner.
2. State changes or context updates call `schedule_rerender`.
3. The loop `run_renders` renders contexts, reconciles children, then runs effects.
4. On unmount, `HookContext.unmount()` runs effect cleanups, removes context subscriptions, and unmounts the subtree.

## DSPy Provider Setup

### OpenAI API Key Configuration

To use this project's AI resources, you need to configure your OpenAI API key.

### 1. Get an API Key

1. Access [OpenAI Platform](https://platform.openai.com/)
2. Log in or create an account
3. Go to "API Keys" in the sidebar menu
4. Click "Create new secret key"
5. Copy the generated key

### 2. Configure the API Key

#### Option A: Environment Variable (Recommended)

```bash
export OPENAI_API_KEY="your_api_key_here"
```

#### Option B: .env File

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_api_key_here
```

**Note:** The `.env` file is in `.gitignore` for security. Never commit your API key to the repository.

### 3. Verify Configuration

To check if the configuration is working:

```bash
python -c "import os; print('API Key configured:', bool(os.getenv('OPENAI_API_KEY')))"
```

### 4. Run the Project

After configuring the API key, run the project:

Run from the repository root (examples are a package):

```bash
# Web version
uv run python -m examples.main_web

# Terminal version
uv run python -m examples.main_terminal

# Tool-calling multi-agent store (terminal)
uv run python -m examples.main_tools

# Chat with durable, branchable memory (see pyreact_durable/README.md)
uv run python -m examples.durable_chat
```

### Tests

```bash
uv run pytest
# End-to-end turn model check against a real LLM via the `claude -p` CLI
uv run python -m scripts.e2e_turns
# Durable memory: kill -9 + restart, branches, LM replay (real LLM)
uv run python scripts/e2e_durable.py
```

## Learning: architectures & optimizers

Same React analogy, one level up: the **architecture** is *how* a component
thinks (a prop you can swap), an **optimizer** is a *build step* that produces
a better program, **hot swap** ships it into the running app (HMR), and
**events** are the trace you debug and learn from (`integrations.learn`).

```python
from integrations import use_agent_call
from integrations.learn import dataset_from_events, split, optimize, exact_match

call = use_agent_call("context, question -> answer", arch="cot")      # or "predict", "react",
#                                         "rlm", "pot", "refine", "best_of_n", your own...
data = dataset_from_events(app.events, "supervisor", min_score=1.0)  # runs + :feedback
train, val = split(data, val=0.3)
result = optimize(router, trainset=train, valset=val, metric=exact_match("name"),
                  optimizer="gepa", budget=40, lm=lm, reflection_lm=lm)  # or mipro, bootstrap...
print(result.report(), result.diff())        # score before -> after, instructions/demos diff
result.save("artifacts/router.json"); result.hot_swap(app, "supervisor")
```

- **Architectures**: `use_module(sig, arch=..., **kw)` + `use_call`, or `use_agent_call(...)`;
  every arch has the same `Call` shape, `llm.*` events and cancellation (RLM runs in a worker
  thread and stops at the next REPL step). `register_architecture(name, factory)`,
  `list_architectures()`.
- **Optimizers**: `optimize(..., optimizer="gepa"|"mipro"|"bootstrap"|"bootstrap_rs"|"simba"|"copro")`,
  `budget="light"` or an int (GEPA: metric calls, MIPRO: trials, bootstrap: demos, ...;
  `python -m integrations.learn list`). `register_optimizer(name, adapter)`.
- **Metrics**: write `metric(example, pred) -> float | bool | (score, "why")` once; it is adapted
  to GEPA's feedback signature and to every other optimizer. Helpers: `exact_match(field)`,
  `contains(field)`, `llm_judge(criteria, lm=...)`.
- **Datasets**: `dataset_from_events(events, key)` takes `app.events`, `Event`s/dicts, a JSONL
  path or a session log; `feedback` events (`send_feedback`, `:feedback - name=shipping`) become
  scores and corrected labels.
- **CLI**: `python -m integrations.learn optimize mod:program --data events.jsonl --module-key
  supervisor --optimizer gepa --budget 40 --out artifacts/x.json`; `inspect x.json`; `diff a.json b.json`.
- **Examples** (`examples/learn/`, dev LM via `claude -p`): `01_architectures`,
  `02_optimizers`, `03_live_feedback_loop [--demo]`, `04_custom_architecture_and_optimizer`.

### Debug checklist

1. **Which program ran?** `:modules` (key, arch, compiled?, swaps) and `:inspect <key>` /
   `inspect_program(program)` (instructions + demos actually in use).
2. **What did the LM see?** `:explain [call_id]` / `explain(last=True)`: exact prompt messages
   and raw completions (RLM: every REPL step). `set_debug(True)` also puts them on `llm.end`.
3. **What happened, in order?** `:events llm.*`, `:turns`, `:usage` (llm events land in the
   session log: latency, tokens, `lm_calls`, `rlm.iteration` progress).
4. **Is the data right?** `dataset_from_events(...)` → check labels (`feedback.expected`
   corrections win) and `feedback_text`; empty → rate more turns with `:feedback`.
5. **Is the metric right?** run it on one example: it must return a number/bool/(score, text);
   `evaluate(program, val, metric, lm=lm)` returns per-example rows with feedback.
6. **Did it improve?** `result.report()` (baseline → score, LM calls) and `result.diff()`;
   GEPA: a reused `log_dir` is never resumed silently (`resume=True` to continue).
7. **Is the new program live?** after `result.hot_swap(app, key)` / `:optimize`, `:modules`
   shows `compiled swaps=1`, and the next `:explain` shows the new instructions.

## Troubleshooting

### Error: "DSPy context not available"

This error indicates that:
1. The API key is not configured
2. The API key is invalid
3. The DSPyProvider was not initialized correctly

**Solutions:**
1. Check if the `OPENAI_API_KEY` variable is defined
2. Confirm that the API key is valid
3. Restart the application after configuring the API key

### Error: "No language model configured"

This error indicates that no language model was configured in the DSPyProvider.

**Solution:** Check if the `DSPyProvider` is being used correctly in the `Root` component.