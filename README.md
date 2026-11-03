# Pyreact

Pyreact is a Python multi‑agent framework with React‑style components and hooks. It coordinates agents via a router and input bus, runs on web and terminal, and integrates with DSPy for LLM‑powered agents.

> **Examples:** see [`examples/README.md`](examples/README.md) for a guided tour (every example runs offline too).

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
  - `use_ref(initial)`: a `Ref` whose `.current` persists across renders (React `useRef`); mutating it does not rerender. Use it for "latest value" boxes read by long-lived callbacks.

### 4) Effects ↔ Agent side‑effects and async work
- React `useEffect` runs after commit when dependencies change.
- Pyreact `use_effect(effect_fn, deps)`: schedules the effect; on commit, `run_effects()` executes cleanups first, then runs the effect (awaiting if it’s async), storing a new cleanup if returned.

### 5) Context/Provider ↔ Shared agent state/services
- React Context shares values across the tree with Providers.
- Pyreact `create_context` (in `pyreact/core/provider.py`) returns a context object with:
  - `Context(value=..., children=[...])`: provider component.
  - `get()`/`set(value)`: read/update. `set` notifies subscribed `HookContext`s via `schedule_rerender`.
- Agents read with `hooks.use_context(Context)`. Subscriptions are tracked so unmount cleans them up.
- **Services are an escape hatch.** Each session also owns a services dict (`input_bus`, `event_bus`, `nav_service`, `turns`, `session`, `refs`, plus what libraries add) reachable with `hooks.get_service(key, factory)`. That is a low-level service locator for library authors implementing hooks, not part of the app-facing API: components use the hooks that wrap it (`use_input`, `use_on_turn`, `use_event` / `use_emit`, `use_navigate` / `use_location`, context, and a `ref=` prop + `use_imperative_handle`). Examples never call it (enforced by a test).
- Every component has a stable address, `hooks.path` (e.g. `App/Chat[chat]`: `Name[key]`, or `Name[#index]` when unkeyed), used for durable state and traces. **Providers are transparent**: context providers, `DSPyProvider`, `ToolProvider`, `HarnessProvider`, `Durable` and any `@component(transparent=True)` add no segment to their descendants' paths, so wrapping a tree in another provider never changes the paths below it. `hooks.full_path` keeps every node. `use_path()` (from `pyreact.core`) returns the calling component's path during render or in its effects, so output components (like `Message`) can tag what they publish without touching the render stack.

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
- The `TurnManager` service (`"turns"`, `pyreact/core/turns.py`) publishes `turn.start` / `turn.end` (`{"status": "done"|"interrupted"|"error"|"blocked", "duration"}`) on the `EventBus`. A turn ends when there are no queued renders and no pending tasks spawned during it (`runtime.spawn(..., turn=...)` tags tasks; default: the current turn; `turn=None` opts out).
- Policy: `bootstrap(App, turn_policy="queue")` (default; inputs are answered one at a time, in order) or `"interrupt"` (a new input cancels the in-flight turn). A component overrides it for its own (and its descendants') work with `use_turn(policy="interrupt"|"queue")`.
- `use_turn()` (from `pyreact.core`) returns the latest turn and rerenders on each new one, so a repeated identical message still triggers effects keyed on the turn. `Keystroke(on_submit=fn)` calls `fn(text)` or `fn(text, turn)`.
- **Handle each input once**: `use_state` drops a value equal to the current one, so don't store the submitted text to trigger work (two identical inputs would look like one). Key the work on the turn instead: `use_on_turn(lambda turn: call.run(question=turn.text))` calls the handler once per turn after commit (async handlers are part of the turn), and `Message(..., trigger="turn")` publishes once per `(turn, text)` (the default `trigger="change"` dedupes on text).
- **Conditionally mounted components**: a component mounted *during* a turn gets that in-flight turn from `use_turn()` (right for a child mounted to answer it). A component that should only react to *later* inputs (a page shown by a navigation, a panel opened by a command) uses `use_turn(since_mount=True)` / `use_on_turn(fn, since_mount=True)`, which skip the turn that mounted it.
- `app.invoke(text, wait=True)` blocks until that turn ends; `app.wait_turn(turn_id)` waits for an earlier one.
- Turn ids are `t1`, `t2`, ... per session (non-default sessions prefix them with `<session id>/`). A process restart starts again at `t1`; to keep ids unique across restarts, configure the session's `TurnManager` with `id_factory=lambda i: f"t-{run_id}-{i}"` or call `turns.advance_to(n)` (`pyreact_durable.Durable` does the latter from its log).

### 10) Durable memory ↔ Persistence at the border
- Agent memory that survives restarts lives in a separate library, [`pyreact_durable`](pyreact_durable/README.md); `pyreact` never imports it.
- `Durable(log="chat.jsonl", children=[...])` provides a pi-style append-only JSONL session log (a tree: branches, labels, checkpoints) and journals `EventBus` events into it.
- `use_persistent_state(initial, key=...)`, `use_persistent_reducer`, `use_conversation()` are `use_state`-like hooks restored from the log; `use_durable()` exposes it for `label` / `checkout` / `fork_file`.
- `durable_lm(lm, log)` journals LM calls so re-running a turn after a crash replays the answer instead of calling the model again.
- Inspect files with `python -m pyreact_durable show chat.jsonl`; examples (short to full) in [`examples/durable/`](examples/durable/): `01_counter` (hello world), `02_branches_sqlite` (SQLite, labels/checkout/fork), `03_web_sessions` (one log per browser tab), `04_llm_chat` (LLM + branches + LM replay).

### 11) Sandboxes ↔ Where agents act (at the border)
- Agents that run code, install packages or keep files between turns get a sandbox from a separate library, [`pyreact_sandbox`](pyreact_sandbox/README.md); `pyreact` never imports it. **Docker first**; E2B / Modal / Daytona / in-house are drop-in drivers (`register_driver`).
- `SandboxProvider(driver="docker", spec=SandboxSpec(image=..., memory=..., setup=[...]), children=[...])` picks the vendor and defaults; `use_sandbox()` gives a component its handle (`await sb.exec(...)`, `run_python`, `write_file` / `read_file`, `snapshot` / `restore`, `suspend` / `resume`). It is an effect with cleanup: created on mount, closed per `on_close="destroy"|"suspend"|"snapshot"` (or by the session's `close_reason`).
- Every exec is a harness boundary of kind `sandbox.exec` (guards, retries, telemetry) and publishes `sandbox.*` events.
- Stateful across restarts: a `SnapshotStore` (memory, file, or the `pyreact_durable` log) records the sandbox; a new runner reattaches it or restores its snapshot (`autosnapshot="turn"`).
- `sandbox_tools(sb)` → `run_shell` / `run_python` / `read_file` / `write_file` for `use_agent`; `SandboxInterpreter(sb)` runs RLM's REPL in the sandbox instead of Deno. Example: [`examples/sandbox/01_coding_agent.py`](examples/sandbox/01_coding_agent.py).

### 12) Suspend / resume ↔ Stateful agents
- An agent is a session's whole state (durable state + conversation, sandbox, pending turns, learned prompts). `session.suspend()` writes it as one manifest through *participants* the libraries register (`pyreact_durable`, `pyreact_sandbox`, `integrations`); `app.resume(session_id)` hands each its fragment before the first render and re-runs the turn that was cut short. `AppRunner(idle_policy="suspend")` makes idle reaping suspend instead of close. See [Stateful agents](#stateful-agents-suspend--resume).

### Lifecycle overview
1. A root agent (component) is wrapped in a `HookContext` by the runner.
2. State changes or context updates call `schedule_rerender`.
3. The loop `run_renders` renders contexts, reconciles children, then runs effects.
4. On unmount, `HookContext.unmount()` runs effect cleanups, removes context subscriptions, and unmounts the subtree.

## Harness: guards, verification, recovery, telemetry

Every call that leaves the render tree is a **boundary** and runs through one
middleware pipeline, the *harness* (`pyreact/core/harness.py`). A middleware
is an async function `(b, call_next)` — like ASGI middleware — that can
observe, time, rewrite, retry, short-circuit or refuse the call, without
touching component code. With no middleware the call is a plain call (an
empty pipeline adds ~2.5 µs per boundary).

### Boundary kinds

| kind | name | `b.inputs` | wired in |
|------|------|------------|----------|
| `turn` | `"turn"` | `{"text": ...}` | `TurnManager` (start → settle of every turn) |
| `llm` | module label (`name=` or e.g. `Predict(question -> answer)`) | the module's inputs | `invoke_module`, so `use_call` / `use_stream` / `use_agent` / `use_agent_call` / RLM runs |
| `tool` | tool name | named arguments | ReAct tools (`to_dspy_tool`, `call_tool`), RLM REPL tools (`call_tool_sync`) |
| `sandbox.exec` | command (`python3`, `ls`...), `python`, `write_file`, `read_file` | `{op, sandbox, cmd\|code\|path, timeout, env, cwd}` | [`pyreact_sandbox`](pyreact_sandbox/README.md) handles (`use_sandbox`, `sandbox_tools`, `SandboxInterpreter`) |

Any dotted kind works for your own boundaries (`sandbox.exec`, `llm.judge`):
`await through("sandbox.exec", "python", {"code": code}, run)`.

Every `Boundary` has `kind`, `name`, `inputs`, `component` (path),
`session_id`, `turn_id`, `attempt`, `id`, `parent_id` and a free-form
`attrs` dict. `parent_id` links nested boundaries — turn → llm → tool
(and llm → `llm.judge` when a middleware calls another model). It comes from
a ContextVar set while a boundary runs, which follows spawned tasks,
`asyncio.to_thread` workers and RLM REPL threads; read it with
`pyreact.core.current_boundary()` (e.g. for tracing span parents).

### Attrs reference (stable names)

| kind | before the call | after `call_next` returns (or raises) |
|------|-----------------|---------------------------------------|
| `turn` | `turn_id`, `index`, `source` | `status` (`done`/`interrupted`/`error`), `duration` (s) |
| `llm` | `call_id`, `module_key`, `arch`, `model`; with a `DSPyProvider(models=...)` registry also `model_key` (registry key in use) and `model_names` (`{key: model}`) | `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`; cumulative across retries), `cost`, `latency` (s, of the last module run), `outputs` (preview), `lm_calls` (prompts + completions) |
| `tool` | `call_id` (of the LLM call that asked for it) | `result` (preview), `latency` (s) |

Recovery middleware adds `retries`, `fallback`, `circuit` (state the call
met), `timed_out`; `fallback_model` rewrites `model_key` / `model`
(see [Error recovery](#error-recovery)).

Set by verifiers (`pyreact.harness.verify`, any kind): `verdicts` (list of
`{verifier, passed, score, threshold, feedback, attempt, action}`); before a
re-run, `rollout_id` (re-sample, don't replay a cache) and `feedback` (condition
the re-run on it). `llm` boundaries honor both (see below).

### Ordering and scope

Process-wide middleware (`install(mw)`, returns an uninstall function) is
outermost, then `HarnessProvider(middleware=[...])` from the root down, then
the innermost provider's list in order, then component hooks (`use_guard`,
`use_verify`, `use_middleware`; `install(mw, innermost=True)` is that layer).
`@middleware(kinds=["llm", "tool.*"])` restricts one to some kinds.

- `llm` calls use the providers above the calling component (captured at
  render) plus the process-wide list *at call time*;
- `tool` calls use the providers of the component that owns the LLM call;
- `turn` boundaries belong to the session, not to a subtree: only
  process-wide middleware sees them.

### What a middleware can do

- **observe / annotate**: read `b.inputs` / `b.attrs`, add your own attrs;
- **transform**: replace `b.inputs` before `call_next`, or the result after;
- **retry**: call `call_next(b)` again (bump `b.attempt`) — the module/tool
  runs again;
- **short-circuit**: return without calling `call_next` (a cache, a canned
  answer: an `llm` middleware returns a `dspy.Prediction`);
- **refuse**: `raise Blocked(reason)`. The message names the boundary and
  the guard: `llm 'qa' blocked by no-pii: <reason>`.
  - `llm`: the `Call` settles with `call.error` = the `Blocked`;
    `llm.blocked` then `llm.error` are published;
  - `tool`: the agent gets `"Error: tool 'x' blocked by ...: <reason>"` as
    the tool's result (so it can adapt); `tool.blocked` is published;
  - `turn`: the turn ends as `blocked` before it starts (`turn.blocked`).

### Writing a middleware

```python
import time
from pyreact.core.harness import Blocked, HarnessProvider, install, middleware

async def timing(b, call_next):                    # 1. timing (any kind)
    t0 = time.perf_counter()
    try:
        return await call_next(b)
    finally:
        print(f"{b.kind} {b.name} {time.perf_counter() - t0:.3f}s parent={b.parent_id}")

@middleware(kinds=["tool"], name="no-orders")      # 2. a guard
async def no_orders(b, call_next):
    if b.name == "place_order" and b.inputs.get("quantity", 0) > 10:
        raise Blocked("orders above 10 units need a human")
    return await call_next(b)

@middleware(kinds=["llm"])                         # 3. a retry
async def retry_once(b, call_next):
    try:
        return await call_next(b)
    except Exception:
        b.attempt += 1
        return await call_next(b)                  # re-runs the module

install(timing)                                     # process-wide (telemetry libs)
HarnessProvider(middleware=[no_orders, retry_once], children=[Store()])  # one subtree
```

### Events vs middleware

Events (`llm.*`, `tool.*`, `turn.*` on the `EventBus`) are **observations**:
published after the fact, consumed by logs, `:usage`, datasets; a listener
cannot change what happens. Middleware sits **in** the call path: it can
act — block, retry, rewrite, answer. Both stay in sync: the events are
published by the same code that owns the boundary, whatever the middleware
did (a blocked call still closes its `llm.start` with `llm.error`). Use
events to watch; use middleware to guard, verify, recover and trace.

### Call identity

`Call.inputs` are the inputs of the in-flight (then settled) call and
`Call.tag` the value passed with them as `run(..., tag=...)` (not sent to
the module, unless its signature has a `tag` input). Tie a result to the
turn that asked for it without extra state:

```python
check.run(comment=text, tag=turn.id)
if check.tag == turn.id and not check.loading:
    verdict = check.result.verdict
```

### Guards & verification

`pyreact.harness` (generic) and `integrations.harness` (the same, plus the
DSPy-aware `llm_judge` / `check_metric`) are ready-made middleware. **Guards**
run before a call and refuse it (`Blocked`) or rewrite its inputs;
**verifiers** run after it and check the result, then raise, retry or flag.

```python
from integrations.harness import (budget, tool_allowlist, input_policy, PII,
                                  verify, schema, assert_that, use_verify)

HarnessProvider(middleware=[                       # a subtree, like any provider
    input_policy(PII, redact=True),                # emails/phones never reach the LM
    tool_allowlist(allow=["lookup_*"]),            # the agent sees "Error: ... not allowed"
    budget(max_tokens_per_turn=20_000),            # later calls in the turn are blocked
], children=[Shop()])

@component
def Clerk():                                       # this component and its subtree
    use_verify([schema({"total": Annotated[float, Field(ge=0)]}),
                assert_that(lambda p: p.answer.endswith("?"), "End with a question.")],
               on_fail="retry_with_feedback", max_attempts=2)
    agent = use_agent(OrderQuote, tools=[...])
```

| guard | blocks / does |
|---|---|
| `guard(fn, kinds=, name=)` | your predicate: `fn(b) -> None/True` allow, `False` or a reason string block (sync or async; also a decorator) |
| `tool_allowlist(allow=, deny=)` | tools not matching the globs (`deny` wins); the reason lists the allowed tools |
| `max_calls(per_turn=, kind="llm")` | calls beyond N per turn (retries of one call count once) |
| `budget(max_tokens_per_turn=, max_cost_per_turn=)` | further calls once the turn's `usage`/`cost` attrs (added after each call) reach a limit; `.spent(turn_id)` |
| `rate_limit(calls, per_seconds, wait=, key=)` | calls over a sliding window (or waits for a slot) |
| `input_policy(regex \| {label: regex} \| fn, redact=)` | inputs matching a policy, or redacts them (`[EMAIL]`); `PII` patterns included |
| `require_confirmation(kinds, ask=, when=, timeout=)` | human in the loop: `ask(b)` (sync or async) returns `True`, or `False`/a reason |

| check (for `verify`) | passes when |
|---|---|
| `schema(Model \| {field: type} \| [fields])` | the result's fields validate (pydantic); feedback names each bad field |
| `assert_that(pred, "message")` | `pred(result)` is truthy (raising counts as failing) |
| `grounded_in(field, source_field, min_overlap=0.6)` | enough of the field's content words appear in the source input |
| `llm_judge(criteria, lm=, threshold=0.7)` | an LM grades it (`integrations.harness`; nested `llm.judge` boundary) |

**Your own check**: `check(b, result)` returns `None`/`True` (pass), `False` or
a feedback string (fail), a score, or `(score, feedback)` (pass iff score ≥
`threshold`). `verify([c1, c2])` runs several as one verifier.

**`on_fail`**: `"raise"` (`VerificationFailed`: the call fails), `"retry"`
(re-run with a new `rollout_id`), `"retry_with_feedback"`, `"flag"` (keep the
result; the failed verdict is in `b.attrs["verdicts"]`). Retries exhausted:
`on_exhausted="raise"` (default) or `"flag"`.

**Retry with feedback**, like `dspy.Refine`: the verifier sets
`b.attrs["feedback"]`, and an `llm` boundary re-runs the module with an extra
input field `feedback` ("feedback on a previous attempt: fix these problems")
added to every predictor call of that attempt, via a wrapped adapter. The
module, its signature and your inputs are untouched; only that attempt's
prompts change. For your own boundary kinds pass `inject=fn(b, feedback)`.

**Hooks vs providers.** `HarnessProvider` scopes middleware to the subtree it
wraps; `use_guard(fn | guard, kinds=)`, `use_verify(check, **options)` and
`use_middleware(mw)` scope it to the calling component and its descendants
(boundaries whose `component` is at or below it, same session), registered on
first render and removed on unmount. Hook middleware is innermost. Plain
functions are always called in their latest version (they can read props and
state); stateful objects (`budget(...)`, `verify(...)`) are kept from the first
render unless `deps=[...]` changes. A call belongs to the component that owns
the call hook, even when another component's effect fires it.

**Events** (on the session and global buses): `guard.blocked` `{guard, reason}`,
`guard.redacted` `{guard, fields, count}`, `guard.confirmed`, `verify.passed`
and `verify.failed` `{verifier, score, threshold, feedback, action}`, each
with `kind, name, boundary_id, call_id, component, attempt`.

**Learning from failures.** `verify(..., to_feedback=True)` publishes a
`feedback` event (the same one `:feedback` sends) when a call *ends* failing,
so `dataset_from_events(events, "clerk", require_feedback=True)` returns the
rejected inputs with the verifier's feedback, and
`integrations.harness.check_metric(check)` turns the same check into the
metric GEPA optimizes against. Use `on_fail="flag"` (or `on_exhausted="flag"`)
to keep those calls' outputs in the dataset.

Example: `uv run python -m examples.harness.01_guarded_agent`.

### Error recovery

Failure handling is explicit and composable, React-style: **middleware**
for boundary calls (`pyreact/core/recovery.py`, any kind; DSPy-aware
`fallback_model` in `integrations/recovery.py`, which re-exports the rest),
an **`ErrorBoundary` component** for render/effect failures, and **resume**
for turns cut short by a crash. Example:
`uv run python -m examples.harness.02_resilient_agent`.

| middleware | does | events |
|---|---|---|
| `retry(max_attempts=3, backoff=exponential(base, max, jitter), retry_on=types\|predicate, kinds=("llm", "tool"))` | re-runs the call (`b.attempt` bumped); never retries `Blocked`, cancellation or `exc.retryable = False`; honors a `Retry-After` hint (`exc.retry_after`, `Retry-After`/`retry-after-ms` headers; capped by `max_retry_after`); an interrupted turn cancels it in its backoff sleep | `recovery.retry`, `recovery.exhausted` |
| `timeout(seconds \| fn(b), kinds=...)` | bounds each run; cancels the inner call (`CancelledError` inside it) and raises `BoundaryTimeout` (a retryable `TimeoutError`). Worker threads (sync tools/LMs) cannot be interrupted: the boundary returns on time, the thread finishes in the background | `recovery.timeout` |
| `circuit_breaker(failure_threshold, reset_timeout, key=lambda b: b.attrs.get("model") or b.name, fallback=None)` | closed → (N consecutive failures) open → fail fast with `CircuitOpen` (`retryable=False`, `retry_after`) or `fallback(b, exc)` → after `reset_timeout` one half-open trial → closed / open. `.state(key)`, `.reset()` | `recovery.circuit` |
| `fallback_model(chain=["default", "fast"], on=...)` (`llm`) | re-runs the call with the next `DSPyProvider(models=...)` key; starts from the call's own model; an open circuit falls back at once | `recovery.fallback` |
| `fallback(fn, on=...)` | returns `fn(b, exc)` instead of failing (a `dspy.Prediction` for `llm`) | `recovery.fallback` |

**Ordering.** Outermost first:

```python
HarnessProvider(middleware=[
    otel,                                         # sees the final outcome, one span per boundary
    guards, verify,                               # a refusal is final: never retried
    fallback_model(chain=["default", "fast"]),    # degrade after retries gave up / circuit open
    retry(max_attempts=3, backoff=exponential(0.5, 8.0)),
    circuit_breaker(failure_threshold=5, reset_timeout=30),  # counts every attempt
    timeout(30),                                  # bounds each attempt
], children=[Agent()])
```

- telemetry outermost sees one boundary with its `retries` / `fallback` /
  `circuit` attrs (put it innermost instead to get one span per attempt);
- guards (`Blocked`) and verification sit outside recovery: a blocked
  call is not retried, and verification judges the final answer
  (retry-with-feedback for *verification* failures belongs to the
  verification middleware);
- `circuit_breaker` inside `retry` so each attempt counts and an open
  circuit stops the retries; inside `fallback_model` so its default key
  (`b.attrs["model"]`) follows the model switch;
- `timeout` innermost: per attempt, not for the whole retry loop.

**How `fallback_model` switches models.** `use_call` passes the provider's
registry to `invoke_module`; the `llm` boundary carries
`b.attrs["model_key"]` and `b.attrs["model_names"]`. A middleware that sets
`b.attrs["model_key"] = "fast"` before `call_next` makes that run use
`models["fast"]`; `b.attrs["model"]`, `llm.end` (`model`, `model_key`) and
the call record name the model that answered. A call with an explicit
`lm=` and no provider registry is passed through untouched.

**`ErrorBoundary`** (`pyreact.core.ErrorBoundary`) — React error
boundaries:

```python
ErrorBoundary(
    fallback=lambda error: Message(text=f"(chart unavailable: {error})"),
    on_error=lambda error, info: log.warning("%s: %s", info["component"], error),
    reset_on="turn",              # or "props" (+ reset_keys=[...]) or fn(error, info) -> bool
    children=[Chart(key="chart")],
)
```

- catches errors of its *descendants* while rendering, in effects (sync, or
  an async effect's task) and in tasks they `spawn` — not its own errors,
  not cleanup errors, not what a hook turns into data (`call.error`);
- on an error: unmounts the children, renders `fallback` (a VNode/list, or a
  function/component given the parameters it declares among `error`,
  `info`, `reset`), calls `on_error(error, info)` (`info`: `component`,
  `phase`, `turn_id`, `boundary`), publishes `error.caught` (with the
  traceback) instead of `error`; the error is not printed and the turn
  does not end as `"error"`;
- resets (re-mounts fresh children) on the next turn (`"turn"`, default),
  when `reset_keys` change (`"props"`), when `reset_on(error, info)` is
  true, or when the fallback calls `reset()`;
- an error of the fallback goes to the next boundary up; errors no boundary
  catches keep the default behavior (last good subtree kept, `error` event,
  turn status `"error"`);
- transparent like providers (`@component(transparent=True)`): wrapping a
  subtree in a boundary does not change its components' paths, so durable
  state keyed by path survives adding one.

**Resume.** A turn the process never finished (crash, `kill -9`, deploy)
can be re-run with the same id and text:

```python
app = bootstrap(App)                        # App mounts Durable(log=...) + durable_lm
app.resume_pending(wait=True)               # {session_id: [turn ids]}
session.resume_turn("t7")                   # one turn (also an interrupted one)
```

The core only defines the hook: a `TurnStore` (`on_turn_start`,
`on_turn_end`, `pending()`, optional `get(turn_id)`) set with
`TurnManager.configure(store=...)`. `pyreact_durable` implements it
(`DurableTurnStore`, attached by `Durable(resume_turns=True)`, the
default) from the `turn.start` / `turn.end` events it already journals: a
start without an end is pending. A resumed turn has `turn.resumed` and
`turn.start` carries `"resumed": true`. Re-running a turn is at-least-once
for its side effects: commit results atomically at the end of the turn,
and put the LM behind `durable_lm` — the calls that answered before the
crash are replayed from the log (no model call, no cost), only the rest
run. `tests/test_resume_turns.py` kills a process with SIGKILL mid-turn and
checks the resumed turn gives the same answer with no repeated model call.

**Telemetry.** Every recovery action is a `recovery.*` event (session bus
and global bus, with `turn_id` and the component path): `:events`,
`Durable` journals (add `"recovery.*"` to `journal=`), sinks and trace
exporters see them. Middleware-level state is on the boundary
(`b.attempt`, `retries`, `fallback`, `circuit`, `timed_out`).

## Stateful agents: suspend / resume

An **agent is a session's whole state**: durable component state and its
conversation, its sandbox (and the files in it), the turns it has queued or
was in the middle of, the prompts it learned. Making a long-running agent
sleep and wake up — on idle, on deploy, on another machine — is one explicit,
observable operation:

```python
from pyreact.boot import AppRunner, DirectoryManifestStore
from pyreact_durable import DurableParticipant
from pyreact_sandbox import SandboxParticipant
from integrations.stateful import CompiledModulesParticipant

app = AppRunner(App, default_session=False,
                participants=[DurableParticipant, SandboxParticipant,
                              lambda s: CompiledModulesParticipant(s, dir="runs/learned")],
                manifest_store=DirectoryManifestStore("runs/manifests"))
s = app.open_session("agent-1")
s.invoke("write fib.py and run it", wait=True)
manifest = s.suspend(on_active="wait")       # -> runs/manifests/agent-1.json, tree unmounted

# ... later, in another process:
s = app.resume("agent-1")                    # same id; state, sandbox, prompts, cut turns back
```

The core keeps the "libraries at the border" rule: `pyreact.boot` only
defines lifecycle hooks, the participant protocol, manifests and their
stores; `pyreact_durable`, `pyreact_sandbox` and `integrations` plug their
state in; nothing in `pyreact/` imports them. Example (two processes, real
LLM + Docker, or offline with `--driver local`):
[`examples/stateful/01_suspend_resume.py`](examples/stateful/01_suspend_resume.py).

### Session lifecycle hooks

`app.on_session(open=fn, close=fn, replay=False)` calls `open(session)` on
the runner loop when a session opens or is resumed — **before its first
render**, so it can subscribe to `session.event_bus`, seed services or add
participants — and `close(session)` after its tree unmounted
(`session.close_reason` is `"closed"`, `"idle"`, `"shutdown"` or
`"suspend"`). It returns a function removing the hooks. `pyreact_otel`'s
`attach_app(app)` uses it to bridge sessions opened later.

Events on the session bus: `session.opened {resumed}`,
`session.suspending {reason, on_active}`, `session.suspended {ref, turn_seq,
participants, carried_turns}`, `session.suspend.failed {error}`,
`session.resumed {turn_seq, participants, degraded, turns}`,
`session.resume.degraded {participant, reason, ...}`, `session.closed {reason}`.

### Suspend and resume

`session.suspend(on_active="wait", timeout=None, store=None, reason="manual")`
(`asuspend` from a loop, `app.suspend_session(id)`):

1. `session.suspending`; new input is refused, queued turns are held;
2. the active turn ends — `on_active="wait"` lets it finish (after `timeout`
   seconds it is interrupted), `"interrupt"` cancels all its tasks now. A turn
   cut short is **carried**: it re-runs (same id, `resumed: true`) on resume;
3. each participant's `suspend(session)` returns a JSON fragment;
4. the manifest is saved to the store; `session.suspended`;
5. the tree unmounts (`close_reason == "suspend"`): providers keep their
   files (`Durable(on_close=...)` retention only applies to `idle`/`closed`);
   then participants' optional `suspended(session)` (e.g. wait for sandbox
   closes). Returns the manifest (plus `ref`).

`app.resume(session_id, manifest=None, store=None, resume_turns=True,
consume=True, wait=False)` (`aresume`): `manifest` is a dict, a path, or
`None` to load it from the store. The session is created with the same id,
`open` hooks register participants, `session.opened {resumed: true}`, each
participant's `resume(session, fragment)` runs **before the first render**,
the tree mounts, optional `resumed(session)` hooks run, carried turns are
re-queued and so are turns a `TurnStore` saw cut short by a crash
(`resume_pending`), then `session.resumed`. With `consume` the manifest is
deleted from the store it came from (an old manifest must not be resumed
twice). Turn numbering continues (`agent-1/t3` after `t2`).

Policies:

- `AppRunner(idle_policy="suspend")`: idle reaping (`close_idle_sessions(ttl)`,
  the web server's reaper) suspends instead of closing (`reason: "idle"`);
  a session that fails to suspend stays open.
- `AppRunner(auto_resume=True)` (default with `idle_policy="suspend"`):
  `open_session(id)` resumes a session that has a manifest — a browser tab
  reconnecting after its session was reaped gets its agent back.
- `on_active`: `"wait"` (default) / `"interrupt"`, with `timeout=` as the
  wait's upper bound.
- `SandboxParticipant(mode="snapshot"|"suspend"|"keep")`: snapshot + destroy
  (survives a machine change), pause and reattach (fast), or leave running.

### Manifest format

```json
{
  "format": "pyreact.manifest/1",
  "session_id": "coder-1",
  "created_at": 1793720847.46, "suspended_at": 1793720877.92,
  "reason": "manual", "turn_seq": 2, "turn_policy": "queue",
  "app": "__main__:make_app.<locals>.App",
  "participants": {
    "turns": {"next_index": 3, "carry": [{"id": "coder-1/t2", "index": 2, "text": "Extend the fib.py ...", "source": "term", "ts": 1793720872.88}]},
    "durable": {"logs": [{"path": "/abs/runs/stateful/logs/coder-1.jsonl", "log_id": "8089a346-…",
                          "leaf": "d88b31a2", "entries": 25, "version": 1, "pending_turns": []}]},
    "sandbox": {"mode": "snapshot", "sandboxes": [{"key": "default", "store_key": "coder-1/default",
                "driver": "docker", "sandbox_id": null, "spec": {"image": "python:3.12-slim", "...": "..."},
                "snapshot": {"driver": "docker", "id": "pyreact-sandbox:pyrsbx-e4835141d5-suspend",
                             "sandbox_id": "pyrsbx-e4835141d5", "label": "suspend", "...": "..."}}]},
    "dspy.compiled": {"modules": [{"key": ["ReAct", "Coder", "coder"], "key_str": "ReAct:Coder:coder",
                      "provider": "App/DSPyProvider[dspy]", "arch": "react",
                      "path": "runs/stateful/learned/coder.json"}]}
  },
  "versions": {"manifest": 1, "pyreact": "0.1.0", "python": "3.13.0",
               "participants": {"turns": "1", "durable": "1", "sandbox": "1", "dspy.compiled": "1"}}
}
```

(From the real run of the example: suspended while the follow-up turn
`coder-1/t2` was in flight, so it is carried.)

Stores: `MemoryManifestStore` (default; same process), `DirectoryManifestStore(dir)`
(one `<id>.json` per session, atomic writes), or any `ManifestStore`
(`save(manifest) -> ref`, `load(id)`, `delete(id)`, `list()`).
`load_manifest(path)` reads and validates one; a newer `format` is refused
(`ManifestError`).

### Participants

| Participant | Name | Fragment | On resume (before the first render) |
|---|---|---|---|
| built in | `turns` | next turn index, carried turns | numbering continues; carried turns re-run after the first render |
| `pyreact_durable.DurableParticipant` | `durable` | per mounted `Durable` log: path, header id, leaf, pending turns | checks the log is there and the same; `Durable` checks it out at the leaf → persistent state + conversation as suspended; `durable_lm` replays a re-run turn's answered calls |
| `pyreact_sandbox.SandboxParticipant` | `sandbox` | per `use_sandbox` handle: driver, spec, snapshot ref / live id | the record wins over the provider's `SnapshotStore`: restore from the snapshot, or reattach + unpause |
| `integrations.stateful.CompiledModulesParticipant` | `dspy.compiled` | per program in `env.compiled`: its `integrations.learn` result path (or one saved under `dir`, or the state inline) | the state is loaded into the module `use_module` / `use_agent` builds for that key; the optimized prompt survives restarts |

Writing one — any object with a `name` and two coroutines:

```python
class Scratchpad:
    name = "scratchpad"            # its key in manifest["participants"]
    version = "1"                  # optional, recorded in manifest["versions"]

    def __init__(self, session):   # factories get the session
        self.session = session

    async def suspend(self, session) -> dict:      # JSON-able, or the suspend fails
        return {"notes": session.services.get("notes", [])}

    async def resume(self, session, fragment):     # before the first render
        session.services["notes"] = fragment["notes"]   # components read it on mount

    # optional: async def suspended(self, session)  -- after the tree unmounted
    # optional: async def resumed(self, session)    -- after the first render

app = AppRunner(App, participants=[Scratchpad])   # or app.add_participant(Scratchpad)
```

The pattern the bundled ones follow: a provider component registers its live
object in a session service on mount (so `suspend` can find it); `resume`
puts what it restored in another service that the provider reads on its
first render.

### Failure semantics

- **Suspend is all or nothing.** If a participant raises, returns something
  that is not JSON, or the store cannot save, nothing is unmounted: held
  turns start again, a turn cut short is re-queued, `session.suspend.failed`
  is published and `SuspendError` raised. (Side effects a participant already
  made stay — e.g. a sandbox snapshot that is simply never referenced.)
- **Resume degrades, it never fails because of one participant.** A fragment
  no registered participant claims, a participant whose `resume` raises, or
  one that reports a partial restore with `session.degrade(name, reason,
  **info)` is recorded in `session.degraded` and published as
  `session.resume.degraded`; the session goes on with what was restored:
  - sandbox snapshot gone (or the paused container, or the driver changed) →
    a **fresh sandbox** boots (`reason: "snapshot missing"`, `"sandbox gone"`,
    `"restore failed"`, `"driver changed"`);
  - durable log missing / replaced / leaf gone → the provider opens the file
    as usual (a fresh log if it is gone);
  - a learned program's result file missing → that module starts from its
    signature (`reason: "result missing"`).
- Resume fails outright only for the envelope: wrong `session_id`,
  unsupported `format` (`ManifestError`), or the session is already open.
- Re-running a carried turn is at-least-once for its side effects, like
  `resume_pending`: answered LLM calls are replayed by `durable_lm`, tools
  run again (make them idempotent).
- Do not `await session.asuspend(on_active="wait")` from inside one of that
  session's turns: it would wait for itself.

## Running the examples

### 1. Pick the LM

Every example gets its LM from `examples/_lm.py`, switched by `PYREACT_EXAMPLE_LM`:

- `claude` (default without an OpenAI key): the `claude -p` CLI as a dev LM, no API key.
- `openai` (default when `OPENAI_API_KEY` is set): `gpt-4o-mini`.
- `offline`: a deterministic fake LM; examples run in seconds with meaningless answers
  (what `tests/test_examples_smoke.py` uses).

```bash
export PYREACT_EXAMPLE_LM=openai OPENAI_API_KEY="your_api_key_here"   # optional
```

In your own app, pass any `dspy.LM` to `DSPyProvider(lm=...)` or
`DSPyProvider(models={"default": ..., "fast": ..., "reasoning": ...})`.

### 2. Run the Project

Run from the repository root (examples are a package). In the terminal every
line is a turn; `:help` lists the dev commands (`:events`, `:usage`, `:turns`,
`:route`, `:nav`, `:q`). Only `:` starts a terminal command: `/word` lines (and
unknown `:word` commands) are app input, so apps can own their slash commands.

```bash
# Routing: pages are routes; an LLM turns free text into a navigation
uv run python -m examples.router_app

# Guardrail: a checker agent decides whether the QA agent may run
uv run python -m examples.qa_guardrail

# Tool-calling multi-agent store: supervisor -> ReAct specialists
uv run python -m examples.tools

# Any of them in the browser (one session per tab), or self-test it
uv run python -m examples.main_web --app router_app   # | qa_guardrail | tools
uv run python -m examples.main_web --check

# Messages and custom events, no LM
uv run python -m examples.message
uv run python -m examples.log

# Durable memory, smallest first (see pyreact_durable/README.md)
uv run python -m examples.durable.01_counter
uv run python -m examples.durable.04_llm_chat

# Sandboxes: a coding agent in Docker; --resume restores it after a restart
uv run python -m examples.sandbox.01_coding_agent "What is the 30th Fibonacci number?"
uv run python -m examples.sandbox.01_coding_agent --resume

# Stateful agents: suspend mid-turn, resume in a new process (--driver local: no Docker)
uv run python -m examples.stateful.01_suspend_resume --demo
```

In your own code, `run_web(App)` serves an app (one session per browser tab);
`make_web_app(App, **opts)` returns the same `(asgi_app, runner)` without
serving it, so a web app can be mounted elsewhere or tested in-process with
starlette's `TestClient` (see `examples/main_web.py --check`).

### Tests

```bash
uv run pytest
# End-to-end turn model check against a real LLM via the `claude -p` CLI
uv run python -m scripts.e2e_turns
# Durable memory: kill -9 + restart, branches, LM replay (real LLM)
uv run python scripts/e2e_durable.py
# Sandbox integration tests need Docker (skipped without it)
uv run pytest -m docker
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
  `02_optimizers`, `03_live_feedback_loop [--demo]`, `04_custom_architecture_and_optimizer`,
  `05_durable_learning_loop` (below).

### The whole loop across restarts (durable + learn)

The two libraries only meet at the border: `pyreact_durable` journals what
happened, `integrations.learn` reads the journal later as a dataset. The app
code knows about neither.

```python
# serve: wrap the tree once; turns, LLM calls and feedback land in the journal
Durable(log="runs/store.jsonl", journal=("turn.*", "llm.*", "feedback"), children=[App()])
send_feedback(turn_id, 0.0, "should be shipping", expected={"name": "shipping"})

# learn (offline, another process): rated calls only; corrections become labels
data = dataset_from_events("runs/store.jsonl", "supervisor", require_feedback=True)
optimize(router, trainset=..., valset=..., metric=route_metric, optimizer="gepa").save(path)

# serve again: load the artifact and hot-swap it at boot
load(path).hot_swap(app, "supervisor")
```

`uv run python -m examples.learn.05_durable_learning_loop all` runs the three
phases with the dev LM. One run (haiku): trained on 12 rated conversations,
held-out routing went 0.62 → 1.00 (8 messages, so a small sample) for 42 GEPA
LM calls; `python -m pyreact_durable events runs/store_router.jsonl --kind
feedback` shows what it learned from.

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

## OpenTelemetry: Langfuse and any OTLP backend

`pyreact_otel` (optional extra: `pip install 'pyreact[otel]'` / `uv sync --extra otel`)
exports every harness boundary as an OpenTelemetry span. A turn is the trace
root (`pyreact.turn`), LLM calls become `chat <model>` spans with GenAI
semantic conventions (model, tokens, cost), and tools become
`execute_tool <name>` spans. It also exports token/latency/cost/error
metrics and bridges `message`/`error`/`feedback`/`session.*` events
(`otel.attach_app(app)` follows sessions opened or resumed later, through
`app.on_session`). Like
`pyreact_durable`, it lives at the border: `pyreact` never imports it.

```python
import pyreact_otel
otel = pyreact_otel.setup(exporter="langfuse")   # LANGFUSE_HOST / _PUBLIC_KEY / _SECRET_KEY
# or exporter="otlp" (OTEL_EXPORTER_OTLP_* env: Jaeger, Phoenix, Honeycomb, Datadog, ...)
```

Prompts and completions are recorded only with `capture_content=True`. Add a
vendor with `register_exporter(name, factory)`. See
[`pyreact_otel/README.md`](pyreact_otel/README.md) for the span/attribute
reference, and [`deploy/langfuse/`](deploy/langfuse/README.md) for a local
Langfuse plus `scripts/e2e_otel_langfuse.py`.

## Troubleshooting

### Error: "DSPy context not available"

The DSPyProvider was not mounted above the component, or its LM could not be
built. For the examples, check `PYREACT_EXAMPLE_LM`: `claude` needs the `claude`
CLI on your PATH, `openai` needs `OPENAI_API_KEY`; `offline` needs nothing.

### Error: "No language model configured"

This error indicates that no language model was configured in the DSPyProvider.

**Solution:** Check if the `DSPyProvider` is being used correctly in the `Root` component.