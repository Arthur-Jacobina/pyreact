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
  - `use_state(initial)`: returns `[state, set_state]`; calls `schedule_rerender` when value changes. As in React, a function is special: `use_state(fn)` is a lazy initializer and `set_state(fn)` an updater (`fn(prev) -> next`). In Python only *functions* count (plain functions, lambdas, bound methods, builtins, `functools.partial`; `pyreact.core.is_updater`): callable objects such as a DSPy module are stored as values, so `set_program(compiled)` just works.
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
- **Services are an escape hatch.** Each session also owns a services dict (`input_bus`, `event_bus`, `nav_service`, `turns`, `session`, `refs`, plus what libraries add) reachable with `hooks.get_service(key, factory)`. That is a low-level service locator for library authors implementing hooks, not part of the app-facing API: components use the hooks that wrap it (`use_input`, `use_on_turn`, `use_event` / `use_emit`, `use_navigate` / `use_location`, context, and a `ref=` prop + `use_imperative_handle`; see [Hooks reference](#hooks-reference)). Examples never call it (enforced by a test).
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
- Every submitted input (`root.send` / `AppRunner.invoke`, terminal, web) becomes a `Turn(id, index, text, source, ts)`; `send`/`asend` (`invoke`/`ainvoke`) return the turn id and InputBus events carry `turn_id`.
- The `TurnManager` service (`"turns"`, `pyreact/core/turns.py`) publishes `turn.start` / `turn.end` (`{"status": "done"|"interrupted"|"error"|"blocked", "duration"}`) on the `EventBus`. A turn ends when there are no queued renders and no pending tasks spawned during it (`runtime.spawn(..., turn=...)` tags tasks; default: the current turn; `turn=None` opts out).
- Policy: `create_root(App, turn_policy="queue")` (default; inputs are answered one at a time, in order) or `"interrupt"` (a new input cancels the in-flight turn). A component overrides it for its own (and its descendants') work with `use_turn(policy="interrupt"|"queue")`.
- `use_turn()` (from `pyreact.core`) returns the latest turn and rerenders on each new one, so a repeated identical message still triggers effects keyed on the turn. `Keystroke(on_submit=fn)` calls `fn(text)` or `fn(text, turn)`.
- **Handle each input once**: `use_state` drops a value equal to the current one, so don't store the submitted text to trigger work (two identical inputs would look like one). Key the work on the turn instead: `use_on_turn(lambda turn: call.run(question=turn.text))` calls the handler once per turn after commit (async handlers are part of the turn), and `Message(..., trigger="turn")` publishes once per `(turn, text)` (the default `trigger="change"` dedupes on text).
- **Conditionally mounted components**: a component mounted *during* a turn gets that in-flight turn from `use_turn()` (right for a child mounted to answer it). A component that should only react to *later* inputs (a page shown by a navigation, a panel opened by a command) uses `use_turn(since_mount=True)` / `use_on_turn(fn, since_mount=True)`, which skip the turn that mounted it.
- `root.send(text, wait=True)` blocks until that turn ends; `root.wait_turn(turn_id)` waits for an earlier one.
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

### 12) Suspend / resume ↔ `createRoot` / `hydrateRoot` + `<Activity>`
- An agent is its tree's state. Components declare how theirs survives with `use_dehydrate(key, dehydrate=..., hydrate=...)`; `create_root(App, store=...)` / `root.suspend()` hide the tree (cleanup reason `"suspend"`), dehydrate it into a JSON snapshot and unmount; `hydrate_root(App, session_id=...)` gives every hook its slice during the first render and re-runs the turn that was cut short. `Activity(mode="hidden")` keeps a subtree's state while it gets no turns. See [Stateful agents](#stateful-agents-roots-use_dehydrate-activity).

### Lifecycle overview
1. A root agent (component) is wrapped in a `HookContext` by the runner.
2. State changes or context updates call `schedule_rerender`.
3. The loop `run_renders` renders contexts, reconciles children, then runs effects.
4. On unmount, `HookContext.unmount()` runs effect cleanups, removes context subscriptions, and unmounts the subtree.

## Hooks reference

If you know React, you know pyreact: every public hook, with its React counterpart where there is one. Core hooks are called as `hooks.use_*` (methods of the component's `HookContext`) or imported from `pyreact.core`; library hooks come from their package. Call them at the top level of a component, in the same order every render.

| Area | Hook | Purpose | React |
|---|---|---|---|
| state | `hooks.use_state(initial)` | `(value, set_value)`; rerenders when the value changes; a *function* is a lazy initializer / an updater, a callable object (e.g. a DSPy module) is a value | `useState` |
| state | `hooks.use_reducer(reducer, initial, init_fn=None, deps=None)` | `(state, dispatch)`; re-init when `deps` change | `useReducer` |
| state | `hooks.use_memo(factory, deps)` / `hooks.use_callback(fn, deps)` | memoized value / function | `useMemo` / `useCallback` |
| effects | `hooks.use_effect(fn, deps)` | side effect after commit (sync or async), returns an optional cleanup | `useEffect` |
| context | `hooks.use_context(Ctx)` | nearest provider's value (`create_context`); rerenders on change | `useContext` |
| context | `use_path()` / `hooks.path` | the component's stable tree path (`App/Chat[chat]`) | `useId`-ish |
| refs | `hooks.use_ref(initial)` | mutable `Ref` (`.current`) that persists, no rerender | `useRef` |
| refs | `use_imperative_handle(ref, factory, deps=None, *, name=None)` | expose a handle through a `ref=` prop (`Ref`, callback, or a name in the session's `"refs"` registry) | `useImperativeHandle` |
| effects | `use_cleanup_reason()` | inside a cleanup: `"unmount"` / `"deps"` / `"hidden"` / `"suspend"` (also passed to a one-argument cleanup) | - |
| state | `use_dehydrate(key, *, dehydrate, hydrate=None, initial=None)` | a slice of the snapshot owned by the component: hydrated value on the first render, `dehydrate()` on suspend | `hydrateRoot` / TanStack `dehydrate` |
| turns | `use_turn(policy=None, *, since_mount=False)` | latest `Turn`; rerenders on each input; per-subtree turn policy | - |
| turns | `use_on_turn(fn, **use_turn_kwargs)` | run `fn(turn)` once per input (async = part of the turn) | - |
| input | `use_input(types=None)` | latest raw input event (`text`/`submit`/...) with a `seq`; rerenders on each | `onChange` |
| events | `use_event(kinds, fn)` | subscribe to the session `EventBus` while mounted (latest `fn` via a ref) | `addEventListener` in `useEffect` |
| events | `use_emit()` | stable `emit(kind, payload)`: publish with the component path as source and the current turn id | `dispatchEvent` |
| routing | `use_route()` / `use_navigate()` | `(route, navigate)` / `navigate(path, params=..., query=...)` | `useNavigate` |
| routing | `use_location()` / `use_route_params()` / `use_query_params()` / `use_fragment()` | current location and its parts | `useLocation` / `useParams` / `useSearchParams` |
| routing | `use_routes_catalog()` | the routes declared under the router (for LLM routing) | - |
| tools | `use_tool(fn, name=..., deps=...)` / `use_tools()` | a tool local to the component / the `ToolProvider` registry | - |
| harness | `use_harness()` | the harness (middleware) in effect for this component | - |
| harness | `use_middleware(mw)` / `use_guard(fn)` / `use_verify(check)` | middleware / guard / verifier for this subtree | - |
| dspy | `use_module(sig, arch=..., name=...)` / `use_predict` / `use_chain_of_thought` | memoized DSPy program (any architecture) | `useMemo` |
| dspy | `use_call(module)` / `use_agent_call(sig, arch=...)` | call a program as an effect: `run` / `result` / `ver` / `loading` / `error` | `useQuery`-ish |
| dspy | `use_stream(module)` | `use_call` that rerenders with partial text | - |
| dspy | `use_agent(sig, tools=...)` / `use_rlm(...)` | ReAct agent over tools / recursive LM agent | - |
| dspy | `use_dspy_env()`, `use_dspy_module`, `use_dspy_call`, `use_dspy_agent` | `DSPyProvider` env and the older tuple-returning API | - |
| durable | `use_persistent_state(initial, key=)` / `use_persistent_reducer(...)` | `use_state` / `use_reducer` restored from the `Durable` log | - |
| durable | `use_conversation(key="history")` / `use_durable()` / `use_durable_lm(lm)` | persisted transcript / the log (label, checkout, fork) / LM replay | - |
| sandbox | `use_sandbox(spec=None, *, key=, scope=, ref=None, ...)` | sandbox handle (effect with cleanup); `ref=` exposes it | - |
| sandbox | `use_sandbox_env()` / `use_durable_snapshot_store()` | nearest `SandboxProvider` env / sandbox records in the `Durable` log | - |
| learn | `use_traced(module, key)` / `use_trace_recorder(path=None)` | calls recorded as `module.call` events / the dataset recorder | - |
| learn | `use_module_version(module_key)` | how many times a program was hot-swapped (rerenders on swap) | HMR |

`hooks.get_service(key, factory)` is deliberately not in this table: it is the low-level escape hatch the hooks above are built on (see [Context/Provider](#5-contextprovider--shared-agent-stateservices)).

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
root = create_root(App, session_id="chat")  # App mounts Durable(log=...) + durable_lm
root.session.resume_pending(wait=True)      # [turn ids]; root.session: the low-level side
root.session.resume_turn("chat/t7")         # one turn (also an interrupted one)
```

(`examples/harness/02_resilient_agent.py` does this after a `kill -9`. A
suspended root needs none of it: `hydrate_root` re-runs the turn it carried.)

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

## Stateful agents: roots, `use_dehydrate`, `Activity`

An **agent is its tree's state**: what its components hold (conversation,
sandbox reference, learned program), plus the turns it has queued or was in
the middle of. Following [ADR 0001](docs/adr/0001-look-like-react.md), the
tree is the source of truth: each component declares how its own state
survives with a hook, and outside code talks to the tree only through its
**root** (cf. React `createRoot` / `hydrateRoot`):

```python
from pyreact.boot import create_root, hydrate_root, DirectorySnapshots
from pyreact.core import component, hooks, use_dehydrate, use_on_turn

@component
def Coder():
    start = use_dehydrate("history", dehydrate=lambda: history.current, initial=[])
    history = hooks.use_ref(list(start))
    use_on_turn(lambda turn: history.current.append(turn.text))
    return []

store = DirectorySnapshots("state/")
root = create_root(App, session_id="coder-1", store=store)
root.send("write fib.py and run it", wait=True)   # a turn; returns its id
root.suspend(on_active="interrupt")              # -> state/coder-1.json, tree unmounted

# ... later, in another process:
root = hydrate_root(App, session_id="coder-1", store=store)   # same id, state back, cut turn re-run
```

### Roots

- `create_root(App, *, session_id=None, store=None, runner=None, **session_opts) -> Root`:
  a fresh tree. `hydrate_root(App, *, session_id, store=None, snapshot=None,
  runner=None, resume_turns=True, consume=True, wait=False)`: a tree hydrated
  from `snapshot` (a dict or a path) or from `store`. Async: `acreate_root` /
  `ahydrate_root`.
- Roots sit on top of `AppRunner` / `Session` (the engine). `runner=None`
  creates a runner owned by the root (stopped with it); pass `runner=r` to
  host many roots on one loop. `session_opts` go to `open_session`
  (`turn_policy="interrupt"`).
- `root.send(text, wait=False, timeout=None)` / `await root.asend(...)`:
  input (returns the turn id); `root.wait_turn(id)`, `root.wait_idle()`.
- `root.ref(name)`: the handle a component exposed with
  `use_imperative_handle(ref, factory, name=name)` (or a `ref="name"` prop),
  read from the session's `"refs"` registry — never `session.services`.
- `root.dehydrate() -> Snapshot`: the live tree's snapshot, nothing unmounted.
- `root.suspend(on_active="wait"|"interrupt", timeout=None) -> Snapshot`:
  1. `session.suspending`; new input is refused, queued turns are held;
  2. the active turn ends — `"wait"` lets it finish (interrupted after
     `timeout` s), `"interrupt"` cancels it now; a cut turn is **carried**;
  3. the tree is hidden like an `Activity`: every effect cleanup runs with
     the reason `"suspend"` (async cleanups awaited) — pause or snapshot what
     you own instead of destroying it;
  4. the tree is dehydrated and saved to `store`; `session.suspended {ref}`;
  5. the tree unmounts.
  If dehydrating or saving fails nothing is unmounted: effects re-run, held
  turns start, the cut turn is re-queued, `session.suspend.failed`,
  `SuspendError`.
- `root.unmount()`; `root.events` / `root.observer` / `root.event_bus`: the
  session's observability, for devtools (not for rendering).
- On hydrate, turn numbering continues (`coder-1/t3` after `t2`) and carried
  turns re-run with the same ids (`turn.resumed`), after the first render.
  Like `resume_pending`, a re-run is at-least-once for its side effects.

### `use_dehydrate` — hooks own their serialization

```python
value = use_dehydrate(key, *, dehydrate, hydrate=None, initial=None)
```

- Registers a slice addressed by `(component path, key)`; the registration
  follows the component's lifecycle. `dehydrate()` (sync or async) returns
  JSON; the latest render's function is used.
- During hydration the hook returns the hydrated value **in the component's
  first render**: `hydrate(slice)` if given, else the slice; otherwise
  `initial`. The value is stable across renders (feed it to `use_state` /
  `use_ref`).
- An async `hydrate` holds the component (it renders nothing, no effects)
  until it resolves; a `Hydrating(fallback=Spinner(), children=[...])`
  ancestor shows its fallback and holds its whole subtree meanwhile.
- **Hydration mismatches** are never fatal: a slice no hook claims, a hook
  with no slice (it starts from `initial`) or a failing `hydrate` publish
  `hydration.mismatch {kind: "unclaimed"|"missing"|"error", path, key}` and
  warn (`HydrationWarning`); `root.mismatches` lists them. Keep paths stable
  with keys (providers, `Activity` and `Hydrating` are transparent).
- Low level: `dehydrate_tree(root_ctx)` / `await adehydrate_tree(root_ctx)`
  build a snapshot from any mounted tree; a non-JSON value or a failing
  `dehydrate` raises `SnapshotError` naming the path and key.

- Libraries mismatch-report what they find later (a sandbox snapshot
  gone when the sandbox boots) with `report_hydration_mismatch(key, message,
  ctx=...)` — it lands in `root.mismatches` too.

### The libraries' slices: the whole agent, no participants

Every library state is dehydrated by the hook / provider that owns it, so
`root.suspend()` carries the whole agent with no runner configuration:

| owner | slice (`path` → `key`) | value | on hydrate |
|---|---|---|---|
| `Durable(log=...)` | `…/Durable[k]` → `"log"` | `{path, log_id, leaf, entries, version, pending_turns}` | log reopened at the leaf before the children render (`durable_lm` replays a re-run turn) |
| `SandboxProvider` | `…/SandboxProvider[k]` → `"sandboxes"` | `{key: {driver, spec, sandbox_id, snapshot}}` | restore from the snapshot / reattach; else fresh + mismatch |
| `use_sandbox(scope="component")` / provider-less | `…/Comp[k]` → `"sandbox"` | one record | same |
| `LearnedProgram` / `LearnedPrograms` | `…/LearnedProgram[name]` → `"programs"` | `{name: path \| {"state": ...} \| None}` (the last swap) | the program is applied on the first render |
| `DSPyProvider` | `…/DSPyProvider[k]` → `"compiled"` | `[{key, key_str, arch, path \| state}]` (`env.compiled`) | applied when `use_module` / `use_agent` build the module |
| the turn manager | `snapshot["turns"]` | `{next_index, policy, carry}` | numbering continues; carried turns re-run once hydration completed |

On suspend, effect cleanups see `use_cleanup_reason() == "suspend"`:
`Durable` records its leaf before closing the log, sandboxes snapshot or
pause per `on_close` (`resolve_policy(on_close, "suspend")` never destroys
implicitly). A hidden `Activity` keeps logs open and sandboxes running.

### `Activity` and cleanup reasons

`Activity(mode="visible"|"hidden", children=[...])` keeps a subtree mounted
with its state while hidden: its effects are cleaned up and do not run, so
it gets **no new turns** (a turn that started while hidden is not delivered
on reveal); `"visible"` re-runs its effects. Cleanups know why they run —
`use_cleanup_reason()` (or a cleanup taking one argument) is `"unmount"`,
`"deps"`, `"hidden"` or `"suspend"`:

```python
def fx():
    sb = start_sandbox()
    def cleanup(reason):
        sb.snapshot() if reason == "suspend" else sb.destroy()
    return cleanup
hooks.use_effect(fx, [])
```

### Snapshot format

```json
{
  "format": "pyreact.snapshot/1",
  "session_id": "coder-1",
  "created_at": 1793726199.04,
  "slices": {
    "App/Coder[coder]": {"history": [{"user": "write fib.py", "agent": "ok: write fib.py"}]}
  },
  "turns": {"next_index": 3, "policy": "queue",
            "carry": [{"id": "coder-1/t2", "index": 2, "text": "slow", "source": "term", "ts": 1793726198.93}]},
  "versions": {"snapshot": 1, "pyreact": "0.1.0", "python": "3.13.0"}
}
```

(Suspended with `on_active="interrupt"` while `coder-1/t2` was in flight, so
it is carried.) Stores: `MemorySnapshots()`, `DirectorySnapshots(dir)` (one
`<id>.json` per session, atomic writes), or anything implementing the
`SnapshotStore` protocol (`save(snapshot) -> ref`, `load(id)`, `delete(id)`,
`list()`). `load_snapshot(path)` / `check_snapshot(d)` validate one; a newer
`format` is refused (`SnapshotError`). A snapshot loaded from a store is
consumed on hydrate (`consume=True`).

### Session lifecycle hooks

`app.on_session(open=fn, close=fn, replay=False)` calls `open(session)` on
the runner loop when a session (or root) opens or is hydrated — **before its
first render** — and `close(session)` after its tree unmounted
(`session.close_reason`: `"closed"`, `"idle"`, `"shutdown"`, `"suspend"`).
`pyreact_otel`'s `attach_app(app)` uses it. Events on the session bus:
`session.opened {resumed}`, `session.suspending`, `session.suspended {ref,
turn_seq, carried_turns}`, `session.suspend.failed {error}`,
`session.resumed {turns, mismatches}`, `hydration.mismatch`,
`session.closed {reason}`.

### Deprecated: participants, `session.suspend()`, `app.resume()`

Before roots, libraries plugged state into `Session.suspend()` /
`AppRunner.resume()` as *participants* writing a `pyreact.manifest/1`:

```python
app = AppRunner(App, participants=[MyParticipant],        # DeprecationWarning
                manifest_store=DirectoryManifestStore("runs/manifests"))
manifest = app.open_session("agent-1").suspend()  # DeprecationWarning -> root.suspend()
s = app.resume("agent-1")                         # DeprecationWarning -> hydrate_root(...)
```

They are kept for one release as thin wrappers over the root machinery
(same events; the manifest carries the tree's slices in
`manifest["slices"]` and the turns in `participants.turns`), and
`participants=` / `add_participant` / `session.suspend` / `app.resume` warn
(`DeprecationWarning`, pointing to ADR 0001). The library
participants (`DurableParticipant`, `CompiledModulesParticipant`,
`SandboxParticipant`) have been removed: `Durable`, `DSPyProvider` /
`LearnedProgram` and `use_sandbox` dehydrate themselves. `idle_policy="suspend"` / `auto_resume` keep working
without warnings. Use roots in new code.

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

Run from the repository root (examples are a package). Every example talks to
its tree through a root (`create_root(App)`, `root.send(...)`, `root.ref(name)`)
and prints its output from `Message` components; `examples/_devtools.py` is the
terminal loop (`run_repl(root)`) and the one place that traces events. In the
terminal every line is a turn; `:help` lists the dev commands (`:events`, `:usage`, `:turns`,
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
a better program, a **learned program is a prop** (hot swap = a re-render,
like HMR), and **events** are the trace you debug and learn from
(`integrations.learn`).

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
result.save("artifacts/router.json")
```

### Learned programs are props (HMR = re-render)

A learned program is declared in the tree, like any other prop or context.
Changing it re-renders the components that use it; nothing outside the tree
mutates a module cache.

```python
from integrations.learn import LearnedProgram, LearnedPrograms, use_learned

@component
def Router():
    router = use_module(RouteSig, arch="cot", name="supervisor",
                        program=use_learned("supervisor"))   # None -> the baseline
    ...

LearnedProgram(key="supervisor", src="artifacts/router.json", children=[Router()])
# many at once, e.g. at the root, fed by state/props:
LearnedPrograms(programs={"supervisor": "artifacts/router.json", "coder": result},
                children=[...])
```

- `src`: a result file (`integrations.learn.load`), an `OptimizationResult`, a program, or
  `None`. `watch=True` polls the file and re-renders when it changes (off by default).
- `program=` on `use_module` / `use_agent` / `use_agent_call` / `use_call` / `use_stream`
  applies the learned state (instructions, demos) onto the component's own module: a ReAct
  agent keeps its live tools. Back to `None` restores the baseline.
- **Hot swap** = the provider's `src` changes. From outside the tree,
  `result.hot_swap(app, "supervisor")` (and `:optimize`) update the state of the provider that
  declares the name — `{name: path}`, one JSON-able value — and its users re-render with the
  new program. Apps without a provider keep the old behavior (the module in the `DSPyEnv` is
  replaced). In the tree, `use_learned_registry().set(name, src)` does the same.

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

# serve again: the artifact is a prop of the tree
LearnedProgram(key="supervisor", src=path, children=[StoreRouter()])
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
7. **Is the new program live?** after a new `src` / `result.hot_swap(app, key)` / `:optimize`,
   `:modules` shows `compiled swaps=1`, `find_registries(app)` shows the provider's
   `{name: path}`, and the next `:explain` shows the new instructions.

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