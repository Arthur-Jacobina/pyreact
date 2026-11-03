# ADR 0001 — Look like React: the tree is the source of truth

Status: accepted · Supersedes the participant/manifest front API of #80–#86

## Context

pyreact's founding principle is the React analogy: agents are components,
memory is hooks, side effects are effects, shared services are context. The
first layers followed it. Later features were bolted on as imperative objects
configured *outside* the tree, and resume (#80–#86) is the clearest case:

```python
AppRunner(App, participants=[DurableParticipant, lambda s: SandboxParticipant(s, ...)],
          manifest_store=DirectoryManifestStore(...))          # duplicates what App declares
h = session.services[HANDLES_SERVICE][f"{session.id}/default"]   # service locator
h.run_sync(lambda: h.write_file("TASK.md", TASK))                 # side effect outside an effect
res.hot_swap(session, key)                                        # imperative mutation
```

## Principles (checklist for every API)

1. **The tree is the single source of truth.** Anything a component declares
   (a durable log, a sandbox, a learned program) is not configured again
   outside the tree.
2. **State is owned by the component that declares it,** through a hook. The
   hook knows how to persist / dehydrate / hydrate its own state.
3. **Data flows down** (props, context). Outside code talks to the tree only
   through the root (`root.send`, `root.suspend`), never by reaching into
   services.
4. **Side effects live in effects** (with cleanup). Imperative access to a
   child goes through a ref (`use_ref` + `use_imperative_handle`), not a
   service lookup.
5. **Change is a re-render:** swapping a program, a model or a policy is a
   prop or context change, not a mutation.
6. **Events are for devtools,** not for rendering. Output is components.

## Decisions

### Roots (cf. `createRoot` / `hydrateRoot`)

```python
root = create_root(App, session_id="coder-1", store=DirectorySnapshots("state/"))
root.send("hello", wait=True)          # was session.invoke
root.suspend(on_active="interrupt")    # Activity hidden + dehydrate -> store
root.unmount()

root = hydrate_root(App, session_id="coder-1", store=DirectorySnapshots("state/"))
```

- `create_root` / `hydrate_root` sit on top of `AppRunner` / `Session` (the
  engine stays). A runner can host many roots (`create_root(..., runner=r)`).
- `root.dehydrate() -> Snapshot` / `hydrate_root(App, snapshot=...)` for
  in-memory use; `store=` persists snapshots (memory / directory; libraries
  may provide more).
- **Snapshot** = the dehydrated tree: `{"format": "pyreact.snapshot/1",
  "session_id", "created_at", "slices": {component_path: {key: value}},
  "turns": {...}, "versions": {...}}`. JSON only.
- **Hydration mismatch** (React's warning): a slice whose path/key no hook
  claims on the first render, or a hook with no slice, is reported as a
  `hydration.mismatch` event (and a warning), never fatal. A hook without a
  slice starts fresh.

### `use_dehydrate` — hooks own their serialization (cf. TanStack Query `dehydrate` / `HydrationBoundary`)

```python
value = use_dehydrate(key, dehydrate=lambda: current_value, hydrate=lambda slice: ...)
```

- Scoped to the component path (+ key). On `root.suspend()/dehydrate()` the
  root walks the tree and calls every registered `dehydrate` (sync or async).
  On `hydrate_root`, `hydrate(slice)` runs **before the component's first
  render completes** (the hook returns the hydrated value during that
  render); async hydrate suspends the subtree (renders a fallback via
  `Suspense`-like `Hydrating(fallback=...)` if present, else nothing) until
  done.
- This is the only core primitive. `Durable` (log + leaf), `use_sandbox`
  (snapshot ref), `use_module` / `use_learned` (compiled program) and the
  turn manager (numbering + carried turns) call it internally. The core never
  imports those libraries.

### `Activity` (cf. React `<Activity mode="hidden"|"visible">`)

```python
Activity(mode="visible" | "hidden", children=[...])
```

- `hidden`: the subtree keeps its state (hooks), its effects' cleanups run
  with `cleanup_reason="hidden"`; no new turns are delivered to it.
  `visible`: effects re-run.
- `root.suspend()` = make the root's activity `hidden` (cleanup reason
  `"suspend"`), dehydrate, unmount. Effects can read `use_cleanup_reason()`
  (or receive the reason as an argument to the cleanup) to pause/snapshot a
  sandbox instead of destroying it.

### Learned programs are props (cf. HMR)

```python
agent = use_agent(Coder, tools=..., program=use_learned("coder"))   # or:
LearnedProgram(key="coder", src="artifacts/coder.json", children=[...])
```

- Hot swap = the `src`/program changes → the component re-renders with the
  new program. `OptimizationResult.hot_swap` remains as a thin wrapper that
  updates the `LearnedProgram` context.

### Hooks instead of service lookups

| instead of | use |
|---|---|
| `hooks.get_service("input_bus", ...)` | `use_input()` / `use_on_turn(fn)` |
| `hooks.get_service("event_bus", ...)` (+ `.subscribe` in components) | `use_event(kind, fn)` (subscribe in an effect) / `use_emit()` |
| `hooks.get_service("nav_service", ...)` | `use_navigate()` / `use_location()` (exist) |
| `session.services[...]` from outside | a `ref=` prop + `use_imperative_handle(ref, lambda: handle)`, read from the root (`root.ref("coder")`) |
| `app.event_bus.subscribe(show)` to print output | `Message` components + a root-level renderer (`root.render_to(console)`); bus subscriptions stay for devtools/observability |

`hooks.get_service` stays as a low-level escape hatch (documented as such);
examples must not use it (enforced by a test).

## Migration

Additive first: new APIs land next to the old ones; participants and
`app.resume` become thin deprecated wrappers over roots + `use_dehydrate`;
examples move to the new APIs; a test forbids `get_service` /
`session.services` / `participants=` in examples.

## Consequences

- One mental model: if you know React, you know where state lives, how it
  survives, and how to change it.
- Libraries stay decoupled (they implement hooks, not participants).
- Cost: a dehydration walk on suspend; hydration mismatches must be handled
  explicitly (reported, never silent).
