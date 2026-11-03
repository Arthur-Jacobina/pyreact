# pyreact_sandbox — pluggable, bootstrappable sandboxes for pyreact agents

Agents that *do* things (run code, install packages, keep files between
turns) need a place to do them that is not your machine. `pyreact_sandbox`
makes that place a first-class resource with a React-style lifecycle: a
provider picks the vendor, a hook gives a component its sandbox, effects
create it, cleanups close it, and a snapshot store brings it back after a
restart. **Docker first**; other vendors are drop-in drivers.

It is a **separate library at the border** (like
[`pyreact_durable`](../pyreact_durable/README.md)): nothing in `pyreact/`
imports it, and it plugs in only through pyreact's public surface (hooks,
`hooks.provide`, the session `EventBus`, the harness `through`).

```python
from pyreact.core import component, hooks
from pyreact_sandbox import SandboxProvider, SandboxSpec, sandbox_tools, use_sandbox
from integrations import use_agent

@component
def Coder():
    sb = use_sandbox()                                     # this session's sandbox
    tools = hooks.use_memo(lambda: sandbox_tools(sb), [id(sb)])
    agent = use_agent("question -> answer", tools=tools)   # ReAct over run_shell, ...
    ...

@component
def App():
    return [SandboxProvider(driver="docker",
                            spec=SandboxSpec(image="python:3.12-slim", memory="512m"),
                            children=[Coder(key="coder")])]
```

Full example (Docker, real LM, restart + restore):
[`examples/sandbox/01_coding_agent.py`](../examples/sandbox/01_coding_agent.py).

## Concepts

| | what | React analogy |
|---|---|---|
| `SandboxSpec` | image, template, env, cpu, memory, network, workdir, ttl, labels, setup | props |
| `SandboxDriver` | creates / attaches sandboxes (Docker, local, E2B, ...) | the renderer (DOM, native) |
| `Sandbox` | one environment: `exec`, files, snapshot, suspend, destroy | a host instance |
| `SandboxProvider` | driver + default spec + policies for a subtree | context provider |
| `use_sandbox()` | a `SandboxHandle` for the component | hook (an effect with cleanup) |
| `SnapshotStore` | `key -> {sandbox_id, snapshot}` record for restarts | persisted state |
| `sandbox.exec` | harness boundary around every exec | middleware / interceptors |
| `sandbox.*` events | lifecycle + exec on the session `EventBus` | DOM events |

## Lifecycle

```
              mount (eager) / first use
                         │
             store.get("<session>/<key>")
          ┌──────────────┼──────────────────────┐
   record.sandbox_id   record.snapshot       nothing
   alive? attach        driver.create(        driver.create(spec)
   (+resume)            spec, snapshot=ref)   (setup cached by spec hash)
   sandbox.restored     sandbox.restored      sandbox.created
   mode=attach          mode=snapshot
          └──────────────┼──────────────────────┘
                         ▼
        exec / run_python / write_file / read_file
        ── through("sandbox.exec", name, inputs) ──► sandbox.exec event
                         │
      end of turn (autosnapshot="turn", if touched; before turn.end)
                 snapshot ─► store.set(...)  ─► sandbox.snapshot
                         │
     unmount (scope="component") / provider unmount / session.closed
                         ▼
            on_close = "destroy"  → destroy           (record keeps the last snapshot)
                     | "suspend"  → pause             (restart reattaches)
                     | "snapshot" → snapshot+destroy  (restart restores)
                     | "keep"     → leave it running
```

- **scope** `"session"` (default): one sandbox per `key` per session/provider,
  shared by every component asking for it. `"component"`: owned by the
  component (default key: its path), closed on unmount.
- **Background loop**: sandbox I/O runs on a private event loop thread, so it
  does not block renders and survives the render loop's shutdown. Closing a
  session while a container is still being created never leaks it.
  `wait_closed()` (also registered at exit) waits for pending
  closes/snapshots.
- **Threads**: `handle.run_sync(lambda: handle.exec("ls"))` from worker
  threads (e.g. an RLM REPL); it runs on the app loop so middleware and
  owner attribution behave as in components.

### Events (session `EventBus`)

```
sandbox.created    {key, sandbox_id, driver, image, template, duration}
sandbox.restored   {key, sandbox_id, driver, mode: "attach"|"snapshot", snapshot, duration}
sandbox.exec       {key, sandbox_id, op, name, command, exit_code, stdout, stderr,
                    duration, timed_out, error, boundary_id}      (turn_id set)
sandbox.snapshot   {key, sandbox_id, snapshot, label, auto}
sandbox.suspended / sandbox.resumed / sandbox.destroyed {key, sandbox_id[, policy]}
sandbox.error      {key, phase: create|restore|store|close:*|autosnapshot, error}
```

### Harness boundary `sandbox.exec`

Every exec-like call is one boundary: kind `sandbox.exec`, name = the command
(`python3`, `ls`...), `python`, `write_file` or `read_file`; `inputs` =
`{op, sandbox, cmd|code|path, timeout, env, cwd, ...}` (a middleware may
rewrite them); attrs `sandbox_id`, `driver`. It nests under the turn / tool /
llm boundary that triggered it (`parent_id`).

```python
@middleware(kinds=["sandbox.exec"])
async def no_curl(b, call_next):
    if b.name in ("curl", "wget"):
        raise Blocked("no downloads from the sandbox", boundary=b, by="policy")
    return await call_next(b)

HarnessProvider(middleware=[no_curl], children=[SandboxProvider(...)])
```

## Driver protocol

```python
class SandboxDriver:
    name: str
    async def create(spec, *, snapshot: SnapshotRef | None = None, sandbox_id=None) -> Sandbox
    async def attach(sandbox_id) -> Sandbox            # SandboxNotFound if gone
    async def list(labels=None) -> list[str]            # optional
    async def delete_snapshot(ref) -> None              # optional
    def gc_sync(**kw) -> dict                           # optional (pyreact_sandbox.gc)
    def available() -> bool

class Sandbox:
    id: str; spec: SandboxSpec; driver: SandboxDriver
    async def exec(cmd: str | list[str], *, timeout=None, env=None, cwd=None, stdin=None) -> ExecResult
    async def run_python(code, ...) -> ExecResult       # default: python3 - <<code
    async def write_file(path, data) / read_file(path) -> bytes   # defaults via exec
    async def snapshot(label=None) -> SnapshotRef       # optional (NotSupported)
    async def restore(ref)                              # optional; else create(spec, snapshot=ref)
    async def suspend() / resume()                      # optional
    async def destroy() / status() -> str               # running|suspended|stopped|destroyed
    def open_process_sync(argv, *, env=None, cwd=None) -> SandboxProcess   # optional (RLM)

ExecResult(exit_code, stdout, stderr, duration, timed_out)   # .ok .output .check()
SnapshotRef(driver, id, sandbox_id, label, created_at, meta) # JSON: to_dict/from_dict
```

Most vendor SDKs are blocking: subclass `BlockingSandbox` / `BlockingDriver`
and implement `exec_sync`, `destroy_sync`, `status_sync`, `create_sync`,
`attach_sync` (+ optional `*_sync`); every async method runs its twin in a
worker thread. Register it:

```python
from pyreact_sandbox import register_driver, get_driver
register_driver("e2b", lambda **kw: E2BDriver(**kw))
SandboxProvider(driver="e2b", ...)        # or driver=E2BDriver(api_key=...)
```

Built-ins:

- **`docker`** (`DockerDriver`, the `docker` CLI; no Python dependency).
  Labels everything `pyreact.sandbox=1`; `--network none` by default,
  `--cpus`/`--memory`/`--pids-limit`, `--cap-drop ALL` +
  `no-new-privileges`, optional `runtime="runsc"` (gVisor) / Kata; `ttl` =
  `sleep <ttl>` entrypoint. Bootstrap: `spec.setup` runs once in a build
  container committed to `pyreact-sandbox-tpl:<setup hash|template>` (6.0s →
  0.4s on the second boot here); `build_template(spec)` pre-builds it.
  Snapshot = `docker commit` → `pyreact-sandbox:<id>-<label>`; restore =
  recreate the container (same name) from it. Suspend = `pause` (or `stop`).
- **`local`** (`LocalDriver`): a temp dir + host subprocesses. **NOT
  ISOLATED** — for tests, offline examples and development only.
- **`FakeDriver`** (`pyreact_sandbox.testing`): in-memory, for unit tests.

### Adding a vendor (sketches, not implemented)

**E2B** (Firecracker microVMs; SDK `e2b`):

```python
class E2BSandbox(BlockingSandbox):
    def __init__(self, driver, sbx, spec): self.driver, self.sbx, self.id, self.spec = driver, sbx, sbx.sandbox_id, spec
    def exec_sync(self, cmd, *, timeout=None, env=None, cwd=None, stdin=None):
        r = self.sbx.commands.run(cmd if isinstance(cmd, str) else shlex.join(cmd),
                                  timeout=timeout, envs=env, cwd=cwd or self.spec.workdir)
        return ExecResult(r.exit_code, r.stdout, r.stderr)
    def write_file_sync(self, path, data): self.sbx.files.write(path, data)
    def read_file_sync(self, path): return self.sbx.files.read(path, format="bytes")
    def suspend_sync(self): self.sbx.beta_pause()            # resume = Sandbox.connect(id)
    def destroy_sync(self): self.sbx.kill()
    def status_sync(self): return Status.RUNNING if self.sbx.is_running() else Status.STOPPED

class E2BDriver(BlockingDriver):
    name = "e2b"
    def create_sync(self, spec, *, snapshot=None, sandbox_id=None):
        sbx = e2b.Sandbox(template=spec.template, envs=dict(spec.env), timeout=spec.ttl or 3600,
                          metadata=dict(spec.labels))
        return E2BSandbox(self, sbx, spec)
    def attach_sync(self, sandbox_id): return E2BSandbox(self, e2b.Sandbox.connect(sandbox_id), SandboxSpec())
```

**Modal** (`modal.Sandbox`): `create_sync` → `modal.Sandbox.create(image=modal.Image.from_registry(spec.image).run_commands(*spec.setup), cpu=spec.cpu, memory=…, block_network=not spec.network, timeout=spec.ttl, app=app)`;
`exec_sync` → `p = sb.exec("sh", "-c", cmd, timeout=…); p.wait(); ExecResult(p.returncode, p.stdout.read(), p.stderr.read())`;
`snapshot_sync` → `img = sb.snapshot_filesystem()` → `SnapshotRef("modal", img.object_id)`, create-from-snapshot = `Sandbox.create(image=modal.Image.from_id(ref.id))`;
`attach_sync` → `modal.Sandbox.from_id(id)`; `destroy_sync` → `sb.terminate()`.

**Daytona** (`daytona` SDK): `create_sync` → `daytona.create(CreateSandboxFromSnapshotParams(snapshot=spec.template, env_vars=…, labels=…, network_block_all=not spec.network, auto_stop_interval=…))`;
`exec_sync` → `sb.process.exec(cmd, cwd=…, env=…, timeout=…)` (`.exit_code`, `.result`);
files → `sb.fs.upload_file / download_file`; `suspend_sync` → `sb.stop()` / resume `sb.start()` (or `archive()`);
`attach_sync` → `daytona.get(id)`; `destroy_sync` → `sb.delete()`.

**Firecracker / in-house**: wrap your control plane the same way; if it only
offers HTTP exec, the defaults for `run_python` / `write_file` / `read_file`
work on top of `exec` alone. Implement `open_process_sync` (a bidirectional
stream: SSH, a websocket, `kubectl exec -i`) to host the RLM interpreter.

## Bootstrappable + stateful: stores and restarts

```python
SandboxProvider(driver="docker", spec=spec, store=FileSnapshotStore("runs/sandboxes.json"),
                autosnapshot="turn", on_close="destroy", children=[...])
```

`SnapshotStore` is three methods (`get`, `set`, `delete`) over JSON records:
`MemorySnapshotStore` (default), `FileSnapshotStore`, or anything else
(Redis, a DB row). The **`pyreact_durable` adapter** keeps the record in the
session log, so the sandbox follows the conversation (checkout a label → the
sandbox restores the snapshot recorded there):

```python
from pyreact_sandbox.contrib.durable import use_durable_snapshot_store

@component
def Agent():
    sb = use_sandbox(store=use_durable_snapshot_store(), autosnapshot="turn")
    conv = use_conversation()
    ...

Durable(log="runs/agent.jsonl", children=[SandboxProvider(driver="docker", ..., children=[Agent()])])
```

`pyreact_sandbox.contrib.durable` imports both libraries lazily; neither
imports the other. Use `autosnapshot="turn"` with it: on shutdown `Durable`
closes its log before the sandbox's close policy runs, so a record written at
close would be dropped (with a warning).

What survives what:

| | process exit (clean) | crash / `kill -9` | `docker` daemon restart |
|---|---|---|---|
| `on_close="suspend"` | paused container, reattached | container keeps running; record → reattached | stopped → `resume` starts it |
| `autosnapshot="turn"` | snapshot of the last turn | snapshot of the last finished turn | snapshot image |
| `on_close="snapshot"` | snapshot at close | (no close) → the record's live id is reattached if alive | snapshot image |

### Suspend / hydrate with the whole agent (roots)

`use_sandbox` / `SandboxProvider` own their state (ADR 0001): on
`root.suspend()` the effect cleanup (`use_cleanup_reason() == "suspend"`)
snapshots or pauses each sandbox per `on_close`, and the provider's
`use_dehydrate` slice records how to bring it back. No participant:

```python
root = create_root(App, session_id="agent", store=DirectorySnapshots("runs/state"))
root.suspend(on_active="interrupt")
root = hydrate_root(App, session_id="agent", store=DirectorySnapshots("runs/state"))
```

```json
"slices": {"App/SandboxProvider[sandbox]": {"sandboxes": {"default": {
    "driver": "docker", "spec": {}, "sandbox_id": null, "snapshot": {"id": "pyreact-sandbox:…"}}}}}
```

The suspend policy is `resolve_policy(on_close, "suspend")`, which **never
destroys implicitly**: `"destroy"` (a plain policy or a mapping's `"*"`)
becomes `"snapshot"`. Name `"suspend"` in a mapping to choose:

| `on_close` for `"suspend"` | at suspend | at hydrate |
|---|---|---|
| `"snapshot"` (also the default `"destroy"`) | snapshot, then destroy | new sandbox from the snapshot |
| `{"suspend": "suspend", ...}` | pause (`docker pause`), keep | reattach + unpause (else the last snapshot) |
| `{"suspend": "keep", ...}` | leave running | reattach |
| `{"suspend": "destroy", ...}` | destroy (nothing carried) | fresh sandbox |

The slice's record wins over the provider's `SnapshotStore` for the same
key. If the snapshot is gone (or the sandbox, or the driver changed), a
fresh sandbox boots and a `hydration.mismatch` (`kind: "error"`,
`reason: "snapshot missing"` ...) plus `session.resume.degraded`
(`participant: "sandbox"`) are published. `root.suspend()` returns once the
sandboxes reached their final state (the dehydrate awaits the closes).
`scope="component"` sandboxes (and provider-less ones) are dehydrated by
`use_sandbox` in the calling component (slice `"sandbox"` / `"sandbox:<key>"`).
Under a hidden `Activity` sandboxes keep running.

## Integrations

**RLM in Docker** — `SandboxInterpreter(sandbox_or_handle)` implements dspy's
`CodeInterpreter`: a stdlib REPL server runs inside the sandbox
(`docker exec -i python3 -u -c …`); `llm_query` / tools / `SUBMIT` are
called back on the host over the same pipe, so the LM's code needs no
network or credentials.

```python
sb = use_sandbox()
rlm = use_module("context, question -> answer", arch="rlm",
                 interpreter=lambda: SandboxInterpreter(sb))
```

**Tools** — `sandbox_tools(handle, timeout=60, max_output=4000, prefix="", include=None)`
→ `run_shell`, `run_python`, `read_file`, `write_file` (async pyreact `Tool`s
for `use_agent(tools=...)` or `ToolProvider`); each is a `tool` boundary with
the `sandbox.exec` boundary nested inside.

## Garbage collection

```python
import pyreact_sandbox as ps
ps.gc()                                    # expired (ttl) + orphaned (dead owner pid) sandboxes
ps.gc(all=True, snapshots=True, templates=True, labels={"app": "x"})   # everything of yours
ps.gc(dry_run=True)
```

Docker: containers carry `pyreact.sandbox.owner=<host>:<pid>`; a container
whose owner process on this host is gone is an orphan unless it is *paused*
(a deliberate `suspend`). Snapshot (`pyreact-sandbox:*`) and template
(`pyreact-sandbox-tpl:*`) images are only removed on request.

## Security notes

- A container is not a VM: kernel exploits escape it. For untrusted code at
  scale use gVisor (`DockerDriver(runtime="runsc")`), Kata, Firecracker
  (E2B), or a vendor. **`LocalDriver` has no isolation at all.**
- Defaults are deny-ish: no network, capped CPU/memory/pids, all capabilities
  dropped, `no-new-privileges`. The container still runs as root *inside*
  (add `extra_args=["--user", "1000:1000", "--read-only", "--tmpfs", "/tmp"]`
  if your image allows it).
- `setup` commands have network (`setup_network=True`) — pin versions; the
  result is cached by the spec hash, so a compromised package stays cached
  until you `gc(templates=True)`.
- Secrets: do not put them in `spec.env` (they end up in container labels
  and `docker inspect`). Keep them host-side and expose capabilities as
  tools; the RLM interpreter already calls tools on the host.
- Snapshots contain everything the agent wrote (including anything it
  fetched or generated); treat snapshot images like the data they hold.
- `sandbox.exec` events carry previews of commands/output (400 chars); the
  journal (`pyreact_durable`) persists them if you journal `sandbox.*`.
- Guard with harness middleware (`Blocked`) for policy (no `curl`, max
  timeout, allowed commands); the boundary runs before anything reaches the
  driver.

## Known gaps

- Snapshot chains: each restore + snapshot cycle adds an image layer (Docker
  caps an image at 127); there is no squash yet.
- `exec` buffers output (no streaming); `sandbox.exec` events show previews.
- `attach` does not prevent two processes from using the same sandbox.
- Docker `suspend_mode="stop"` containers look like orphans to `gc()` once
  their owner exits (paused ones do not).
- No file upload/download of directories (use `exec` + `tar`).
