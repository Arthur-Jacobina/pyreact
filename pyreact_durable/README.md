# pyreact_durable — durable agent memory at the border of the flow

Agent memory that survives crashes and restarts, can branch and be rewound,
and journals every LM call so re-running a turn replays instead of re-paying.
It is a **separate library**: nothing in `pyreact/` imports it, and it plugs
in only through pyreact's public surface (hooks, a provider component, the
session `EventBus`) and, optionally, by wrapping a `dspy.BaseLM`.

```python
from pyreact.core import component
from pyreact_durable import Durable, use_conversation, use_persistent_state, use_durable

@component
def Chat():
    turns, set_turns = use_persistent_state(0, key="turns")
    conv = use_conversation()                 # persisted [{"role", "content"}]
    log = use_durable()                       # log.label(...), log.checkout(...)
    ...

@component
def App():
    return [Durable(log="chat.jsonl", children=[Chat(key="chat")])]
```

Kill the process any way you like (`kill -9` included), start it again: `Chat`
comes back with the same `turns` and transcript.

## Examples

Smallest first, in [`examples/durable/`](../examples/durable/) (each module
docstring has the run and inspect commands):

| example | teaches | LLM |
|---|---|---|
| `01_counter` | `use_persistent_state` / `use_persistent_reducer` under `Durable(log=...)` | no |
| `02_branches_sqlite` | SQLite backend; `/label`, `/checkout`, `/fork`, `/tree` via `use_durable()` | no |
| `03_web_sessions` | `run_web` + `Durable(log_factory=...)`: one log per browser tab | yes |
| `04_llm_chat` | everything: `use_conversation`, branches, `durable_lm` replay | yes |

```bash
uv run python -m examples.durable.01_counter   # type + + -, :q, run again
```

## Layers

| module | depends on | role |
|---|---|---|
| `pyreact_durable.log` | stdlib only | the file format and `SessionLog` API |
| `pyreact_durable.react` | `pyreact` public API | `Durable` provider, persistent hooks |
| `pyreact_durable.lm` | `dspy` (lazily) | `durable_lm` record / replay |
| `python -m pyreact_durable` | `log` | CLI |

`import pyreact_durable` loads only the storage layer; the React and LM names
are resolved lazily on first access.

## File format (v1)

Inspired by [pi's session format](https://pi.dev/docs/latest/session): an
append-only JSONL file. Line 1 is the header; every other line is an entry.

```jsonl
{"type":"session","version":1,"id":"4f8185e0-80f3-42f1-8ab4-d702bc1ce8f4","timestamp":"2026-11-03T04:40:01.120Z"}
{"type":"event","id":"9a1c0b7e","parentId":null,"timestamp":"…","kind":"turn.start","payload":{"index":1,"text":"my favorite color is teal"},"turn_id":"t1","source":"term","ts":1793680801.1}
{"type":"llm","id":"7e667548","parentId":"9a1c0b7e","timestamp":"…","key":"54dde7d9a0c4e1f2","occurrence":0,"model":"claude-cli/haiku","response":{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"[[ ## answer ## ]]\nGot it…"}}],"usage":{},"model":"claude-cli/haiku"},"latency":4.1,"request":{"prompt":null,"messages":[…],"kwargs":{}}}
{"type":"state","id":"45ef500d","parentId":"7e667548","timestamp":"…","path":"App/Chat[chat]","key":"history","value":[{"role":"user","content":"my favorite color is teal"},{"role":"assistant","content":"Got it…"}]}
{"type":"label","id":"1a582c79","parentId":"45ef500d","timestamp":"…","targetId":"45ef500d","label":"base"}
{"type":"leaf","id":"0df787b2","parentId":"60ec4dc4","timestamp":"…","targetId":"45ef500d"}
{"type":"checkpoint","id":"c0ffee01","parentId":"45ef500d","timestamp":"…","state":{"App/Chat[chat]":{"history":[…]}},"summary":"after onboarding"}
{"type":"session_info","id":"5b2e9d10","parentId":"c0ffee01","timestamp":"…","name":"support chat"}
{"type":"custom","id":"6d3a4f21","parentId":"c0ffee01","timestamp":"…","customType":"note","data":{"any":"json"}}
```

**Header** `{"type":"session","version","id" (uuid4),"timestamp"}` plus any
`meta` given at creation. Files made by `fork_file` add `parentSession` (the
source file's path), `parentSessionId` and `forkedFrom` (the entry id).

**Entries** all carry `type`, `id` (8 hex chars), `parentId` (`null` for a
root) and `timestamp` (ISO-8601 UTC). Together they form a **tree**.

| type | fields | kind |
|---|---|---|
| `state` | `path`, `key`, `value` (JSON) or `deleted: true` | content |
| `event` | `kind`, `payload`, `turn_id`, `source`, `ts` (a journaled bus event) | content |
| `llm` | `key`, `occurrence`, `model`, `response`, `latency`, `request?` | content |
| `checkpoint` | `state` (full `{path: {key: value}}` snapshot), `summary?` | content |
| `custom` | `customType`, `data` | content |
| `label` | `targetId`, `label` (`targetId: null` removes it) | meta |
| `session_info` | `name` | meta |
| `leaf` | `targetId` (`null` = before the first entry) | meta |

Rules:

- A **content** entry is a child of the current leaf (or of an explicit
  earlier entry — that's a *branch*) and becomes the new leaf.
- A **meta** entry records the leaf it was written at as `parentId` but never
  becomes the leaf. A `leaf` entry moves the position without adding content,
  so `checkout` survives restarts.
- On load, the leaf is decided by the last content or `leaf` entry in file
  order. The latest `label` per name wins; the latest `session_info` names
  the session.
- **State at an entry** = walk entry → root, stop at the nearest
  `checkpoint`, fold its snapshot plus the `state` entries after it (root →
  entry order). With checkpoints, rebuilding state is O(entries since the
  checkpoint); results are cached per entry.
- **Versioning**: the header `version` is the format version. Loading an
  older file applies registered migrations (`register_migration(from_version)`
  or `SessionLog.open(..., migrations={0: fn})`, each `fn(header, entries) ->
  (header, entries)`), then rewrites the file atomically. Newer versions are
  refused.

### Durability guarantees

- One `os.write` per entry on an `O_APPEND` descriptor; `fsync=True` to also
  fsync each entry.
- **Single writer**: an exclusive, non-blocking `flock` on the file;
  `SessionLocked` if another writer (another process, or another open in the
  same process) holds it. The OS drops the lock when the process dies, so a
  `kill -9`'d app restarts without cleanup. Readers (`readonly=True`, the
  CLI's read commands) never lock.
- **Crash tolerance**: a torn or garbled trailing line is ignored, reported
  (`log.recovered` + a `RuntimeWarning`) and truncated before the next write;
  a valid last line missing only its newline is kept. A garbled line in the
  middle is skipped and reported too. Never fatal.

### Backends: JSONL or SQLite

The suffix picks the backend; the format and every API are identical:

```python
SessionLog.open("chat.jsonl")    # one JSON line per entry (default)
SessionLog.open("chat.sqlite")   # same lines, one row each (.sqlite / .sqlite3 / .db)
Durable(log="state.db", children=[...])
```

`SQLiteSessionLog` stores exactly the lines a `.jsonl` file would (header
first) in a `lines(seq, line)` table, so tree semantics and migrations are
shared. Each append is one committed `INSERT` (WAL; `synchronous=FULL` with
`fsync=True`), so there are no torn lines; the single-writer guard is a `flock`
on `<file>.lock`; `readonly=True` readers (e.g. the CLI) work while the app
writes. Convert between backends with `fork_file`:

```python
SessionLog.open("chat.sqlite").fork_file("chat.jsonl")   # export
SessionLog.open("chat.jsonl").fork_file("chat.db")       # import
```

Pick JSONL for greppable, diffable logs; SQLite when many sessions live in one
process or files get large (no whole-file reads by other tools).

## Storage API (`pyreact_durable.log`)

```python
log = SessionLog.open(path, create=True, fsync=False, readonly=False,
                      auto_checkpoint=None, meta=None)
log = SessionLog.create(path, meta={"app": "chat"})

log.leaf                                  # current entry id (None when empty)
log.append(type, parent_id=..., **fields) # -> entry dict (validated JSON)
log.set_state(path, key, value); log.delete_state(path, key)
log.state_at(entry_id=None)               # {path: {key: value}} (copy)
log.get_state(path, key, default, entry_id=None)
log.events(kinds=None, entry_id=None)     # event entries on the path; "turn.*" ok
log.path_to(entry_id=None)                # root -> entry
log.label(name, entry_id=None); log.unlabel(name); log.labels
log.resolve("label" | "id" | "id-prefix" | "@leaf" | "@root")
log.checkout(ref)                         # writes a `leaf` entry
log.fork_file(new_path, at=None)          # new file, parentSession header
log.checkpoint(summary=None)              # compaction snapshot
log.tree(); log.branch_points(); log.stats()
log.render(width=60)                      # the tree as text (same as `show`)
log.set_name("support chat"); log.name
log.subscribe(fn)                         # fn(entry, moved) after each append
log.close()                               # also a context manager
```

## pyreact border (`pyreact_durable.react`)

- **`Durable(log=SessionLog|path, journal=DEFAULT_JOURNAL, children=[...])`** —
  a provider component. It provides the log to its subtree (via
  `hooks.provide`), subscribes (in an effect) to the session's `EventBus` and
  appends an `event` entry for each kind in `journal` (default
  `input.*`, `message`, `turn.*`, `llm.*`, `tool.*`, `feedback`, `error`;
  payloads are made JSON-safe). Logs it opened from a path are closed on
  unmount; a `SessionLog` you pass stays yours. It also watches the log: when
  the leaf *jumps* (checkout / branch) it rerenders its subtree so persistent
  hooks reload from the new position.
- **Turn ids continue across restarts**: on mount `Durable` calls
  `TurnManager.advance_to(last journaled turn index + 1)`, so a restarted app
  numbers its turns `t5`, `t6`, ... instead of reusing `t1` (journaled
  `turn_id`s stay unique; needs `turn.*` in `journal`, the default).
- **Multiple sessions** (one app tree per user on one `AppRunner`): give each
  session its own file with a factory — it receives the session id:

  ```python
  Durable(log_factory=lambda sid: f"logs/{sid}.jsonl", children=[...])
  ```

- **`use_persistent_state(initial, *, key)`** → `(value, set_value)`, like
  `use_state`. Address = `(hooks.path, key)`; give components stable `key=`
  props so the path does not depend on sibling order. **Providers are
  transparent**: `Durable`, `DSPyProvider`, `ToolProvider`, `HarnessProvider`,
  every `create_context` provider and your own `@component(transparent=True)`
  components add no path segment, so `Chat(key="chat")` under `App` is
  `App/Chat[chat]` no matter which providers sit in between — wrapping the
  tree in another provider later does not orphan its state. Logs written
  before this rule stored state under the every-node path
  (`App/Durable[durable]/DSPyProvider[dspy]/_Provider[#0]/Chat[chat]`,
  `hooks.full_path`); restore falls back to it and the next set rewrites it
  under the new path, so old logs keep working without a migration. First render restores
  from the leaf; each effective set appends a `state` entry. Non-JSON values
  raise `DurableStateError` naming the component path and key. **Without a
  `Durable` above** it behaves exactly like `use_state` and warns once.
- **`use_persistent_reducer(reducer, initial, *, key, init_fn=None)`** →
  `(state, dispatch)`; the resulting state (not the action) is persisted.
- **`use_conversation(key="history")`** → `Conversation` with `.messages`,
  `.append(role, content, **meta)`, `.extend(messages)` (one entry: record a
  question/answer pair atomically), `.clear()`, `.as_chat_messages()`,
  `.as_dspy_history(input_field="question", output_field="answer")` (lazy
  dspy import; user→assistant pairs).
- **`use_durable()`** → the log (or `None`), e.g. for a `/fork <label>`
  command: `use_durable().checkout(label)`.
- **`use_durable_lm(lm, mode=...)`** → `durable_lm(lm, <provider log>)`,
  memoized.

## Durable execution at the LM boundary (`pyreact_durable.lm`)

`durable_lm(lm, log, mode="replay_or_record")` wraps any `dspy.BaseLM` and is
itself a `dspy.BaseLM` (use it in `DSPyProvider(lm=...)` or `dspy.context`).

- Each call is keyed by `sha256(model, prompt/messages, relevant kwargs)` —
  transport kwargs (`api_key`, `num_retries`, `timeout`, …) are ignored — plus
  its **occurrence**: the wrapper counts calls per key in this run.
- `record`: call the model, append an `llm` entry.
  `replay`: return the occurrence-th journaled response with that key on the
  root→leaf path; never call the model (`ReplayMissError` on a miss).
  `replay_or_record` (default): replay if journaled, else record.
  `passthrough`: no journaling.
- If the branch changes under the wrapper (the leaf it last saw is no longer
  on the path, e.g. after a checkout), the occurrence counters reset.
- `dlm.stats` = `{"replayed", "recorded", "inner_calls"}`; `dlm.reset()`.

The scenario this buys (verified by `scripts/e2e_durable.py`): the process
dies after the model answered and the answer was journaled, but before the
agent's state update was written. On restart the agent's state is as before
the turn, the turn is re-run, the identical request hashes to the same key,
and the answer is replayed — **0 model calls, identical answer**, and the
turn completes exactly once.

## CLI

```bash
python -m pyreact_durable show chat.jsonl            # tree, labels, ⑂ branch points, * leaf
python -m pyreact_durable state chat.jsonl --at base # folded state as JSON
python -m pyreact_durable events chat.jsonl --kind turn.* --kind message
python -m pyreact_durable label chat.jsonl before-refactor [--at 1a58]
python -m pyreact_durable checkout chat.jsonl before-refactor
python -m pyreact_durable fork chat.jsonl experiment.jsonl --at base
python -m pyreact_durable checkpoint chat.jsonl --summary "after onboarding"
```

Read commands open the file read-only (safe while the app runs); write
commands take the writer lock, so they fail cleanly while an app holds it.
`show` prints `pyreact_durable.render_tree(log)` (also `log.render()`), so an
app can show the same tree (e.g. `say(log.render())` for a `/tree` command):

```
session 4f8185e0-…  v1  52 entries  2 branch point(s)  leaf=50072721
· 9a1c0b7e event      turn.start Please remember this: my favorite color is…
· 7e667548 llm        claude-cli/haiku 54dde7d9#0 → [[ ## answer ## ]] Got it, …
· 45ef500d state      …/DurableChat[chat]#history = [{"role": "user", …
  2ac49877 event      turn.start /label base  [base]  ⑂2
  ├─ 5d1ae1d4 event      turn.end {"index": 3, …}
  │  …
* └─ 50072721 event      turn.end {"index": 1, …}  <- leaf
```

## Design: why at the border of the flow

- **Zero coupling.** pyreact stays a pure in-memory runtime; durability is an
  opt-in layer composed *around* it. `tests/test_durable_lm.py` asserts that
  no module under `pyreact/` mentions or imports `pyreact_durable`. Removing
  the package changes nothing for apps that don't use it.
- **Only public seams.** State goes through `use_state`/`use_ref`/`use_memo`/
  `use_effect`; addressing through `hooks.path`; sharing through
  `hooks.provide`/`use_context`; observation through the session `EventBus`;
  LM durability through the `dspy.BaseLM` interface. These are the same seams
  any third-party extension would use, so the library also documents (and
  tests) that they are sufficient.
- **The log is the source of truth, components are caches.** Hooks read their
  value from the leaf and write entries; branch switches are a leaf move plus
  a rerender. Nothing is serialized from component internals (closures, LM
  clients, DSPy modules), so the file stays small, inspectable and portable.
- **Append-only tree, like pi.** Branching, rewinding and labels cost one
  line; history is never rewritten (except a one-time format migration); a
  crash can only tear the last line. The same file serves memory (state),
  audit (events), replay (llm) and debugging (CLI).
- **Replay at the LM boundary, Temporal-style.** LM calls are the expensive,
  non-deterministic side effect; journaling them makes re-execution
  deterministic and cheap without the runtime knowing.

## Known gaps

- Persistent addresses are component paths: renaming a component, changing a
  `key`, reordering unkeyed siblings or moving it under another non-provider
  component orphans its state (it stays in the file; write a migration or
  move it with `set_state`). Adding/removing providers does not. Two
  components with the same name and key under the same visible parent (e.g.
  in two sibling providers) share an address: keep keys unique there.
- Values must be JSON. Each set writes the whole value (fine for transcripts
  of normal size; checkpoints bound reload cost, not file size). No
  compaction that drops old lines yet.
- Labels taken mid-turn (e.g. from a `/label` command) mark the position at
  that moment; the rest of that turn's journaled events follow it, so later
  checkouts of that label create an extra (harmless) branch point.
- `durable_lm`: a crash between the model answering and the `llm` entry being
  written still re-calls the model (at-least-once in that window;
  `fsync=True` narrows it). Identical requests at the same occurrence replay
  the same answer by design; pass `mode="record"` or vary `rollout_id` for
  fresh samples. Streaming responses are not journaled chunk by chunk.
  Responses are stored in a simplified chat-completions shape (`content`,
  `tool_calls`, `reasoning_content`, `usage`); `model_type="responses"` is
  stored raw and rebuilt as attribute objects.
- Single writer per file: two processes cannot append to the same session at
  once (use `fork_file` or one file per session).
- `flock` is POSIX; on platforms without `fcntl` the writer guard is a no-op.
