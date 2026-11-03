"""Append-only, tree-structured JSONL session log (pure storage).

Inspired by pi's session format (https://pi.dev/docs/latest/session). This
module has no dependency on pyreact or dspy: it is a plain file format plus a
small API, usable from any process (the app, a CLI, a notebook, a test).

File layout::

    {"type":"session","version":1,"id":"<uuid>","timestamp":"<iso>", ...}
    {"type":"state","id":"a1b2c3d4","parentId":null,"timestamp":"...","path":"App/Chat[chat]","key":"history","value":[]}
    {"type":"event","id":"0f9e8d7c","parentId":"a1b2c3d4","timestamp":"...","kind":"turn.start",...}
    {"type":"label","id":"...","parentId":"0f9e8d7c","timestamp":"...","targetId":"0f9e8d7c","label":"before-fork"}

- Line 1 is the header. Every other line is an entry with an 8-hex ``id`` and a
  ``parentId`` (``null`` for a root). Entries form a **tree**: branching is
  appending a child of an earlier entry, in the same file.
- **Content** entries (``state``, ``event``, ``checkpoint``, ``llm``,
  ``custom``) are children of the current leaf and become the new leaf.
- **Meta** entries (``label``, ``session_info``, ``leaf``) record their
  ``parentId`` (the leaf when written) but never become the leaf. A ``leaf``
  entry moves the current position (``targetId``) without new content, so
  branch switches survive restarts.
- The leaf on load is decided by the last content or ``leaf`` entry in file
  order.
- State at an entry = fold of ``state`` entries on the root→entry path,
  starting from the nearest ``checkpoint`` (a full snapshot).

Durability: one ``os.write`` per entry on an ``O_APPEND`` descriptor (optional
``fsync``); an exclusive ``flock`` guards against two writers; a torn/garbled
trailing line (crash mid-write) is dropped and reported, never fatal.

Backends: ``SessionLog.open(path)`` picks JSONL or SQLite from the suffix
(``.sqlite`` / ``.sqlite3`` / ``.db`` -> :class:`SQLiteSessionLog`, which stores
the same lines one row each). Everything else is shared.
"""

from __future__ import annotations

import copy
import json
import os
import secrets
import threading
import uuid
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

try:  # POSIX advisory locks
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "FORMAT_VERSION",
    "CONTENT_TYPES",
    "META_TYPES",
    "MIGRATIONS",
    "SessionLog",
    "SessionLogError",
    "SessionLocked",
    "register_migration",
    "kind_matches",
]

FORMAT_VERSION = 1

CONTENT_TYPES = frozenset({"state", "event", "checkpoint", "llm", "custom"})
META_TYPES = frozenset({"label", "session_info", "leaf"})
ENTRY_TYPES = CONTENT_TYPES | META_TYPES

_RESERVED = frozenset({"type", "id", "parentId", "timestamp"})
_HEADER_RESERVED = frozenset({"type", "version", "id", "timestamp"})

PathLike = Union[str, "os.PathLike[str]"]
Migration = Callable[[dict, List[dict]], Tuple[dict, List[dict]]]

# from_version -> fn(header, entries) -> (header, entries) at from_version + 1
MIGRATIONS: Dict[int, Migration] = {}


def register_migration(from_version: int) -> Callable[[Migration], Migration]:
    """Register ``fn(header, entries) -> (header, entries)`` upgrading a file
    from ``from_version`` to ``from_version + 1`` (applied on load)."""

    def deco(fn: Migration) -> Migration:
        MIGRATIONS[int(from_version)] = fn
        return fn

    return deco


class SessionLogError(Exception):
    """Invalid file, invalid entry or misuse of a :class:`SessionLog`."""


class SessionLocked(SessionLogError):
    """Another writer holds the log (single-writer guard)."""


def now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def kind_matches(kind: str, patterns: Optional[Iterable[str]]) -> bool:
    """``"turn.*"`` matches ``turn.start``; ``"*"`` matches everything."""
    if patterns is None:
        return True
    for p in patterns:
        if p == "*" or p == kind or (p.endswith(".*") and kind.startswith(p[:-1])):
            return True
    return False


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class SessionLog:
    """An open session file. Construct with :meth:`open` or :meth:`create`.

    Thread-safe (one internal lock); one writer process at a time.
    """

    def __init__(self, path: Path, *, fsync: bool = False, readonly: bool = False,
                 auto_checkpoint: Optional[int] = None) -> None:
        # Use SessionLog.open / SessionLog.create.
        self.path = Path(path)
        self.fsync = fsync
        self.readonly = readonly
        self.auto_checkpoint = auto_checkpoint
        self.header: dict = {}
        self.entries: List[dict] = []
        self.by_id: Dict[str, dict] = {}
        self.children: Dict[Optional[str], List[str]] = {}
        self.recovered: List[dict] = []  # garbled lines dropped on load
        self.migrated_from: Optional[int] = None
        self._leaf: Optional[str] = None
        self._labels: Dict[str, Optional[str]] = {}
        self._name: Optional[str] = None
        self._ckpt_dist: Dict[str, int] = {}
        self._state_cache: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.RLock()
        self._fd: Optional[int] = None
        self._subs: List[Callable[[dict, bool], None]] = []

    # ------------------------------------------------------------------ open
    @classmethod
    def open(
        cls,
        path: PathLike,
        *,
        create: bool = True,
        fsync: bool = False,
        readonly: bool = False,
        meta: Optional[dict] = None,
        lock: bool = True,
        auto_checkpoint: Optional[int] = None,
        migrations: Optional[Dict[int, Migration]] = None,
    ) -> "SessionLog":
        """Open (and by default create) a session file.

        - ``readonly=True``: no lock, no writes, no tail repair (safe while an
          app is writing the file, e.g. from the CLI).
        - ``lock=True`` (writable): take an exclusive ``flock``; raises
          :class:`SessionLocked` if another writer holds it.
        - ``auto_checkpoint=N``: append a checkpoint whenever the leaf is N
          content entries past the last one.
        - ``meta``: extra header fields when the file is created.
        """
        p = Path(path)
        cls = _backend_for(cls, p)
        if cls._is_empty(p):
            if readonly or not create:
                raise FileNotFoundError(str(p))
            return cls.create(p, meta=meta, fsync=fsync, lock=lock,
                              auto_checkpoint=auto_checkpoint)
        log = cls(p, fsync=fsync, readonly=readonly, auto_checkpoint=auto_checkpoint)
        if not readonly:
            log._open_fd(lock)
        try:
            log._load(MIGRATIONS if migrations is None else migrations)
        except BaseException:
            log.close()
            raise
        return log

    @classmethod
    def create(
        cls,
        path: PathLike,
        *,
        meta: Optional[dict] = None,
        fsync: bool = False,
        lock: bool = True,
        parent_session: Optional[str] = None,
        auto_checkpoint: Optional[int] = None,
    ) -> "SessionLog":
        """Create a new session file (fails if a non-empty one exists).

        The suffix picks the backend: ``.sqlite`` / ``.sqlite3`` / ``.db`` ->
        :class:`SQLiteSessionLog`, anything else -> JSONL."""
        p = Path(path)
        cls = _backend_for(cls, p)
        if not cls._is_empty(p):
            raise SessionLogError(f"{p} already exists")
        p.parent.mkdir(parents=True, exist_ok=True)
        log = cls(p, fsync=fsync, auto_checkpoint=auto_checkpoint)
        log._open_fd(lock)
        header = {"type": "session", "version": FORMAT_VERSION, "id": str(uuid.uuid4()),
                  "timestamp": now_iso()}
        if parent_session is not None:
            header["parentSession"] = parent_session
        for k, v in (meta or {}).items():
            if k in _HEADER_RESERVED:
                raise SessionLogError(f"meta key {k!r} is reserved")
            header[k] = v
        try:
            if not log._storage_empty():  # lost a creation race
                raise SessionLogError(f"{p} already exists")
            log._write_line(_dumps(header))
        except BaseException:
            log.close()
            raise
        log.header = header
        return log

    # -------------------------------------------------- storage hooks (JSONL)
    @classmethod
    def _is_empty(cls, p: Path) -> bool:
        return not p.exists() or p.stat().st_size == 0

    def _storage_empty(self) -> bool:
        return os.fstat(self._fd).st_size == 0

    def _read_lines(self) -> Tuple[bytes, List[bytes]]:
        raw = self.path.read_bytes()
        return raw, raw.split(b"\n")

    def _reset_storage(self) -> None:
        os.ftruncate(self._fd, 0)

    def _open_fd(self, lock: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        if lock and fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise SessionLocked(
                    f"{self.path} is locked by another writer (single-writer guard); "
                    "open it with readonly=True to inspect it"
                ) from None
        self._fd = fd

    def close(self) -> None:
        """Release the lock and the file descriptor (idempotent)."""
        with self._lock:
            fd, self._fd = self._fd, None
            if fd is not None:
                try:
                    os.close(fd)  # also releases the flock
                except OSError:
                    pass

    @property
    def closed(self) -> bool:
        return self._fd is None

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<SessionLog {str(self.path)!r} entries={len(self.entries)} leaf={self._leaf}>"

    # ------------------------------------------------------------------ load
    def _load(self, migrations: Dict[int, Migration]) -> None:
        raw, lines = self._read_lines()
        # ``lines[-1]`` is b"" iff the file ends with a newline.
        torn_tail = lines[-1] != b""
        parsed: List[Tuple[int, dict]] = []
        bad: List[Tuple[int, bytes]] = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict) or "type" not in obj:
                    raise ValueError("not an entry")
            except ValueError:
                bad.append((i, line))
                continue
            parsed.append((i, obj))

        last_line = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
        tail_bad = [b for b in bad if b[0] == last_line]
        for i, line in bad:
            info = {"line": i + 1, "bytes": len(line),
                    "preview": line[:80].decode("utf-8", "replace"),
                    "trailing": i == last_line}
            self.recovered.append(info)
        if self.recovered:
            warnings.warn(
                f"{self.path}: ignored {len(self.recovered)} garbled line(s): "
                + ", ".join(f"line {r['line']}{' (trailing)' if r['trailing'] else ''}"
                            for r in self.recovered),
                RuntimeWarning,
                stacklevel=3,
            )

        if not parsed or parsed[0][1].get("type") != "session":
            if not parsed and not self.readonly:
                # Only garbage (crash while writing the header): start over.
                self._reset_storage()
                header = {"type": "session", "version": FORMAT_VERSION,
                          "id": str(uuid.uuid4()), "timestamp": now_iso()}
                self._write_line(_dumps(header))
                self.header = header
                return
            raise SessionLogError(f"{self.path}: missing session header")

        header = parsed[0][1]
        entries = [obj for _, obj in parsed[1:]]
        version = int(header.get("version", 0))
        if version > FORMAT_VERSION:
            raise SessionLogError(
                f"{self.path}: format version {version} is newer than supported "
                f"({FORMAT_VERSION})"
            )
        if version < FORMAT_VERSION:
            self.migrated_from = version
            while version < FORMAT_VERSION:
                fn = migrations.get(version)
                if fn is None:
                    raise SessionLogError(f"no migration from format version {version}")
                header, entries = fn(dict(header), [dict(e) for e in entries])
                version += 1
                header["version"] = version

        self.header = header
        for e in entries:
            self._index(e)

        if self.readonly:
            return
        if self.migrated_from is not None:
            self._rewrite()  # persist the upgraded format atomically
        elif tail_bad or torn_tail:
            self._repair_tail(raw, tail_bad)

    def _repair_tail(self, raw: bytes, tail_bad) -> None:
        """Make the file end right after the last good line, newline-terminated."""
        if tail_bad:
            cut = raw.rstrip(b"\n").rfind(b"\n") + 1  # start of the bad last line
            os.ftruncate(self._fd, cut)
        else:  # good JSON missing only its "\n"
            os.write(self._fd, b"\n")

    def _rewrite(self) -> None:
        tmp = self.path.with_name(self.path.name + ".migrating")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_dumps(self.header) + "\n")
            for e in self.entries:
                f.write(_dumps(e) + "\n")
            f.flush()
            os.fsync(f.fileno())
        # Keep holding the lock: swap the descriptor to the new inode.
        old = self._fd
        os.replace(tmp, self.path)
        self._fd = None
        self._open_fd(lock=True)
        if old is not None:
            os.close(old)

    def _index(self, e: dict) -> None:
        eid = e.get("id")
        if not isinstance(eid, str):
            return
        self.entries.append(e)
        self.by_id[eid] = e
        parent = e.get("parentId")
        etype = e.get("type")
        if etype in CONTENT_TYPES:
            self.children.setdefault(parent, []).append(eid)
            if etype == "checkpoint":
                self._ckpt_dist[eid] = 0
            else:
                self._ckpt_dist[eid] = (self._ckpt_dist.get(parent, 0) + 1) if parent else 1
            self._leaf = eid
        elif etype == "leaf":
            target = e.get("targetId")
            if target is None or target in self.by_id:
                self._leaf = target
        elif etype == "label":
            name = e.get("label")
            if isinstance(name, str):
                self._labels[name] = e.get("targetId")
                if e.get("targetId") is None:
                    self._labels.pop(name, None)
        elif etype == "session_info":
            self._name = e.get("name", self._name)

    # ----------------------------------------------------------------- write
    def _write_line(self, line: str) -> None:
        if self._fd is None:
            raise SessionLogError("log is closed" if not self.readonly else "log is read-only")
        data = (line + "\n").encode("utf-8")
        view = memoryview(data)
        while view:
            n = os.write(self._fd, view)  # O_APPEND: positioned atomically
            view = view[n:]
        if self.fsync:
            os.fsync(self._fd)

    def _new_id(self) -> str:
        while True:
            eid = secrets.token_hex(4)
            if eid not in self.by_id:
                return eid

    _LEAF = object()

    def append(self, type: str, *, parent_id: Any = _LEAF, **fields: Any) -> dict:
        """Append an entry and return it.

        Content entries become children of ``parent_id`` (default: the
        current leaf) and the new leaf; pass an earlier id to branch. Meta
        entries never move the leaf (except ``leaf`` entries, via
        ``targetId``). ``fields`` must be JSON-serializable.
        """
        if type not in ENTRY_TYPES:
            raise SessionLogError(f"unknown entry type {type!r}")
        bad = _RESERVED.intersection(fields)
        if bad:
            raise SessionLogError(f"reserved field(s) {sorted(bad)}")
        with self._lock:
            prev_leaf = self._leaf
            parent = prev_leaf if parent_id is SessionLog._LEAF else parent_id
            if parent is not None and parent not in self.by_id:
                raise SessionLogError(f"unknown parent entry {parent!r}")
            if type == "leaf":
                target = fields.get("targetId")
                if target is not None and target not in self.by_id:
                    raise SessionLogError(f"unknown entry {target!r}")
            entry = {"type": type, "id": self._new_id(), "parentId": parent,
                     "timestamp": now_iso(), **fields}
            try:
                line = _dumps(entry)
            except (TypeError, ValueError) as exc:
                raise SessionLogError(f"{type} entry is not JSON-serializable: {exc}") from exc
            self._write_line(line)
            self._index(entry)
            moved = (type == "leaf" and self._leaf != prev_leaf) or (
                type in CONTENT_TYPES and parent != prev_leaf
            )
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(entry, moved)
            except Exception as exc:  # pragma: no cover - listener bug
                warnings.warn(f"SessionLog listener failed: {exc!r}", RuntimeWarning)
        if (type in CONTENT_TYPES and type != "checkpoint" and self.auto_checkpoint
                and self._ckpt_dist.get(entry["id"], 0) >= self.auto_checkpoint):
            self.checkpoint()
        return entry

    def subscribe(self, fn: Callable[[dict, bool], None]) -> Callable[[], None]:
        """Call ``fn(entry, moved)`` after every append. ``moved`` is True when
        the leaf jumped (checkout or branching append) rather than advanced."""
        with self._lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)

        return unsubscribe

    # ------------------------------------------------------------------ tree
    @property
    def id(self) -> str:
        return self.header.get("id", "")

    @property
    def version(self) -> int:
        return int(self.header.get("version", 0))

    @property
    def leaf(self) -> Optional[str]:
        """Id of the current position (``None`` for an empty log)."""
        return self._leaf

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def labels(self) -> Dict[str, Optional[str]]:
        """``{label: entry_id}`` (latest label entry per name wins)."""
        return dict(self._labels)

    def labels_of(self, entry_id: str) -> List[str]:
        return [n for n, t in self._labels.items() if t == entry_id]

    def set_name(self, name: str) -> dict:
        return self.append("session_info", name=name)

    def resolve(self, ref: Optional[str]) -> Optional[str]:
        """Entry id for a label, an id, or a unique id prefix (≥4 chars).
        ``None``/``"@leaf"`` → the leaf; ``"@root"`` → ``None`` (before
        everything)."""
        if ref is None or ref == "@leaf":
            return self._leaf
        if ref == "@root":
            return None
        if ref in self._labels:
            return self._labels[ref]
        if ref in self.by_id:
            return ref
        if len(ref) >= 4:
            hits = [i for i in self.by_id if i.startswith(ref)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise SessionLogError(f"ambiguous id prefix {ref!r}: {hits[:5]}")
        raise SessionLogError(f"no label or entry {ref!r}")

    def iter_up(self, entry_id: Optional[str] = None) -> Iterator[dict]:
        """Entries from ``entry_id`` (default: leaf) up to the root."""
        eid = self._leaf if entry_id is None else self.resolve(entry_id)
        seen = set()
        while eid is not None:
            if eid in seen:  # corrupt cycle: stop rather than spin
                raise SessionLogError(f"cycle at entry {eid}")
            seen.add(eid)
            e = self.by_id[eid]
            yield e
            eid = e.get("parentId")

    def path_to(self, entry_id: Optional[str] = None) -> List[dict]:
        """Root→entry list of entries (default: root→leaf)."""
        out = list(self.iter_up(entry_id))
        out.reverse()
        return out

    def label(self, name: str, entry_id: Optional[str] = None) -> dict:
        """Bookmark ``entry_id`` (default: the leaf) as ``name`` (moves an
        existing label)."""
        target = self.resolve(entry_id) if entry_id is not None else self._leaf
        if target is None:
            raise SessionLogError("nothing to label: the log is empty")
        return self.append("label", targetId=target, label=name)

    def unlabel(self, name: str) -> dict:
        return self.append("label", targetId=None, label=name)

    def checkout(self, ref: Optional[str]) -> Optional[str]:
        """Move the leaf to a label/id (writes a ``leaf`` entry); the next
        content entry starts a branch there. Returns the new leaf id."""
        target = self.resolve(ref)
        self.append("leaf", targetId=target)
        return target

    # ----------------------------------------------------------------- state
    def state_at(self, entry_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """``{path: {key: value}}`` at ``entry_id`` (default: leaf): the
        nearest checkpoint's snapshot plus the ``state`` entries after it on
        the path. Returns a fresh copy."""
        target = self._leaf if entry_id is None else self.resolve(entry_id)
        return copy.deepcopy(self._state(target))

    def get_state(self, path: str, key: str, default: Any = None,
                  entry_id: Optional[str] = None) -> Any:
        """One value from :meth:`state_at` (copied), or ``default``."""
        target = self._leaf if entry_id is None else self.resolve(entry_id)
        node = self._state(target).get(path)
        if node is None or key not in node:
            return default
        return copy.deepcopy(node[key])

    def has_state(self, path: str, key: str, entry_id: Optional[str] = None) -> bool:
        target = self._leaf if entry_id is None else self.resolve(entry_id)
        return key in self._state(target).get(path, {})

    def _state(self, target: Optional[str]) -> Dict[str, Dict[str, Any]]:
        if target is None:
            return {}
        with self._lock:
            hit = self._state_cache.get(target)
            if hit is not None:
                self._state_cache.move_to_end(target)
                return hit
            chain: List[dict] = []
            base: Dict[str, Dict[str, Any]] = {}
            for e in self.iter_up(target):
                cached = self._state_cache.get(e["id"])
                if cached is not None:
                    base = cached
                    break
                if e.get("type") == "checkpoint":
                    base = e.get("state") or {}
                    break
                chain.append(e)
            state = {p: dict(kv) for p, kv in base.items()}
            for e in reversed(chain):
                if e.get("type") == "state":
                    node = state.setdefault(e["path"], {})
                    if e.get("deleted"):
                        node.pop(e["key"], None)
                    else:
                        node[e["key"]] = e.get("value")
            self._state_cache[target] = state
            while len(self._state_cache) > 32:
                self._state_cache.popitem(last=False)
            return state

    def set_state(self, path: str, key: str, value: Any) -> dict:
        """Append a ``state`` entry (``value`` must be JSON-serializable)."""
        try:
            _dumps(value)
        except (TypeError, ValueError) as exc:
            raise SessionLogError(
                f"state {path!r} / {key!r} is not JSON-serializable: {exc}"
            ) from exc
        return self.append("state", path=path, key=key, value=value)

    def delete_state(self, path: str, key: str) -> dict:
        return self.append("state", path=path, key=key, deleted=True)

    def checkpoint(self, summary: Optional[str] = None) -> dict:
        """Append a full state snapshot (compaction): later loads of the state
        start here instead of folding the whole history."""
        fields: Dict[str, Any] = {"state": self._state(self._leaf)}
        if summary is not None:
            fields["summary"] = summary
        return self.append("checkpoint", **fields)

    # ---------------------------------------------------------------- events
    def events(self, kinds: Optional[Iterable[str]] = None,
               entry_id: Optional[str] = None) -> List[dict]:
        """Journaled ``event`` entries on the root→entry path, oldest first,
        optionally filtered by kind (exact, ``"prefix.*"`` or ``"*"``)."""
        pats = None if kinds is None else ([kinds] if isinstance(kinds, str) else list(kinds))
        target = self._leaf if entry_id is None else self.resolve(entry_id)
        if target is None:
            return []
        return [e for e in self.path_to(target)
                if e.get("type") == "event" and kind_matches(e.get("kind", ""), pats)]

    # ------------------------------------------------------------------ fork
    def fork_file(self, path: PathLike, at: Optional[str] = None, *,
                  meta: Optional[dict] = None, lock: bool = True) -> "SessionLog":
        """New session file whose history is the root→``at`` path of this one
        (default: the leaf). Its header carries ``parentSession`` (this file's
        path), ``parentSessionId`` and ``forkedFrom``. Entry ids are kept;
        labels pointing into the copied path are copied too. Returns the new,
        open log."""
        target = self._leaf if at is None else self.resolve(at)
        hist = self.path_to(target) if target is not None else []
        extra = {"parentSessionId": self.id, "forkedFrom": target, **(meta or {})}
        new = SessionLog.create(path, meta=extra, parent_session=str(self.path),
                                lock=lock, fsync=self.fsync,
                                auto_checkpoint=self.auto_checkpoint)
        try:
            ids = set()
            for e in hist:
                new._write_line(_dumps(e))
                new._index(e)
                ids.add(e["id"])
            for name, tid in self._labels.items():
                if tid in ids:
                    new.append("label", targetId=tid, label=name)
            if self._name:
                new.set_name(self._name)
        except BaseException:
            new.close()
            raise
        return new

    # ------------------------------------------------------------------ view
    def tree(self) -> List[dict]:
        """Content-entry tree for rendering: a list of root nodes
        ``{"entry", "children", "labels", "is_leaf", "on_path"}`` (built
        iteratively; deep histories are fine)."""
        on_path = {e["id"] for e in self.iter_up()} if self._leaf else set()
        nodes: Dict[str, dict] = {}
        for e in self.entries:
            if e.get("type") in CONTENT_TYPES:
                nodes[e["id"]] = {"entry": e, "children": [],
                                  "labels": self.labels_of(e["id"]),
                                  "is_leaf": e["id"] == self._leaf,
                                  "on_path": e["id"] in on_path}
        roots = []
        for eid, node in nodes.items():
            parent = node["entry"].get("parentId")
            if parent in nodes:
                nodes[parent]["children"].append(node)
            else:
                roots.append(node)
        return roots

    def branch_points(self) -> List[str]:
        return [p for p, kids in self.children.items() if p is not None and len(kids) > 1]

    def stats(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for e in self.entries:
            counts[e.get("type", "?")] = counts.get(e.get("type", "?"), 0) + 1
        return {"entries": len(self.entries), "types": counts,
                "branch_points": len(self.branch_points()),
                "labels": len(self._labels), "leaf": self._leaf}

    def render(self, width: int = 60) -> str:
        """The entry tree as text (what ``python -m pyreact_durable show``
        prints): labels, ``⑂`` branch points, ``*`` leaf."""
        from .render import render_tree

        return "\n".join(render_tree(self, width))


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------
SQLITE_SUFFIXES = (".sqlite", ".sqlite3", ".db")


def _backend_for(cls: type, p: Path) -> type:
    """``SessionLog.open/create`` pick the backend from the file suffix;
    explicit subclasses are respected."""
    if cls is SessionLog and p.suffix.lower() in SQLITE_SUFFIXES:
        return SQLiteSessionLog
    return cls


class SQLiteSessionLog(SessionLog):
    """The same session format, stored in SQLite: one row per JSON line.

    Rows hold exactly the lines a ``.jsonl`` file would (header first), so the
    tree semantics, migrations and every API are shared with the JSONL backend;
    ``fork_file("x.jsonl")`` exports and ``fork_file("x.sqlite")`` imports.

    - each append is one committed INSERT (WAL; ``synchronous=FULL`` with
      ``fsync=True``, else ``NORMAL``), so there are no torn lines to repair
    - single writer: an exclusive ``flock`` on ``<file>.lock``
    - ``readonly=True`` opens ``mode=ro``; WAL lets it read while the app writes
    """

    _SCHEMA = "CREATE TABLE IF NOT EXISTS lines (seq INTEGER PRIMARY KEY AUTOINCREMENT, line TEXT NOT NULL)"

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self._db = None
        self._lock_fd: Optional[int] = None

    # -- storage hooks ------------------------------------------------------
    @classmethod
    def _is_empty(cls, p: Path) -> bool:
        if not p.exists() or p.stat().st_size == 0:
            return True
        import sqlite3

        try:
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                row = con.execute("SELECT count(*) FROM lines").fetchone()
                return not row or row[0] == 0
            finally:
                con.close()
        except sqlite3.Error:
            return True  # no table yet

    def _storage_empty(self) -> bool:
        return self._db.execute("SELECT count(*) FROM lines").fetchone()[0] == 0

    def _connect(self, readonly: bool):
        import sqlite3

        if readonly:
            return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True,
                                   check_same_thread=False)
        con = sqlite3.connect(str(self.path), isolation_level=None,
                              check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(f"PRAGMA synchronous={'FULL' if self.fsync else 'NORMAL'}")
        con.execute(self._SCHEMA)
        return con

    def _open_fd(self, lock: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if lock and fcntl is not None:
            lock_path = str(self.path) + ".lock"
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise SessionLocked(
                    f"{self.path} is locked by another writer (single-writer guard); "
                    "open it with readonly=True to inspect it"
                ) from None
            self._lock_fd = fd
        self._db = self._connect(readonly=False)
        self._fd = -1  # "open" marker for the shared code paths

    def close(self) -> None:
        with self._lock:
            db, self._db = self._db, None
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass
            fd, self._lock_fd = self._lock_fd, None
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fd = None

    def _read_lines(self) -> Tuple[bytes, List[bytes]]:
        con = self._db if self._db is not None else self._connect(readonly=True)
        try:
            rows = con.execute("SELECT line FROM lines ORDER BY seq").fetchall()
        finally:
            if con is not self._db:
                con.close()
        lines = [r[0].encode("utf-8") for r in rows] + [b""]  # "ends with newline"
        return b"", lines

    def _reset_storage(self) -> None:
        self._db.execute("DELETE FROM lines")

    def _repair_tail(self, raw: bytes, tail_bad) -> None:
        # Rows are committed atomically, so only a hand-edited row can be bad.
        if tail_bad:
            self._db.execute("DELETE FROM lines WHERE seq = (SELECT max(seq) FROM lines)")

    def _write_line(self, line: str) -> None:
        if self._db is None:
            raise SessionLogError("log is closed" if not self.readonly else "log is read-only")
        self._db.execute("INSERT INTO lines(line) VALUES (?)", (line,))

    def _rewrite(self) -> None:
        db = self._db
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("DELETE FROM lines")
            db.executemany(
                "INSERT INTO lines(line) VALUES (?)",
                [(_dumps(self.header),)] + [(_dumps(e),) for e in self.entries],
            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
