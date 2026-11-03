"""Stateful sessions: checkpoint participants, manifests and their stores.

An *agent* is a session's whole state: durable component state, the
conversation, its sandboxes, pending turns, learned modules... Each piece is
owned by a library at the border (durable memory, sandboxes, ...), so the
core only defines how they take part in one explicit operation:

- :meth:`Session.suspend` asks every **participant** registered on the
  session (``session.participants.add(p)``) for a JSON-able *fragment*,
  writes a **manifest** to a :class:`ManifestStore` and unmounts the tree;
- :meth:`AppRunner.resume` re-opens the session with the same id, hands each
  participant its fragment **before the first render**, mounts the tree and
  re-runs the turns that were cut short.

Participant protocol (duck-typed)::

    class MyParticipant:
        name = "mine"                 # key in manifest["participants"]
        version = "1"                 # optional, recorded in manifest["versions"]

        async def suspend(self, session) -> dict: ...         # JSON-able fragment
        async def resume(self, session, fragment) -> None: ... # before first render
        async def suspended(self, session) -> None: ...        # optional: after unmount
        async def resumed(self, session) -> None: ...          # optional: after first render

Participants are registered when a session opens (before it mounts), usually
from a factory given to the runner::

    app = AppRunner(App, participants=[MyParticipant], manifest_store=store)
    # == app.on_session(open=lambda s: s.participants.add(MyParticipant(s)))

Manifest (``MANIFEST_FORMAT``)::

    {"format": "pyreact.manifest/1", "session_id": "agent-1",
     "created_at": 1793680000.0, "suspended_at": 1793680123.4,
     "turn_seq": 3, "turn_policy": "queue", "app": "examples.x:App", "reason": "manual",
     "participants": {"turns": {...}, "durable": {...}, "sandbox": {...}},
     "versions": {"manifest": 1, "pyreact": "0.1.0", "python": "3.13.1",
                  "participants": {"durable": "1", ...}}}

Nothing here imports a persistence library; they plug in through this
protocol (libraries at the border).
"""

from __future__ import annotations

import abc
import copy
import json
import os
import platform
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union
from urllib.parse import quote

if TYPE_CHECKING:  # pragma: no cover
    from .session import Session

__all__ = [
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "ManifestError",
    "ManifestStore",
    "MemoryManifestStore",
    "DirectoryManifestStore",
    "Participants",
    "SuspendError",
    "TurnsParticipant",
    "check_manifest",
    "load_manifest",
]

MANIFEST_VERSION = 1
MANIFEST_FORMAT = f"pyreact.manifest/{MANIFEST_VERSION}"


class SuspendError(RuntimeError):
    """A suspend could not complete; the session is left open and running."""


class ManifestError(ValueError):
    """Not a manifest this version of pyreact can resume."""


def _pyreact_version() -> str:
    try:
        from importlib.metadata import version

        return version("pyreact")
    except Exception:  # pragma: no cover - not installed as a dist
        return "unknown"


def versions(participants: "Participants") -> Dict[str, Any]:
    return {
        "manifest": MANIFEST_VERSION,
        "pyreact": _pyreact_version(),
        "python": platform.python_version(),
        "participants": {p.name: getattr(p, "version", None) for p in participants},
    }


def check_manifest(manifest: Any) -> Dict[str, Any]:
    """Validate the envelope of a manifest; returns it. Raises :class:`ManifestError`."""
    if not isinstance(manifest, dict):
        raise ManifestError(f"a manifest is a dict, got {type(manifest).__name__}")
    fmt = manifest.get("format")
    if not isinstance(fmt, str) or not fmt.startswith("pyreact.manifest/"):
        raise ManifestError(f"not a pyreact manifest (format={fmt!r})")
    try:
        ver = int(fmt.rsplit("/", 1)[1])
    except ValueError:
        raise ManifestError(f"bad manifest format {fmt!r}") from None
    if ver > MANIFEST_VERSION:
        raise ManifestError(f"manifest format {fmt!r} is newer than this pyreact "
                            f"({MANIFEST_FORMAT})")
    if not manifest.get("session_id"):
        raise ManifestError("manifest has no session_id")
    if not isinstance(manifest.get("participants", {}), dict):
        raise ManifestError("manifest['participants'] must be a dict")
    return manifest


def load_manifest(path: Union[str, "os.PathLike[str]"]) -> Dict[str, Any]:
    """Read and validate a manifest JSON file."""
    with open(path, encoding="utf-8") as f:
        return check_manifest(json.load(f))


def _jsonable(fragment: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(fragment, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SuspendError(f"participant {name!r} returned a fragment that is not JSON "
                           f"({type(exc).__name__}: {exc})") from None


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------
class Participants:
    """The checkpoint participants of one session (``session.participants``).

    Ordered by registration: ``suspend`` and ``resume`` are called in that
    order. Names are unique (they key the manifest's ``participants``).
    """

    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def add(self, participant: Any, *, replace: bool = False) -> Any:
        """Register ``participant`` (see the module docs); returns it."""
        name = getattr(participant, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"participant {participant!r} needs a non-empty str .name")
        for meth in ("suspend", "resume"):
            if not callable(getattr(participant, meth, None)):
                raise TypeError(f"participant {name!r} has no {meth}() method")
        with self._lock:
            if name in self._items and not replace and self._items[name] is not participant:
                raise ValueError(f"a participant named {name!r} is already registered")
            self._items[name] = participant
        return participant

    def remove(self, participant_or_name: Any) -> bool:
        name = (participant_or_name if isinstance(participant_or_name, str)
                else getattr(participant_or_name, "name", None))
        with self._lock:
            return self._items.pop(name, None) is not None

    def get(self, name: str) -> Optional[Any]:
        return self._items.get(name)

    def names(self) -> List[str]:
        return list(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[Any]:
        with self._lock:
            return iter(list(self._items.values()))

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<Participants {self.names()}>"


class TurnsParticipant:
    """Built-in participant (always registered, name ``"turns"``).

    Fragment: ``{"next_index": n, "carry": [turn, ...]}`` where ``carry`` are
    the turns the suspend cut short (the active turn when it was interrupted)
    followed by the queued ones that never started. On resume, numbering
    continues at ``next_index`` and the carried turns are queued again (as
    ``resumed`` turns, same ids) right after the first render. Turns a crash
    left unfinished are the ``TurnStore``'s job (``resume_pending``).
    """

    name = "turns"
    version = "1"

    def __init__(self) -> None:
        self._carry: List[Any] = []

    async def suspend(self, session: "Session") -> Dict[str, Any]:
        carry = [*session._suspend_carry, *session.turns.pending()]
        seen, out = set(), []
        for t in carry:
            if t.id not in seen:
                seen.add(t.id)
                out.append({k: v for k, v in asdict(t).items() if k != "resumed"})
        return {"next_index": session.turns.next_index(), "carry": out}

    async def resume(self, session: "Session", fragment: Dict[str, Any]) -> None:
        from pyreact.core.turns import Turn

        nxt = int(fragment.get("next_index") or 1)
        session.turns.advance_to(nxt)
        self._carry = [Turn(**{k: v for k, v in t.items() if k in Turn.__dataclass_fields__})
                       for t in fragment.get("carry") or []]

    async def resumed(self, session: "Session") -> None:
        carry, self._carry = self._carry, []
        if not session._resume_turns:
            return
        for t in carry:
            turn = replace(t, resumed=True, ts=time.time())
            session.turns.advance_to(turn.index + 1)
            session.turns.enqueue(turn)
            session._resumed_turns.append(turn.id)


# ---------------------------------------------------------------------------
# Manifest stores
# ---------------------------------------------------------------------------
class ManifestStore(abc.ABC):
    """Where suspended sessions' manifests live, keyed by session id.

    ``save`` returns a reference (a path, a key...) shown in the
    ``session.suspended`` event. Anything that can persist JSON works.
    """

    @abc.abstractmethod
    def save(self, manifest: Dict[str, Any]) -> str: ...

    @abc.abstractmethod
    def load(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    @abc.abstractmethod
    def delete(self, session_id: str) -> None: ...

    @abc.abstractmethod
    def list(self) -> List[str]: ...

    def __contains__(self, session_id: object) -> bool:
        return isinstance(session_id, str) and self.load(session_id) is not None


class MemoryManifestStore(ManifestStore):
    """In-process (the default): suspend/resume within one process."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def save(self, manifest):
        with self._lock:
            self._data[manifest["session_id"]] = copy.deepcopy(manifest)
        return f"memory:{manifest['session_id']}"

    def load(self, session_id):
        with self._lock:
            m = self._data.get(session_id)
            return copy.deepcopy(m) if m is not None else None

    def delete(self, session_id):
        with self._lock:
            self._data.pop(session_id, None)

    def list(self):
        with self._lock:
            return list(self._data)


class DirectoryManifestStore(ManifestStore):
    """One ``<dir>/<session id>.json`` per session (id percent-encoded),
    written atomically. Survives restarts; readable by any process."""

    def __init__(self, directory: Union[str, "os.PathLike[str]"]) -> None:
        self.dir = Path(directory)

    def path_for(self, session_id: str) -> Path:
        return self.dir / f"{quote(str(session_id), safe='')}.json"

    def save(self, manifest):
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(manifest["session_id"])
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
        os.replace(tmp, path)
        return str(path)

    def load(self, session_id):
        path = self.path_for(session_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def delete(self, session_id):
        try:
            self.path_for(session_id).unlink()
        except FileNotFoundError:
            pass

    def list(self):
        from urllib.parse import unquote

        if not self.dir.is_dir():
            return []
        return sorted(unquote(p.stem) for p in self.dir.glob("*.json"))

    def __repr__(self) -> str:
        return f"<DirectoryManifestStore {self.dir}>"
