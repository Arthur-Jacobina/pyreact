"""The border between pyreact and the session log.

Everything here is built on pyreact's *public* surface only:
``hooks.use_state / use_ref / use_memo / use_effect / use_callback /
use_context``,
``hooks.provide``, ``hooks.path`` / ``hooks.full_path``, ``hooks.get_service``,
``use_dehydrate`` / ``use_cleanup_reason`` and the session's ``EventBus``.
Nothing in ``pyreact`` knows this module exists.

    @component
    def App():
        return [Durable(log="chat.jsonl", children=[Chat(key="chat")])]

    @component
    def Chat():
        count, set_count = use_persistent_state(0, key="count")
        conv = use_conversation()            # persisted message list
        log = use_durable()                  # labels / checkout / fork
        ...
"""

from __future__ import annotations

import json
import threading
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from pyreact.core import (
    TurnManager,
    component,
    create_context,
    hooks,
    is_updater,
    use_cleanup_reason,
    use_dehydrate,
    use_event,
)

from .log import SessionLog, archive_log, delete_log, kind_matches, safe_session_id

__all__ = [
    "DEFAULT_JOURNAL",
    "Durable",
    "DurableHandle",
    "DurableStateError",
    "Conversation",
    "use_durable",
    "use_durable_lm",
    "use_persistent_state",
    "use_persistent_reducer",
    "use_conversation",
]

DEFAULT_JOURNAL: Tuple[str, ...] = (
    "input.*", "message", "turn.*", "llm.*", "tool.*", "feedback", "error",
)

_DurableCtx = create_context(default=None, name="Durable")
_MISSING = object()
_UNSET = object()
_WARNED: set = set()


class DurableStateError(TypeError):
    """A persistent hook was given a value that cannot be stored as JSON."""


class DurableHandle:
    """Value provided by :func:`Durable`: the log plus a generation counter
    bumped whenever the leaf jumps (checkout / branch), so persistent hooks
    reload their values from the new position."""

    __slots__ = ("log", "generation")

    def __init__(self, log: SessionLog, generation: int) -> None:
        self.log = log
        self.generation = generation

    def __repr__(self) -> str:
        return f"<DurableHandle {self.log!r} gen={self.generation}>"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
ON_CLOSE = ("keep", "archive", "delete")


def _session() -> Any:
    return hooks.get_service("session", lambda: None)


def _session_id() -> str:
    return str(getattr(_session(), "id", None) or "default")


def _open(log: Any, log_factory: Optional[Callable[[str], Any]], session_id: str):
    """-> (SessionLog, owned). Logs we open from a path/factory are ours to close.
    The factory gets the session id escaped for use in a file name."""
    src = log_factory(safe_session_id(session_id)) if log_factory is not None else log
    if src is None:
        raise ValueError("Durable needs log=SessionLog|path or log_factory=fn(session_id)")
    if isinstance(src, SessionLog):
        return src, log_factory is not None
    if isinstance(src, (str, Path)):
        return SessionLog.open(src), True
    raise TypeError(f"Durable: unsupported log {src!r}")


# Deprecated: {resolved path: {"leaf": id}} filled by DurableParticipant from a
# manifest written before Durable dehydrated itself (read for one release).
RESTORE_SERVICE = "pyreact_durable.restore"


def _log_key(slog: SessionLog) -> str:
    return str(Path(slog.path).resolve())


def _checkout(slog: SessionLog, leaf: Optional[str]) -> None:
    if leaf and leaf != slog.leaf and leaf in slog.by_id and not slog.readonly:
        slog.checkout(leaf)


def _log_record(slog: SessionLog, turns: Any) -> Dict[str, Any]:
    """The provider's slice: where its log is and where its leaf was."""
    store = getattr(turns, "store", None)
    pending: List[str] = []
    if getattr(store, "log", None) is slog and not slog.closed:
        pending = [t.id for t in store.pending()]
    return {"path": _log_key(slog), "log_id": slog.id, "leaf": slog.leaf,
            "entries": len(slog.entries), "version": slog.version,
            "pending_turns": pending}


class LogHydrationError(RuntimeError):
    """The log a snapshot points at cannot be reopened as it was."""


def _degrade(session: Any, reason: str, **info: Any) -> None:
    fn = getattr(session, "degrade", None)
    if callable(fn):
        try:
            fn("durable", reason, **info)
        except Exception:
            pass


def _check_record(rec: Any, session: Any) -> Optional[Dict[str, Any]]:
    """``hydrate`` of the provider's slice: the log must still be there and
    be the same log (header id). A leaf no longer in it is dropped (the log
    opens at its current leaf). Raising makes it a ``hydration.mismatch``
    (the provider then opens its log as usual, a fresh one if it is gone)."""
    if not isinstance(rec, dict) or not rec.get("path"):
        return None
    path = Path(rec["path"])
    if not path.exists():
        _degrade(session, "log missing", path=str(path))
        raise LogHydrationError(f"log missing: {path}")
    try:
        log = SessionLog.open(path, readonly=True)
    except Exception as exc:
        _degrade(session, "log unreadable", path=str(path), error=f"{type(exc).__name__}: {exc}")
        raise LogHydrationError(f"log unreadable: {path} ({exc})") from exc
    try:
        if rec.get("log_id") and log.id != rec["log_id"]:
            _degrade(session, "log replaced", path=str(path), expected=rec["log_id"],
                     found=log.id)
            raise LogHydrationError(f"log replaced: {path} (header {log.id}, "
                                    f"expected {rec['log_id']})")
        leaf = rec.get("leaf")
        if leaf and leaf not in log.by_id:
            _degrade(session, "leaf missing", path=str(path), leaf=leaf)
            rec = {**rec, "leaf": None}
    finally:
        log.close()
    return dict(rec)


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj, allow_nan=False)
        return obj
    except (TypeError, ValueError):
        pass

    def fallback(o: Any) -> Any:
        for attr in ("model_dump", "to_dict", "toDict"):
            fn = getattr(o, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        return repr(o)

    try:
        return json.loads(json.dumps(obj, default=fallback))
    except (TypeError, ValueError):
        return repr(obj)


def _last_turn_index(log: SessionLog) -> int:
    """Highest turn index journaled in ``log`` (any branch), 0 if none."""
    last = 0
    for e in log.entries:
        if e.get("type") == "event" and str(e.get("kind", "")).startswith("turn."):
            idx = (e.get("payload") or {}).get("index")
            if isinstance(idx, int) and idx > last:
                last = idx
    return last


def _retire(slog: SessionLog, on_close: Any, archive_dir: Any, session_id: str) -> None:
    """Apply the ``on_close`` retention policy to a closed, owned log."""
    try:
        if on_close == "archive":
            archive_log(slog.path, archive_dir)
        elif on_close == "delete":
            delete_log(slog.path)
        elif callable(on_close):
            on_close(slog.path, session_id)
    except Exception as exc:  # retention must never break unmounting
        warnings.warn(f"Durable(on_close={on_close!r}) failed for {slog.path}: {exc}",
                      RuntimeWarning, stacklevel=2)


@component(transparent=True)
def Durable(log=None, log_factory=None, journal=DEFAULT_JOURNAL, on_close="keep",
            archive_dir=None, resume_turns=True, children=None):
    """Provide a :class:`SessionLog` to the subtree and journal bus events.

    - ``log``: a :class:`SessionLog` (caller owns it) or a path (opened on
      mount, closed on unmount).
    - ``log_factory``: ``fn(session_id) -> SessionLog | path`` — one log per
      session when the same app is mounted in several sessions, e.g.
      ``Durable(log_factory=lambda sid: f"logs/{sid}.jsonl", children=[...])``.
      Logs it returns are closed on unmount. The factory receives the id
      escaped by :func:`~pyreact_durable.log.safe_session_id` (ids made of
      ``[A-Za-z0-9_-]`` are unchanged; ``/``, ``.``... are percent-encoded),
      so it is safe to put in a path.
    - ``on_close``: what happens to a log *this provider opened* (from a path
      or the factory) when its **session is closed** — reaped after its TTL
      (``session.close_reason == "idle"``) or ``close_session()``
      (``"closed"``); never on runner shutdown or when only this subtree
      unmounts:

      * ``"keep"`` (default): leave the file;
      * ``"archive"``: move it (and backend sidecars) to ``archive_dir``
        (default ``<log dir>/archive/``);
      * ``"delete"``: remove it;
      * ``fn(path, session_id)``: your own policy, called after closing.
    - ``journal``: EventBus kinds appended as ``event`` entries (``"x.*"``
      prefixes allowed; ``None`` or ``()`` disables journaling).

    On mount the session's ``TurnManager`` continues numbering after the
    highest turn index journaled in the log (``TurnManager.advance_to``), so
    turn ids (``t1``, ``t2``, ...) are not reused across restarts.

    Suspend / hydrate (ADR 0001): the provider dehydrates a reference to its
    log (``use_dehydrate("log")``: ``{path, log_id, leaf, entries, version,
    pending_turns}``), so ``root.suspend()`` carries it and ``hydrate_root``
    reopens the log at that leaf before the children's first render. A log
    that is gone or was replaced is a ``hydration.mismatch`` (and
    ``session.resume.degraded``): the provider opens its log as usual. Under
    a hidden ``Activity`` the log stays open.

    - ``resume_turns`` (default ``True``, needs ``turn.*`` in ``journal``):
      attach a :class:`~pyreact_durable.turns.DurableTurnStore` to the
      session's ``TurnManager`` (unless it has a store already), so turns
      cut short by a crash can be re-run with ``app.resume_pending()`` /
      ``session.resume_turn(turn_id)``.
    """
    if not (on_close in ON_CLOSE or callable(on_close)):
        raise ValueError(f"Durable: on_close must be one of {ON_CLOSE} or a callable, "
                         f"got {on_close!r}")
    session = _session()
    sid = _session_id() if log_factory is not None else None
    # Keyed on the log identity / path and the session, not on the factory's
    # identity (an inline lambda is a new object every render).
    log_src = log if isinstance(log, SessionLog) else (str(log) if log is not None else None)
    turns = hooks.get_service("turns", TurnManager)
    # The provider's own state: a reference to its log (ADR 0001). A suspend
    # dehydrates it ({path, log_id, leaf, entries, version, pending_turns});
    # hydrate_root reopens the log at that leaf before the children render.
    live = hooks.use_ref(None)  # {"log": SessionLog, "record": captured at suspend}
    hydrated = use_dehydrate(
        "log",
        dehydrate=lambda: (live.current or {}).get("record") or _log_record(
            live.current["log"], turns),
        hydrate=lambda rec: _check_record(rec, session),
    )

    def _open_and_restore():
        slog, owned = _open(log, log_factory, sid)
        legacy = hooks.get_service(RESTORE_SERVICE, dict).pop(_log_key(slog), None)
        rec = hydrated if live.current is None else None  # first open only
        if rec and rec.get("path") != _log_key(slog):
            _degrade(session, "log moved", path=_log_key(slog), expected=rec.get("path"))
        elif rec:
            _checkout(slog, rec.get("leaf"))
        elif legacy:
            _checkout(slog, legacy.get("leaf"))
        return slog, owned

    opened = hooks.use_memo(_open_and_restore, [log_src, log_factory is None, sid])
    slog, owned = opened
    live.current = {"log": slog, "record": None}

    gen, set_gen = hooks.use_state(0)
    retention = hooks.use_ref(None)
    retention.current = (on_close, archive_dir)

    def _own():
        def _close():
            reason = use_cleanup_reason()
            if reason == "hidden":  # Activity: keep the log open, state is kept
                return
            if reason == "suspend":  # the tree is about to be dehydrated
                live.current = {"log": slog, "record": _log_record(slog, turns)}
            if not owned:
                return
            slog.close()
            policy, where = retention.current
            reason = getattr(session, "close_reason", None)
            if policy != "keep" and reason in ("idle", "closed"):
                _retire(slog, policy, where, str(getattr(session, "id", "")))

        return _close

    hooks.use_effect(_own, [slog])

    def _watch():
        return slog.subscribe(lambda _entry, moved: moved and set_gen(lambda g: g + 1))

    hooks.use_effect(_watch, [slog])

    kinds = tuple(journal) if journal else ()  # () subscribes to nothing

    def _journal(ev) -> None:
        if slog.closed:
            return
        slog.append(
            "event", kind=ev.kind, payload=_jsonable(ev.payload), turn_id=ev.turn_id,
            source=ev.source, ts=ev.ts,
        )

    use_event(kinds, _journal)

    def _continue_turn_ids():
        last = _last_turn_index(slog)
        if last:
            turns.advance_to(last + 1)

    hooks.use_effect(_continue_turn_ids, [slog, turns])

    tracks_turns = bool(resume_turns) and all(
        any(kind_matches(k, [p]) for p in kinds) for k in ("turn.start", "turn.end"))

    def _attach_turn_store():
        if not tracks_turns or getattr(turns, "store", "n/a") is not None:
            return None
        from .turns import DurableTurnStore

        store = DurableTurnStore(slog)
        turns.configure(store=store)

        def _detach():
            if turns.store is store:
                turns.store = None

        return _detach

    hooks.use_effect(_attach_turn_store, [slog, turns, tracks_turns])

    hooks.provide(_DurableCtx, DurableHandle(slog, gen))
    return list(children or [])


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def use_durable() -> Optional[SessionLog]:
    """The nearest :func:`Durable`'s log (``None`` without a provider)."""
    handle = hooks.use_context(_DurableCtx)
    return handle.log if handle is not None else None


def _warn_no_provider(hook: str, key: str) -> None:
    if hook in _WARNED:
        return
    _WARNED.add(hook)
    warnings.warn(
        f"{hook}(key={key!r}) at {hooks.path} has no Durable provider above it; "
        "state is in-memory only (behaves like use_state). Wrap the tree in "
        "pyreact_durable.Durable(log=...) to persist it.",
        RuntimeWarning,
        stacklevel=3,
    )


def _check_json(value: Any, path: str, key: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DurableStateError(
            f"persistent state {key!r} of component {path!r} is not JSON-serializable "
            f"({type(exc).__name__}: {exc}). Store plain dict/list/str/int/float/bool/None."
        ) from None


def _restore(log: SessionLog, path: str, full_path: str, key: str) -> Any:
    """Value at ``(path, key)``; falls back to ``full_path`` (where logs
    written before providers became transparent stored it, e.g.
    ``App/Durable[d]/DSPyProvider[dspy]/_Provider[#0]/Chat[chat]``). The next
    set writes under ``path``, so old logs migrate on first write."""
    restored = log.get_state(path, key, _MISSING)
    if restored is _MISSING and full_path != path:
        restored = log.get_state(full_path, key, _MISSING)
    return restored


def _use_box(hook: str, key: str, make_initial: Callable[[], Any]) -> Tuple[dict, Callable]:
    """Shared machinery: a per-component box holding the current value, kept
    in sync with the log position, and a ``commit(new)`` that persists."""
    handle = hooks.use_context(_DurableCtx)
    path = hooks.path
    ref = hooks.use_ref(None)
    if ref.current is None:  # lazy init: the RLock is created once
        ref.current = {"sync": _UNSET, "value": None, "lock": threading.RLock(),
                       "mounted": True, "log": None}
    box = ref.current
    _tick, bump = hooks.use_state(0)

    log = handle.log if handle is not None else None
    if log is None:
        _warn_no_provider(hook, key)
    sync = (id(log), handle.generation) if handle is not None else None
    with box["lock"]:
        if box["sync"] != sync:
            if log is not None:
                restored = _restore(log, path, hooks.full_path, key)
                box["value"] = make_initial() if restored is _MISSING else restored
            elif box["sync"] is _UNSET:
                box["value"] = make_initial()
            box["sync"] = sync
        box["log"] = log
        box["path"] = path

    def _lifecycle():
        box["mounted"] = True

        def _unmounted():
            box["mounted"] = False

        return _unmounted

    hooks.use_effect(_lifecycle, [])

    def commit(compute: Callable[[Any], Any]) -> None:
        if not box["mounted"]:
            return
        with box["lock"]:
            cur = box["value"]
            new = compute(cur)
            if new == cur:
                return
            slog = box["log"]
            if slog is not None:
                _check_json(new, box["path"], key)
                slog.set_state(box["path"], key, new)
            box["value"] = new
        bump(lambda n: n + 1)

    return box, commit


def use_persistent_state(initial: Any, *, key: str):
    """``use_state`` whose value survives restarts.

    Addressed by ``(hooks.path, key)``: give components stable ``key=`` props
    so their path does not depend on sibling order. Providers (``Durable``,
    ``DSPyProvider``, context providers...) add no path segment, so wrapping
    the tree in more providers later keeps the address (``App/Chat[chat]``).
    Values stored by older versions under the every-node path
    (``hooks.full_path``) are still found. On first render (and after
    a checkout) the value is restored from ``log.state_at(leaf)``; setting it
    appends a ``state`` entry (values must be JSON-serializable, else
    :class:`DurableStateError`). Without a :func:`Durable` above, behaves like
    ``use_state`` (and warns once).

    Returns ``(value, set_value)``; ``set_value`` accepts a value or a
    function of the previous value (an updater: see
    :func:`pyreact.core.is_updater`).
    """
    box, commit = _use_box("use_persistent_state", key, lambda: initial)

    def set_value(val):  # functions are updaters; callable objects are values
        commit(lambda cur: val(cur) if is_updater(val) else val)

    setter = hooks.use_callback(set_value, [])
    return box["value"], setter


def use_persistent_reducer(reducer: Callable[[Any, Any], Any], initial: Any, *, key: str,
                           init_fn: Optional[Callable[[Any], Any]] = None):
    """``use_reducer`` whose state survives restarts (the *state* is
    persisted after each effective dispatch, not the actions). ``init_fn``
    runs only when nothing was restored."""
    box, commit = _use_box(
        "use_persistent_reducer", key,
        lambda: init_fn(initial) if init_fn is not None else initial,
    )
    box["reducer"] = reducer

    def dispatch(action):
        commit(lambda cur: box["reducer"](cur, action))

    dispatcher = hooks.use_callback(dispatch, [])
    return box["value"], dispatcher


class Conversation:
    """Persisted chat transcript returned by :func:`use_conversation`.

    ``messages`` is a list of ``{"role", "content", **meta}`` dicts. Updates
    are functional (safe from any thread) and each call is one ``state``
    entry, so use :meth:`extend` to record a question/answer pair atomically.
    """

    def __init__(self, messages: List[dict], set_messages: Callable) -> None:
        self.messages: List[dict] = list(messages)
        self._set = set_messages

    def append(self, role: str, content: Any, **meta: Any) -> None:
        msg = {"role": role, "content": content, **meta}
        self._set(lambda ms: [*ms, msg])

    def extend(self, messages: Iterable[dict]) -> None:
        new = [dict(m) for m in messages]
        self._set(lambda ms: [*ms, *new])

    def clear(self) -> None:
        self._set([])

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.messages)

    def as_chat_messages(self) -> List[Dict[str, Any]]:
        """``[{"role", "content"}]`` (OpenAI chat format)."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def as_dspy_history(self, input_field: str = "question", output_field: str = "answer"):
        """``dspy.History`` of user→assistant pairs mapped to the signature's
        input/output field names (unpaired messages are skipped)."""
        import dspy

        pairs: List[dict] = []
        pending: Optional[dict] = None
        for m in self.messages:
            if m.get("role") == "user":
                pending = m
            elif m.get("role") == "assistant" and pending is not None:
                pairs.append({input_field: pending["content"], output_field: m["content"]})
                pending = None
        return dspy.History(messages=pairs)

    def __repr__(self) -> str:
        return f"<Conversation {len(self.messages)} messages>"


def use_conversation(key: str = "history") -> Conversation:
    """A persisted message list (see :class:`Conversation`)."""
    messages, set_messages = use_persistent_state([], key=key)
    return Conversation(messages or [], set_messages)


def use_durable_lm(lm: Any, *, mode: str = "replay_or_record", **kwargs: Any):
    """``durable_lm(lm, <provider log>)``, memoized; returns ``lm`` unchanged
    without a :func:`Durable` provider."""
    log = use_durable()

    def _make():
        if log is None:
            return lm
        from .lm import durable_lm

        return durable_lm(lm, log, mode=mode, **kwargs)

    return hooks.use_memo(_make, [id(lm), id(log), mode])

