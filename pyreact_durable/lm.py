"""Durable execution at the LM boundary (record / replay).

``durable_lm(lm, log)`` wraps any ``dspy.BaseLM``. Each call is keyed by a
hash of (model, prompt/messages, relevant kwargs); the wrapper keeps a
per-key *cursor* (how many calls with that key it has made in this run).

- **record**: call the model, append an ``llm`` entry
  ``{key, occurrence, model, request, response}`` on the current branch.
- **replay**: return the ``occurrence``-th journaled response with that key
  on the root→leaf path, without calling the model.
- **replay_or_record** (default): replay when journaled, else record.
- **passthrough**: no journaling.

So re-running the same code along the same branch (e.g. after a crash, or
re-running a turn whose state update was lost) gets the journaled answers
back: once a response is recorded, that LLM call happens exactly once. The
window between the model answering and the entry hitting the file is still
at-least-once (use ``SessionLog(fsync=True)`` to shrink it to the write).

The cursor resets when the branch changes under the wrapper (the leaf it last
saw is no longer on the current path, e.g. after a checkout).

This is the only module that imports dspy, and only when a wrapper is built.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .log import SessionLog

__all__ = ["MODES", "DEFAULT_IGNORED_KWARGS", "ReplayMissError", "durable_lm", "call_key",
           "llm_entries"]

MODES = ("replay_or_record", "record", "replay", "passthrough")

# Transport/config kwargs that do not change what the model is asked.
DEFAULT_IGNORED_KWARGS = frozenset({
    "api_key", "api_base", "base_url", "headers", "timeout", "request_timeout",
    "num_retries", "cache", "callbacks", "stream",
})


class ReplayMissError(LookupError):
    """``mode="replay"`` and no journaled response matches the call."""


def _plain(obj: Any) -> Any:
    """Best-effort JSON-able copy of an SDK object."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return _plain(dump())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: _plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


def _ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def call_key(model: str, prompt: Any, messages: Any, kwargs: Dict[str, Any],
             ignored: Iterable[str] = DEFAULT_IGNORED_KWARGS) -> str:
    """Stable 16-hex key of an LM request."""
    ign = set(ignored)
    rel = {k: v for k, v in sorted(kwargs.items()) if k not in ign}
    blob = json.dumps({"model": model, "prompt": prompt, "messages": messages, "kwargs": rel},
                      sort_keys=True, default=repr, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _serialize(resp: Any, model_type: str) -> Dict[str, Any]:
    if model_type != "chat":
        return {"raw": _plain(resp)}
    choices = []
    for c in getattr(resp, "choices", None) or []:
        if isinstance(c, dict):
            choices.append(_plain(c))
            continue
        item: Dict[str, Any] = {"finish_reason": getattr(c, "finish_reason", None)}
        msg = getattr(c, "message", None)
        if msg is not None:
            item["message"] = {
                "role": getattr(msg, "role", "assistant"),
                "content": getattr(msg, "content", None),
                "tool_calls": _plain(getattr(msg, "tool_calls", None)),
                "reasoning_content": getattr(msg, "reasoning_content", None),
            }
        else:
            item["text"] = getattr(c, "text", None)
        choices.append(item)
    usage = _plain(getattr(resp, "usage", None)) or {}
    return {"choices": choices, "usage": usage if isinstance(usage, dict) else {},
            "model": getattr(resp, "model", None)}


def _rebuild(data: Dict[str, Any]) -> Any:
    if "raw" in data:
        resp = _ns(data["raw"])
    else:
        choices = []
        for c in data.get("choices", []):
            if "message" in c:
                m = dict(c["message"])
                m["tool_calls"] = _ns(m.get("tool_calls")) or None
                choices.append(SimpleNamespace(message=SimpleNamespace(**m),
                                               finish_reason=c.get("finish_reason")))
            else:
                choices.append(dict(c))
        resp = SimpleNamespace(choices=choices, usage=dict(data.get("usage") or {}),
                               model=data.get("model"))
    resp._hidden_params = {"response_cost": None, "replayed": True}
    return resp


_CLS = None


def _durable_cls():
    global _CLS
    if _CLS is not None:
        return _CLS
    import dspy

    class DurableLM(dspy.BaseLM):
        """A ``dspy.BaseLM`` that journals/replays another one. See module doc."""

        def __init__(self, inner: Any, log: SessionLog, *, mode: str = "replay_or_record",
                     ignored_kwargs: Iterable[str] = DEFAULT_IGNORED_KWARGS,
                     store_request: bool = True) -> None:
            if mode not in MODES:
                raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
            super().__init__(model=inner.model,
                             model_type=getattr(inner, "model_type", "chat"), cache=False)
            self.kwargs = dict(getattr(inner, "kwargs", {}) or {})
            self.inner = inner
            self.log = log
            self.mode = mode
            self.ignored_kwargs = frozenset(ignored_kwargs)
            self.store_request = store_request
            self.stats = {"replayed": 0, "recorded": 0, "inner_calls": 0}
            self._cursor: Dict[str, int] = {}
            self._anchor: Optional[str] = None
            self._mu = threading.Lock()

        # -- capabilities follow the wrapped LM
        @property
        def supports_function_calling(self) -> bool:
            return bool(getattr(self.inner, "supports_function_calling", False))

        @property
        def supports_reasoning(self) -> bool:
            return bool(getattr(self.inner, "supports_reasoning", False))

        @property
        def supports_response_schema(self) -> bool:
            return bool(getattr(self.inner, "supports_response_schema", False))

        @property
        def supported_params(self) -> set:
            return set(getattr(self.inner, "supported_params", set()) or set())

        def __deepcopy__(self, memo):  # dspy's lm.copy() deep-copies the LM
            twin = type(self)(self.inner, self.log, mode=self.mode,
                              ignored_kwargs=self.ignored_kwargs,
                              store_request=self.store_request)
            twin.kwargs = dict(self.kwargs)
            return twin

        def reset(self) -> None:
            """Forget the replay cursors (next calls replay from the start of
            the current branch)."""
            with self._mu:
                self._cursor.clear()
                self._anchor = None

        # -- core
        def _claim(self, prompt, messages, kwargs) -> Tuple[str, int, Optional[dict]]:
            key = call_key(self.inner.model, prompt, messages,
                           {**self.kwargs, **kwargs}, self.ignored_kwargs)
            with self._mu:
                path = self.log.path_to()
                ids = {e["id"] for e in path}
                if self._anchor is not None and self._anchor not in ids:
                    self._cursor.clear()  # branch changed under us
                occ = self._cursor.get(key, 0)
                self._cursor[key] = occ + 1
                hit = None
                if self.mode in ("replay", "replay_or_record"):
                    found = [e for e in path if e.get("type") == "llm" and e.get("key") == key]
                    hit = found[occ] if occ < len(found) else None
                self._anchor = self.log.leaf
            return key, occ, hit

        def _replay(self, hit: dict) -> Any:
            self.stats["replayed"] += 1
            return _rebuild(hit["response"])

        def _miss(self, key: str, occ: int) -> None:
            raise ReplayMissError(
                f"durable_lm(mode='replay'): no journaled response for call {key} "
                f"(occurrence {occ}) on the current branch of {self.log.path}"
            )

        def _record(self, key, occ, prompt, messages, kwargs, resp, latency) -> None:
            fields: Dict[str, Any] = {
                "key": key, "occurrence": occ, "model": self.inner.model,
                "response": _serialize(resp, self.model_type),
                "latency": round(latency, 3),
            }
            if self.store_request:
                fields["request"] = {"prompt": prompt, "messages": _plain(messages),
                                     "kwargs": _plain({k: v for k, v in kwargs.items()
                                                       if k not in self.ignored_kwargs})}
            self.log.append("llm", **fields)
            with self._mu:
                self._anchor = self.log.leaf
            self.stats["recorded"] += 1

        def forward(self, prompt=None, messages=None, **kwargs):
            if self.mode == "passthrough":
                self.stats["inner_calls"] += 1
                return self.inner.forward(prompt=prompt, messages=messages, **kwargs)
            key, occ, hit = self._claim(prompt, messages, kwargs)
            if hit is not None:
                return self._replay(hit)
            if self.mode == "replay":
                self._miss(key, occ)
            t0 = time.time()
            self.stats["inner_calls"] += 1
            resp = self.inner.forward(prompt=prompt, messages=messages, **kwargs)
            self._record(key, occ, prompt, messages, kwargs, resp, time.time() - t0)
            return resp

        async def aforward(self, prompt=None, messages=None, **kwargs):
            inner_async = type(self.inner).aforward is not dspy.BaseLM.aforward
            if self.mode == "passthrough":
                self.stats["inner_calls"] += 1
                if inner_async:
                    return await self.inner.aforward(prompt=prompt, messages=messages, **kwargs)
                return await asyncio.to_thread(self.inner.forward, prompt=prompt,
                                               messages=messages, **kwargs)
            key, occ, hit = self._claim(prompt, messages, kwargs)
            if hit is not None:
                return self._replay(hit)
            if self.mode == "replay":
                self._miss(key, occ)
            t0 = time.time()
            self.stats["inner_calls"] += 1
            if inner_async:
                resp = await self.inner.aforward(prompt=prompt, messages=messages, **kwargs)
            else:
                resp = await asyncio.to_thread(self.inner.forward, prompt=prompt,
                                               messages=messages, **kwargs)
            self._record(key, occ, prompt, messages, kwargs, resp, time.time() - t0)
            return resp

        def __repr__(self) -> str:
            return f"<DurableLM {self.model} mode={self.mode} {self.stats}>"

    _CLS = DurableLM
    return _CLS


def durable_lm(lm: Any, log: SessionLog, *, mode: str = "replay_or_record",
               ignored_kwargs: Iterable[str] = DEFAULT_IGNORED_KWARGS,
               store_request: bool = True):
    """Wrap ``lm`` (a ``dspy.BaseLM``) so its calls are journaled in ``log``
    and replayed on re-runs along the same branch. See the module doc."""
    return _durable_cls()(lm, log, mode=mode, ignored_kwargs=ignored_kwargs,
                          store_request=store_request)


def __getattr__(name: str):
    if name == "DurableLM":
        return _durable_cls()
    raise AttributeError(name)


def llm_entries(log: SessionLog, entry_id: Optional[str] = None) -> List[dict]:
    """Journaled ``llm`` entries on the root→entry path (default: leaf)."""
    return [e for e in log.path_to(entry_id) if e.get("type") == "llm"]

