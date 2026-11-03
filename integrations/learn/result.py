"""``OptimizationResult``: what an optimizer built, and how to ship it.

``.diff()`` shows per-predictor instructions and demos before -> after,
``.report()`` a summary, ``.save()`` / :func:`load` persist it as JSON
(program state + metadata + the baseline state, so ``diff`` works offline),
``.hot_swap(target)`` makes it the live program (HMR): it updates the
``LearnedProgram`` provider that declares it (a re-render), else replaces the
module in the app's ``DSPyEnv`` (legacy).
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dspy

__all__ = ["OptimizationResult", "load", "predictor_view", "diff_states", "format_demo"]

FORMAT = "pyreact.learn/1"


# ---------------------------------------------------------------------------
# State views (work on live programs and on saved JSON alike)
# ---------------------------------------------------------------------------
def _state_of(program_or_state: Any) -> Dict[str, Any]:
    if program_or_state is None:
        return {}
    if isinstance(program_or_state, dict):
        return program_or_state
    return program_or_state.dump_state()


def predictor_view(program_or_state: Any) -> Dict[str, Dict[str, Any]]:
    """``{predictor: {"instructions": str, "demos": [dict], "fields": [str]}}``."""
    state = _state_of(program_or_state)
    if "signature" in state and "demos" in state:  # a bare Predict
        state = {"self": state}
    out: Dict[str, Dict[str, Any]] = {}
    for name, st in state.items():
        if not isinstance(st, dict) or "signature" not in st:
            continue
        sig = st.get("signature") or {}
        out[name] = {
            "instructions": sig.get("instructions", ""),
            "demos": list(st.get("demos") or []),
            "fields": [f.get("prefix", "").rstrip(":") or f.get("name", "")
                       for f in sig.get("fields", []) if isinstance(f, dict)],
        }
    return out


def format_demo(demo: Dict[str, Any], width: int = 100) -> str:
    items = [(k, v) for k, v in demo.items() if not k.startswith("_") and k != "augmented"]
    text = ", ".join(f"{k}={str(v)!r}" for k, v in items)
    return textwrap.shorten(text, width=width, placeholder=" …")


def _wrap(prefix: str, text: str, width: int = 96) -> List[str]:
    lines: List[str] = []
    for para in (text or "").splitlines() or [""]:
        chunk = textwrap.wrap(para, width=width) or [""]
        lines += [prefix + c for c in chunk]
    return lines


def diff_states(before: Any, after: Any, *, labels=("before", "after")) -> str:
    """Readable per-predictor diff of instructions and demos."""
    a, b = predictor_view(before), predictor_view(after)
    out: List[str] = []
    for name in list(dict.fromkeys([*a, *b])):
        pa, pb = a.get(name), b.get(name)
        out.append(f"predictor {name!r}")
        if pa is None or pb is None:
            out.append(f"  only in {labels[1] if pa is None else labels[0]}")
            continue
        ia, ib = pa["instructions"].strip(), pb["instructions"].strip()
        if ia == ib:
            out.append("  instructions: (unchanged)")
            out += _wrap("    ", textwrap.shorten(ia, 300, placeholder=" …"))
        else:
            out.append("  instructions:")
            out += _wrap("  - ", ia)
            out += _wrap("  + ", ib)
        da, db = pa["demos"], pb["demos"]
        sa = [format_demo(d) for d in da]
        sb = [format_demo(d) for d in db]
        if sa == sb:
            out.append(f"  demos: {len(da)} (unchanged)")
        else:
            out.append(f"  demos: {len(da)} -> {len(db)}")
            for s in sa:
                if s not in sb:
                    out.append(f"  - {s}")
            for s in sb:
                if s not in sa:
                    out.append(f"  + {s}")
    return "\n".join(out) if out else "(no predictors)"


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    program: Any
    optimizer: str
    baseline: Optional[float] = None
    score: Optional[float] = None
    calls: int = 0
    seconds: float = 0.0
    budget: Any = None
    train_size: int = 0
    val_size: int = 0
    metric: str = ""
    arch: str = ""
    module_key: Any = None
    details: Dict[str, Any] = field(default_factory=dict)
    baseline_program: Any = None
    baseline_state: Optional[Dict[str, Any]] = None
    signature: str = ""
    path: Optional[str] = None

    def __post_init__(self):
        if self.baseline_state is None and self.baseline_program is not None:
            self.baseline_state = self.baseline_program.dump_state()
        if not self.signature and self.program is not None:
            from integrations.architectures import signature_of, signature_text

            self.signature = signature_text(signature_of(self.program))

    # -- views --------------------------------------------------------------
    @property
    def state(self) -> Dict[str, Any]:
        return self.program.dump_state() if self.program is not None else dict(self._saved_state)

    _saved_state: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def improved(self) -> Optional[bool]:
        if self.baseline is None or self.score is None:
            return None
        return self.score > self.baseline

    @property
    def delta(self) -> Optional[float]:
        if self.baseline is None or self.score is None:
            return None
        return self.score - self.baseline

    def diff(self, other: Any = None) -> str:
        """Instructions + demos, baseline -> optimized (or this -> ``other``)."""
        if other is None:
            return diff_states(self.baseline_state or {}, self.state)
        other_state = other.state if isinstance(other, OptimizationResult) else _state_of(other)
        return diff_states(self.state, other_state, labels=("this", "other"))

    def report(self) -> str:
        d = self.delta
        arrow = "" if d is None else f"  ({'+' if d >= 0 else ''}{d:.3f})"
        lines = [
            f"optimizer   {self.optimizer}  (budget={self.budget})",
            f"program     {self.arch or '?'}  {self.signature}",
            f"key         {self.module_key}" if self.module_key is not None else None,
            f"data        train={self.train_size}  val={self.val_size}  metric={self.metric}",
            f"score       {_fmt(self.baseline)} -> {_fmt(self.score)}{arrow}",
            f"cost        {self.calls} LM calls, {self.seconds:.1f}s"
            + (f"  {self.details.get('calls_by_phase')}" if self.details.get("calls_by_phase") else ""),
        ]
        vals = self.details.get("val_aggregate_scores")
        if vals:
            lines.append(f"candidates  {len(vals)} (val scores: {', '.join(f'{v:.2f}' for v in vals)})")
        views = predictor_view(self.state)
        for name, v in views.items():
            first = (v["instructions"].strip().splitlines() or [""])[0]
            lines.append(f"  [{name}] demos={len(v['demos'])}  "
                         f"{textwrap.shorten(first, 80, placeholder=' …')}")
        if self.path:
            lines.append(f"saved       {self.path}")
        return "\n".join(x for x in lines if x is not None)

    def __str__(self) -> str:
        return self.report()

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        from integrations.feedback import key_str

        sig_cls = None
        from integrations.architectures import signature_of

        sig = signature_of(self.program) if self.program is not None else None
        if isinstance(sig, type) and getattr(sig, "__module__", "").split(".")[0] not in ("dspy",):
            sig_cls = f"{sig.__module__}:{sig.__qualname__}"
        details = {k: v for k, v in self.details.items() if k not in ("val_rows", "baseline_rows")}
        return {
            "format": FORMAT,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "optimizer": self.optimizer,
            "budget": self.budget,
            "baseline": self.baseline,
            "score": self.score,
            "calls": self.calls,
            "seconds": self.seconds,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "metric": self.metric,
            "arch": self.arch,
            "signature": self.signature,
            "signature_class": sig_cls,
            "module_key": None if self.module_key is None else key_str(self.module_key),
            "details": details,
            "state": self.state,
            "baseline_state": self.baseline_state,
        }

    def save(self, path: str) -> str:
        """Write JSON (program state + metadata + baseline state) to ``path``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        self.path = path
        return path

    # -- shipping -----------------------------------------------------------
    def hot_swap(self, target: Any, key: Any = None) -> List[Any]:
        """Make :attr:`program` the live program for ``key`` (HMR).

        A thin wrapper: when ``target`` (a ``Root``, an ``AppRunner`` /
        ``Session``, or ``None`` for headless trees) mounts a ``LearnedProgram`` /
        ``LearnedPrograms`` provider that declares ``key`` (or whose
        ``use_learned`` users match it), that provider's state is updated
        and its subtree re-renders with this program -- the same thing as
        re-rendering it with ``src=<this result>``. Returns the paths of the
        components that read it.

        Otherwise (legacy, apps without a provider) the module cached under
        ``key`` in the app's ``DSPyEnv`` is replaced in place; ``target`` may
        also be a ``DSPyEnv`` for that. ``key``: the learned name or module
        key (tuple, ``"Cls:sig:scope"`` string, or just the scope /
        component key, e.g. ``"supervisor"``); default: ``module_key``.
        """
        from integrations.dspy_integration import DSPyEnv

        from .hmr import hot_swap_into
        from .programs import swap_learned

        for attr, val in (("_pyreact_result_path", self.path),
                          ("_pyreact_optimizer", self.optimizer)):
            if val:
                try:
                    setattr(self.program, attr, val)  # a suspend can point at the file
                except Exception:
                    pass
        key = key if key is not None else self.module_key
        if not isinstance(target, DSPyEnv):
            users = swap_learned(target, key, self)
            if users is not None:
                return users
        return hot_swap_into(target, key, self.program)


def load(path: str, program: Any = None) -> OptimizationResult:
    """Load a saved result. The program is rebuilt from ``program`` (a module
    or a zero-arg factory) or else from the saved signature + architecture."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "state" not in data:
        # a plain dspy `module.save(path)` file: wrap it
        data = {"format": "dspy", "optimizer": "?", "state": data}
    prog = program() if callable(program) and not isinstance(program, dspy.Module) else program
    if prog is None:
        prog = _rebuild(data)
    if prog is not None:
        try:
            prog.load_state(data["state"])
        except Exception as exc:
            raise ValueError(f"could not load state from {path} into {type(prog).__name__}: {exc}")
    res = OptimizationResult(
        program=prog,
        optimizer=data.get("optimizer", "?"),
        baseline=data.get("baseline"),
        score=data.get("score"),
        calls=data.get("calls", 0),
        seconds=data.get("seconds", 0.0),
        budget=data.get("budget"),
        train_size=data.get("train_size", 0),
        val_size=data.get("val_size", 0),
        metric=data.get("metric", ""),
        arch=data.get("arch", ""),
        module_key=data.get("module_key"),
        details=data.get("details") or {},
        baseline_state=data.get("baseline_state"),
        signature=data.get("signature", ""),
        path=path,
    )
    res._saved_state = data["state"]
    return res


def _rebuild(data: Dict[str, Any]) -> Any:
    from integrations.architectures import build_module

    arch, sig = data.get("arch"), data.get("signature_class") or data.get("signature")
    if not arch or not sig:
        return None
    if data.get("signature_class"):
        try:
            import importlib

            mod_name, _, qual = data["signature_class"].partition(":")
            obj = importlib.import_module(mod_name)
            for part in qual.split("."):
                obj = getattr(obj, part)
            sig = obj
        except Exception:
            sig = data.get("signature")
    try:
        return build_module(sig, arch)
    except Exception:
        return None
