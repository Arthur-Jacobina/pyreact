"""Dev tools: find modules in a running app, hot-swap by key, terminal commands.

Terminal commands (registered on import of :mod:`integrations.learn` through
``pyreact.boot.terminal.register_command``)::

    :modules                          modules in the app: key, arch, compiled?, swaps
    :inspect <key>                    instructions / demos of a module
    :explain [call_id]                exact prompt + raw completion of a call (default: last)
    :feedback <+|-|0..1> [@call|@<call_id>] [text] [field=value ...]
                                      feedback on the last turn (or call); field=value
                                      is a correction (becomes the label)
    :optimize <key> [optimizer] [budget]
                                      optimize from this session's runs + feedback in a
                                      worker thread; publishes optimize.* events and
                                      hot-swaps the result when it is not worse

``<key>`` is anything that identifies a module: its full key
(``ChainOfThought:RouteSig:App/...#supervisor``), its component key / scope
(``supervisor``) or its label. Configure what ``:optimize`` uses with
:func:`configure_learning` (metric, LMs, extra examples, defaults).
"""

from __future__ import annotations

import shlex
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from integrations.feedback import key_str

from .hmr import hot_swap_into, list_modules, resolve_module

__all__ = [
    "configure_learning",
    "learning_config",
    "run_optimization",
    "install_commands",
    "wait_optimizations",
    "LAST_RESULTS",
]


# ---------------------------------------------------------------------------
# Learning config (what :optimize uses)
# ---------------------------------------------------------------------------
@dataclass
class LearnConfig:
    metric: Any = None  # default: exact_match() over label fields
    lm: Any = None  # default: the env's default LM
    reflection_lm: Any = None  # default: lm
    optimizer: str = "gepa"
    budget: Any = "light"
    trainset: List[Any] = field(default_factory=list)  # extra examples, always included
    valset: Optional[List[Any]] = None
    min_examples: int = 3
    min_score: float = 0.5  # feedback score that counts as "good" (corrections always count)
    val_fraction: float = 0.3
    save_dir: Optional[str] = "artifacts/learn"
    swap_if_worse: bool = False
    opts: Dict[str, Any] = field(default_factory=dict)


_CONFIG: Dict[Optional[str], LearnConfig] = {None: LearnConfig()}


def configure_learning(key: Optional[str] = None, **kw: Any) -> LearnConfig:
    """Set what ``:optimize`` uses (globally, or for one module key/scope)."""
    base = _CONFIG.get(key) or LearnConfig(**{f: getattr(_CONFIG[None], f)
                                              for f in LearnConfig.__dataclass_fields__})
    for k, v in kw.items():
        if k not in LearnConfig.__dataclass_fields__:
            raise TypeError(f"unknown learning option {k!r}")
        setattr(base, k, v)
    _CONFIG[key] = base
    return base


def learning_config(key: Any = None) -> LearnConfig:
    if key is not None:
        ks = key_str(key)
        for k, cfg in _CONFIG.items():
            if k is not None and (k == ks or ks.endswith(k) or ks.split(":")[-1].endswith(k)):
                return cfg
    return _CONFIG[None]


# ---------------------------------------------------------------------------
# Terminal commands
# ---------------------------------------------------------------------------
GRAY, RESET, BOLD, GREEN, RED, YELLOW = "\x1b[90m", "\x1b[0m", "\x1b[1m", "\x1b[32m", "\x1b[31m", "\x1b[33m"
LAST_RESULTS: Dict[str, Any] = {}
_RUNNING: Dict[str, threading.Thread] = {}


def _log_of(app: Any):
    return getattr(app, "events", None)


def _bus_of(app: Any):
    return getattr(app, "event_bus", None)


def _last_call_event(app: Any):
    log = _log_of(app)
    if log is None:
        return None
    evs = log.recent(None, ("llm.end", "llm.error"))
    return evs[-1] if evs else None


def cmd_modules(app: Any, args: str = "") -> None:
    rows = list_modules(app)
    if not rows:
        print(f"{GRAY}no DSPy modules yet (they appear after the first render){RESET}")
        return
    print(f"\n{BOLD}=== Modules ({len(rows)}) ==={RESET}")
    for r in rows:
        flag = f"{GREEN}compiled{RESET}" if r["compiled"] else f"{GRAY}base{RESET}"
        swaps = f" swaps={r['swaps']}" if r["swaps"] else ""
        print(f"  {r['arch']:<10} {flag}{swaps}  {r['key']}")
    print()


def cmd_inspect(app: Any, args: str = "") -> None:
    from .debug import inspect_program

    key = args.strip() or None
    try:
        _env, k, mod = resolve_module(app, key)
    except LookupError as exc:
        print(f"{RED}{exc}{RESET}")
        return
    print(f"\n{BOLD}=== {key_str(k)} ==={RESET}")
    print(inspect_program(mod))
    print()


def cmd_explain(app: Any, args: str = "") -> None:
    from .debug import explain

    cid = args.strip() or None
    if cid is None:
        ev = _last_call_event(app)
        cid = ev.payload.get("call_id") if ev is not None else None
    try:
        print(explain(cid, events=_log_of(app)))
    except LookupError as exc:
        print(f"{RED}{exc}{RESET}")


def _parse_feedback(args: str):
    try:
        toks = shlex.split(args)
    except ValueError:
        toks = args.split()
    if not toks:
        raise ValueError("usage: :feedback <+|-|0..1> [@call|@<call_id>] [text] [field=value ...]")
    sign, rest = toks[0], toks[1:]
    if sign in ("+", "good", "up"):
        score = 1.0
    elif sign in ("-", "bad", "down"):
        score = 0.0
    else:
        try:
            score = float(sign)
        except ValueError:
            raise ValueError(f"first argument must be +, - or a score, got {sign!r}") from None
    target, expected, words = None, {}, []
    for t in rest:
        if t.startswith("@") and len(t) > 1:
            target = t[1:]
        elif "=" in t and not t.startswith("=") and " " not in t.split("=", 1)[0]:
            k, v = t.split("=", 1)
            expected[k] = v
        else:
            words.append(t)
    return score, target, " ".join(words), expected


def cmd_feedback(app: Any, args: str = "") -> None:
    from integrations.feedback import send_feedback

    try:
        score, target, text, expected = _parse_feedback(args)
    except ValueError as exc:
        print(f"{RED}{exc}{RESET}")
        return
    ev = _last_call_event(app)
    if target in (None, "call", "turn"):
        if ev is None:
            print(f"{RED}no call to give feedback on yet{RESET}")
            return
        if target == "call" or (target is None and not ev.turn_id):
            target = ev.payload.get("call_id")
        else:
            target = ev.turn_id
    bus = _bus_of(app)
    send_feedback(target, score, text, expected=expected or None, bus=bus)
    what = "turn" if ev is not None and target == ev.turn_id else "call"
    print(f"{GRAY}[learn]{RESET} feedback {score:g} on {what} {target}"
          + (f" (correction {expected})" if expected else "")
          + (f": {text}" if text else ""))


def run_optimization(app: Any, key: str, optimizer: Optional[str] = None,
                     budget: Any = None, *, block: bool = False) -> Optional[threading.Thread]:
    """What ``:optimize`` does (also usable from code / tests)."""
    from .datasets import dataset_from_events, split
    from .metrics import exact_match
    from .optimizers import optimize

    env, k, module = resolve_module(app, key)
    ks = key_str(k)
    if ks in _RUNNING and _RUNNING[ks].is_alive():
        print(f"{YELLOW}already optimizing {ks}{RESET}")
        return _RUNNING[ks]
    cfg = learning_config(k)
    optimizer = optimizer or cfg.optimizer
    budget = cfg.budget if budget is None else budget
    data = dataset_from_events(_log_of(app), k, min_score=cfg.min_score,
                               require_feedback=True)
    data = [*data, *cfg.trainset]
    if len(data) < cfg.min_examples:
        print(f"{RED}only {len(data)} example(s) for {ks} (need {cfg.min_examples}).{RESET} "
              "Chat a bit and rate answers with :feedback + / :feedback - field=value, "
              "or configure_learning(trainset=[...]).")
        return None
    if cfg.valset:
        train, val = data, list(cfg.valset)
    else:
        train, val = split(data, val=cfg.val_fraction)
    metric = cfg.metric or exact_match()
    lm = cfg.lm or env.lm_for()
    reflection_lm = cfg.reflection_lm or lm
    bus = _bus_of(app)

    def pub(kind, payload):
        if bus is not None:
            bus.publish(kind, {"key": ks, **payload}, source="learn")

    def work():
        pub("optimize.start", {"optimizer": optimizer, "budget": budget,
                               "train": len(train), "val": len(val)})
        print(f"{GRAY}[learn]{RESET} optimizing {ks} with {optimizer} (budget={budget}, "
              f"train={len(train)}, val={len(val)}) …")
        try:
            res = optimize(module, trainset=train, valset=val or None, metric=metric,
                           optimizer=optimizer, budget=budget, lm=lm,
                           reflection_lm=reflection_lm, module_key=k,
                           on_progress=lambda phase, info: pub("optimize.progress",
                                                               {"phase": phase, **info}),
                           **cfg.opts)
        except Exception as exc:
            pub("optimize.error", {"error": f"{type(exc).__name__}: {exc}"})
            print(f"{RED}[learn] optimize failed: {exc}{RESET}")
            traceback.print_exc(limit=3)
            return
        LAST_RESULTS[ks] = res
        path = None
        if cfg.save_dir:
            import os

            scope = ks.split(":")[-1].split("/")[-1].replace("#", "-") or "module"
            path = res.save(os.path.join(cfg.save_dir,
                                         f"{scope}-{optimizer}-{time.strftime('%H%M%S')}.json"))
        ok = res.score is None or res.baseline is None or res.score >= res.baseline
        swapped = False
        if ok or cfg.swap_if_worse:
            hot_swap_into(app, k, res.program)
            swapped = True
        pub("optimize.end", {"optimizer": optimizer, "baseline": res.baseline,
                             "score": res.score, "calls": res.calls, "seconds": res.seconds,
                             "swapped": swapped, "path": path})
        print(f"\n{BOLD}=== optimize {ks} ==={RESET}\n{res.report()}\n\n{res.diff()}\n")
        print(f"{GREEN}[learn] hot-swapped: the next call uses the optimized program{RESET}"
              if swapped else f"{YELLOW}[learn] not better than baseline: kept the current "
                               f"program (configure_learning(swap_if_worse=True) to force){RESET}")

    t = threading.Thread(target=work, name=f"optimize:{ks}", daemon=True)
    _RUNNING[ks] = t
    t.start()
    if block:
        t.join()
    return t


def wait_optimizations(timeout: Optional[float] = None) -> bool:
    """Block until every ``:optimize`` worker finished (True if all done)."""
    end = None if timeout is None else time.time() + timeout
    for t in list(_RUNNING.values()):
        t.join(None if end is None else max(0.0, end - time.time()))
    return all(not t.is_alive() for t in _RUNNING.values())


def cmd_optimize(app: Any, args: str = "") -> None:
    toks = args.split()
    if not toks:
        print(f"{RED}usage: :optimize <key> [optimizer] [budget]{RESET}")
        cmd_modules(app)
        return
    key = toks[0]
    optimizer = toks[1] if len(toks) > 1 else None
    budget: Any = None
    if len(toks) > 2:
        budget = int(toks[2]) if toks[2].isdigit() else toks[2]
    try:
        run_optimization(app, key, optimizer, budget)
    except LookupError as exc:
        print(f"{RED}{exc}{RESET}")


def install_commands() -> None:
    """Register :modules / :inspect / :explain / :feedback / :optimize."""
    from pyreact.boot.terminal import register_command

    register_command("modules", cmd_modules, "DSPy modules: key, arch, compiled?, swaps")
    register_command("inspect", cmd_inspect, "<key>  instructions + demos of a module")
    register_command("explain", cmd_explain, "[call_id]  exact prompt + completion (default: last)")
    register_command("feedback", cmd_feedback,
                     "<+|-|score> [@call] [text] [field=value]  rate the last turn/call")
    register_command("optimize", cmd_optimize,
                     "<key> [optimizer] [budget]  optimize from feedback, then hot-swap")
