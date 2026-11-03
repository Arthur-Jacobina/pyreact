"""``optimize()``: every DSPy optimizer behind one call.

In React terms an optimizer is a *build step*: it takes a program (a
component's "how it thinks") plus examples and a metric and produces a better
program, which you then hot-swap into the running app (HMR)::

    result = optimize(program, trainset=train, valset=val, metric=exact_match("label"),
                      optimizer="gepa", budget=40, lm=lm, reflection_lm=lm)
    print(result.report()); print(result.diff())
    result.save("artifacts/router.json")
    result.hot_swap(app, "supervisor")

Built-in optimizers and what ``budget`` means for each:

    =============  ==========================================  ==========================
    name           DSPy optimizer                              budget (int)
    =============  ==========================================  ==========================
    gepa           dspy.GEPA (reflective prompt evolution)     max_metric_calls
    mipro          dspy.MIPROv2 (instructions + demos, Optuna) num_trials
    bootstrap      dspy.BootstrapFewShot (self-generated demos) max_bootstrapped_demos
    bootstrap_rs   dspy.BootstrapFewShotWithRandomSearch       num_candidate_programs
    simba          dspy.SIMBA (stochastic introspective ascent) max_steps
    copro          dspy.COPRO (coordinate instruction ascent)  depth
    =============  ==========================================  ==========================

``budget="light" | "medium" | "heavy"`` uses the optimizer's own presets
(GEPA/MIPRO ``auto``) or a small/medium/large value for the others.

Add your own with :func:`register_optimizer`.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextvars
import inspect
import os
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import dspy

from dspy.utils.callback import BaseCallback

from .metrics import Metric, MetricError, as_metric

__all__ = [
    "optimize",
    "evaluate",
    "register_optimizer",
    "unregister_optimizer",
    "list_optimizers",
    "get_optimizer",
    "OptimizerSpec",
    "BudgetError",
    "OptimizeError",
    "count_lm_calls",
]

Budget = Union[str, int]
PRESETS = ("light", "medium", "heavy")


class OptimizeError(ValueError):
    """Bad optimize() inputs (empty data, missing LM, ...)."""


class BudgetError(OptimizeError):
    """The budget cannot buy a meaningful optimization run."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    adapter: Callable[..., Any]
    doc: str = ""
    needs_reflection_lm: bool = False
    budget_doc: str = ""


_OPTIMIZERS: Dict[str, OptimizerSpec] = {}


def register_optimizer(
    name: str,
    adapter: Callable[..., Any],
    *,
    doc: str = "",
    needs_reflection_lm: bool = False,
    budget_doc: str = "",
    replace: bool = True,
) -> OptimizerSpec:
    """Register ``adapter`` under ``name`` for ``optimize(optimizer=name)``.

    ``adapter(program, *, trainset, valset, metric, budget, lm, reflection_lm,
    log_dir, num_threads, seed, **opts)`` returns the optimized program or
    ``(program, details_dict)``. ``metric`` is a :class:`Metric` (a standard
    ``(example, pred, trace=None) -> float`` callable; ``metric.for_gepa()`` /
    ``metric.feedback(...)`` give feedback text). It runs inside
    ``dspy.context(lm=lm)`` on a private copy of the program.
    """
    if not callable(adapter):
        raise TypeError("optimizer adapter must be callable")
    if not replace and name in _OPTIMIZERS:
        raise ValueError(f"optimizer {name!r} is already registered")
    spec = OptimizerSpec(name, adapter, doc or (adapter.__doc__ or "").strip(),
                         needs_reflection_lm, budget_doc)
    _OPTIMIZERS[name] = spec
    return spec


def unregister_optimizer(name: str) -> None:
    _OPTIMIZERS.pop(name, None)


def get_optimizer(name: str) -> OptimizerSpec:
    try:
        return _OPTIMIZERS[name]
    except KeyError:
        raise KeyError(f"unknown optimizer {name!r}; registered: {sorted(_OPTIMIZERS)} "
                       "(add one with register_optimizer(name, adapter))") from None


def list_optimizers() -> Dict[str, str]:
    """``{name: one-line doc}`` of every registered optimizer."""
    return {n: (s.doc.splitlines()[0] if s.doc else "") for n, s in _OPTIMIZERS.items()}


# ---------------------------------------------------------------------------
# LM call accounting: a DSPy callback scoped with dspy.context (propagates to
# DSPy's worker threads; LM copies made by optimizers are counted too)
# ---------------------------------------------------------------------------
class _CountCallback(BaseCallback):
    def __init__(self) -> None:
        self.n = 0
        self._lock = threading.Lock()

    def on_lm_start(self, call_id, instance, inputs):
        with self._lock:
            self.n += 1


class count_lm_calls:  # noqa: N801 - used as a context manager
    """``with count_lm_calls() as c: ...; c.n`` -- LM calls made in this context
    (and in DSPy worker threads it spawns)."""

    def __init__(self) -> None:
        self.cb = _CountCallback()
        self._ctx = None

    @property
    def n(self) -> int:
        return self.cb.n

    def __enter__(self) -> "count_lm_calls":
        current = list(dspy.settings.get("callbacks", None) or [])
        self._ctx = dspy.context(callbacks=[*current, self.cb])
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*exc)


# ---------------------------------------------------------------------------
# Evaluation (our own: never swallows metric type errors)
# ---------------------------------------------------------------------------
def evaluate(
    program: Any,
    dataset: Sequence[dspy.Example],
    metric: Union[Callable, Metric],
    *,
    lm: Any = None,
    num_threads: int = 4,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Average score of ``program`` on ``dataset`` + per-example rows.

    Program errors score 0 (recorded in the row); a metric returning a wrong
    type raises :class:`MetricError` immediately.
    """
    metric = as_metric(metric)
    if not dataset:
        return 0.0, []

    def one(ex):
        try:
            if lm is not None:
                with dspy.context(lm=lm):
                    pred = program(**ex.inputs())
            else:
                pred = program(**ex.inputs())
        except Exception as exc:
            return {"inputs": dict(ex.inputs()), "pred": None, "score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}"[:300]}
        score, fb = metric.feedback(ex, pred)
        outs = {k: v for k, v in pred.items()} if hasattr(pred, "items") else {"value": pred}
        return {"inputs": dict(ex.inputs()), "pred": outs, "score": score, "feedback": fb,
                "error": None}

    with cf.ThreadPoolExecutor(max_workers=max(1, num_threads)) as pool:
        futs = [pool.submit(contextvars.copy_context().run, one, ex) for ex in dataset]
        rows = [f.result() for f in futs]  # re-raises MetricError
    return sum(r["score"] for r in rows) / len(rows), rows


# ---------------------------------------------------------------------------
# optimize()
# ---------------------------------------------------------------------------
def _as_examples(data: Iterable[Any], program: Any, what: str) -> List[dspy.Example]:
    from integrations.architectures import signature_of

    out = []
    sig = signature_of(program)
    input_names = None
    if sig is not None:
        try:
            input_names = list(dspy.ensure_signature(sig).input_fields)
        except Exception:
            input_names = None
    for i, ex in enumerate(data or []):
        if isinstance(ex, dict):
            ex = dspy.Example(**ex)
        if not isinstance(ex, dspy.Example):
            raise OptimizeError(f"{what}[{i}] must be a dspy.Example or dict, got {type(ex).__name__}")
        if not getattr(ex, "_input_keys", None):
            if not input_names:
                raise OptimizeError(f"{what}[{i}] has no input fields: use "
                                    "dspy.Example(...).with_inputs('question', ...)")
            ex = ex.with_inputs(*[k for k in input_names if k in ex])
        out.append(ex)
    return out


def _prepare_log_dir(log_dir: Optional[str], resume: bool) -> Optional[str]:
    if not log_dir:
        return None
    if os.path.isdir(log_dir) and os.listdir(log_dir) and not resume:
        base = os.path.join(log_dir, time.strftime("run-%Y%m%d-%H%M%S"))
        fresh, i = base, 1
        while os.path.exists(fresh):
            i += 1
            fresh = f"{base}-{i}"
        warnings.warn(
            f"log_dir {log_dir!r} already holds a previous run; GEPA would silently RESUME it "
            f"(budget, candidates and scores carried over). Writing to {fresh!r} instead -- "
            "pass resume=True to continue the old run.",
            UserWarning, stacklevel=3,
        )
        return fresh
    return log_dir


def optimize(
    program: Any,
    *,
    trainset: Iterable[Any],
    metric: Callable,
    valset: Optional[Iterable[Any]] = None,
    optimizer: str = "gepa",
    budget: Budget = "light",
    lm: Any = None,
    reflection_lm: Any = None,
    log_dir: Optional[str] = None,
    resume: bool = False,
    num_threads: int = 4,
    seed: int = 0,
    evaluate_program: bool = True,
    baseline: Optional[float] = None,
    on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    module_key: Any = None,
    arch: Optional[str] = None,
    **opts: Any,
):
    """Optimize ``program`` and return an :class:`OptimizationResult`.

    - ``program``: a ``dspy.Module`` (left untouched; a copy is optimized) or a
      signature string (built with ``arch=``, default ``"predict"``).
    - ``trainset`` / ``valset``: ``dspy.Example`` (or dicts; inputs inferred
      from the signature). Scores (``.baseline`` / ``.score``) are on
      ``valset`` (or ``trainset`` when there is none).
    - ``metric(example, pred) -> float | bool | (score, feedback) |
      Prediction(score=, feedback=)`` -- adapted to every optimizer.
    - ``budget``: ``"light"|"medium"|"heavy"`` or an int (meaning per optimizer:
      see the module docstring).
    - ``lm``: student LM (default: ``dspy.settings.lm``); ``reflection_lm``:
      the LM that proposes new instructions (required by GEPA).
    - ``log_dir``: optimizer logs/checkpoints. A non-empty ``log_dir`` is not
      resumed unless ``resume=True`` (a fresh sub-directory is used + warning).
    - ``baseline``: known baseline score (skips the baseline evaluation);
      ``evaluate_program=False`` skips both evaluations.
    - ``on_progress(phase, info)``: ``"baseline"``, ``"compile"``, ``"evaluate"``,
      ``"done"`` callbacks (the ``:optimize`` dev command turns them into events).
    - ``**opts``: passed to the optimizer (constructor or compile, by name).
    """
    from integrations.architectures import arch_of, build_module

    from .result import OptimizationResult

    if isinstance(program, str):
        program = build_module(program, arch or "predict")
    if not isinstance(program, dspy.Module):
        raise OptimizeError(f"program must be a dspy.Module or a signature string, got {program!r}")
    spec = get_optimizer(optimizer)
    train = _as_examples(trainset, program, "trainset")
    if not train:
        raise OptimizeError(
            "trainset is empty: collect examples first (e.g. dataset_from_events(app.events, "
            "module_key) after a few runs + feedback) or pass a list of dspy.Example."
        )
    val = _as_examples(valset, program, "valset") if valset is not None else None
    if val is not None and not val:
        val = None
    metric = as_metric(metric)
    student_lm = lm if lm is not None else dspy.settings.lm
    if student_lm is None:
        raise OptimizeError("No LM: pass lm=... or dspy.configure(lm=...).")
    if spec.needs_reflection_lm and reflection_lm is None:
        raise OptimizeError(
            f"optimizer {optimizer!r} needs reflection_lm=... -- the LM that reads metric "
            "feedback and proposes new instructions (use a strong model; reflection_lm=lm "
            "reuses the student)."
        )
    if isinstance(budget, str) and budget not in PRESETS:
        raise BudgetError(f"budget must be one of {PRESETS} or an int, got {budget!r}")
    if not isinstance(budget, str) and (not isinstance(budget, int) or budget < 1):
        raise BudgetError(f"budget must be a positive int or one of {PRESETS}, got {budget!r}")

    def progress(phase: str, **info):
        if on_progress is not None:
            try:
                on_progress(phase, info)
            except Exception:
                pass

    eval_set = val or train
    baseline_program = program.deepcopy()
    log_dir = _prepare_log_dir(log_dir, resume)
    t0 = time.time()
    phase_calls: Dict[str, int] = {}
    baseline_rows: List[Dict[str, Any]] = []
    with count_lm_calls() as total:
        if evaluate_program and baseline is None:
            progress("baseline", n=len(eval_set))
            with count_lm_calls() as c:
                baseline, baseline_rows = evaluate(program, eval_set, metric, lm=student_lm,
                                                   num_threads=num_threads)
            phase_calls["baseline"] = c.n
        progress("compile", optimizer=optimizer, budget=budget, train=len(train),
                 val=len(val or []), baseline=baseline)
        student = program.deepcopy()
        with count_lm_calls() as c:
            with dspy.context(lm=student_lm):
                out = spec.adapter(
                    student, trainset=train, valset=val, metric=metric, budget=budget,
                    lm=student_lm, reflection_lm=reflection_lm, log_dir=log_dir,
                    num_threads=num_threads, seed=seed, **opts,
                )
        phase_calls["compile"] = c.n
        compiled, details = out if isinstance(out, tuple) else (out, {})
        if compiled is None:
            raise OptimizeError(f"optimizer {optimizer!r} returned no program")
        for attr in ("_pyreact_arch", "_pyreact_signature", "_pyreact_key", "_pyreact_name"):
            if hasattr(program, attr) and not hasattr(compiled, attr):
                try:
                    setattr(compiled, attr, getattr(program, attr))
                except Exception:
                    pass
        score, rows = None, []
        if evaluate_program:
            progress("evaluate", n=len(eval_set))
            with count_lm_calls() as c:
                score, rows = evaluate(compiled, eval_set, metric, lm=student_lm,
                                       num_threads=num_threads)
            phase_calls["evaluate"] = c.n
    result = OptimizationResult(
        program=compiled,
        baseline_program=baseline_program,
        optimizer=optimizer,
        baseline=baseline,
        score=score,
        calls=total.n,
        seconds=round(time.time() - t0, 2),
        budget=budget,
        train_size=len(train),
        val_size=len(val or []),
        metric=metric.name,
        arch=arch_of(program),
        module_key=module_key if module_key is not None else getattr(program, "_pyreact_key", None),
        details={**(details or {}), "calls_by_phase": phase_calls, "log_dir": log_dir,
                 "val_rows": rows, "baseline_rows": baseline_rows},
    )
    progress("done", score=score, baseline=baseline, calls=result.calls)
    return result


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------
def _split_opts(opts: Dict[str, Any], *targets: Callable) -> List[Dict[str, Any]]:
    """Route ``opts`` to the callables whose signature names them."""
    outs: List[Dict[str, Any]] = [{} for _ in targets]
    for k, v in opts.items():
        for i, t in enumerate(targets):
            if k in inspect.signature(t).parameters:
                outs[i][k] = v
                break
        else:
            raise OptimizeError(f"unknown option {k!r} for this optimizer")
    return outs


def _preset(budget: Budget, light: int, medium: int, heavy: int) -> int:
    if isinstance(budget, int):
        return budget
    return {"light": light, "medium": medium, "heavy": heavy}[budget]


def _gepa(program, *, trainset, valset, metric: Metric, budget, lm, reflection_lm, log_dir,
          num_threads, seed, **opts):
    """Reflective prompt evolution (dspy.GEPA): reads metric feedback, rewrites instructions."""
    kw: Dict[str, Any] = {}
    n_val = len(valset or trainset)
    if isinstance(budget, str):
        kw["auto"] = budget
    else:
        need = n_val + 2 * min(3, len(trainset)) + 1
        if budget < need:
            raise BudgetError(
                f"GEPA budget {budget} is too small: scoring the seed program alone costs "
                f"len(valset)={n_val} metric calls, and each reflection step another "
                f"~{2 * min(3, len(trainset))}. Use budget >= {need} (or a smaller valset)."
            )
        kw["max_metric_calls"] = budget
    kw.setdefault("reflection_minibatch_size", min(3, len(trainset)))
    kw.update(opts)
    gepa = dspy.GEPA(metric=metric.for_gepa(), reflection_lm=reflection_lm,
                     num_threads=num_threads, log_dir=log_dir, seed=seed, track_stats=True, **kw)
    compiled = gepa.compile(program, trainset=trainset, valset=valset)
    details: Dict[str, Any] = {}
    dr = getattr(compiled, "detailed_results", None)
    if dr is not None:
        details["val_aggregate_scores"] = list(getattr(dr, "val_aggregate_scores", []) or [])
        details["best_idx"] = getattr(dr, "best_idx", None)
        details["candidates"] = len(getattr(dr, "candidates", []) or [])
        details["total_metric_calls"] = getattr(dr, "total_metric_calls", None)
    return compiled, details


def _mipro(program, *, trainset, valset, metric, budget, lm, reflection_lm, log_dir,
           num_threads, seed, **opts):
    """Instruction + demo search with Bayesian optimization (dspy.MIPROv2)."""
    try:  # fail before spending LM calls on proposals
        import optuna  # noqa: F401
    except ImportError:
        raise OptimizeError("optimizer 'mipro' needs the optional dependency optuna: "
                            "`uv add optuna` (or pip install 'dspy[optuna]').") from None
    init_o, comp_o = _split_opts(opts, dspy.MIPROv2.__init__, dspy.MIPROv2.compile)
    init = dict(metric=metric, prompt_model=reflection_lm or lm, task_model=lm,
                num_threads=num_threads, seed=seed or 9, log_dir=log_dir, verbose=False,
                max_bootstrapped_demos=2, max_labeled_demos=2)
    comp: Dict[str, Any] = {}
    if isinstance(budget, str):
        init["auto"] = budget
    else:
        init["auto"] = None
        init["num_candidates"] = max(2, min(budget, 6))
        comp["num_trials"] = budget
    init.update(init_o)
    if valset is None:
        if len(trainset) < 2:
            raise OptimizeError("MIPROv2 needs >= 2 training examples (or a valset)")
        cut = max(1, len(trainset) // 2)
        trainset, valset = trainset[:cut], trainset[cut:]
    comp.setdefault("minibatch", len(valset) > 35)
    comp.setdefault("minibatch_size", min(35, len(valset)))
    comp.update(comp_o)
    tp = dspy.MIPROv2(**init)
    compiled = tp.compile(program, trainset=trainset, valset=valset, **comp)
    return compiled, {"best_score": getattr(compiled, "score", None)}


def _bootstrap(program, *, trainset, valset, metric, budget, lm, reflection_lm, log_dir,
               num_threads, seed, **opts):
    """Self-generated few-shot demos that pass the metric (dspy.BootstrapFewShot)."""
    n = _preset(budget, 2, 4, 8)
    kw = dict(metric=metric, max_bootstrapped_demos=n, max_labeled_demos=n)
    kw.update(opts)
    compiled = dspy.BootstrapFewShot(**kw).compile(program, trainset=trainset)
    return compiled, {}


def _bootstrap_rs(program, *, trainset, valset, metric, budget, lm, reflection_lm, log_dir,
                  num_threads, seed, **opts):
    """Bootstrapped demo sets + random search over them (BootstrapFewShotWithRandomSearch)."""
    n = _preset(budget, 2, 6, 12)
    kw = dict(metric=metric, num_candidate_programs=n, max_bootstrapped_demos=2,
              max_labeled_demos=2, num_threads=num_threads)
    kw.update(opts)
    compiled = dspy.BootstrapFewShotWithRandomSearch(**kw).compile(
        program, trainset=trainset, valset=valset)
    return compiled, {}


def _simba(program, *, trainset, valset, metric, budget, lm, reflection_lm, log_dir,
           num_threads, seed, **opts):
    """Stochastic introspective mini-batch ascent: rules + demos (dspy.SIMBA)."""
    if len(trainset) < 2:
        raise OptimizeError("SIMBA needs >= 2 training examples")
    steps = _preset(budget, 2, 6, 12)
    kw = dict(metric=metric, max_steps=steps, bsize=min(32, len(trainset)),
              num_candidates=min(4, len(trainset)), num_threads=num_threads,
              prompt_model=reflection_lm)
    kw.update(opts)
    compiled = dspy.SIMBA(**kw).compile(program, trainset=trainset, seed=seed)
    return compiled, {}


def _copro(program, *, trainset, valset, metric, budget, lm, reflection_lm, log_dir,
           num_threads, seed, **opts):
    """Coordinate ascent over instructions (dspy.COPRO)."""
    depth = _preset(budget, 1, 2, 3)
    kw = dict(metric=metric, breadth=opts.pop("breadth", 2), depth=depth,
              prompt_model=reflection_lm or lm, track_stats=False)
    kw.update(opts)
    compiled = dspy.COPRO(**kw).compile(
        program, trainset=trainset,
        eval_kwargs={"num_threads": num_threads, "display_progress": False},
    )
    return compiled, {}


register_optimizer("gepa", _gepa, needs_reflection_lm=True, budget_doc="max_metric_calls")
register_optimizer("mipro", _mipro, budget_doc="num_trials")
register_optimizer("bootstrap", _bootstrap, budget_doc="max_bootstrapped_demos")
register_optimizer("bootstrap_rs", _bootstrap_rs, budget_doc="num_candidate_programs")
register_optimizer("simba", _simba, budget_doc="max_steps")
register_optimizer("copro", _copro, budget_doc="depth")

__all__ += ["MetricError"]
