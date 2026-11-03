"""Learning, the React way: architectures are props, optimizers are build steps.

    architecture  how a component thinks        use_module(sig, arch="cot" | "rlm" | ...)
    optimizer     build step -> better program  optimize(program, trainset=..., metric=...)
    learned prog  a prop / context              LearnedProgram(key="supervisor", src="x.json")
                                                use_module(sig, program=use_learned("supervisor"))
    hot swap      HMR = a re-render             src changes -> users re-render with it
                                                (result.hot_swap(app, "supervisor") updates
                                                 that provider's state)
    events        the trace you debug/learn from dataset_from_events(app.events, key)
                                                explain(call_id) / inspect_program(program)

Quick start::

    from integrations.learn import optimize, exact_match, dataset_from_events, split

    data = dataset_from_events("events.jsonl", "supervisor", min_score=1.0)
    train, val = split(data, val=0.3)
    result = optimize(router, trainset=train, valset=val, metric=exact_match("name"),
                      optimizer="gepa", budget=40, lm=lm, reflection_lm=lm)
    print(result.report()); print(result.diff())
    result.save("artifacts/router.json")
    # in the app: LearnedProgram(key="supervisor", src="artifacts/router.json", children=[...])
    #             use_module(RouteSig, program=use_learned("supervisor"))
    result.hot_swap(app, "supervisor")  # live: re-renders the provider with the new src

Importing this package also registers the ``:modules``, ``:inspect``,
``:explain``, ``:feedback`` and ``:optimize`` terminal commands.
CLI: ``python -m integrations.learn --help``.
"""

from integrations.architectures import (
    build_module,
    get_architecture,
    list_architectures,
    register_architecture,
    unregister_architecture,
)
from integrations.hooks import use_agent_call, use_module
from integrations.telemetry import set_debug

from .datasets import call_records, dataset_from_events, iter_events, split
from .debug import explain, inspect_program
from .devtools import (
    LAST_RESULTS,
    configure_learning,
    install_commands,
    run_optimization,
    wait_optimizations,
)
from .hmr import find_envs, hot_swap_into, list_modules, resolve_module
from .metrics import Metric, MetricError, as_metric, contains, exact_match, llm_judge
from .optimizers import (
    BudgetError,
    OptimizeError,
    count_lm_calls,
    evaluate,
    get_optimizer,
    list_optimizers,
    optimize,
    register_optimizer,
    unregister_optimizer,
)
from .programs import (
    LearnedProgram,
    LearnedPrograms,
    LearnedRegistry,
    SavedProgram,
    find_registries,
    swap_learned,
    use_learned,
    use_learned_registry,
)
from .result import OptimizationResult, diff_states, load

install_commands()

__all__ = [
    # architectures
    "use_module",
    "use_agent_call",
    "build_module",
    "register_architecture",
    "unregister_architecture",
    "get_architecture",
    "list_architectures",
    # optimizers
    "optimize",
    "evaluate",
    "register_optimizer",
    "unregister_optimizer",
    "get_optimizer",
    "list_optimizers",
    "OptimizationResult",
    "load",
    # learned programs are props
    "LearnedProgram",
    "LearnedPrograms",
    "LearnedRegistry",
    "use_learned",
    "use_learned_registry",
    "find_registries",
    "swap_learned",
    "SavedProgram",
    "diff_states",
    "count_lm_calls",
    "BudgetError",
    "OptimizeError",
    # metrics
    "Metric",
    "MetricError",
    "as_metric",
    "exact_match",
    "contains",
    "llm_judge",
    # datasets
    "dataset_from_events",
    "call_records",
    "iter_events",
    "split",
    # debugging / dev tools
    "inspect_program",
    "explain",
    "set_debug",
    "configure_learning",
    "find_envs",
    "list_modules",
    "resolve_module",
    "hot_swap_into",
    "run_optimization",
    "wait_optimizations",
    "install_commands",
    "LAST_RESULTS",
]
