from .app_runner import AppRunner


def bootstrap(app_component_fn, *, fps: int = 20, turn_policy: str = "queue") -> AppRunner:
    """Create and start an AppRunner for the given root component.

    ``turn_policy``: ``"queue"`` (default) processes inputs one turn at a
    time; ``"interrupt"`` cancels the in-flight turn when a new input arrives.

    trace=True habilita tracing de render por padrão.
    """
    return AppRunner(app_component_fn, fps=fps, turn_policy=turn_policy)
