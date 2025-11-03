from .app_runner import AppRunner


def bootstrap(app_component_fn, *, fps: int = 20) -> AppRunner:
    """Create and start an AppRunner for the given root component.

    trace=True habilita tracing de render por padrão.
    """
    return AppRunner(app_component_fn, fps=fps)
