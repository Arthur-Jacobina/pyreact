from .app_runner import AppRunner
from .session import DEFAULT_SESSION_ID


def bootstrap(
    app_component_fn,
    *,
    fps: int = 20,
    turn_policy: str = "queue",
    session_id: str = DEFAULT_SESSION_ID,
    default_session: bool = True,
) -> AppRunner:
    """Create and start an AppRunner for the given root component.

    ``turn_policy``: ``"queue"`` (default) processes inputs one turn at a
    time; ``"interrupt"`` cancels the in-flight turn when a new input arrives.
    ``session_id`` names the default session.
    ``default_session=False`` starts the loop without mounting anything; open
    sessions with ``runner.open_session(...)``.
    """
    return AppRunner(
        app_component_fn,
        fps=fps,
        turn_policy=turn_policy,
        session_id=session_id,
        default_session=default_session,
    )
