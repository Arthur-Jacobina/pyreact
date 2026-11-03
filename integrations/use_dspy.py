import asyncio
from collections.abc import Callable
from typing import Optional, Any
import inspect
import dspy
from pyreact.core.core import hooks, _context_stack
from pyreact.core.runtime import spawn, current_turn_id
from integrations.dspy_integration import use_dspy_env


def use_dspy_call(
    module, *, model: Optional[str] = None, lm: Optional[Any] = None
) -> tuple[Callable[..., None], tuple, bool, Optional[Exception]]:
    """
    Policy: replace (the last call wins; the previous in-flight call is cancelled).
    Returns (run, result, loading, error) where ``result`` is ``(prediction, ver)``.
    ``ver`` increases only when a call settles (success or error), so effects
    that depend on ``result`` fire exactly once per completed call.
    """

    def reducer(state, action):
        typ = action["type"]
        if typ == "start":
            return {
                "status": "loading",
                "result": state["result"],
                "error": None,
                "ver": state["ver"],
            }
        if typ == "success":
            return {
                "status": "idle",
                "result": action["result"],
                "error": None,
                "ver": state["ver"] + 1,
            }
        if typ == "error":
            return {
                "status": "idle",
                "result": None,
                "error": action["error"],
                "ver": state["ver"] + 1,
            }
        return state

    initial = {"status": "idle", "result": None, "error": None, "ver": 0}
    state, dispatch = hooks.use_reducer(reducer, initial)

    alive_ref = hooks.use_memo(lambda: {"alive": True}, [])
    task_ref = hooks.use_memo(lambda: {"task": None}, [])

    def _mount_cleanup():
        def _un():
            alive_ref["alive"] = False
            t = task_ref["task"]
            if t is not None and not t.done():
                t.cancel()

        return _un

    hooks.use_effect(_mount_cleanup, [])

    # Get environment during render (not in async function)
    env = use_dspy_env()
    owner = _context_stack.get()  # component calling the hook (turn policy, errors)

    async def _do_call(inputs: dict):
        current_task = asyncio.current_task()

        try:
            if lm is not None:
                selected_lm = lm
            elif env is not None and model is not None:
                selected_lm = env.models.get(model, env.models.get("default"))
            elif env is not None:
                selected_lm = env.models.get("default")
            else:
                selected_lm = None  # fall back to dspy global default if configured

            async def _call():
                if selected_lm is not None:
                    ctx_mgr = dspy.context(lm=selected_lm)
                else:
                    # Provide more helpful error message for SSR scenarios
                    if env is None:
                        raise RuntimeError(
                            "DSPy context not available. This may happen during server-side rendering. "
                            "Please ensure DSPyProvider is properly configured with API keys."
                        )
                    elif not env.models.get("default"):
                        raise RuntimeError(
                            "DSPy provider not fully initialized. This is normal during server-side rendering. "
                            "The application will work correctly once the client-side JavaScript loads."
                        )
                    else:
                        raise RuntimeError(
                            f"No language model configured. Available models: {list(env.models.keys())}. "
                            "Please check your DSPyProvider configuration."
                        )

                with ctx_mgr:
                    acall = getattr(module, "acall", None)
                    if acall is not None and inspect.iscoroutinefunction(acall):
                        return await acall(**inputs)
                    apredict = getattr(module, "apredict", None)
                    if apredict is not None and inspect.iscoroutinefunction(apredict):
                        return await apredict(**inputs)
                    predict = getattr(module, "predict", None)
                    if callable(predict):
                        return await asyncio.to_thread(predict, **inputs)
                    if callable(module):
                        return await asyncio.to_thread(module, **inputs)
                    raise AttributeError(
                        "DSPy module has no acall/apredict/predict/__call__ to invoke"
                    )

            result = await _call()

            if alive_ref["alive"] and task_ref["task"] is current_task:
                dispatch({"type": "success", "result": result})

        except Exception as e:
            if alive_ref["alive"] and task_ref["task"] is current_task:
                dispatch({"type": "error", "error": e})

    def _run(**inputs):
        prev = task_ref["task"]
        if prev is not None and not prev.done():
            prev.cancel()  # superseded: don't keep paying for a stale LLM call
        dispatch({"type": "start"})
        # Attributed to the turn that started the call, so turn.end waits for
        # it (and an interrupt cancels it).
        t = spawn(_do_call(inputs), owner=owner, name="dspy_call", turn=current_turn_id())
        task_ref["task"] = t

    run = hooks.use_callback(_run, deps=[id(module)])
    return (
        run,
        (state["result"], state["ver"]),
        state["status"] == "loading",
        state["error"],
    )
