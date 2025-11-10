from pyreact.core.core import hooks
from pyreact.input.bus import InputBus, Event


def use_keystroke(on_submit=None):
    """
    Hook to handle keyboard input.
    
    Args:
        on_submit: Optional callback that receives the submitted text when Enter is pressed.
                  The callback is invoked in an effect, so it won't cause extra renders.
    
    Returns:
        A tuple of (current_text, submit_count) for observability if needed.
    """
    state, set_state = hooks.use_state({"text": "", "submit_ver": 0})
    bus: InputBus = hooks.get_service("input_bus", InputBus)

    # Bus handler - memoized to avoid recreating on every render
    def _handle(ev: Event):
        t = ev.get("type")
        v = ev.get("value", "") or ""
        if t == "submit":
            set_state(lambda s: {"text": v, "submit_ver": s["submit_ver"] + 1})

    handler = hooks.use_callback(_handle, deps=[])

    # Subscribe to input bus
    def _bus_effect():
        return bus.subscribe(handler)

    hooks.use_effect(_bus_effect, [handler])

    # Call the on_submit callback when submit_ver changes
    def _on_submit_effect():
        if (
            on_submit is not None and state["submit_ver"] > 0
        ):  # triggers only after at least one submit
            on_submit(state["text"])

    hooks.use_effect(_on_submit_effect, [state["submit_ver"]])

    return state["text"], state["submit_ver"]
