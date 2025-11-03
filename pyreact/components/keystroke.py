from pyreact.core.core import component, hooks
from pyreact.input.bus import InputBus, Event


@component
def Keystroke(on_submit=None):
    state, set_state = hooks.use_state({"text": "", "submit_ver": 0})
    bus: InputBus = hooks.get_service("input_bus", InputBus)

    # Bus handler
    def _handle(ev: Event):
        t = ev.get("type")
        v = ev.get("value", "") or ""
        if t == "submit":
            set_state(lambda s: {"text": v, "submit_ver": s["submit_ver"] + 1})

    handler = hooks.use_callback(_handle, deps=[])

    def _bus_effect():
        return bus.subscribe(handler)

    hooks.use_effect(_bus_effect, [handler])

    def _on_submit_effect():
        if (
            on_submit is not None and state["submit_ver"] > 0
        ):  # on_submit: triggers only after at least one submit
            on_submit(state["text"])

    hooks.use_effect(_on_submit_effect, [state["submit_ver"]])

    return []
