from pyreact.core.core import component, hooks
from typing import Literal


@component
def Log(text: str, trigger: Literal["change", "mount"] = "change"):
    deps = [] if trigger == "mount" else [text]
    hooks.use_effect(lambda: print(text), deps)
    return []
