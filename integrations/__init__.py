"""DSPy hooks layer for pyreact.

Modules are component memory, calls are effects, LMs/adapters are context,
tools are capabilities, and every call is an event on the EventBus.
"""

from .architectures import (
    build_module,
    list_architectures,
    register_architecture,
)
from .dspy_integration import DSPyContext, DSPyEnv, DSPyProvider, use_dspy_env, use_dspy_module
from .hooks import (
    Call,
    StreamCall,
    call_tool,
    invoke_module,
    to_dspy_tool,
    use_agent,
    use_agent_call,
    use_call,
    use_chain_of_thought,
    use_module,
    use_predict,
    use_stream,
)
from .use_dspy import use_dspy_call
from .dspy_agent import use_dspy_agent

__all__ = [
    "build_module",
    "list_architectures",
    "register_architecture",
    "DSPyContext",
    "DSPyEnv",
    "DSPyProvider",
    "use_dspy_env",
    "use_dspy_module",
    "Call",
    "StreamCall",
    "call_tool",
    "invoke_module",
    "to_dspy_tool",
    "use_agent",
    "use_agent_call",
    "use_call",
    "use_chain_of_thought",
    "use_module",
    "use_predict",
    "use_stream",
    "use_dspy_call",
    "use_dspy_agent",
]
