"""Attribute names (one place, so the README reference stays honest).

``gen_ai.*`` follow the OpenTelemetry GenAI semantic conventions; ``pyreact.*``
are this library's own. Vendor-specific keys (``langfuse.*``) are added by
enrichers (see :mod:`pyreact_otel.exporters`), never here.
"""

from __future__ import annotations

# --- GenAI semantic conventions -------------------------------------------
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_SYSTEM = "gen_ai.system"  # deprecated alias of provider.name; still read by many backends
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"
ERROR_TYPE = "error.type"
SESSION_ID = "session.id"
USER_ID = "user.id"

OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"
OP_INVOKE_AGENT = "invoke_agent"

# --- pyreact -----------------------------------------------------------------
BOUNDARY_KIND = "pyreact.boundary.kind"
BOUNDARY_NAME = "pyreact.boundary.name"
BOUNDARY_ID = "pyreact.boundary.id"
BOUNDARY_PARENT_ID = "pyreact.boundary.parent_id"
COMPONENT = "pyreact.component"
SESSION = "pyreact.session_id"
TURN = "pyreact.turn_id"
ATTEMPT = "pyreact.attempt"
CALL_ID = "pyreact.call_id"
MODULE_KEY = "pyreact.module_key"
ARCH = "pyreact.arch"
LM_CALLS = "pyreact.lm_calls"
TOTAL_TOKENS = "pyreact.usage.total_tokens"
COST = "pyreact.cost"
CANCELLED = "pyreact.cancelled"
BLOCKED_REASON = "pyreact.blocked.reason"
BLOCKED_BY = "pyreact.blocked.by"
INPUTS = "pyreact.inputs"  # content: only with capture_content=True
OUTPUTS = "pyreact.outputs"  # content: only with capture_content=True
