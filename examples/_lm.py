"""One switch for the language model every example uses.

``PYREACT_EXAMPLE_LM`` picks it:

- ``claude`` (default without an OpenAI key): the ``claude -p`` dev LM
  (``integrations/claude_cli_lm.py``), no API key needed.
- ``openai`` (default when ``OPENAI_API_KEY`` is set): ``gpt-4o-mini``.
- ``offline``: :class:`OfflineLM`, a deterministic fake that answers any
  DSPy signature with well-formed placeholder values. Examples run end to end
  in seconds with no network; answers are meaningless. The examples smoke test
  (``tests/test_examples_smoke.py``) uses it.

Usage in an example::

    from examples._lm import example_lm, example_models, OFFLINE

    lm = example_lm()                    # one LM
    models = example_models()            # {"default", "fast", "reasoning"}
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import dspy

__all__ = ["OFFLINE", "OfflineLM", "example_lm", "example_models", "lm_kind"]


def lm_kind() -> str:
    kind = os.getenv("PYREACT_EXAMPLE_LM", "").strip().lower()
    if kind:
        if kind not in ("claude", "openai", "offline"):
            raise ValueError(f"PYREACT_EXAMPLE_LM must be claude, openai or offline, got {kind!r}")
        return kind
    return "openai" if os.getenv("OPENAI_API_KEY") else "claude"


OFFLINE = lm_kind() == "offline"


def example_lm(model: str = "haiku") -> Any:
    """The LM for examples (see module docstring). ``model`` is the claude
    alias (``haiku`` / ``sonnet``); with OpenAI, ``haiku`` maps to
    gpt-4o-mini and anything else to gpt-4o."""
    kind = lm_kind()
    if kind == "offline":
        return OfflineLM()
    if kind == "openai":
        name = "openai/gpt-4o-mini" if model == "haiku" else "openai/gpt-4o"
        return dspy.LM(name, api_key=os.getenv("OPENAI_API_KEY"))
    from integrations.claude_cli_lm import ClaudeCLILM

    return ClaudeCLILM(model)


def example_models() -> Dict[str, Any]:
    """``{"default", "fast", "reasoning"}`` for ``DSPyProvider(models=...)``."""
    fast = example_lm("haiku")
    return {"default": fast, "fast": fast, "reasoning": fast}


# ---------------------------------------------------------------------------
# Offline fake LM
# ---------------------------------------------------------------------------
_FIELD_RE = re.compile(r"^\d+\. `(\w+)` \((.*)\):", re.M)


def _output_fields(system: str) -> List[Tuple[str, str]]:
    """``[(name, type)]`` from the adapter's "Your output fields are:" block."""
    if "Your output fields are:" not in system:
        return []
    block = system.split("Your output fields are:", 1)[1]
    block = re.split(r"\n(?:All interactions|In adhering)", block, maxsplit=1)[0]
    return _FIELD_RE.findall(block)


def _literal_options(type_str: str) -> Optional[List[str]]:
    m = re.match(r"Literal\[(.*)\]$", type_str.strip())
    if not m:
        return None
    return [o.strip().strip("'\"") for o in m.group(1).split(",") if o.strip()]


def _value(name: str, type_str: str, seed: str) -> Any:
    t = type_str.strip()
    options = _literal_options(t)
    if options:
        if "finish" in options:  # ReAct: end the loop immediately
            return "finish"
        digest = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
        return options[digest % len(options)]  # deterministic per input
    low = t.lower()
    if low.startswith(("list", "tuple", "set")):
        return []
    if low.startswith("dict"):
        return {}
    if low == "int":
        return 1
    if low == "float":
        return 1.0
    if low == "bool":
        return True
    return f"offline {name}"


def _render(fields: List[Tuple[str, str]], seed: str, chat: bool) -> str:
    values = {n: _value(n, t, seed) for n, t in fields}
    if not chat:
        return json.dumps(values, ensure_ascii=False)
    parts = []
    for n, v in values.items():
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        parts.append(f"[[ ## {n} ## ]]\n{text}")
    parts.append("[[ ## completed ## ]]")
    return "\n\n".join(parts)


class OfflineLM(dspy.BaseLM):
    """Answers any DSPy signature with placeholder values, instantly.

    Reads the output fields (and their types) from the adapter's system
    message and fills them: ``Literal`` -> one of the options (chosen
    deterministically from the input; ``finish`` for ReAct), numbers -> 1,
    lists/dicts -> empty, strings -> ``"offline <field>"``. Speaks both the
    chat-marker and the JSON adapter formats. For tests and smoke runs only.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(model="offline/fake", cache=False, **kw)
        self.calls = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls += 1
        messages = messages or [{"role": "user", "content": prompt or ""}]
        system = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        seed = str(messages[-1].get("content", ""))
        fields = _output_fields(system)
        if fields:
            chat = "[[ ## completed ## ]]" in system
            text = _render(fields, seed, chat)
        else:
            text = "offline answer"
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=self.model,
            _hidden_params={},
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)
