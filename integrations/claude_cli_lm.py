"""DSPy LM backed by the `claude -p` CLI.

Dev/test convenience: lets examples and end-to-end tests run real LLM calls
without an API key. Not for production (one subprocess per call, no caching).

    from integrations.claude_cli_lm import ClaudeCLILM
    lm = ClaudeCLILM("haiku")
"""

import asyncio
import subprocess
import time
from types import SimpleNamespace
import dspy

CALLS = []


def _flatten(messages, prompt):
    if messages is None:
        return None, prompt or ""
    sys_parts, convo = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
        if m["role"] == "system":
            sys_parts.append(c)
        else:
            convo.append(f"[{m['role'].upper()}]\n{c}")
    return "\n\n".join(sys_parts), "\n\n".join(convo)


def _wrap(text, model):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage={},
        model=model,
        _hidden_params={},
    )


class ClaudeCLILM(dspy.BaseLM):
    def __init__(self, model="haiku", **kw):
        super().__init__(model=f"claude-cli/{model}", cache=False, **kw)
        self.cli_model = model

    def _argv(self, system):
        a = [
            "claude",
            "-p",
            "--model",
            self.cli_model,
            "--tools",
            "",
            "--no-session-persistence",
            "--output-format",
            "text",
        ]
        if system:
            a += ["--system-prompt", system]
        return a

    def forward(self, prompt=None, messages=None, **kwargs):
        system, user = _flatten(messages, prompt)
        t0 = time.time()
        out = subprocess.run(
            self._argv(system), input=user, capture_output=True, text=True, timeout=180
        )
        CALLS.append({"t": time.time() - t0, "rc": out.returncode, "sync": True})
        if out.returncode:
            raise RuntimeError(out.stderr[-500:])
        return _wrap(out.stdout.strip(), self.model)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        system, user = _flatten(messages, prompt)
        t0 = time.time()
        p = await asyncio.create_subprocess_exec(
            *self._argv(system),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            so, se = await asyncio.wait_for(p.communicate(user.encode()), 180)
        except BaseException:
            # cancelled / timed out: don't leave orphaned CLI processes behind
            if p.returncode is None:
                p.kill()
            raise
        CALLS.append({"t": time.time() - t0, "rc": p.returncode, "sync": False})
        if p.returncode:
            raise RuntimeError(se.decode()[-500:])
        return _wrap(so.decode().strip(), self.model)
