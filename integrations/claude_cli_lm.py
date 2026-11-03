"""DSPy LM backed by the `claude -p` CLI.

Dev/test convenience: lets examples and end-to-end tests run real LLM calls
without an API key. Not for production (one subprocess per call, no caching).

    from integrations.claude_cli_lm import ClaudeCLILM
    lm = ClaudeCLILM("haiku")

Usage: the CLI reports no token counts, so by default responses carry an
APPROXIMATE char-based usage dict (``"estimated": True``); pass
``approx_usage=False`` for empty usage.
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


def _estimate(text):
    # ~4 chars/token. APPROXIMATE: the CLI's text output reports no usage.
    return (len(text) + 3) // 4 if text else 0


def estimate_usage(system, user, output):
    """Char-based usage estimate, flagged with ``"estimated": True``."""
    p = _estimate(system or "") + _estimate(user or "")
    c = _estimate(output or "")
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "estimated": True,
    }


def _wrap(text, model, usage=None):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=dict(usage or {}),
        model=model,
        _hidden_params={},
    )


class ClaudeCLILM(dspy.BaseLM):
    def __init__(self, model="haiku", *, approx_usage=True, **kw):
        super().__init__(model=f"claude-cli/{model}", cache=False, **kw)
        self.cli_model = model
        self.approx_usage = approx_usage

    def _usage(self, system, user, output):
        return estimate_usage(system, user, output) if self.approx_usage else {}

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
        text = out.stdout.strip()
        return _wrap(text, self.model, self._usage(system, user, text))

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
        text = so.decode().strip()
        return _wrap(text, self.model, self._usage(system, user, text))
