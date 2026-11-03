"""In-sandbox Python REPL server (stdlib only; run as ``python3 -u -c <this>``).

Protocol: one JSON object per line. Host -> server::

    {"op": "exec", "code": str, "variables": {...}, "tools": [name...],
     "outputs": [{"name", "type"?}...]}
    {"op": "tool_result", "ok": bool, "value": any, "error": str}

Server -> host::

    {"op": "ready"}
    {"op": "tool_call", "name": str, "args": [...], "kwargs": {...}}
    {"op": "result", "output": str | None}
    {"op": "final", "output": {...}}                      # SUBMIT(...) was called
    {"op": "error", "type": str, "message": str, "syntax": bool}

The namespace persists across ``exec`` requests. User ``print`` output is
captured; the protocol uses a private duplicate of fd 1 and fd 1 itself is
pointed at stderr, so stray output (subprocesses) cannot corrupt it.
"""

import ast
import io
import json
import os
import sys
import traceback

_proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
_stdin = sys.stdin


def _send(msg):
    _proto.write(json.dumps(msg, default=repr) + "\n")
    _proto.flush()


class _Final(BaseException):
    pass


def _tool(name):
    def call(*args, **kwargs):
        _send({"op": "tool_call", "name": name, "args": list(args), "kwargs": kwargs})
        line = _stdin.readline()
        if not line:
            raise SystemExit(0)
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or f"tool {name} failed")
        return reply.get("value")

    call.__name__ = name
    return call


def _submit_for(outputs):
    names = [o["name"] for o in outputs or []]

    def SUBMIT(*args, **kwargs):  # noqa: N802
        if not names:
            if len(args) == 1 and not kwargs:
                raise _Final({"output": args[0]})
            raise _Final(dict(kwargs) if kwargs else {"output": list(args)})
        if len(args) > len(names):
            raise TypeError(f"SUBMIT() takes {len(names)} arguments ({', '.join(names)})")
        out = dict(zip(names, args))
        out.update(kwargs)
        missing = [n for n in names if n not in out]
        if missing:
            raise TypeError(f"SUBMIT() missing: {', '.join(missing)}")
        raise _Final(out)

    return SUBMIT


def _run(ns, code):
    tree = ast.parse(code, mode="exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<sandbox>", "exec"), ns)
    if last is not None:
        return eval(compile(last, "<sandbox>", "eval"), ns)
    return None


def main():
    ns = {"__name__": "__main__"}
    _send({"op": "ready"})
    for line in _stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req.get("op") != "exec":
            continue
        ns.update(req.get("variables") or {})
        for name in req.get("tools") or []:
            ns[name] = _tool(name)
        ns["SUBMIT"] = _submit_for(req.get("outputs"))
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            value = _run(ns, req.get("code") or "")
        except _Final as fin:
            sys.stdout = old
            _send({"op": "final", "output": fin.args[0]})
            continue
        except SyntaxError as exc:
            sys.stdout = old
            _send({"op": "error", "type": "SyntaxError", "message": str(exc), "syntax": True})
            continue
        except BaseException as exc:  # noqa: BLE001 - report everything to the host
            sys.stdout = old
            tb = traceback.format_exception_only(type(exc), exc)
            _send({"op": "error", "type": type(exc).__name__, "message": "".join(tb).strip(),
                   "stdout": buf.getvalue(), "syntax": False})
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                return
            continue
        sys.stdout = old
        out = buf.getvalue()
        if value is not None:
            out = out + (value if isinstance(value, str) else repr(value))
        _send({"op": "result", "output": out or None})


if __name__ == "__main__":
    main()
