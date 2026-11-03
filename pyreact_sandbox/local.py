"""``LocalDriver``: a "sandbox" that is a temp directory + host subprocesses.

**NOT ISOLATED.** Commands run as your user, on your machine, with your
network, and can read or write anything you can. ``cpu`` / ``memory`` /
``network`` are ignored. Use it for tests, offline examples and development
only; use :class:`~pyreact_sandbox.docker.DockerDriver` (or a vendor) for
untrusted code (anything an LLM writes).

Layout (``root`` defaults to ``$PYREACT_SANDBOX_LOCAL_DIR`` or
``<tmp>/pyreact-sandbox-local``)::

    <root>/<sandbox id>/meta.json      spec, status, owner, expiry
    <root>/<sandbox id>/fs/            the sandbox workdir (exec cwd)
    <root>/_snapshots/<snapshot id>/   copies of fs/ (snapshot / restore)
    <root>/_templates/<setup key>/     bootstrapped fs/ (setup cache)

The spec's ``workdir`` (``/workspace``) maps to ``fs/``: file APIs accept
paths under it or relative paths, and refuse other absolute paths. Shell
commands are not confined: use relative paths in them. ``python3`` /
``python`` resolve to the current interpreter.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._util import labels_match, owner_alive, owner_tag, safe_tag
from .base import (
    BlockingDriver,
    BlockingSandbox,
    ExecResult,
    PopenProcess,
    SandboxError,
    SandboxNotFound,
    SandboxSetupError,
    SandboxSpec,
    SnapshotRef,
    Status,
    new_sandbox_id,
)

__all__ = ["LocalDriver", "LocalSandbox"]


def _default_root() -> Path:
    env = os.environ.get("PYREACT_SANDBOX_LOCAL_DIR")
    return Path(env) if env else Path(tempfile.gettempdir()) / "pyreact-sandbox-local"


class LocalSandbox(BlockingSandbox):
    def __init__(self, driver: "LocalDriver", sandbox_id: str, spec: SandboxSpec) -> None:
        self.driver = driver
        self.id = sandbox_id
        self.spec = spec
        self.dir = driver.root / sandbox_id
        self.fs = self.dir / "fs"

    # ------------------------------------------------------------- metadata
    def _meta(self) -> Dict[str, Any]:
        try:
            return json.loads((self.dir / "meta.json").read_text())
        except FileNotFoundError:
            raise SandboxNotFound(self.id) from None

    def _set_meta(self, **changes: Any) -> None:
        meta = {**self._meta(), **changes}
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, self.dir / "meta.json")

    def _check_running(self) -> None:
        st = self.status_sync()
        if st != Status.RUNNING:
            raise SandboxError(f"sandbox {self.id} is {st}")

    # ---------------------------------------------------------------- paths
    def host_path(self, path: str) -> Path:
        """Map a sandbox path (under ``spec.workdir`` or relative) to the host."""
        wd = self.spec.workdir.rstrip("/") or "/"
        p = str(path)
        if p == wd:
            rel = ""
        elif p.startswith(wd + "/"):
            rel = p[len(wd) + 1:]
        elif os.path.isabs(p):
            raise SandboxError(
                f"LocalDriver confines file paths to {wd} (got {p!r}); use a relative path")
        else:
            rel = p
        full = (self.fs / rel).resolve()
        if full != self.fs.resolve() and self.fs.resolve() not in full.parents:
            raise SandboxError(f"path escapes the sandbox: {p!r}")
        return full

    def _env(self, env: Optional[Mapping[str, str]]) -> Dict[str, str]:
        bindir = os.path.dirname(sys.executable)
        base = {
            "PATH": bindir + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.fs),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONUNBUFFERED": "1",
        }
        return {**base, **dict(self.spec.env), **dict(env or {})}

    def _argv(self, cmd: Any) -> List[str]:
        if isinstance(cmd, str):
            return ["/bin/sh", "-c", cmd]
        argv = [str(a) for a in cmd]
        if argv and argv[0] in ("python", "python3"):
            argv[0] = sys.executable
        return argv

    # ------------------------------------------------------------------ ops
    def exec_sync(self, cmd, *, timeout=None, env=None, cwd=None, stdin=None) -> ExecResult:
        self._check_running()
        workdir = self.host_path(cwd) if cwd else self.fs
        data = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            self._argv(cmd), cwd=workdir, env=self._env(env),
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        timed_out = False
        try:
            out, err = proc.communicate(input=data, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            out, err = proc.communicate()
        return ExecResult(
            exit_code=124 if timed_out else proc.returncode,
            stdout=out.decode("utf-8", "replace"), stderr=err.decode("utf-8", "replace"),
            duration=time.perf_counter() - t0, timed_out=timed_out,
        )

    def run_python_sync(self, code, *, timeout=None, env=None, cwd=None) -> ExecResult:
        return self.exec_sync([sys.executable, "-"], timeout=timeout, env=env, cwd=cwd,
                              stdin=code)

    def write_file_sync(self, path, data) -> None:
        self._check_running()
        p = self.host_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode("utf-8") if isinstance(data, str) else bytes(data))

    def read_file_sync(self, path) -> bytes:
        self._check_running()
        return self.host_path(path).read_bytes()

    def snapshot_sync(self, label=None) -> SnapshotRef:
        self._check_running()
        snap_id = f"{self.id}-{safe_tag(label or str(int(time.time() * 1000)))}"
        dest = self.driver.root / "_snapshots" / snap_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.fs, dest / "fs", symlinks=True)
        meta = {"sandbox_id": self.id, "label": label, "spec": self.spec.to_dict(),
                "created_at": time.time()}
        (dest / "meta.json").write_text(json.dumps(meta))
        return SnapshotRef(driver=self.driver.name, id=snap_id, sandbox_id=self.id, label=label)

    def restore_sync(self, ref) -> None:
        src = self.driver._snapshot_fs(ref)
        shutil.rmtree(self.fs, ignore_errors=True)
        shutil.copytree(src, self.fs, symlinks=True)
        self._set_meta(status=Status.RUNNING)

    def suspend_sync(self) -> None:
        self._meta()
        self._set_meta(status=Status.SUSPENDED)

    def resume_sync(self) -> None:
        self._meta()
        self._set_meta(status=Status.RUNNING)

    def destroy_sync(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def status_sync(self) -> str:
        try:
            meta = self._meta()
        except SandboxNotFound:
            return Status.DESTROYED
        exp = meta.get("expires_at")
        if exp and time.time() > exp:
            return Status.STOPPED
        return meta.get("status", Status.UNKNOWN)

    def open_process_sync(self, argv: Sequence[str], *, env=None, cwd=None) -> PopenProcess:
        self._check_running()
        workdir = self.host_path(cwd) if cwd else self.fs
        popen = subprocess.Popen(
            self._argv(list(argv)), cwd=workdir, env=self._env(env), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        return PopenProcess(popen)


class LocalDriver(BlockingDriver):
    """Temp-dir sandboxes on the host. **NOT ISOLATED** (see module docs)."""

    name = "local"

    def __init__(self, root: Optional[os.PathLike] = None) -> None:
        self.root = Path(root) if root is not None else _default_root()

    def __repr__(self) -> str:
        return f"<LocalDriver {self.root}>"

    def _snapshot_fs(self, ref: SnapshotRef) -> Path:
        src = self.root / "_snapshots" / ref.id / "fs"
        if not src.is_dir():
            raise SandboxNotFound(f"snapshot {ref.id} not found")
        return src

    def create_sync(self, spec, *, snapshot=None, sandbox_id=None) -> LocalSandbox:
        spec = SandboxSpec.coerce(spec)
        sid = sandbox_id or new_sandbox_id("local")
        sb = LocalSandbox(self, sid, spec)
        if sb.dir.exists():
            raise SandboxError(f"sandbox {sid} already exists")
        sb.dir.mkdir(parents=True)
        meta = {"id": sid, "spec": spec.to_dict(), "status": Status.RUNNING,
                "owner": owner_tag(), "created_at": time.time(),
                "expires_at": time.time() + spec.ttl if spec.ttl else None,
                "labels": {"pyreact.sandbox": "1", **dict(spec.labels)}}
        (sb.dir / "meta.json").write_text(json.dumps(meta))
        try:
            if snapshot is not None:
                shutil.copytree(self._snapshot_fs(snapshot), sb.fs, symlinks=True)
            else:
                self._bootstrap(sb)
        except BaseException:
            sb.destroy_sync()
            raise
        return sb

    def _bootstrap(self, sb: LocalSandbox) -> None:
        spec = sb.spec
        if not spec.setup:
            sb.fs.mkdir(parents=True)
            return
        cache = self.root / "_templates" / safe_tag(spec.template or spec.setup_key())
        if (cache / "fs").is_dir():
            shutil.copytree(cache / "fs", sb.fs, symlinks=True)
            return
        sb.fs.mkdir(parents=True)
        for cmd in spec.setup:
            r = sb.exec_sync(cmd)
            if not r.ok:
                raise SandboxSetupError(f"setup command failed: {cmd!r}\n{r.output[-2000:]}", r)
        tmp = cache.with_name(cache.name + f".tmp{os.getpid()}")
        shutil.copytree(sb.fs, tmp / "fs", symlinks=True)
        try:
            os.replace(tmp, cache)
        except OSError:  # another process won the race
            shutil.rmtree(tmp, ignore_errors=True)

    def attach_sync(self, sandbox_id) -> LocalSandbox:
        try:
            meta = json.loads((self.root / sandbox_id / "meta.json").read_text())
        except (FileNotFoundError, NotADirectoryError):
            raise SandboxNotFound(sandbox_id) from None
        return LocalSandbox(self, sandbox_id, SandboxSpec.from_dict(meta["spec"]))

    def _metas(self) -> List[Dict[str, Any]]:
        out = []
        if not self.root.is_dir():
            return out
        for d in self.root.iterdir():
            if d.name.startswith("_") or not d.is_dir():
                continue
            try:
                out.append(json.loads((d / "meta.json").read_text()))
            except (OSError, ValueError):
                out.append({"id": d.name, "labels": {}, "owner": None, "broken": True})
        return out

    def list_sync(self, labels=None) -> List[str]:
        return [m["id"] for m in self._metas() if labels_match(m.get("labels", {}), labels)]

    def delete_snapshot_sync(self, ref) -> None:
        shutil.rmtree(self.root / "_snapshots" / ref.id, ignore_errors=True)

    def gc_sync(self, *, all: bool = False, snapshots: bool = False, templates: bool = False,
                labels: Optional[Mapping[str, str]] = None, older_than: Optional[float] = None,
                dry_run: bool = False, **_: Any) -> Dict[str, List[str]]:
        now = time.time()
        removed: Dict[str, List[str]] = {"sandboxes": [], "snapshots": [], "templates": []}
        for m in self._metas():
            if not labels_match(m.get("labels", {}), labels):
                continue
            if older_than is not None and now - m.get("created_at", 0) < older_than:
                continue
            expired = bool(m.get("expires_at")) and now > m["expires_at"]
            orphan = not owner_alive(m.get("owner")) and m.get("status") != Status.SUSPENDED
            if all or expired or orphan or m.get("broken"):
                removed["sandboxes"].append(m["id"])
                if not dry_run:
                    shutil.rmtree(self.root / m["id"], ignore_errors=True)
        for kind, flag in (("snapshots", snapshots), ("templates", templates)):
            base = self.root / f"_{kind}"
            if not flag or not base.is_dir():
                continue
            for d in base.iterdir():
                if kind == "snapshots" and labels:
                    try:
                        spec = json.loads((d / "meta.json").read_text()).get("spec", {})
                    except (OSError, ValueError):
                        spec = {}
                    if not labels_match(spec.get("labels", {}), labels):
                        continue
                removed[kind].append(d.name)
                if not dry_run:
                    shutil.rmtree(d, ignore_errors=True)
        return removed
