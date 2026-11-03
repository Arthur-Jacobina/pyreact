"""``DockerDriver``: sandboxes as Docker containers, via the ``docker`` CLI.

No Python dependency (the CLI is the API surface; it also talks to remote
daemons via ``DOCKER_HOST`` / contexts). Every resource it creates is labeled
``pyreact.sandbox=1`` so :func:`pyreact_sandbox.gc` can find it.

- **create**: ``docker run -d`` of ``sleep <ttl|infinity>`` (with ``--init``)
  from the spec's bootstrapped image; ``--network none`` unless
  ``spec.network``; ``--cpus`` / ``--memory`` / ``--pids-limit``; hardened by
  default (``--cap-drop ALL``, ``no-new-privileges``). The container *name*
  is the sandbox id, so it is stable across restore-in-place.
- **bootstrap**: ``spec.setup`` runs once in a build container (network per
  ``spec.setup_network``), which is committed to
  ``pyreact-sandbox-tpl:<spec.setup_key()>`` (or ``:<spec.template>``). The
  next boot with the same spec reuses that image (no setup, ~1s).
- **exec**: ``docker exec`` (``sh -c`` for strings); timeouts are enforced
  inside the container with ``timeout -k`` (the process really dies), plus a
  host-side safety net.
- **snapshot**: ``docker commit`` to ``pyreact-sandbox:<id>-<label>``;
  **restore** recreates the container (same name) from it, or
  ``create(spec, snapshot=ref)`` boots a new one. Note: each restore+snapshot
  cycle adds a layer (Docker caps an image at 127 layers).
- **suspend** / **resume**: ``docker pause`` / ``unpause`` (default) or
  ``stop`` / ``start`` (``suspend_mode="stop"``).
- **destroy**: ``docker rm -f`` (``destroy_sync(snapshots=True)`` also
  removes its snapshot images).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import threading
import time
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

__all__ = ["DockerDriver", "DockerSandbox", "LABEL"]

LABEL = "pyreact.sandbox"
SNAPSHOT_REPO = "pyreact-sandbox"
TEMPLATE_REPO = "pyreact-sandbox-tpl"

_STATUS = {
    "running": Status.RUNNING,
    "restarting": Status.RUNNING,
    "paused": Status.SUSPENDED,
    "exited": Status.STOPPED,
    "dead": Status.STOPPED,
    "created": Status.STOPPED,
    "removing": Status.DESTROYED,
}


def _daemon_error(stderr: str) -> bool:
    s = stderr.lstrip()
    return s.startswith("Error response from daemon") or s.startswith("Error: No such")


def _parse_ts(text: str) -> float:
    try:
        base = text[:19]
        dt = _dt.datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


class DockerSandbox(BlockingSandbox):
    def __init__(self, driver: "DockerDriver", sandbox_id: str, spec: SandboxSpec) -> None:
        self.driver = driver
        self.id = sandbox_id
        self.spec = spec
        self._has_timeout: Optional[bool] = None

    # ------------------------------------------------------------------ exec
    def _exec_args(self, *, interactive: bool, env=None, cwd=None) -> List[str]:
        args = ["exec"]
        if interactive:
            args.append("-i")
        args += ["-w", cwd or self.spec.workdir]
        for k, v in dict(env or {}).items():
            args += ["-e", f"{k}={v}"]
        args.append(self.id)
        return args

    def _supports_timeout(self) -> bool:
        if self._has_timeout is None:
            r = self.driver._run(["exec", self.id, "sh", "-c", "command -v timeout"], check=False)
            self._has_timeout = r.returncode == 0
        return self._has_timeout

    def exec_sync(self, cmd, *, timeout=None, env=None, cwd=None, stdin=None) -> ExecResult:
        argv = ["sh", "-c", cmd] if isinstance(cmd, str) else [str(a) for a in cmd]
        if timeout is not None and self._supports_timeout():
            argv = ["timeout", "-k", "2", f"{max(float(timeout), 0.01):g}", *argv]
        data = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
        args = self._exec_args(interactive=data is not None, env=env, cwd=cwd) + argv
        t0 = time.perf_counter()
        try:
            r = self.driver._run(args, input=data, check=False,
                                 timeout=None if timeout is None else float(timeout) + 15)
        except subprocess.TimeoutExpired as exc:
            return ExecResult(124, (exc.stdout or b"").decode("utf-8", "replace"),
                              (exc.stderr or b"").decode("utf-8", "replace"),
                              time.perf_counter() - t0, timed_out=True)
        dur = time.perf_counter() - t0
        out = r.stdout.decode("utf-8", "replace")
        err = r.stderr.decode("utf-8", "replace")
        if r.returncode != 0 and _daemon_error(err):
            if "No such container" in err:
                raise SandboxNotFound(self.id)
            raise SandboxError(f"docker exec failed: {err.strip()}")
        timed_out = (timeout is not None and r.returncode in (124, 137)
                     and dur >= float(timeout) * 0.9)
        return ExecResult(r.returncode, out, err, dur, timed_out=timed_out)

    def read_file_sync(self, path) -> bytes:
        r = self.driver._run(self._exec_args(interactive=False) + ["cat", "--", path],
                             check=False)
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", "replace").strip()
            if _daemon_error(err):
                raise SandboxError(err)
            raise FileNotFoundError(f"{path}: {err}")
        return r.stdout

    def open_process_sync(self, argv: Sequence[str], *, env=None, cwd=None) -> PopenProcess:
        args = [self.driver.docker] + self._exec_args(interactive=True, env=env, cwd=cwd)
        popen = subprocess.Popen(args + [str(a) for a in argv], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return PopenProcess(popen)

    # ------------------------------------------------------------- lifecycle
    def status_sync(self) -> str:
        r = self.driver._run(["inspect", "-f", "{{.State.Status}}", self.id], check=False)
        if r.returncode != 0:
            return Status.DESTROYED
        return _STATUS.get(r.stdout.decode().strip(), Status.UNKNOWN)

    def snapshot_sync(self, label=None) -> SnapshotRef:
        label = label or time.strftime("%Y%m%d%H%M%S")
        tag = f"{SNAPSHOT_REPO}:{safe_tag(self.id, 60)}-{safe_tag(label, 60)}"
        paused = self.status_sync() == Status.SUSPENDED
        change = (f"LABEL {LABEL}=1 {LABEL}.kind=snapshot {LABEL}.source={self.id} "
                  f"{LABEL}.label={safe_tag(label)}")
        args = ["commit", f"--pause={'false' if paused else 'true'}", "-c", change, self.id, tag]
        r = self.driver._run(args, check=False, timeout=600)
        if r.returncode != 0:
            raise SandboxError(f"docker commit failed: {r.stderr.decode().strip()}")
        image_id = r.stdout.decode().strip()
        return SnapshotRef(driver=self.driver.name, id=tag, sandbox_id=self.id, label=label,
                           meta={"image_id": image_id})

    def restore_sync(self, ref) -> None:
        self.driver._require_image(ref.id)
        self.driver._run(["rm", "-f", self.id], check=False)
        self.driver._start_container(self.id, self.spec, ref.id, snapshot=ref)
        self._has_timeout = None

    def suspend_sync(self) -> None:
        cmd = "pause" if self.driver.suspend_mode == "pause" else "stop"
        if self.status_sync() == Status.SUSPENDED:
            return
        self.driver._run([cmd, self.id])

    def resume_sync(self) -> None:
        st = self.status_sync()
        if st == Status.SUSPENDED:
            self.driver._run(["unpause", self.id])
        elif st == Status.STOPPED:
            self.driver._run(["start", self.id])
        elif st == Status.DESTROYED:
            raise SandboxNotFound(self.id)

    def destroy_sync(self, snapshots: bool = False) -> None:
        self.driver._run(["rm", "-f", "-v", self.id], check=False)
        if snapshots:
            for ref in self.driver._images({f"{LABEL}.kind": "snapshot",
                                            f"{LABEL}.source": self.id}):
                self.driver._try(["rmi", ref])


class DockerDriver(BlockingDriver):
    """Docker containers via the ``docker`` CLI.

    - ``docker``: CLI path (default ``$PYREACT_SANDBOX_DOCKER`` or ``docker``).
    - ``hardened``: drop all capabilities and set ``no-new-privileges``.
    - ``pids_limit``: max processes per sandbox (fork bombs).
    - ``suspend_mode``: ``"pause"`` (freeze, keeps memory) or ``"stop"``.
    - ``runtime``: OCI runtime, e.g. ``"runsc"`` (gVisor) or ``"kata"``
      for stronger isolation than plain containers.
    - ``extra_args``: appended to ``docker run`` (``["--read-only"]``...).
    """

    name = "docker"

    def __init__(self, docker: Optional[str] = None, *, hardened: bool = True,
                 pids_limit: Optional[int] = 256, suspend_mode: str = "pause",
                 runtime: Optional[str] = None, extra_args: Sequence[str] = (),
                 setup_timeout: float = 900) -> None:
        if suspend_mode not in ("pause", "stop"):
            raise ValueError("suspend_mode must be 'pause' or 'stop'")
        self.docker = docker or os.environ.get("PYREACT_SANDBOX_DOCKER", "docker")
        self.hardened = hardened
        self.pids_limit = pids_limit
        self.suspend_mode = suspend_mode
        self.runtime = runtime
        self.extra_args = list(extra_args)
        self.setup_timeout = setup_timeout
        self._available: Optional[bool] = None
        self._build_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"<DockerDriver {self.docker}>"

    # ------------------------------------------------------------------ cli
    def _run(self, args: Sequence[str], *, input: Optional[bytes] = None, check: bool = True,
             timeout: Optional[float] = 120) -> subprocess.CompletedProcess:
        try:
            r = subprocess.run([self.docker, *args], input=input, capture_output=True,
                               timeout=timeout)
        except FileNotFoundError:
            raise SandboxError(f"docker CLI not found ({self.docker!r})") from None
        if check and r.returncode != 0:
            err = r.stderr.decode("utf-8", "replace").strip()
            if "No such container" in err:
                raise SandboxNotFound(err)
            raise SandboxError(f"docker {args[0]} failed: {err}")
        return r

    def _try(self, args: Sequence[str], timeout: float = 300) -> int:
        """Best-effort command (cleanup paths): exit code, -1 on timeout."""
        try:
            return self._run(args, check=False, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return -1

    def available(self) -> bool:
        if self._available is None:
            try:
                r = self._run(["version", "--format", "{{.Server.Version}}"], check=False,
                              timeout=10)
                self._available = r.returncode == 0
            except (SandboxError, subprocess.TimeoutExpired):
                self._available = False
        return self._available

    def _image_exists(self, ref: str) -> bool:
        return self._run(["image", "inspect", ref], check=False).returncode == 0

    def _require_image(self, ref: str) -> None:
        if not self._image_exists(ref):
            raise SandboxNotFound(f"image {ref} not found")

    def _ensure_image(self, ref: str) -> None:
        if not self._image_exists(ref):
            r = self._run(["pull", "-q", ref], check=False, timeout=900)
            if r.returncode != 0:
                raise SandboxError(f"docker pull {ref} failed: {r.stderr.decode().strip()}")

    def _images(self, labels: Mapping[str, str]) -> List[str]:
        args = ["images", "--format", "{{.Repository}}:{{.Tag}}"]
        for k, v in labels.items():
            args += ["--filter", f"label={k}={v}"]
        r = self._run(args, check=False)
        return [x for x in r.stdout.decode().split() if x and not x.endswith(":<none>")]

    # ----------------------------------------------------------- containers
    def _limits(self, spec: SandboxSpec) -> List[str]:
        args: List[str] = []
        if spec.cpu:
            args += ["--cpus", f"{spec.cpu:g}"]
        if spec.memory:
            args += ["--memory", str(spec.memory), "--memory-swap", str(spec.memory)]
        if self.pids_limit:
            args += ["--pids-limit", str(self.pids_limit)]
        if self.hardened:
            args += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
        if self.runtime:
            args += ["--runtime", self.runtime]
        return args

    @staticmethod
    def _network(network: Any) -> List[str]:
        if network is False or network is None:
            return ["--network", "none"]
        if network is True:
            return []
        return ["--network", str(network)]

    def _start_container(self, sid: str, spec: SandboxSpec, image: str, *,
                         snapshot: Optional[SnapshotRef] = None) -> None:
        labels = {
            LABEL: "1", f"{LABEL}.kind": "sandbox", f"{LABEL}.id": sid,
            f"{LABEL}.owner": owner_tag(), f"{LABEL}.spec": json.dumps(spec.to_dict()),
            **{str(k): str(v) for k, v in spec.labels.items()},
        }
        if spec.ttl:
            labels[f"{LABEL}.expires"] = f"{time.time() + spec.ttl:.0f}"
        if snapshot is not None:
            labels[f"{LABEL}.snapshot"] = snapshot.id
        args = ["run", "-d", "--init", "--name", sid, "-w", spec.workdir]
        for k, v in labels.items():
            args += ["--label", f"{k}={v}"]
        for k, v in spec.env.items():
            args += ["-e", f"{k}={v}"]
        args += self._network(spec.network) + self._limits(spec) + self.extra_args
        args += ["--entrypoint", "sleep", image, f"{spec.ttl:.0f}" if spec.ttl else "infinity"]
        self._run(args, timeout=300)

    def _bootstrap(self, spec: SandboxSpec) -> str:
        """The image to boot ``spec`` from (built + cached when it has setup)."""
        if not spec.image:
            raise SandboxError("DockerDriver needs spec.image")
        if not spec.setup:
            self._ensure_image(spec.image)
            return spec.image
        tag = f"{TEMPLATE_REPO}:{safe_tag(spec.template) if spec.template else spec.setup_key()}"
        if self._image_exists(tag):
            return tag
        with self._locks_lock:
            lock = self._build_locks.setdefault(tag, threading.Lock())
        with lock:
            if self._image_exists(tag):
                return tag
            self._ensure_image(spec.image)
            build = new_sandbox_id("pyrsbx-build")
            args = ["run", "-d", "--init", "--name", build, "-w", spec.workdir,
                    "--label", f"{LABEL}=1", "--label", f"{LABEL}.kind=build",
                    "--label", f"{LABEL}.owner={owner_tag()}"]
            for k, v in spec.labels.items():
                args += ["--label", f"{k}={v}"]
            for k, v in spec.env.items():
                args += ["-e", f"{k}={v}"]
            setup_net = (True if spec.setup_network and not isinstance(spec.network, str)
                         else spec.network)
            args += self._network(setup_net)
            args += ["--entrypoint", "sleep", spec.image, "infinity"]
            self._run(args, timeout=300)
            try:
                for cmd in spec.setup:
                    r = self._run(["exec", "-w", spec.workdir, build, "sh", "-c", cmd],
                                  check=False, timeout=self.setup_timeout)
                    if r.returncode != 0:
                        res = ExecResult(r.returncode, r.stdout.decode("utf-8", "replace"),
                                         r.stderr.decode("utf-8", "replace"))
                        raise SandboxSetupError(
                            f"setup command failed ({r.returncode}): {cmd!r}\n"
                            f"{res.output[-2000:]}", res)
                change = (f"LABEL {LABEL}=1 {LABEL}.kind=template "
                          f"{LABEL}.setup_key={spec.setup_key()}")
                self._run(["commit", "-c", change, build, tag], timeout=600)
            finally:
                self._run(["rm", "-f", build], check=False)
        return tag

    # -------------------------------------------------------------- driver
    def create_sync(self, spec, *, snapshot=None, sandbox_id=None) -> DockerSandbox:
        spec = SandboxSpec.coerce(spec)
        sid = sandbox_id or new_sandbox_id()
        if snapshot is not None:
            self._require_image(snapshot.id)
            image = snapshot.id
        else:
            image = self._bootstrap(spec)
        try:
            self._start_container(sid, spec, image, snapshot=snapshot)
        except BaseException:
            self._run(["rm", "-f", sid], check=False)
            raise
        return DockerSandbox(self, sid, spec)

    def build_template(self, spec) -> str:
        """Bootstrap ``spec`` now (e.g. at deploy time); returns the image tag."""
        return self._bootstrap(SandboxSpec.coerce(spec))

    def attach_sync(self, sandbox_id) -> DockerSandbox:
        r = self._run(["inspect", "-f", "{{json .Config.Labels}}", sandbox_id], check=False)
        if r.returncode != 0:
            raise SandboxNotFound(sandbox_id)
        labels = json.loads(r.stdout.decode() or "{}") or {}
        if labels.get(LABEL) != "1":
            raise SandboxNotFound(f"{sandbox_id} is not a pyreact sandbox")
        try:
            spec = SandboxSpec.from_dict(json.loads(labels.get(f"{LABEL}.spec", "{}")))
        except ValueError:
            spec = SandboxSpec()
        return DockerSandbox(self, sandbox_id, spec)

    def list_sync(self, labels=None) -> List[str]:
        args = ["ps", "-a", "--format", "{{.Names}}", "--filter", f"label={LABEL}.kind=sandbox"]
        for k, v in (labels or {}).items():
            args += ["--filter", f"label={k}={v}"]
        return self._run(args).stdout.decode().split()

    def delete_snapshot_sync(self, ref) -> None:
        self._try(["rmi", ref.id])

    # ------------------------------------------------------------------ gc
    def gc_sync(self, *, all: bool = False, snapshots: bool = False, templates: bool = False,
                labels: Optional[Mapping[str, str]] = None, older_than: Optional[float] = None,
                dry_run: bool = False, **_: Any) -> Dict[str, List[str]]:
        """Remove our containers that are expired (ttl), orphaned (owner
        process on this host is dead, unless paused = deliberately
        suspended), leftover build containers, or everything with
        ``all=True``. ``snapshots`` / ``templates`` also remove those images
        (images still used by a container are skipped)."""
        removed: Dict[str, List[str]] = {"containers": [], "images": []}
        if not self.available():
            return removed
        now = time.time()
        ids = self._run(["ps", "-aq", "--filter", f"label={LABEL}=1"]).stdout.decode().split()
        infos = []
        if ids:
            r = self._run(["inspect", *ids], check=False)
            try:
                infos = json.loads(r.stdout.decode() or "[]")
            except ValueError:
                infos = []
        for info in infos:
            lab = (info.get("Config") or {}).get("Labels") or {}
            if not labels_match(lab, labels):
                continue
            name = (info.get("Name") or "").lstrip("/") or info.get("Id", "")[:12]
            created = _parse_ts(info.get("Created", ""))
            if older_than is not None and now - created < older_than:
                continue
            state = (info.get("State") or {}).get("Status", "")
            exp = lab.get(f"{LABEL}.expires")
            expired = bool(exp) and now > float(exp)
            dead_owner = not owner_alive(lab.get(f"{LABEL}.owner"))
            build = lab.get(f"{LABEL}.kind") == "build"
            orphan = dead_owner and (state != "paused" or build)
            if all or expired or orphan:
                removed["containers"].append(name)
                if not dry_run:
                    self._try(["rm", "-f", "-v", info["Id"]])
        kinds = [k for k, flag in (("snapshot", snapshots), ("template", templates)) if flag]
        for kind in kinds:
            for ref in self._images({f"{LABEL}.kind": kind, **(labels or {})}):
                if dry_run:
                    removed["images"].append(ref)
                elif self._try(["rmi", ref]) == 0:
                    removed["images"].append(ref)
        if kinds and not dry_run:  # untagged leftovers of removed snapshot chains
            r = self._run(["images", "-q", "--filter", "dangling=true",
                           "--filter", f"label={LABEL}=1"], check=False)
            for img in set(r.stdout.decode().split()):
                self._try(["rmi", img])
        return removed
