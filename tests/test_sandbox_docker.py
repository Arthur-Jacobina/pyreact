"""DockerDriver integration tests (real containers). Skipped without Docker.

Run only these: ``uv run python -m pytest -m docker``. Everything created is
labeled with this run's id and removed with ``pyreact_sandbox.gc`` at the end.
"""

import asyncio
import json
import socket
import subprocess
import time
import uuid

import pytest

import pyreact_sandbox as ps
from pyreact_sandbox import SandboxError, SandboxNotFound, SandboxSpec, Status
from pyreact_sandbox.docker import DockerDriver

pytestmark = pytest.mark.docker

RUN = uuid.uuid4().hex[:8]
LABELS = {"pyreact.test": RUN}
IMAGE = "python:3.12-slim"


def _docker_ok() -> bool:
    try:
        return DockerDriver().available()
    except Exception:
        return False


if not _docker_ok():
    pytest.skip("Docker is not available", allow_module_level=True)


def run(coro):
    return asyncio.run(coro)


def inspect(name):
    out = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
    return json.loads(out.stdout)[0] if out.returncode == 0 else None


@pytest.fixture(scope="module")
def docker():
    d = DockerDriver()
    yield d
    report = ps.gc([d], all=True, snapshots=True, templates=True, labels=LABELS)
    left = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=pyreact.test={RUN}"],
                          capture_output=True, text=True).stdout.split()
    imgs = subprocess.run(["docker", "images", "-q", "--filter", f"label=pyreact.test={RUN}"],
                          capture_output=True, text=True).stdout.split()
    assert not left and not imgs, (report, left, imgs)


def spec(**kw):
    return SandboxSpec(image=IMAGE, labels=LABELS, **kw)


def test_lifecycle_limits_network_files_and_timeout(docker):
    async def main():
        sb = await docker.create(spec(env={"GREETING": "hi"}, cpu=0.5, memory="256m"))
        try:
            assert await sb.status() == Status.RUNNING
            info = inspect(sb.id)
            hc = info["HostConfig"]
            assert hc["NetworkMode"] == "none"
            assert hc["NanoCpus"] == 500_000_000 and hc["Memory"] == 256 * 1024 * 1024
            assert hc["PidsLimit"] == 256 and "ALL" in (hc["CapDrop"] or [])
            labels = info["Config"]["Labels"]
            assert labels["pyreact.sandbox"] == "1" and labels["pyreact.test"] == RUN

            r = await sb.exec("echo $GREETING; pwd")
            assert r.ok and r.stdout == "hi\n/workspace\n"
            r = await sb.exec(["python3", "-c", "import sys; print(sys.version_info[:2])"])
            assert r.stdout.strip() == "(3, 12)"
            await sb.write_file("/workspace/app/data.bin", b"\x00\xffbin")
            assert await sb.read_file("app/data.bin") == b"\x00\xffbin"
            await sb.write_file("main.py", "print(sum(range(10)))")
            assert (await sb.exec("python3 main.py")).stdout == "45\n"
            assert (await sb.run_python("print(6 * 7)")).stdout == "42\n"
            with pytest.raises(FileNotFoundError):
                await sb.read_file("nope.txt")

            net = await sb.run_python(
                "import socket\n"
                "try:\n socket.create_connection(('1.1.1.1', 53), 2); print('online')\n"
                "except OSError as e: print('offline', type(e).__name__)\n")
            assert net.stdout.startswith("offline"), net

            t0 = time.perf_counter()
            r = await sb.exec("sleep 30; echo never", timeout=1)
            assert r.timed_out and "never" not in r.stdout
            assert time.perf_counter() - t0 < 10
            procs = await sb.exec("cat /proc/[0-9]*/cmdline | tr '\\0' ' '")
            assert "sleep 30" not in procs.stdout  # killed inside the container

            again = await docker.attach(sb.id)
            assert again.spec.env == {"GREETING": "hi"}
            assert sb.id in await docker.list(LABELS)
        finally:
            await sb.destroy()
        assert await sb.status() == Status.DESTROYED
        assert inspect(sb.id) is None
        with pytest.raises(SandboxNotFound):
            await docker.attach(sb.id)

    run(main())


def test_snapshot_restore_suspend_resume(docker):
    async def main():
        sb = await docker.create(spec())
        try:
            await sb.write_file("state.txt", "v1")
            ref = await sb.snapshot("v1")
            assert ref.id == f"pyreact-sandbox:{sb.id}-v1" and ref.sandbox_id == sb.id
            await sb.write_file("state.txt", "v2")
            await sb.restore(ref)  # in place: same id, state rolled back
            assert await sb.read_file("state.txt") == b"v1"
            assert inspect(sb.id)["HostConfig"]["NetworkMode"] == "none"

            clone = await docker.create(spec(), snapshot=ref)
            try:
                assert await clone.read_file("state.txt") == b"v1"
            finally:
                await clone.destroy()

            await sb.suspend()
            assert await sb.status() == Status.SUSPENDED
            with pytest.raises(SandboxError):
                await sb.exec("true")
            ref2 = await sb.snapshot("while-paused")  # commit works on a paused container
            await sb.resume()
            assert (await sb.exec("cat state.txt")).stdout == "v1"
        finally:
            sb.destroy_sync(snapshots=True)
        imgs = subprocess.run(["docker", "images", "-q", ref.id, ref2.id], capture_output=True,
                              text=True).stdout.split()
        assert imgs == []

    run(main())


def test_bootstrap_setup_is_cached_by_spec_hash(docker):
    s = spec(setup=(f"echo {RUN} > /opt/marker", "pip --version > /opt/pip.txt"))

    async def main():
        t0 = time.perf_counter()
        a = await docker.create(s)
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = await docker.create(s)
        second = time.perf_counter() - t0
        try:
            tag = f"pyreact-sandbox-tpl:{s.setup_key()}"
            assert inspect(tag)["Config"]["Labels"]["pyreact.sandbox.kind"] == "template"
            for sb in (a, b):
                assert (await sb.read_file("/opt/marker")).decode().strip() == RUN
            return first, second
        finally:
            await a.destroy()
            await b.destroy()

    first, second = run(main())
    print(f"boot with setup: {first:.2f}s, cached: {second:.2f}s")
    assert second < first


def test_failed_setup_raises_and_leaves_nothing(docker):
    s = spec(setup=(f"echo {RUN}", "exit 9"))
    with pytest.raises(ps.SandboxSetupError) as exc:
        docker.create_sync(s)
    assert exc.value.result.exit_code == 9
    left = subprocess.run(["docker", "ps", "-aq", "--filter", "label=pyreact.sandbox.kind=build",
                           "--filter", f"label=pyreact.test={RUN}"], capture_output=True,
                          text=True).stdout.split()
    assert left == []


def test_gc_collects_orphans_and_expired_but_keeps_live(docker, monkeypatch):
    live = docker.create_sync(spec())
    expired = docker.create_sync(spec(ttl=1))
    import pyreact_sandbox.docker as dmod

    monkeypatch.setattr(dmod, "owner_tag", lambda: f"{socket.gethostname()}:999999")
    orphan = docker.create_sync(spec())
    monkeypatch.undo()
    time.sleep(2.5)  # `sleep 1` (ttl) exits
    assert expired.status_sync() == Status.STOPPED
    dry = ps.gc([docker], labels=LABELS, dry_run=True)["docker"]["containers"]
    assert sorted(dry) == sorted([expired.id, orphan.id])
    out = ps.gc([docker], labels=LABELS)["docker"]["containers"]
    assert sorted(out) == sorted([expired.id, orphan.id])
    assert live.status_sync() == Status.RUNNING
    live.destroy_sync()


def test_rlm_interpreter_runs_inside_the_container(docker):
    from dspy.primitives.code_interpreter import FinalOutput

    from pyreact_sandbox.interpreter import SandboxInterpreter

    sb = docker.create_sync(spec())
    try:
        with SandboxInterpreter(sb, tools={"shout": lambda text: text.upper()},
                                output_fields=[{"name": "answer"}], timeout=30) as it:
            assert it.execute("import platform; print(platform.system())") == "Linux\n"
            assert it.execute("x = shout(word)", {"word": "hi"}) is None
            assert it.execute("open('/workspace/x.txt', 'w').write(x); SUBMIT(x)") == \
                FinalOutput({"answer": "HI"})
        assert sb.read_file_sync("x.txt") == b"HI"
    finally:
        sb.destroy_sync()


def _shell_app(docker, on_close):
    from pyreact.core import component, hooks, use_turn
    from pyreact_sandbox.react import SandboxProvider, use_sandbox

    @component
    def Shell():
        sb = use_sandbox()
        turn = use_turn()
        bus = hooks.get_service("event_bus", lambda: None)

        async def act():
            r = await sb.exec(turn.text)
            bus.publish("message", {"text": r.stdout.strip(), "sender": "sandbox"})

        hooks.use_effect(lambda: act() if turn else None, [turn and turn.id])
        return []

    @component
    def App():
        return [SandboxProvider(key="sbx", driver=docker, spec=spec(), on_close=on_close,
                                children=[Shell(key="sh")])]

    return App


@pytest.mark.parametrize("on_close,mode", [
    ("destroy", "snapshot"),                              # suspend snapshots, never destroys
    ({"suspend": "suspend", "*": "destroy"}, "suspend"),  # docker pause, reattach
])
def test_root_suspend_hydrate_brings_the_container_back(docker, tmp_path, on_close, mode):
    from pyreact.boot import DirectorySnapshots, create_root, hydrate_root
    from pyreact_sandbox.react import wait_closed

    store = DirectorySnapshots(tmp_path / "state")
    root = create_root(_shell_app(docker, on_close), session_id="dk", store=store)
    root.send("sh -c 'echo persisted > /tmp/state.txt'", wait=True, timeout=120)
    snap = root.suspend()  # no participants: the provider dehydrates its sandboxes
    rec = snap["slices"]["App/SandboxProvider[sbx]"]["sandboxes"]["default"]
    if mode == "suspend":
        assert inspect(rec["sandbox_id"])["State"]["Paused"] is True
    else:
        assert rec["sandbox_id"] is None
        assert rec["snapshot"]["id"].startswith("pyreact-sandbox:")
    wait_closed(60)

    r = hydrate_root(_shell_app(docker, on_close), session_id="dk", store=store)
    try:
        r.send("cat /tmp/state.txt", wait=True, timeout=120)
        texts = [e.payload["text"] for e in r.events.recent(None, ["message"])]
        assert texts == ["persisted"]
        assert r.mismatches == [] and r.session.degraded == []
        restored = [e.payload for e in r.events.recent(None, ["sandbox.restored"])]
        assert restored[0]["mode"] == ("attach" if mode == "suspend" else "snapshot")
    finally:
        r.unmount()  # on_close: destroy
        wait_closed(60)


def test_root_hydrate_with_a_lost_snapshot_boots_fresh(docker, tmp_path):
    import warnings

    from pyreact.boot import create_root, hydrate_root
    from pyreact.core import HydrationWarning
    from pyreact_sandbox.react import wait_closed

    root = create_root(_shell_app(docker, "destroy"), session_id="lost")
    root.send("sh -c 'echo x > /tmp/state.txt'", wait=True, timeout=120)
    snap = root.suspend()
    wait_closed(60)
    ref = snap["slices"]["App/SandboxProvider[sbx]"]["sandboxes"]["default"]["snapshot"]
    subprocess.run(["docker", "rmi", "-f", ref["id"]], capture_output=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HydrationWarning)
        r = hydrate_root(_shell_app(docker, "destroy"), session_id="lost", snapshot=snap)
        try:
            r.send("sh -c 'cat /tmp/state.txt 2>/dev/null || echo fresh'", wait=True,
                   timeout=120)
            texts = [e.payload["text"] for e in r.events.recent(None, ["message"])]
            assert texts == ["fresh"]
            assert [m["reason"] for m in r.mismatches] == ["snapshot missing"]
            assert [d["reason"] for d in r.session.degraded] == ["snapshot missing"]
        finally:
            r.unmount()
            wait_closed(60)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deprecated_participant_suspend_resume_still_works(docker, tmp_path):
    from pyreact.boot import AppRunner, DirectoryManifestStore
    from pyreact_sandbox.react import wait_closed

    def make():
        return AppRunner(_shell_app(docker, "destroy"), default_session=False,
                         participants=[lambda s: ps.SandboxParticipant(s, mode="snapshot")],
                         manifest_store=DirectoryManifestStore(tmp_path / "m"))

    app = make()
    try:
        s = app.open_session("dk")
        s.invoke("sh -c 'echo persisted > /tmp/state.txt'", wait=True, timeout=120)
        m = s.suspend()
        [rec] = m["participants"]["sandbox"]["sandboxes"]
        assert rec["snapshot"]["id"].startswith("pyreact-sandbox:")
    finally:
        app.shutdown()
        wait_closed(60)

    app = make()
    try:
        texts = []
        app.on_session(open=lambda sess: sess.event_bus.subscribe(
            lambda ev: texts.append(ev.payload["text"]), ["message"]))
        r = app.resume("dk")
        r.invoke("cat /tmp/state.txt", wait=True, timeout=120)
        assert texts == ["persisted"] and r.degraded == []
        app.close_session("dk")
    finally:
        app.shutdown()
        wait_closed(60)