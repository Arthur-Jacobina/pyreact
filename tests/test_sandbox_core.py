"""pyreact_sandbox core: spec, registry, LocalDriver lifecycle, gc (no Docker)."""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import pyreact_sandbox as ps
from pyreact_sandbox import (
    ExecResult,
    LocalDriver,
    SandboxError,
    SandboxNotFound,
    SandboxSetupError,
    SandboxSpec,
    SnapshotRef,
    Status,
)
from pyreact_sandbox.base import command_name

ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def local(tmp_path):
    return LocalDriver(root=tmp_path / "sbx")


def test_spec_coerce_merge_roundtrip_and_setup_key():
    spec = SandboxSpec.coerce({"image": "alpine", "setup": "echo hi", "env": {"A": "1"}})
    assert spec.setup == ("echo hi",) and spec.network is False and spec.workdir == "/workspace"
    merged = spec.with_(env={"B": "2"}, labels={"x": "y"})
    assert merged.env == {"A": "1", "B": "2"} and merged.labels == {"x": "y"}
    assert SandboxSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    # setup key: depends on image/setup/env, not on resources or labels
    assert spec.setup_key() == spec.with_(cpu=4, memory="2g", labels={"a": "b"}).setup_key()
    assert spec.setup_key() != spec.with_(setup=("echo other",)).setup_key()
    assert SandboxSpec.coerce(None) == SandboxSpec()


def test_exec_result_and_snapshot_ref_helpers():
    r = ExecResult(1, "out\n", "err")
    assert not r.ok and r.output == "out\nerr"
    with pytest.raises(SandboxError):
        r.check()
    assert ExecResult(0, "x").check().ok
    ref = SnapshotRef(driver="local", id="s1", sandbox_id="a", label="l", meta={"k": 1})
    assert SnapshotRef.from_dict(json.loads(json.dumps(ref.to_dict()))) == ref
    assert command_name("python3 -c 'x'") == "python3"
    assert command_name(["/usr/bin/ls", "-l"]) == "ls"


def test_registry_register_get_and_errors(local):
    assert "local" in ps.list_drivers()
    ps.register_driver("mine", lambda **kw: local)
    try:
        assert ps.get_driver("mine") is local
        assert ps.get_driver(local) is local
        with pytest.raises(ValueError):
            ps.register_driver("mine", lambda **kw: local)
        ps.register_driver("mine", lambda **kw: LocalDriver(**kw), replace=True)
        d = ps.get_driver("mine", root="/tmp/x")
        assert isinstance(d, LocalDriver) and str(d.root) == "/tmp/x"
    finally:
        ps.unregister_driver("mine")
    with pytest.raises(KeyError):
        ps.get_driver("nope")
    assert ps.get_driver("local") is ps.get_driver("local")  # shared instance


def test_local_lifecycle_exec_files_status(local):
    async def main():
        sb = await local.create(SandboxSpec(env={"GREETING": "hello"}))
        assert await sb.status() == Status.RUNNING
        r = await sb.exec("echo $GREETING && pwd")
        assert r.ok and r.stdout.splitlines()[0] == "hello"
        assert Path(r.stdout.splitlines()[1]).resolve() == sb.fs.resolve()
        r = await sb.exec(["python3", "-c", "import sys; print(sys.argv[1])", "argv"])
        assert r.stdout == "argv\n"
        await sb.write_file("/workspace/pkg/mod.py", "X = 41\n")
        await sb.write_file("data.bin", b"\x00\x01")
        assert await sb.read_file("pkg/mod.py") == b"X = 41\n"
        assert await sb.read_file("/workspace/data.bin") == b"\x00\x01"
        r = await sb.run_python("import pkg.mod as m; print(m.X + 1)", cwd="/workspace")
        assert r.stdout.strip() == "42", r
        r = await sb.exec("cat -", stdin="piped")
        assert r.stdout == "piped"
        r = await sb.exec("exit 3")
        assert r.exit_code == 3 and not r.ok
        with pytest.raises(SandboxError):
            await sb.read_file("/etc/passwd")  # confined file API
        with pytest.raises(SandboxError):
            await sb.write_file("../escape.txt", "x")
        # attach from "another process" (a new driver object on the same root)
        again = await LocalDriver(root=local.root).attach(sb.id)
        assert (await again.exec("cat pkg/mod.py")).stdout == "X = 41\n"
        assert sb.id in await local.list()
        await sb.destroy()
        assert await sb.status() == Status.DESTROYED
        with pytest.raises(SandboxNotFound):
            await local.attach(sb.id)

    run(main())


def test_local_timeout_kills_the_process_group(local):
    async def main():
        sb = await local.create(SandboxSpec())
        t0 = time.perf_counter()
        r = await sb.exec("sleep 5; echo never", timeout=0.5)
        assert r.timed_out and r.exit_code == 124 and "never" not in r.stdout
        assert time.perf_counter() - t0 < 3
        await sb.destroy()

    run(main())


def test_local_snapshot_restore_in_place_and_create_from_snapshot(local):
    async def main():
        sb = await local.create(SandboxSpec())
        await sb.write_file("state.txt", "v1")
        ref = await sb.snapshot("v1")
        assert ref.driver == "local" and ref.sandbox_id == sb.id and ref.label == "v1"
        await sb.write_file("state.txt", "v2")
        await sb.restore(ref)
        assert await sb.read_file("state.txt") == b"v1"
        clone = await local.create(sb.spec, snapshot=ref)
        assert clone.id != sb.id and await clone.read_file("state.txt") == b"v1"
        await local.delete_snapshot(ref)
        with pytest.raises(SandboxNotFound):
            await local.create(sb.spec, snapshot=ref)
        await sb.suspend()
        assert await sb.status() == Status.SUSPENDED
        with pytest.raises(SandboxError):
            await sb.exec("true")
        await sb.resume()
        assert (await sb.exec("true")).ok
        await sb.destroy()
        await clone.destroy()

    run(main())


def test_local_bootstrap_is_cached_by_setup_key(local, tmp_path):
    counter = tmp_path / "count"
    spec = SandboxSpec(setup=(f"echo x >> {counter}", "echo ready > marker"))

    async def main():
        a = await local.create(spec)
        b = await local.create(spec)
        assert await a.read_file("marker") == b"ready\n" == await b.read_file("marker")
        await a.destroy()
        await b.destroy()

    run(main())
    assert counter.read_text() == "x\n"  # setup ran once; 2nd boot copied the cache
    with pytest.raises(SandboxSetupError):
        run(local.create(SandboxSpec(setup=("exit 7",))))
    assert local.list_sync() == []  # failed bootstrap leaves nothing behind


def test_local_gc_orphans_expired_and_all(local):
    async def main():
        keep = await local.create(SandboxSpec(labels={"t": "1"}))
        expired = await local.create(SandboxSpec(ttl=0.01, labels={"t": "1"}))
        orphan = await local.create(SandboxSpec(labels={"t": "1"}))
        meta = json.loads((orphan.dir / "meta.json").read_text())
        meta["owner"] = f"{__import__('socket').gethostname()}:999999"
        (orphan.dir / "meta.json").write_text(json.dumps(meta))
        await keep.snapshot("s")
        return keep, expired, orphan

    keep, expired, orphan = run(main())
    time.sleep(0.05)
    assert run(expired.status()) == Status.STOPPED
    dry = ps.gc([local], dry_run=True)["local"]
    assert sorted(dry["sandboxes"]) == sorted([expired.id, orphan.id])
    assert local.list_sync() and len(local.list_sync()) == 3
    out = ps.gc([local])["local"]
    assert sorted(out["sandboxes"]) == sorted([expired.id, orphan.id])
    assert local.list_sync() == [keep.id]
    out = ps.gc([local], all=True, snapshots=True, labels={"t": "1"})["local"]
    assert out["sandboxes"] == [keep.id] and len(out["snapshots"]) == 1
    assert local.list_sync() == []


def test_interactive_process(local):
    sb = local.create_sync(SandboxSpec())
    proc = sb.open_process_sync(["python3", "-u", "-c",
                                 "import sys\nfor l in sys.stdin: print(l.strip().upper())"])
    proc.write(b"hello\n")
    assert proc.readline() == b"HELLO\n"
    proc.close()
    sb.destroy_sync()


def test_pyreact_never_imports_pyreact_sandbox():
    hits = [str(p) for p in (ROOT / "pyreact").rglob("*.py")
            if "pyreact_sandbox" in p.read_text(encoding="utf-8")]
    assert hits == []
    code = (
        "import sys, pyreact, pyreact.core, pyreact.boot, pyreact.tools\n"
        "assert not any(m.startswith('pyreact_sandbox') for m in sys.modules)\n"
        "import pyreact_sandbox\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('dspy', 'pyreact_durable')]\n"
        "assert not bad, bad\n"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0
    # the core layer is stdlib-only (no pyreact either)
    code2 = (
        "import sys, pyreact_sandbox, pyreact_sandbox.local\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('pyreact', 'dspy')]\n"
        "assert not bad, bad\n"
    )
    assert subprocess.run([sys.executable, "-c", code2], cwd=ROOT).returncode == 0
