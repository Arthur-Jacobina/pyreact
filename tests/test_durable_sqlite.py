"""The SQLite backend stores the same session format (one row per JSON line)."""

import sqlite3
import subprocess
import sys

import pytest

from pyreact_durable import SessionLocked, SessionLog
from pyreact_durable.log import SQLiteSessionLog

from test_durable_react import _counter_app, run_app  # noqa: F401  (fixture)


def _rows(path):
    con = sqlite3.connect(path)
    try:
        return [r[0] for r in con.execute("SELECT line FROM lines ORDER BY seq")]
    finally:
        con.close()


def test_suffix_picks_backend(tmp_path):
    for name in ("s.sqlite", "s.sqlite3", "s.db"):
        log = SessionLog.open(tmp_path / name)
        assert isinstance(log, SQLiteSessionLog)
        log.close()
    log = SessionLog.open(tmp_path / "s.jsonl")
    assert type(log) is SessionLog
    log.close()


def test_append_reload_branch_and_labels(tmp_path):
    path = tmp_path / "s.sqlite"
    log = SessionLog.open(path)
    log.set_state("App", "n", 1)
    base = log.label("base")["targetId"]
    log.set_state("App", "n", 2)
    log.checkout("base")
    log.set_state("App", "n", 99)  # a second branch from "base"
    log.close()

    log = SessionLog.open(path)
    assert log.get_state("App", "n") == 99
    assert log.resolve("base") == base
    assert log.branch_points() == [base]
    log.checkout(log.children[base][0])  # back to the first branch
    log.close()
    assert SessionLog.open(path, readonly=True).get_state("App", "n") == 2

    rows = _rows(path)
    assert rows[0].startswith('{"type":"session"') or '"type": "session"' in rows[0]
    assert len(rows) == 1 + len(SessionLog.open(path, readonly=True).entries)


def test_checkpoint_equivalence(tmp_path):
    log = SessionLog.open(tmp_path / "s.db")
    for i in range(5):
        log.set_state("A", "i", i)
    before = log.state_at()
    log.checkpoint("compact")
    log.set_state("A", "j", 1)
    log.close()
    reopened = SessionLog.open(tmp_path / "s.db", readonly=True)
    assert reopened.state_at() == {"A": {**before["A"], "j": 1}}


def test_single_writer_and_concurrent_reader(tmp_path):
    path = tmp_path / "s.sqlite"
    writer = SessionLog.open(path)
    writer.set_state("A", "x", 1)
    with pytest.raises(SessionLocked):
        SessionLog.open(path)
    # A reader in another process sees committed rows while the writer is open.
    code = (
        "from pyreact_durable import SessionLog;"
        f"print(SessionLog.open({str(path)!r}, readonly=True).get_state('A','x'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "1", out.stderr
    writer.close()
    SessionLog.open(path).close()  # lock released


def test_fork_file_exports_and_imports_between_backends(tmp_path):
    src = SessionLog.open(tmp_path / "s.sqlite")
    src.set_state("A", "n", 1)
    src.label("one")
    src.set_state("A", "n", 2)
    exported = src.fork_file(tmp_path / "s.jsonl")
    assert type(exported) is SessionLog
    assert exported.get_state("A", "n") == 2 and "one" in exported.labels
    exported.close()
    imported = SessionLog.open(tmp_path / "s.jsonl").fork_file(tmp_path / "back.db")
    assert isinstance(imported, SQLiteSessionLog)
    assert imported.get_state("A", "n") == 2
    assert imported.header["parentSession"].endswith("s.jsonl")


def test_persistent_hooks_restore_with_sqlite(tmp_path, run_app):  # noqa: F811
    path = str(tmp_path / "app.sqlite")
    seen1, seen2 = [], []
    app = run_app(_counter_app(path, seen1))
    for v in ("2", "3"):
        app.invoke(v, wait=True, timeout=5)
    app.shutdown()

    app2 = run_app(_counter_app(path, seen2))
    app2.wait_idle(timeout=5)
    assert seen2[0] == (2, {"sum": 5})  # restored from SQLite on first render


def test_cli_reads_sqlite(tmp_path):
    path = tmp_path / "s.sqlite"
    log = SessionLog.open(path)
    log.set_state("App", "n", 7)
    log.close()
    out = subprocess.run(
        [sys.executable, "-m", "pyreact_durable", "state", str(path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "7" in out.stdout
