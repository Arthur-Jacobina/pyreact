import json
import subprocess
import sys
import warnings

import pytest

from pyreact_durable.log import (
    FORMAT_VERSION,
    SessionLocked,
    SessionLog,
    SessionLogError,
)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "s.jsonl"


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_header_and_append_reload(path):
    with SessionLog.open(path, meta={"app": "t"}) as log:
        a = log.set_state("App", "n", 1)
        b = log.append("event", kind="message", payload={"text": "hi"}, turn_id="t1",
                       source="term")
        assert log.leaf == b["id"] and b["parentId"] == a["id"]
    lines = _lines(path)
    assert lines[0]["type"] == "session" and lines[0]["version"] == FORMAT_VERSION
    assert lines[0]["app"] == "t" and len(lines[0]["id"]) == 36
    assert all(len(e["id"]) == 8 and int(e["id"], 16) >= 0 for e in lines[1:])
    assert lines[1]["parentId"] is None
    with SessionLog.open(path) as log:
        assert log.leaf == b["id"]
        assert log.state_at() == {"App": {"n": 1}}
        assert [e["kind"] for e in log.events()] == ["message"]
        assert log.events("turn.*") == []


def test_one_line_per_entry_and_non_json_rejected(path):
    with SessionLog.open(path) as log:
        log.set_state("A", "k", {"nested": [1, 2]})
        with pytest.raises(SessionLogError, match="'A' / 'k'"):
            log.set_state("A", "k", object())
        with pytest.raises(SessionLogError):
            log.append("nope")
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.parametrize("tail", [b'{"type":"state","id":"dead', b"\x00\x00garbage\n"])
def test_crash_truncated_tail_is_ignored_and_repaired(path, tail):
    with SessionLog.open(path) as log:
        log.set_state("A", "n", 1)
        good = log.set_state("A", "n", 2)
    with open(path, "ab") as f:
        f.write(tail)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        log = SessionLog.open(path)
    assert any("garbled" in str(x.message) for x in w)
    assert log.recovered and log.recovered[0]["trailing"]
    assert log.leaf == good["id"] and log.state_at() == {"A": {"n": 2}}
    log.set_state("A", "n", 3)  # appends cleanly after repair
    log.close()
    with SessionLog.open(path) as log2:
        assert log2.recovered == [] and log2.state_at() == {"A": {"n": 3}}


def test_valid_last_line_without_newline_is_kept(path):
    with SessionLog.open(path) as log:
        e = log.set_state("A", "n", 1)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with SessionLog.open(path) as log:
        assert log.leaf == e["id"] and not log.recovered
        log.set_state("A", "n", 2)
    assert SessionLog.open(path, readonly=True).state_at() == {"A": {"n": 2}}


def test_single_writer_guard(path):
    log = SessionLog.open(path)
    try:
        with pytest.raises(SessionLocked):
            SessionLog.open(path)
        # another *process* is refused too
        code = (
            "import sys; from pyreact_durable.log import SessionLog, SessionLocked\n"
            f"try:\n SessionLog.open({str(path)!r})\nexcept SessionLocked:\n sys.exit(7)\n"
        )
        assert subprocess.run([sys.executable, "-c", code]).returncode == 7
        ro = SessionLog.open(path, readonly=True)  # readers never block
        with pytest.raises(SessionLogError):
            ro.set_state("A", "n", 1)
    finally:
        log.close()
    SessionLog.open(path).close()  # released on close


def test_branching_and_checkout_survive_reopen(path):
    with SessionLog.open(path) as log:
        log.set_state("A", "n", 1)
        base = log.label("base")["targetId"]
        log.set_state("A", "n", 2)
        main = log.label("main")["targetId"]
        assert log.checkout("base") == base
        alt = log.set_state("A", "n", 99)
        assert alt["parentId"] == base
        assert log.branch_points() == [base]
        log.checkout("main")
        assert log.state_at() == {"A": {"n": 2}}
    with SessionLog.open(path) as log:
        assert log.leaf == main
        assert log.state_at(alt["id"]) == {"A": {"n": 99}}
        assert [e["id"] for e in log.path_to(alt["id"])] == [base, alt["id"]]
        log.checkout(alt["id"][:5])  # unique prefix
    with SessionLog.open(path) as log:
        assert log.leaf == alt["id"]


def test_branch_by_explicit_parent_and_labels(path):
    with SessionLog.open(path) as log:
        r = log.set_state("A", "x", 0)
        log.set_state("A", "x", 1)
        side = log.append("custom", parent_id=r["id"], customType="note", data={"a": 1})
        assert log.leaf == side["id"]
        log.label("l1", r["id"])
        log.label("l1", side["id"])  # relabel moves it
        log.label("gone")
        log.unlabel("gone")
        assert log.labels == {"l1": side["id"]}
        assert log.resolve("l1") == side["id"]
        with pytest.raises(SessionLogError):
            log.resolve("missing")
    with SessionLog.open(path, readonly=True) as log:
        assert log.labels == {"l1": side["id"]}
        assert log.labels_of(side["id"]) == ["l1"]


def test_checkpoint_equivalence(tmp_path):
    def build(p, ckpt):
        log = SessionLog.open(p)
        for i in range(30):
            log.set_state(f"C{i % 3}", "v", i)
            if i == 10:
                log.set_state("C9", "gone", 1)
            if i == 12:
                log.delete_state("C9", "gone")
            if ckpt and i % 7 == 0:
                log.checkpoint(summary=f"at {i}")
        return log

    a, b = build(tmp_path / "a.jsonl", False), build(tmp_path / "b.jsonl", True)
    assert a.state_at() == b.state_at()
    assert any(e["type"] == "checkpoint" for e in b.entries)
    a.close()
    b.close()
    rb = SessionLog.open(tmp_path / "b.jsonl", readonly=True)
    assert rb.state_at() == SessionLog.open(tmp_path / "a.jsonl", readonly=True).state_at()


def test_auto_checkpoint(path):
    with SessionLog.open(path, auto_checkpoint=5) as log:
        for i in range(12):
            log.set_state("A", "n", i)
        assert sum(e["type"] == "checkpoint" for e in log.entries) == 2
        assert log.state_at() == {"A": {"n": 11}}


def test_fork_file_header(tmp_path):
    src = tmp_path / "src.jsonl"
    with SessionLog.open(src) as log:
        log.set_name("demo")
        log.set_state("A", "n", 1)
        at = log.label("here")["targetId"]
        log.set_state("A", "n", 2)
        with log.fork_file(tmp_path / "dst.jsonl", at="here") as new:
            assert new.leaf == at and new.state_at() == {"A": {"n": 1}}
            new.set_state("A", "n", 5)
        src_id = log.id
    h = _lines(tmp_path / "dst.jsonl")[0]
    assert h["parentSession"] == str(src) and h["parentSessionId"] == src_id
    assert h["forkedFrom"] == at and h["id"] != src_id
    with SessionLog.open(tmp_path / "dst.jsonl") as new:
        assert new.labels == {"here": at} and new.name == "demo"
        assert new.state_at() == {"A": {"n": 5}}
    with SessionLog.open(src, readonly=True) as log:
        assert log.state_at() == {"A": {"n": 2}}  # source untouched


def test_tree_structure(path):
    with SessionLog.open(path) as log:
        a = log.set_state("A", "n", 1)
        log.set_state("A", "n", 2)
        log.checkout(a["id"])
        c = log.set_state("A", "n", 3)
        roots = log.tree()
    assert len(roots) == 1 and roots[0]["entry"]["id"] == a["id"]
    kids = roots[0]["children"]
    assert len(kids) == 2 and kids[1]["is_leaf"] and kids[1]["entry"]["id"] == c["id"]
    assert kids[1]["on_path"] and not kids[0]["on_path"]


def test_migration_on_load(path):
    path.write_text(
        json.dumps({"type": "session", "version": 0, "id": "x", "timestamp": "t"}) + "\n"
        + json.dumps({"type": "kv", "id": "aaaaaaaa", "parentId": None, "timestamp": "t",
                      "k": "n", "v": 1}) + "\n"
    )

    def v0_to_v1(header, entries):
        out = []
        for e in entries:
            if e["type"] == "kv":
                e = {"type": "state", "id": e["id"], "parentId": e["parentId"],
                     "timestamp": e["timestamp"], "path": "App", "key": e["k"], "value": e["v"]}
            out.append(e)
        return header, out

    with SessionLog.open(path, migrations={0: v0_to_v1}) as log:
        assert log.migrated_from == 0 and log.state_at() == {"App": {"n": 1}}
        log.set_state("App", "n", 2)
    assert _lines(path)[0]["version"] == FORMAT_VERSION
    with SessionLog.open(path) as log:  # no migration needed anymore
        assert log.migrated_from is None and log.state_at() == {"App": {"n": 2}}
    path.write_text(json.dumps({"type": "session", "version": 0, "id": "y"}) + "\n")
    with pytest.raises(SessionLogError, match="no migration"):
        SessionLog.open(path, migrations={})


def test_newer_version_refused(path):
    path.write_text(json.dumps({"type": "session", "version": FORMAT_VERSION + 1}) + "\n")
    with pytest.raises(SessionLogError, match="newer"):
        SessionLog.open(path)


def test_cli_roundtrip(tmp_path, capsys):
    from pyreact_durable.__main__ import main

    p = tmp_path / "c.jsonl"
    with SessionLog.open(p) as log:
        log.set_state("App", "n", 1)
        log.append("event", kind="message", payload={"text": "hello"})
        first = log.leaf
        log.set_state("App", "n", 2)
    assert main(["label", str(p), "one", "--at", first]) == 0
    assert main(["checkout", str(p), "one"]) == 0
    assert main(["state", str(p)]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 2)[2]) == {"App": {"n": 1}}
    assert main(["events", str(p), "--kind", "message"]) == 0
    assert "hello" in capsys.readouterr().out
    assert main(["fork", str(p), str(tmp_path / "f.jsonl"), "--at", "one"]) == 0
    assert main(["show", str(p)]) == 0
    out = capsys.readouterr().out
    assert "[one]" in out and "<- leaf" in out
    assert main(["checkout", str(p), "nope"]) == 2


def test_render_tree_is_public(tmp_path):
    import pyreact_durable
    from pyreact_durable import render_tree
    from pyreact_durable.__main__ import render_tree as cli_render_tree

    assert cli_render_tree is render_tree  # the CLI uses the public one
    assert "render_tree" in pyreact_durable.__all__
    with SessionLog.open(tmp_path / "r.jsonl") as log:
        log.set_state("App/Chat[chat]", "n", 1)
        log.label("base")
        log.set_state("App/Chat[chat]", "n", 2)
        log.checkout("base")
        log.set_state("App/Chat[chat]", "n", 3)
        text = log.render(width=40)
        assert text == "\n".join(render_tree(log, width=40))
    lines = text.splitlines()
    assert lines[0].startswith("session ") and "1 branch point(s)" in lines[0]
    assert any("[base]" in ln and "⑂2" in ln for ln in lines)
    assert sum("<- leaf" in ln for ln in lines) == 1
