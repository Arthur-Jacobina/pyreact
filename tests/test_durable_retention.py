"""Session-log retention: Durable(on_close=...) and safe session ids."""

import pytest

from pyreact.boot import AppRunner
from pyreact.core import component
from pyreact_durable import Durable, archive_log, safe_session_id, use_persistent_state


def test_safe_session_id():
    assert safe_session_id("abc_DEF-123") == "abc_DEF-123"
    assert safe_session_id("default") == "default"
    assert safe_session_id("../etc/passwd") == "%2E%2E%2Fetc%2Fpasswd"
    assert safe_session_id(".hidden") == "%2Ehidden"
    assert safe_session_id("a b") == "a%20b"
    assert safe_session_id("") == "%00"
    assert "/" not in safe_session_id("x/" * 10)


def _app(logs, **durable_kw):
    @component
    def Who():
        use_persistent_state(0, key="n")[1](1)  # write something
        return []

    @component
    def App():
        return [Durable(key="d", log_factory=lambda sid: f"{logs}/{sid}.sqlite",
                        children=[Who(key="w")], **durable_kw)]

    return App


def test_session_close_reasons():
    @component
    def Empty():
        return []

    runner = AppRunner(Empty, default_session=False)
    try:
        a, b = runner.open_session("aaaa"), runner.open_session("bbbb")
        runner.close_session("aaaa")
        assert a.close_reason == "closed"
        assert runner.close_idle_sessions(0) == ["bbbb"] and b.close_reason == "idle"
        c = runner.open_session("cccc")
    finally:
        runner.shutdown()
    assert c.close_reason == "shutdown"


@pytest.mark.parametrize("policy", ["archive", "delete"])
def test_on_close_policy_applies_to_closed_sessions_only(tmp_path, policy):
    logs = tmp_path / "logs"
    runner = AppRunner(_app(logs, on_close=policy), default_session=False)
    try:
        runner.open_session("closed1").wait_idle(timeout=5)
        runner.open_session("idle01").wait_idle(timeout=5)
        runner.open_session("kept01").wait_idle(timeout=5)
        assert {p.name for p in logs.glob("*.sqlite")} == {
            "closed1.sqlite", "idle01.sqlite", "kept01.sqlite"}
        runner.close_session("closed1")
        runner.close_idle_sessions(0, keep=lambda s: s.id == "kept01")
    finally:
        runner.shutdown()  # shutdown is not a session close: kept01 stays
    left = {p.name for p in logs.iterdir() if p.is_file()}
    assert "kept01.sqlite" in left and not any(n.startswith(("closed1", "idle01")) for n in left)
    if policy == "archive":
        archived = {p.name for p in (logs / "archive").iterdir()}
        assert {"closed1.sqlite", "idle01.sqlite"} <= archived
        assert not any(n.endswith(".lock") for n in archived)
    else:
        assert not (logs / "archive").exists()


def test_on_close_callable_and_escaped_factory_ids(tmp_path):
    seen = []
    runner = AppRunner(_app(tmp_path, on_close=lambda path, sid: seen.append((path.name, sid))),
                       default_session=False)
    try:
        runner.open_session("../escape").wait_idle(timeout=5)
        runner.close_session("../escape")
    finally:
        runner.shutdown()
    assert seen == [("%2E%2E%2Fescape.sqlite", "../escape")]
    assert (tmp_path / "%2E%2E%2Fescape.sqlite").exists()
    assert not (tmp_path.parent / "escape.sqlite").exists()


def test_archive_log_never_overwrites(tmp_path):
    for i in range(2):
        (tmp_path / "s.jsonl").write_text(f"{i}\n")
        archive_log(tmp_path / "s.jsonl")
    assert sorted(p.name for p in (tmp_path / "archive").iterdir()) == ["s-1.jsonl", "s.jsonl"]


def test_on_close_rejects_unknown_policy(tmp_path):
    runner = AppRunner(_app(tmp_path, on_close="shred"), default_session=False)
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runner.open_session("x" * 8).wait_idle(timeout=5)
        assert not list(tmp_path.glob("*.sqlite"))  # render failed before opening
    finally:
        runner.shutdown()
