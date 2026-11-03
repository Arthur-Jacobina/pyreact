"""The terminal REPL: ``:`` lines are terminal commands, everything else
(``/word`` included) is app input."""

import asyncio

from pyreact.boot import parse_command, read_terminal_and_invoke


class FakeApp:
    events = None
    is_running = True

    def __init__(self):
        self.inputs = []

    async def ainvoke(self, text, *, wait=False):
        self.inputs.append(text)

    def current_route(self):
        return {"path": "/", "query": {}, "fragment": ""}


def _run(monkeypatch, lines, **kw):
    feed = iter(lines)

    def fake_input(_prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    app = FakeApp()
    asyncio.run(read_terminal_and_invoke(app, **kw))
    return app


def test_parse_command():
    assert parse_command(":events llm 5") == ("events", "llm 5")
    assert parse_command("  :q ") == ("q", "")
    assert parse_command("/label base") is None
    assert parse_command("hello") is None
    assert parse_command(":") is None
    assert parse_command("/tree", (":", "/")) == ("tree", "")


def test_slash_lines_reach_the_app(monkeypatch, capsys):
    app = _run(monkeypatch, ["/label base", "hi", ":route", "/tree", ":q", "never"])
    assert app.inputs == ["/label base", "hi", "/tree"]
    out = capsys.readouterr().out
    assert "unknown command" not in out
    assert "Route" in out  # :route is still a terminal command


def test_unknown_colon_command_reaches_the_app_with_a_hint(monkeypatch, capsys):
    app = _run(monkeypatch, [":nosuch thing", ":"])
    assert app.inputs == [":nosuch thing"]
    assert ":help" in capsys.readouterr().out


def test_legacy_slash_prefix_opt_in(monkeypatch, capsys):
    app = _run(monkeypatch, ["/route", "/q", "x"], command_prefixes=(":", "/"))
    assert app.inputs == []
    assert "Route" in capsys.readouterr().out
