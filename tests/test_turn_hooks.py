"""Handling the same input twice: use_on_turn and Message(trigger="turn")."""

import pytest

from pyreact.boot import bootstrap
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_on_turn, use_turn


@pytest.fixture
def run_app():
    apps = []

    def _start(fn, **kw):
        app = bootstrap(fn, **kw)
        apps.append(app)
        return app

    yield _start
    for app in apps:
        app.shutdown()


def test_use_on_turn_fires_once_per_turn_even_for_repeated_text(run_app):
    handled = []

    @component
    def Agent():
        n, set_n = hooks.use_state(0)  # unrelated rerenders must not refire

        def on_turn(turn):
            handled.append((turn.id, turn.text))
            set_n(lambda v: v + 1)

        use_on_turn(on_turn)
        return []

    app = run_app(Agent)
    ids = [app.invoke(t, wait=True, timeout=5) for t in ("yes", "yes", "no")]
    assert handled == list(zip(ids, ["yes", "yes", "no"]))


def test_use_on_turn_async_handler_is_part_of_the_turn(run_app):
    import asyncio

    done = []

    @component
    def Agent():
        async def on_turn(turn):
            await asyncio.sleep(0.05)
            done.append(turn.text)

        use_on_turn(on_turn)
        return []

    app = run_app(Agent)
    app.invoke("a", wait=True, timeout=5)
    assert done == ["a"]  # the turn waited for the handler
    app.invoke("a", wait=True, timeout=5)
    assert done == ["a", "a"]


def test_message_trigger_turn_publishes_once_per_turn(run_app):
    @component
    def Echo():
        turn = use_turn()
        if turn is None:
            return []
        return [
            Message(key="per-turn", text=f"got {turn.text}", trigger="turn", sender="assistant"),
            Message(key="on-change", text=f"seen {turn.text}", sender="assistant"),
        ]

    app = run_app(Echo)
    ids = [app.invoke(t, wait=True, timeout=5) for t in ("hi", "hi", "yo")]
    msgs = [(e.payload["text"], e.turn_id) for e in app.events.recent(None, ["message"])]
    assert [m for m in msgs if m[0].startswith("got")] == [
        ("got hi", ids[0]), ("got hi", ids[1]), ("got yo", ids[2])]
    # the default (trigger="change") dedupes equal text across turns
    assert [m[0] for m in msgs if m[0].startswith("seen")] == ["seen hi", "seen yo"]
