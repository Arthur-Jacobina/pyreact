"""Turn ids stay unique across restarts (id_factory / advance_to / Durable)."""

from pyreact.boot import bootstrap
from pyreact.core import component, use_turn
from pyreact.core.turns import TurnManager
from pyreact_durable import Durable, SessionLog


def test_id_factory_and_advance_to():
    tm = TurnManager(id_factory=lambda i: f"t-run1-{i}")
    assert tm.new_turn("a").id == "t-run1-1"
    tm.configure(id_prefix="s/")
    tm.advance_to(10)
    t = tm.new_turn("b")
    assert (t.id, t.index) == ("s/t-run1-10", 10)
    tm.advance_to(3)  # never goes back
    assert tm.new_turn("c").index == 11
    assert TurnManager().new_turn("x").id == "t1"  # default unchanged


def test_durable_continues_turn_numbering_after_restart(tmp_path):
    path = str(tmp_path / "turns.jsonl")

    @component
    def Agent():
        use_turn()
        return []

    @component
    def App():
        return [Durable(key="d", log=path, children=[Agent(key="a")])]

    ids = []
    for _run in range(2):
        app = bootstrap(App)
        try:
            ids += [app.invoke(t, wait=True, timeout=5) for t in ("x", "y")]
        finally:
            app.shutdown()
    assert ids == ["t1", "t2", "t3", "t4"]
    with SessionLog.open(path, readonly=True) as log:
        starts = [e["turn_id"] for e in log.events(["turn.start"])]
    assert starts == ids  # no journaled turn id is reused
