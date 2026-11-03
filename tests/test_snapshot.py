"""dehydrate_tree / Snapshot format / hydration mismatches."""

import asyncio
import json
import warnings

import pytest

from pyreact.core import (
    SNAPSHOT_FORMAT,
    EventBus,
    HookContext,
    SnapshotError,
    TurnManager,
    adehydrate_tree,
    check_snapshot,
    component,
    dehydrate_tree,
    hooks,
    runtime,
    use_dehydrate,
)
from pyreact.core.dehydrate import begin_hydration, finish_hydration
from pyreact.core.runtime import schedule_rerender
from pyreact.core.turns import Turn


def _app(setters, seen):
    @component
    def Notes(owner):
        start = use_dehydrate("notes", dehydrate=lambda: notes, initial=[])
        notes, set_notes = hooks.use_state(start)
        setters[owner] = set_notes
        seen[owner] = notes
        return []

    @component
    def App():
        return [Notes(key="alice", owner="alice"), Notes(key="bob", owner="bob")]

    return App


async def _mount(fn, slices=None, services=None):
    root = HookContext(fn.__name__, fn, services=services)
    if slices is not None:
        begin_hydration(root, slices)
    schedule_rerender(root)
    await runtime.run_until_idle()
    return root


def test_dehydrate_then_hydrate_round_trip():
    setters, seen = {}, {}
    App = _app(setters, seen)

    async def main():
        root = await _mount(App)
        setters["alice"](["buy milk", ("a", "tuple")])
        await runtime.run_until_idle()
        snap = dehydrate_tree(root, session_id="s1")
        root.unmount()
        seen.clear()
        root2 = await _mount(App, snap)
        return snap, finish_hydration(root2)

    snap, mismatches = asyncio.run(main())
    assert set(snap) == {"format", "session_id", "created_at", "slices", "turns", "versions"}
    assert snap["format"] == SNAPSHOT_FORMAT == "pyreact.snapshot/1"
    assert snap["session_id"] == "s1"
    assert snap["slices"] == {"App/Notes[alice]": {"notes": ["buy milk", ["a", "tuple"]]},
                              "App/Notes[bob]": {"notes": []}}
    assert snap["versions"]["snapshot"] == 1
    json.dumps(snap)
    assert seen == {"alice": ["buy milk", ["a", "tuple"]], "bob": []}
    assert mismatches == []
    assert check_snapshot(snap) is snap


def test_not_json_names_path_and_key():
    @component
    def App():
        use_dehydrate("handle", dehydrate=lambda: object())
        return []

    root = asyncio.run(_mount(App))
    with pytest.raises(SnapshotError, match=r"use_dehydrate\('handle'\) of App .*not JSON"):
        dehydrate_tree(root)


def test_failing_and_async_dehydrate():
    async def later():
        await asyncio.sleep(0)
        return {"ok": 1}

    @component
    def App():
        use_dehydrate("a", dehydrate=later)
        return []

    @component
    def Bad():
        use_dehydrate("b", dehydrate=lambda: 1 / 0)
        return []

    async def main():
        root = await _mount(App)
        with pytest.raises(SnapshotError, match="async dehydrate"):
            dehydrate_tree(root)
        snap = await adehydrate_tree(root)
        bad = await _mount(Bad)
        with pytest.raises(SnapshotError, match=r"use_dehydrate\('b'\) of Bad: dehydrate\(\) failed"):
            dehydrate_tree(bad)
        return snap

    assert asyncio.run(main())["slices"] == {"App": {"a": {"ok": 1}}}


def test_duplicate_paths_are_an_error():
    @component
    def Leaf():
        use_dehydrate("v", dehydrate=lambda: 1)
        return []

    @component
    def App():
        return [Leaf(key="same"), Leaf(key="same")]

    root = asyncio.run(_mount(App))
    with pytest.raises(SnapshotError, match="distinct keys"):
        dehydrate_tree(root)


def test_unclaimed_and_missing_are_published_and_warned():
    bus = EventBus()
    events = []
    bus.subscribe(lambda ev: events.append(ev.payload), ["hydration.*"])
    setters, seen = {}, {}
    App = _app(setters, seen)

    async def main():
        root = await _mount(App, {"App/Notes[alice]": {"notes": ["x"]},
                                  "App/Gone[g]": {"v": 1}},
                            services={"event_bus": bus})
        return finish_hydration(root)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mismatches = asyncio.run(main())
    assert [(m["kind"], m["path"], m["key"]) for m in mismatches] == [
        ("missing", "App/Notes[bob]", "notes"), ("unclaimed", "App/Gone[g]", "v")]
    assert [e["kind"] for e in events] == ["missing", "unclaimed"]
    assert len(caught) == 2 and seen["alice"] == ["x"]


def test_pending_async_hydrate_dehydrates_its_original_slice():
    gate = {}

    @component
    def App():
        async def hydrate(s):
            await gate["ev"].wait()
            return s

        use_dehydrate("v", dehydrate=lambda: "never called", hydrate=hydrate)
        return []

    async def main():
        gate["ev"] = asyncio.Event()
        root = HookContext("App", App)
        begin_hydration(root, {"App": {"v": {"kept": True}}})
        schedule_rerender(root)
        await asyncio.sleep(0)
        await runtime.run_renders()
        snap = dehydrate_tree(root)
        gate["ev"].set()
        await runtime.run_until_idle()
        return snap

    assert asyncio.run(main())["slices"] == {"App": {"v": {"kept": True}}}


def test_check_snapshot_errors():
    with pytest.raises(SnapshotError):
        check_snapshot([])
    with pytest.raises(SnapshotError, match="not a pyreact snapshot"):
        check_snapshot({"format": "pyreact.manifest/1"})
    with pytest.raises(SnapshotError, match="newer"):
        check_snapshot({"format": "pyreact.snapshot/2"})
    with pytest.raises(SnapshotError, match="slices"):
        check_snapshot({"format": "pyreact.snapshot/1", "slices": {"App": 3}})


def test_turn_manager_dehydrate_and_hydrate():
    tm = TurnManager()
    tm.id_prefix = "s/"
    cut = tm.new_turn("cut")
    queued = tm.new_turn("queued")
    tm._queue.append(queued)
    state = tm.dehydrate(carry=[cut])
    assert state["next_index"] == 3 and state["policy"] == "queue"
    assert [t["id"] for t in state["carry"]] == ["s/t1", "s/t2"]
    assert "resumed" not in state["carry"][0]
    tm2 = TurnManager()
    turns = tm2.hydrate(json.loads(json.dumps(state)))
    assert all(isinstance(t, Turn) and t.resumed for t in turns)
    assert [t.text for t in turns] == ["cut", "queued"] and tm2.next_index() == 3
