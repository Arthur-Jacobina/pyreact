"""dataset_from_events / split. No network."""

import asyncio
import json

import pytest
from dspy.utils.dummies import DummyLM

from integrations import DSPyProvider, use_agent_call
from integrations.feedback import TracedModule, send_feedback
from integrations.learn import dataset_from_events, split
from pyreact.core import EventBus, HookContext, component, runtime
from pyreact.core.events import Event
from pyreact.core.runtime import schedule_rerender
from pyreact.observability.log import EventLog

KEY = "ChainOfThought:RouteSig:App/Home#supervisor"


def _llm_pair(cid, message, name, turn):
    return [
        {"kind": "llm.start", "turn_id": turn,
         "payload": {"call_id": cid, "module": "cot(RouteSig)", "module_key": KEY,
                     "inputs": {"message": message}}},
        {"kind": "llm.end", "turn_id": turn,
         "payload": {"call_id": cid, "module": "cot(RouteSig)", "module_key": KEY,
                     "outputs": {"reasoning": "...", "name": name}}},
    ]


def _events():
    evs = []
    evs += _llm_pair("c1", "quais produtos?", "catalog", "t1")
    evs += _llm_pair("c2", "comprar e saber o frete", "order", "t2")
    evs += _llm_pair("c3", "ainda tem camisa?", "order", "t3")
    evs += _llm_pair("x1", "other module", "zzz", "t3")
    evs[-2]["payload"]["module_key"] = evs[-1]["payload"]["module_key"] = "Predict:Other:App#o"
    evs.append({"kind": "feedback", "payload": {"target": "t1", "score": 1.0, "text": "ok"}})
    evs.append({"kind": "feedback", "payload": {"target": "c2", "score": 0.0,
                                                "text": "frete -> shipping",
                                                "expected": {"name": "shipping",
                                                             "unrelated": "x"}}})
    return evs


def test_dataset_from_event_dicts_joins_feedback_and_corrections():
    ds = dataset_from_events(_events(), KEY)
    assert [e.message for e in ds] == ["quais produtos?", "comprar e saber o frete",
                                       "ainda tem camisa?"]
    assert [e.name for e in ds] == ["catalog", "shipping", "order"]  # correction wins
    assert "reasoning" not in ds[0].labels() and "unrelated" not in ds[1]
    assert set(ds[0].inputs().keys()) == {"message"}
    assert ds[0].feedback_score == 1.0 and ds[1].feedback_text == "frete -> shipping"
    assert ds[1].recorded_outputs["name"] == "order"
    # good runs + corrected ones only
    assert [e.call_id for e in dataset_from_events(_events(), KEY, min_score=1.0)] == ["c1", "c2"]
    assert len(dataset_from_events(_events(), KEY, require_feedback=True)) == 2
    # address by scope / component key or full key; other modules excluded
    assert len(dataset_from_events(_events(), "supervisor")) == 3
    assert len(dataset_from_events(_events(), ("ChainOfThought", "RouteSig",
                                               "App/Home#supervisor"))) == 3
    assert [e.message for e in dataset_from_events(_events(), "o")] == ["other module"]


def test_dataset_from_event_objects_jsonl_and_logs(tmp_path):
    evs = [Event(kind=d["kind"], payload=d["payload"], turn_id=d.get("turn_id"))
           for d in _events()]
    assert len(dataset_from_events(evs, KEY)) == 3
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e.to_dict()) for e in evs) + "\n")
    assert [e.name for e in dataset_from_events(str(path), KEY)] == ["catalog", "shipping",
                                                                     "order"]
    # durable-log style envelopes and an EventLog work too
    wrapped = [{"event": d} for d in _events()]
    assert len(dataset_from_events(wrapped, KEY)) == 3
    bus, log = EventBus(), EventLog()
    log.attach(bus)
    for e in evs:
        bus.emit(e)
    assert len(dataset_from_events(log, KEY)) == 3


def test_dataset_from_traced_module_calls():
    bus = EventBus()
    log = EventLog().attach(bus)

    class Router:
        def __call__(self, message):
            import dspy

            return dspy.Prediction(name="order")

    traced = TracedModule(Router(), "router", bus=bus)
    traced(message="frete?")
    send_feedback(traced.last_call_id, 0, "wrong", expected={"name": "shipping"}, bus=bus)
    (ex,) = dataset_from_events(log, "router")
    assert ex.message == "frete?" and ex.name == "shipping"


def test_dataset_from_a_live_app_run():
    events = []
    HookContext.get_service("event_bus", EventBus).subscribe(events.append)
    hs = {}

    @component
    def Clf():
        hs["call"] = use_agent_call("text -> label", arch="predict")
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=DummyLM([{"label": "pos"}, {"label": "neg"}]),
                             children=[Clf(key="clf")])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await runtime.run_until_idle()
        for t in ("good", "bad"):
            hs["call"].run(text=t)
            await runtime.run_until_idle()

    asyncio.run(main())
    ds = dataset_from_events(events, "clf")
    assert [(e.text, e.label) for e in ds] == [("good", "pos"), ("bad", "neg")]


def test_split_is_deterministic_and_non_empty():
    items = list(range(10))
    tr, va = split(items, val=0.3, seed=1)
    assert len(tr) == 7 and len(va) == 3 and sorted(tr + va) == items
    assert split(items, val=0.3, seed=1) == (tr, va)
    assert split(items, val=0.3, seed=2) != (tr, va)
    assert [len(x) for x in split([1, 2], val=0.01)] == [1, 1]
    assert split([1], val=0.5) == ([1], [])
    assert [len(x) for x in split(items, val=4)] == [6, 4]


def test_truncated_inputs_warn():
    evs = _llm_pair("c9", "x" * 10 + "…", "catalog", "t9")
    with pytest.warns(UserWarning, match="truncated"):
        dataset_from_events(evs, KEY)
