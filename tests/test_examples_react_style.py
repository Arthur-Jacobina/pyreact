"""Guard: examples look like React (ADR 0001, "Hooks instead of service lookups").

Examples are what users copy, so they use only the app-facing API: hooks
(``use_input``, ``use_on_turn``, ``use_event`` / ``use_emit``,
``use_navigate``, context, ``ref=`` + ``use_imperative_handle``) and roots.
They must not use the low-level service locator (``get_service(``), reach
into ``session.services`` from outside the tree, configure what the tree
declares again outside it (``participants=``), or subscribe to the event bus
by hand (``event_bus.subscribe(`` / ``events.subscribe(``): output is
``Message`` components, and devtools tracing goes through the one shared
helper, ``examples/_devtools.py``.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

FORBIDDEN = {
    "get_service(": re.compile(r"\bget_service\("),
    "session.services": re.compile(r"\bsession\.services\b"),
    "participants=": re.compile(r"\bparticipants\s*="),
    "event_bus.subscribe": re.compile(r"\b(?:event_bus|events)\.subscribe\("),
}

# label -> files allowed to use it (relative to the repository root)
ALLOWED = {
    "event_bus.subscribe": {"examples/_devtools.py"},  # the one devtools subscriber
}


def violations_in(text: str, name: str = "<text>"):
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for label, pattern in FORBIDDEN.items():
            if name in ALLOWED.get(label, ()):
                continue
            if pattern.search(line):
                out.append(f"{name}:{lineno}: {label}: {line.strip()}")
    return out


def _example_files():
    for path in sorted(EXAMPLES.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".md"):
            yield path


def test_scanner_finds_each_forbidden_pattern():
    sample = (
        'bus = hooks.get_service("event_bus", EventBus)\n'
        'h = session.services["x"]\n'
        "AppRunner(App, participants=[P])\n"
        "app.event_bus.subscribe(show, ['message'])\n"
        "root.events.subscribe(show, replay=True)\n"
        "emit = use_emit()  # fine\n"
        "use_event('tool.end', fn)  # fine\n"
        "s.services_count = 1  # fine\n"
    )
    found = violations_in(sample)
    assert [v.split(": ")[1] for v in found] == ["get_service(", "session.services",
                                                  "participants=", "event_bus.subscribe",
                                                  "event_bus.subscribe"]


def test_devtools_helper_may_subscribe():
    line = "root.events.subscribe(show, kinds, replay=True)\n"
    assert violations_in(line, "examples/_devtools.py") == []
    assert violations_in(line, "examples/tools.py") != []


def test_examples_do_not_use_service_lookups_or_participants():
    found = []
    for path in _example_files():
        rel = str(path.relative_to(ROOT))
        found += violations_in(path.read_text(encoding="utf-8"), rel)
    assert not found, "examples must use hooks / roots, not:\n" + "\n".join(found)
