"""Guard: examples look like React (ADR 0001, "Hooks instead of service lookups").

Examples are what users copy, so they use only the app-facing API: hooks
(``use_input``, ``use_on_turn``, ``use_event`` / ``use_emit``,
``use_navigate``, context, ``ref=`` + ``use_imperative_handle``) and roots.
They must not use the low-level service locator (``get_service(``), reach
into ``session.services`` from outside the tree, or configure what the tree
declares again outside it (``participants=``).
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

FORBIDDEN = {
    "get_service(": re.compile(r"\bget_service\("),
    "session.services": re.compile(r"\bsession\.services\b"),
    "participants=": re.compile(r"\bparticipants\s*="),
}


def violations_in(text: str, name: str = "<text>"):
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for label, pattern in FORBIDDEN.items():
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
        "emit = use_emit()  # fine\n"
        "s.services_count = 1  # fine\n"
    )
    found = violations_in(sample)
    assert [v.split(": ")[1] for v in found] == ["get_service(", "session.services",
                                                  "participants="]


@pytest.mark.xfail(strict=False, reason="examples are migrated in a later PR")
def test_examples_do_not_use_service_lookups_or_participants():
    found = []
    for path in _example_files():
        rel = str(path.relative_to(ROOT))
        found += violations_in(path.read_text(encoding="utf-8"), rel)
    assert not found, "examples must use hooks / roots, not:\n" + "\n".join(found)
