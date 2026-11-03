"""use_imperative_handle + the ref= prop convention + the session refs registry."""

import pytest

from pyreact.boot import bootstrap
from pyreact.core import REFS_SERVICE, Ref, component, hooks, use_imperative_handle, use_on_turn


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


class Handle:
    def __init__(self, label):
        self.label = label


def _toggling_app(child):
    """App rendering ``child(label)`` until a turn says "hide"; a turn text
    other than "hide" becomes the label."""

    @component
    def App():
        label, set_label = hooks.use_state("a")
        use_on_turn(lambda t: set_label(t.text))
        return [] if label == "hide" else [child(label)]

    return App


def test_ref_object_set_after_mount_updated_on_deps_and_cleared_on_unmount(run_app):
    r = Ref()
    made = []

    @component
    def Child(label, ref=None):
        def make():
            made.append(label)
            return Handle(label)

        use_imperative_handle(ref, make, [label])
        return []

    app = run_app(_toggling_app(lambda label: Child(key="c", label=label, ref=r)))
    assert r.current.label == "a"
    first = r.current
    app.invoke("a", wait=True, timeout=5)  # same deps: same handle
    assert r.current is first and made == ["a"]
    app.invoke("b", wait=True, timeout=5)  # deps changed: new handle
    assert r.current.label == "b" and made == ["a", "b"]
    app.invoke("hide", wait=True, timeout=5)
    assert r.current is None


def test_parent_reads_child_handle_through_use_ref(run_app):
    seen = []

    @component
    def Child(ref=None):
        use_imperative_handle(ref, lambda: Handle("child"), [])
        return []

    @component
    def Parent():
        r = hooks.use_ref()
        use_on_turn(lambda t: seen.append(r.current.label))  # after the child mounted
        return [Child(key="c", ref=r)]

    app = run_app(Parent)
    app.invoke("go", wait=True, timeout=5)
    assert seen == ["child"]


def test_callback_ref_called_with_handle_then_none(run_app):
    calls = []

    @component
    def Child(label, ref=None):
        use_imperative_handle(ref, lambda: label, [label])
        return []

    app = run_app(_toggling_app(lambda label: Child(key="c", label=label, ref=calls.append)))
    app.invoke("b", wait=True, timeout=5)
    app.invoke("hide", wait=True, timeout=5)
    assert calls == ["a", None, "b", None]


def test_named_ref_registered_per_session_and_removed_on_unmount(run_app):
    @component
    def Child(label, ref=None):
        use_imperative_handle(ref, lambda: Handle(label), [label])
        return []

    app = run_app(_toggling_app(lambda label: Child(key="c", label=label, ref="coder")),
                  default_session=False)
    s1, s2 = app.open_session("one"), app.open_session("two")
    reg1, reg2 = s1.services[REFS_SERVICE], s2.services[REFS_SERVICE]
    assert reg1 is not reg2
    assert reg1["coder"].label == "a" and reg2["coder"] is not reg1["coder"]
    s1.invoke("b", wait=True, timeout=5)
    assert reg1["coder"].label == "b" and reg2["coder"].label == "a"
    s1.invoke("hide", wait=True, timeout=5)
    assert "coder" not in reg1 and "coder" in reg2


def test_name_kw_with_ref_object_and_no_deps_reruns_every_render(run_app):
    r = Ref()
    made = []

    @component
    def Child(label, ref=None):
        use_imperative_handle(ref, lambda: made.append(label) or Handle(label), name="h")
        return []

    app = run_app(_toggling_app(lambda label: Child(key="c", label=label, ref=r)))
    reg = app.default_session.services[REFS_SERVICE]
    assert reg["h"] is r.current
    app.invoke("a", wait=True, timeout=5)  # rerender, same label: deps=None reruns
    assert len(made) >= 2 and reg["h"] is r.current


def test_duplicate_name_warns(run_app):
    @component
    def Child(label):
        use_imperative_handle("same", lambda: label, [])
        return []

    @component
    def App():
        return [Child(key="a", label="a"), Child(key="b", label="b")]

    with pytest.warns(RuntimeWarning, match="already registered"):
        run_app(App)


def test_bad_ref_type_raises():
    from pyreact.core.hook import HookContext

    @component
    def Child():
        use_imperative_handle(42, lambda: 1, [])
        return []

    errors = []
    root = HookContext("Child", Child, services={})
    try:
        root.render()
    except TypeError as exc:
        errors.append(exc)
    assert errors and "ref must be" in str(errors[0])
