from sim.actions import Action, assemble_actions


def _base():
    return [Action("a", lambda *x: None),
            Action("b", lambda *x: None, requires=("a",)),
            Action("c", lambda *x: None, requires=("b",))]


def test_insert_before_target():
    extra = Action("x", lambda *x: None, before="c")
    names = [a.name for a in assemble_actions(_base(), [extra])]
    assert names == ["a", "b", "x", "c"]


def test_insert_after_target():
    extra = Action("x", lambda *x: None, after="a")
    names = [a.name for a in assemble_actions(_base(), [extra])]
    assert names == ["a", "x", "b", "c"]


def test_gate_adds_requires_to_target():
    extra = Action("x", lambda *x: None, before="c", mode="gate")
    out = {a.name: a for a in assemble_actions(_base(), [extra])}
    assert "x" in out["c"].requires


def test_branch_does_not_touch_target_requires():
    extra = Action("x", lambda *x: None, before="c", mode="branch")
    out = {a.name: a for a in assemble_actions(_base(), [extra])}
    assert "x" not in out["c"].requires


def test_unknown_hook_target_raises():
    extra = Action("x", lambda *x: None, before="nope")
    try:
        assemble_actions(_base(), [extra])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_base_unchanged_when_no_extras():
    assert [a.name for a in assemble_actions(_base(), [])] == ["a", "b", "c"]
