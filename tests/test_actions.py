import numpy as np

from sim.actions import Action, run_session


def test_runner_runs_in_order_and_gates_on_requires():
    log = []

    def mk(name):
        def run(agent, market, rng, session):
            log.append(name)
        return run

    actions = [
        Action("a", mk("a")),
        Action("b", mk("b"), requires=("a",)),
        Action("c", mk("c"), requires=("missing",)),  # never eligible
    ]
    done = run_session(agent=None, market=None, rng=np.random.default_rng(0), actions=actions)
    assert log == ["a", "b"]        # c gated out, order preserved
    assert done == {"a", "b"}


def test_session_dict_threads_between_actions():
    def produce(agent, market, rng, session):
        session["candidates"] = [1, 2, 3]

    captured = {}

    def capture(agent, market, rng, session):
        captured.update(session)

    run_session(None, None, np.random.default_rng(0),
                [Action("produce", produce), Action("cap", capture, requires=("produce",))])
    assert captured["candidates"] == [1, 2, 3]


def test_default_funnel_names_and_order():
    from sim.actions import default_consumer_funnel
    names = [a.name for a in default_consumer_funnel()]
    assert names == ["visit", "list", "search", "view", "consideration", "buy"]


def test_default_funnel_preconditions():
    from sim.actions import default_consumer_funnel
    by = {a.name: a for a in default_consumer_funnel()}
    assert by["visit"].requires == ()
    assert by["list"].requires == ("visit",)
    assert by["search"].requires == ("visit",)
    assert by["view"].requires == ("search",)
    assert by["consideration"].requires == ("view",)
    assert by["buy"].requires == ("consideration",)
