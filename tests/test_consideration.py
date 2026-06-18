import numpy as np

from sim.consideration import quality_ranked_shortlist


class _L:
    def __init__(self, id, quality):
        self.id = id
        self.quality = quality


class _A:
    def __init__(self, engagement):
        self.engagement = engagement


def _listings(qualities):
    return [_L(i, q) for i, q in enumerate(qualities)]


def test_empty_viewed_returns_empty():
    out = quality_ranked_shortlist(_A(7.0), [], market=None, rng=np.random.default_rng(0))
    assert out == []


def test_ranks_by_quality_desc_with_id_tiebreak():
    viewed = [_L(0, 100.0), _L(1, 300.0), _L(2, 300.0), _L(3, 50.0)]
    # huge engagement -> mu huge -> k >> len(viewed) -> returns all, fully ordered
    out = quality_ranked_shortlist(_A(1e6), viewed, market=None, rng=np.random.default_rng(0))
    assert [l.id for l in out] == [1, 2, 0, 3]      # 300(id1), 300(id2), 100, 50


def test_caps_at_k_and_never_exceeds_viewed():
    viewed = _listings([float(i) for i in range(10)])
    rng = np.random.default_rng(0)
    lens = [len(quality_ranked_shortlist(_A(7.0), viewed, None, rng)) for _ in range(2000)]
    assert max(lens) <= len(viewed)
    assert 3.0 < float(np.mean(lens)) < 7.0         # mu ~= 5 at engagement 7


def test_higher_engagement_gives_bigger_shortlist():
    viewed = _listings([float(i) for i in range(10)])
    rng = np.random.default_rng(1)
    lo = np.mean([len(quality_ranked_shortlist(_A(2.0), viewed, None, rng)) for _ in range(2000)])
    hi = np.mean([len(quality_ranked_shortlist(_A(20.0), viewed, None, rng)) for _ in range(2000)])
    assert lo < hi


def test_deterministic_for_fixed_rng():
    viewed = _listings([float(i) for i in range(8)])
    a = quality_ranked_shortlist(_A(7.0), viewed, None, np.random.default_rng(5))
    b = quality_ranked_shortlist(_A(7.0), viewed, None, np.random.default_rng(5))
    assert [l.id for l in a] == [l.id for l in b]


def test_spec_default_curation_is_quality_ranked_shortlist():
    from datetime import datetime
    from sim.spec import MarketplaceSpec
    spec = MarketplaceSpec(start=datetime(2026, 1, 1))
    assert spec.curation is quality_ranked_shortlist


def test_act_consideration_uses_market_curation():
    # A custom curation strategy must be what populates session["consideration"].
    from sim.actions import _act_consideration

    sentinel = [object(), object()]

    class _M:
        def curation(self, agent, viewed, market, rng):
            assert market is self
            return sentinel

    session = {"viewed": [object(), object(), object()]}
    _act_consideration(agent=None, market=_M(), rng=None, session=session)
    assert session["consideration"] is sentinel


def test_market_exposes_curation_from_spec():
    import numpy as np
    import simpy
    from datetime import datetime
    from sim.engine import Clock, Market
    from sim.events import EventRecorder
    from sim.spec import MarketplaceSpec

    strat = lambda agent, viewed, market, rng: list(viewed)[:1]
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0, curation=strat)
    m = Market(env=simpy.Environment(), rng=np.random.default_rng(0),
               clock=Clock(spec.start), recorder=EventRecorder(), spec=spec)
    assert m.curation is strat


def _market_with_buyer(value_factor=1.0):
    import numpy as np
    import simpy
    from datetime import datetime
    from sim.engine import Clock, Market
    from sim.events import EventRecorder
    from sim.spec import MarketplaceSpec
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=simpy.Environment(), rng=np.random.default_rng(0),
               clock=Clock(spec.start), recorder=EventRecorder(), spec=spec)
    buyer = m.spawn_user()
    buyer.value_factor = value_factor
    return m, buyer


def test_explicit_buys_single_best_surplus_only():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)            # wtp == quality
    a = m.add_listing(quality=500.0, price=400.0, seller_id=999)    # surplus +100
    best = m.add_listing(quality=500.0, price=300.0, seller_id=999)  # surplus +200 (the pick)
    dear = m.add_listing(quality=500.0, price=600.0, seller_id=999)  # surplus -100
    session = {"consideration": [a, best, dear]}
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert best.transactions == 1 and not best.is_live          # only the best is bought
    assert a.transactions == 0 and dear.transactions == 0       # at most one purchase


def test_explicit_buys_nothing_when_all_underwater():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)
    x = m.add_listing(quality=500.0, price=600.0, seller_id=999)
    y = m.add_listing(quality=500.0, price=700.0, seller_id=999)
    session = {"consideration": [x, y]}
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert x.transactions == 0 and y.transactions == 0


def test_buy_excludes_negotiated_listings():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)
    claimed = m.add_listing(quality=500.0, price=300.0, seller_id=999)  # best, but negotiated
    other = m.add_listing(quality=500.0, price=450.0, seller_id=999)    # the fallback pick
    session = {"consideration": [claimed, other], "negotiated": {claimed.id}}
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert claimed.transactions == 0                            # excluded
    assert other.transactions == 1                              # best of the rest


def test_implicit_buys_at_most_the_single_best():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)
    buyer.engagement = 1e6                                      # p_buy ~ 1 -> coin flip passes
    a = m.add_listing(quality=400.0, price=10.0, seller_id=999)
    best = m.add_listing(quality=900.0, price=10.0, seller_id=999)   # highest surplus
    session = {"consideration": [a, best]}
    buy_action("implicit").run(buyer, m, m.rng, session)
    assert best.transactions == 1
    assert a.transactions == 0                                  # only one, and it's the best


def test_curation_strategy_is_exported():
    import sim
    from sim import quality_ranked_shortlist as exported
    assert "quality_ranked_shortlist" in sim.__all__
    assert exported is quality_ranked_shortlist


def test_explicit_buy_tie_breaks_to_lowest_id():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)
    # two identical surplus listings (same quality & price) -> tie -> lowest id wins
    first = m.add_listing(quality=500.0, price=300.0, seller_id=999)   # lower id
    second = m.add_listing(quality=500.0, price=300.0, seller_id=999)  # higher id
    assert first.id < second.id
    session = {"consideration": [second, first]}                      # order shouldn't matter
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert first.transactions == 1 and second.transactions == 0
