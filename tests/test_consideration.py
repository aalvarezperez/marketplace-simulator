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
