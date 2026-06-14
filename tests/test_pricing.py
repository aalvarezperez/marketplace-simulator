from datetime import datetime

import numpy as np

from sim.pricing import EndogenousPrice, LinearPriceModel, fit_price_model
from sim.spec import Property


class _L:
    def __init__(self, quality, price):
        self.quality = quality
        self.price = price


class _M:
    def __init__(self, listings):
        self.listings = listings


def test_property_literal_backward_compatible():
    assert Property(7).draw(np.random.default_rng(0)) == 7
    # passing a context must not break literals / dists
    assert Property(7).draw(np.random.default_rng(0), context={"x": 1}) == 7


def test_fit_returns_none_without_price_variation():
    assert fit_price_model(_M([])) is None
    # prices identical -> nothing to learn -> None (so the caller uses the prior)
    assert fit_price_model(_M([_L(100, 500), _L(200, 500)])) is None


def test_quality_anchored_prior_when_uninformative():
    ep = EndogenousPrice(prior_sigma=0.4)
    lo = [ep.draw_with_context(np.random.default_rng(s),
                               {"market": _M([]), "quality": 100.0}) for s in range(30)]
    hi = [ep.draw_with_context(np.random.default_rng(s),
                               {"market": _M([]), "quality": 1000.0}) for s in range(30)]
    assert all(p > 0 for p in lo + hi)
    assert np.mean(hi) > np.mean(lo)   # prior is quality-anchored
    assert np.std(lo) > 0              # and carries spread (not a constant)


def test_endogenous_price_increases_with_quality():
    m = _M([_L(q, 2.0 * q) for q in (100, 200, 300, 400, 500)])  # price = 2*quality
    ep = EndogenousPrice()
    p_low = ep.draw_with_context(np.random.default_rng(0),
                                 {"market": m, "quality": 100.0})
    p_high = ep.draw_with_context(np.random.default_rng(0),
                                  {"market": m, "quality": 500.0})
    assert p_high > p_low


def test_bias_scales_price_down():
    m = _M([_L(q, 2.0 * q) for q in (100, 200, 300)])
    ep = EndogenousPrice()
    full = ep.draw_with_context(np.random.default_rng(0),
                                {"market": m, "quality": 200.0, "bias": 1.0})
    biased = ep.draw_with_context(np.random.default_rng(0),
                                  {"market": m, "quality": 200.0, "bias": 0.9})
    assert biased < full


def test_emergent_prices_track_quality_in_full_run():
    from sim.engine import Marketplace
    from sim.spec import MarketplaceSpec
    mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                                n_seed_users=300, until=5.0, seed=1))
    mkt.run()
    q = np.array([l.quality for l in mkt.market.listings], dtype=float)
    p = np.array([l.price for l in mkt.market.listings], dtype=float)
    assert p.std() > 0                      # prices did NOT collapse to a constant
    assert np.corrcoef(q, p)[0, 1] > 0.3    # price tracks quality
