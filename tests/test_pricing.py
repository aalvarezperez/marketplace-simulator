import numpy as np

from sim.pricing import EndogenousPrice, NaiveMeanPriceModel, fit_price_model
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


def test_naive_fallback_when_no_listings():
    ep = EndogenousPrice(fallback_mean=500.0)
    price = ep.draw_with_context(np.random.default_rng(0),
                                 {"market": _M([]), "quality": 100.0})
    assert price == 500.0


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
