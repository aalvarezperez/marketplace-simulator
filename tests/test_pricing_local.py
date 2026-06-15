import numpy as np

from sim.pricing import default_pricing


class _L:
    def __init__(self, quality, price):
        self.quality = quality
        self.price = price
        self.is_live = True


class _M:
    def __init__(self, listings):
        self.listings = listings


def test_cold_start_prior_is_quality_anchored_and_positive():
    m = _M([])
    vals = [default_pricing(None, 500.0, m, np.random.default_rng(s)) for s in range(50)]
    assert all(v > 0 for v in vals)
    assert 250 < np.median(vals) < 1000          # centered near quality=500, with spread
    assert np.std(vals) > 0


def test_uses_median_of_k_nearest_quality_when_market_is_deep():
    listings = [_L(q, q) for q in range(100, 1300, 100)]   # 12 listings, price = quality
    listings.append(_L(500.0, 99999.0))                    # outlier ask
    m = _M(listings)
    price = default_pricing(None, 500.0, m, np.random.default_rng(0), k=10)
    assert price < 2000                                    # median ignores the outlier
    assert 300 < price < 800                               # ~ near quality 500


def test_price_increases_with_quality_in_deep_market():
    listings = [_L(q, q) for q in range(100, 1300, 100)]
    m = _M(listings)
    lo = default_pricing(None, 200.0, m, np.random.default_rng(0), k=6)
    hi = default_pricing(None, 1000.0, m, np.random.default_rng(0), k=6)
    assert hi > lo
