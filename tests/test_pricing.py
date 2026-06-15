from datetime import datetime

import numpy as np
from scipy.stats import spearmanr

from sim.engine import Marketplace
from sim.spec import MarketplaceSpec


def test_spec_default_pricing_is_callable():
    from sim.pricing import default_pricing
    s = MarketplaceSpec(start=datetime(2026, 1, 1))
    assert s.pricing is default_pricing


def test_prices_track_quality_without_inflation():
    mkt = Marketplace.from_spec(MarketplaceSpec(
        start=datetime(2026, 1, 1), n_seed_users=300, until=5.0, seed=1))
    mkt.run()
    ls = mkt.market.listings
    q = np.array([l.quality for l in ls], dtype=float)
    p = np.array([l.price for l in ls], dtype=float)
    ratio = np.median(p / q)
    assert 0.6 < ratio < 1.6            # NOT the old ~2.0 inflation
    # default_pricing uses median-of-k-neighbours: monotone but non-linear,
    # so Spearman rank correlation is the right measure (Pearson undershoots).
    assert spearmanr(q, p).statistic > 0.4


def test_default_marketplace_now_converts():
    mkt = Marketplace.from_spec(MarketplaceSpec(
        start=datetime(2026, 1, 1), n_seed_users=300, until=5.0, seed=1))
    events = mkt.run()
    n_tx = sum(1 for e in events if e.event_type == "transaction")
    assert n_tx > 0                      # direct-buy conversion no longer ~0


def test_custom_pricing_callable_is_used():
    mkt = Marketplace.from_spec(MarketplaceSpec(
        start=datetime(2026, 1, 1), n_seed_users=20, until=2.0, seed=1,
        pricing=lambda seller, quality, market, rng: 7.0))
    mkt.run()
    assert all(l.price == 7.0 for l in mkt.market.listings)
