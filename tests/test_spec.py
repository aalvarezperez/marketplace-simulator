from datetime import datetime

from sim.spec import MarketplaceSpec, Property


def test_default_spec_constructs():
    s = MarketplaceSpec(start=datetime(2026, 1, 1))
    assert s.seed == 0
    assert s.n_seed_users > 0
    assert isinstance(s.engagement, Property)
    assert isinstance(s.listing_quality, Property)


def test_spec_overrides_and_wraps_literals():
    s = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=10, seed=5,
                        listing_price=500)
    assert s.n_seed_users == 10
    assert s.seed == 5
    # a bare literal is wrapped into a Property
    assert isinstance(s.listing_price, Property)
    import numpy as np
    assert s.listing_price.draw(np.random.default_rng(0)) == 500
