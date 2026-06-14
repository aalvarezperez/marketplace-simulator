from datetime import datetime

import numpy as np
import simpy

from sim.agents import p_churn
from sim.engine import Clock, Market, Marketplace
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def test_p_churn_higher_for_low_engagement():
    assert p_churn(0.5) > p_churn(50)


def test_listing_expires_after_ttl():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0,
                           listing_ttl_days=2.0)
    m = Market(env=env, rng=np.random.default_rng(0), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    seller = m.spawn_user()
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=seller.id)
    env.run(until=5.0)
    assert not listing.is_live
    assert any(e.event_type == "listing_expired" for e in m.recorder.events)


def test_churn_and_reactivation_in_full_run():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200,
                           until=60.0, seed=4, reactivation_scale_days=10.0)
    events = Marketplace.from_spec(spec).run()
    kinds = {e.event_type for e in events}
    assert "churned" in kinds
    assert "reactivated" in kinds


def test_full_lifecycle_reproducible():
    def run():
        spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=150,
                               until=30.0, seed=8)
        return [(e.event_type, e.actor_id) for e in Marketplace.from_spec(spec).run()]
    assert run() == run()
