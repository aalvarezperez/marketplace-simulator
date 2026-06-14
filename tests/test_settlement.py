from datetime import datetime

import numpy as np
import simpy

from sim.engine import Clock, Market, Marketplace
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def test_unactioned_proposal_expires():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0,
                           proposal_expiry_days=2.0)
    m = Market(env=env, rng=np.random.default_rng(0), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    buyer, seller = m.spawn_user(), m.spawn_user()
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=seller.id)
    p = m.make_proposal(buyer, seller, listing, amount=45.0)  # starts an expiry timer
    # never routed/settled -> it must expire
    env.run(until=5.0)
    assert p.status == "expired"


def test_proposals_reach_paid_in_full_run():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200,
                           until=12.0, seed=5)
    events = Marketplace.from_spec(spec).run()
    kinds = {e.event_type for e in events}
    assert "accepted" in kinds
    assert "paid" in kinds


def test_stock_never_negative():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200,
                           until=12.0, seed=5)
    mkt = Marketplace.from_spec(spec)
    mkt.run()
    assert all(l.stock >= 0 for l in mkt.market.listings)


def test_settlement_runs_are_reproducible():
    def run():
        spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=150,
                               until=8.0, seed=9)
        return [(e.event_type, e.actor_id, e.entity_id)
                for e in Marketplace.from_spec(spec).run()]
    assert run() == run()
