from datetime import datetime

import numpy as np
import simpy

from sim.agents import User
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def _market(seed=0):
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    return Market(env=env, rng=np.random.default_rng(seed),
                  clock=Clock(spec.start), recorder=EventRecorder(), spec=spec)


def test_add_listing_is_live_and_listed():
    m = _market()
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=1)
    assert listing.is_live
    assert listing in m.live_listings()


def test_match_listings_orders_by_quality_desc():
    m = _market()
    m.add_listing(quality=10.0, price=1.0, seller_id=1)
    m.add_listing(quality=99.0, price=1.0, seller_id=1)
    top = m.match_listings(k=1)
    assert len(top) == 1 and top[0].quality == 99.0


def test_transact_decrements_stock_and_emits_event():
    m = _market()
    user = User(id=1, engagement=5.0, response_time=1.0)
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=2)
    m.transact(user, listing)
    assert listing.stock == 0
    assert not listing.is_live
    assert any(e.event_type == "transaction" for e in m.recorder.events)
