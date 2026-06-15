from datetime import datetime

import numpy as np
import simpy

from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec
from sim.willingness import default_willingness


class _L:
    def __init__(self, quality, price):
        self.quality = quality
        self.price = price


class _A:
    def __init__(self, value_factor):
        self.value_factor = value_factor


def test_default_willingness_is_quality_times_factor():
    wtp = default_willingness(_A(1.4), _L(quality=500.0, price=999), market=None)
    assert wtp == 500.0 * 1.4          # intrinsic, monetary, ignores price


def test_spawned_agents_get_value_factor():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=env, rng=np.random.default_rng(0), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    u = m.spawn_user()
    assert isinstance(u.value_factor, float) and u.value_factor > 0


def test_market_wtp_uses_the_spec_willingness():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0,
                           willingness=lambda agent, listing, market: 123.0)
    m = Market(env=env, rng=np.random.default_rng(0), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    u = m.spawn_user()
    listing = m.add_listing(quality=500.0, price=400.0, seller_id=u.id)
    assert m.wtp(u, listing) == 123.0
