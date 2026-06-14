from datetime import datetime

import numpy as np
import simpy

from sim.agents import User, user_lifecycle
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def _market(seed=1):
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=env, rng=np.random.default_rng(seed),
               clock=Clock(spec.start), recorder=EventRecorder(), spec=spec)
    return env, m


def test_lifecycle_produces_visit_and_view_events():
    env, m = _market()
    m.add_listing(quality=1000.0, price=10.0, seller_id=999)
    user = User(id=1, engagement=50.0, response_time=1.0)
    m.users.append(user)
    env.process(user_lifecycle(env, user, m, m.rng))
    env.run(until=14)
    kinds = {e.event_type for e in m.recorder.events}
    assert "visit" in kinds
    assert "view" in kinds  # engagement=50 -> p_view ~ 0.98, near-certain over 14 days
