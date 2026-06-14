from datetime import datetime

import numpy as np
import simpy

from sim.agents import User, p_list, user_lifecycle
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def test_p_list_monotonic_in_engagement():
    assert p_list(10) > p_list(1)


def test_seller_lists_during_run():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=env, rng=np.random.default_rng(1), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    u = User(id=0, engagement=80.0, response_time=1.0)
    m.users.append(u)
    env.process(user_lifecycle(env, u, m, m.rng))
    env.run(until=30)
    assert any(e.event_type == "list" for e in m.recorder.events)
    assert len(m.listings) > 0
