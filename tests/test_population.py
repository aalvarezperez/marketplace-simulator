from datetime import datetime

import numpy as np
import simpy

from sim.agents import population_arrival
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def test_population_arrival_adds_users_over_time():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0,
                           arrival_rate=10.0)
    m = Market(env=env, rng=np.random.default_rng(3), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    env.process(population_arrival(env, m, m.rng))
    env.run(until=7)
    assert len(m.users) > 0
    assert any(e.event_type == "register" for e in m.recorder.events)


def test_zero_arrival_rate_adds_no_users():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0,
                           arrival_rate=0.0)
    m = Market(env=env, rng=np.random.default_rng(3), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    env.process(population_arrival(env, m, m.rng))
    env.run(until=7)
    assert len(m.users) == 0
