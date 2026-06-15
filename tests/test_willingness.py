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


def test_buy_action_factory_fidelities():
    from sim.actions import buy_action
    assert buy_action().fidelity == "explicit"
    assert buy_action("implicit").fidelity == "implicit"
    assert buy_action().name == "buy"


def test_explicit_buy_prices_out_low_wtp():
    import numpy as np
    import simpy
    from datetime import datetime
    from sim.actions import buy_action
    from sim.engine import Clock, Market
    from sim.events import EventRecorder
    from sim.spec import MarketplaceSpec

    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=env, rng=np.random.default_rng(0), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    buyer = m.spawn_user()
    buyer.value_factor = 1.0                       # WTP = quality
    cheap = m.add_listing(quality=500.0, price=400.0, seller_id=999)   # 500 >= 400 -> buy
    dear = m.add_listing(quality=500.0, price=600.0, seller_id=999)    # 500 < 600 -> skip
    session = {"consideration": [cheap, dear]}
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert cheap.transactions == 1 and not cheap.is_live
    assert dear.transactions == 0 and dear.is_live


def test_default_funnel_buy_is_explicit():
    from sim.actions import default_consumer_funnel
    buy = {a.name: a for a in default_consumer_funnel()}["buy"]
    assert buy.fidelity == "explicit"
