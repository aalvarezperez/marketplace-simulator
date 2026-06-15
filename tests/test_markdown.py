from datetime import datetime

import numpy as np
import simpy

from sim.agents import User, markdown_listing
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def _market(seed=0, **kw):
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0, **kw)
    m = Market(env=env, rng=np.random.default_rng(seed), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    return env, m


def test_seller_patience_drawn_and_positive():
    env, m = _market(seed=1)
    u = m.spawn_user()
    assert isinstance(u.patience, float) and u.patience > 0


def test_default_patience_scales_with_until():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), until=20.0)
    draws = [float(spec.seller_patience.draw(np.random.default_rng(s))) for s in range(200)]
    assert 2.5 < np.mean(draws) < 5.5          # default ~ norm(loc=until*0.2=4.0, ...)


def test_unsold_listing_marks_down_over_time():
    env, m = _market(seed=2, markdown_pct=0.1)
    seller = m.spawn_user()
    listing = m.add_listing(quality=500.0, price=1000.0, seller_id=seller.id)
    env.process(markdown_listing(env, listing, m, patience=1.0))
    env.run(until=3.5)
    assert listing.price < 1000.0
    assert any(e.event_type == "markdown" for e in m.recorder.events)


def test_sold_listing_stops_marking_down():
    env, m = _market(seed=3, markdown_pct=0.1)
    seller = m.spawn_user()
    listing = m.add_listing(quality=500.0, price=1000.0, seller_id=seller.id)
    env.process(markdown_listing(env, listing, m, patience=1.0))
    env.run(until=1.5)
    p_after_one = listing.price
    listing.is_live = False
    env.run(until=10.0)
    assert listing.price == p_after_one


def test_markdown_events_occur_in_full_run():
    from sim.engine import Marketplace
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200, until=8.0, seed=1)
    events = Marketplace.from_spec(spec).run()
    assert any(e.event_type == "markdown" for e in events)


def test_markdown_disabled_when_pct_zero():
    from sim.engine import Marketplace
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=100, until=6.0, seed=1,
                           markdown_pct=0.0)
    events = Marketplace.from_spec(spec).run()
    assert not any(e.event_type == "markdown" for e in events)


def test_markdown_run_is_reproducible():
    from sim.engine import Marketplace
    def run():
        spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=120, until=6.0, seed=4)
        return [(e.event_type, e.actor_id, e.entity_id) for e in Marketplace.from_spec(spec).run()]
    assert run() == run()
