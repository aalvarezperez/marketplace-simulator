from datetime import datetime

import numpy as np
import simpy

from sim.agents import Proposal
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def _market(seed=0):
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    return Market(env=env, rng=np.random.default_rng(seed), clock=Clock(spec.start),
                  recorder=EventRecorder(), spec=spec)


def test_spawned_user_has_store_inbox():
    m = _market()
    u = m.spawn_user()
    assert isinstance(u.inbox, simpy.Store)


def test_make_proposal_increments_ids_and_starts_created():
    m = _market()
    buyer, seller = m.spawn_user(), m.spawn_user()
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=seller.id)
    p0 = m.make_proposal(buyer, seller, listing, amount=45.0)
    p1 = m.make_proposal(buyer, seller, listing, amount=40.0)
    assert p0.status == "created" and p0.amount == 45.0
    assert p1.id == p0.id + 1


def test_send_to_seller_routes_into_seller_inbox():
    m = _market()
    buyer, seller = m.spawn_user(), m.spawn_user()
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=seller.id)
    p = m.make_proposal(buyer, seller, listing, amount=45.0)
    m.send_to_seller(p)
    assert p.status == "with_seller"
    assert p in seller.inbox.items


def test_send_to_buyer_routes_into_buyer_inbox():
    m = _market()
    buyer, seller = m.spawn_user(), m.spawn_user()
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=seller.id)
    p = m.make_proposal(buyer, seller, listing, amount=45.0)
    m.send_to_buyer(p)
    assert p.status == "with_buyer"
    assert p in buyer.inbox.items


def test_event_carries_optional_payload():
    m = _market()
    m.emit("bid", actor_id=1, entity_id=2, other_id=3, payload={"amount": 45.0})
    ev = m.recorder.events[-1]
    assert ev.payload == {"amount": 45.0}
