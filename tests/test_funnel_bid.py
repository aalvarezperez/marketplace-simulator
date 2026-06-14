from datetime import datetime

from sim.agents import p_bid, p_lead
from sim.engine import Marketplace
from sim.spec import MarketplaceSpec


def test_p_lead_and_p_bid_monotonic_in_engagement():
    assert p_lead(10) > p_lead(1)
    assert p_bid(10) > p_bid(1)


def test_lead_and_bid_events_and_proposals_in_full_run():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200,
                           until=7.0, seed=3)
    mkt = Marketplace.from_spec(spec)
    events = mkt.run()
    kinds = {e.event_type for e in events}
    assert "lead" in kinds
    assert "bid" in kinds
    assert mkt.market._next_proposal_id > 0   # bids created proposals (now settled by Epic E)
    # a bid event carries its amount in the payload
    bid_events = [e for e in events if e.event_type == "bid"]
    assert bid_events and bid_events[0].payload is not None and "amount" in bid_events[0].payload


def test_runs_with_bidding_are_reproducible():
    def run():
        spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=150,
                               until=6.0, seed=11)
        return [(e.event_type, e.actor_id, e.entity_id) for e in Marketplace.from_spec(spec).run()]
    assert run() == run()
