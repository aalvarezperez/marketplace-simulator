from datetime import datetime

from sim.engine import Marketplace
from sim.spec import MarketplaceSpec


def test_default_run_has_no_assignment_events():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=50, until=3.0, seed=1)
    mkt = Marketplace.from_spec(spec)
    events = mkt.run()
    assert not [e for e in events if e.event_type == "assignment"]


def test_variant_weights_allocates_both_variants():
    # variant_weights shims into the "default" experiment; resolving each seed
    # user populates the ledger with both buckets.
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=400, seed=2,
                           variant_weights={"CONTROL": 0.5, "B": 0.5})
    mkt = Marketplace.from_spec(spec)
    store = mkt.market.assignment_store
    seen = [store.resolve("default", u, time=0.0) for u in mkt.market.users]
    assert set(seen) == {"CONTROL", "B"}
    frac_b = seen.count("B") / len(seen)
    assert 0.3 < frac_b < 0.7                              # roughly balanced


def test_allocation_is_reproducible():
    def variants():
        spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200, seed=7,
                               variant_weights={"CONTROL": 0.5, "B": 0.5})
        mkt = Marketplace.from_spec(spec)
        store = mkt.market.assignment_store
        return [store.resolve("default", u, time=0.0) for u in mkt.market.users]
    assert variants() == variants()
