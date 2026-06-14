from collections import Counter
from datetime import datetime

from sim.engine import Marketplace
from sim.spec import MarketplaceSpec


def test_default_is_all_control_and_stamped_on_events():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=50, until=3.0, seed=1)
    mkt = Marketplace.from_spec(spec)
    events = mkt.run()
    assert all(u.variant == "CONTROL" for u in mkt.market.users)
    actor_events = [e for e in events if e.actor_id is not None]
    assert actor_events
    assert all(e.payload and e.payload.get("variant") == "CONTROL" for e in actor_events)


def test_weighted_split_assigns_both_variants():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=400, seed=2,
                           variant_weights={"CONTROL": 0.5, "B": 0.5})
    mkt = Marketplace.from_spec(spec)          # users seeded synchronously at build
    counts = Counter(u.variant for u in mkt.market.users)
    assert set(counts) == {"CONTROL", "B"}
    frac_b = counts["B"] / sum(counts.values())
    assert 0.3 < frac_b < 0.7                   # roughly balanced


def test_variant_assignment_is_reproducible():
    def variants():
        spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200, seed=7,
                               variant_weights={"CONTROL": 0.5, "B": 0.5})
        return [u.variant for u in Marketplace.from_spec(spec).market.users]
    assert variants() == variants()
