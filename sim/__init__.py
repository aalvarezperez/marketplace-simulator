"""SimPy-based marketplace simulation engine (experimental).

    from sim import Marketplace, MarketplaceSpec, negotiate_action

    mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1)))
    events = mkt.run()
    mkt.summary()                 # {'visit': ..., 'view': ..., 'transaction': ...}
    mkt.write_jsonl("events.jsonl")
"""
from sim.actions import negotiate_action
from sim.allocation import (Assignment, AssignmentStore, ClusterRandomization,
                            Experiment, Exposure, SimpleRandomization, Switchback,
                            bucket)
from sim.consideration import quality_ranked_shortlist
from sim.engine import Marketplace
from sim.pricing import default_pricing
from sim.spec import MarketplaceSpec, Property
from sim.willingness import default_willingness

__all__ = [
    "Marketplace", "MarketplaceSpec", "Property",
    "negotiate_action", "default_pricing", "default_willingness", "quality_ranked_shortlist",
    "Experiment", "SimpleRandomization", "ClusterRandomization", "Switchback",
    "AssignmentStore", "Assignment", "Exposure", "bucket",
]
