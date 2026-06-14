import threading
from datetime import datetime, timedelta

from sim.engine import Marketplace
from sim.spec import MarketplaceSpec


def _small_spec(seed=0):
    return MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200,
                           until=7.0, arrival_rate=5.0, seed=seed)


def test_run_produces_event_stream():
    events = Marketplace.from_spec(_small_spec()).run()
    assert len(events) > 0
    assert "visit" in {e.event_type for e in events}


def test_events_within_run_window():
    spec = _small_spec()
    events = Marketplace.from_spec(spec).run()
    start, end = spec.start, spec.start + timedelta(days=spec.until)
    for e in events:
        assert start <= e.sim_time <= end


def test_runs_are_reproducible():
    def run():
        return [(e.event_type, e.actor_id, e.entity_id)
                for e in Marketplace.from_spec(_small_spec(seed=7)).run()]
    assert run() == run()


def test_no_threads_spawned():
    before = threading.active_count()
    Marketplace.from_spec(_small_spec()).run()
    assert threading.active_count() == before
