"""Run the SimPy slice: print a funnel summary, dump events, verify reproducibility.

Usage (from repo root):  python scripts/run_slice.py
"""
import os
import sys
import time
from collections import Counter
from datetime import datetime

# Allow running as a plain script (`python scripts/run_slice.py`) from any cwd:
# CPython puts this file's dir (scripts/) on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.engine import Marketplace
from sim.spec import MarketplaceSpec


def _signature(events):
    return [(e.event_type, e.actor_id, e.entity_id) for e in events]


def main():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=1000,
                           until=7.0, arrival_rate=20.0, seed=42)

    t0 = time.perf_counter()
    mkt = Marketplace.from_spec(spec)
    events = mkt.run()
    elapsed = time.perf_counter() - t0

    counts = Counter(e.event_type for e in events)
    print("event counts:", dict(counts))
    print("total events:", len(events))
    print("users at end:", len(mkt.market.users))
    print(f"wall-clock: {elapsed:.2f}s for 1000 seed users / 7 days")

    mkt.market.recorder.write_jsonl("slice_events.jsonl")
    print("wrote slice_events.jsonl")

    again = Marketplace.from_spec(spec).run()
    print("reproducible:", _signature(events) == _signature(again))


if __name__ == "__main__":
    main()
