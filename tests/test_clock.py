from datetime import datetime, timedelta

from sim.engine import Clock


def test_clock_maps_zero_to_start():
    start = datetime(2026, 1, 1)
    assert Clock(start).to_datetime(0) == start


def test_clock_maps_fractional_days():
    start = datetime(2026, 1, 1)
    assert Clock(start).to_datetime(1.5) == start + timedelta(days=1.5)
