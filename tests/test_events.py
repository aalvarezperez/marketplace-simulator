import json
from datetime import datetime

from sim.events import Event, EventRecorder


def test_recorder_collects_in_order():
    r = EventRecorder()
    r.record(Event(datetime(2026, 1, 1), "visit", actor_id=1))
    r.record(Event(datetime(2026, 1, 2), "view", actor_id=1, entity_id=5))
    assert [e.event_type for e in r.events] == ["visit", "view"]


def test_write_jsonl(tmp_path):
    r = EventRecorder()
    r.record(Event(datetime(2026, 1, 1, 12, 0, 0), "view", actor_id=1, entity_id=5))
    path = tmp_path / "events.jsonl"
    r.write_jsonl(path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event_type"] == "view"
    assert rec["actor_id"] == 1
    assert rec["entity_id"] == 5
    assert rec["sim_time"] == "2026-01-01T12:00:00"
