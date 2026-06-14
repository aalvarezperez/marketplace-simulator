import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Event:
    sim_time: datetime
    event_type: str
    actor_id: Optional[int] = None
    entity_id: Optional[int] = None
    other_id: Optional[int] = None


class EventRecorder:
    """In-memory, thread-free, deterministic event sink."""

    def __init__(self):
        self._events = []

    def record(self, event):
        self._events.append(event)

    @property
    def events(self):
        return list(self._events)

    def write_jsonl(self, path):
        with open(path, "w") as f:
            for e in self._events:
                f.write(json.dumps({
                    "sim_time": e.sim_time.isoformat(),
                    "event_type": e.event_type,
                    "actor_id": e.actor_id,
                    "entity_id": e.entity_id,
                    "other_id": e.other_id,
                }) + "\n")
