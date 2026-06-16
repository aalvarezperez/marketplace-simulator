import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Event:
    """One thing that happened, at one calendar instant. The unit of output.

    Deliberately generic (no marketplace vocabulary in the shape): an ``event_type``
    string plus up to three integer ids, read by convention as
    ``actor`` (who acted), ``entity`` (the listing/proposal acted on), and ``other``
    (the counterparty, e.g. the seller). ``payload`` carries any extras — always the
    A/B ``variant`` of the actor, plus per-type fields like ``price`` or ``amount``.
    Frozen so a recorded event can never be mutated after the fact.
    """
    sim_time: datetime
    event_type: str
    actor_id: Optional[int] = None
    entity_id: Optional[int] = None
    other_id: Optional[int] = None
    payload: Optional[dict] = None


class EventRecorder:
    """In-memory, thread-free, deterministic event sink."""

    def __init__(self):
        self._events = []

    def record(self, event):
        """Append one ``Event``. Append-order is the deterministic event order."""
        self._events.append(event)

    @property
    def events(self):
        """A shallow copy of the recorded events, in occurrence order."""
        return list(self._events)

    def write_jsonl(self, path):
        """Write the stream to ``path`` as JSON lines, one event per row.

        ``sim_time`` is serialized as an ISO-8601 string; all other fields pass
        through unchanged. Round-trips into pandas via ``pd.read_json(lines=True)``.
        """
        with open(path, "w") as f:
            for e in self._events:
                f.write(json.dumps({
                    "sim_time": e.sim_time.isoformat(),
                    "event_type": e.event_type,
                    "actor_id": e.actor_id,
                    "entity_id": e.entity_id,
                    "other_id": e.other_id,
                    "payload": e.payload,
                }) + "\n")
