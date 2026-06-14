# SimPy Re-platform — Phase 1 Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, deterministic, continuous-time SimPy slice of the marketplace — a declarative `MarketplaceSpec` that seeds agents, spawns more mid-run, and drives an `arrive → view → transact` funnel that emits a timestamped event stream.

**Architecture:** A new `sim/` package built on SimPy 4.x. One `simpy.Environment` owns a deterministic event loop; one seeded `numpy.random.Generator` drives every stochastic draw. Each `User` is a persistent generator process that wakes on an engagement-driven exponential schedule and runs a session against the live listing pool. A population-arrival process mints new users over time. Events are stamped with a real `datetime` (via a `Clock` mapping sim-days → calendar) and collected in an in-memory `EventRecorder`. The legacy `classes.py` is untouched.

**Tech Stack:** Python 3, SimPy 4.x, numpy, scipy (frozen distributions as property generators), pytest.

---

## Decisions resolving the PRD's open questions

- **Base time unit = days.** `env.now` is in days; `Clock` maps it to a `datetime` via `start + timedelta(days=now)`. Chosen to match the existing engagement scale (`ENGAGEMENT_TIME_UNIT = 28` days).
- **Stop condition = `until` (float sim-days).** `env.run(until=spec.until)`.
- **Event sink = in-memory `EventRecorder`** (canonical), with optional `write_jsonl(path)`. The legacy `logger_setup.EventLogger` is **intentionally not used** in the slice: its background daemon flush thread breaks the "no threads spawned" check, and its singleton + file handler leak state across runs/tests, breaking determinism. A streaming/file sink is revisited in the parity phase.
- **Perf check** at 1000 users / 7 days happens in the smoke harness (Task 11); record the ceiling there.

## File structure

- `requirements.txt` — pin `simpy`, plus `numpy`, `scipy`, `pytest`. (Create.)
- `sim/__init__.py` — package marker. (Create.)
- `sim/spec.py` — `Property` (fixed-as-dynamic) + `MarketplaceSpec`. (Create.)
- `sim/events.py` — `Event` dataclass + in-memory `EventRecorder`. (Create.)
- `sim/agents.py` — `User`/`Listing` dataclasses, funnel probability functions, `user_lifecycle` + `population_arrival` SimPy processes. Imports `func.sigmoid`. (Create.)
- `sim/engine.py` — `Clock`, `Market` (runtime state + `emit`/`transact`/`spawn_user`), `Marketplace` (`from_spec`/`run`). Imports from `sim.agents`. (Create.)
- `tests/test_*.py` — one test module per unit. (Create.)
- `scripts/run_slice.py` — smoke + reproducibility harness. (Create.)
- `CLAUDE.md` — add a short "Experimental SimPy engine" section. (Modify.)

**Import direction (no cycles):** `engine.py` imports `agents.py`; `agents.py` receives the `market` object by duck-typing and imports nothing from `engine.py`.

All `pytest` / `python` commands are run **from the repo root** so `import sim...` and `import func` both resolve.

---

### Task 1: Project setup — manifest, package, install

**Files:**
- Create: `requirements.txt`
- Create: `sim/__init__.py`

- [ ] **Step 1: Create the dependency manifest**

`requirements.txt`:
```
# SimPy slice engine
simpy>=4,<5
numpy
scipy
pytest
# Note: the legacy classes.py additionally needs scikit-learn, python-json-logger,
# and jupyter_server; those are not required for the sim/ slice or its tests.
```

- [ ] **Step 2: Create the package marker**

`sim/__init__.py`:
```python
"""SimPy-based marketplace simulation engine (experimental)."""
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: installs `simpy`, `numpy`, `scipy`, `pytest` without error.

- [ ] **Step 4: Verify SimPy imports**

Run: `python -c "import simpy; print(simpy.__version__)"`
Expected: prints a 4.x version string.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt sim/__init__.py
git commit -m "build: add simpy slice deps and sim package skeleton"
```

---

### Task 2: `Property` — the fixed-as-dynamic abstraction

**Files:**
- Create: `sim/spec.py`
- Test: `tests/test_properties.py`

- [ ] **Step 1: Write the failing test**

`tests/test_properties.py`:
```python
import numpy as np
from scipy.stats import norm

from sim.spec import Property


def test_property_literal_returns_value():
    assert Property(7).draw(np.random.default_rng(0)) == 7


def test_property_distribution_is_deterministic_with_seed():
    a = Property(norm(loc=10, scale=2)).draw(np.random.default_rng(42))
    b = Property(norm(loc=10, scale=2)).draw(np.random.default_rng(42))
    assert a == b


def test_property_callable_receives_rng():
    p = Property(lambda rng: rng.integers(100, 200))
    v = p.draw(np.random.default_rng(1))
    assert 100 <= v < 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_properties.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.spec'` (or `ImportError` for `Property`).

- [ ] **Step 3: Write minimal implementation**

`sim/spec.py`:
```python
class Property:
    """Fixed-as-dynamic: a literal, a scipy frozen distribution (has .rvs),
    or a callable(rng). A literal is the degenerate generator."""

    def __init__(self, value):
        self.value = value

    def draw(self, rng):
        v = self.value
        if hasattr(v, "rvs"):
            return v.rvs(random_state=rng)
        if callable(v):
            return v(rng)
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_properties.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/spec.py tests/test_properties.py
git commit -m "feat(sim): add Property fixed-as-dynamic value abstraction"
```

---

### Task 3: `Clock` — sim-days → datetime

**Files:**
- Create: `sim/engine.py`
- Test: `tests/test_clock.py`

- [ ] **Step 1: Write the failing test**

`tests/test_clock.py`:
```python
from datetime import datetime, timedelta

from sim.engine import Clock


def test_clock_maps_zero_to_start():
    start = datetime(2026, 1, 1)
    assert Clock(start).to_datetime(0) == start


def test_clock_maps_fractional_days():
    start = datetime(2026, 1, 1)
    assert Clock(start).to_datetime(1.5) == start + timedelta(days=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.engine'`.

- [ ] **Step 3: Write minimal implementation**

Create `sim/engine.py` with just the `Clock` (the rest of the module is added in later tasks):
```python
from datetime import timedelta


class Clock:
    """Maps SimPy float time (in days) to calendar datetimes."""

    def __init__(self, start):
        self.start = start

    def to_datetime(self, now):
        return self.start + timedelta(days=float(now))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_clock.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/engine.py tests/test_clock.py
git commit -m "feat(sim): add Clock mapping sim-days to datetime"
```

---

### Task 4: `Event` + `EventRecorder`

**Files:**
- Create: `sim/events.py`
- Test: `tests/test_events.py`

- [ ] **Step 1: Write the failing test**

`tests/test_events.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.events'`.

- [ ] **Step 3: Write minimal implementation**

`sim/events.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_events.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/events.py tests/test_events.py
git commit -m "feat(sim): add timestamped Event and in-memory EventRecorder"
```

---

### Task 5: `MarketplaceSpec` dataclass + defaults

**Files:**
- Modify: `sim/spec.py` (append `MarketplaceSpec`)
- Test: `tests/test_spec.py`

- [ ] **Step 1: Write the failing test**

`tests/test_spec.py`:
```python
from datetime import datetime

from sim.spec import MarketplaceSpec, Property


def test_default_spec_constructs():
    s = MarketplaceSpec(start=datetime(2026, 1, 1))
    assert s.seed == 0
    assert s.n_seed_users > 0
    assert isinstance(s.engagement, Property)
    assert isinstance(s.listing_quality, Property)


def test_spec_overrides_and_wraps_literals():
    s = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=10, seed=5,
                        listing_price=500)
    assert s.n_seed_users == 10
    assert s.seed == 5
    # a bare literal is wrapped into a Property
    assert isinstance(s.listing_price, Property)
    import numpy as np
    assert s.listing_price.draw(np.random.default_rng(0)) == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spec.py -v`
Expected: FAIL with `ImportError: cannot import name 'MarketplaceSpec'`.

- [ ] **Step 3: Write minimal implementation**

Append to `sim/spec.py` (keep the existing `Property` class at the top of the file):
```python
from dataclasses import dataclass, field
from datetime import datetime

from scipy.stats import gamma, lognorm, norm, poisson


def _as_property(v):
    return v if isinstance(v, Property) else Property(v)


@dataclass
class MarketplaceSpec:
    start: datetime
    seed: int = 0
    n_seed_users: int = 1000
    until: float = 7.0            # sim-days to run
    arrival_rate: float = 5.0     # new users per day (population arrival)
    engagement: Property = field(
        default_factory=lambda: Property(gamma(a=2, scale=7 / 2)))
    response_time: Property = field(
        default_factory=lambda: Property(gamma(a=2, scale=1 / 2)))
    listings_per_user: Property = field(
        default_factory=lambda: Property(poisson(mu=0.6)))
    listing_quality: Property = field(
        default_factory=lambda: Property(lognorm(s=0.6, scale=500)))
    listing_price: Property = field(
        default_factory=lambda: Property(norm(loc=500, scale=100)))

    def __post_init__(self):
        self.engagement = _as_property(self.engagement)
        self.response_time = _as_property(self.response_time)
        self.listings_per_user = _as_property(self.listings_per_user)
        self.listing_quality = _as_property(self.listing_quality)
        self.listing_price = _as_property(self.listing_price)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_spec.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/spec.py tests/test_spec.py
git commit -m "feat(sim): add MarketplaceSpec with property-generator defaults"
```

---

### Task 6: Funnel probability functions

**Files:**
- Create: `sim/agents.py`
- Test: `tests/test_funnel.py`

- [ ] **Step 1: Write the failing test**

`tests/test_funnel.py`:
```python
from sim.agents import p_buy, p_view


def test_p_view_monotonic_in_engagement():
    assert p_view(10) > p_view(1)


def test_p_buy_monotonic_in_engagement():
    assert p_buy(10) > p_buy(1)


def test_probabilities_in_unit_interval():
    for e in [0.01, 1, 5, 50]:
        assert 0.0 <= p_view(e) <= 1.0
        assert 0.0 <= p_buy(e) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_funnel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.agents'`.

- [ ] **Step 3: Write minimal implementation**

Create `sim/agents.py` (funnel section only; processes are added in Task 8):
```python
import math

from func import sigmoid

ENGAGEMENT_TIME_UNIT = 28.0   # days; sets the engagement -> visit-rate scale
EPS = 1e-9
VIEW_BASE, VIEW_SLOPE = 0.95, 1.0
BUY_BASE, BUY_SLOPE = 0.175, 1.0
SESSION_K = 10                # listings shown per session


def p_view(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(VIEW_BASE) + VIEW_SLOPE * math.log(e)))


def p_buy(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(BUY_BASE) + BUY_SLOPE * math.log(e)))


def _decide(p, rng):
    return rng.random() < p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_funnel.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/agents.py tests/test_funnel.py
git commit -m "feat(sim): add engagement-driven funnel probabilities"
```

---

### Task 7: Entities + `Market` runtime (state, emit, transact)

**Files:**
- Modify: `sim/agents.py` (add `User`, `Listing`)
- Modify: `sim/engine.py` (add `Market`)
- Test: `tests/test_market.py`

- [ ] **Step 1: Write the failing test**

`tests/test_market.py`:
```python
from datetime import datetime

import numpy as np
import simpy

from sim.agents import User
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def _market(seed=0):
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    return Market(env=env, rng=np.random.default_rng(seed),
                  clock=Clock(spec.start), recorder=EventRecorder(), spec=spec)


def test_add_listing_is_live_and_listed():
    m = _market()
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=1)
    assert listing.is_live
    assert listing in m.live_listings()


def test_match_listings_orders_by_quality_desc():
    m = _market()
    m.add_listing(quality=10.0, price=1.0, seller_id=1)
    m.add_listing(quality=99.0, price=1.0, seller_id=1)
    top = m.match_listings(k=1)
    assert len(top) == 1 and top[0].quality == 99.0


def test_transact_decrements_stock_and_emits_event():
    m = _market()
    user = User(id=1, engagement=5.0, response_time=1.0)
    listing = m.add_listing(quality=100.0, price=50.0, seller_id=2)
    m.transact(user, listing)
    assert listing.stock == 0
    assert not listing.is_live
    assert any(e.event_type == "transaction" for e in m.recorder.events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market.py -v`
Expected: FAIL with `ImportError: cannot import name 'User'` (or `'Market'`).

- [ ] **Step 3a: Add entities to `sim/agents.py`**

Insert at the top of `sim/agents.py`, above the funnel constants:
```python
from dataclasses import dataclass


@dataclass
class User:
    id: int
    engagement: float
    response_time: float


@dataclass
class Listing:
    id: int
    quality: float
    price: float
    seller_id: int
    stock: int = 1
    is_live: bool = True
    views: int = 0
    transactions: int = 0
```

(Keep the existing `import math`, `from func import sigmoid`, constants, and funnel functions below this.)

- [ ] **Step 3b: Add `Market` to `sim/engine.py`**

Append to `sim/engine.py` (after `Clock`):
```python
from sim.agents import Listing, User, user_lifecycle
from sim.events import Event


class Market:
    """Runtime state + helpers shared by all agent processes."""

    def __init__(self, env, rng, clock, recorder, spec):
        self.env = env
        self.rng = rng
        self.clock = clock
        self.recorder = recorder
        self.spec = spec
        self.users = []
        self.listings = []
        self._next_user_id = 0
        self._next_listing_id = 0

    def emit(self, event_type, actor_id=None, entity_id=None, other_id=None):
        self.recorder.record(Event(
            self.clock.to_datetime(self.env.now),
            event_type, actor_id, entity_id, other_id,
        ))

    def live_listings(self):
        return [l for l in self.listings if l.is_live]

    def match_listings(self, k):
        return sorted(self.live_listings(),
                      key=lambda l: l.quality, reverse=True)[:k]

    def add_listing(self, quality, price, seller_id):
        listing = Listing(id=self._next_listing_id, quality=float(quality),
                          price=float(price), seller_id=seller_id)
        self._next_listing_id += 1
        self.listings.append(listing)
        return listing

    def spawn_user(self):
        user = User(
            id=self._next_user_id,
            engagement=float(self.spec.engagement.draw(self.rng)),
            response_time=float(self.spec.response_time.draw(self.rng)),
        )
        self._next_user_id += 1
        self.users.append(user)
        self.emit("register", actor_id=user.id)
        self.env.process(user_lifecycle(self.env, user, self, self.rng))
        return user

    def transact(self, user, listing):
        listing.stock -= 1
        listing.transactions += 1
        if listing.stock <= 0:
            listing.is_live = False
        self.emit("transaction", actor_id=user.id,
                  entity_id=listing.id, other_id=listing.seller_id)
```

> Note: this import references `user_lifecycle`, which is created in Task 8. `tests/test_market.py` does not call `spawn_user`, but the module-level import must resolve — so Task 8 must land before `test_market.py` is run if you run the full suite. Run this task's test in isolation as written in Step 4; the full-suite green bar arrives at Task 8.

- [ ] **Step 4: Run test to verify it passes**

First add a temporary stub so the import resolves, then run. Append to `sim/agents.py`:
```python
def user_lifecycle(env, user, market, rng):
    yield env.timeout(0)
```
Run: `python -m pytest tests/test_market.py -v`
Expected: 3 passed. (The stub is replaced with the real loop in Task 8.)

- [ ] **Step 5: Commit**

```bash
git add sim/agents.py sim/engine.py tests/test_market.py
git commit -m "feat(sim): add User/Listing entities and Market runtime"
```

---

### Task 8: `user_lifecycle` process (session loop)

**Files:**
- Modify: `sim/agents.py` (replace the `user_lifecycle` stub; add `_run_session`)
- Test: `tests/test_lifecycle.py`

- [ ] **Step 1: Write the failing test**

`tests/test_lifecycle.py`:
```python
from datetime import datetime

import numpy as np
import simpy

from sim.agents import User, user_lifecycle
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def _market(seed=1):
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=env, rng=np.random.default_rng(seed),
               clock=Clock(spec.start), recorder=EventRecorder(), spec=spec)
    return env, m


def test_lifecycle_produces_visit_and_view_events():
    env, m = _market()
    m.add_listing(quality=1000.0, price=10.0, seller_id=999)
    user = User(id=1, engagement=50.0, response_time=1.0)
    m.users.append(user)
    env.process(user_lifecycle(env, user, m, m.rng))
    env.run(until=14)
    kinds = {e.event_type for e in m.recorder.events}
    assert "visit" in kinds
    assert "view" in kinds  # engagement=50 -> p_view ~ 0.98, near-certain over 14 days
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lifecycle.py -v`
Expected: FAIL — the stub `user_lifecycle` records no events, so `"visit" in kinds` is `False` (AssertionError).

- [ ] **Step 3: Replace the stub with the real process**

In `sim/agents.py`, delete the temporary stub from Task 8 Step 4 and add:
```python
def user_lifecycle(env, user, market, rng):
    """Persistent per-agent process: wake on an engagement-driven schedule,
    run a session, repeat for the life of the simulation."""
    while True:
        scale = ENGAGEMENT_TIME_UNIT / max(user.engagement, EPS)
        yield env.timeout(float(rng.exponential(scale)))
        _run_session(user, market, rng)


def _run_session(user, market, rng):
    market.emit("visit", actor_id=user.id)
    for listing in market.match_listings(SESSION_K):
        if not listing.is_live:
            continue
        if _decide(p_view(user.engagement), rng):
            listing.views += 1
            market.emit("view", actor_id=user.id,
                        entity_id=listing.id, other_id=listing.seller_id)
            if listing.is_live and _decide(p_buy(user.engagement), rng):
                market.transact(user, listing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lifecycle.py tests/test_market.py -v`
Expected: all passed (the real `user_lifecycle` also satisfies `test_market.py`'s import).

- [ ] **Step 5: Commit**

```bash
git add sim/agents.py tests/test_lifecycle.py
git commit -m "feat(sim): add engagement-driven user lifecycle session process"
```

---

### Task 9: `population_arrival` process (acquisition)

**Files:**
- Modify: `sim/agents.py` (add `population_arrival`)
- Test: `tests/test_population.py`

- [ ] **Step 1: Write the failing test**

`tests/test_population.py`:
```python
from datetime import datetime

import numpy as np
import simpy

from sim.agents import population_arrival
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def test_population_arrival_adds_users_over_time():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0,
                           arrival_rate=10.0)
    m = Market(env=env, rng=np.random.default_rng(3), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    env.process(population_arrival(env, m, m.rng))
    env.run(until=7)
    assert len(m.users) > 0
    assert any(e.event_type == "register" for e in m.recorder.events)


def test_zero_arrival_rate_adds_no_users():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0,
                           arrival_rate=0.0)
    m = Market(env=env, rng=np.random.default_rng(3), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    env.process(population_arrival(env, m, m.rng))
    env.run(until=7)
    assert len(m.users) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_population.py -v`
Expected: FAIL with `ImportError: cannot import name 'population_arrival'`.

- [ ] **Step 3: Write minimal implementation**

Append to `sim/agents.py`:
```python
def population_arrival(env, market, rng):
    """Mint new users over time at spec.arrival_rate (users per day)."""
    rate = market.spec.arrival_rate
    while rate > 0:
        yield env.timeout(float(rng.exponential(1.0 / rate)))
        market.spawn_user()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_population.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/agents.py tests/test_population.py
git commit -m "feat(sim): add population arrival process for mid-run acquisition"
```

---

### Task 10: `Marketplace.from_spec` + `run` (wire it together)

**Files:**
- Modify: `sim/engine.py` (add `Marketplace`; add top-of-file imports)
- Test: `tests/test_marketplace.py`

- [ ] **Step 1: Write the failing test**

`tests/test_marketplace.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_marketplace.py -v`
Expected: FAIL with `ImportError: cannot import name 'Marketplace'`.

- [ ] **Step 3: Write minimal implementation**

Add these imports to the **top** of `sim/engine.py` (above `Clock`):
```python
import numpy as np
import simpy
```
Then append to `sim/engine.py`:
```python
from sim.agents import population_arrival


class Marketplace:
    """User-facing handle: build from a spec, run, read the event stream."""

    def __init__(self, market):
        self.market = market

    @classmethod
    def from_spec(cls, spec):
        env = simpy.Environment()
        rng = np.random.default_rng(spec.seed)
        market = Market(env=env, rng=rng, clock=Clock(spec.start),
                        recorder=EventRecorder(), spec=spec)
        # Seed users at t0 (each starts its own lifecycle process) + their listings.
        for _ in range(spec.n_seed_users):
            user = market.spawn_user()
            for _ in range(int(spec.listings_per_user.draw(rng))):
                market.add_listing(
                    quality=spec.listing_quality.draw(rng),
                    price=spec.listing_price.draw(rng),
                    seller_id=user.id,
                )
        env.process(population_arrival(env, market, rng))
        return cls(market)

    def run(self, until=None):
        self.market.env.run(until=self.market.spec.until if until is None else until)
        return self.events

    @property
    def events(self):
        return self.market.recorder.events
```
Also add this import near the other `sim.` imports in `sim/engine.py` if not already present (Task 7 added `from sim.events import Event`; extend it):
```python
from sim.events import Event, EventRecorder
```

- [ ] **Step 4: Run the full suite to verify everything passes**

Run: `python -m pytest -v`
Expected: all tests across all modules pass (properties, clock, events, spec, funnel, market, lifecycle, population, marketplace).

- [ ] **Step 5: Commit**

```bash
git add sim/engine.py tests/test_marketplace.py
git commit -m "feat(sim): add Marketplace.from_spec and deterministic run"
```

---

### Task 11: Smoke + reproducibility + perf harness

**Files:**
- Create: `scripts/run_slice.py`

- [ ] **Step 1: Write the harness**

`scripts/run_slice.py`:
```python
"""Run the SimPy slice: print a funnel summary, dump events, verify reproducibility.

Usage (from repo root):  python scripts/run_slice.py
"""
import time
from collections import Counter
from datetime import datetime

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
```

- [ ] **Step 2: Run the harness**

Run: `python scripts/run_slice.py`
Expected: prints non-zero event counts including `visit`, `view`, `transaction`, `register`; `reproducible: True`; writes `slice_events.jsonl`; records the wall-clock (this is the recorded perf ceiling note for the PRD's open question #4).

- [ ] **Step 3: Sanity-check the dumped stream**

Run: `head -n 3 slice_events.jsonl`
Expected: three JSON objects, each with an ISO `sim_time`, an `event_type`, and ids.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_slice.py
git commit -m "feat(sim): add slice smoke + reproducibility harness"
```

> Note: `slice_events.jsonl` is a run artifact — do not commit it. If it is not already ignored, add a line `slice_events.jsonl` (or `*.jsonl`) to `.gitignore` in this step.

---

### Task 12: Document the experimental engine in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a section describing the SimPy slice**

Insert the following section into `CLAUDE.md` immediately **before** the `## Files` section:
```markdown
## Experimental SimPy engine (`sim/`)

A continuous-time re-platform of the daily-loop engine lives in `sim/` (branch
`experimental/simpy-replatform`). It is **additive** — the legacy `classes.py` is
untouched. Design + scope: `docs/specs/2026-06-14-simpy-replatform-prd.md`.

- `sim/spec.py` — `Property` (literal | scipy dist | callable) and `MarketplaceSpec`.
- `sim/events.py` — `Event` + in-memory `EventRecorder` (no threads; optional `write_jsonl`).
- `sim/agents.py` — `User`/`Listing`, funnel probabilities, `user_lifecycle` +
  `population_arrival` SimPy processes.
- `sim/engine.py` — `Clock` (sim-days → datetime), `Market` runtime, `Marketplace.from_spec`/`run`.

Run it:
```python
from datetime import datetime
from sim.engine import Marketplace
from sim.spec import MarketplaceSpec

mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                            n_seed_users=1000, until=7.0, seed=42))
events = mkt.run()   # list[Event], each with a datetime sim_time
```

Tests: `python -m pytest` from the repo root. Smoke harness: `python scripts/run_slice.py`.
The engine is single-threaded and deterministic: same spec + same `seed` → identical events.
```

- [ ] **Step 2: Verify the docs render and tests still pass**

Run: `python -m pytest -q`
Expected: all tests pass (docs change is inert).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the experimental sim/ engine in CLAUDE.md"
```

---

## Self-Review

**1. Spec coverage (PRD §7 requirements):**
- R1 `Marketplace.from_spec` → Task 10. ✓
- R2 seed N at t0 + population arrival → Task 10 seeding loop + Task 9. ✓
- R3 engagement-driven continuous schedule → Task 8 (`scale = ENGAGEMENT_TIME_UNIT / engagement`). ✓
- R4 slice funnel arrive→view→transact → Task 8 `_run_session`. ✓
- R5 datetime-stamped events with type + ids → Task 4 `Event` + Task 7 `Market.emit`. ✓
- R6 literal-or-generator property → Task 2 `Property`. ✓
- R7 deterministic same seed → Task 10 `test_runs_are_reproducible`. ✓
- R8 single-threaded, no threads → Task 10 `test_no_threads_spawned`; EventLogger deliberately unused. ✓
- R9 target population → Task 11 harness at 1000 users / 7 days. ✓
- R10 `classes.py` untouched → no task modifies it. ✓
- PRD §9 reproducibility harness → Task 11. ✓

**2. Placeholder scan:** No TBD/TODO/"add error handling"/"write tests for the above". Every code step shows complete code; every run step shows the exact command + expected result. The one temporary stub (Task 7 Step 4) is explicitly created and explicitly removed in Task 8 Step 3. ✓

**3. Type consistency:** `Property.draw(rng)`, `MarketplaceSpec` field names (`engagement`, `response_time`, `listings_per_user`, `listing_quality`, `listing_price`, `arrival_rate`, `until`, `n_seed_users`, `seed`, `start`), `Event(sim_time, event_type, actor_id, entity_id, other_id)`, `Market.{emit,live_listings,match_listings,add_listing,spawn_user,transact}`, `User(id, engagement, response_time)`, `Listing(id, quality, price, seller_id, stock, is_live, views, transactions)`, `user_lifecycle(env, user, market, rng)`, `population_arrival(env, market, rng)`, `Marketplace.{from_spec, run, events, market}` — all names match across the tasks that define and use them. ✓
```
