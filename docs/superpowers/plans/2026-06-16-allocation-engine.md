# Allocation Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class, swappable variant **allocation** (deterministic hash bucketing, pluggable randomization designs, and an in-memory assignment store as the source of truth) to the `sim/` engine, as a pure data-generation concern.

**Architecture:** A new self-contained module `sim/allocation.py` holds a deterministic `bucket()` hash, three `AllocationStrategy` built-ins (simple / cluster / switchback), an `Experiment` registry dataclass, an `Assignment` truth row, and an `AssignmentStore` (cache + append-only ledger). The runtime `Market` builds the store and exposes one lookup, `Market.variant(subject, exp_key)`, which resolves lazily, caches the result, and emits an `assignment` event as a *projection* of the ledger. The action funnel is untouched. `variant_weights` keeps working via a one-experiment shim.

**Tech Stack:** Python 3.10+, `numpy`, `hashlib` (md5), `dataclasses`, SimPy (already present), `pytest`. Interpreter = the conda base `python` (has numpy/scipy/scikit-learn/simpy/pytest).

**Spec:** `docs/superpowers/specs/2026-06-16-allocation-engine-design.md`

**Invariants to hold throughout:** deterministic (md5 + sim-time only in resolution; single seeded `numpy` rng for all sampling; the one new rng draw is `cluster` at spawn, and only when `cluster` is non-literal); single-threaded; counter ids; `classes.py` frozen (never edit). Same spec + same seed → byte-identical event stream for runs that do not set a real `variant_weights` split.

**Run tests with:** `python -m pytest -q` from the repo root. Existing suite = 82 tests; all stay green except the deliberate `tests/test_variant.py` rewrite in Task 6.

---

## File Structure

- **Create `sim/allocation.py`** — the entire allocation subsystem: `bucket`, `SimpleRandomization`, `ClusterRandomization`, `Switchback`, `Experiment`, `Assignment`, `AssignmentStore`. No imports from other `sim/` modules (avoids cycles; `sim/spec.py` and `sim/engine.py` import *from* it).
- **Create `tests/test_allocation.py`** — unit tests for bucketing, strategies, the experiment dataclass, and the store (built incrementally across Tasks 1–4).
- **Modify `sim/spec.py`** — add `experiments` and `cluster` fields; coerce `cluster` to `Property`; synthesize the `default` experiment from `variant_weights`.
- **Modify `sim/engine.py`** — `Market` builds the store and gains `variant()`; `spawn_user` draws `cluster` and no longer assigns `variant`; delete `_assign_variant`; `emit` drops the variant stamp.
- **Modify `sim/agents.py`** — `User` gains `cluster`, loses `variant`.
- **Modify `sim/__init__.py`** — export the new public names.
- **Rewrite `tests/test_variant.py`** — assert against the store model instead of `user.variant`.

---

## Task 1: `bucket()` — deterministic hash assignment

**Files:**
- Create: `sim/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_allocation.py`:

```python
from collections import Counter

from sim.allocation import bucket


def test_bucket_is_deterministic():
    v = {"CONTROL": 0.5, "B": 0.5}
    a = bucket("user-7", v, salt="exp1")
    b = bucket("user-7", v, salt="exp1")
    assert a == b
    assert a in v


def test_bucket_respects_weights_roughly():
    v = {"CONTROL": 0.8, "B": 0.2}
    counts = Counter(bucket(f"user-{i}", v, salt="exp1") for i in range(4000))
    frac_b = counts["B"] / sum(counts.values())
    assert 0.15 < frac_b < 0.25          # ~0.2, generous bound


def test_bucket_salt_changes_assignment():
    v = {"CONTROL": 0.5, "B": 0.5}
    # Over many keys, two different salts must not produce identical assignments.
    same = sum(bucket(f"u{i}", v, salt="A") == bucket(f"u{i}", v, salt="B")
               for i in range(500))
    assert same < 500                    # independent salts -> not all equal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.allocation'`.

- [ ] **Step 3: Write minimal implementation**

Create `sim/allocation.py`:

```python
"""Variant allocation: deterministic hash bucketing, pluggable randomization
designs, and an in-memory assignment store (the source of truth).

Allocation answers "which variant is unit U in for experiment E at time T?".
It is design-driven, looked up on demand, and cached. It is NOT exposure (the
action-consequence layer) and defines no treatment effect. The event stream is a
projection of the store, not the truth.

Self-contained: imports only stdlib + numpy, so sim.spec / sim.engine import it
without cycles.
"""
import hashlib

import numpy as np


def bucket(key, variants, salt):
    """Map a string key to a variant by content hashing.

    Same (key, salt) -> same variant, always; order-independent and stable across
    processes (this is why we use md5, not Python's process-salted ``hash()``).
    ``variants`` is {name: weight}; weights need not sum to 1 (they are normalized).
    """
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF                 # deterministic uniform in [0, 1]
    names = list(variants)
    cum = np.cumsum(np.array([variants[n] for n in names], dtype=float))
    cum /= cum[-1]
    return names[min(int(np.searchsorted(cum, x)), len(names) - 1)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add sim/allocation.py tests/test_allocation.py
git commit -m "feat(allocation): deterministic bucket() hash assignment"
```

---

## Task 2: Allocation strategies (simple / cluster / switchback)

**Files:**
- Modify: `sim/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
from sim.allocation import (SimpleRandomization, ClusterRandomization,
                            Switchback)

VARS = {"CONTROL": 0.5, "B": 0.5}


def test_simple_is_sticky_and_time_invariant():
    s = SimpleRandomization()
    assert s.window(0.0) is None
    assert s.window(9.9) is None
    v1 = s.assign("user-3", cluster_key="c0", window=None, variants=VARS, salt="e")
    v2 = s.assign("user-3", cluster_key="c1", window=None, variants=VARS, salt="e")
    assert v1 == v2                      # depends on the unit key, not the cluster


def test_cluster_shares_variant_within_cluster():
    s = ClusterRandomization()
    # Different units, same cluster -> same variant.
    a = s.assign("user-1", cluster_key="north", window=None, variants=VARS, salt="e")
    b = s.assign("user-2", cluster_key="north", window=None, variants=VARS, salt="e")
    assert a == b


def test_switchback_flips_by_window():
    s = Switchback(period=1.0)
    assert s.window(0.4) == 0
    assert s.window(1.0) == 1
    assert s.window(2.7) == 2
    # Same window -> same variant; the assignment depends on the window, not the unit.
    v0a = s.assign("user-1", cluster_key="c", window=0, variants=VARS, salt="e")
    v0b = s.assign("user-2", cluster_key="c", window=0, variants=VARS, salt="e")
    assert v0a == v0b
    assert s.window_bounds(3) == (3.0, 4.0)


def test_switchback_per_cluster_differs():
    s = Switchback(period=1.0, per_cluster=True)
    # Same window, different clusters can differ; the key includes the cluster.
    distinct = {s.assign("u", cluster_key=c, window=0, variants=VARS, salt="e")
                for c in range(20)}
    assert distinct == {"CONTROL", "B"}  # both variants appear across clusters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: FAIL with `ImportError: cannot import name 'SimpleRandomization'`.

- [ ] **Step 3: Write minimal implementation**

Append to `sim/allocation.py`:

```python
class SimpleRandomization:
    """Per-unit Bernoulli assignment. Time-invariant, sticky to the unit."""

    def window(self, t):
        return None

    def assign(self, unit_key, cluster_key, window, variants, salt):
        return bucket(str(unit_key), variants, salt)


class ClusterRandomization:
    """Every unit in a cluster shares the cluster's variant. Time-invariant."""

    def window(self, t):
        return None

    def assign(self, unit_key, cluster_key, window, variants, salt):
        return bucket(str(cluster_key), variants, salt)


class Switchback:
    """The market (or each cluster) flips variant every ``period`` sim-days."""

    def __init__(self, period=1.0, per_cluster=False):
        self.period = period
        self.per_cluster = per_cluster

    def window(self, t):
        return int(t // self.period)

    def assign(self, unit_key, cluster_key, window, variants, salt):
        key = f"{cluster_key}:{window}" if self.per_cluster else str(window)
        return bucket(key, variants, salt)

    def window_bounds(self, window):
        return (window * self.period, (window + 1) * self.period)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add sim/allocation.py tests/test_allocation.py
git commit -m "feat(allocation): simple/cluster/switchback strategies"
```

---

## Task 3: `Experiment` dataclass

**Files:**
- Modify: `sim/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
from sim.allocation import Experiment


class _Subj:
    def __init__(self, id, cluster=0):
        self.id = id
        self.cluster = cluster


def test_experiment_defaults():
    e = Experiment(key="wtp", variants={"CONTROL": 0.5, "B": 0.5})
    assert isinstance(e.strategy, SimpleRandomization)
    assert e.salt == "wtp"               # defaults to the key
    assert e.start == 0.0 and e.end is None
    assert e.eligibility is None
    s = _Subj(id=42, cluster=7)
    assert e.subject_key(s) == 42        # default extractor = .id
    assert e.cluster_key(s) == 7         # default extractor = .cluster


def test_experiment_custom_extractors_and_salt():
    e = Experiment(key="k", variants={"A": 1.0}, salt="custom",
                   subject_key=lambda s: f"u{s.id}",
                   cluster_key=lambda s: "fixed")
    s = _Subj(id=5)
    assert e.salt == "custom"
    assert e.subject_key(s) == "u5"
    assert e.cluster_key(s) == "fixed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: FAIL with `ImportError: cannot import name 'Experiment'`.

- [ ] **Step 3: Write minimal implementation**

Add the import at the top of `sim/allocation.py` (just below `import hashlib`):

```python
from dataclasses import dataclass
```

Then append to `sim/allocation.py`:

```python
@dataclass
class Experiment:
    """Declarative description of one experiment's allocation.

    Names in ``variants`` are entirely user-defined; nothing is privileged. There
    is no ``unit`` enum and no baked-in baseline name: granularity is set by the
    ``subject_key`` / ``cluster_key`` extractors plus what the caller passes to
    ``AssignmentStore.resolve``. When the experiment is inactive / ineligible /
    absent, resolution returns the caller's ``default`` (see AssignmentStore).
    """
    key: str
    variants: dict                       # {'CONTROL': .5, 'B': .5}
    strategy: object = None              # an AllocationStrategy; default SimpleRandomization()
    salt: str = None                     # default = key (each experiment hashes independently)
    start: float = 0.0                   # active window, sim-days (inclusive)
    end: float = None                    # active window end, sim-days (exclusive); None = no end
    eligibility: object = None           # predicate(subject, market) -> bool; default everyone
    subject_key: object = None           # subject -> hashing key; default lambda s: s.id
    cluster_key: object = None           # subject -> cluster key; default lambda s: getattr(s,'cluster',0)

    def __post_init__(self):
        if self.strategy is None:
            self.strategy = SimpleRandomization()
        if self.salt is None:
            self.salt = self.key
        if self.subject_key is None:
            self.subject_key = lambda s: s.id
        if self.cluster_key is None:
            self.cluster_key = lambda s: getattr(s, "cluster", 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add sim/allocation.py tests/test_allocation.py
git commit -m "feat(allocation): Experiment dataclass with generic extractors"
```

---

## Task 4: `Assignment` + `AssignmentStore` (source of truth)

**Files:**
- Modify: `sim/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
from sim.allocation import Assignment, AssignmentStore


class _FakeMarket:
    """Minimal stand-in: AssignmentStore only needs market.emit for the log."""
    def __init__(self):
        self.events = []

    def emit(self, event_type, actor_id=None, entity_id=None, other_id=None, payload=None):
        self.events.append((event_type, actor_id, payload))


def _store(experiments):
    m = _FakeMarket()
    return AssignmentStore(experiments, m), m


def test_resolve_returns_variant_and_logs_once():
    exp = Experiment(key="wtp", variants={"CONTROL": 0.5, "B": 0.5})
    store, m = _store([exp])
    s = _Subj(id=1, cluster=0)
    v1 = store.resolve("wtp", s, time=0.0)
    v2 = store.resolve("wtp", s, time=0.5)        # same window (None) -> cached, no recompute
    assert v1 == v2 and v1 in ("CONTROL", "B")
    assert len(store.ledger()) == 1               # written once
    assert [e for e in m.events if e[0] == "assignment"]  # logged a projection


def test_unknown_experiment_returns_default_no_row():
    store, m = _store([])
    assert store.resolve("missing", _Subj(1), time=0.0, default=None) is None
    assert store.ledger() == []


def test_inactive_window_returns_default_no_row():
    exp = Experiment(key="e", variants={"A": 1.0}, start=5.0, end=10.0)
    store, m = _store([exp])
    assert store.resolve("e", _Subj(1), time=0.0) is None     # before start
    assert store.resolve("e", _Subj(1), time=10.0) is None    # at end (exclusive)
    assert store.ledger() == []


def test_eligibility_gates_assignment():
    exp = Experiment(key="e", variants={"A": 1.0},
                     eligibility=lambda subj, mkt: subj.id % 2 == 0)
    store, m = _store([exp])
    assert store.resolve("e", _Subj(2), time=0.0) == "A"
    assert store.resolve("e", _Subj(3), time=0.0) is None
    assert len(store.ledger()) == 1


def test_switchback_makes_a_row_per_window():
    exp = Experiment(key="sb", variants={"CONTROL": 0.5, "B": 0.5},
                     strategy=Switchback(period=1.0))
    store, m = _store([exp])
    s = _Subj(id=1)
    store.resolve("sb", s, time=0.2)              # window 0
    store.resolve("sb", s, time=0.7)              # window 0 again -> cached
    store.resolve("sb", s, time=1.3)              # window 1 -> new row
    assert len(store.ledger()) == 2


def test_valid_bounds_sticky_vs_switchback():
    sticky = Experiment(key="s", variants={"A": 1.0}, start=2.0, end=8.0)
    store, _ = _store([sticky])
    store.resolve("s", _Subj(1), time=3.0)
    a = store.ledger()[0]
    assert (a.valid_from, a.valid_to) == (2.0, 8.0)

    sb = Experiment(key="b", variants={"A": 1.0},
                    strategy=Switchback(period=1.0), start=0.0, end=10.0)
    store2, _ = _store([sb])
    store2.resolve("b", _Subj(1), time=3.4)       # window 3 -> bounds (3,4)
    b = store2.ledger()[0]
    assert (b.valid_from, b.valid_to) == (3.0, 4.0)
    assert b.window == 3 and b.assigned_at == 3.4


def test_current_returns_latest():
    exp = Experiment(key="sb", variants={"CONTROL": 0.5, "B": 0.5},
                     strategy=Switchback(period=1.0))
    store, _ = _store([exp])
    s = _Subj(id=1)
    store.resolve("sb", s, time=0.2)
    store.resolve("sb", s, time=1.3)
    cur = store.current("sb", subject_id=1)
    assert isinstance(cur, Assignment) and cur.window == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: FAIL with `ImportError: cannot import name 'Assignment'`.

- [ ] **Step 3: Write minimal implementation**

Append to `sim/allocation.py`:

```python
@dataclass(frozen=True)
class Assignment:
    """The truth row written the moment a subject is allocated."""
    experiment: str
    subject_id: object        # the hashed unit's id
    variant: str
    cluster: object
    window: object            # None for time-invariant designs; window index for switchback
    assigned_at: float        # sim-time of first resolution
    valid_from: float         # when this assignment takes effect (None = open)
    valid_to: float           # when it stops (None = open-ended)


class AssignmentStore:
    """Authoritative cache + append-only ledger. Written once per allocation,
    queried cheaply thereafter. The event stream is a projection of this, not the
    truth.
    """

    def __init__(self, experiments, market):
        self._exp = {e.key: e for e in experiments}
        self._market = market
        self._cache = {}     # (exp_key, subject_id, window) -> Assignment   <- O(1) lookup
        self._ledger = []    # append-only list[Assignment]                  <- the persistent truth

    def resolve(self, exp_key, subject, time, default=None):
        exp = self._exp.get(exp_key)
        if exp is None:
            return default
        window = exp.strategy.window(time)
        ckey = (exp_key, subject.id, window)
        hit = self._cache.get(ckey)
        if hit is not None:                                  # already allocated -> read truth
            return hit.variant
        if not (exp.start <= time and (exp.end is None or time < exp.end)):
            return default                                   # outside active window
        if exp.eligibility is not None and not exp.eligibility(subject, self._market):
            return default                                   # ineligible
        variant = exp.strategy.assign(
            exp.subject_key(subject), exp.cluster_key(subject), window, exp.variants, exp.salt)
        vfrom, vto = self._valid_bounds(exp, window)
        a = Assignment(exp_key, subject.id, variant, exp.cluster_key(subject),
                       window, time, vfrom, vto)
        self._cache[ckey] = a
        self._ledger.append(a)                               # write truth once
        self._market.emit("assignment", actor_id=subject.id, payload={
            "experiment": a.experiment, "variant": a.variant, "cluster": a.cluster,
            "window": a.window, "valid_from": a.valid_from, "valid_to": a.valid_to,
        })                                                   # log = projection of the truth
        return variant

    def _valid_bounds(self, exp, window):
        if window is None:
            return (exp.start, exp.end)
        wb = getattr(exp.strategy, "window_bounds", lambda w: (None, None))
        wfrom, wto = wb(window)
        lo = exp.start if wfrom is None else max(exp.start, wfrom)
        hi = wto if exp.end is None else (exp.end if wto is None else min(exp.end, wto))
        return (lo, hi)

    def current(self, exp_key, subject_id):
        """Most recent Assignment for a subject in an experiment, or None."""
        hits = [a for (e, s, _w), a in self._cache.items()
                if e == exp_key and s == subject_id]
        return max(hits, key=lambda a: a.assigned_at) if hits else None

    def ledger(self):
        """The full append-only assignment record (for export to a dataframe)."""
        return list(self._ledger)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: PASS (16 passed).

- [ ] **Step 5: Commit**

```bash
git add sim/allocation.py tests/test_allocation.py
git commit -m "feat(allocation): Assignment row + AssignmentStore (cache + ledger)"
```

---

## Task 5: Spec wiring — `experiments`, `cluster`, `variant_weights` shim

**Files:**
- Modify: `sim/spec.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
from datetime import datetime

from sim.spec import MarketplaceSpec, Property


def test_spec_no_split_has_no_experiments():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1))
    assert spec.experiments == []


def test_variant_weights_synthesizes_default_experiment():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1),
                           variant_weights={"CONTROL": 0.5, "B": 0.5})
    assert len(spec.experiments) == 1
    e = spec.experiments[0]
    assert e.key == "default"
    assert e.variants == {"CONTROL": 0.5, "B": 0.5}
    assert isinstance(e.strategy, SimpleRandomization)


def test_explicit_experiments_win_over_variant_weights():
    custom = Experiment(key="mine", variants={"CONTROL": 0.5, "B": 0.5})
    spec = MarketplaceSpec(start=datetime(2026, 1, 1),
                           variant_weights={"CONTROL": 0.5, "B": 0.5},
                           experiments=[custom])
    assert [e.key for e in spec.experiments] == ["mine"]   # variant_weights ignored


def test_cluster_is_coerced_to_property():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1))
    assert isinstance(spec.cluster, Property)
    assert spec.cluster.draw(rng=None) == 0                # literal 0; rng untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py -k "spec or variant_weights or cluster or experiments" -q`
Expected: FAIL with `AttributeError: 'MarketplaceSpec' object has no attribute 'experiments'`.

- [ ] **Step 3: Write minimal implementation**

In `sim/spec.py`, add this import near the other `sim.*` imports (below `from sim.willingness import default_willingness`):

```python
from sim.allocation import Experiment, SimpleRandomization
```

Add two fields to `MarketplaceSpec`, immediately after the existing `markdown_pct: float = 0.1` line:

```python
    experiments: list = field(default_factory=list)     # allocation Experiment registry
    cluster: Property = field(default_factory=lambda: Property(0))   # generic cluster key per agent
```

Extend `__post_init__`. The current body ends with:

```python
        if self.seller_patience is None:
            self.seller_patience = Property(norm(loc=self.until * 0.2, scale=self.until * 0.1))
        else:
            self.seller_patience = _as_property(self.seller_patience)
```

Append, after that block, inside `__post_init__`:

```python
        self.cluster = _as_property(self.cluster)
        # Back-compat: a real variant_weights split with no explicit experiments
        # synthesizes one simple-randomization experiment named "default".
        # When experiments are given explicitly, variant_weights is ignored.
        if not self.experiments and len(self.variant_weights) > 1:
            self.experiments = [Experiment(key="default",
                                           variants=dict(self.variant_weights),
                                           strategy=SimpleRandomization())]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: PASS (20 passed).

- [ ] **Step 5: Commit**

```bash
git add sim/spec.py tests/test_allocation.py
git commit -m "feat(allocation): spec experiments + cluster fields + variant_weights shim"
```

---

## Task 6: Engine + agents wiring; rewrite `test_variant.py`

**Files:**
- Modify: `sim/agents.py` (the `User` dataclass)
- Modify: `sim/engine.py` (`Market.__init__`, `emit`, delete `_assign_variant`, `spawn_user`, add `variant()`)
- Rewrite: `tests/test_variant.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test (integration) and rewrite `test_variant.py`**

Append to `tests/test_allocation.py`:

```python
from sim.allocation import Experiment as _Exp
from sim.engine import Marketplace


def test_market_variant_lookup_and_ledger_end_to_end():
    # A variant-aware willingness callable makes 'B' value items more, so the
    # buy step reads market.variant(...) -> allocation gets resolved + logged.
    def uplift(agent, listing, market):
        base = listing.quality * agent.value_factor
        return base * 1.2 if market.variant(agent, "wtp") == "B" else base

    def build():
        spec = MarketplaceSpec(
            start=datetime(2026, 1, 1), n_seed_users=300, until=5.0, seed=11,
            willingness=uplift,
            experiments=[_Exp(key="wtp", variants={"CONTROL": 0.5, "B": 0.5})])
        return Marketplace.from_spec(spec)

    mkt = build()
    events = mkt.run()
    assignment_events = [e for e in events if e.event_type == "assignment"]
    assert assignment_events                               # allocation produced data
    ledger = mkt.market.assignment_store.ledger()
    variants = {a.variant for a in ledger}
    assert variants == {"CONTROL", "B"}                    # both buckets allocated

    # Determinism: same spec + seed -> byte-identical stream.
    a = [(e.event_type, e.actor_id, e.entity_id) for e in build().run()]
    b = [(e.event_type, e.actor_id, e.entity_id) for e in build().run()]
    assert a == b


def test_no_experiments_means_no_assignment_events():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=50, until=3.0, seed=1)
    mkt = Marketplace.from_spec(spec)
    events = mkt.run()
    assert not [e for e in events if e.event_type == "assignment"]
    # Lookup on an unconfigured experiment returns the caller's default.
    user = mkt.market.users[0]
    assert mkt.market.variant(user, "anything", default=None) is None
```

Replace the **entire contents** of `tests/test_variant.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_variant.py tests/test_allocation.py -q`
Expected: FAIL — `tests/test_variant.py` fails because `mkt.market.assignment_store` does not exist yet; the new `test_allocation.py` integration tests fail on the missing `assignment_store` / `Market.variant`.

- [ ] **Step 3: Write minimal implementation**

**3a. `sim/agents.py`** — update the `User` dataclass. Replace:

```python
@dataclass
class User:
    """A marketplace participant — generic, so the same object is buyer and seller.

    Heterogeneity lives in the per-agent draws: ``engagement`` drives every implicit
    funnel rate and the wake-up cadence; ``response_time`` is the latency (days)
    before this user reacts to its inbox; ``value_factor`` scales willingness-to-pay;
    ``patience`` is the days-unsold a listing waits before this seller marks it down.
    ``inbox`` is a ``simpy.Store`` (assigned at spawn). ``state`` toggles
    active/dormant across churn and reactivation. ``variant`` is the A/B tag.
    """
    id: int
    engagement: float
    response_time: float
    inbox: object = None   # a simpy.Store, assigned at spawn time
    variant: str = "CONTROL"
    state: str = "active"      # active | dormant
    value_factor: float = 1.0
    patience: float = 0.0
```

with:

```python
@dataclass
class User:
    """A marketplace participant — generic, so the same object is buyer and seller.

    Heterogeneity lives in the per-agent draws: ``engagement`` drives every implicit
    funnel rate and the wake-up cadence; ``response_time`` is the latency (days)
    before this user reacts to its inbox; ``value_factor`` scales willingness-to-pay;
    ``patience`` is the days-unsold a listing waits before this seller marks it down.
    ``inbox`` is a ``simpy.Store`` (assigned at spawn). ``state`` toggles
    active/dormant across churn and reactivation. ``cluster`` is the unit's
    cluster key (drawn at spawn), used by cluster/switchback allocation designs.
    """
    id: int
    engagement: float
    response_time: float
    inbox: object = None   # a simpy.Store, assigned at spawn time
    cluster: object = 0    # cluster key for allocation; drawn at spawn from spec.cluster
    state: str = "active"      # active | dormant
    value_factor: float = 1.0
    patience: float = 0.0
```

**3b. `sim/engine.py` import** — add `AssignmentStore` to the allocation-free import block. After the existing line `from sim.actions import assemble_actions, default_consumer_funnel, run_session as _run_session_actions`, add:

```python
from sim.allocation import AssignmentStore
```

**3c. `Market.__init__`** — after the line `self.markdown_pct = spec.markdown_pct`, add:

```python
        self.experiments = spec.experiments
        self.assignment_store = AssignmentStore(spec.experiments, self)
```

**3d. Delete `_assign_variant`** entirely (the whole method, lines beginning `def _assign_variant(self, rng):` through its `return names[int(rng.choice(len(names), p=w))]`).

**3e. Add `Market.variant`** — insert immediately after `run_session` (before `emit`):

```python
    def variant(self, subject, exp_key, default=None):
        """Look up ``subject``'s variant for experiment ``exp_key`` at the current
        sim-time. Resolves + caches + logs on first read (per switchback window);
        O(1) thereafter. Returns ``default`` when the experiment is unknown,
        inactive, or the subject is ineligible. This is the whole allocation surface."""
        return self.assignment_store.resolve(exp_key, subject, self.env.now, default)
```

**3f. Simplify `emit`** — replace the whole method with the variant stamp removed:

```python
    def emit(self, event_type, actor_id=None, entity_id=None, other_id=None, payload=None):
        """Record an ``Event`` stamped with the current calendar time. Behavioral
        events are lean; the ``assignment`` event (a projection of the
        AssignmentStore) is the only variant-bearing log."""
        self.recorder.record(Event(
            self.clock.to_datetime(self.env.now),
            event_type, actor_id, entity_id, other_id, payload,
        ))
```

**3g. `spawn_user`** — replace the line `user.variant = self._assign_variant(self.rng)` with:

```python
        user.cluster = self.spec.cluster.draw(self.rng)
```

Also update the `spawn_user` docstring phrase "give it an inbox + A/B variant" to "give it an inbox + cluster key".

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — all tests green (the rewritten `test_variant.py`, the new `test_allocation.py` integration tests, and the rest of the 82-test regression suite). The deterministic `cluster=Property(0)` literal draws nothing from the rng, so non-experiment runs are byte-identical to before.

- [ ] **Step 5: Commit**

```bash
git add sim/agents.py sim/engine.py tests/test_variant.py tests/test_allocation.py
git commit -m "feat(allocation): wire store into Market; User.cluster; drop legacy variant"
```

---

## Task 7: Public exports

**Files:**
- Modify: `sim/__init__.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
def test_public_exports_are_importable():
    import sim
    from sim import (Experiment, SimpleRandomization, ClusterRandomization,
                     Switchback, AssignmentStore, Assignment, bucket)
    for name in ("Experiment", "SimpleRandomization", "ClusterRandomization",
                 "Switchback", "AssignmentStore", "Assignment", "bucket"):
        assert name in sim.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py::test_public_exports_are_importable -q`
Expected: FAIL with `ImportError: cannot import name 'Experiment' from 'sim'`.

- [ ] **Step 3: Write minimal implementation**

Replace the contents of `sim/__init__.py` below the module docstring with:

```python
from sim.actions import negotiate_action
from sim.allocation import (Assignment, AssignmentStore, ClusterRandomization,
                            Experiment, SimpleRandomization, Switchback, bucket)
from sim.engine import Marketplace
from sim.pricing import default_pricing
from sim.spec import MarketplaceSpec, Property
from sim.willingness import default_willingness

__all__ = [
    "Marketplace", "MarketplaceSpec", "Property",
    "negotiate_action", "default_pricing", "default_willingness",
    "Experiment", "SimpleRandomization", "ClusterRandomization", "Switchback",
    "AssignmentStore", "Assignment", "bucket",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/__init__.py tests/test_allocation.py
git commit -m "feat(allocation): export allocation public API from sim"
```

---

## Task 8: Final regression + determinism + smoke verification

**Files:**
- Test only (no production code changes)

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: PASS — all tests green (82 pre-existing minus the 3 old `test_variant.py` tests, plus the rewritten `test_variant.py` (3) and the new `test_allocation.py` suite).

- [ ] **Step 2: Determinism sweep across designs**

Run this one-off check (paste into a shell):

```bash
python - <<'PY'
from datetime import datetime
from sim import (Marketplace, MarketplaceSpec, Property, Experiment,
                 ClusterRandomization, Switchback)
from scipy.stats import randint

def uplift(agent, listing, market):
    base = listing.quality * agent.value_factor
    return base * 1.2 if market.variant(agent, "e") == "B" else base

designs = {
    "cluster":    dict(experiments=[Experiment(key="e", variants={"CONTROL": .5, "B": .5},
                                               strategy=ClusterRandomization())],
                       cluster=Property(randint(0, 10))),
    "switchback": dict(experiments=[Experiment(key="e", variants={"CONTROL": .5, "B": .5},
                                               strategy=Switchback(period=1.0))]),
}
for name, extra in designs.items():
    def build():
        return Marketplace.from_spec(MarketplaceSpec(
            start=datetime(2026, 1, 1), n_seed_users=300, until=6.0, seed=5,
            willingness=uplift, **extra))
    a = [(e.event_type, e.actor_id, e.entity_id) for e in build().run()]
    b = [(e.event_type, e.actor_id, e.entity_id) for e in build().run()]
    print(f"{name:11s} identical={a == b}  events={len(a)}")
PY
```

Expected: each line prints `identical=True` with a non-trivial event count.

- [ ] **Step 3: Smoke harness still runs**

Run: `python scripts/run_slice.py`
Expected: runs to completion and reports `reproducible: True` (the harness uses no experiments, so behavior is unchanged).

- [ ] **Step 4: Commit (if any doc/notes changed; otherwise skip)**

No code changes in this task. If `python -m pytest -q` is green and the determinism sweep prints `identical=True`, the feature is complete. Proceed to the final code review and `superpowers:finishing-a-development-branch`.

---

## Notes for the implementer

- **Do not edit `classes.py`** (frozen legacy engine) or `func.py` at the repo root.
- **Do not touch the action funnel** (`sim/actions.py`) or `run_session`. Allocation is lookup-only; exposure (the action-consequence layer) is a separate future spec.
- **Interpreter:** use `python` (conda base), not `python3`.
- **`sim/allocation.py` must not import from other `sim/` modules** — keep it dependency-free so `sim/spec.py` and `sim/engine.py` can import it without cycles.
- If the determinism sweep ever prints `identical=False`, the most likely cause is an rng draw that varies with allocation; resolution must use only md5 + sim-time. Stop and fix before continuing.
