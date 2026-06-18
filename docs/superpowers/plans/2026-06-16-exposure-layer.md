# Exposure Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exposure layer on top of the shipped allocation engine — Eppo-style, where reading a unit's variant logs an exposure as a side effect, the exposure point is the call site, and `auto_expose=False` is a peek-without-logging escape hatch.

**Architecture:** Layer exposure on top of allocation **without touching `AssignmentStore.resolve`** (allocation stays pristine). The store gains a separate exposure ledger + `expose()` (resolve, then log an exposure once per `(exp, subject, window)`) + a `read()` router (auto-expose iff the experiment opts in). `Market.variant` routes through `read()`; `Market.expose` is the explicit surface. The action funnel is untouched.

**Tech Stack:** Python 3.10+, `dataclasses`, `numpy`, SimPy, `pytest`. Interpreter = the conda base `python`. Run tests with `python -m pytest` from the repo root.

**Spec:** `docs/superpowers/specs/2026-06-16-exposure-layer-design.md`

**Invariants to hold:** deterministic (exposure adds no randomness; `exposure` event emitted right after `assignment` at the same sim-time); single seeded `numpy` rng; single-threaded; counter ids; `classes.py` frozen; **`AssignmentStore.resolve` and the action funnel (`sim/actions.py`, `run_session`) are NOT changed**. All existing 108 tests must stay green.

**Helpers already present in `tests/test_allocation.py`** (reuse them, do not redefine): `_Subj` (has `.id`, `.cluster`), `_FakeMarket` (records `(event_type, actor_id, payload)` tuples in `.events`, has `.emit`), `_store(experiments)` (returns `(AssignmentStore, _FakeMarket)`). Also already imported there: `Experiment`, `SimpleRandomization`, `ClusterRandomization`, `Switchback`, `Assignment`, `AssignmentStore`, `datetime`, `MarketplaceSpec`, `Property`, `Marketplace`, `pytest`.

---

## File Structure

- **Modify `sim/allocation.py`** — add `Experiment.auto_expose` field; add `Exposure` frozen dataclass; extend `AssignmentStore.__init__` with `_exposed` + `_exposure_ledger`; add `expose()`, `exposures()`, `read()`. `resolve()` unchanged.
- **Modify `sim/engine.py`** — `Market.variant` delegates to `store.read`; add `Market.expose`.
- **Modify `sim/__init__.py`** — export `Exposure`.
- **Modify `tests/test_allocation.py`** — append exposure tests (store-level and Market-level).

---

## Task 1: Data model — `Experiment.auto_expose` + `Exposure`

**Files:**
- Modify: `sim/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
from sim.allocation import Exposure


def test_experiment_auto_expose_defaults_true():
    e = Experiment(key="wtp", variants={"CONTROL": 0.5, "B": 0.5})
    assert e.auto_expose is True
    e2 = Experiment(key="reco", variants={"A": 1.0}, auto_expose=False)
    assert e2.auto_expose is False


def test_exposure_row_fields():
    ex = Exposure(experiment="wtp", subject_id=7, variant="B", cluster=3,
                  window=None, exposed_at=2.5)
    assert (ex.experiment, ex.subject_id, ex.variant, ex.cluster,
            ex.window, ex.exposed_at) == ("wtp", 7, "B", 3, None, 2.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py::test_experiment_auto_expose_defaults_true tests/test_allocation.py::test_exposure_row_fields -q`
Expected: FAIL — `AttributeError: 'Experiment' object has no attribute 'auto_expose'` and `ImportError: cannot import name 'Exposure'`.

- [ ] **Step 3: Write minimal implementation**

In `sim/allocation.py`, add the `auto_expose` field to the `Experiment` dataclass. The field block currently ends with `cluster_key`; add `auto_expose` after it:

```python
    subject_key: object = None           # subject -> hashing key; default lambda s: s.id
    cluster_key: object = None           # subject -> cluster key; default lambda s: getattr(s,'cluster',0)
    auto_expose: bool = True             # True: reading the variant also logs an exposure (Eppo default)
```

Then add the `Exposure` dataclass immediately AFTER the existing `Assignment` dataclass (after its `valid_to` field, before `class AssignmentStore:`):

```python
@dataclass(frozen=True)
class Exposure:
    """An exposure row: a unit actually encountered experiment E in this window.

    By default emitted coincident with allocation (reading the variant exposes you);
    with auto_expose=False it is emitted only when market.expose is called at the
    chosen surface, so exposure becomes a strict subset of allocation.
    """
    experiment: str
    subject_id: object
    variant: str
    cluster: object
    window: object            # None for time-invariant designs; window index for switchback
    exposed_at: float         # sim-time of first exposure (== assigned_at when auto-exposed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py::test_experiment_auto_expose_defaults_true tests/test_allocation.py::test_exposure_row_fields -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add sim/allocation.py tests/test_allocation.py
git commit -m "feat(exposure): Experiment.auto_expose field + Exposure row"
```

---

## Task 2: `AssignmentStore` exposure methods (`expose`, `exposures`, `read`)

**Files:**
- Modify: `sim/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
def test_expose_logs_assignment_and_exposure_once():
    exp = Experiment(key="wtp", variants={"CONTROL": 0.5, "B": 0.5})
    store, m = _store([exp])
    s = _Subj(id=1, cluster=0)
    v1 = store.expose("wtp", s, time=0.0)
    v2 = store.expose("wtp", s, time=0.5)         # same window -> idempotent
    assert v1 == v2 and v1 in ("CONTROL", "B")
    assert len(store.exposures()) == 1            # exposure logged once
    assert len(store.ledger()) == 1              # allocation still once
    assert sum(1 for e in m.events if e[0] == "exposure") == 1
    assert sum(1 for e in m.events if e[0] == "assignment") == 1
    ex = store.exposures()[0]
    assert ex.experiment == "wtp" and ex.subject_id == 1 and ex.exposed_at == 0.0


def test_expose_unknown_or_inactive_logs_no_exposure():
    exp = Experiment(key="e", variants={"A": 1.0}, start=5.0, end=10.0)
    store, m = _store([exp])
    assert store.expose("missing", _Subj(1), time=0.0) is None      # unknown experiment
    assert store.expose("e", _Subj(1), time=0.0) is None            # before start
    assert store.exposures() == []
    assert not [e for e in m.events if e[0] == "exposure"]


def test_expose_ineligible_logs_no_exposure():
    exp = Experiment(key="e", variants={"A": 1.0},
                     eligibility=lambda subj, mkt: subj.id % 2 == 0)
    store, m = _store([exp])
    assert store.expose("e", _Subj(3), time=0.0) is None
    assert store.exposures() == []
    assert store.expose("e", _Subj(2), time=0.0) == "A"
    assert len(store.exposures()) == 1


def test_switchback_exposes_once_per_window():
    exp = Experiment(key="sb", variants={"CONTROL": 0.5, "B": 0.5},
                     strategy=Switchback(period=1.0))
    store, m = _store([exp])
    s = _Subj(id=1)
    store.expose("sb", s, time=0.2)               # window 0
    store.expose("sb", s, time=0.7)               # window 0 -> idempotent
    store.expose("sb", s, time=1.3)               # window 1 -> new exposure
    assert len(store.exposures()) == 2
    assert sorted(ex.window for ex in store.exposures()) == [0, 1]


def test_read_auto_exposes_only_when_opted_in():
    auto = Experiment(key="auto", variants={"A": 1.0})                  # auto_expose=True
    quiet = Experiment(key="quiet", variants={"A": 1.0}, auto_expose=False)
    store, m = _store([auto, quiet])
    store.read("auto", _Subj(1), time=0.0)
    store.read("quiet", _Subj(2), time=0.0)
    # both allocated...
    assert len(store.ledger()) == 2
    # ...but only the auto_expose experiment logged an exposure
    exposed = {ex.experiment for ex in store.exposures()}
    assert exposed == {"auto"}
    # the quiet one exposes only on an explicit expose()
    store.expose("quiet", _Subj(2), time=0.0)
    assert {ex.experiment for ex in store.exposures()} == {"auto", "quiet"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py -k "expose or read_auto or switchback_exposes" -q`
Expected: FAIL — `AttributeError: 'AssignmentStore' object has no attribute 'expose'`.

- [ ] **Step 3: Write minimal implementation**

In `sim/allocation.py`, extend `AssignmentStore.__init__`. It currently ends with:

```python
        self._cache = {}     # (exp_key, subject_id, window) -> Assignment   <- O(1) lookup
        self._ledger = []    # append-only list[Assignment]                  <- the persistent truth
```

Add two lines after `self._ledger = []`:

```python
        self._exposed = set()        # {(exp_key, subject_id, window)} -> exposed once per window
        self._exposure_ledger = []   # append-only list[Exposure]
```

Then add three methods to `AssignmentStore`. Insert them immediately AFTER the existing `ledger` method (the last method in the class):

```python
    def expose(self, exp_key, subject, time, default=None):
        """Allocate (via resolve) and log an exposure once per (exp, subject, window).

        Returns the variant. Logs nothing when the unit is not actually in the
        experiment (unknown / inactive / ineligible -> resolve left no cache entry).
        Idempotent per window: repeat calls return the variant but add no new row/event.
        """
        variant = self.resolve(exp_key, subject, time, default)
        exp = self._exp.get(exp_key)
        if exp is None:
            return variant
        window = exp.strategy.window(time)
        ckey = (exp_key, subject.id, window)
        if ckey not in self._cache:           # not actually allocated (inactive/ineligible)
            return variant
        if ckey not in self._exposed:
            self._exposed.add(ckey)
            a = self._cache[ckey]             # the Assignment resolve just created/returned
            ex = Exposure(a.experiment, a.subject_id, a.variant, a.cluster, a.window, time)
            self._exposure_ledger.append(ex)
            self._market.emit("exposure", actor_id=subject.id, payload={
                "experiment": a.experiment, "variant": a.variant,
                "cluster": a.cluster, "window": a.window,
            })
        return variant

    def exposures(self):
        """The full append-only exposure record (for export to a dataframe)."""
        return list(self._exposure_ledger)

    def read(self, exp_key, subject, time, default=None):
        """The default public read path: allocate, and auto-expose iff the experiment
        opts in (auto_expose). Keeps the routing inside the store so the engine never
        touches private state."""
        exp = self._exp.get(exp_key)
        if exp is not None and exp.auto_expose:
            return self.expose(exp_key, subject, time, default)
        return self.resolve(exp_key, subject, time, default)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py -q`
Expected: PASS (all green — the new exposure tests plus every prior allocation test, since `resolve` is unchanged).

- [ ] **Step 5: Commit**

```bash
git add sim/allocation.py tests/test_allocation.py
git commit -m "feat(exposure): AssignmentStore expose/exposures/read (resolve untouched)"
```

---

## Task 3: Market surface — `variant` routes to `read`; add `expose`

**Files:**
- Modify: `sim/engine.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
def test_market_variant_auto_exposes_end_to_end():
    def uplift(agent, listing, market):
        base = listing.quality * agent.value_factor
        return base * 1.2 if market.variant(agent, "wtp") == "B" else base

    def build():
        spec = MarketplaceSpec(
            start=datetime(2026, 1, 1), n_seed_users=300, until=5.0, seed=11,
            willingness=uplift,
            experiments=[Experiment(key="wtp", variants={"CONTROL": 0.5, "B": 0.5})])
        return Marketplace.from_spec(spec)

    mkt = build()
    events = mkt.run()
    assert [e for e in events if e.event_type == "assignment"]
    assert [e for e in events if e.event_type == "exposure"]        # auto-exposed
    assert mkt.market.assignment_store.exposures()                  # ledger populated
    # Determinism holds with exposure events in the stream.
    a = [(e.event_type, e.actor_id, e.entity_id) for e in build().run()]
    b = [(e.event_type, e.actor_id, e.entity_id) for e in build().run()]
    assert a == b


def test_market_auto_expose_false_defers_exposure():
    spec = MarketplaceSpec(
        start=datetime(2026, 1, 1), n_seed_users=10, until=1.0, seed=3,
        experiments=[Experiment(key="reco", variants={"CONTROL": 0.5, "B": 0.5},
                                auto_expose=False)])
    mkt = Marketplace.from_spec(spec)
    store = mkt.market.assignment_store
    user = mkt.market.users[0]
    # Reading the variant allocates but does NOT expose.
    v = mkt.market.variant(user, "reco")
    assert v in ("CONTROL", "B")
    assert store.ledger() and not store.exposures()
    # Explicit expose at the surface logs the exposure.
    v2 = mkt.market.expose(user, "reco")
    assert v2 == v
    assert len(store.exposures()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py::test_market_auto_expose_false_defers_exposure -q`
Expected: FAIL — `AttributeError: 'Market' object has no attribute 'expose'` (and the auto-expose test finds no `exposure` events because `variant` still calls `resolve`).

- [ ] **Step 3: Write minimal implementation**

In `sim/engine.py`, replace the existing `Market.variant` method:

```python
    def variant(self, subject, exp_key, default=None):
        """Look up ``subject``'s variant for experiment ``exp_key`` at the current
        sim-time. Resolves + caches + logs on first read (per switchback window);
        O(1) thereafter. Returns ``default`` when the experiment is unknown,
        inactive, or the subject is ineligible. This is the whole allocation surface."""
        return self.assignment_store.resolve(exp_key, subject, self.env.now, default)
```

with this version (routes through `read`, and adds `expose`):

```python
    def variant(self, subject, exp_key, default=None):
        """Look up ``subject``'s variant for ``exp_key`` at the current sim-time.
        By default (auto_expose=True) reading also logs an exposure (the Eppo
        getAssignment model); for auto_expose=False experiments this allocates only.
        Returns ``default`` when the experiment is unknown, inactive, or ineligible."""
        return self.assignment_store.read(exp_key, subject, self.env.now, default)

    def expose(self, subject, exp_key, default=None):
        """Explicitly expose ``subject`` to ``exp_key`` at the current sim-time — the
        surface for auto_expose=False experiments. Allocates if needed, logs the
        exposure once per (exp, subject, window), and returns the variant."""
        return self.assignment_store.expose(exp_key, subject, self.env.now, default)
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — all green. The two new Market tests pass; the prior end-to-end allocation test (`test_market_variant_lookup_and_ledger_end_to_end`) still passes (it now also produces `exposure` events, which its assertions tolerate, and determinism still holds).

- [ ] **Step 5: Commit**

```bash
git add sim/engine.py tests/test_allocation.py
git commit -m "feat(exposure): Market.variant routes through read; add Market.expose"
```

---

## Task 4: Export `Exposure`

**Files:**
- Modify: `sim/__init__.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_allocation.py`:

```python
def test_exposure_is_exported():
    import sim
    from sim import Exposure
    assert "Exposure" in sim.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_allocation.py::test_exposure_is_exported -q`
Expected: FAIL — `ImportError: cannot import name 'Exposure' from 'sim'`.

- [ ] **Step 3: Write minimal implementation**

In `sim/__init__.py`, the allocation import currently reads:

```python
from sim.allocation import (Assignment, AssignmentStore, ClusterRandomization,
                            Experiment, SimpleRandomization, Switchback, bucket)
```

Replace it with (adds `Exposure`):

```python
from sim.allocation import (Assignment, AssignmentStore, ClusterRandomization,
                            Experiment, Exposure, SimpleRandomization, Switchback,
                            bucket)
```

And in the `__all__` list, add `"Exposure"` next to `"Assignment"`. The allocation line of `__all__` currently reads:

```python
    "AssignmentStore", "Assignment", "bucket",
```

Replace it with:

```python
    "AssignmentStore", "Assignment", "Exposure", "bucket",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_allocation.py::test_exposure_is_exported -q`
Expected: PASS. Also verify: `python -c "from sim import Exposure; print('ok')"` prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add sim/__init__.py tests/test_allocation.py
git commit -m "feat(exposure): export Exposure from sim"
```

---

## Task 5: Final regression + determinism verification

**Files:**
- Test only (no production code changes)

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: PASS — all green.

- [ ] **Step 2: Determinism + exposure sweep across designs**

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
    "simple":     dict(experiments=[Experiment(key="e", variants={"CONTROL": .5, "B": .5})],
                       willingness=uplift),
    "cluster":    dict(experiments=[Experiment(key="e", variants={"CONTROL": .5, "B": .5},
                                               strategy=ClusterRandomization())],
                       cluster=Property(randint(0, 10)), willingness=uplift),
    "switchback": dict(experiments=[Experiment(key="e", variants={"CONTROL": .5, "B": .5},
                                               strategy=Switchback(period=1.0))], willingness=uplift),
}
for name, extra in designs.items():
    def build():
        return Marketplace.from_spec(MarketplaceSpec(
            start=datetime(2026, 1, 1), n_seed_users=300, until=6.0, seed=5, **extra))
    a = [(e.event_type, e.actor_id, e.entity_id) for e in build().run()]
    b = [(e.event_type, e.actor_id, e.entity_id) for e in build().run()]
    m = build(); m.run()
    asg = sum(1 for e in m.events if e.event_type == "assignment")
    exp = sum(1 for e in m.events if e.event_type == "exposure")
    print(f"{name:11s} identical={a == b}  assignment={asg}  exposure={exp}")
PY
```

Expected: each line prints `identical=True`, with `assignment` and `exposure` counts both > 0 (and equal, since these experiments auto-expose 1:1 with allocation in a single window — switchback may differ as windows turn).

- [ ] **Step 3: Smoke harness still runs**

Run: `python scripts/run_slice.py`
Expected: runs to completion and reports `reproducible: True` (the harness configures no experiments, so it emits neither assignment nor exposure events — behavior unchanged).

- [ ] **Step 4: No commit needed.** If the full suite is green and the sweep prints `identical=True`, the feature is complete. Proceed to the final code review and `superpowers:finishing-a-development-branch`.

---

## Notes for the implementer

- **Do not edit `classes.py`**, root `func.py`, the action funnel (`sim/actions.py`), or `run_session`.
- **Do not change `AssignmentStore.resolve`** — exposure layers strictly on top of it. If you find yourself editing `resolve`, stop: the design keeps allocation pristine.
- **Interpreter:** use `python` (conda base), not `python3`.
- Exposure adds no randomness. If the determinism sweep ever prints `identical=False`, something introduced rng/wall-clock into the exposure path — stop and fix.
- The `ckey in self._cache` check in `expose()` is how "really allocated" is distinguished from "default fallback" — keep it; it is robust even if a variant is named like the `default` value.
