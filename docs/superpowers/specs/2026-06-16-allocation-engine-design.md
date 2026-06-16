# Allocation Engine — Design Spec

**Date:** 2026-06-16
**Branch:** `experimental/simpy-replatform` (no merge to `main` until the v2.0 release is greenlit)
**Status:** approved design; next step is an implementation plan via writing-plans.

---

## 1. Purpose

The simulator is a **data generator** and a **sandbox for experimentation methodology**. A researcher
should be able to pick a *randomization design* — simple per-unit, cluster, switchback, or one they
write themselves — run the marketplace under it, and get the resulting **allocation data** out. The
research payoff is studying how the design interacts with marketplace **interference** (treating buyer
A leaks to buyer B through shared inventory, stock depletion, and the emergent comparable-median
price), which is exactly why cluster/switchback designs exist. The engine already produces that
interference for free.

**This spec covers allocation only.** Allocation is the design's deterministic answer to *"which
variant is unit U in for experiment E at time T?"* The engine performs no statistics — estimation and
error analysis are the downstream consumer's job (R / notebook). The allocation the engine produces is
itself data.

### Allocation vs exposure (and why exposure is out of scope)

These are two layers and this spec is only the first:

- **Allocation** — a *property of a unit under a design*. Deterministic, looked up on demand, cached
  as the source of truth. Not an event in the funnel. **← this spec.**
- **Exposure** — a real *event*: the unit actually hit the treated surface. A **consequence of an
  action** (visited, viewed an item) that fires after the action, consults the allocation, and logs an
  exposure. Different timestamp, different table, usually a subset of allocated units. **← a separate
  follow-on spec. Not built here. No funnel changes here.**

---

## 2. Architecture overview

One new module, `sim/allocation.py`, holds the whole allocation subsystem. The runtime `Market` gains a
store and one lookup method; the spec gains two fields; the `User` gains a cluster and loses its scalar
`variant`. The action funnel is **not** touched.

```
sim/allocation.py   bucket(), AllocationStrategy built-ins, Experiment, Assignment, AssignmentStore
sim/spec.py         + experiments, + cluster; variant_weights -> "default" experiment shim
sim/engine.py       Market builds the store; + Market.variant(); draws cluster at spawn;
                    drops the per-event variant stamp; from_spec feeds experiments to the store
sim/agents.py       User gains `cluster`, loses `variant`
sim/__init__.py     export Experiment + the three strategies (+ Assignment, AssignmentStore, bucket)
tests/              rewrite test_variant.py; new test_allocation.py
```

Consumers of allocation (a variant-aware `willingness`/`pricing` callable today; the exposure layer
later) call `market.variant(subject, exp_key)`. Nothing in this spec auto-calls it from the funnel.

---

## 3. Deterministic bucketing + allocation strategies (`sim/allocation.py`)

### 3.1 `bucket`

The one primitive. Maps a string key to a variant by content hashing — no rng, order-independent,
stable across processes. (The engine's "no `hash()`" invariant forbids Python's process-salted
`hash()`; a content hash like md5 is deterministic and is the right tool here.)

```python
import hashlib
import numpy as np

def bucket(key, variants, salt):
    """Map a string key to a variant by hashing. Same (key, salt) -> same variant, always.

    `variants` is {name: weight}; weights need not sum to 1 (they are normalized).
    """
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF                 # deterministic uniform in [0, 1]
    names = list(variants)
    cum = np.cumsum(np.array([variants[n] for n in names], dtype=float))
    cum /= cum[-1]
    return names[min(int(np.searchsorted(cum, x)), len(names) - 1)]
```

### 3.2 The strategy protocol

A strategy *is the randomization design*. It is purely the design — `variants`, `salt`, and the active
window live on the `Experiment` and are passed in. A strategy implements:

- `window(time) -> hashable | None` — which time-window we are in. `None` for time-invariant designs.
- `assign(unit_key, cluster_key, window, variants, salt) -> variant` — the variant for this
  (unit, cluster, window).
- `window_bounds(window) -> (from, to)` — *optional*; the `[from, to)` sim-time span of a window for
  the validity record. Default `(None, None)`.

A user adds a new design (stratified, stepped-wedge, saturation, …) by writing a class with these
methods. No engine change.

### 3.3 Built-in strategies

```python
class SimpleRandomization:
    """Per-unit Bernoulli assignment. Time-invariant, sticky to the unit."""
    def window(self, t):
        return None
    def assign(self, unit_key, cluster_key, window, variants, salt):
        return bucket(str(unit_key), variants, salt)


class ClusterRandomization:
    """Every unit in a cluster shares the cluster's variant. Time-invariant, sticky to the cluster."""
    def window(self, t):
        return None
    def assign(self, unit_key, cluster_key, window, variants, salt):
        return bucket(str(cluster_key), variants, salt)


class Switchback:
    """The market (or each cluster) flips variant every `period` sim-days."""
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

`SimpleRandomization` and `ClusterRandomization` are the degenerate (time-invariant, `window() is
None`) case of the same machinery that `Switchback` makes time-varying.

---

## 4. `Experiment` (`sim/allocation.py`)

Declarative description of one experiment. No privileged entity type and no privileged baseline name —
both were hardcodes we removed. Granularity is decided by `subject_key`/`cluster_key` extractors plus
what the caller passes.

```python
from dataclasses import dataclass, field

@dataclass
class Experiment:
    key: str
    variants: dict                                   # {'CONTROL': .5, 'B': .5}; names are user-defined
    strategy: object = None                          # an AllocationStrategy; default SimpleRandomization()
    salt: str = None                                 # default = key (each experiment hashes independently)
    start: float = 0.0                               # active window, sim-days (inclusive)
    end: float = None                                # active window end, sim-days (exclusive); None = no end
    eligibility: object = None                       # predicate(subject, market) -> bool; default everyone
    subject_key: object = None                       # subject -> hashing key; default lambda s: s.id
    cluster_key: object = None                       # subject -> cluster key; default lambda s: getattr(s, "cluster", 0)

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

"Randomize by user" = pass the user (default extractors). "By cluster" = a `ClusterRandomization`
strategy (ignores `subject_key`). "By seller/listing/session" = pass that entity / supply extractors.
Targeting = `eligibility`. There is no `control` field: when an experiment is inactive, ineligible, or
absent, resolution returns the **caller's** `default` (see §6), default `None` = "not in this
experiment".

---

## 5. `Assignment` — the truth row (`sim/allocation.py`)

What gets persisted the moment a subject is allocated. Frozen.

```python
@dataclass(frozen=True)
class Assignment:
    experiment: str
    subject_id: object        # the hashed unit's id
    variant: str
    cluster: object
    window: object            # None for time-invariant designs; window index for switchback
    assigned_at: float        # sim-time of first resolution
    valid_from: float         # when this assignment takes effect (None = open)
    valid_to: float           # when it stops (None = open-ended)
```

This carries the "who got allocated, to what, when, and from-when-till-when" record the design is meant
to study.

---

## 6. `AssignmentStore` — the source of truth (`sim/allocation.py`)

The store is the **authoritative cache**, written once per allocation and queried cheaply forever
after. The event stream is *not* the truth; the store is. Assignment events are a projection of the
store's ledger, emitted for downstream export.

```python
class AssignmentStore:
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
        if hit is not None:                                  # already allocated -> read truth, no recompute
            return hit.variant
        if not (exp.start <= time and (exp.end is None or time < exp.end)):
            return default                                   # outside active window: not in experiment
        if exp.eligibility is not None and not exp.eligibility(subject, self._market):
            return default                                   # ineligible: not in experiment
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
        wfrom, wto = exp.strategy.window_bounds(window)
        # intersect the window span with the experiment's active window
        lo = exp.start if wfrom is None else max(exp.start, wfrom)
        hi = wto if exp.end is None else (exp.end if wto is None else min(exp.end, wto))
        return (lo, hi)

    def current(self, exp_key, subject_id):
        """Most recent Assignment for a subject in an experiment, or None."""
        hits = [a for (e, s, _w), a in self._cache.items() if e == exp_key and s == subject_id]
        return max(hits, key=lambda a: a.assigned_at) if hits else None

    def ledger(self):
        """The full append-only assignment record (for export to a dataframe)."""
        return list(self._ledger)
```

**Switchback falls out for free:** when the clock crosses into a new window the cache key changes, so a
fresh `Assignment` is resolved and appended. A subject that acts across three windows has three truth
rows; a sticky design has exactly one. Resolution touches only the hash and the sim-time — **no rng,
no clock mutation** — so it is fully deterministic.

---

## 7. Engine integration (`sim/engine.py`, `sim/agents.py`)

- **`Market.__init__`** builds the store: `self.assignment_store = AssignmentStore(spec.experiments,
  self)`; expose `self.experiments = spec.experiments`.
- **`Market.variant(subject, exp_key, default=None)`** → `self.assignment_store.resolve(exp_key,
  subject, self.env.now, default)`. This is the entire public lookup surface.
- **`spawn_user`** draws the cluster — `user.cluster = self.spec.cluster.draw(self.rng)` — and **no
  longer assigns `user.variant`**. `_assign_variant` is deleted.
- **`emit`** **no longer stamps `variant`** on events. Behavioral events stay lean; the `assignment`
  event is the only variant-bearing log.
- **`User`** gains `cluster: object = 0` and **loses `variant`**.

The action funnel and `run_session` are unchanged.

---

## 8. Backward compatibility

- **`variant_weights` shim.** `MarketplaceSpec.__post_init__`: if `experiments` is empty and
  `variant_weights` has more than one variant, synthesize
  `[Experiment(key="default", variants=variant_weights, strategy=SimpleRandomization())]`. A default
  single-`CONTROL` `variant_weights` synthesizes **nothing** (no experiment) — matching today's
  short-circuit, drawing nothing, generating no allocation rows.
- **`cluster` default is the literal `0`.** A literal `Property` does not touch the rng, so existing
  runs draw nothing new and stay **byte-identical**.
- **rng-stream change for explicit `variant_weights` splits.** The old engine consumed the rng at spawn
  (`rng.choice`) to assign a variant; allocation now uses content hashing and consumes no rng. So a run
  that set a real `variant_weights` split shifts its downstream rng stream. The common cases —
  all-`CONTROL` / no split — are unaffected. This is an accepted, documented change.

---

## 9. Determinism & invariants (held)

- Allocation uses md5 + sim-time only — reproducible and order-independent; *more* deterministic than an
  rng draw.
- Single seeded `numpy` rng still drives all sampling; the only new draw is `cluster` at spawn, and only
  when `cluster` is non-literal.
- Single-threaded; counter ids; in-memory store; `classes.py` untouched.
- Same spec + same seed → byte-identical event stream (for runs that don't set a real split; see §8).

---

## 10. Test plan (`tests/`)

**New `tests/test_allocation.py`:**
- `bucket` determinism: same `(key, salt)` → same variant; repeated calls equal.
- `bucket` balance: over many keys the split ≈ weights (loose bound); different `salt` → independent split.
- `SimpleRandomization` sticky: same subject → same variant at different times.
- `ClusterRandomization`: all subjects sharing a cluster get the same variant; it reflects cluster, not subject id.
- `Switchback`: variant is constant within a window and changes across windows; `window_bounds` correct.
- Store caching: a second resolve in the same window adds no ledger row; a switchback resolve in a new window adds one.
- Validity: `valid_from`/`valid_to` correct for sticky (= experiment window) and switchback (= window span ∩ experiment window).
- Eligibility gating: ineligible subject → `default`, **no** ledger row.
- Active window: `time < start` or `time >= end` → `default`, **no** ledger row.
- Determinism: a full run with one experiment + a variant-aware willingness callable → two runs byte-identical.

**Rewrite `tests/test_variant.py`** to the store model:
- No experiments configured → `market.variant(user, "anything")` is `None`; no `assignment` events emitted.
- `variant_weights` split → synthesizes the `default` experiment; over the seed users the ledger shows both variants, roughly balanced; reproducible across two builds.

---

## 11. Out of scope (explicit non-goals)

- **Exposure** — the action-consequence layer that fires post-visit/post-view, consults allocation, and
  logs exposures. Its own follow-on spec. No funnel changes here.
- **Treatment effects** — what a variant *does* to behavior is data-generation choice made by the user
  in a swappable callable (`willingness`/`pricing`); the engine defines none.
- **Statistics** — estimation, bias, Type-I/power, variance, design effect. All downstream (R/notebook).
- **Extra strategies** — stratified, stepped-wedge, saturation/two-sided. User-supplyable via the
  strategy protocol; not built here.
- **Listing/seller-level materialization beyond the extractors** — `subject_key`/`cluster_key` make
  other units expressible, but no dedicated seller/listing exposure plumbing is added here.

---

## 12. Open questions (resolve at plan time)

1. **Both `experiments` and `variant_weights` set** — prefer `experiments` and ignore `variant_weights`,
   or raise? (Lean: prefer `experiments`, ignore the legacy field.)
2. **`cluster` ergonomics** — ship only the `Property` field (user writes `Property(randint(0, K))`), or
   add a convenience `n_clusters: int` knob? (Lean: `Property` only, to keep the spec surface small.)
3. **`assignment` event time** — `assigned_at` is sim-time (float); the event's own `sim_time` is the
   calendar datetime. Confirm both are wanted in the export (they are redundant but convenient).
