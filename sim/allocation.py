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
from dataclasses import dataclass

import numpy as np


def bucket(key, variants, salt):
    """Map a string key to a variant by content hashing.

    Same (key, salt) -> same variant, always; order-independent and stable across
    processes (this is why we use md5, not Python's process-salted ``hash()``).
    ``variants`` is {name: weight}; weights need not sum to 1 (they are normalized).
    """
    if not variants:
        raise ValueError("variants must be a non-empty {name: weight} mapping")
    names = list(variants)
    weights = np.array([variants[n] for n in names], dtype=float)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("variant weights must be non-negative and sum to a positive value")
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF                 # deterministic uniform in [0, 1]
    cum = np.cumsum(weights)
    cum /= cum[-1]
    return names[min(int(np.searchsorted(cum, x)), len(names) - 1)]


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
        if period <= 0:
            raise ValueError("Switchback period must be > 0")
        self.period = period
        self.per_cluster = per_cluster

    def window(self, t):
        return int(t // self.period)

    def assign(self, unit_key, cluster_key, window, variants, salt):
        key = f"{cluster_key}:{window}" if self.per_cluster else str(window)
        return bucket(key, variants, salt)

    def window_bounds(self, window):
        return (window * self.period, (window + 1) * self.period)


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
    auto_expose: bool = True             # True: reading the variant also logs an exposure (Eppo default)

    def __post_init__(self):
        if self.strategy is None:
            self.strategy = SimpleRandomization()
        if self.salt is None:
            self.salt = self.key
        if self.subject_key is None:
            self.subject_key = lambda s: s.id
        if self.cluster_key is None:
            self.cluster_key = lambda s: getattr(s, "cluster", 0)


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


class AssignmentStore:
    """Authoritative cache + append-only ledger. Written once per allocation,
    queried cheaply thereafter. The event stream is a projection of this, not the
    truth.
    """

    def __init__(self, experiments, market):
        keys = [e.key for e in experiments]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"duplicate experiment keys: {dupes}")
        self._exp = {e.key: e for e in experiments}
        self._market = market
        self._cache = {}     # (exp_key, subject_id, window) -> Assignment   <- O(1) lookup
        self._ledger = []    # append-only list[Assignment]                  <- the persistent truth
        self._exposed = set()        # {(exp_key, subject_id, window)} -> exposed once per window
        self._exposure_ledger = []   # append-only list[Exposure]

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
