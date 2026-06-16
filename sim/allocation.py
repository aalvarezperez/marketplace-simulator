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
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF                 # deterministic uniform in [0, 1]
    names = list(variants)
    cum = np.cumsum(np.array([variants[n] for n in names], dtype=float))
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
