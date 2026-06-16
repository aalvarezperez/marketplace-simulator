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
