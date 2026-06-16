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
