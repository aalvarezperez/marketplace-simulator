import numpy as np


def default_pricing(seller, quality, market, rng, k=10, prior_sigma=0.3):
    """Price a new listing at the median ask of the K nearest-quality LIVE listings.

    Sellers read observable asks (sale prices aren't visible). A local median is
    robust to overpriced outliers and does not extrapolate, so it sits at the
    item's going rate (a stable fixed point) instead of running away. Cold-start
    (fewer than k live listings) falls back to a quality-anchored prior with
    lognormal noise to seed price dispersion. `seller` is unused for now; it's in
    the signature so a later version can scope comparables to the seller's category
    or apply a reservation floor. Deterministic: rng is only touched on cold-start.
    """
    live = [l for l in market.listings if l.is_live]
    if len(live) >= k:
        nearest = sorted(live, key=lambda l: abs(l.quality - quality))[:k]
        return float(np.median([l.price for l in nearest]))
    return float(quality * np.exp(rng.normal(0.0, prior_sigma)))
