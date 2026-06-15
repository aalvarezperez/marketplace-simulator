# Phase 3 · Plan 4 — Seller Listing Pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the regression-based `EndogenousPrice` (which inflated prices to ~2× value) with a decoupled `pricing` callable. Default: a seller prices a new listing at the **median ask of the K nearest-quality live listings**; cold-start (thin market) → a quality-anchored prior with lognormal noise. Fixes the inflation, preserves the price-out dynamic, and is swappable per the same pattern as `willingness`.

**Architecture:** `pricing(seller, quality, market, rng) -> price` is a callable on `MarketplaceSpec` (default `default_pricing`). Sellers read the **observable asks** (live listings) — realistic, since sale prices aren't visible — and take a robust **local median** (same category later; nearby-quality now), which is a stable `k=1` fixed point with no extrapolation. `EndogenousPrice`/`LinearPriceModel`/`fit_price_model` are deleted. Self-correction via liquidity/time-to-sold and a platform-recommended-price strategy are future plans the callable already supports.

**Tech Stack:** Python 3, SimPy, numpy, pytest. `python` = conda base, from repo root. Resolves Phase 3 spec §10.

---

## File structure
- **Modify** `sim/pricing.py` — delete `EndogenousPrice`/`LinearPriceModel`/`fit_price_model`; add `default_pricing(seller, quality, market, rng, k=10, prior_sigma=0.3)`.
- **Modify** `sim/spec.py` — drop the `EndogenousPrice` import + `listing_price` Property field (+ its `__post_init__` line); add `pricing: object = default_pricing`.
- **Modify** `sim/engine.py` — `Market.__init__` stores `self.pricing = spec.pricing`; `create_listing_for` and the `from_spec` seeding loop call `self.pricing(user, quality, self, rng)`.
- **Rewrite** `tests/test_pricing.py` — for `default_pricing` (drop the EndogenousPrice tests).

---

### Task 1: `default_pricing` (add alongside; suite stays green)

**Files:** Modify `sim/pricing.py`; Create `tests/test_pricing_local.py`.

- [ ] **Step 1: failing test** — `tests/test_pricing_local.py`:
```python
import numpy as np

from sim.pricing import default_pricing


class _L:
    def __init__(self, quality, price):
        self.quality = quality
        self.price = price
        self.is_live = True


class _M:
    def __init__(self, listings):
        self.listings = listings


def test_cold_start_prior_is_quality_anchored_and_positive():
    m = _M([])
    vals = [default_pricing(None, 500.0, m, np.random.default_rng(s)) for s in range(50)]
    assert all(v > 0 for v in vals)
    assert 250 < np.median(vals) < 1000          # centered near quality=500, with spread
    assert np.std(vals) > 0


def test_uses_median_of_k_nearest_quality_when_market_is_deep():
    # 12 listings; price = quality so nearest-quality medians track quality, robust to an outlier
    listings = [_L(q, q) for q in range(100, 1300, 100)]   # 12 of them
    listings.append(_L(500.0, 99999.0))                    # outlier ask
    m = _M(listings)
    price = default_pricing(None, 500.0, m, np.random.default_rng(0), k=10)
    assert price < 2000                                    # median ignores the 99999 outlier
    assert 300 < price < 800                               # ~ near quality 500


def test_price_increases_with_quality_in_deep_market():
    listings = [_L(q, q) for q in range(100, 1300, 100)]
    m = _M(listings)
    lo = default_pricing(None, 200.0, m, np.random.default_rng(0), k=6)
    hi = default_pricing(None, 1000.0, m, np.random.default_rng(0), k=6)
    assert hi > lo
```

- [ ] **Step 2: run** `python -m pytest tests/test_pricing_local.py -v` → FAIL (`default_pricing` missing).

- [ ] **Step 3: implement** — add to `sim/pricing.py` (keep existing code for now; it's removed in Task 2):
```python
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
```
(`import numpy as np` is already at the top of `sim/pricing.py`.)

- [ ] **Step 4: run** `python -m pytest tests/test_pricing_local.py -v` (expect 3 passed), then FULL suite `python -m pytest -q` — no regressions (EndogenousPrice still in place and used).

- [ ] **Step 5: commit**
```bash
git add sim/pricing.py tests/test_pricing_local.py
git commit -m "feat(sim): default_pricing = median of comparable asks (Phase 3 Plan 4)"
```

---

### Task 2: wire `pricing` into spec/engine; delete `EndogenousPrice`

**Files:** Modify `sim/spec.py`, `sim/engine.py`, `sim/pricing.py`; Rewrite `tests/test_pricing.py`.

- [ ] **Step 1: failing test** — rewrite `tests/test_pricing.py` entirely to:
```python
from datetime import datetime

import numpy as np

from sim.engine import Marketplace
from sim.spec import MarketplaceSpec


def test_spec_default_pricing_is_callable():
    from sim.pricing import default_pricing
    s = MarketplaceSpec(start=datetime(2026, 1, 1))
    assert s.pricing is default_pricing


def test_prices_track_quality_without_inflation():
    mkt = Marketplace.from_spec(MarketplaceSpec(
        start=datetime(2026, 1, 1), n_seed_users=300, until=5.0, seed=1))
    mkt.run()
    ls = mkt.market.listings
    q = np.array([l.quality for l in ls], dtype=float)
    p = np.array([l.price for l in ls], dtype=float)
    ratio = np.median(p / q)
    assert 0.6 < ratio < 1.6            # NOT the old ~2.0 inflation
    assert np.corrcoef(q, p)[0, 1] > 0.3


def test_default_marketplace_now_converts():
    # with prices near value (not 2x), willingness (value_factor ~ 1) clears some asks
    mkt = Marketplace.from_spec(MarketplaceSpec(
        start=datetime(2026, 1, 1), n_seed_users=300, until=5.0, seed=1))
    events = mkt.run()
    n_tx = sum(1 for e in events if e.event_type == "transaction")
    assert n_tx > 0                      # direct-buy conversion is no longer ~0


def test_custom_pricing_callable_is_used():
    mkt = Marketplace.from_spec(MarketplaceSpec(
        start=datetime(2026, 1, 1), n_seed_users=20, until=2.0, seed=1,
        pricing=lambda seller, quality, market, rng: 7.0))
    mkt.run()
    assert all(l.price == 7.0 for l in mkt.market.listings)
```

- [ ] **Step 2: run** `python -m pytest tests/test_pricing.py -v` → FAIL (`MarketplaceSpec` has no `pricing`; still uses `listing_price`).

- [ ] **Step 3: implement.**
(a) `sim/spec.py`: change the import `from sim.pricing import EndogenousPrice` to `from sim.pricing import default_pricing`. Delete the `listing_price` field and its `self.listing_price = _as_property(self.listing_price)` line in `__post_init__`. Add a field (with the scalar fields, e.g. after `willingness`): `pricing: object = default_pricing`.
(b) `sim/engine.py`:
- In `Market.__init__`, add `self.pricing = spec.pricing` (with the other field assignments).
- In `create_listing_for`, replace the price line:
```python
    def create_listing_for(self, user, rng):
        quality = self.spec.listing_quality.draw(rng)
        price = self.pricing(user, quality, self, rng)
        listing = self.add_listing(quality=quality, price=price, seller_id=user.id)
        self.emit("list", actor_id=user.id, entity_id=listing.id)
        return listing
```
- In `Marketplace.from_spec`, the seeding loop becomes:
```python
        for _ in range(spec.n_seed_users):
            user = market.spawn_user()
            for _ in range(int(spec.listings_per_user.draw(rng))):
                quality = spec.listing_quality.draw(rng)
                price = market.pricing(user, quality, market, rng)
                market.add_listing(quality=quality, price=price, seller_id=user.id)
```
(c) `sim/pricing.py`: DELETE `EndogenousPrice`, `LinearPriceModel`, and `fit_price_model` (keep only `default_pricing` and the `numpy` import).

- [ ] **Step 4: run** the FULL suite `python -m pytest -q`. ALL must pass — including the rewritten `tests/test_pricing.py`, plus reproducibility, no-threads, stock, willingness, lifecycle, settlement. Grep to confirm no stragglers: `grep -rn EndogenousPrice sim/ tests/` returns nothing. If a real failure occurs, report BLOCKED — don't weaken tests.

- [ ] **Step 5: verify the fix end-to-end:**
```bash
python - <<'PY'
from datetime import datetime
from collections import Counter
import numpy as np
from sim.engine import Marketplace
from sim.spec import MarketplaceSpec
mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026,1,1), n_seed_users=300, until=5.0, seed=1))
ev = mkt.run()
ls = mkt.market.listings
q = np.array([l.quality for l in ls]); p = np.array([l.price for l in ls])
print("price/quality median:", round(float(np.median(p/q)),2), "(was ~2.0)")
print("transactions:", Counter(e.event_type for e in ev)["transaction"], "(was ~0)")
PY
```
Expected: ratio ≈ 1 (not 2), transactions > 0. Record both in the report.

- [ ] **Step 6: commit**
```bash
git add sim/spec.py sim/engine.py sim/pricing.py tests/test_pricing.py
git commit -m "feat(sim): seller pricing via pricing callable; delete EndogenousPrice (Phase 3 Plan 4)"
```

---

## Self-Review
- **Spec coverage:** decoupled pricing callable + default median-of-comparable-asks + cold-start prior → Tasks 1–2; resolves §10 inflation (verified ratio ≈ 1, conversion > 0); category-scoping of comparables is deferred to the sub-markets plan (the `seller` arg reserves the hook); liquidity correction + platform-recommended price are future plans.
- **Placeholders:** none — complete code, exact commands, an end-to-end numeric check with expected before/after.
- **Type/name consistency:** `default_pricing(seller, quality, market, rng, k=10, prior_sigma=0.3)`, `MarketplaceSpec.pricing` (callable, default `default_pricing`), `Market.pricing`, `create_listing_for`/seeding call `self.pricing(user, quality, self, rng)`. `EndogenousPrice` fully removed (grep clean). The `listing_quality` Property stays; only `listing_price` is dropped.
- **Determinism:** median is deterministic; the cold-start prior is the only rng draw (warm-up); reproducibility tests guard it.

## Note
Removing the `listing_price` field is a breaking spec change (anyone passing `listing_price=...` must switch to `pricing=...`). Acceptable — we're pre-v2.0 on the experimental branch.
