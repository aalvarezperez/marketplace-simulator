# Phase 3 · Plan 5 — Liquidity Correction (stale-listing markdown) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Self-correcting prices via liquidity. An unsold listing marks its price down on a **seller-driven, heterogeneous** clock (each seller's own patience); the *up* direction emerges for free from the median-of-comparable pricing (cleared cheap stock raises the live-comparable median). Two-sided, with one explicit mechanism.

**Architecture:** A new per-seller disposition `seller_patience` (a `Property`, **normally distributed by default**, `norm(loc=until×0.2, scale=until×0.1)`, configurable, clamped to a positive floor). A per-listing `markdown_listing` SimPy process drops `price` by `markdown_pct` every `seller.patience` days while unsold; stops on sale/expiry. Markdowns flow into `default_pricing`'s comparable median, so a stale market drifts down and a liquid one drifts up — emergent two-sidedness. Deterministic (rng only for the patience draw at spawn).

**Tech Stack:** Python 3, SimPy, numpy, scipy, pytest. `python` = conda base, from repo root.

---

## File structure
- **Modify** `sim/spec.py` — import `norm`; add `seller_patience: Property = None` (defaulted in `__post_init__` to `norm(loc=until×0.2, scale=until×0.1)`); add `markdown_pct: float = 0.1`.
- **Modify** `sim/agents.py` — `User` gains `patience: float = 0.0`; add `MIN_PATIENCE` constant and the `markdown_listing` process.
- **Modify** `sim/engine.py` — `Market.__init__` stores `self.markdown_pct`; `spawn_user` draws `patience` (clamped); `add_listing` starts the markdown process for real sellers when `markdown_pct > 0`.
- **Create** `tests/test_markdown.py`.

---

### Task 1: `seller_patience` disposition + `markdown_listing` process

**Files:** Modify `sim/spec.py`, `sim/agents.py`, `sim/engine.py`; Create `tests/test_markdown.py`.

- [ ] **Step 1: failing test** — `tests/test_markdown.py`:
```python
from datetime import datetime

import numpy as np
import simpy

from sim.agents import User, markdown_listing
from sim.engine import Clock, Market
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec


def _market(seed=0, **kw):
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0, **kw)
    m = Market(env=env, rng=np.random.default_rng(seed), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    return env, m


def test_seller_patience_drawn_and_positive():
    env, m = _market(seed=1)
    u = m.spawn_user()
    assert isinstance(u.patience, float) and u.patience > 0


def test_default_patience_scales_with_until():
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), until=20.0)
    # default is norm(loc=until*0.2,...): mean ~4.0 days for until=20
    draws = [float(spec.seller_patience.draw(np.random.default_rng(s))) for s in range(200)]
    assert 2.5 < np.mean(draws) < 5.5


def test_unsold_listing_marks_down_over_time():
    env, m = _market(seed=2, markdown_pct=0.1)
    seller = m.spawn_user()
    seller.patience = 1.0                         # fixed patience for a clean assertion
    listing = m.add_listing(quality=500.0, price=1000.0, seller_id=seller.id)
    env.process(markdown_listing(env, listing, m, patience=1.0))
    env.run(until=3.5)                            # ~3 markdown steps
    assert listing.price < 1000.0                 # dropped
    assert any(e.event_type == "markdown" for e in m.recorder.events)


def test_sold_listing_stops_marking_down():
    env, m = _market(seed=3, markdown_pct=0.1)
    seller = m.spawn_user()
    listing = m.add_listing(quality=500.0, price=1000.0, seller_id=seller.id)
    env.process(markdown_listing(env, listing, m, patience=1.0))
    env.run(until=1.5)                            # one markdown
    p_after_one = listing.price
    listing.is_live = False                       # "sold"
    env.run(until=10.0)
    assert listing.price == p_after_one           # no further markdowns once not live
```

- [ ] **Step 2: run** `python -m pytest tests/test_markdown.py -v` → FAIL (`markdown_listing` / `seller_patience` / `User.patience` missing).

- [ ] **Step 3: implement.**
(a) `sim/spec.py`: add `norm` to the scipy import → `from scipy.stats import gamma, lognorm, norm, poisson`. Add fields to `MarketplaceSpec`:
```python
    seller_patience: Property = None    # days unsold before a markdown; default set in __post_init__
    markdown_pct: float = 0.1
```
In `__post_init__`, add (defaults to a normal scaled by the run length; honours a user override):
```python
        if self.seller_patience is None:
            self.seller_patience = Property(norm(loc=self.until * 0.2, scale=self.until * 0.1))
        else:
            self.seller_patience = _as_property(self.seller_patience)
```
(b) `sim/agents.py`: add `patience: float = 0.0` to `User`. Add a constant near the others: `MIN_PATIENCE = 0.25`. Add the process (after `listing_expiry`):
```python
def markdown_listing(env, listing, market, patience):
    """Seller-driven liquidity correction: while unsold, drop the price by
    market.markdown_pct every `patience` days. Stops on sale/expiry. The up-move
    is emergent (cleared cheap stock raises the comparable median)."""
    while listing.is_live:
        yield env.timeout(max(patience, MIN_PATIENCE))
        if listing.is_live:
            listing.price *= (1.0 - market.markdown_pct)
            market.emit("markdown", actor_id=listing.seller_id, entity_id=listing.id,
                        payload={"price": listing.price})
```
(c) `sim/engine.py`:
- In `Market.__init__`, add `self.markdown_pct = spec.markdown_pct`.
- In `spawn_user`, draw the seller's patience (clamped positive), alongside the other draws:
```python
        from sim.agents import MIN_PATIENCE
        user.patience = max(float(self.spec.seller_patience.draw(self.rng)), MIN_PATIENCE)
```
(If a top-level import is cleaner, import `MIN_PATIENCE` with the other `sim.agents` imports instead of inline.)
- Do NOT wire `add_listing` yet (Task 2). This task only adds the machinery + unit-tests the process directly.

- [ ] **Step 4: run** `python -m pytest tests/test_markdown.py -v` (expect 4 passed), then FULL suite `python -m pytest -q` — no regressions (markdown not yet started by `add_listing`; spawn now draws an extra rng value, but same-seed runs stay reproducible).

- [ ] **Step 5: commit**
```bash
git add sim/spec.py sim/agents.py sim/engine.py tests/test_markdown.py
git commit -m "feat(sim): seller_patience disposition + markdown_listing process (Phase 3 Plan 5)"
```

---

### Task 2: wire markdown into listing creation + verify the dynamic

**Files:** Modify `sim/engine.py`; Modify `tests/test_markdown.py` (append).

- [ ] **Step 1: failing test** — append to `tests/test_markdown.py`:
```python
def test_markdown_events_occur_in_full_run():
    from sim.engine import Marketplace
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=200, until=8.0, seed=1)
    events = Marketplace.from_spec(spec).run()
    assert any(e.event_type == "markdown" for e in events)


def test_markdown_disabled_when_pct_zero():
    from sim.engine import Marketplace
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=100, until=6.0, seed=1,
                           markdown_pct=0.0)
    events = Marketplace.from_spec(spec).run()
    assert not any(e.event_type == "markdown" for e in events)


def test_markdown_run_is_reproducible():
    from sim.engine import Marketplace
    def run():
        spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=120, until=6.0, seed=4)
        return [(e.event_type, e.actor_id, e.entity_id) for e in Marketplace.from_spec(spec).run()]
    assert run() == run()
```

- [ ] **Step 2: run** `python -m pytest tests/test_markdown.py -v` → the new `test_markdown_events_occur_in_full_run` FAILS (markdown not started in `add_listing` yet).

- [ ] **Step 3: implement** in `sim/engine.py` `add_listing` — start the markdown process for real sellers when enabled. After the existing `listing_expiry` start and before `return listing`:
```python
        if self.markdown_pct > 0:
            seller = self.get_user(seller_id)
            if seller is not None:
                from sim.agents import markdown_listing
                self.env.process(markdown_listing(self.env, listing, self, seller.patience))
```
(Or import `markdown_listing` at the top with the other `sim.agents` imports.)

- [ ] **Step 4: run** the FULL suite `python -m pytest -q`. ALL must pass (markdown events are additive; reproducibility/no-threads/stock/pricing/willingness/lifecycle/settlement hold). If a real failure occurs, report BLOCKED — don't weaken tests.

- [ ] **Step 5: verify the liquidity dynamic** — markdown should pull prices down when the market is illiquid (low buyer valuations) and less so when liquid:
```bash
python - <<'PY'
from datetime import datetime
from collections import Counter
import numpy as np
from scipy.stats import lognorm
from sim.engine import Marketplace
from sim.spec import MarketplaceSpec
for vf in (0.7, 1.3):     # low vs high demand (buyer value_factor scale)
    spec = MarketplaceSpec(start=datetime(2026,1,1), n_seed_users=300, until=8.0, seed=1,
                           value_factor=lognorm(s=0.3, scale=vf))
    mkt = Marketplace.from_spec(spec); ev = mkt.run()
    ls = mkt.market.listings
    ratio = np.median([l.price/l.quality for l in ls]) if ls else float('nan')
    n_md = Counter(e.event_type for e in ev)["markdown"]
    print(f"demand(value_factor scale)={vf}: median price/quality={ratio:.2f}, markdowns={n_md}")
PY
```
Expected: low-demand run shows MORE markdowns and a LOWER price/quality than the high-demand run — prices self-correcting toward what clears. Record both lines.

- [ ] **Step 6: commit**
```bash
git add sim/engine.py tests/test_markdown.py
git commit -m "feat(sim): start markdown on listing creation; liquidity self-correction (Phase 3 Plan 5)"
```

---

## Self-Review
- **Spec coverage:** seller-driven heterogeneous patience (per-seller `Property`) → Task 1; normally distributed default scaled to `until`, configurable → `__post_init__`; markdown mechanism + emergent up-move → Tasks 1–2; verified the demand-responsive dynamic → Task 2 Step 5.
- **Placeholders:** none — complete code, exact commands, an empirical dynamic check with an expected direction.
- **Type/name consistency:** `MarketplaceSpec.seller_patience` (Property) + `.markdown_pct` (float), `User.patience`, `MIN_PATIENCE`, `markdown_listing(env, listing, market, patience)`, `Market.markdown_pct`, `add_listing` starts the process for real sellers. The normal default lives in `__post_init__` because it needs `self.until`.
- **Determinism:** patience is one rng draw at spawn (clamped); markdown timer is rng-free; reproducibility tests guard it. `markdown_pct=0` cleanly disables.

## Notes
- **Floor:** `MIN_PATIENCE=0.25` days stops the normal's left tail from producing zero/negative patience (which would spam markdowns). 
- **Self-arresting:** markdown only runs while unsold, so it stops once a price is low enough to clear within ~patience — the equilibrium is a liquidity-clearing price, no hard price floor required.
- **Future:** explicit per-seller "raise after a fast sale" and the platform-recommended-price strategy remain separate, optional plans.
