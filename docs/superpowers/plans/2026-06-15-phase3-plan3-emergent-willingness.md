# Phase 3 · Plan 3 — Emergent Willingness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the `p_buy` coin-flip with an emergent buy decision: each agent has an intrinsic, sticky willingness-to-pay and buys iff `WTP ≥ price`. Conversion stops being a rate and becomes a consequence of valuations meeting prices — so a price surge prices out low-WTP agents.

**Architecture:** WTP is a configurable monetary function `willingness(agent, listing, market) -> currency`, default `v(m) × value_factor(x)` with `v(m) = listing.quality` (quality is the item's intrinsic value, in currency — the shared anchor that makes WTP and price commensurable by design) and `value_factor` a per-agent disposition (`Property`, ~lognormal ≈ 1). The default ignores the live price → WTP is sticky. `buy` becomes the **explicit**-fidelity action (willingness); the `p_buy` sigmoid is kept as the **implicit** alternative inside a `buy_action(fidelity)` factory. `view`/`lead`/`bid` stay implicit. No discrete-choice machinery.

**Tech Stack:** Python 3, SimPy, numpy, scipy, pytest. `python` = conda base, from repo root. Spec §2.4.

---

## File structure
- **Create** `sim/willingness.py` — `default_willingness(agent, listing, market)`.
- **Modify** `sim/spec.py` — `MarketplaceSpec` gains `value_factor` (`Property`, default `lognorm(s=0.3, scale=1.0)`) and `willingness` (callable, default `default_willingness`); wrap `value_factor` in `__post_init__`.
- **Modify** `sim/agents.py` — `User` gains `value_factor: float = 1.0`.
- **Modify** `sim/engine.py` — `spawn_user` draws `value_factor`; `Market.__init__` stores `self.willingness = spec.willingness`; add `Market.wtp(agent, listing)`.
- **Modify** `sim/actions.py` — replace module-level `_act_buy` with a `buy_action(fidelity="explicit")` factory (explicit = willingness, implicit = `p_buy`); `default_consumer_funnel()` uses `buy_action("explicit")`.
- **Create** `tests/test_willingness.py`.

---

### Task 1: willingness function + `value_factor` disposition + `Market.wtp`

**Files:** Create `sim/willingness.py`; Modify `sim/spec.py`, `sim/agents.py`, `sim/engine.py`; Create `tests/test_willingness.py`.

- [ ] **Step 1: failing test** — `tests/test_willingness.py`:
```python
from datetime import datetime

import numpy as np
import simpy

from sim.engine import Clock, Market, Marketplace
from sim.events import EventRecorder
from sim.spec import MarketplaceSpec
from sim.willingness import default_willingness


class _L:
    def __init__(self, quality, price):
        self.quality = quality
        self.price = price


class _A:
    def __init__(self, value_factor):
        self.value_factor = value_factor


def test_default_willingness_is_quality_times_factor():
    wtp = default_willingness(_A(1.4), _L(quality=500.0, price=999), market=None)
    assert wtp == 500.0 * 1.4          # intrinsic, monetary, ignores price


def test_spawned_agents_get_value_factor():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=env, rng=np.random.default_rng(0), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    u = m.spawn_user()
    assert isinstance(u.value_factor, float) and u.value_factor > 0


def test_market_wtp_uses_the_spec_willingness():
    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0,
                           willingness=lambda agent, listing, market: 123.0)
    m = Market(env=env, rng=np.random.default_rng(0), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    u = m.spawn_user()
    listing = m.add_listing(quality=500.0, price=400.0, seller_id=u.id)
    assert m.wtp(u, listing) == 123.0
```

- [ ] **Step 2: run** `python -m pytest tests/test_willingness.py -v` → FAIL (`sim.willingness` missing / no `value_factor` / no `Market.wtp`).

- [ ] **Step 3: implement.**
(a) `sim/willingness.py`:
```python
def default_willingness(agent, listing, market):
    """Intrinsic, sticky willingness-to-pay in currency.

    v(m) = listing.quality (quality is the item's intrinsic monetary value — the
    shared anchor that keeps WTP on the same scale as price). Scaled by this agent's
    own value_factor. Deliberately ignores the live market price, so agents can be
    priced out when prices surge. Override on the spec for non-linear / richer forms.
    """
    return listing.quality * agent.value_factor
```
(b) `sim/spec.py` — add the import `from sim.willingness import default_willingness` (near the `from sim.pricing import EndogenousPrice` line); add fields to `MarketplaceSpec` (put `value_factor` with the Property fields and `willingness` with them):
```python
    value_factor: Property = field(
        default_factory=lambda: Property(lognorm(s=0.3, scale=1.0)))
    willingness: object = default_willingness
```
and in `__post_init__` add: `self.value_factor = _as_property(self.value_factor)`.
(c) `sim/agents.py` — add to `User`:
```python
    value_factor: float = 1.0
```
(d) `sim/engine.py`:
- In `Market.__init__`, store the willingness fn (with the other field assignments): `self.willingness = spec.willingness`.
- In `spawn_user`, draw the factor and set it on the user (alongside `engagement`/`response_time`):
```python
        user.value_factor = float(self.spec.value_factor.draw(self.rng))
```
- Add a helper:
```python
    def wtp(self, agent, listing):
        return self.willingness(agent, listing, self)
```

- [ ] **Step 4: run** `python -m pytest tests/test_willingness.py -v` (expect 3 passed), then FULL suite `python -m pytest -q`. The buy step still uses `p_buy` at this point (changed in Task 2), so nothing else moves yet — confirm no regressions. (Adding the `value_factor` draw in `spawn_user` shifts the RNG stream; reproducibility tests still pass since they compare same-seed runs.)

- [ ] **Step 5: commit**
```bash
git add sim/willingness.py sim/spec.py sim/agents.py sim/engine.py tests/test_willingness.py
git commit -m "feat(sim): willingness fn + value_factor disposition + Market.wtp (Phase 3 Plan 3)"
```

---

### Task 2: `buy` becomes explicit (willingness); `p_buy` is the implicit path

**Files:** Modify `sim/actions.py`; Modify `tests/test_willingness.py` (append).

- [ ] **Step 1: failing test** — append to `tests/test_willingness.py`:
```python
def test_buy_action_factory_fidelities():
    from sim.actions import buy_action
    assert buy_action().fidelity == "explicit"
    assert buy_action("implicit").fidelity == "implicit"
    assert buy_action().name == "buy"


def test_explicit_buy_prices_out_low_wtp(monkeypatch=None):
    # WTP = quality * value_factor; an ask above WTP must NOT sell, below WTP must sell.
    import numpy as np
    import simpy
    from datetime import datetime
    from sim.actions import buy_action
    from sim.engine import Clock, Market
    from sim.events import EventRecorder
    from sim.spec import MarketplaceSpec

    env = simpy.Environment()
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=env, rng=np.random.default_rng(0), clock=Clock(spec.start),
               recorder=EventRecorder(), spec=spec)
    buyer = m.spawn_user()
    buyer.value_factor = 1.0                       # WTP = quality
    cheap = m.add_listing(quality=500.0, price=400.0, seller_id=999)   # WTP 500 >= 400 -> buy
    dear = m.add_listing(quality=500.0, price=600.0, seller_id=999)    # WTP 500 < 600 -> skip
    session = {"consideration": [cheap, dear]}
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert cheap.transactions == 1 and not cheap.is_live
    assert dear.transactions == 0 and dear.is_live


def test_default_funnel_buy_is_explicit():
    from sim.actions import default_consumer_funnel
    buy = {a.name: a for a in default_consumer_funnel()}["buy"]
    assert buy.fidelity == "explicit"
```

- [ ] **Step 2: run** `python -m pytest tests/test_willingness.py -v` → the 3 new FAIL (`buy_action` missing).

- [ ] **Step 3: implement** in `sim/actions.py`. Remove the module-level `_act_buy` function and replace it with a factory; update `default_consumer_funnel()` to call it.
```python
def buy_action(fidelity="explicit"):
    """The buy step. explicit: buy iff willingness >= price (emergent, no rng).
    implicit: the legacy p_buy(engagement) coin-flip (a cheap stand-in)."""
    def _run(agent, market, rng, session):
        negotiated = session.get("negotiated", set())
        for listing in session.get("consideration", []):
            if not listing.is_live or listing.id in negotiated:
                continue
            if fidelity == "implicit":
                bought = _decide(p_buy(agent.engagement), rng)
            else:
                bought = market.wtp(agent, listing) >= listing.price
            if bought:
                market.transact(agent, listing)
    return Action("buy", _run, requires=("consideration",), fidelity=fidelity)
```
In `default_consumer_funnel()`, change the buy entry from `Action("buy", _act_buy, requires=("consideration",))` to `buy_action("explicit")`.

- [ ] **Step 4: run** `python -m pytest tests/test_willingness.py -v` (all pass), then FULL suite `python -m pytest -q`. ALL must pass. Notes:
  - `buy` no longer draws rng in the explicit path; reproducibility tests still hold (same-seed runs identical).
  - `test_stock_never_negative` holds (the `is_live` guard in `transact`).
  - Transactions now emerge from `value_factor` vs price; confirm `transaction` still appears in default runs (some agents have `value_factor` high enough). If a real failure occurs, report BLOCKED — don't weaken tests.

- [ ] **Step 5: verify the emergent dynamic** — run:
```bash
python - <<'PY'
from datetime import datetime
from collections import Counter
from sim.engine import Marketplace
from sim.spec import MarketplaceSpec
for vf_scale in (0.6, 1.0, 1.5):     # shift the WTP distribution down/up
    from scipy.stats import lognorm
    spec = MarketplaceSpec(start=datetime(2026,1,1), n_seed_users=300, until=5.0, seed=1,
                           value_factor=lognorm(s=0.3, scale=vf_scale))
    ev = Marketplace.from_spec(spec).run()
    c = Counter(e.event_type for e in ev)
    print(f"value_factor scale={vf_scale}: transactions={c['transaction']}")
PY
```
Expected: transactions rise monotonically with the `value_factor` scale (higher WTP → more clear the price) — demonstrating emergent, sticky demand. Record the three numbers in the report.

- [ ] **Step 6: commit**
```bash
git add sim/actions.py tests/test_willingness.py
git commit -m "feat(sim): explicit willingness buy (p_buy becomes implicit) (Phase 3 Plan 3)"
```

---

## Self-Review
- **Spec coverage (§2.4):** willingness disposition + explicit decide replacing the sigmoid → Tasks 1–2; `p_buy` retained as the implicit-fidelity path → `buy_action("implicit")`; per-action fidelity → the `fidelity` field on `buy_action`. (Making `view` explicit too is deferred — not required; the headline is buy.)
- **Placeholders:** none — complete code, exact commands, and an empirical dynamic check (Step 5) with an expected monotonic result.
- **Type/name consistency:** `default_willingness(agent, listing, market)`, `MarketplaceSpec.value_factor` (Property) + `.willingness` (callable), `User.value_factor`, `Market.willingness` + `Market.wtp(agent, listing)`, `buy_action(fidelity)` returning `Action(name="buy", ..., fidelity=...)`. The `negotiate`/`assemble_actions` hooks still find `buy` by name. `value_factor` default `lognorm(s=0.3, scale=1.0)` uses scipy already imported in `spec.py`.
- **Determinism:** explicit buy draws no rng; `value_factor` is one draw at spawn; reproducibility tests guard it. WTP is sticky (ignores live price) by design, giving price-out dynamics.

## Known nuance (calibration, not a blocker)
Explicit buy is per-listing, so a high-`value_factor` agent may buy multiple affordable listings in one session. Matches the prior per-listing structure; if "one purchase per visit" is wanted, switch to best-surplus selection later. Flag in the report.
