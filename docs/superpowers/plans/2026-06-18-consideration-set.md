# Consideration-Set Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the buyer shortlist viewed listings (a swappable curation strategy, default `quality_ranked_shortlist`) and then make a single rational purchase (`argmax(wtp − price)`), instead of copying all viewed listings and buying every affordable one.

**Architecture:** A new `sim/consideration.py` holds the shipped curation strategy (top-k viewed by quality, k drawn Poisson with mu rising in engagement). `_act_consideration` delegates to a swappable `market.curation`. `buy_action` is rewritten to pick the single utility-maximizing listing. The funnel list and `run_session` are unchanged.

**Tech Stack:** Python 3.10+, `numpy`, SimPy, `pytest`. Interpreter = the conda base `python`. Run tests with `python -m pytest` from the repo root.

**Spec:** `docs/superpowers/specs/2026-06-18-consideration-set-design.md`

**Invariants:** deterministic (all ordering ties broken by `id`; the new `rng.poisson` shortlist draw comes from the single seeded rng); single-threaded; counter ids; `classes.py` and root `func.py` frozen; the **funnel list and `run_session` structure are unchanged** (only the bodies of `_act_consideration` and `buy_action` change).

**Deliberate behavior change (not a regression):** consideration now caps the set (was copy-all) and adds one `rng.poisson` draw per session; `buy` now makes ≤1 purchase per session (was buy-all-affordable, baseline ~236 transactions on the 200-user/5-day/seed-1 run). Event counts and the rng stream shift. The existing test suite is expected to **stay green** (no test asserts exact transaction counts, multi-buy, or copy-all; `test_explicit_buy_prices_out_low_wtp` survives because the affordable listing is also the argmax). If any existing test breaks, update it to the new model — do **not** weaken determinism or the controlled proposal/settlement pipeline tests.

---

## File Structure

- **Create `sim/consideration.py`** — constants + `quality_ranked_shortlist(agent, viewed, market, rng)`. Imports only `numpy` (self-contained, no `sim` imports → no cycle).
- **Create `tests/test_consideration.py`** — unit tests for the strategy.
- **Modify `sim/spec.py`** — import the strategy; add `curation` field.
- **Modify `sim/engine.py`** — `Market.__init__` sets `self.curation`.
- **Modify `sim/actions.py`** — `_act_consideration` delegates to `market.curation`; rewrite `buy_action`.
- **Modify `sim/__init__.py`** — export `quality_ranked_shortlist`.
- **Modify `tests/test_willingness.py`** — only if Task 3 shows it needs it (it should not; verify).

---

## Task 1: `quality_ranked_shortlist` curation strategy

**Files:**
- Create: `sim/consideration.py`
- Test: `tests/test_consideration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consideration.py`:

```python
import numpy as np

from sim.consideration import quality_ranked_shortlist


class _L:
    def __init__(self, id, quality):
        self.id = id
        self.quality = quality


class _A:
    def __init__(self, engagement):
        self.engagement = engagement


def _listings(qualities):
    return [_L(i, q) for i, q in enumerate(qualities)]


def test_empty_viewed_returns_empty():
    out = quality_ranked_shortlist(_A(7.0), [], market=None, rng=np.random.default_rng(0))
    assert out == []


def test_ranks_by_quality_desc_with_id_tiebreak():
    viewed = [_L(0, 100.0), _L(1, 300.0), _L(2, 300.0), _L(3, 50.0)]
    # huge engagement -> mu huge -> k >> len(viewed) -> returns all, fully ordered
    out = quality_ranked_shortlist(_A(1e6), viewed, market=None, rng=np.random.default_rng(0))
    assert [l.id for l in out] == [1, 2, 0, 3]      # 300(id1), 300(id2), 100, 50


def test_caps_at_k_and_never_exceeds_viewed():
    viewed = _listings([float(i) for i in range(10)])
    rng = np.random.default_rng(0)
    lens = [len(quality_ranked_shortlist(_A(7.0), viewed, None, rng)) for _ in range(2000)]
    assert max(lens) <= len(viewed)
    assert 3.0 < float(np.mean(lens)) < 7.0         # mu ~= 5 at engagement 7


def test_higher_engagement_gives_bigger_shortlist():
    viewed = _listings([float(i) for i in range(10)])
    rng = np.random.default_rng(1)
    lo = np.mean([len(quality_ranked_shortlist(_A(2.0), viewed, None, rng)) for _ in range(2000)])
    hi = np.mean([len(quality_ranked_shortlist(_A(20.0), viewed, None, rng)) for _ in range(2000)])
    assert lo < hi


def test_deterministic_for_fixed_rng():
    viewed = _listings([float(i) for i in range(8)])
    a = quality_ranked_shortlist(_A(7.0), viewed, None, np.random.default_rng(5))
    b = quality_ranked_shortlist(_A(7.0), viewed, None, np.random.default_rng(5))
    assert [l.id for l in a] == [l.id for l in b]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consideration.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.consideration'`.

- [ ] **Step 3: Write minimal implementation**

Create `sim/consideration.py`:

```python
"""Consideration-set curation: which viewed listings the agent seriously evaluates.

The shipped strategy ranks by quality (a stand-in for relevance, which has no
property yet) and keeps an engagement-sized shortlist. It is a swappable callable
on the spec (``curation``); it is the default by assignment, not by name. Imports
only numpy so sim.spec can import it without a cycle.
"""
import numpy as np

CONSIDERATION_MU_AT_REF = 5.0   # target shortlist size at the reference engagement
REF_ENGAGEMENT = 7.0            # the default engagement mean (gamma(a=2, scale=7/2)); anchors mu


def quality_ranked_shortlist(agent, viewed, market, rng):
    """Curate the consideration set: the top-k of ``viewed`` by quality.

    k is a per-session draw, Poisson with mu rising linearly in engagement
    (mu = CONSIDERATION_MU_AT_REF * engagement / REF_ENGAGEMENT), so a more engaged
    agent considers more; k is capped naturally by len(viewed). Ties broken by
    listing id, so the result is deterministic. ``market`` is unused here (present
    for parity with the other strategy callables / power users).
    """
    if not viewed:
        return []
    mu = CONSIDERATION_MU_AT_REF * agent.engagement / REF_ENGAGEMENT
    k = int(rng.poisson(mu))
    ranked = sorted(viewed, key=lambda l: (-l.quality, l.id))   # quality desc; id breaks ties
    return ranked[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_consideration.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add sim/consideration.py tests/test_consideration.py
git commit -m "feat(consideration): quality_ranked_shortlist curation strategy"
```

---

## Task 2: Wire curation into the spec, engine, and consideration step

**Files:**
- Modify: `sim/spec.py`
- Modify: `sim/engine.py`
- Modify: `sim/actions.py` (`_act_consideration` only)
- Test: `tests/test_consideration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consideration.py`:

```python
def test_spec_default_curation_is_quality_ranked_shortlist():
    from datetime import datetime
    from sim.spec import MarketplaceSpec
    spec = MarketplaceSpec(start=datetime(2026, 1, 1))
    assert spec.curation is quality_ranked_shortlist


def test_act_consideration_uses_market_curation():
    # A custom curation strategy must be what populates session["consideration"].
    from sim.actions import _act_consideration

    sentinel = [object(), object()]

    class _M:
        def curation(self, agent, viewed, market, rng):
            assert market is self
            return sentinel

    session = {"viewed": [object(), object(), object()]}
    _act_consideration(agent=None, market=_M(), rng=None, session=session)
    assert session["consideration"] is sentinel


def test_market_exposes_curation_from_spec():
    import numpy as np
    import simpy
    from datetime import datetime
    from sim.engine import Clock, Market
    from sim.events import EventRecorder
    from sim.spec import MarketplaceSpec

    strat = lambda agent, viewed, market, rng: list(viewed)[:1]
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0, curation=strat)
    m = Market(env=simpy.Environment(), rng=np.random.default_rng(0),
               clock=Clock(spec.start), recorder=EventRecorder(), spec=spec)
    assert m.curation is strat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consideration.py -k "spec_default or act_consideration_uses or exposes_curation" -q`
Expected: FAIL — `AttributeError: 'MarketplaceSpec' object has no attribute 'curation'`.

- [ ] **Step 3: Write minimal implementation**

**3a. `sim/spec.py` import.** Next to the existing `from sim.willingness import default_willingness` line, add:

```python
from sim.consideration import quality_ranked_shortlist
```

**3b. `sim/spec.py` field.** The fields currently read:

```python
    willingness: object = default_willingness
    pricing: object = default_pricing
```

Add a `curation` field immediately after `pricing`:

```python
    willingness: object = default_willingness
    pricing: object = default_pricing
    curation: object = quality_ranked_shortlist
```

**3c. `sim/engine.py`.** In `Market.__init__`, the lines currently read:

```python
        self.willingness = spec.willingness
        self.pricing = spec.pricing
        self.markdown_pct = spec.markdown_pct
```

Add `self.curation` after `self.pricing`:

```python
        self.willingness = spec.willingness
        self.pricing = spec.pricing
        self.curation = spec.curation
        self.markdown_pct = spec.markdown_pct
```

**3d. `sim/actions.py`.** Replace the current `_act_consideration`:

```python
def _act_consideration(agent, market, rng, session):
    """Form the consideration set the buy/negotiate steps act on. Currently a copy of
    everything viewed; the seam where top-k-by-willingness curation will plug in."""
    session["consideration"] = list(session.get("viewed", []))
```

with the delegating version:

```python
def _act_consideration(agent, market, rng, session):
    """Form the consideration set via the market's curation strategy — a shortlist of
    what was viewed, which the buy/negotiate steps then act on."""
    session["consideration"] = market.curation(agent, session.get("viewed", []), market, rng)
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — all green. (Consideration now caps the set; `buy` still buys all affordable *within the shortlist* until Task 3. No existing test asserts counts, so the suite stays green; determinism self-equality holds.)

- [ ] **Step 5: Commit**

```bash
git add sim/spec.py sim/engine.py sim/actions.py tests/test_consideration.py
git commit -m "feat(consideration): swappable curation; _act_consideration delegates"
```

---

## Task 3: Rational single-best `buy_action`

**Files:**
- Modify: `sim/actions.py` (`buy_action` only)
- Test: `tests/test_consideration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consideration.py`:

```python
def _market_with_buyer(value_factor=1.0):
    import numpy as np
    import simpy
    from datetime import datetime
    from sim.engine import Clock, Market
    from sim.events import EventRecorder
    from sim.spec import MarketplaceSpec
    spec = MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=0)
    m = Market(env=simpy.Environment(), rng=np.random.default_rng(0),
               clock=Clock(spec.start), recorder=EventRecorder(), spec=spec)
    buyer = m.spawn_user()
    buyer.value_factor = value_factor
    return m, buyer


def test_explicit_buys_single_best_surplus_only():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)            # wtp == quality
    a = m.add_listing(quality=500.0, price=400.0, seller_id=999)   # surplus +100
    best = m.add_listing(quality=500.0, price=300.0, seller_id=999)  # surplus +200 (the pick)
    dear = m.add_listing(quality=500.0, price=600.0, seller_id=999)  # surplus -100
    session = {"consideration": [a, best, dear]}
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert best.transactions == 1 and not best.is_live          # only the best is bought
    assert a.transactions == 0 and dear.transactions == 0       # at most one purchase


def test_explicit_buys_nothing_when_all_underwater():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)
    x = m.add_listing(quality=500.0, price=600.0, seller_id=999)
    y = m.add_listing(quality=500.0, price=700.0, seller_id=999)
    session = {"consideration": [x, y]}
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert x.transactions == 0 and y.transactions == 0


def test_buy_excludes_negotiated_listings():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)
    claimed = m.add_listing(quality=500.0, price=300.0, seller_id=999)  # best, but negotiated
    other = m.add_listing(quality=500.0, price=450.0, seller_id=999)    # the fallback pick
    session = {"consideration": [claimed, other], "negotiated": {claimed.id}}
    buy_action("explicit").run(buyer, m, m.rng, session)
    assert claimed.transactions == 0                            # excluded
    assert other.transactions == 1                              # best of the rest


def test_implicit_buys_at_most_the_single_best():
    from sim.actions import buy_action
    m, buyer = _market_with_buyer(value_factor=1.0)
    buyer.engagement = 1e6                                      # p_buy ~ 1 -> coin flip passes
    a = m.add_listing(quality=400.0, price=10.0, seller_id=999)
    best = m.add_listing(quality=900.0, price=10.0, seller_id=999)   # highest surplus
    session = {"consideration": [a, best]}
    buy_action("implicit").run(buyer, m, m.rng, session)
    assert best.transactions == 1
    assert a.transactions == 0                                  # only one, and it's the best
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consideration.py -k "single_best or all_underwater or excludes_negotiated or implicit_buys" -q`
Expected: FAIL — the current `buy_action` buys every affordable listing, so `a.transactions == 0` fails in `test_explicit_buys_single_best_surplus_only`.

- [ ] **Step 3: Write minimal implementation**

In `sim/actions.py`, replace the entire current `buy_action`:

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

with the single-best version:

```python
def buy_action(fidelity="explicit"):
    """The buy step. The agent makes ONE rational choice: the single utility-maximizing
    listing in its consideration set (argmax of wtp - price, excluding negotiated /
    sold-out ones). explicit: buy that best iff its surplus >= 0 (emergent, no rng).
    implicit: pick the same best but gate on the p_buy(engagement) coin flip (a cheap
    stand-in). At most one purchase per session."""
    def _run(agent, market, rng, session):
        negotiated = session.get("negotiated", set())
        candidates = [l for l in session.get("consideration", [])
                      if l.is_live and l.id not in negotiated]
        if not candidates:
            return
        best = max(candidates, key=lambda l: (market.wtp(agent, l) - l.price, l.id))
        if fidelity == "implicit":
            if _decide(p_buy(agent.engagement), rng):
                market.transact(agent, best)
        elif market.wtp(agent, best) - best.price >= 0:
            market.transact(agent, best)
    return Action("buy", _run, requires=("consideration",), fidelity=fidelity)
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — all green, including the existing `test_explicit_buy_prices_out_low_wtp` in `tests/test_willingness.py` (its affordable listing is also the argmax, the dear one is excluded either way). If any existing test fails because it assumed buy-all-affordable or copy-all consideration, update that test to the single-best / shortlist model — do not change determinism tests or the proposal/settlement pipeline tests. Report any test you change and why.

- [ ] **Step 5: Commit**

```bash
git add sim/actions.py tests/test_consideration.py
git commit -m "feat(consideration): buy picks the single utility-maximizing listing"
```

---

## Task 4: Export `quality_ranked_shortlist`

**Files:**
- Modify: `sim/__init__.py`
- Test: `tests/test_consideration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consideration.py`:

```python
def test_curation_strategy_is_exported():
    import sim
    from sim import quality_ranked_shortlist as exported
    assert "quality_ranked_shortlist" in sim.__all__
    assert exported is quality_ranked_shortlist
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consideration.py::test_curation_strategy_is_exported -q`
Expected: FAIL — `ImportError: cannot import name 'quality_ranked_shortlist' from 'sim'`.

- [ ] **Step 3: Write minimal implementation**

In `sim/__init__.py`, add an import after the existing `from sim.allocation import (...)` block (keep alphabetical order of the import lines tidy; place it before `from sim.engine import Marketplace`):

```python
from sim.consideration import quality_ranked_shortlist
```

And in `__all__`, add `"quality_ranked_shortlist"` to the strategy line. The line currently reads:

```python
    "negotiate_action", "default_pricing", "default_willingness",
```

Replace it with:

```python
    "negotiate_action", "default_pricing", "default_willingness", "quality_ranked_shortlist",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_consideration.py::test_curation_strategy_is_exported -q`
Expected: PASS. Also verify: `python -c "from sim import quality_ranked_shortlist; print('ok')"` prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add sim/__init__.py tests/test_consideration.py
git commit -m "feat(consideration): export quality_ranked_shortlist from sim"
```

---

## Task 5: Final regression + determinism + behavior verification

**Files:**
- Test only (no production code changes)

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: PASS — all green.

- [ ] **Step 2: Behavior + determinism check**

Run this one-off (paste into a shell):

```bash
python - <<'PY'
from datetime import datetime
from collections import Counter
from sim import Marketplace, MarketplaceSpec

def run():
    return [(e.event_type, e.actor_id, e.entity_id)
            for e in Marketplace.from_spec(MarketplaceSpec(
                start=datetime(2026, 1, 1), n_seed_users=200, until=5.0, seed=1)).run()]

a, b = run(), run()
print("deterministic:", a == b)
print("event counts:", dict(Counter(t for t, _, _ in a)))
PY
```

Expected: `deterministic: True`. Transaction count is materially lower than the old buy-all-affordable baseline (~236 on this run) — that is the intended single-best behavior, not a regression.

- [ ] **Step 3: Smoke harness**

Run: `python scripts/run_slice.py`
Expected: runs to completion, reports `reproducible: True`. Transaction counts shift down vs before (single-best buy + capped consideration) — expected.

- [ ] **Step 4: No commit needed.** If the suite is green and the run is deterministic, the feature is complete. Proceed to the final code review and `superpowers:finishing-a-development-branch`.

---

## Notes for the implementer

- **Do not edit `classes.py`**, root `func.py`, the funnel list, or `run_session`. Only the bodies of `_act_consideration` and `buy_action` change in `sim/actions.py`.
- **`sim/consideration.py` imports only `numpy`** — no `sim` imports, so `sim/spec.py` can import it without a cycle.
- **Interpreter:** use `python` (conda base), not `python3`.
- This is a deliberate behavior change: fewer transactions, shifted rng stream. Determinism (run-to-run equality) must still hold; if it ever fails, a non-rng or non-id-ordered path crept in — stop and fix.
- If an existing test breaks (it should not), update it to the new model and report it; never weaken the determinism or proposal/settlement pipeline tests to make something pass.
