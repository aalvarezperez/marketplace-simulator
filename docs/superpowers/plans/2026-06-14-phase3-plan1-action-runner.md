# Phase 3 · Plan 1 — Action Primitive & Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a simple `Action` primitive + a session runner, and re-express the consumer funnel (`visit → list → search → view → consideration → buy`) as a declarative list of `Action`s — replacing the hardcoded `_run_session`, while keeping the 49-test suite green, deterministic, and single-threaded.

**Architecture:** A new `sim/actions.py` holds the `Action` dataclass, a `run_session` runner that walks an ordered action list gating each on its `requires`, and `default_consumer_funnel()` that builds the funnel from `Action`s whose `run` callables wrap the *current* funnel logic (sigmoid decisions retained for now). The engine holds the action list on `Market` and `user_lifecycle` invokes `market.run_session(...)`. Import direction: `actions.py → agents.py` (uses `p_*`/`_decide`); `engine.py → actions.py`; `agents.py` imports neither (no cycle).

**Tech Stack:** Python 3, SimPy, numpy, pytest. `python` = conda base interpreter (from repo root).

**Spec:** `docs/superpowers/specs/2026-06-14-phase3-action-primitives-design.md` (§2.1 Action, §2.2 consumer funnel). Subsequent plans (roadmap at the end) cover §2.3 hooks, §2.4 emergent willingness, §3 sub-markets, §5 API.

**Note — not byte-preserving:** the funnel restructures to *view-all-then-buy* (a real consideration set) instead of the current per-listing interleave, so the RNG draw order changes. That's intended. The 49 tests assert *structure* (event kinds present, within window, reproducible, no threads, stock ≥ 0), all of which hold; determinism (same seed → identical run) is preserved.

---

## File structure

- **Create** `sim/actions.py` — `Action` dataclass, `run_session(agent, market, rng, actions)`, the `_act_*` funnel callables, `default_consumer_funnel()`.
- **Modify** `sim/agents.py` — delete `_run_session`; `user_lifecycle` calls `market.run_session(user, rng)`. (`p_*`, `_decide`, `SESSION_K`, `BID_BIAS` stay — the action callables import them.)
- **Modify** `sim/engine.py` — `Market.__init__` sets `self.actions = default_consumer_funnel()`; add `Market.run_session(self, user, rng)`; import from `sim.actions` at top.
- **Create** `tests/test_actions.py` — runner gating/order + funnel shape + engine integration.

---

### Task 1: `Action` primitive + `run_session` runner

**Files:**
- Create: `sim/actions.py`
- Test: `tests/test_actions.py`

- [ ] **Step 1: Write the failing test**

`tests/test_actions.py`:
```python
import numpy as np

from sim.actions import Action, run_session


def test_runner_runs_in_order_and_gates_on_requires():
    log = []

    def mk(name):
        def run(agent, market, rng, session):
            log.append(name)
        return run

    actions = [
        Action("a", mk("a")),
        Action("b", mk("b"), requires=("a",)),
        Action("c", mk("c"), requires=("missing",)),  # never eligible
    ]
    done = run_session(agent=None, market=None, rng=np.random.default_rng(0), actions=actions)
    assert log == ["a", "b"]        # c gated out, order preserved
    assert done == {"a", "b"}


def test_session_dict_threads_between_actions():
    def produce(agent, market, rng, session):
        session["candidates"] = [1, 2, 3]

    def consume(agent, market, rng, session):
        session["seen"] = list(session["candidates"])

    actions = [Action("produce", produce), Action("consume", consume, requires=("produce",))]
    run_session(None, None, np.random.default_rng(0), actions)
    # no assertion error means the shared session dict carried state; verify via a capture
    captured = {}

    def capture(agent, market, rng, session):
        captured.update(session)

    run_session(None, None, np.random.default_rng(0),
                [Action("produce", produce), Action("cap", capture, requires=("produce",))])
    assert captured["candidates"] == [1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_actions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sim.actions'`.

- [ ] **Step 3: Write minimal implementation**

`sim/actions.py`:
```python
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class Action:
    """A named step an agent performs in a session.

    `run(agent, market, rng, session)` performs the decision + effect + events for
    this step. `requires` names prior actions that must have run this session before
    this one is eligible. `fidelity` is informational for now ("explicit" | "implicit");
    the emergent decision arrives in a later plan.
    """
    name: str
    run: Callable
    requires: Tuple[str, ...] = ()
    fidelity: str = "explicit"


def run_session(agent, market, rng, actions):
    """Walk the action list once. An action runs only if every name in its `requires`
    has already run this session. Returns the set of action names that ran."""
    done = set()
    session = {}
    for action in actions:
        if all(req in done for req in action.requires):
            action.run(agent, market, rng, session)
            done.add(action.name)
    return done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_actions.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/actions.py tests/test_actions.py
git commit -m "feat(sim): Action primitive + session runner (Phase 3 Plan 1)"
```

---

### Task 2: Consumer funnel as `Action`s

**Files:**
- Modify: `sim/actions.py` (add the funnel callables + `default_consumer_funnel`)
- Test: `tests/test_actions.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_actions.py`:
```python
def test_default_funnel_names_and_order():
    from sim.actions import default_consumer_funnel
    names = [a.name for a in default_consumer_funnel()]
    assert names == ["visit", "list", "search", "view", "consideration", "buy"]


def test_default_funnel_preconditions():
    from sim.actions import default_consumer_funnel
    by = {a.name: a for a in default_consumer_funnel()}
    assert by["visit"].requires == ()
    assert by["list"].requires == ("visit",)
    assert by["search"].requires == ("visit",)
    assert by["view"].requires == ("search",)
    assert by["consideration"].requires == ("view",)
    assert by["buy"].requires == ("consideration",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_actions.py -v`
Expected: FAIL — `ImportError: cannot import name 'default_consumer_funnel'`.

- [ ] **Step 3: Write minimal implementation**

Append to `sim/actions.py` (the funnel callables wrap the *current* `_run_session` logic; they import the funnel helpers from `sim.agents`):
```python
from sim.agents import (BID_BIAS, SESSION_K, _decide, p_bid, p_buy, p_lead,
                        p_list, p_view)


def _act_visit(agent, market, rng, session):
    market.emit("visit", actor_id=agent.id)


def _act_list(agent, market, rng, session):
    if _decide(p_list(agent.engagement), rng):
        market.create_listing_for(agent, rng)


def _act_search(agent, market, rng, session):
    session["candidates"] = market.match_listings(SESSION_K)


def _act_view(agent, market, rng, session):
    viewed = []
    for listing in session.get("candidates", []):
        if not listing.is_live:
            continue
        if _decide(p_view(agent.engagement), rng):
            listing.views += 1
            market.emit("view", actor_id=agent.id, entity_id=listing.id,
                        other_id=listing.seller_id)
            viewed.append(listing)
    session["viewed"] = viewed


def _act_consideration(agent, market, rng, session):
    session["consideration"] = list(session.get("viewed", []))


def _act_buy(agent, market, rng, session):
    for listing in session.get("consideration", []):
        if not listing.is_live:
            continue
        if _decide(p_buy(agent.engagement), rng):
            market.transact(agent, listing)
        elif _decide(p_lead(agent.engagement), rng):
            listing.leads += 1
            market.emit("lead", actor_id=agent.id, entity_id=listing.id,
                        other_id=listing.seller_id)
            if _decide(p_bid(agent.engagement), rng):
                seller = market.get_user(listing.seller_id)
                if seller is not None and seller.inbox is not None:
                    amount = BID_BIAS * listing.price
                    proposal = market.make_proposal(buyer=agent, seller=seller,
                                                    listing=listing, amount=amount)
                    market.send_to_seller(proposal)
                    listing.bids += 1
                    market.emit("bid", actor_id=agent.id, entity_id=listing.id,
                                other_id=seller.id,
                                payload={"proposal_id": proposal.id, "amount": amount})


def default_consumer_funnel():
    return [
        Action("visit", _act_visit),
        Action("list", _act_list, requires=("visit",)),
        Action("search", _act_search, requires=("visit",)),
        Action("view", _act_view, requires=("search",)),
        Action("consideration", _act_consideration, requires=("view",)),
        Action("buy", _act_buy, requires=("consideration",)),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_actions.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/actions.py tests/test_actions.py
git commit -m "feat(sim): consumer funnel as Action list (Phase 3 Plan 1)"
```

---

### Task 3: Wire the runner into the engine; retire `_run_session`

**Files:**
- Modify: `sim/engine.py` (import actions; `Market.actions`; `Market.run_session`)
- Modify: `sim/agents.py` (delete `_run_session`; `user_lifecycle` calls `market.run_session`)
- Test: `tests/test_actions.py` (append integration test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_actions.py`:
```python
def test_engine_runs_funnel_via_action_runner():
    from datetime import datetime
    from sim.engine import Marketplace
    from sim.spec import MarketplaceSpec
    events = Marketplace.from_spec(
        MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=100, until=5.0, seed=1)).run()
    kinds = {e.event_type for e in events}
    assert {"visit", "view"} <= kinds   # funnel ran through the action runner


def test_action_runner_full_run_reproducible():
    from datetime import datetime
    from sim.engine import Marketplace
    from sim.spec import MarketplaceSpec

    def run():
        return [(e.event_type, e.actor_id, e.entity_id)
                for e in Marketplace.from_spec(
                    MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=80,
                                    until=5.0, seed=3)).run()]
    assert run() == run()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_actions.py::test_engine_runs_funnel_via_action_runner -v`
Expected: FAIL — at this point `user_lifecycle` still calls the old `_run_session`; the new test passes only structurally, but run it now to confirm the suite state before wiring. (If it already passes because the funnel kinds happen to match, proceed; the real verification is Step 4's full suite after the refactor.)

- [ ] **Step 3: Write the implementation**

In `sim/engine.py`, add to the top imports:
```python
from sim.actions import default_consumer_funnel, run_session as _run_session_actions
```
In `Market.__init__`, after `self.spec = spec` (with the other instance fields), add:
```python
        self.actions = default_consumer_funnel()
```
Add a method to `Market`:
```python
    def run_session(self, user, rng):
        return _run_session_actions(user, self, rng, self.actions)
```

In `sim/agents.py`:
- Delete the entire `_run_session(user, market, rng)` function (its logic now lives in `sim/actions.py`).
- In `user_lifecycle`, replace the call `_run_session(user, market, rng)` with `market.run_session(user, rng)`. The function becomes:
```python
def user_lifecycle(env, user, market, rng):
    while user.state == "active":
        scale = ENGAGEMENT_TIME_UNIT / max(user.engagement, EPS)
        yield env.timeout(float(rng.exponential(scale)))
        if user.state != "active":
            break
        market.run_session(user, rng)
        if _decide(p_churn(user.engagement), rng):
            market.churn_user(user)
            break
```

- [ ] **Step 4: Run the FULL suite to verify everything passes**

Run: `python -m pytest -q`
Expected: all tests pass (the prior 49 + the new action tests). In particular `test_runs_are_reproducible`, `test_no_threads_spawned`, `test_stock_never_negative`, `test_emergent_prices_track_quality_in_full_run`, and the lifecycle/variant/settlement tests stay green — the funnel produces the same event *kinds* deterministically, just restructured to view-all-then-buy.

- [ ] **Step 5: Verify the harness + determinism**

Run: `python scripts/run_slice.py`
Expected: prints non-zero counts including `visit`, `list`, `view`, `transaction`, and `bid`/`lead`; `reproducible: True`.

- [ ] **Step 6: Commit**

```bash
git add sim/engine.py sim/agents.py tests/test_actions.py
git commit -m "refactor(sim): drive sessions through the Action runner (Phase 3 Plan 1)"
```

---

## Self-Review

**1. Spec coverage (this plan):** §2.1 `Action` → Task 1. §2.2 consumer funnel as `Action`s → Task 2. Engine uses the runner → Task 3. §2.3 (hooks), §2.4 (emergent willingness), §3 (sub-markets), §4/§5 (properties already done / API) are **out of scope for Plan 1** — see roadmap below. No spec requirement for Plan 1 is unaddressed.

**2. Placeholder scan:** none — every code step shows complete code; every run step has an exact command + expected result. The Task 3 Step 2 note explains why the failing-test step is soft for a refactor (the existing suite is the real guard at Step 4).

**3. Type/name consistency:** `Action(name, run, requires, fidelity)`, `run_session(agent, market, rng, actions)`, the `_act_*(agent, market, rng, session)` signature, `default_consumer_funnel()`, `Market.actions`, `Market.run_session(user, rng)` — consistent across tasks. The funnel callables call only existing `Market` methods (`emit`, `create_listing_for`, `match_listings`, `transact`, `get_user`, `make_proposal`, `send_to_seller`) and existing `agents` helpers (`p_list/p_view/p_buy/p_lead/p_bid`, `_decide`, `SESSION_K`, `BID_BIAS`), all verified present in the current code.

---

## Phase 3 roadmap (subsequent plans — authored when reached)

- **Plan 2 — Pluggable actions + hooks (§2.3).** Add `before`/`after` + `mode` (`gate`/`branch`) to `Action`; an insertion step in the runner; extract `negotiate` (the lead→bid→proposal branch) out of `_act_buy` into a registered branch action; `MarketplaceSpec.actions` to register extras. Tests: with/without `negotiate`.
- **Plan 3 — Emergent willingness (§2.4).** Add a simple per-agent willingness disposition; an `explicit` decide for `view`/`buy` (willingness vs price/quality/competition) replacing the sigmoids, which become the `implicit`-fidelity rates. Per-action fidelity switch. Tests: emergent conversion; explicit-vs-implicit both coherent. (Also dissolves the Phase 2 negotiation-calibration issue.)
- **Plan 4 — Two-layer sub-markets (§3).** `MarketplaceSpec.submarket_weights`; assign agents/listings to a sub-market; scope `search`/matching per sub-market. Tests: matching isolation; single-sub-market = today's behaviour.
- **Plan 5 — API polish + consideration semantics (§5).** Finalize `consideration` curation (top-k by willingness), the `actions` registration ergonomics, fidelity defaults, and docs/harness. Tests: API surface; CLAUDE.md update.

Each plan keeps the regression suite green, stays deterministic/single-threaded, and leaves `classes.py` frozen. v2.0 merges to `main` only on explicit approval.
