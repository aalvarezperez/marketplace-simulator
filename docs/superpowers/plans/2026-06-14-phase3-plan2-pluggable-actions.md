# Phase 3 · Plan 2 — Pluggable Actions & Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let users add actions that hook into the base consumer funnel without editing it. `negotiate` (the lead→bid→proposal branch) leaves the base funnel and becomes an opt-in registered action; the default marketplace is the generic buy-only consumer journey.

**Architecture:** `Action` gains `before`/`after` (placement) + `mode` (`branch` default | `gate`). `assemble_actions(base, extras)` inserts each extra at its hooked position (and, for `gate`, adds the extra to the target's `requires`). `negotiate_action()` is a shipped library action (`before="buy"`, `mode="branch"`) that claims listings it engages, via a `session["negotiated"]` set; `_act_buy` skips claimed listings. `MarketplaceSpec.actions` carries extras; `Market` assembles base + extras.

**Tech Stack:** Python 3, SimPy, numpy, pytest. `python` = conda base, from repo root. Spec §2.3.

---

## File structure
- **Modify** `sim/actions.py` — `Action` hook fields; `assemble_actions`; extract `_act_negotiate` + `negotiate_action()`; make `_act_buy` buy-only (skip negotiated). `default_consumer_funnel()` stays buy-only.
- **Modify** `sim/spec.py` — `MarketplaceSpec.actions: list = field(default_factory=list)`.
- **Modify** `sim/engine.py` — `Market.__init__` assembles `default_consumer_funnel()` + `spec.actions`.
- **Modify** `tests/test_funnel_bid.py` — register `negotiate_action()` to see lead/bid; assert default has none.
- **Modify** `scripts/run_slice.py` — register `negotiate_action()` so the demo shows the full funnel.
- **Create** `tests/test_hooks.py` — assemble placement + gate-requires; negotiate present/absent.

---

### Task 1: `Action` hooks + `assemble_actions`

**Files:** Modify `sim/actions.py`; Create `tests/test_hooks.py`.

- [ ] **Step 1: failing test** — `tests/test_hooks.py`:
```python
from sim.actions import Action, assemble_actions


def _base():
    return [Action("a", lambda *x: None),
            Action("b", lambda *x: None, requires=("a",)),
            Action("c", lambda *x: None, requires=("b",))]


def test_insert_before_target():
    extra = Action("x", lambda *x: None, before="c")
    names = [a.name for a in assemble_actions(_base(), [extra])]
    assert names == ["a", "b", "x", "c"]


def test_insert_after_target():
    extra = Action("x", lambda *x: None, after="a")
    names = [a.name for a in assemble_actions(_base(), [extra])]
    assert names == ["a", "x", "b", "c"]


def test_gate_adds_requires_to_target():
    extra = Action("x", lambda *x: None, before="c", mode="gate")
    out = {a.name: a for a in assemble_actions(_base(), [extra])}
    assert "x" in out["c"].requires        # gate: c now requires x


def test_branch_does_not_touch_target_requires():
    extra = Action("x", lambda *x: None, before="c", mode="branch")
    out = {a.name: a for a in assemble_actions(_base(), [extra])}
    assert "x" not in out["c"].requires


def test_unknown_hook_target_raises():
    extra = Action("x", lambda *x: None, before="nope")
    try:
        assemble_actions(_base(), [extra])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_base_unchanged_when_no_extras():
    assert [a.name for a in assemble_actions(_base(), [])] == ["a", "b", "c"]
```

- [ ] **Step 2: run** `python -m pytest tests/test_hooks.py -v` → FAIL (`Action` has no `before`; no `assemble_actions`).

- [ ] **Step 3: implement.** In `sim/actions.py`, extend the `Action` dataclass and add `assemble_actions`. Add `import dataclasses` at the top. New `Action`:
```python
@dataclass
class Action:
    """A named step an agent performs in a session.

    run(agent, market, rng, session) performs the decision + effect + events.
    `requires`: prior action names that must have run this session.
    `fidelity`: "explicit" | "implicit" (informational for now).
    `before`/`after`: name of a base action this extra hooks around.
    `mode`: "branch" (insert alongside; coordinate via session state) or
            "gate" (also add this action to the target's `requires`).
    """
    name: str
    run: Callable
    requires: Tuple[str, ...] = ()
    fidelity: str = "explicit"
    before: str = None
    after: str = None
    mode: str = "branch"
```
Add:
```python
def assemble_actions(base, extras):
    """Return base + extras, each extra inserted at its before/after hook.
    A `gate` extra also gets appended to its target action's `requires`."""
    actions = list(base)
    for extra in extras:
        target = extra.before or extra.after
        idx = next((i for i, a in enumerate(actions) if a.name == target), None)
        if idx is None:
            raise ValueError(
                f"action {extra.name!r} hooks onto unknown action {target!r}")
        insert_at = idx if extra.before else idx + 1
        actions.insert(insert_at, extra)
        if extra.mode == "gate" and extra.before:
            tgt = actions[insert_at + 1]
            actions[insert_at + 1] = dataclasses.replace(
                tgt, requires=tuple(tgt.requires) + (extra.name,))
    return actions
```

- [ ] **Step 4: run** `python -m pytest tests/test_hooks.py -v` → 6 passed. Then `python -m pytest -q` → no regressions (Action's new fields are defaulted; nothing else changed yet).

- [ ] **Step 5: commit**
```bash
git add sim/actions.py tests/test_hooks.py
git commit -m "feat(sim): Action hooks + assemble_actions (Phase 3 Plan 2)"
```

---

### Task 2: extract `negotiate` as a pluggable branch

**Files:** Modify `sim/actions.py`; Modify `tests/test_hooks.py` (append).

- [ ] **Step 1: failing test** — append to `tests/test_hooks.py`:
```python
def test_negotiate_action_shape():
    from sim.actions import negotiate_action
    n = negotiate_action()
    assert n.name == "negotiate"
    assert n.before == "buy"
    assert n.mode == "branch"
    assert n.requires == ("consideration",)


def test_negotiate_inserts_before_buy():
    from sim.actions import assemble_actions, default_consumer_funnel, negotiate_action
    names = [a.name for a in assemble_actions(default_consumer_funnel(), [negotiate_action()])]
    assert names == ["visit", "list", "search", "view", "consideration", "negotiate", "buy"]
```

- [ ] **Step 2: run** `python -m pytest tests/test_hooks.py -v` → the 2 new FAIL (`negotiate_action` missing).

- [ ] **Step 3: implement.** In `sim/actions.py`: (a) replace `_act_buy` with a **buy-only** version that skips negotiated listings; (b) add `_act_negotiate` and `negotiate_action()`. `default_consumer_funnel()` is unchanged (still ends at `buy`, no negotiate).
```python
def _act_buy(agent, market, rng, session):
    negotiated = session.get("negotiated", set())
    for listing in session.get("consideration", []):
        if not listing.is_live or listing.id in negotiated:
            continue
        if _decide(p_buy(agent.engagement), rng):
            market.transact(agent, listing)


def _act_negotiate(agent, market, rng, session):
    negotiated = session.setdefault("negotiated", set())
    for listing in session.get("consideration", []):
        if not listing.is_live or listing.id in negotiated:
            continue
        if _decide(p_lead(agent.engagement), rng):
            listing.leads += 1
            market.emit("lead", actor_id=agent.id, entity_id=listing.id,
                        other_id=listing.seller_id)
            negotiated.add(listing.id)            # claim it; buy will skip it
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


def negotiate_action():
    return Action("negotiate", _act_negotiate, requires=("consideration",),
                  before="buy", mode="branch")
```

- [ ] **Step 4: run** `python -m pytest tests/test_hooks.py -v` → all pass. Then `python -m pytest -q`. NOTE: `tests/test_funnel_bid.py` will now FAIL because the default funnel no longer negotiates — that is fixed in Task 3. Confirm the only failures are in `test_funnel_bid.py`.

- [ ] **Step 5: commit**
```bash
git add sim/actions.py tests/test_hooks.py
git commit -m "feat(sim): negotiate as a pluggable branch action (Phase 3 Plan 2)"
```

---

### Task 3: wire `spec.actions` + fix consumers

**Files:** Modify `sim/spec.py`, `sim/engine.py`, `tests/test_funnel_bid.py`, `scripts/run_slice.py`.

- [ ] **Step 1: failing test** — append to `tests/test_funnel_bid.py`:
```python
def test_default_funnel_has_no_negotiation():
    from datetime import datetime
    from sim.engine import Marketplace
    from sim.spec import MarketplaceSpec
    events = Marketplace.from_spec(
        MarketplaceSpec(start=datetime(2026, 1, 1), n_seed_users=120, until=6.0, seed=3)).run()
    kinds = {e.event_type for e in events}
    assert "bid" not in kinds and "lead" not in kinds   # consumer-only by default
```
And update the EXISTING `test_lead_and_bid_events_and_proposals_in_full_run` and `test_runs_with_bidding_are_reproducible` to register negotiation — change their spec construction to include the action. At the top of the file add `from sim.actions import negotiate_action`, and in those two tests pass `actions=[negotiate_action()]` to `MarketplaceSpec(...)`.

- [ ] **Step 2: run** `python -m pytest tests/test_funnel_bid.py -v` → FAIL (`MarketplaceSpec` has no `actions`).

- [ ] **Step 3: implement.**
(a) `sim/spec.py` — add to `MarketplaceSpec` (with the scalar/dict fields):
```python
    actions: list = field(default_factory=list)
```
(b) `sim/engine.py` — change `Market.__init__`'s funnel line from `self.actions = default_consumer_funnel()` to assemble extras. Update the import to also bring in `assemble_actions`, and set:
```python
        self.actions = assemble_actions(default_consumer_funnel(), spec.actions)
```
(import line becomes: `from sim.actions import assemble_actions, default_consumer_funnel, run_session as _run_session_actions`)
(c) `scripts/run_slice.py` — register negotiation in the demo spec so it shows the full funnel: add `from sim.spec import MarketplaceSpec` already present; add `from sim.actions import negotiate_action` and pass `actions=[negotiate_action()]` into the `MarketplaceSpec(...)` in `main()`.

- [ ] **Step 4: run** the FULL suite `python -m pytest -q` → all pass (incl. the updated funnel_bid tests, `test_default_funnel_has_no_negotiation`, reproducibility, no-threads, stock, pricing, lifecycle, settlement). If a non-trivial test fails, report BLOCKED — don't weaken it.

- [ ] **Step 5: verify harness** `python scripts/run_slice.py` → `reproducible: True`, and `lead`/`bid` present (negotiation registered).

- [ ] **Step 6: commit**
```bash
git add sim/spec.py sim/engine.py tests/test_funnel_bid.py scripts/run_slice.py
git commit -m "feat(sim): register pluggable actions via spec.actions (Phase 3 Plan 2)"
```

---

## Self-Review
- **Spec coverage:** §2.3 pluggable actions + declared hooks (before/after, gate/branch) → Task 1; negotiate as a shipped branch → Task 2; `spec.actions` registration + default = generic consumer funnel → Task 3.
- **Placeholders:** none — complete code per step, exact commands + expected results, and the planned cross-task failure (Task 2 Step 4 → Task 3) is called out explicitly.
- **Type/name consistency:** `Action(name, run, requires, fidelity, before, after, mode)`, `assemble_actions(base, extras)`, `negotiate_action()`, `session["negotiated"]` (set of `listing.id`), `MarketplaceSpec.actions`, `Market.actions = assemble_actions(default_consumer_funnel(), spec.actions)` — consistent. `_act_buy`/`_act_negotiate` call only existing `Market`/agents helpers.
- **Determinism:** `negotiate` runs before `buy`; both draw from the passed `rng` in list order; no new randomness source. Reproducibility tests guard it.
