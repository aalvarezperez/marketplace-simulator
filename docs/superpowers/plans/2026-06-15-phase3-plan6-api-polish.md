# Phase 3 · Plan 6 — API Polish (A + C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ergonomics, no behavior change. (A) One-import surface + top-level results (`from sim import ...`, `mkt.write_jsonl`, `mkt.summary`). (C) A grouped `MarketplaceSpec` docstring documenting the knobs. (B — consideration curation — deferred.)

**Tech Stack:** Python, pytest. `python` = conda base, from repo root.

---

### Task 1: top-level exports + result helpers + spec docstring

**Files:** Modify `sim/__init__.py`, `sim/engine.py`, `sim/spec.py`, `scripts/run_slice.py`, `CLAUDE.md`; Create `tests/test_api.py`.

- [ ] **Step 1: failing test** — `tests/test_api.py`:
```python
from datetime import datetime


def test_top_level_imports():
    from sim import (Marketplace, MarketplaceSpec, Property, negotiate_action,
                     default_pricing, default_willingness)
    assert all(x is not None for x in (Marketplace, MarketplaceSpec, Property,
                                       negotiate_action, default_pricing, default_willingness))


def test_summary_returns_event_counts():
    from sim import Marketplace, MarketplaceSpec
    mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                                n_seed_users=50, until=3.0, seed=1))
    mkt.run()
    s = mkt.summary()
    assert isinstance(s, dict) and s.get("visit", 0) > 0


def test_write_jsonl_top_level(tmp_path):
    from sim import Marketplace, MarketplaceSpec
    mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                                n_seed_users=20, until=2.0, seed=1))
    mkt.run()
    p = tmp_path / "e.jsonl"
    mkt.write_jsonl(p)
    assert p.exists() and p.read_text().strip()


def test_spec_has_grouped_docstring():
    from sim import MarketplaceSpec
    assert MarketplaceSpec.__doc__ and len(MarketplaceSpec.__doc__) > 100
```

- [ ] **Step 2: run** `python -m pytest tests/test_api.py -v` → FAIL (no top-level exports / `summary` / `write_jsonl`).

- [ ] **Step 3: implement.**
(a) `sim/__init__.py` (replace contents):
```python
"""SimPy-based marketplace simulation engine (experimental).

    from sim import Marketplace, MarketplaceSpec, negotiate_action

    mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1)))
    events = mkt.run()
    mkt.summary()                 # {'visit': ..., 'view': ..., 'transaction': ...}
    mkt.write_jsonl("events.jsonl")
"""
from sim.actions import negotiate_action
from sim.engine import Marketplace
from sim.pricing import default_pricing
from sim.spec import MarketplaceSpec, Property
from sim.willingness import default_willingness

__all__ = [
    "Marketplace", "MarketplaceSpec", "Property",
    "negotiate_action", "default_pricing", "default_willingness",
]
```
(b) `sim/engine.py` — add two methods to `Marketplace` (after the `events` property):
```python
    def write_jsonl(self, path):
        """Dump the event stream to JSON lines."""
        self.market.recorder.write_jsonl(path)

    def summary(self):
        """Event-type counts for the run, e.g. {'visit': 1816, 'view': 2351, ...}."""
        from collections import Counter
        return dict(Counter(e.event_type for e in self.events))
```
(c) `sim/spec.py` — add a grouped docstring to `MarketplaceSpec` (immediately under `class MarketplaceSpec:`), documenting the knobs by group. Keep field order unchanged:
```python
    """Declarative marketplace definition. Pass to ``Marketplace.from_spec``.

    Run controls:
      start                  - calendar start (datetime, required)
      seed                   - RNG seed (deterministic runs)
      n_seed_users, until    - seed population; sim-days to run
      arrival_rate           - new users per day

    Agent dispositions (Property: literal | scipy dist | callable | context-model):
      engagement, response_time, value_factor, seller_patience

    Funnel / lifecycle:
      proposal_expiry_days, reactivation_scale_days, listing_ttl_days
      variant_weights        - A/B split, e.g. {"CONTROL": .5, "B": .5}

    Seller pricing / behavior:
      pricing                - pricing(seller, quality, market, rng) -> price
      willingness            - willingness(agent, listing, market) -> WTP
      markdown_pct           - stale-listing markdown step (0 disables)

    Composition:
      actions                - extra Action()s to register, e.g. [negotiate_action()]
      listings_per_user, listing_quality
    """
```
(d) `scripts/run_slice.py` — dogfood the new surface: replace the `from sim.engine ... / from sim.spec ... / from sim.actions ...` imports with `from sim import Marketplace, MarketplaceSpec, negotiate_action`, and replace `mkt.market.recorder.write_jsonl("slice_events.jsonl")` with `mkt.write_jsonl("slice_events.jsonl")`. (Keep the existing sys.path bootstrap and the rest of `main()`.)
(e) `CLAUDE.md` — in the "Experimental SimPy engine" run snippet, change `from sim.engine import Marketplace` + `from sim.spec import MarketplaceSpec` to the single line `from sim import Marketplace, MarketplaceSpec`.

- [ ] **Step 4: run** `python -m pytest tests/test_api.py -v` (4 passed), then FULL suite `python -m pytest -q` (no regressions — existing `from sim.engine`/`from sim.spec` imports still work alongside the new top-level ones). Then `python scripts/run_slice.py` → still `reproducible: True`.

- [ ] **Step 5: commit**
```bash
git add sim/__init__.py sim/engine.py sim/spec.py scripts/run_slice.py CLAUDE.md tests/test_api.py
git commit -m "feat(sim): top-level imports + mkt.write_jsonl/summary + spec docstring (Phase 3 Plan 6)"
```

---

## Self-Review
- **Spec coverage:** A (one-import via `sim/__init__.py`; `Marketplace.write_jsonl`/`summary`; dogfood harness + CLAUDE.md) + C (grouped `MarketplaceSpec` docstring). B (consideration curation) intentionally deferred.
- **Placeholders:** none — complete code, exact commands.
- **No behavior change / determinism:** only adds exports, two read-only helper methods, and docs; no engine logic touched. No circular import (submodules import specific `sim.X`, never the `sim` package).
- **Type/name consistency:** `from sim import Marketplace, MarketplaceSpec, Property, negotiate_action, default_pricing, default_willingness`; `Marketplace.write_jsonl(path)`, `Marketplace.summary()`.
