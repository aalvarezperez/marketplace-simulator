# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agent-based **marketplace simulator** (think OLX/Adevinta-style classifieds) that generates
synthetic marketplace data from the ground up: heterogeneous users visit, list, view, bid, and
transact. The repo is the **simulation engine only** — downstream analysis (previously some R
example scripts) has been stripped out.

There are **two engines**:
- **`sim/` — the active engine (v2.0):** a deterministic, single-threaded, **continuous-time
  SimPy** model where agents + actions are the primitives and the funnel/conversion/prices emerge.
  This is what you should read and extend. See "Active engine" below.
- **`classes.py` — the legacy engine, frozen:** the original discrete-daily-loop implementation,
  kept as a reference for behaviour parity. Do not edit it. Sections below labelled `classes.py`
  describe this legacy engine.

## Running things

**Active engine (`sim/`):** `pip install -r requirements.txt` (`simpy`, `numpy`, `scipy`,
`scikit-learn`, `pytest`), then drive it from Python — see the "Active engine" section for the
canonical snippet. Tests: `python -m pytest` from the repo root (deterministic, ~80 tests). Use
`python` = the conda base interpreter, not `python3`. Smoke harness: `python scripts/run_slice.py`.

**Legacy engine (`classes.py`, frozen):** notebook-driven (`test_nb.ipynb`), no test suite; the
canonical entry was:

```python
from scipy.stats import gamma
from classes import Marketplace, Category

market = Marketplace(name="olx",
                     engagement_heterogeneity=gamma(a=2, scale=7/2),
                     response_heterogeneity=gamma(a=2, scale=1/2))
market.reset()
market.initialise(n_users=50000, listing_kwargs={'category': Category.ELECTRONICS})
market.run_n_day(1)
```
It additionally imports `python-json-logger` and `jupyter_server` (a dead import — see below).

## Design principles (`Design principles.md`)

The author's intent, which the code only partly realises yet — respect it when extending:

1. **Generic entities.** `User`/`Listing` are deliberately domain-agnostic. A "seller" could be a
   booking.com partner or a Marktplaats seller; a "buyer" likewise. Don't bake in classifieds-only
   assumptions.
2. **Fixed is a special case of dynamic.** Every property should be settable either as a one-off
   value or via a model. `quality` and `price` already show this (a literal, or a statistical
   model — see `price_model` vs `NaiveMeanPriceModel`). New properties should follow the same
   pattern rather than hardcoding a scalar.
3. **Recursive marketplace structure.** A `Marketplace` is meant to be nestable — a market can be
   a sub-market of a bigger one, so a market can be a mixture of sub-markets. The current
   `Marketplace` is flat; this is the direction, not the current state.

## Legacy engine architecture (`classes.py`, frozen)

The whole legacy engine lives in one file. Three layers:

**1. Agents & objects** — `User`, `Listing`, `Proposal` all use `__slots__`. Heterogeneity is the
core idea: each `User` has an `engagement` and `response_time` drawn from gamma distributions at
`initialise()`. Almost every behavioural probability is `sigmoid(log(base_rate) + slope*log(engagement|quality))`,
so engagement/quality drive the entire funnel.

**2. `Marketplace` — discrete daily loop.** `run_n_day(n)` iterates days. Each day:
- `open()` samples three overlapping cohorts from the user base via engagement-driven
  `1 - exp(-engagement/scale)` probabilities: **visitors**, **buyers**, **sellers**.
- `create_listing_match_set(k=100)` takes the top-quality live listings and probabilistically
  displays them (the marketplace "ranking/match" step).
- Cohorts run concurrently in a `ThreadPoolExecutor`: visitors `check_inbox`, sellers `list`,
  buyers `curate_listings` then `engage_with_listing`.
- `clean_up_gone_listings()` drops sold-out (`is_live=False`) listings.

**3. The funnel** (`User.engage_with_listing`): `view` → `make_lead` → `bid`. A `bid` creates a
`Proposal` priced below ask; proposals flow **created → with_seller → accepted → with_buyer →
paid**. Sellers pick the highest bid per listing in `evaluate_proposals`; buyers settle accepted
ones in `check_accepted_proposals`. Direct buys go through `transact`.

**Pricing** is endogenous ("free market" assumption): a `User` fits a `LinearRegression` of
price~quality on the top-10 listings it can "see" (`_pricing_research_prices`), caching it as
`self.price_model`, and falls back to `NaiveMeanPriceModel` when there's nothing to learn from.
Prices emerge from supply/demand rather than being assigned.

**A/B testing** is built in via the `Variant` enum (`CONTROL/B/C/D`) on each `User`.

**Object creation** goes through thin factories: `RegistrationFlow.complete()` builds a `User`,
`SellYourItemFlow.complete()` builds a `Listing` — and `User.list()` actually routes through the
latter. The `start()` methods are empty stubs (placeholders for future conversion steps).

## Hardcoded assumptions

Category is **not** fully parameterised. `initialise(listing_kwargs=...)` only sets the category
of the *seed* listings. The daily loop hardcodes `Category.ELECTRONICS` (`run_n_day` → `m, c =
self, Category.ELECTRONICS`), so every listing created on a running day is ELECTRONICS regardless.
The pricing fallback in `_pricing_research_prices` also hardcodes the ELECTRONICS price
distribution. Changing category for real means touching all three spots.

## Concurrency gotcha

`run_n_day` uses threads, but the per-user thread-safe variants (`User.list_w_thread_lock`,
`check_inbox_w_thread_lock`, the `_lock`/`_last_*_day` slots) are **commented out**. The active
path is not day-idempotent, and profiling (`test_nb.ipynb`) shows ~99% of time in
`_thread.lock.acquire` — it's GIL-bound, so the `ThreadPoolExecutor` buys almost no parallelism.
Treat this code as single-effective-threaded when reasoning about it.

Also note: `classes.py` imports `User` from `jupyter_server.auth` at the top, but that name is
immediately shadowed by the local `class User` (defined lower in the same file). The local class
is what actually runs — the import is dead weight.

## Logging

`logger_setup.EventLogger` is a thread-safe **singleton** with a background flush thread, writing
JSON lines to `marketplace.log`. Most `logger.info(...)` calls in `classes.py` are commented out,
so the log is usually empty.

## Active engine — SimPy continuous-time (`sim/`)

`sim/` is the **active engine** (v2.0, branch `experimental/simpy-replatform`); the legacy
`classes.py` daily-loop engine is **frozen** as a reference (do not edit it). The replatform moved
off the GIL-bound discrete-day loop to a deterministic, single-threaded, continuous-time SimPy
model. Design trail (point-in-time records) in `docs/specs/` and `docs/superpowers/plans/`: the PRD,
the Phase 2 parity roadmap, the Phase 3 action-primitives design spec, and per-plan implementation
plans.

**Core idea: agents and their actions are the only primitives; the funnel, conversion, and prices
are emergent consequences — never configured rates.** A "rate" exists only as the *implicit*
fidelity stand-in for a step you chose not to simulate. (See the memory notes for the rationale.)

**The funnel is a declarative list of `Action`s** walked by a runner (`run_session`), each gated on
its `requires`. Default consumer funnel: `visit → list → search → view → consideration → buy`. Extra
actions register via `spec.actions` and declare a hook (`before`/`after`, `gate`/`branch`);
`negotiate_action()` ships as a *branch before `buy`* (the classifieds add-on: `lead → bid →
Proposal`). Base actions are never edited — open/closed.

**Conversion emerges; there is no buy-rate.** `buy` is the *explicit* step: an agent buys iff its
willingness `wtp(agent, listing, market) ≥ listing.price`. Default willingness =
`quality × value_factor` — **intrinsic and sticky** (it ignores the live price), so a price surge
prices out low-`value_factor` agents (real demand elasticity). `value_factor` is a per-agent
`Property`. `view`/`lead`/`bid` remain *implicit* sigmoids; flip any step's fidelity per what you study.

**Seller pricing** is a decoupled, swappable callable `pricing(seller, quality, market, rng)`.
Default `default_pricing`: list at the **median ask of the K nearest-quality live listings** (sellers
price off observable *asks*, since sale prices aren't visible); cold-start → a quality-anchored
prior. This replaced the old regression model (`EndogenousPrice`, **deleted**) which extrapolated and
ran prices away to ~2× value.

**Liquidity correction:** an unsold listing marks down by `markdown_pct` every `seller.patience` days
(`markdown_listing`); `patience` is a per-seller `Property` (normal by default, scaled to `until`).
The *up* direction is emergent — cleared cheap stock leaves the live set, raising the comparable
median. Two-sided from one explicit mechanism.

**Other behaviour:** `Proposal`s route buyer↔seller through SimPy `Store` inboxes with `response_time`
latency + expiry; A/B `variant` (spec `variant_weights`) is stamped on every event; agents
`churn → dormant → reactivate`; listings expire at a TTL.

Files:
- `sim/spec.py` — `Property` (literal | scipy dist | callable | context-model, via `draw(rng, context)`)
  + `MarketplaceSpec` (all knobs; see its docstring).
- `sim/actions.py` — `Action`, `run_session`, `assemble_actions`, `default_consumer_funnel`,
  `negotiate_action`, `buy_action`.
- `sim/willingness.py` — `default_willingness` (intrinsic sticky WTP).
- `sim/pricing.py` — `default_pricing` (median of comparable asks).
- `sim/agents.py` — `User`/`Listing`/`Proposal`; funnel probabilities; SimPy processes
  (`user_lifecycle`, `population_arrival`, `settlement_process`, `proposal_expiry`, `reactivation`,
  `listing_expiry`, `markdown_listing`).
- `sim/engine.py` — `Clock`, `Market` runtime, `Marketplace` (`from_spec`/`run`/`events`/`summary`/`write_jsonl`).
- `sim/events.py` — `Event` (+ `payload`) + in-memory `EventRecorder`. Deliberately **not**
  `logger_setup.EventLogger` (its daemon thread + singleton break determinism and the no-threads guarantee).

**Invariants (hold them when extending):** single seeded `numpy` rng → byte-identical reproducible
runs; single-threaded (no threads/locks); counter-based ids (never `hash()`/`datetime.now()`);
generic entities (no classifieds baked into the engine); base time unit = **days** (`env.now`/`until`).

Run it:
```python
from datetime import datetime
from sim import Marketplace, MarketplaceSpec, negotiate_action

mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                            n_seed_users=1000, until=7.0, seed=42,
                                            actions=[negotiate_action()]))   # negotiation is optional
events = mkt.run()        # list[Event]; also mkt.summary() and mkt.write_jsonl(path)
```
Tests: `python -m pytest` (use `python` = the conda base interpreter; has numpy/scipy/scikit-learn/simpy/pytest).
Smoke harness: `python scripts/run_slice.py`. Same spec + same `seed` → identical events.

## Files

- `classes.py` — the legacy daily-loop engine (Marketplace, User, Listing, Proposal, Variant); frozen.
- `sim/` — the experimental SimPy engine (see section above).
- `func.py` — `sigmoid()` (shared by both engines).
- `logger_setup.py` — threaded singleton `EventLogger` (legacy engine only).
- `test_nb.ipynb` — run + profile the legacy sim.
- `scripts/run_slice.py` — smoke + reproducibility harness for the SimPy engine.
- `requirements.txt` — deps for the SimPy slice (`simpy`, `numpy`, `scipy`, `pytest`).
- `Design principles.md` — author's design intent (generic entities, fixed-as-dynamic, recursive markets).
- `docs/` — PRD + implementation plan for the SimPy re-platform.
