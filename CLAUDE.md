# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agent-based **marketplace simulator** (think OLX/Adevinta-style classifieds). It generates
synthetic marketplace data from the ground up: heterogeneous users visit, list, view, bid, and
transact over a discrete daily loop. The repo is the **simulation engine only** — downstream
analysis (previously some R example scripts) has been stripped out to keep it clean.

## Running things

There is **no requirements file, no test suite, and no `main.py`**. The simulation is driven from
a Jupyter notebook; the canonical entry sequence (see `test_nb.ipynb`) is:

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

Python deps are implicit (no manifest): `numpy`, `scipy`, `scikit-learn`, `python-json-logger`,
and `jupyter_server`.

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

## Architecture (`classes.py`)

The whole engine lives in one file. Three layers:

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

## Experimental SimPy engine (`sim/`)

A continuous-time re-platform of the daily-loop engine lives in `sim/` (branch
`experimental/simpy-replatform`). It is **additive** — the legacy `classes.py` is
untouched. Design + scope: `docs/specs/2026-06-14-simpy-replatform-prd.md`; the
implementation plan: `docs/superpowers/plans/2026-06-14-simpy-replatform-slice.md`.

- `sim/spec.py` — `Property` (literal | scipy dist | callable) and `MarketplaceSpec`.
- `sim/events.py` — `Event` + in-memory `EventRecorder` (no threads; optional `write_jsonl`).
  Deliberately does **not** use `logger_setup.EventLogger` (its daemon thread + singleton
  break determinism and the no-threads guarantee).
- `sim/agents.py` — `User`/`Listing`, funnel probabilities, `user_lifecycle` +
  `population_arrival` SimPy processes.
- `sim/engine.py` — `Clock` (sim-days → datetime), `Market` runtime, `Marketplace.from_spec`/`run`.

Run it (from the repo root, using `python` = the conda base interpreter that has numpy/scipy):

```python
from datetime import datetime
from sim.engine import Marketplace
from sim.spec import MarketplaceSpec

mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                            n_seed_users=1000, until=7.0, seed=42))
events = mkt.run()   # list[Event], each with a datetime sim_time
```

Tests: `python -m pytest` from the repo root. Smoke harness: `python scripts/run_slice.py`.
The engine is single-threaded and deterministic: same spec + same `seed` → identical events.
Base time unit is **days** (`env.now` in days; `until` is sim-days).

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
