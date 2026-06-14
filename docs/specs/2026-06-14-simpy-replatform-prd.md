# PRD — SimPy Re-platform of the Marketplace Simulator

**Date:** 2026-06-14
**Branch:** `experimental/simpy-replatform` (off a clean `main`)
**Status:** Approved

---

## 1. Context

The engine today (`classes.py`) is a hand-rolled **discrete daily loop**. Each day it samples
visitor/buyer/seller cohorts via `1 - exp(-engagement/scale)`, fans them out over a
`ThreadPoolExecutor`, and advances one day. Three problems block extension:

- **Concurrency is broken.** The executor is GIL-bound (~99% of time in `_thread.lock.acquire`);
  the thread-safe per-user variants are commented out, so the loop is not day-idempotent and buys
  no real parallelism.
- **Time is too coarse.** A "day" is the atomic unit. `response_time` is drawn per user and stored
  in `__slots__` but **never used** — latency was intended and never modelled. Proposal expiry,
  reply lag, return visits, intraday dynamics cannot be expressed.
- **No real config surface.** The user passes two gamma distributions + a `listing_kwargs` dict.
  Everything else is hardcoded: `Category.ELECTRONICS`, funnel sigmoid rates, the quality/price
  draws. Adding a property means editing engine internals.

**Decision (settled in brainstorming):** re-platform the simulation onto **SimPy** (process-based
discrete-event simulation). SimPy gives a single-threaded, deterministic, continuous-time event
loop — killing the concurrency mess and making time first-class. The work lands on an
**experimental branch** with the old engine **frozen** so the two can be compared and the change
reverted freely.

> Note: SimPy replaces the *loop / clock / concurrency*. It does **not** provide the config layer —
> the declarative spec API below is designed on top of it.

---

## 2. Goals

1. Continuous-time, event-driven simulation on **SimPy 4.x**. Deterministic given a seed.
2. **Declarative spec object** as the end-user API: build one `MarketplaceSpec`, call
   `Marketplace.from_spec(...)`, run. Introspectable and shareable as data.
3. **Calendar-aware (datetime)** clock; every event carries a real timestamp.
4. **Timestamped event stream** as the canonical output for downstream analysis.
5. Population dynamics: **seed N agents at t0** *and* **spawn new agents mid-run** via a process.
   Full agent lifecycle (active → dormant/churn → reactivation) + listing expiry is the **target**.
6. Realise the **fixed-as-dynamic** principle: any entity property is a literal *or* a generator
   (distribution / callable / model), resolved through one abstraction.
7. **Phased delivery**: a thin vertical slice first to de-risk the paradigm, then feature parity.

---

## 3. Non-goals (explicit, this PRD)

- **Recursive / nested sub-markets.** Design stays compatible with principle #3; nesting is not built.
- **Multi-category de-hardcoding.** Removing the baked-in `ELECTRONICS` assumption is a separate effort.
- **Endogenous pricing in the slice.** The `LinearRegression` price~quality model is ported in the
  parity phase, not the slice.
- **Improving the old threaded engine.** `classes.py` is frozen as-is for A/B comparison.
- **User-configurable funnel rates in v1.** View/lead/bid sigmoid base-rates and slopes stay
  engine constants; they get exposed in the spec in a later milestone.

---

## 4. Users & primary use case

A researcher/analyst emulating a marketplace (OLX, Marktplaats, booking.com, …) to generate
synthetic, event-level data for downstream analysis — funnel conversion, supply/demand dynamics,
and built-in A/B experiments. They want to *declare* the marketplace (entities, their property
generators, population/arrival behaviour, run window) and get a reproducible timestamped event log.

---

## 5. Design overview

### 5.1 Substrate
- A SimPy `Environment` owns the clock and event queue. Single-threaded, deterministic.
- **Calendar mapping:** spec provides a `start: datetime`; SimPy float time advances in a fixed
  base unit (proposed: seconds) and is rendered to real timestamps for events. v1 = **timestamps
  only**: rates are flat across the calendar, with clean hooks left for weekday/seasonal modulation
  later.
- Reproducibility: a single `seed` seeds both the numpy RNG and any stochastic scheduling.

### 5.2 Agent process model — persistent lifecycle per agent
- Each `User` is a long-lived generator process. Lifecycle: **active** (schedules its next action
  via an inter-arrival draw whose rate rises with `engagement`) → acts → reschedules; can go
  **dormant/churn**; can **reactivate**. (Slice ships a thin version: active + acts; churn/expiry/
  reactivation land in parity.)
- **Seed:** batch-create N agents at t0, each launches its lifecycle process.
- **Acquisition:** a population-arrival process mints new agents over time at a spec-configured rate.
- The old daily `1 - exp(-engagement/scale)` cohort probabilities become **continuous-time arrival
  rates** (e.g. exponential inter-arrival, rate ∝ engagement).

### 5.3 Continuous-time funnel
- Keep the existing sigmoid funnel (`view → make_lead → bid`), engine constants in v1.
- `response_time` is finally used: seller reply latency, buyer settle latency, and **proposal
  expiry** become scheduled timeouts. Inboxes/proposal queues map to SimPy `Store`s in the parity
  phase.

### 5.4 Config — the declarative spec
- `MarketplaceSpec` (dataclass) holds: entity **property generators**, population/arrival params,
  lifecycle params, run controls (`start` datetime, stop condition), and `seed`.
- A `Property` abstraction realises **fixed-as-dynamic**: resolves a literal **or** samples a
  distribution / calls a model. A literal is just the degenerate generator.
- `Marketplace.from_spec(spec)` builds the environment, seeds agents, registers processes, returns
  a runnable market.

### 5.5 Output — timestamped event stream
- Reuse the existing `logger_setup.EventLogger` (thread-safe JSON-lines singleton). Every
  view/lead/bid/proposal/transaction is emitted with: sim datetime, event type, and entity ids.
  Aggregation happens downstream in analysis, not in the engine.

### 5.6 Module layout
- New package, e.g. `sim/`:
  - `sim/spec.py` — `MarketplaceSpec`, `Property` generators.
  - `sim/engine.py` — environment builder, `Marketplace.from_spec`, `run`.
  - `sim/agents.py` — agent lifecycle processes + funnel behaviour.
  - `sim/events.py` — event type definitions (or thin wrappers over `EventLogger`).
- **Reuse:** `func.sigmoid`, `logger_setup.EventLogger`.
- **Frozen, untouched:** `classes.py`.
- Add a dependency manifest (repo has none today) pinning `simpy` + existing implicit deps
  (`numpy`, `scipy`, `scikit-learn`, `python-json-logger`).

---

## 6. Phasing

- **Phase 0 — branch hygiene.** Commit `main` to a clean baseline (the R-strip deletions +
  `CLAUDE.md` + `Design principles.md`), then branch `experimental/simpy-replatform`. Add the
  dependency manifest with `simpy`.
- **Phase 1 — vertical slice.** Minimal end-to-end on SimPy: spec → seed agents → continuous
  arrivals → `arrive → view → transact` → timestamped event stream → reproducible across two
  seeded runs. Thin lifecycle (sold listings removed). **Exit criteria:** runs deterministically,
  emits a non-empty timestamped event stream, and holds up at the target population size.
- **Phase 2 — parity.** Full funnel (`lead → bid → proposal → transaction`), seller/buyer inboxes
  via `Store`, proposal lifecycle + expiry, A/B `Variant` split, endogenous pricing port, and full
  agent lifecycle (churn / listing expiry / reactivation).
- **Phase 3 — later (out of this PRD).** User-configurable rates, calendar effects
  (weekday/seasonality), recursive markets, multi-category.

---

## 7. Requirements

**Functional**
- R1. `Marketplace.from_spec(MarketplaceSpec)` constructs and returns a runnable market.
- R2. Spec seeds N agents at t0 and runs a population-arrival process minting new agents over time.
- R3. Each agent acts on a continuous-time schedule driven by its `engagement`.
- R4. Slice funnel: `arrive → view → transact`; parity funnel adds `lead → bid → proposal → paid`.
- R5. Every event is logged with a real datetime timestamp, event type, and entity ids.
- R6. Property values accept a literal **or** a generator through one `Property` abstraction.

**Non-functional**
- R7. Deterministic: same spec + same seed → byte-identical event stream.
- R8. Single-threaded; no `ThreadPoolExecutor`, no locks.
- R9. Handles the target seed population (validate against ~50k in the slice; record the ceiling).
- R10. Old `classes.py` remains importable and unchanged.

---

## 8. Open questions (resolve during implementation)

1. **Base time unit** for the SimPy float clock — seconds (proposed) vs days.
2. **Stop condition** — run-until-`datetime`, run-for-duration, or until-N-events.
3. **Event sink** — JSON-lines file (reuse `EventLogger`) vs in-memory structure handed back for
   analysis. Proposed: file, with an optional in-memory collector.
4. **Perf ceiling** at the persistent-process model — confirm in the slice; if 50k persistent
   generators are too heavy, fall back to arrival-spawned session processes.

---

## 9. Verification

- **Slice:** run a small spec (e.g. 1k agents, 7 sim-days). Assert: event stream non-empty;
  per-process timestamps monotonic; two runs with the same seed produce identical output; no
  threads spawned. Sanity-check aggregate funnel shape against the old engine's range.
- **Parity:** a proposal reaches `paid`; `Variant` assignment splits the population; the ported
  pricing model yields a sensible price~quality relationship.
- **Reproducibility harness:** a tiny script/notebook that runs the same seed twice and diffs the
  event logs.

---

## 10. Critical files

- **New:** `sim/spec.py`, `sim/engine.py`, `sim/agents.py`, `sim/events.py`, dependency manifest
  (e.g. `requirements.txt` or `pyproject.toml`).
- **Reuse:** `func.py` (`sigmoid`), `logger_setup.py` (`EventLogger`).
- **Frozen:** `classes.py`.
