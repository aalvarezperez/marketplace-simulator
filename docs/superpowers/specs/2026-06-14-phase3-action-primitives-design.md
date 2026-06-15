# Phase 3 Design — Action Primitives & Emergent Marketplace

**Date:** 2026-06-14
**Branch:** `experimental/simpy-replatform` (v2.0 line; no merge to `main` until explicitly approved)
**Status:** Design — awaiting review before planning
**Predecessors:** PRD `docs/specs/2026-06-14-simpy-replatform-prd.md`; Phase 2 roadmap `docs/specs/2026-06-14-phase2-roadmap.md`.

---

## 1. Context

Phase 2 brought the SimPy engine to feature parity, but the funnel is **hardcoded** in
`sim/agents.py::_run_session` and its conversions are **sigmoid rates of engagement**. Two problems
this design fixes:

- **Rates are knobs, not consequences.** Conversion should *emerge* from agents interacting with
  each other and with stock — not be set by a `p_buy`/`p_view` constant. (A rate is something you
  *measure* from a run, never something you configure.)
- **The funnel is baked in.** `view`/`bid` are classifieds-specific yet live in engine code. A user
  can't compose a different marketplace dynamic without editing classes.

**Guiding principle for this design: keep it simple.** We are always emulating a *marketplace* —
this is not a general-purpose agent framework. Build the smallest thing that makes agents+actions
the primitives, the funnel emergent, and the specific bits pluggable. No DSLs, no discrete-choice
econometrics, no arbitrary recursion.

---

## 2. Primitives: agents and their actions

Agents and the actions they take are the **only** primitives. Everything downstream — the funnel,
conversion, prices, market dynamics — is an emergent consequence and is *measured*, never set.

### 2.1 `Action`
A small object:
- `name` — e.g. `view`, `buy`.
- `requires` — prior action(s) on the same target; **defaults to the funnel predecessor** (so
  declaring order *is* the basic precondition; richer predicates are optional).
- `fidelity` — `"explicit"` or `"implicit"`.
- `decide(agent, option, market) -> bool` — used when **explicit** (the emergent decision).
- `rate` — a flat propensity used when **implicit** (the cheap stand-in).
- `effect(agent, option, market)` — the state change + event it produces.
- `target` — what it operates on (`listing` | `none` | `proposal` | …); some actions produce a set
  (search → candidates, consideration → a curated set) that later actions consume.

### 2.2 Consumer funnel — concrete predefined default
Shipped as an ordered list of `Action`s, all built from the same primitive (inspectable,
overrideable — but **not** a meta-framework):

`visit → search → view → consideration → buy`

`search` returns candidates from the agent's sub-market; `consideration` curates a set; `buy` takes
the best option **if the agent's willingness clears**. Drop-out at any step is the emergent
conversion — there is no conversion *rate* anywhere in the core path. A single sub-market with this
funnel reproduces today's observable behaviour.

### 2.3 Pluggable actions
Register an extra action that **declares its own hook**; base actions are never edited (open/closed):
- position: `before` / `after` an existing action,
- mode: `gate` (mandatory) or `branch` (alternative).

`negotiate` ships as a **branch before `buy`**: present → some agents negotiate instead of buying
now (reusing the Phase 2 `Proposal`/settlement machinery); absent → the funnel runs untouched.

### 2.4 Emergent decision + explicit/implicit fidelity
When an action is available:
- **explicit** → the agent evaluates the option through a **simple willingness/disposition** check
  (e.g. quality-vs-price against a per-agent reservation), competing for scarce stock. Conversion is
  a genuine consequence of agent × option × competition. *Deliberately simple* — no utility/logit
  framework.
- **implicit** → a flat `rate` stand-in (this is where the old sigmoid lands).

You make **explicit only the step your experiment perturbs** — whatever your method/intervention
targets — and leave the rest **implicit and cheap**. This is the core knob of the whole design, and
it also dissolves the Phase 2 "negotiation never settles" calibration problem: buy/negotiate
outcomes emerge from willingness meeting price and stock, not from fixed competing rates.

---

## 3. Sub-markets (two layers)

A `Marketplace` holds **sub-markets**; a sub-market is a **partition label** (category, origin/
destination pair) that acts as a **matching boundary**: `search` only returns listings in the
agent's sub-market.
- Agents and listings each belong to one sub-market (a buyer drawn in via a categorical at spawn; a
  listing inherits its seller's sub-market).
- The same agent/action machinery runs per sub-market; sub-markets are independent matching pools;
  metrics roll up to the market.
- **Two layers only — no recursion.** A single sub-market is the degenerate default (today's flat
  market), so nothing breaks.

---

## 4. Properties

No new abstraction — reuse `Property.draw(rng, context)` (literal | scipy dist | callable |
context-model) from Epic B. Phase 3 leans on it: any agent/listing property is **fixed or a
generator**, and `context` is what lets a **dependency arise** (`quality ~ seller.engagement`,
`price ~ market`). The sub-market is exposed in `context`.

---

## 5. User-facing API

One declarative spec, kept crisp and extensible where it matters:
- `MarketplaceSpec` gains `submarket_weights` (categorical; omit → single market) and an `actions`
  list for extras (the default consumer funnel is always present).
- An extra action declares its hook, e.g. `Action("negotiate", before="buy", mode="branch",
  fidelity="explicit", decide=..., effect=...)`.
- Fidelity is set **per action** (`explicit`/`implicit`); the core funnel ships explicit; flip the
  steps you are *not* studying to implicit for speed.
- Entry stays `Marketplace.from_spec(spec)`; same timestamped event stream; same determinism.

---

## 6. Non-goals (this design)

- **No general-purpose action DSL / rules engine** beyond `requires` + a simple hook.
- **No discrete-choice / econometric utility framework** — willingness stays a simple check.
- **No market recursion** beyond two layers.
- **No configurable funnel rates as the primary model** — a rate exists *only* as the implicit
  stand-in for a step you chose not to simulate.
- **No change** to the SimPy substrate, event-stream format, or determinism guarantees.

---

## 7. Migration & compatibility

- Rearchitect `sim/agents.py::_run_session` into a small **action runner** that walks the registered
  action list per session, scoped to the agent's sub-market. The Phase 2 behaviours (proposals,
  settlement, pricing, variants, churn/expiry) are preserved — the funnel actions wrap them; e.g.
  `buy`/`negotiate` reuse `transact` and the `Proposal` pipeline.
- The hardcoded sigmoid `p_view`/`p_buy`/`p_lead`/`p_bid` become the **implicit-fidelity rates**;
  the explicit path replaces them with the willingness decision.
- The existing **49 tests are the regression harness**: the default consumer funnel (single
  sub-market, explicit core) must keep producing the funnel event stream, and runs must stay
  byte-identical reproducible under the single seeded `rng`. New tests cover: action ordering/
  preconditions, a pluggable `negotiate` branch, two-sub-market matching isolation, and
  explicit-vs-implicit fidelity producing comparable funnels.

---

## 8. Open questions (resolve at plan time)

1. **Willingness functional form** — quality/price ratio vs `quality − λ·price` vs reservation
   price; pick the simplest that yields sensible emergent conversion (calibration, not architecture).
2. **Consideration-set semantics** — top-k by willingness, or a sampled set; how `buy` picks among
   competing buyers for one stock unit (FCFS via SimPy ordering is the likely answer).
3. **Action runner shape** — how set-producing actions (`search`, `consideration`) feed per-item
   actions (`view`) cleanly without special-casing.
4. **How much Phase 2 calibration to fold in** — whether to retire `BUY_BASE`/`p_*` sigmoids
   entirely or keep them as the shipped implicit defaults.

---

## 9. Verification

- Default spec (single sub-market, explicit funnel): full event stream present, deterministic
  (same seed → identical), no threads — the Phase 2 suite stays green.
- Pluggable `negotiate` branch: registering it produces negotiation/proposal events; removing it
  leaves a clean buy-only funnel.
- Two sub-markets: an agent only ever views/buys listings in its own sub-market (matching isolation).
- Fidelity: a step run `implicit` (rate) vs `explicit` (willingness) both produce a coherent funnel;
  flipping fidelity changes cost, not the pipeline's validity.

---

## 10. Price-inflation — RESOLVED by Plan 4 (median-of-comparable pricing)

> **Update (Plan 4, 2026-06-15):** RESOLVED. `EndogenousPrice` (regression on top-k-by-quality →
> extrapolation/outlier inflation to ~2× value) was deleted and replaced with a decoupled `pricing`
> callable; the default `default_pricing` lists at the **median ask of the K nearest-quality live
> listings** (cold-start → quality-anchored prior). Sellers price off observable asks (realistic),
> but the local median is a stable fixed point with no extrapolation. Verified: price/quality median
> ≈ 0.82 (was ~2.0), direct-buy conversion restored (was ~0). The **liquidity / time-to-sold**
> correction (stale → markdown, fast → raise) and a **platform-recommended-price** strategy remain
> future plans — the callable already accommodates both. Original analysis kept below for history.



**Observed (after Plan 3):** with the emergent willingness buy, the default marketplace makes
~0 direct purchases. Diagnostic on a default run: `price/quality` ratio ≈ **2.0** (median), while
WTP ≈ `quality × value_factor` with `value_factor` ≈ 1 — so almost no ask clears, and demand
collapses.

**Root cause (not the willingness model — that's correct and tested):** the Epic B
`EndogenousPrice` model has sellers price off *other listings' asks* (regression of price~quality on
the top-k-by-quality listings). That's a **positive feedback with no demand anchor** — high asks pull
the regression up, new listings price higher, which become the new top, and prices drift to ~2×
intrinsic value (`v = quality`). Sticky WTP can't follow → conversion dies. It's the price-surge
dynamic, but *runaway*.

**Fix (deferred to the supply-side work — do NOT patch with a hardcoded anchor):** sellers should set
price **following the market outcome**, not merely quality/other-asks — i.e. respond to realized
**demand** (what actually sells, unsold-inventory pressure, competition), which introduces a
**negative feedback**: listings that don't move get repriced *down* toward what buyers will pay. This
tethers price to WTP endogenously, restores two-sided elasticity, and keeps transient price-out
without permanent inflation. No artificial value-anchor needed.

**Action item:** add a supply-side pricing plan (a future "Plan 6 — market-following seller pricing")
that replaces/extends `EndogenousPrice` with a demand-aware repricing strategy. Until then, the
direct-buy path will under-convert by default; the negotiation path still clears (settlement accepts
bids independent of WTP). Calibrate/verify direct-buy conversion once this lands.
