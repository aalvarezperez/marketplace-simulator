# Phase 2 Roadmap — SimPy Engine to Feature Parity (v2.0)

**Date:** 2026-06-14
**Branch:** `experimental/simpy-replatform` (stays open until the whole v2.0 major release lands; no merge to `main` before then)
**Predecessors:** PRD `docs/specs/2026-06-14-simpy-replatform-prd.md`; slice plan `docs/superpowers/plans/2026-06-14-simpy-replatform-slice.md` (done — 22/22 tests, reviewed).

---

## Context

Phase 1 shipped a deterministic, continuous-time **vertical slice**: a declarative `MarketplaceSpec`,
seed-at-t0 + mid-run population arrival, a persistent per-agent lifecycle process, the
`arrive → view → transact` funnel, and a timestamped in-memory event stream. Phase 2 brings the
SimPy engine to **feature parity** with the legacy `classes.py` funnel — and does it by porting
*toward the right abstractions*, not by copying the old daily-loop code. v2.0 is the release where
the new engine fully replaces the old one in capability.

The legacy engine (`classes.py`) is the **functional reference** to port (read-only — it stays
frozen). Phase 2 reuses its behavioural maths but re-expresses scheduling in continuous time.

---

## Vision & principles — the north star (every task serves these)

1. **Generic entities.** `User`/`Listing`/`Proposal` are roles, not classifieds. No domain-specific
   assumptions bleed into the engine. Events are generic (`type` + ids), not OLX-specific.
2. **Fixed is a special case of dynamic.** Every property — including **price** — is a literal *or*
   a generator. Phase 2's pricing port is the first real test of this: a fixed price and a fitted
   price model must be the *same shape* behind one abstraction.
3. **Recursive markets.** Stay flat in v2.0, but never block nesting: no global singletons, agents
   only ever touch their `Market` via the handle passed in, so a sub-market is just another `Market`.

**Design-for-final-state rule:** a Phase 2 choice that would make Phase 3 (configurable rates,
property dependencies, calendar effects, recursive/multi-category markets) *harder* is wrong — pick
the one that leaves those doors open.

---

## Final state (what v2.0 looks like)

A user writes one `MarketplaceSpec` and runs a continuous-time market where: users arrive over time;
sellers list items priced by an **endogenous model**; buyers visit, view, lead, and bid; **proposals**
flow buyer↔seller through `Store` inboxes with **real response latency and expiry**; deals settle and
stock depletes; users **churn, go dormant, reactivate**, and listings **expire**; every actor carries
an **A/B variant**; and the whole run emits one deterministic, timestamped event stream that
reproduces the legacy funnel's behaviour — now in continuous time.

---

## Epics & tasks (each knows its neighbours)

Legend: **Reuse** = the legacy `classes.py` symbol whose *maths/logic* to port (not its scheduling).

### Epic A — Supply: sellers list mid-run  *(standalone; unblocks inventory)*
- **A1.** In a session, a user may `list` with probability `sigmoid(log(base) + slope·log(engagement))`,
  creating a `Listing` via the spec's quality/price generators; it joins the live pool.
- **A2.** Emit a `list` event; the count of listings a user makes follows an engagement-driven draw.
- **Depends on:** nothing. **Feeds:** the whole funnel (without it, inventory only depletes).
- **Reuse:** `User.list`, `Marketplace._k_listing_per_user`, `initialise` quality/price draws.
- **Final-state:** listing creation goes through the property generators (fixed-as-dynamic). `category`
  stays a spec field — do **not** hardcode `ELECTRONICS` (multi-category is Phase 3, but don't bake it shut).

### Epic B — Pricing as a dynamic property  *(the fixed-as-dynamic core)*
- **B1.** Evolve the `Property` abstraction to support **context-dependent** generators:
  `draw(rng, context=None)`, where `context` exposes market/seller state. Literal / scipy-dist /
  plain-callable paths keep working unchanged (backward compatible). This is the seam Phase 3's
  `Depends("seller.engagement", …)` and time-varying properties plug into.
- **B2.** Port the endogenous pricing as a generator-shaped strategy: `PriceModel` fits
  `LinearRegression` of price~quality over the top-k *visible* listings; `NaiveMeanPriceModel` is the
  fallback when there's nothing to learn from. Both satisfy the same `draw(rng, context)` interface.
- **B3.** Wire pricing in: sellers price new listings (A) via the model; buyers bid below ask (D) via
  the model with a bias factor.
- **Depends on:** A (something to price). **Feeds:** D (bid amount), E (settlement). **Foundational for Phase 3.**
- **Reuse:** `_pricing_research_prices`, `_pricing_determine_price`, `_pricing_set_price`,
  `NaiveMeanPriceModel`, `category_price_distributions`.
- **Final-state:** THE realization of principle #2 — price literal and price model are one shape.

### Epic C — Proposal entity + inbox substrate  *(messaging primitive)*
- **C1.** `Proposal` entity (buyer, seller, listing, amount, status); port the status lifecycle
  `created → with_seller → accepted → with_buyer → paid` plus an `expired` terminal state.
- **C2.** Per-user **inbox as a SimPy `Store`**; `send_to_seller` / `send_to_buyer` push to Stores.
  Any role can own an inbox (generic).
- **Depends on:** nothing structural (can land alongside B). **Feeds:** D, E.
- **Reuse:** `Proposal`, `send_to_seller`, `send_to_buyer`, `set_status`, `generate_proposal_id`
  (but use a **deterministic counter id**, not `hash()` — determinism guardrail).
- **Final-state:** `Store` is the continuous-time messaging substrate; no daily inbox sweeps.

### Epic D — Full buyer funnel: lead → bid  *(extends the slice funnel)*
- **D1.** Extend `_run_session`: after `view`, `make_lead` (sigmoid), then `bid` (sigmoid) → create a
  `Proposal` priced via B, `send_to_seller` via C.
- **D2.** Emit `lead` and `bid` events.
- **Depends on:** B (price), C (proposal/inbox). **Feeds:** E.
- **Reuse:** `User.make_lead`, `User.bid`.

### Epic E — Settlement processes (latency-driven)  *(why SimPy exists)*
- **E1.** Seller settlement: a seller reacts to its inbox `Store`; after a `response_time` delay it
  evaluates proposals per listing, accepts the highest bid, sends the accepted proposal back to the buyer.
- **E2.** Buyer settlement: after its own latency, the buyer pays accepted proposals → `transaction`/
  `paid`, decrements stock, flips `is_live` at zero. Also keep the direct-buy path from the slice.
- **E3.** **Proposal expiry:** a scheduled timeout moves stale proposals to `expired`.
- **Depends on:** C, D. **This is the payoff** — `response_time` finally drives behaviour; settlement is
  event-driven, not a daily batch.
- **Reuse:** `evaluate_proposals` (highest-bid-per-listing), `check_accepted_proposals`, `_pay_proposal`,
  `transact`. Replace the legacy day-batch sweep with `Store.get`-driven reactions.
- **Final-state:** continuous-time settlement; the negative-stock guard the slice review flagged lands here.

### Epic F — A/B Variant  *(independent; slot after funnel exists)*
- **F1.** Assign a `Variant` (`CONTROL/B/C/D`) at spawn via a spec-configurable assigner (default:
  all `CONTROL`; optionally weighted split). Store on `User`.
- **F2.** Stamp the variant on every emitted event so experiments are analyzable downstream.
- **Depends on:** funnel present (so variant has behaviour to tag). **Light.**
- **Reuse:** `Variant` enum.
- **Final-state:** variant stays generic; in Phase 3 it modulates configurable rates.

### Epic G — Full agent lifecycle  *(the "full lifecycle" target)*
- **G1.** Churn / dormancy: the lifecycle process can transition a user to **dormant** (stops
  scheduling sessions) via an engagement-driven hazard.
- **G2.** Reactivation: dormant users can reactivate after a drawn delay.
- **G3.** Listing expiry: listings expire after a TTL (scheduled timeout flips `is_live`, emits
  `listing_expired`).
- **Depends on:** funnel stable (do last). Independent of B/C/D internals.
- **Final-state:** explicit lifecycle states; matches the "full lifecycle" decision from the PRD.

---

## Dependency graph & build order

```
A ──┐
    ├──> D ──> E
B ──┤        (C feeds D and E)
C ──┘
F  : after D/E (independent, light)
G  : after E   (independent, lifecycle)
```

Recommended sequence: **A → B → C → D → E → F → G.** A gives inventory; B+C are the substrates D/E
need; D/E complete the funnel and are the SimPy payoff; F and G are independent finishers.

---

## Cross-cutting invariants (hold across every epic)

- **Determinism:** one seeded `np.random.default_rng`; every new stochastic path draws from it. No
  `set`/`dict`-order dependence, no `hash()`/`datetime.now()` ids — use counters (the slice's pattern).
- **Single-threaded:** `Store` + processes only. No threads, no locks, no `EventLogger` singleton.
- **Generic & flat:** entities are roles; `Market` is the only context handle agents touch; no globals
  (keeps recursive markets reachable).
- **Spec-driven:** new knobs (list rate, variant split, churn hazard, TTL, proposal-expiry window) are
  `MarketplaceSpec` fields with sensible defaults — though funnel sigmoid rates stay engine constants
  until Phase 3, per the PRD.
- **TDD + per-epic commits**, and extend `scripts/run_slice.py` so the smoke harness shows the new
  funnel stages (`list`, `lead`, `bid`, `paid`, `expired`, `churned`, `listing_expired`).

---

## Per-epic verification

- **A:** mid-run `list` events appear; live-listing count rises during a run, not just at t0.
- **B:** `Property.draw(rng)` literal/dist/callable still pass; `PriceModel` returns higher price for
  higher quality; `NaiveMeanPriceModel` fallback triggers on empty visible set; determinism holds.
- **C/D:** a `bid` creates a `with_seller` proposal in the seller's inbox `Store`.
- **E:** a proposal reaches `paid`; stock decrements and never goes negative; an un-actioned proposal
  reaches `expired`; same seed → identical event stream (incl. proposal flow).
- **F:** variant distribution matches the configured split; every event carries a variant tag.
- **G:** dormant users stop emitting; some reactivate; listings hit `listing_expired` at their TTL.
- **Whole release:** `python -m pytest` green; `python scripts/run_slice.py` shows the full funnel with
  `reproducible: True`; aggregate funnel shape sanity-checks against the legacy engine's ranges.

---

## Open questions (resolve at per-epic design time)

1. **Variant split policy** — default all-CONTROL vs configurable weighted split in v2.0.
2. **Churn model** — hazard form (constant vs engagement-driven) and whether reactivation is in v2.0 or
   a fast-follow.
3. **Proposal expiry window** — fixed spec value vs derived from `response_time`.
4. **Event payload growth** — keep the flat `Event(type, actor, entity, other)` shape and encode
   variant/amount in `other`/a payload dict? Decide before F to avoid churn.

---

## Next step

This roadmap is the **planning artifact**. Each epic then gets its own detailed TDD implementation plan
via the writing-plans skill, executed subagent-by-subagent on this branch — same loop as the slice.
Recommended: plan + build epics in the A→B→C→D→E→F→G order, reviewing between epics. v2.0 merges to
`main` only after G and the whole-release verification pass.
```
