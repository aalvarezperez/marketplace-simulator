# Consideration-Set Curation — Design Spec

**Date:** 2026-06-18
**Branch:** `experimental/simpy-replatform` (no merge to `main` until the v2.0 release is greenlit)
**Status:** approved design; next step is an implementation plan via writing-plans.

---

## 1. Purpose

Today the `consideration` funnel step copies **all** viewed listings, and `buy` then purchases
**every** listing whose `wtp ≥ price`. Two changes make the buyer behave like a real one:

1. **Curation** — the consideration set becomes a **shortlist**: a swappable strategy (like
   `willingness` / `pricing`) that picks which viewed listings the agent seriously evaluates. The
   shipped strategy ranks by **quality** (a stand-in for relevance, which has no property yet) and
   keeps an **engagement-sized** number of them.
2. **Rational buy** — instead of buying everything affordable, the agent makes **one** choice:
   `argmax(wtp − price)` over the consideration set, bought iff that best surplus is ≥ 0. At most one
   purchase per session via `buy`.

So: *shortlist by attention, then pick the single best deal.* Both stay generic, deterministic, and
funnel-as-actions; the funnel structure is unchanged (the two base actions are edited in place, not
added to).

---

## 2. Curation strategy (`sim/consideration.py`)

A new module parallel to `sim/pricing.py` and `sim/willingness.py`, holding the shipped strategy.
The strategy is a callable `curation(agent, viewed, market, rng) -> list[Listing]` returning the
consideration set (ordered, capped). It is configured on the spec; **it is the default by
assignment, not by name.**

```python
import numpy as np

CONSIDERATION_MU_AT_REF = 5.0   # target shortlist size at the reference engagement
REF_ENGAGEMENT = 7.0            # the default engagement mean (gamma(a=2, scale=7/2)); anchors mu


def quality_ranked_shortlist(agent, viewed, market, rng):
    """Curate the consideration set: the top-k viewed listings by quality.

    Quality stands in for relevance (no relevance property exists yet). The shortlist
    size k is a per-session draw, Poisson with mu rising linearly in engagement, so a
    more engaged agent considers more (mu ~= CONSIDERATION_MU_AT_REF at a typical
    agent). k is capped naturally by len(viewed) (<= SESSION_K). Ties broken by
    listing id, so the result is deterministic.
    """
    if not viewed:
        return []
    mu = CONSIDERATION_MU_AT_REF * agent.engagement / REF_ENGAGEMENT
    k = int(rng.poisson(mu))
    ranked = sorted(viewed, key=lambda l: (-l.quality, l.id))   # quality desc; id breaks ties
    return ranked[:k]
```

- `mu = 5 · engagement / 7`. Disengaged agents draw ~0 (consider nothing → buy nothing); highly
  engaged agents consider everything they viewed. The `7` anchors to the **default** engagement
  distribution; if a user changes that distribution they may want to retune — the escape hatch is
  that the whole callable is swappable (set `spec.curation = your_strategy`).
- `market` is in the signature for parity with the other strategy callables and for power users; the
  shipped strategy does not read it.
- `sim/consideration.py` imports only `numpy` — self-contained, no `sim` imports (like
  `willingness.py`), so `sim/spec.py` imports it without a cycle.

---

## 3. Rational buy (`sim/actions.py` — `buy_action`)

`buy_action` changes from "buy every affordable listing" to a single rational pick over the
consideration set, excluding listings the negotiation branch already claimed.

```python
def buy_action(fidelity="explicit"):
    """The buy step. The agent picks the single utility-maximizing option from its
    consideration set (argmax of wtp - price) and buys it: explicitly iff that surplus
    is >= 0, or implicitly via the p_buy coin flip. At most one purchase per session."""
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
        else:
            if market.wtp(agent, best) - best.price >= 0:
                market.transact(agent, best)
    return Action("buy", _run, requires=("consideration",), fidelity=fidelity)
```

- `argmax(wtp − price)` with `id` as the deterministic tie-breaker.
- **Explicit:** buy `best` iff its surplus ≥ 0 (equivalently `wtp ≥ price`). If even the best deal is
  underwater, buy nothing.
- **Implicit:** still picks the same single `best`, but gates on the `p_buy` coin flip — the
  cheap fidelity stand-in, now also capped at one purchase.
- Negotiation interaction unchanged in spirit: `negotiate` (a branch before `buy`) claims listings
  into `session["negotiated"]`; `buy` excludes them, so the two paths never double-count.

---

## 4. Wiring (`_act_consideration`, `Market`, spec)

- `sim/actions.py` — `_act_consideration` delegates to the configured strategy:
  ```python
  def _act_consideration(agent, market, rng, session):
      """Form the consideration set via the market's curation strategy (a shortlist of
      what was viewed), which the buy/negotiate steps then act on."""
      session["consideration"] = market.curation(agent, session.get("viewed", []), market, rng)
  ```
- `sim/spec.py` — add field `curation: object = quality_ranked_shortlist` (import from
  `sim.consideration`), placed beside `willingness` / `pricing`.
- `sim/engine.py` — `Market.__init__` adds `self.curation = spec.curation` (beside
  `self.willingness` / `self.pricing`).
- `sim/__init__.py` — export `quality_ranked_shortlist`.

The funnel list and `run_session` are otherwise unchanged.

---

## 5. Determinism, back-compat, impact

- **Deterministic.** All ordering ties broken by `id`; the new `rng.poisson` shortlist draw comes
  from the single seeded rng. Same spec + same seed → byte-identical stream.
- **This is a real behavior change** (not purely additive):
  - `consideration` now **caps** the set (was copy-all), and adds **one `rng.poisson` draw per
    session** — so the rng stream shifts and event counts change for every run, including the
    smoke-harness default.
  - `buy` now makes **at most one** purchase per session (was buy-all-affordable) — fewer
    `transaction` events; conversion drops toward a per-session-one-decision shape.
- **Test fallout (handled in the plan):** existing tests that assert "buys multiple per session",
  exact transaction counts, or the copy-all consideration behavior will be updated to the new model.
  Determinism tests (run-to-run equality) still hold. The controlled negotiation/settlement pipeline
  test is unaffected (it drives proposals directly, not the buy argmax).

---

## 6. Out of scope (non-goals)

- **A real relevance / affinity property** — quality is the deliberate stand-in. A relevance
  property + relevance-ranked curation is a future feature; the swappable `curation` slot is where it
  plugs in.
- **Separately swappable buy choice-rule** — the `argmax(wtp − price)` rational pick is the explicit
  default, baked into `buy_action`. Making the choice rule itself pluggable is future.
- **Renaming `default_pricing` / `default_willingness`** — left as-is to avoid churning the public
  API; only the new strategy follows the name-by-behavior convention.
- **Multi-item baskets** — one purchase per session via `buy` (negotiation can still add its own
  separate path). No basket/cart modeling.

---

## 7. Test plan (`tests/`)

- `quality_ranked_shortlist`: ranks by quality desc; caps at the drawn k; ties broken by id;
  empty `viewed` → `[]`; higher engagement → larger expected shortlist (statistical, loose bound);
  deterministic for a fixed rng.
- `_act_consideration`: writes `session["consideration"]` from the strategy; respects a custom
  `spec.curation` override (e.g. a strategy returning a fixed slice).
- `buy_action` explicit: with a hand-built consideration set, buys exactly the `argmax(wtp − price)`
  and only when its surplus ≥ 0; buys nothing when all surpluses < 0; never buys a negotiated listing.
- `buy_action` implicit: picks the single best and gates on `p_buy` (use a controlled engagement /
  monkeypatched decision to make it deterministic) — at most one purchase.
- End-to-end: a full run produces `transaction` events, ≤ 1 buy-driven purchase per session on
  average is consistent with the new model, and two runs with the same seed are byte-identical.
- Determinism sweep unaffected by experiments/allocation still prints `identical=True`.
