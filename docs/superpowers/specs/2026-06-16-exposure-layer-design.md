# Exposure Layer — Design Spec

**Date:** 2026-06-16
**Branch:** `experimental/simpy-replatform` (no merge to `main` until the v2.0 release is greenlit)
**Status:** approved design; next step is an implementation plan via writing-plans.
**Builds on:** the allocation engine — `docs/superpowers/specs/2026-06-16-allocation-engine-design.md` (shipped).

---

## 1. Purpose

Allocation answers *"which variant is unit U in?"* (a design-driven property, looked up via
`market.variant`, cached in the `AssignmentStore`). **Exposure** answers a different question:
*"did U actually encounter the experiment?"* — a real event, a **consequence of an action**
(reading the treated variant), not a funnel step. This spec adds the exposure layer on top of
allocation **without touching `AssignmentStore.resolve`** (allocation stays pristine).

**Default: exposure == allocation.** When an experiment is not further defined, reading a unit's
variant through the public surface (`market.variant`) both *allocates* and *exposes* it, coincident
— exposure is 1:1 with allocation. When **further defined** (`auto_expose=False`), reading the
variant allocates only; the user fires exposure deliberately at the real surface via
`market.expose(...)`, so exposure becomes a strict subset of allocation.

This is still a pure data-generation concern: the engine emits `exposure` events and keeps an
exposure ledger; it computes no statistics and defines no treatment effect.

### This mirrors how Eppo's SDKs work (the chosen model)

Eppo's SDK fuses assignment and exposure: you call `getAssignment(flag, subject, ...)` **at the
point of exposure** (where the user actually hits the treated surface), and the SDK both buckets the
subject *and* fires an `assignmentLogger` side-effect that becomes the exposure row — deduped, and
skipped when the subject isn't in an allocation. Eppo's headline guidance is precisely *"call it at
the exposure point, not eagerly"*, so logging == "actually saw it".

Our design is the same:
- `market.variant(subject, key)` is Eppo's `getAssignment` — by default it allocates **and** logs the
  exposure. **The exposure point is the call site**: you set it by placing this call where the
  treated surface is, exactly as in Eppo. There is no separate "exposure config" — placement *is* the
  configuration.
- The `EventRecorder` / `emit("exposure", ...)` is our `assignmentLogger` sink.
- `auto_expose=False` is an **escape hatch Eppo barely exposes**: read the variant *without* logging
  an exposure (allocate quietly), then fire the exposure deliberately later via `market.expose(...)`
  at the true surface. Use it only when you must peek at the variant before the exposure point.

### Timing model (resolved during brainstorming)

Allocation is **lazy** — it materializes on the first read of `market.variant` (typically from the
user's treatment-effect callable, which is the realistic exposure surface). Accepted consequence:
no read ⇒ no allocation ⇒ no exposure. By default, that same first read also emits the exposure, so
allocation and exposure coincide (same sim-time, same session) — the Eppo behavior. `market.expose`
(with `auto_expose=False`) decouples them — exposure fires later, at a chosen surface, as a subset.

---

## 2. Architecture overview

```
sim/allocation.py   + Experiment.auto_expose (bool, default True)
                    + Exposure (frozen row)
                    + AssignmentStore: _exposed set, _exposure_ledger, expose(), exposures()
                    (resolve() is UNCHANGED — pure allocation)
sim/engine.py       Market.variant routes to store.expose when the experiment auto-exposes;
                    new Market.expose(subject, exp_key, default) -> the explicit surface
sim/__init__.py     export Exposure
tests/              new tests in tests/test_allocation.py (exposure behavior)
```

No new file; exposure is tightly coupled to allocation and lives in `sim/allocation.py`. The action
funnel (`sim/actions.py`) and `run_session` are **not** touched — exposure is wired by where the
user calls `market.variant` / `market.expose`, not by a funnel step.

---

## 3. Data model (`sim/allocation.py`)

### 3.1 `Experiment.auto_expose`

Add one field to the existing `Experiment` dataclass:

```python
    auto_expose: bool = True             # True: reading the variant also exposes (exposure == allocation)
```

`auto_expose=True` (default): `market.variant` allocates and exposes coincidentally.
`auto_expose=False`: `market.variant` allocates only; exposure must be fired explicitly via
`market.expose`.

### 3.2 `Exposure` row

```python
@dataclass(frozen=True)
class Exposure:
    """An exposure event row: U actually encountered experiment E in this window."""
    experiment: str
    subject_id: object
    variant: str
    cluster: object
    window: object            # None for time-invariant designs; window index for switchback
    exposed_at: float         # sim-time of first exposure (== assigned_at when auto-exposed)
```

### 3.3 `AssignmentStore` additions

`__init__` gains, alongside the existing `_cache` / `_ledger`:

```python
        self._exposed = set()        # {(exp_key, subject_id, window)} -> exposed once per window
        self._exposure_ledger = []   # append-only list[Exposure]
```

`resolve(...)` is **unchanged** (pure allocation — cache + ledger + `assignment` event).

New `expose(...)`:

```python
    def expose(self, exp_key, subject, time, default=None):
        """Allocate (via resolve) and log an exposure once per (exp, subject, window).

        Returns the variant. Logs nothing when the unit is not actually in the
        experiment (unknown / inactive / ineligible -> resolve returned the default
        and left no cache entry). Idempotent per window: repeat calls return the
        variant but add no new Exposure row / event.
        """
        variant = self.resolve(exp_key, subject, time, default)
        exp = self._exp.get(exp_key)
        if exp is None:
            return variant
        window = exp.strategy.window(time)
        ckey = (exp_key, subject.id, window)
        if ckey not in self._cache:           # not actually allocated (inactive/ineligible)
            return variant
        if ckey not in self._exposed:
            self._exposed.add(ckey)
            a = self._cache[ckey]             # the Assignment resolve just created/returned
            ex = Exposure(a.experiment, a.subject_id, a.variant, a.cluster, a.window, time)
            self._exposure_ledger.append(ex)
            self._market.emit("exposure", actor_id=subject.id, payload={
                "experiment": a.experiment, "variant": a.variant,
                "cluster": a.cluster, "window": a.window,
            })
        return variant

    def exposures(self):
        """The full append-only exposure record (for export to a dataframe)."""
        return list(self._exposure_ledger)

    def read(self, exp_key, subject, time, default=None):
        """The default public read path: allocate, and auto-expose iff the experiment
        opts in (auto_expose). Keeps the auto-expose routing inside the store so the
        engine never touches private state."""
        exp = self._exp.get(exp_key)
        if exp is not None and exp.auto_expose:
            return self.expose(exp_key, subject, time, default)
        return self.resolve(exp_key, subject, time, default)
```

The "real bucket vs default fallback" distinction is done by checking `ckey in self._cache`
(`resolve` only writes the cache when a unit is genuinely allocated), so it is robust even if a
variant is literally named like the `default` value.

---

## 4. Market surface (`sim/engine.py`)

```python
    def variant(self, subject, exp_key, default=None):
        """Look up subject's variant for exp_key at the current sim-time. By default
        (auto_expose=True) reading also exposes the unit (exposure == allocation);
        for auto_expose=False experiments this allocates only."""
        return self.assignment_store.read(exp_key, subject, self.env.now, default)

    def expose(self, subject, exp_key, default=None):
        """Explicitly expose subject to exp_key at the current sim-time (the surface for
        auto_expose=False experiments). Allocates if needed; returns the variant."""
        return self.assignment_store.expose(exp_key, subject, self.env.now, default)
```

`market.expose` takes `subject` first, matching `market.variant`.

**Setting the exposure point.** There is no exposure-point config — like Eppo, **the point is the
call site**. You "set" exposure by placing the read where the treated surface is, and you express the
"logic" with ordinary code around it (it has full access to the listing / slot / session state in
scope, which a config field could not see).

```python
# DEFAULT (auto_expose=True) — the Eppo getAssignment pattern: read = allocate + log exposure.
# The exposure point is wherever you put this call (here, the willingness surface).
def willingness(agent, listing, market):
    base = listing.quality * agent.value_factor
    return base * 1.15 if market.variant(agent, "wtp") == "B" else base   # allocate + expose, here

# ESCAPE HATCH (auto_expose=False) — peek without logging, then expose at the true surface.
# Use only when you must read the variant before the exposure point.
spec = MarketplaceSpec(..., experiments=[
    Experiment(key="reco", variants={"CONTROL": .5, "B": .5}, auto_expose=False)])

def reco_action(agent, market, rng, session):
    for listing in session.get("consideration", []):
        if listing_in_reco_slot(listing):         # <- exposure LOGIC: arbitrary code, sees local state
            v = market.expose(agent, "reco")      # <- exposure POINT: the call site; logs once/window
            if v == "B":
                ...apply treatment...
```

Declarative alternatives (an `expose_when` predicate on the `Experiment`, or an `expose_on='action'`
funnel hook) were considered and rejected: Eppo has neither, the predicate can't see local action
state without threading it back to the call site anyway, and the action hook would touch the funnel
and mismatch per-listing granularity. The call-site primitive subsumes both and can be sugared later
if real repetition appears. See §6.

---

## 5. Semantics, determinism, back-compat

- **Once per `(exp, subject, window)`.** A unit is exposed the first time; repeat reads / `expose`
  calls return the variant but log nothing new. Switchback re-exposes on the first read in each new
  window (the cache + `_exposed` keys include the window); sticky designs expose once. (Exposure
  *frequency* counting is intentionally out of scope.)
- **Deterministic.** Exposure introduces no randomness. The `exposure` event is emitted immediately
  after the `assignment` event at the same sim-time. Same spec + same seed → byte-identical stream.
- **Back-compat.** `resolve()` is untouched, so every existing allocation test passes unchanged
  (those call `resolve` directly and/or filter `assignment` events). No-experiment and direct
  `resolve` paths emit no exposure. A run whose effect callable reads `market.variant` on an
  `auto_expose=True` experiment now *also* emits `exposure` events — purely additive; determinism
  holds. The allocation determinism sweep (cluster/switchback) still prints `identical=True`.

---

## 6. Out of scope (non-goals)

- **Treatment effects** — what a variant does to behavior remains the user's swappable callable.
- **Statistics** — estimation / bias / power / Type-I all downstream (R / notebook).
- **Exposure-frequency counting** — once-per-window only; no per-encounter exposure log.
- **Declarative exposure config** — neither an `expose_on='action'` funnel hook nor an `expose_when`
  predicate on the `Experiment`. Exposure is wired by *where you call* `market.variant`/`market.expose`
  (the Eppo model), matching the lazy "consequence of an action" model. No post-action hook
  machinery, no funnel changes. Either can be added later as sugar over the call-site primitive if a
  real need appears.
- **Eager allocation** — allocation stays lazy (materialized on first read), per the allocation spec.

---

## 7. Test plan (additions to `tests/test_allocation.py`)

- Default `auto_expose=True`: `market.variant` on an active experiment emits both an `assignment`
  and an `exposure` event at the same sim-time; the exposure ledger has the unit.
- `auto_expose=False`: `market.variant` emits an `assignment` but **no** `exposure`; a subsequent
  `market.expose` emits the `exposure` (subset behavior).
- Idempotent: two `market.expose` calls in the same window produce exactly one exposure row / event.
- Switchback: `market.expose` across two windows produces two exposure rows.
- Not-in-experiment: `market.expose` for an unknown / inactive / ineligible unit returns the default
  and logs no exposure.
- Determinism: a full run with an `auto_expose=True` experiment + a variant-reading willingness
  callable is byte-identical across two builds, and includes `exposure` events.
- Export: `store.exposures()` returns the exposure ledger; `Exposure` is importable from `sim`.

---

## 8. Open questions

None outstanding. The two confirmed during brainstorming: once-per-window exposure semantics, and
`market.expose(subject, exp_key)` argument order (subject first).
