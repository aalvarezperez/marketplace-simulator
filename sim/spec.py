class Property:
    """Fixed-as-dynamic: a literal, a scipy frozen distribution (has .rvs),
    or a callable(rng). A literal is the degenerate generator."""

    def __init__(self, value):
        self.value = value

    def draw(self, rng, context=None):
        """Sample one value from the wrapped generator using ``rng``.

        Dispatch by capability, most specific first: a context-model (has
        ``draw_with_context``) lets the value depend on live state; a scipy frozen
        dist (has ``rvs``) is sampled with ``rng`` as its ``random_state``; a plain
        callable is called as ``v(rng)``; anything else is a literal returned as-is.
        Always drawing from the passed ``rng`` is what keeps runs deterministic.
        """
        v = self.value
        if hasattr(v, "draw_with_context"):
            return v.draw_with_context(rng, context)
        if hasattr(v, "rvs"):
            return v.rvs(random_state=rng)
        if callable(v):
            return v(rng)
        return v


from dataclasses import dataclass, field
from datetime import datetime

from scipy.stats import gamma, lognorm, norm, poisson

from sim.pricing import default_pricing
from sim.willingness import default_willingness
from sim.consideration import quality_ranked_shortlist
from sim.allocation import Experiment, SimpleRandomization


def _as_property(v):
    """Coerce a raw spec value into a ``Property`` (idempotent for one already)."""
    return v if isinstance(v, Property) else Property(v)


@dataclass
class MarketplaceSpec:
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
    start: datetime
    seed: int = 0
    n_seed_users: int = 1000
    until: float = 7.0            # sim-days to run
    arrival_rate: float = 5.0     # new users per day (population arrival)
    proposal_expiry_days: float = 3.0
    reactivation_scale_days: float = 30.0
    listing_ttl_days: float = 30.0      # set to None to disable listing expiry
    variant_weights: dict = field(default_factory=lambda: {"CONTROL": 1.0})
    actions: list = field(default_factory=list)
    engagement: Property = field(
        default_factory=lambda: Property(gamma(a=2, scale=7 / 2)))
    response_time: Property = field(
        default_factory=lambda: Property(gamma(a=2, scale=1 / 2)))
    listings_per_user: Property = field(
        default_factory=lambda: Property(poisson(mu=0.6)))
    listing_quality: Property = field(
        default_factory=lambda: Property(lognorm(s=0.6, scale=500)))
    value_factor: Property = field(
        default_factory=lambda: Property(lognorm(s=0.3, scale=1.0)))
    willingness: object = default_willingness
    pricing: object = default_pricing
    curation: object = quality_ranked_shortlist
    seller_patience: Property = None    # days unsold before a markdown; default set in __post_init__
    markdown_pct: float = 0.1
    experiments: list = field(default_factory=list)     # allocation Experiment registry
    cluster: Property = field(default_factory=lambda: Property(0))   # generic cluster key per agent

    def __post_init__(self):
        """Wrap the disposition fields in ``Property`` so users can pass a bare
        literal / scipy dist / callable. ``seller_patience`` has no fixed default:
        it is anchored to the run length (mean = 20% of ``until``) so markdowns
        actually fire within a run, then coerced like the rest.
        """
        self.engagement = _as_property(self.engagement)
        self.response_time = _as_property(self.response_time)
        self.listings_per_user = _as_property(self.listings_per_user)
        self.listing_quality = _as_property(self.listing_quality)
        self.value_factor = _as_property(self.value_factor)
        if self.seller_patience is None:
            self.seller_patience = Property(norm(loc=self.until * 0.2, scale=self.until * 0.1))
        else:
            self.seller_patience = _as_property(self.seller_patience)
        self.cluster = _as_property(self.cluster)
        # Back-compat: a real variant_weights split with no explicit experiments
        # synthesizes one simple-randomization experiment named "default".
        # When experiments are given explicitly, variant_weights is ignored.
        if not self.experiments and len(self.variant_weights) > 1:
            self.experiments = [Experiment(key="default",
                                           variants=dict(self.variant_weights),
                                           strategy=SimpleRandomization())]
