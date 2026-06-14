class Property:
    """Fixed-as-dynamic: a literal, a scipy frozen distribution (has .rvs),
    or a callable(rng). A literal is the degenerate generator."""

    def __init__(self, value):
        self.value = value

    def draw(self, rng):
        v = self.value
        if hasattr(v, "rvs"):
            return v.rvs(random_state=rng)
        if callable(v):
            return v(rng)
        return v


from dataclasses import dataclass, field
from datetime import datetime

from scipy.stats import gamma, lognorm, norm, poisson


def _as_property(v):
    return v if isinstance(v, Property) else Property(v)


@dataclass
class MarketplaceSpec:
    start: datetime
    seed: int = 0
    n_seed_users: int = 1000
    until: float = 7.0            # sim-days to run
    arrival_rate: float = 5.0     # new users per day (population arrival)
    engagement: Property = field(
        default_factory=lambda: Property(gamma(a=2, scale=7 / 2)))
    response_time: Property = field(
        default_factory=lambda: Property(gamma(a=2, scale=1 / 2)))
    listings_per_user: Property = field(
        default_factory=lambda: Property(poisson(mu=0.6)))
    listing_quality: Property = field(
        default_factory=lambda: Property(lognorm(s=0.6, scale=500)))
    listing_price: Property = field(
        default_factory=lambda: Property(norm(loc=500, scale=100)))

    def __post_init__(self):
        self.engagement = _as_property(self.engagement)
        self.response_time = _as_property(self.response_time)
        self.listings_per_user = _as_property(self.listings_per_user)
        self.listing_quality = _as_property(self.listing_quality)
        self.listing_price = _as_property(self.listing_price)
