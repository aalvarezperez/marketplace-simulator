from dataclasses import dataclass


@dataclass
class User:
    """A marketplace participant — generic, so the same object is buyer and seller.

    Heterogeneity lives in the per-agent draws: ``engagement`` drives every implicit
    funnel rate and the wake-up cadence; ``response_time`` is the latency (days)
    before this user reacts to its inbox; ``value_factor`` scales willingness-to-pay;
    ``patience`` is the days-unsold a listing waits before this seller marks it down.
    ``inbox`` is a ``simpy.Store`` (assigned at spawn). ``state`` toggles
    active/dormant across churn and reactivation. ``variant`` is the A/B tag.
    """
    id: int
    engagement: float
    response_time: float
    inbox: object = None   # a simpy.Store, assigned at spawn time
    variant: str = "CONTROL"
    state: str = "active"      # active | dormant
    value_factor: float = 1.0
    patience: float = 0.0


@dataclass
class Listing:
    """An item for sale. ``quality`` is its intrinsic monetary value — the shared
    anchor both pricing and willingness-to-pay are denominated in. ``price`` is the
    current ask (mutated down by markdowns). ``is_live`` flips false when stock hits
    zero or the TTL expires, which removes it from the comparable/match set. The
    ``views/leads/bids/transactions`` counters accumulate funnel activity per listing.
    """
    id: int
    quality: float
    price: float
    seller_id: int
    stock: int = 1
    is_live: bool = True
    views: int = 0
    transactions: int = 0
    leads: int = 0
    bids: int = 0


@dataclass
class Proposal:
    """A bid in flight between a buyer and a seller (the negotiation add-on).

    ``amount`` is the offered price (below the ask). ``status`` walks a pipeline:
    ``created -> with_seller`` (in seller inbox) ``-> accepted -> with_buyer`` (routed
    back) ``-> paid``. Terminal off-ramps: ``rejected``/``lost`` (listing died before
    each side acted) and ``expired`` (neither side acted within the expiry window).
    """
    id: int
    buyer: object
    seller: object
    listing: object
    amount: float
    status: str = "created"   # created -> with_seller -> accepted -> with_buyer -> paid (or expired)


import math

from sim.func import sigmoid

ENGAGEMENT_TIME_UNIT = 28.0   # days; sets the engagement -> visit-rate scale
EPS = 1e-9
MIN_PATIENCE = 0.25
VIEW_BASE, VIEW_SLOPE = 0.95, 1.0
BUY_BASE, BUY_SLOPE = 0.175, 1.0
LIST_BASE, LIST_SLOPE = 0.1, 1.0
LEAD_BASE, LEAD_SLOPE = 0.35, 1.0
BID_BASE, BID_SLOPE = 0.175, 1.0
BID_BIAS = 0.9               # buyers bid below the ask
SESSION_K = 10                # listings shown per session
CHURN_BASE, CHURN_SLOPE = 0.1, 1.0


# --- Implicit-fidelity rates -------------------------------------------------
# Each is the engagement-driven probability of a funnel step we chose NOT to model
# explicitly: sigmoid(log(base) + slope * log(engagement)). These are stand-ins, not
# configured truths — model a step explicitly (e.g. buy via willingness >= price)
# when your experiment perturbs it, and leave the rest as these cheap coin-flips.

def p_view(engagement):
    """P(view a shown listing) — rises with engagement."""
    e = max(engagement, EPS)
    return float(sigmoid(math.log(VIEW_BASE) + VIEW_SLOPE * math.log(e)))


def p_buy(engagement):
    """P(direct buy) — the implicit alternative to explicit willingness-vs-price.

    Only used when ``buy_action`` is built with ``fidelity="implicit"``; the default
    funnel buys explicitly, so emergent conversion ignores this.
    """
    e = max(engagement, EPS)
    return float(sigmoid(math.log(BUY_BASE) + BUY_SLOPE * math.log(e)))


def p_list(engagement):
    """P(a visiting user lists an item this session)."""
    e = max(engagement, EPS)
    return float(sigmoid(math.log(LIST_BASE) + LIST_SLOPE * math.log(e)))


def p_lead(engagement):
    """P(contact a seller about a considered listing) — the lead step of negotiation."""
    e = max(engagement, EPS)
    return float(sigmoid(math.log(LEAD_BASE) + LEAD_SLOPE * math.log(e)))


def p_bid(engagement):
    """P(turn a lead into a bid), conditional on having made the lead."""
    e = max(engagement, EPS)
    return float(sigmoid(math.log(BID_BASE) + BID_SLOPE * math.log(e)))


def p_churn(engagement):
    """P(go dormant after a session) — note the MINUS slope: high engagement churns less."""
    e = max(engagement, EPS)
    return float(sigmoid(math.log(CHURN_BASE) - CHURN_SLOPE * math.log(e)))


def _decide(p, rng):
    """One Bernoulli trial: True with probability ``p``, drawn from ``rng``."""
    return rng.random() < p


def user_lifecycle(env, user, market, rng):
    """Persistent per-agent process: wake on an engagement-driven schedule,
    run a session, repeat until churn or sim end."""
    while user.state == "active":
        scale = ENGAGEMENT_TIME_UNIT / max(user.engagement, EPS)
        yield env.timeout(float(rng.exponential(scale)))
        if user.state != "active":
            break
        market.run_session(user, rng)
        if _decide(p_churn(user.engagement), rng):
            market.churn_user(user)
            break


def population_arrival(env, market, rng):
    """Mint new users over time at spec.arrival_rate (users per day)."""
    rate = market.spec.arrival_rate
    while rate > 0:
        yield env.timeout(float(rng.exponential(1.0 / rate)))
        market.spawn_user()


def reactivation(env, user, market, rng):
    """Bring a dormant user back after an exponential delay, then restart its
    lifecycle. Scheduled by ``churn_user``; mean delay is ``reactivation_scale_days``.
    """
    yield env.timeout(float(rng.exponential(max(market.spec.reactivation_scale_days, EPS))))
    user.state = "active"
    market.emit("reactivated", actor_id=user.id)
    env.process(user_lifecycle(env, user, market, rng))


def listing_expiry(env, listing, market):
    """Take a listing off the market at its TTL (``listing_ttl_days``) if still
    unsold. A no-op once it has already sold out. Skipped entirely when the spec
    sets ``listing_ttl_days = None``.
    """
    yield env.timeout(max(market.spec.listing_ttl_days, EPS))
    if listing.is_live:
        listing.is_live = False
        market.emit("listing_expired", actor_id=listing.seller_id, entity_id=listing.id)


def markdown_listing(env, listing, market, patience):
    """Seller-driven liquidity correction: while unsold, drop the price by
    market.markdown_pct every `patience` days. Stops on sale/expiry. The up-move
    is emergent (cleared cheap stock raises the comparable median)."""
    while listing.is_live:
        yield env.timeout(max(patience, MIN_PATIENCE))
        if listing.is_live:
            listing.price *= (1.0 - market.markdown_pct)
            market.emit("markdown", actor_id=listing.seller_id, entity_id=listing.id,
                        payload={"price": listing.price})


def settlement_process(env, user, market, rng):
    """React to this user's inbox: as seller, accept incoming bids after a
    response_time latency; as buyer, pay accepted proposals after a latency."""
    while True:
        proposal = yield user.inbox.get()
        if proposal.status == "with_seller":
            yield env.timeout(max(user.response_time, EPS))
            market.evaluate_proposal(proposal)
        elif proposal.status == "with_buyer":
            yield env.timeout(max(user.response_time, EPS))
            market.settle_proposal(proposal)
        # any other status (e.g. expired): drop it


def proposal_expiry(env, proposal, market):
    """Move a proposal to 'expired' if it hasn't terminated by the expiry window."""
    yield env.timeout(max(market.spec.proposal_expiry_days, EPS))
    if proposal.status in ("created", "with_seller", "accepted", "with_buyer"):
        proposal.status = "expired"
        market.emit("proposal_expired", actor_id=proposal.buyer.id,
                    entity_id=proposal.listing.id, other_id=proposal.seller.id,
                    payload={"proposal_id": proposal.id})

