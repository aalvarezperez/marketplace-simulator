from dataclasses import dataclass


@dataclass
class User:
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


def p_view(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(VIEW_BASE) + VIEW_SLOPE * math.log(e)))


def p_buy(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(BUY_BASE) + BUY_SLOPE * math.log(e)))


def p_list(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(LIST_BASE) + LIST_SLOPE * math.log(e)))


def p_lead(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(LEAD_BASE) + LEAD_SLOPE * math.log(e)))


def p_bid(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(BID_BASE) + BID_SLOPE * math.log(e)))


def p_churn(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(CHURN_BASE) - CHURN_SLOPE * math.log(e)))


def _decide(p, rng):
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
    yield env.timeout(float(rng.exponential(max(market.spec.reactivation_scale_days, EPS))))
    user.state = "active"
    market.emit("reactivated", actor_id=user.id)
    env.process(user_lifecycle(env, user, market, rng))


def listing_expiry(env, listing, market):
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

