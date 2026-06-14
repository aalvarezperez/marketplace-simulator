from dataclasses import dataclass


@dataclass
class User:
    id: int
    engagement: float
    response_time: float
    inbox: object = None   # a simpy.Store, assigned at spawn time


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


@dataclass
class Proposal:
    id: int
    buyer: object
    seller: object
    listing: object
    amount: float
    status: str = "created"   # created -> with_seller -> accepted -> with_buyer -> paid (or expired)


import math

from func import sigmoid

ENGAGEMENT_TIME_UNIT = 28.0   # days; sets the engagement -> visit-rate scale
EPS = 1e-9
VIEW_BASE, VIEW_SLOPE = 0.95, 1.0
BUY_BASE, BUY_SLOPE = 0.175, 1.0
LIST_BASE, LIST_SLOPE = 0.1, 1.0
SESSION_K = 10                # listings shown per session


def p_view(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(VIEW_BASE) + VIEW_SLOPE * math.log(e)))


def p_buy(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(BUY_BASE) + BUY_SLOPE * math.log(e)))


def p_list(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(LIST_BASE) + LIST_SLOPE * math.log(e)))


def _decide(p, rng):
    return rng.random() < p


def user_lifecycle(env, user, market, rng):
    """Persistent per-agent process: wake on an engagement-driven schedule,
    run a session, repeat for the life of the simulation."""
    while True:
        scale = ENGAGEMENT_TIME_UNIT / max(user.engagement, EPS)
        yield env.timeout(float(rng.exponential(scale)))
        _run_session(user, market, rng)


def _run_session(user, market, rng):
    market.emit("visit", actor_id=user.id)
    if _decide(p_list(user.engagement), rng):
        market.create_listing_for(user, rng)
    for listing in market.match_listings(SESSION_K):
        if not listing.is_live:
            continue
        if _decide(p_view(user.engagement), rng):
            listing.views += 1
            market.emit("view", actor_id=user.id,
                        entity_id=listing.id, other_id=listing.seller_id)
            if listing.is_live and _decide(p_buy(user.engagement), rng):
                market.transact(user, listing)


def population_arrival(env, market, rng):
    """Mint new users over time at spec.arrival_rate (users per day)."""
    rate = market.spec.arrival_rate
    while rate > 0:
        yield env.timeout(float(rng.exponential(1.0 / rate)))
        market.spawn_user()

