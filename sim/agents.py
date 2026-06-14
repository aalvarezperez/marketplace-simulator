from dataclasses import dataclass


@dataclass
class User:
    id: int
    engagement: float
    response_time: float


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


import math

from func import sigmoid

ENGAGEMENT_TIME_UNIT = 28.0   # days; sets the engagement -> visit-rate scale
EPS = 1e-9
VIEW_BASE, VIEW_SLOPE = 0.95, 1.0
BUY_BASE, BUY_SLOPE = 0.175, 1.0
SESSION_K = 10                # listings shown per session


def p_view(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(VIEW_BASE) + VIEW_SLOPE * math.log(e)))


def p_buy(engagement):
    e = max(engagement, EPS)
    return float(sigmoid(math.log(BUY_BASE) + BUY_SLOPE * math.log(e)))


def _decide(p, rng):
    return rng.random() < p


# TEMPORARY stub — replaced in Task 8
def user_lifecycle(env, user, market, rng):
    yield env.timeout(0)

