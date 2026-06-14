import heapq
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from itertools import compress, groupby, repeat
from typing import List

import numpy as np
from jupyter_server.auth import User
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from func import sigmoid
from logger_setup import EventLogger

ENGAGEMENT_TIME_UNIT = 28
logger = EventLogger("marketplace.log", flush_interval=5)


class Category(Enum):
    ELECTRONICS = 1
    COLLECTIBLES = 2


category_price_distributions = {
    Category.ELECTRONICS: norm(loc=500, scale=100),
    Category.COLLECTIBLES: norm(loc=100, scale=30)
}


class Listing:
    __slots__ = ('quality', 'category', 'price', 'id', 'seller', 'is_live',
                 'visits', 'views', 'leads', 'bids', 'proposals', 'transactions', 'stock')

    def __init__(self, quality: float, category: Category, price: float, seller=None, is_live=True, id=None, views=0,
                 leads=0, bids=0, proposals=0, transactions=0, stock=1):
        self.quality = quality
        self.category = category
        if price is None:
            price = category_price_distributions[self.category].rvs(size=1).round()[0]
        self.price = price
        self.id = generate_listing_id(self.quality, self.category.value, float(self.price)) if id is None else id
        self.seller = seller
        self.is_live = is_live
        self.views = views
        self.leads = leads
        self.bids = bids
        self.proposals = proposals
        self.transactions = transactions
        self.stock = stock

    # def __lt__(self, other):
    #     return self.quality < other.quality

    def __hash__(self):
        return self.id

    def to_dict(self):
        return {
            'id': self.id,
            'quality': self.quality,
            'category': self.category,
            'price': self.price,
            'views': self.views,
            'leads': self.leads,
            'bids': self.bids,
            'transactions': self.transactions
        }


class Marketplace:
    def __init__(self,
                 name: str,
                 response_heterogeneity,
                 engagement_heterogeneity
                 ):
        self.response_heterogeneity = response_heterogeneity
        self.engagement_heterogeneity = engagement_heterogeneity
        self.name = name
        self.users = None
        self.listings = None
        self.active_days = 0
        self._rng = np.random.default_rng()
        self._visitor_engagement = None

    @staticmethod
    def _k_listing_per_user(engagement):
        p = sigmoid(.3 * engagement)
        X = np.random.binomial(n=1, p=p)
        poisson_process = np.random.poisson(lam=engagement * .45)
        return poisson_process * X

    def initialise(self, n_users, listing_kwargs):
        rng = self._rng if hasattr(self, '_rng') else np.random.default_rng()
        self.users = []
        self.listings = []

        engagement = self.engagement_heterogeneity.rvs(size=n_users)
        user_listings = self._k_listing_per_user(engagement)
        response = self.response_heterogeneity.rvs(size=n_users)

        for e, r, k in zip(engagement, response, user_listings):
            user = User(engagement=e, response_time=r)
            self.users.append(user)
            num_listings = int(k)
            qualities = np.exp(np.log(500) + 0.01 * np.log(e) + rng.normal(0, 0.6, size=num_listings))
            prices = np.exp(np.log(500) + 0.01 * np.log(qualities) + rng.normal(0, 0.6, size=num_listings))
            for quality, price in zip(qualities, prices):
                self.listings.append(Listing(quality=quality, price=price, seller=user, **listing_kwargs))

        # logger.info(f"Marketplace initiated [u: {len(self.users)}, l: {len(self.listings)}]")

    def run_n_day(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be greater than zero")

        def visitor_process(visitor: User) -> None:
            visitor.check_inbox()
            # visitor.check_inbox_w_thread_lock(day)

        m, c = self, Category.ELECTRONICS

        def seller_process(seller: User, market: Marketplace = m, category: Category = c) -> None:
            seller.list(market, category)
            # seller.list_w_thread_lock(market, category, day)

        def buyer_process(buyer: User, listings: [Listing], market: Marketplace = m) -> None:
            curated_listings = buyer.curate_listings(listings)
            buyer.engage_with_listing(curated_listings, market)

        for day in range(n):
            visitors, buyers, sellers = self.open()
            listings = self.create_listing_match_set(k=100)
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(visitor_process, visitors)
                executor.map(seller_process, sellers)
                executor.map(buyer_process, buyers, repeat(listings))
            self.clean_up_gone_listings()

    def create_listing_match_set(self, k: int = 100) -> [Listing]:
        listings = self.listings
        best_k_listings = heapq.nlargest(n=k, iterable=listings, key=lambda x: x.quality)
        quality = np.array([listing.quality for listing in best_k_listings])
        p = sigmoid(np.log(.05) + np.log(1.2) * quality)
        display_mask = (np.random.binomial(n=1, p=p) == 1)
        listings = list(compress(best_k_listings, display_mask))
        return listings

    def create_visitor_set(self):
        engagement = np.array([user.engagement for user in self.users])
        self._visitor_engagement = engagement
        p = 1 - np.exp(-engagement / ENGAGEMENT_TIME_UNIT)
        visit_mask = (self._rng.binomial(n=1, p=p, size=len(p)) == 1)
        visitors = list(compress(self.users, visit_mask))
        # logger.info(f"visitor set size: {np.sum(visit_mask)}")
        return visitors

    def create_seller_set(self, visitors: [User], scale=ENGAGEMENT_TIME_UNIT * 2) -> [User]:
        if self._visitor_engagement is not None:
            engagement = self._visitor_engagement
        else:
            engagement = np.array([user.engagement for user in visitors])
        p = 1 - np.exp(-engagement * 1 / scale)
        list_mask = (self._rng.binomial(n=1, p=p, size=len(p)) == 1)
        sellers = list(compress(visitors, list_mask))
        # logger.info(f"Seller set prop: {np.mean(list_mask)}")
        return sellers

    def create_buyer_set(self, visitors: [User], scale=ENGAGEMENT_TIME_UNIT * 1.5) -> [User]:
        if self._visitor_engagement is not None:
            engagement = self._visitor_engagement
        else:
            engagement = np.array([user.engagement for user in visitors])
        p = 1 - np.exp(-engagement * 1 / scale)
        buy_mask = (self._rng.binomial(n=1, p=p, size=len(p)) == 1)
        buyers = list(compress(visitors, buy_mask))
        # logger.info(f"Buyer set prop: {np.mean(buy_mask)}")
        return buyers

    def open(self):
        self.active_days += 1
        visitors = self.create_visitor_set()
        buyers = self.create_buyer_set(visitors)
        sellers = self.create_seller_set(visitors)
        return visitors, buyers, sellers

    def reset(self):
        self.users = None
        self.listings = None

    def clean_up_gone_listings(self):
        self.listings = [listing for listing in self.listings if listing.is_live]


class RegistrationFlow:
    """factory for users"""

    @staticmethod
    def start():
        pass

    @staticmethod
    def complete(**user_kwargs):
        return User(**user_kwargs)


class SellYourItemFlow:
    """factory for listings"""

    def __init__(self):
        self.database = []

    @staticmethod
    def start():
        pass

    @staticmethod
    def complete(quality: float, category: Category, price: float) -> Listing:
        return Listing(quality, category, price)


def generate_user_id(engagement: float, response_time: float) -> int:
    return hash(datetime.timestamp(datetime.now()) + engagement + response_time)


def generate_listing_id(quality: float, price: float, category_value: int) -> int:
    return hash(datetime.timestamp(datetime.now()) + quality + price + category_value)


def generate_proposal_id(listing_id: int, buyer_id: int, seller_id: int, amount: float) -> int:
    return hash(listing_id + buyer_id + seller_id + amount)


class Variant(Enum):
    CONTROL = 0
    B = 1
    C = 2
    D = 3


class User:
    __slots__ = (
        'id', 'visits', 'engagement', 'response_time', 'listings', 'purchases', 'variant', 'inbox', 'price_model',
        '_lock', '_last_listed_day', '_last_inbox_checked_day')

    def __init__(self, engagement, response_time, inbox=None, purchases=None, listings=None, id=None, visits=0):
        self.visits: int = visits
        self.engagement: float = engagement
        self.response_time: float = response_time
        self.id: int = generate_user_id(self.engagement, self.response_time) if id is None else id
        self.listings = [] if listings is None else listings
        self.purchases = purchases if purchases is not None else []
        self.variant: Variant = Variant.CONTROL
        self.inbox: List = [] if inbox is None else inbox
        self.price_model = None
        # self._lock = threading.Lock()
        # self._last_listed_day = -1
        # self._last_inbox_checked_day = -1

    def to_dict(self):
        return {
            'id': self.id,
            'engagement': self.engagement,
            'response_time': self.response_time,
            'visits': self.visits,
            'n_listings_listings': len(self.listings),
            'n_purchases': len(self.purchases),
            'variant': self.variant
        }

    @classmethod
    def activate(cls):
        raise NotImplementedError('This method has not been implemented yet.')

    def list(self, market: Marketplace, category: Category) -> None:
        """place a listing"""
        # SellYourItemFlow.start() #  here to calculate listing conversion in future version
        user_engagement = self.engagement
        quality = np.random.lognormal(mean=user_engagement ** .2)
        price = self._pricing_set_price(quality=quality, market=market)
        listing = SellYourItemFlow.complete(quality, category, price)
        self.listings.append(listing)

    def list_w_thread_lock(self, market: Marketplace, category: Category, day: int) -> None:
        """
        Attempt to list an item for this user, but only once per `day`.
        Thread-safe with a lock to prevent multiple listings in parallel.
        """
        with self._lock:
            if self._last_listed_day == day:
                return  # Already listed today; skip
            self._last_listed_day = day

        # If we get here, we haven't listed yet today; call the `list` method
        self.list(market, category)

    def view(self, listing: Listing) -> int:
        """views item"""
        p = sigmoid(np.log(.95) + 1 * np.log(self.engagement))
        do_view = np.random.binomial(n=1, p=p)
        listing.views += do_view
        # logger.info(f"p_view: {p}; fact: {do_view}")
        return do_view

    def curate_listings(self, listings: [Listing], k: int = 10) -> [Listing]:
        listings = heapq.nlargest(n=k, iterable=listings,
                                  key=lambda x: x.quality)  # emulate limited user memory and interest
        quality = [listing.quality for listing in listings]
        p = sigmoid(np.log(.2) + 1.1 * np.log(quality))
        curated_mask = (np.random.binomial(n=1, p=p) == 1)
        listings = list(compress(listings, curated_mask))
        return listings

    def make_lead(self, listing: Listing) -> int:
        """asks seller a question via chat"""
        p = sigmoid(np.log(.35) + 1 * np.log(self.engagement))
        do_reply = np.random.binomial(n=1, p=p)
        listing.leads += do_reply
        # logger.info(f"p_lead: {p}; fact: {do_reply}")
        return do_reply

    def bid(self, listing: Listing, market: Marketplace) -> int:
        """makes a proposal"""
        p = sigmoid(np.log(.175) + 1 * np.log(self.engagement))
        do_bid = np.random.binomial(n=1, p=p)
        if do_bid == 1:
            amount = self._pricing_set_price(listing.quality, market, bias=.90)  # TODO: make bias user driven
            proposal = Proposal(amount=amount, listing=listing, buyer=self, seller=listing.seller)
            proposal.send_to_seller()
            listing.bids += 1
            # logger.info(f"p_bid: {p}; fact: {do_bid}")
        return do_bid

    def transact(self, listing: Listing):
        """buys item directly"""
        listing.transactions += 1
        listing.stock -= 1
        if listing.stock == 0:
            listing.is_live = False
        self.purchases.append(listing)

    def _pay_proposal(self, proposal):
        listing = proposal.listing
        listing.transactions += 1
        listing.stock -= 1
        if listing.stock == 0:
            listing.is_live = False
        proposal.set_status("paid")
        self.purchases.append(proposal)

    def check_accepted_proposals(self):
        if len(self.inbox) > 0:
            proposals = [proposal for proposal in self.inbox if proposal.status == "accepted"]
            for proposal in proposals:
                self._pay_proposal(proposal)

    def evaluate_proposals(self):
        if len(self.inbox) > 0:
            proposals = [proposal for proposal in self.inbox if proposal.status == "with_seller"]
            listing_proposal_groups = groupby(proposals, lambda x: x.listing)
            winning_proposals = {max(proposal) for listing, proposal in listing_proposal_groups}
            for proposal in winning_proposals:
                proposal.set_status("accepted")  # marked them accepted
                proposal.send_to_buyer()
            self.inbox = []  # reset inbox after approving winning proposals
        else:
            pass

    def check_inbox(self):
        self.evaluate_proposals()
        self.check_accepted_proposals()

    def check_inbox_w_thread_lock(self, day):
        with self._lock:
            if self._last_inbox_checked_day == day:
                return
            self._last_inbox_checked_day = day

        self.check_inbox()

    def _pricing_research_prices(self, market):
        if self.price_model is not None:
            price_model = self.price_model
        else:
            listings = market.listings
            listing_set = heapq.nlargest(n=10, iterable=listings,
                                         key=lambda x: x.quality)  # emulate user short working memory: < 10 items
            listing_set = listing_set if listing_set is not None else []
            price = np.array([listing.price for listing in listing_set])
            quality = np.array([listing.quality for listing in listing_set]).reshape(-1, 1)

            if len(listing_set) > 0:
                price_model = LinearRegression()
                price_model.fit(y=price, X=quality)
                self.price_model = price_model
            else:
                mean_price = category_price_distributions[Category.ELECTRONICS].mean()
                price_model = NaiveMeanPriceModel(mean_price)
        return price_model

    def _pricing_determine_price(self, quality: float, price_model, bias: float = 1) -> float:
        quality = np.array(quality).reshape(-1, 1)
        return bias * price_model.predict(quality)

    def _pricing_set_price(self, quality, market, bias: float = 1) -> float:
        price_model = self._pricing_research_prices(market)
        final_price = self._pricing_determine_price(quality, price_model, bias=bias)
        return final_price

    def engage_with_listing(self, listings: [Listing], market: Marketplace) -> None:
        for listing in listings:
            do_view = self.view(listing)
            if do_view == 1:
                do_reply = self.make_lead(listing)
                if do_reply == 1:
                    self.bid(listing, market)


class Proposal:
    __slots__ = ('amount', 'listing', 'buyer', 'seller', 'status', 'id')

    def __init__(self, amount: float, listing: Listing, seller: User, buyer: User, status: str = 'created',
                 id: int = None) -> None:
        self.listing = listing
        self.buyer = buyer
        self.seller = seller
        self.amount = amount
        self.status = status  # one of: created, with_seller, with_buyer
        if id is None:
            self.id: int = generate_proposal_id(
                self.listing.id,
                self.buyer.id,
                self.seller.id,
                float(self.amount)
            )
        else:
            id

    def __lt__(self, other):
        return self.amount < other.amount

    def __hash__(self):
        return self.id

    def set_status(self, status: str):
        self.status = status

    def send_to_seller(self):
        self.set_status("with_seller")
        self.seller.inbox.append(self)

    def send_to_buyer(self):
        self.set_status("with_buyer")
        self.buyer.inbox.append(self)


class NaiveMeanPriceModel:
    def __init__(self, value):
        self.value = value

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.value
