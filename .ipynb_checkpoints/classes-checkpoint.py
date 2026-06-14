import math
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Market:
    def __init__(self,
                 engagement_heterogeneity,
                 engagement_heterogeneity_arg_dict,
                 listing_quality,
                 listing_quality_arg_dict):
        self.engagement_heterogeneity = engagement_heterogeneity
        self.engagement_heterogeneity_arg_dict = engagement_heterogeneity_arg_dict
        self.listing_quality = listing_quality
        self.listing_quality_arg_dict = listing_quality_arg_dict
        self.users = None
        self.listings = None

    def init(self, n_users, n_listings):
        self.users = [User(engagement=self.engagement_heterogeneity(**self.engagement_heterogeneity_arg_dict))
                      for _ in range(n_users)]

        self.listings = [Listing(quality=self.listing_quality(**self.listing_quality_arg_dict))
                         for _ in range(n_listings)]

    def display_listing(self):
        quality = [listing.quality for listing in self.listings]
        p = [q / sum(quality) for q in quality]
        return np.random.choice(self.listings, size=1, p=p)

    def remove_listing(self, listing):
        self.listings.pop(listing)

    def reset(self):
        self.users = None
        self.listings = None


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
    def complete(listing_quality):
        return Listing(listing_quality)


def generate_id():
    return datetime.timestamp(datetime.now()) * 1000


@dataclass
class Listing:
    quality: float
    id: int = None
    is_live: bool = True
    views: int = 0
    leads: int = 0
    proposals: int = 0
    transactions: int = 0
    stock: int = 1

    def __post_init__(self):
        self.id = generate_id()


class ListingSet:
    """A generic container for listings"""

    def __init__(self, content: List[Listing] = None):
        self.content = content or []

    def add(self, item: Listing):
        """adds an item to content"""
        self.content.append(item)


class Inventory(ListingSet):
    """contains seller's listings"""
    pass


class Purchases(ListingSet):
    """contains buyer's purchases"""
    pass


@dataclass
class User:
    id: int = None
    visits: int = 0
    engagement: float = None
    listings = None
    purchases = None
    variant: bool = None

    def __post_init__(self):
        self.id = generate_id()
        self.listings = []
        self.purchases = []

    def activate(self):
        raise NotImplementedError('This method has not been implemented yet.')

    def place_listing(self, listing_quality):
        """place a listing"""

        SellYourItemFlow.start()
        listing = SellYourItemFlow.complete(listing_quality)
        self.listings.append(listing)

        return listing

    def reply(self, listing: Listing):
        """asks seller a question via chat"""
        listing.leads += 1

    def buy(self, listing: Listing):
        listing.transactions += 1
        listing.stock -= 1
        if listing.stock == 0:
            listing.is_live = False
        """buys item directly"""
        self.purchases.append(listing)

    def view(self, listing: Listing):
        """buys item directly"""
        listing.views += 1
