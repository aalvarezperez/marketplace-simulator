from datetime import timedelta

import numpy as np
import simpy


class Clock:
    """Maps SimPy float time (in days) to calendar datetimes."""

    def __init__(self, start):
        self.start = start

    def to_datetime(self, now):
        return self.start + timedelta(days=float(now))


from sim.agents import Listing, User, user_lifecycle
from sim.agents import population_arrival
from sim.events import Event, EventRecorder


class Market:
    """Runtime state + helpers shared by all agent processes."""

    def __init__(self, env, rng, clock, recorder, spec):
        self.env = env
        self.rng = rng
        self.clock = clock
        self.recorder = recorder
        self.spec = spec
        self.users = []
        self.listings = []
        self._next_user_id = 0
        self._next_listing_id = 0

    def emit(self, event_type, actor_id=None, entity_id=None, other_id=None):
        self.recorder.record(Event(
            self.clock.to_datetime(self.env.now),
            event_type, actor_id, entity_id, other_id,
        ))

    def live_listings(self):
        return [l for l in self.listings if l.is_live]

    def match_listings(self, k):
        return sorted(self.live_listings(),
                      key=lambda l: l.quality, reverse=True)[:k]

    def add_listing(self, quality, price, seller_id):
        listing = Listing(id=self._next_listing_id, quality=float(quality),
                          price=float(price), seller_id=seller_id)
        self._next_listing_id += 1
        self.listings.append(listing)
        return listing

    def spawn_user(self):
        user = User(
            id=self._next_user_id,
            engagement=float(self.spec.engagement.draw(self.rng)),
            response_time=float(self.spec.response_time.draw(self.rng)),
        )
        self._next_user_id += 1
        self.users.append(user)
        self.emit("register", actor_id=user.id)
        self.env.process(user_lifecycle(self.env, user, self, self.rng))
        return user

    def transact(self, user, listing):
        listing.stock -= 1
        listing.transactions += 1
        if listing.stock <= 0:
            listing.is_live = False
        self.emit("transaction", actor_id=user.id,
                  entity_id=listing.id, other_id=listing.seller_id)


class Marketplace:
    """User-facing handle: build from a spec, run, read the event stream."""

    def __init__(self, market):
        self.market = market

    @classmethod
    def from_spec(cls, spec):
        env = simpy.Environment()
        rng = np.random.default_rng(spec.seed)
        market = Market(env=env, rng=rng, clock=Clock(spec.start),
                        recorder=EventRecorder(), spec=spec)
        # Seed users at t0 (each starts its own lifecycle process) + their listings.
        for _ in range(spec.n_seed_users):
            user = market.spawn_user()
            for _ in range(int(spec.listings_per_user.draw(rng))):
                market.add_listing(
                    quality=spec.listing_quality.draw(rng),
                    price=spec.listing_price.draw(rng),
                    seller_id=user.id,
                )
        env.process(population_arrival(env, market, rng))
        return cls(market)

    def run(self, until=None):
        self.market.env.run(until=self.market.spec.until if until is None else until)
        return self.events

    @property
    def events(self):
        return self.market.recorder.events
