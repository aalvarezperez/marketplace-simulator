from datetime import timedelta

import numpy as np
import simpy


class Clock:
    """Maps SimPy float time (in days) to calendar datetimes."""

    def __init__(self, start):
        self.start = start

    def to_datetime(self, now):
        return self.start + timedelta(days=float(now))


from sim.agents import Listing, Proposal, User, user_lifecycle
from sim.agents import population_arrival, settlement_process, proposal_expiry
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
        self.users_by_id = {}
        self.listings = []
        self._next_user_id = 0
        self._next_listing_id = 0
        self._next_proposal_id = 0

    def _assign_variant(self, rng):
        weights = self.spec.variant_weights
        names = list(weights.keys())
        if len(names) == 1:           # no real split -> no rng draw (keeps default runs unchanged)
            return names[0]
        w = np.array([weights[n] for n in names], dtype=float)
        w = w / w.sum()
        return names[int(rng.choice(len(names), p=w))]

    def emit(self, event_type, actor_id=None, entity_id=None, other_id=None, payload=None):
        if actor_id is not None:
            actor = self.users_by_id.get(actor_id)
            if actor is not None:
                if payload is None:
                    payload = {"variant": actor.variant}
                elif "variant" not in payload:
                    payload = {**payload, "variant": actor.variant}
        self.recorder.record(Event(
            self.clock.to_datetime(self.env.now),
            event_type, actor_id, entity_id, other_id, payload,
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

    def create_listing_for(self, user, rng):
        quality = self.spec.listing_quality.draw(rng)
        price = self.spec.listing_price.draw(
            rng, context={"market": self, "quality": quality, "seller": user})
        listing = self.add_listing(quality=quality, price=price, seller_id=user.id)
        self.emit("list", actor_id=user.id, entity_id=listing.id)
        return listing

    def spawn_user(self):
        user = User(
            id=self._next_user_id,
            engagement=float(self.spec.engagement.draw(self.rng)),
            response_time=float(self.spec.response_time.draw(self.rng)),
        )
        self._next_user_id += 1
        user.inbox = simpy.Store(self.env)
        user.variant = self._assign_variant(self.rng)
        self.users.append(user)
        self.users_by_id[user.id] = user
        self.emit("register", actor_id=user.id)
        self.env.process(user_lifecycle(self.env, user, self, self.rng))
        self.env.process(settlement_process(self.env, user, self, self.rng))
        return user

    def make_proposal(self, buyer, seller, listing, amount):
        proposal = Proposal(id=self._next_proposal_id, buyer=buyer, seller=seller,
                            listing=listing, amount=float(amount))
        self._next_proposal_id += 1
        self.env.process(proposal_expiry(self.env, proposal, self))
        return proposal

    def send_to_seller(self, proposal):
        proposal.status = "with_seller"
        proposal.seller.inbox.put(proposal)

    def get_user(self, user_id):
        return self.users_by_id.get(user_id)

    def send_to_buyer(self, proposal):
        proposal.status = "with_buyer"
        proposal.buyer.inbox.put(proposal)

    def transact(self, user, listing):
        listing.stock -= 1
        listing.transactions += 1
        if listing.stock <= 0:
            listing.is_live = False
        self.emit("transaction", actor_id=user.id,
                  entity_id=listing.id, other_id=listing.seller_id)

    def evaluate_proposal(self, proposal):
        """Seller side: accept if the listing is still live, else reject."""
        if proposal.status != "with_seller":
            return
        listing = proposal.listing
        if listing.is_live:
            self.emit("accepted", actor_id=proposal.seller.id, entity_id=listing.id,
                      other_id=proposal.buyer.id,
                      payload={"proposal_id": proposal.id, "amount": proposal.amount})
            self.send_to_buyer(proposal)   # sets status with_buyer, routes to buyer inbox
        else:
            proposal.status = "rejected"
            self.emit("proposal_rejected", actor_id=proposal.seller.id,
                      entity_id=listing.id, other_id=proposal.buyer.id,
                      payload={"proposal_id": proposal.id})

    def settle_proposal(self, proposal):
        """Buyer side: pay if the listing is still live, else the deal is lost."""
        if proposal.status != "with_buyer":
            return
        listing = proposal.listing
        if listing.is_live:
            listing.stock -= 1
            listing.transactions += 1
            if listing.stock <= 0:
                listing.is_live = False
            proposal.status = "paid"
            self.emit("paid", actor_id=proposal.buyer.id, entity_id=listing.id,
                      other_id=proposal.seller.id,
                      payload={"proposal_id": proposal.id, "amount": proposal.amount})
        else:
            proposal.status = "lost"
            self.emit("proposal_lost", actor_id=proposal.buyer.id, entity_id=listing.id,
                      other_id=proposal.seller.id, payload={"proposal_id": proposal.id})


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
                quality = spec.listing_quality.draw(rng)
                price = spec.listing_price.draw(
                    rng, context={"market": market, "quality": quality})
                market.add_listing(quality=quality, price=price, seller_id=user.id)
        env.process(population_arrival(env, market, rng))
        return cls(market)

    def run(self, until=None):
        self.market.env.run(until=self.market.spec.until if until is None else until)
        return self.events

    @property
    def events(self):
        return self.market.recorder.events
