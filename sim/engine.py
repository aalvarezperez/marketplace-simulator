from datetime import timedelta

import numpy as np
import simpy


class Clock:
    """Maps SimPy float time (in days) to calendar datetimes."""

    def __init__(self, start):
        self.start = start

    def to_datetime(self, now):
        """Convert a SimPy time (``env.now``, in days) to a calendar datetime."""
        return self.start + timedelta(days=float(now))


from sim.agents import Listing, Proposal, User, user_lifecycle, MIN_PATIENCE
from sim.agents import population_arrival, settlement_process, proposal_expiry
from sim.agents import reactivation, listing_expiry, markdown_listing
from sim.events import Event, EventRecorder
from sim.actions import assemble_actions, default_consumer_funnel, run_session as _run_session_actions
from sim.allocation import AssignmentStore


class Market:
    """Runtime state + helpers shared by all agent processes."""

    def __init__(self, env, rng, clock, recorder, spec):
        self.env = env
        self.rng = rng
        self.clock = clock
        self.recorder = recorder
        self.spec = spec
        self.actions = assemble_actions(default_consumer_funnel(), spec.actions)
        self.willingness = spec.willingness
        self.pricing = spec.pricing
        self.markdown_pct = spec.markdown_pct
        self.experiments = spec.experiments
        self.assignment_store = AssignmentStore(spec.experiments, self)
        self.users = []
        self.users_by_id = {}
        self.listings = []
        self._next_user_id = 0
        self._next_listing_id = 0
        self._next_proposal_id = 0

    def run_session(self, user, rng):
        """Run the assembled action funnel once for ``user`` (called by the lifecycle)."""
        return _run_session_actions(user, self, rng, self.actions)

    def variant(self, subject, exp_key, default=None):
        """Look up ``subject``'s variant for ``exp_key`` at the current sim-time.
        By default (auto_expose=True) reading also logs an exposure (the Eppo
        getAssignment model); for auto_expose=False experiments this allocates only.
        Returns ``default`` when the experiment is unknown, inactive, or ineligible."""
        return self.assignment_store.read(exp_key, subject, self.env.now, default)

    def expose(self, subject, exp_key, default=None):
        """Explicitly expose ``subject`` to ``exp_key`` at the current sim-time — the
        surface for auto_expose=False experiments. Allocates if needed, logs the
        exposure once per (exp, subject, window), and returns the variant."""
        return self.assignment_store.expose(exp_key, subject, self.env.now, default)

    def emit(self, event_type, actor_id=None, entity_id=None, other_id=None, payload=None):
        """Record an ``Event`` stamped with the current calendar time. Behavioral
        events are lean; the ``assignment`` event (a projection of the
        AssignmentStore) is the only variant-bearing log."""
        self.recorder.record(Event(
            self.clock.to_datetime(self.env.now),
            event_type, actor_id, entity_id, other_id, payload,
        ))

    def live_listings(self):
        """All listings still on the market (in stock and not expired)."""
        return [l for l in self.listings if l.is_live]

    def match_listings(self, k):
        """The market's ranking/match step: the top-``k`` live listings by quality.
        This is what a searching agent is shown (the candidate set)."""
        return sorted(self.live_listings(),
                      key=lambda l: l.quality, reverse=True)[:k]

    def wtp(self, agent, listing):
        """This agent's willingness-to-pay for this listing, via the spec's
        ``willingness`` callable. The buy decision compares it against the ask."""
        return self.willingness(agent, listing, self)

    def add_listing(self, quality, price, seller_id):
        """Register a listing and start its background processes: TTL expiry (unless
        disabled) and, for a real seller when ``markdown_pct > 0``, the per-seller
        liquidity markdown clocked on that seller's ``patience``."""
        listing = Listing(id=self._next_listing_id, quality=float(quality),
                          price=float(price), seller_id=seller_id)
        self._next_listing_id += 1
        self.listings.append(listing)
        if self.spec.listing_ttl_days is not None:
            self.env.process(listing_expiry(self.env, listing, self))
        if self.markdown_pct > 0:
            seller = self.get_user(seller_id)
            if seller is not None:
                self.env.process(markdown_listing(self.env, listing, self, seller.patience))
        return listing

    def create_listing_for(self, user, rng):
        """Mid-run listing creation (the ``list`` funnel step): draw a quality, price
        it via the spec's ``pricing`` callable, register it, emit ``list``."""
        quality = self.spec.listing_quality.draw(rng)
        price = self.pricing(user, quality, self, rng)
        listing = self.add_listing(quality=quality, price=price, seller_id=user.id)
        self.emit("list", actor_id=user.id, entity_id=listing.id)
        return listing

    def spawn_user(self):
        """Mint a user: draw its dispositions (engagement, response_time, value_factor,
        patience), give it an inbox + cluster key, register it, and kick off its two
        long-lived processes — the session lifecycle and the inbox settlement loop.
        Used both for the t0 seed population and for mid-run arrivals."""
        user = User(
            id=self._next_user_id,
            engagement=float(self.spec.engagement.draw(self.rng)),
            response_time=float(self.spec.response_time.draw(self.rng)),
        )
        self._next_user_id += 1
        user.value_factor = float(self.spec.value_factor.draw(self.rng))
        user.patience = max(float(self.spec.seller_patience.draw(self.rng)), MIN_PATIENCE)
        user.inbox = simpy.Store(self.env)
        user.cluster = self.spec.cluster.draw(self.rng)
        self.users.append(user)
        self.users_by_id[user.id] = user
        self.emit("register", actor_id=user.id)
        self.env.process(user_lifecycle(self.env, user, self, self.rng))
        self.env.process(settlement_process(self.env, user, self, self.rng))
        return user

    def make_proposal(self, buyer, seller, listing, amount):
        """Create a ``Proposal`` (counter id) and arm its expiry timer. Doesn't route
        it anywhere yet — the caller follows with ``send_to_seller``."""
        proposal = Proposal(id=self._next_proposal_id, buyer=buyer, seller=seller,
                            listing=listing, amount=float(amount))
        self._next_proposal_id += 1
        self.env.process(proposal_expiry(self.env, proposal, self))
        return proposal

    def send_to_seller(self, proposal):
        """Route a proposal into the seller's inbox (status -> ``with_seller``)."""
        proposal.status = "with_seller"
        proposal.seller.inbox.put(proposal)

    def get_user(self, user_id):
        """Look up a user by id, or ``None`` if unknown."""
        return self.users_by_id.get(user_id)

    def churn_user(self, user):
        """Make a user dormant (stops scheduling sessions) and schedule its eventual
        reactivation. Emits ``churned``."""
        user.state = "dormant"
        self.emit("churned", actor_id=user.id)
        self.env.process(reactivation(self.env, user, self, self.rng))

    def send_to_buyer(self, proposal):
        """Route an accepted proposal back to the buyer's inbox (status -> ``with_buyer``)."""
        proposal.status = "with_buyer"
        proposal.buyer.inbox.put(proposal)

    def transact(self, user, listing):
        """Complete a direct buy: decrement stock (delisting at zero), bump the
        listing's transaction count, emit ``transaction``."""
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
        """Build a ready-to-run marketplace from a ``MarketplaceSpec``.

        Creates the SimPy environment and the single seeded RNG, spawns the t0 seed
        population (each with its drawn listings), and arms the population-arrival
        process. Nothing advances until ``run()``. Same spec + same ``seed`` ->
        byte-identical run.
        """
        env = simpy.Environment()
        rng = np.random.default_rng(spec.seed)
        market = Market(env=env, rng=rng, clock=Clock(spec.start),
                        recorder=EventRecorder(), spec=spec)
        # Seed users at t0 (each starts its own lifecycle process) + their listings.
        for _ in range(spec.n_seed_users):
            user = market.spawn_user()
            for _ in range(int(spec.listings_per_user.draw(rng))):
                quality = spec.listing_quality.draw(rng)
                price = market.pricing(user, quality, market, rng)
                market.add_listing(quality=quality, price=price, seller_id=user.id)
        env.process(population_arrival(env, market, rng))
        return cls(market)

    def run(self, until=None):
        """Advance the simulation to ``until`` days (default ``spec.until``) and return
        the event stream. Re-callable with a larger ``until`` to continue a run."""
        self.market.env.run(until=self.market.spec.until if until is None else until)
        return self.events

    @property
    def events(self):
        """The recorded event stream so far (list of ``Event``, in occurrence order)."""
        return self.market.recorder.events

    def write_jsonl(self, path):
        """Dump the event stream to JSON lines."""
        self.market.recorder.write_jsonl(path)

    def summary(self):
        """Event-type counts for the run, e.g. {'visit': 1816, 'view': 2351, ...}."""
        from collections import Counter
        return dict(Counter(e.event_type for e in self.events))
