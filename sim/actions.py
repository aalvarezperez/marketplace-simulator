from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class Action:
    """A named step an agent performs in a session.

    `run(agent, market, rng, session)` performs the decision + effect + events for
    this step. `requires` names prior actions that must have run this session before
    this one is eligible. `fidelity` is informational for now ("explicit" | "implicit");
    the emergent decision arrives in a later plan.
    """
    name: str
    run: Callable
    requires: Tuple[str, ...] = ()
    fidelity: str = "explicit"


def run_session(agent, market, rng, actions):
    """Walk the action list once. An action runs only if every name in its `requires`
    has already run this session. Returns the set of action names that ran."""
    done = set()
    session = {}
    for action in actions:
        if all(req in done for req in action.requires):
            action.run(agent, market, rng, session)
            done.add(action.name)
    return done


from sim.agents import (BID_BIAS, SESSION_K, _decide, p_bid, p_buy, p_lead,  # noqa: E402
                        p_list, p_view)


def _act_visit(agent, market, rng, session):
    market.emit("visit", actor_id=agent.id)


def _act_list(agent, market, rng, session):
    if _decide(p_list(agent.engagement), rng):
        market.create_listing_for(agent, rng)


def _act_search(agent, market, rng, session):
    session["candidates"] = market.match_listings(SESSION_K)


def _act_view(agent, market, rng, session):
    viewed = []
    for listing in session.get("candidates", []):
        if not listing.is_live:
            continue
        if _decide(p_view(agent.engagement), rng):
            listing.views += 1
            market.emit("view", actor_id=agent.id, entity_id=listing.id,
                        other_id=listing.seller_id)
            viewed.append(listing)
    session["viewed"] = viewed


def _act_consideration(agent, market, rng, session):
    session["consideration"] = list(session.get("viewed", []))


def _act_buy(agent, market, rng, session):
    for listing in session.get("consideration", []):
        if not listing.is_live:
            continue
        if _decide(p_buy(agent.engagement), rng):
            market.transact(agent, listing)
        elif _decide(p_lead(agent.engagement), rng):
            listing.leads += 1
            market.emit("lead", actor_id=agent.id, entity_id=listing.id,
                        other_id=listing.seller_id)
            if _decide(p_bid(agent.engagement), rng):
                seller = market.get_user(listing.seller_id)
                if seller is not None and seller.inbox is not None:
                    amount = BID_BIAS * listing.price
                    proposal = market.make_proposal(buyer=agent, seller=seller,
                                                    listing=listing, amount=amount)
                    market.send_to_seller(proposal)
                    listing.bids += 1
                    market.emit("bid", actor_id=agent.id, entity_id=listing.id,
                                other_id=seller.id,
                                payload={"proposal_id": proposal.id, "amount": amount})


def default_consumer_funnel():
    return [
        Action("visit", _act_visit),
        Action("list", _act_list, requires=("visit",)),
        Action("search", _act_search, requires=("visit",)),
        Action("view", _act_view, requires=("search",)),
        Action("consideration", _act_consideration, requires=("view",)),
        Action("buy", _act_buy, requires=("consideration",)),
    ]
