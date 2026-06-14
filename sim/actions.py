import dataclasses
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class Action:
    """A named step an agent performs in a session.

    run(agent, market, rng, session) performs the decision + effect + events.
    `requires`: prior action names that must have run this session.
    `fidelity`: "explicit" | "implicit" (informational for now).
    `before`/`after`: name of a base action this extra hooks around.
    `mode`: "branch" (insert alongside; coordinate via session state) or
            "gate" (also add this action to the target's `requires`).
    """
    name: str
    run: Callable
    requires: Tuple[str, ...] = ()
    fidelity: str = "explicit"
    before: str = None
    after: str = None
    mode: str = "branch"


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
    negotiated = session.get("negotiated", set())
    for listing in session.get("consideration", []):
        if not listing.is_live or listing.id in negotiated:
            continue
        if _decide(p_buy(agent.engagement), rng):
            market.transact(agent, listing)


def _act_negotiate(agent, market, rng, session):
    negotiated = session.setdefault("negotiated", set())
    for listing in session.get("consideration", []):
        if not listing.is_live or listing.id in negotiated:
            continue
        if _decide(p_lead(agent.engagement), rng):
            listing.leads += 1
            market.emit("lead", actor_id=agent.id, entity_id=listing.id,
                        other_id=listing.seller_id)
            negotiated.add(listing.id)            # claim it; buy will skip it
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


def negotiate_action():
    return Action("negotiate", _act_negotiate, requires=("consideration",),
                  before="buy", mode="branch")


def assemble_actions(base, extras):
    """Return base + extras, each extra inserted at its before/after hook.
    A `gate` extra also gets appended to its target action's `requires`."""
    actions = list(base)
    for extra in extras:
        target = extra.before or extra.after
        idx = next((i for i, a in enumerate(actions) if a.name == target), None)
        if idx is None:
            raise ValueError(
                f"action {extra.name!r} hooks onto unknown action {target!r}")
        insert_at = idx if extra.before else idx + 1
        actions.insert(insert_at, extra)
        if extra.mode == "gate" and extra.before:
            tgt = actions[insert_at + 1]
            actions[insert_at + 1] = dataclasses.replace(
                tgt, requires=tuple(tgt.requires) + (extra.name,))
    return actions


def default_consumer_funnel():
    return [
        Action("visit", _act_visit),
        Action("list", _act_list, requires=("visit",)),
        Action("search", _act_search, requires=("visit",)),
        Action("view", _act_view, requires=("search",)),
        Action("consideration", _act_consideration, requires=("view",)),
        Action("buy", _act_buy, requires=("consideration",)),
    ]
