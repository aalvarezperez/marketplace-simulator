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


# Funnel steps. Each is an Action's `run`: (agent, market, rng, session) -> None,
# reading/writing the per-session dict to hand state to later steps. They never
# return; their effects are events + mutations + (for some) session keys.

def _act_visit(agent, market, rng, session):
    """Open the session: the agent shows up. Always fires (it's the entry point)."""
    market.emit("visit", actor_id=agent.id)


def _act_list(agent, market, rng, session):
    """Seller side: with engagement-driven probability, create a priced listing."""
    if _decide(p_list(agent.engagement), rng):
        market.create_listing_for(agent, rng)


def _act_search(agent, market, rng, session):
    """The marketplace's ranking step: put the top-K live listings (by quality) into
    ``session['candidates']`` for the agent to consider."""
    session["candidates"] = market.match_listings(SESSION_K)


def _act_view(agent, market, rng, session):
    """View each candidate with probability ``p_view``; collect the seen ones into
    ``session['viewed']`` and bump per-listing view counts + emit ``view`` events."""
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
    """Form the consideration set via the market's curation strategy — a shortlist of
    what was viewed, which the buy/negotiate steps then act on."""
    session["consideration"] = market.curation(agent, session.get("viewed", []), market, rng)


def buy_action(fidelity="explicit"):
    """The buy step. The agent makes ONE rational choice: the single utility-maximizing
    listing in its consideration set (argmax of wtp - price, excluding negotiated /
    sold-out ones). explicit: buy that best iff its surplus >= 0 (emergent, no rng).
    implicit: pick the same best but gate on the p_buy(engagement) coin flip (a cheap
    stand-in). At most one purchase per session."""
    def _run(agent, market, rng, session):
        negotiated = session.get("negotiated", set())
        candidates = [l for l in session.get("consideration", [])
                      if l.is_live and l.id not in negotiated]
        if not candidates:
            return
        best = max(candidates, key=lambda l: (market.wtp(agent, l) - l.price, l.id))
        if fidelity == "implicit":
            if _decide(p_buy(agent.engagement), rng):
                market.transact(agent, best)
        elif market.wtp(agent, best) - best.price >= 0:
            market.transact(agent, best)
    return Action("buy", _run, requires=("consideration",), fidelity=fidelity)


def _act_negotiate(agent, market, rng, session):
    """The classifieds add-on: lead -> bid -> Proposal, as a branch before ``buy``.

    For each considered listing the agent may make a lead (``p_lead``) and then a bid
    (``p_bid``). A bid creates a ``Proposal`` at ``BID_BIAS * ask`` and drops it in the
    seller's inbox for latency-driven settlement. Each touched listing id is recorded
    in ``session['negotiated']`` so the later ``buy`` step skips it — the two paths to
    a sale don't double-count the same listing in one session.
    """
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
    """Build the negotiation ``Action`` to pass in ``spec.actions``.

    A branch inserted before ``buy`` (it claims listings via session state rather
    than gating buy). Opt-in: omit it for a plain visit->...->buy funnel.
    """
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
    """The base funnel every market starts with: visit -> list -> search -> view ->
    consideration -> buy, wired by ``requires``. ``buy`` is explicit (willingness vs
    price). Returned fresh each call; extras from the spec are woven in by
    ``assemble_actions``. Don't edit the base steps — add Actions instead (open/closed).
    """
    return [
        Action("visit", _act_visit),
        Action("list", _act_list, requires=("visit",)),
        Action("search", _act_search, requires=("visit",)),
        Action("view", _act_view, requires=("search",)),
        Action("consideration", _act_consideration, requires=("view",)),
        buy_action("explicit"),
    ]
