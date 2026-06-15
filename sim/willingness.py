def default_willingness(agent, listing, market):
    """Intrinsic, sticky willingness-to-pay in currency.

    v(m) = listing.quality (quality is the item's intrinsic monetary value — the
    shared anchor that keeps WTP on the same scale as price). Scaled by this agent's
    own value_factor. Deliberately ignores the live market price, so agents can be
    priced out when prices surge. Override on the spec for non-linear / richer forms.
    """
    return listing.quality * agent.value_factor
