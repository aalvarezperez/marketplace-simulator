"""Consideration-set curation: which viewed listings the agent seriously evaluates.

The shipped strategy ranks by quality (a stand-in for relevance, which has no
property yet) and keeps an engagement-sized shortlist. It is a swappable callable
on the spec (``curation``); it is the default by assignment, not by name. Imports
only numpy so sim.spec can import it without a cycle.
"""
import numpy as np

CONSIDERATION_MU_AT_REF = 5.0   # target shortlist size at the reference engagement
REF_ENGAGEMENT = 7.0            # the default engagement mean (gamma(a=2, scale=7/2)); anchors mu


def quality_ranked_shortlist(agent, viewed, market, rng):
    """Curate the consideration set: the top-k of ``viewed`` by quality.

    k is a per-session draw, Poisson with mu rising linearly in engagement
    (mu = CONSIDERATION_MU_AT_REF * engagement / REF_ENGAGEMENT), so a more engaged
    agent considers more; k is capped naturally by len(viewed). Ties broken by
    listing id, so the result is deterministic. ``market`` is unused here (present
    for parity with the other strategy callables / power users).
    """
    if not viewed:
        return []
    mu = CONSIDERATION_MU_AT_REF * agent.engagement / REF_ENGAGEMENT
    k = int(rng.poisson(mu))
    ranked = sorted(viewed, key=lambda l: (-l.quality, l.id))   # quality desc; id breaks ties
    return ranked[:k]
