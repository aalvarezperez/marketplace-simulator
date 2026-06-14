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
