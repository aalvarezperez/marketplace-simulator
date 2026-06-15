# marketplace-simulator

Simulate marketplace data from the ground up — an agent-based simulator where heterogeneous users
visit, list, view, bid, and transact, and the funnel, conversion, and prices **emerge** from agents
interacting with supply.

Two engines:
- **`sim/`** — the active engine (v2.0): a deterministic, single-threaded, **continuous-time SimPy**
  model. Agents + actions are the primitives; behaviour emerges. This is the one to use.
- **`classes.py`** — the original discrete-daily-loop engine, kept frozen as a reference.

```python
from datetime import datetime
from sim import Marketplace, MarketplaceSpec, negotiate_action

mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                            n_seed_users=1000, until=7.0, seed=42,
                                            actions=[negotiate_action()]))
events = mkt.run()
mkt.summary()                 # event-type counts
```

`pip install -r requirements.txt`, then `python -m pytest`. See `CLAUDE.md` for the architecture
and `docs/` for the design trail.
