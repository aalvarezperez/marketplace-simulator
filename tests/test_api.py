from datetime import datetime


def test_top_level_imports():
    from sim import (Marketplace, MarketplaceSpec, Property, negotiate_action,
                     default_pricing, default_willingness)
    assert all(x is not None for x in (Marketplace, MarketplaceSpec, Property,
                                       negotiate_action, default_pricing, default_willingness))


def test_summary_returns_event_counts():
    from sim import Marketplace, MarketplaceSpec
    mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                                n_seed_users=50, until=3.0, seed=1))
    mkt.run()
    s = mkt.summary()
    assert isinstance(s, dict) and s.get("visit", 0) > 0


def test_write_jsonl_top_level(tmp_path):
    from sim import Marketplace, MarketplaceSpec
    mkt = Marketplace.from_spec(MarketplaceSpec(start=datetime(2026, 1, 1),
                                                n_seed_users=20, until=2.0, seed=1))
    mkt.run()
    p = tmp_path / "e.jsonl"
    mkt.write_jsonl(p)
    assert p.exists() and p.read_text().strip()


def test_spec_has_grouped_docstring():
    from sim import MarketplaceSpec
    assert MarketplaceSpec.__doc__ and len(MarketplaceSpec.__doc__) > 100
