from collections import Counter

from sim.allocation import bucket


def test_bucket_is_deterministic():
    v = {"CONTROL": 0.5, "B": 0.5}
    a = bucket("user-7", v, salt="exp1")
    b = bucket("user-7", v, salt="exp1")
    assert a == b
    assert a in v


def test_bucket_respects_weights_roughly():
    v = {"CONTROL": 0.8, "B": 0.2}
    counts = Counter(bucket(f"user-{i}", v, salt="exp1") for i in range(4000))
    frac_b = counts["B"] / sum(counts.values())
    assert 0.15 < frac_b < 0.25          # ~0.2, generous bound


def test_bucket_salt_changes_assignment():
    v = {"CONTROL": 0.5, "B": 0.5}
    # Over many keys, two different salts must not produce identical assignments.
    same = sum(bucket(f"u{i}", v, salt="A") == bucket(f"u{i}", v, salt="B")
               for i in range(500))
    assert same < 500                    # independent salts -> not all equal

from sim.allocation import (SimpleRandomization, ClusterRandomization,
                            Switchback)

VARS = {"CONTROL": 0.5, "B": 0.5}


def test_simple_is_sticky_and_time_invariant():
    s = SimpleRandomization()
    assert s.window(0.0) is None
    assert s.window(9.9) is None
    v1 = s.assign("user-3", cluster_key="c0", window=None, variants=VARS, salt="e")
    v2 = s.assign("user-3", cluster_key="c1", window=None, variants=VARS, salt="e")
    assert v1 == v2                      # depends on the unit key, not the cluster


def test_cluster_shares_variant_within_cluster():
    s = ClusterRandomization()
    # Different units, same cluster -> same variant.
    a = s.assign("user-1", cluster_key="north", window=None, variants=VARS, salt="e")
    b = s.assign("user-2", cluster_key="north", window=None, variants=VARS, salt="e")
    assert a == b


def test_switchback_flips_by_window():
    s = Switchback(period=1.0)
    assert s.window(0.4) == 0
    assert s.window(1.0) == 1
    assert s.window(2.7) == 2
    # Same window -> same variant; the assignment depends on the window, not the unit.
    v0a = s.assign("user-1", cluster_key="c", window=0, variants=VARS, salt="e")
    v0b = s.assign("user-2", cluster_key="c", window=0, variants=VARS, salt="e")
    assert v0a == v0b
    assert s.window_bounds(3) == (3.0, 4.0)


def test_switchback_per_cluster_differs():
    s = Switchback(period=1.0, per_cluster=True)
    # Same window, different clusters can differ; the key includes the cluster.
    distinct = {s.assign("u", cluster_key=c, window=0, variants=VARS, salt="e")
                for c in range(20)}
    assert distinct == {"CONTROL", "B"}  # both variants appear across clusters
