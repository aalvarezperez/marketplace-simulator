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
