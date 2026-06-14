import numpy as np
from scipy.stats import norm

from sim.spec import Property


def test_property_literal_returns_value():
    assert Property(7).draw(np.random.default_rng(0)) == 7


def test_property_distribution_is_deterministic_with_seed():
    a = Property(norm(loc=10, scale=2)).draw(np.random.default_rng(42))
    b = Property(norm(loc=10, scale=2)).draw(np.random.default_rng(42))
    assert a == b


def test_property_callable_receives_rng():
    p = Property(lambda rng: rng.integers(100, 200))
    v = p.draw(np.random.default_rng(1))
    assert 100 <= v < 200
