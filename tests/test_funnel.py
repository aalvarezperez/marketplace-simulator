from sim.agents import p_buy, p_view


def test_p_view_monotonic_in_engagement():
    assert p_view(10) > p_view(1)


def test_p_buy_monotonic_in_engagement():
    assert p_buy(10) > p_buy(1)


def test_probabilities_in_unit_interval():
    for e in [0.01, 1, 5, 50]:
        assert 0.0 <= p_view(e) <= 1.0
        assert 0.0 <= p_buy(e) <= 1.0
