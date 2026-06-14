class Property:
    """Fixed-as-dynamic: a literal, a scipy frozen distribution (has .rvs),
    or a callable(rng). A literal is the degenerate generator."""

    def __init__(self, value):
        self.value = value

    def draw(self, rng):
        v = self.value
        if hasattr(v, "rvs"):
            return v.rvs(random_state=rng)
        if callable(v):
            return v(rng)
        return v
