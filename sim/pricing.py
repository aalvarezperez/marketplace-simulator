import numpy as np
from sklearn.linear_model import LinearRegression


class NaiveMeanPriceModel:
    """Constant price; used when there is nothing to learn from."""

    def __init__(self, value):
        self.value = float(value)

    def predict_price(self, quality):
        return self.value


class LinearPriceModel:
    """Wraps a fitted price~quality regression."""

    def __init__(self, reg):
        self._reg = reg

    def predict_price(self, quality):
        return float(self._reg.predict(np.array([[float(quality)]]))[0])


def fit_price_model(market, top_k=10, fallback_mean=500.0):
    """Fit price~quality on the top_k visible listings; fall back to naive mean."""
    listings = sorted(market.listings, key=lambda l: l.quality, reverse=True)[:top_k]
    if listings:
        X = np.array([[l.quality] for l in listings], dtype=float)
        y = np.array([l.price for l in listings], dtype=float)
        return LinearPriceModel(LinearRegression().fit(X, y))
    return NaiveMeanPriceModel(fallback_mean)


class EndogenousPrice:
    """A Property value: price emerges from a model fit on visible listings.

    Requires context={'market': ..., 'quality': ...}; optional 'bias' scales the
    result (buyers bid below ask with bias < 1). Deterministic: fit/predict do not
    consume the rng.
    """

    def __init__(self, fallback_mean=500.0, top_k=10, bias=1.0):
        self.fallback_mean = fallback_mean
        self.top_k = top_k
        self.bias = bias

    def draw_with_context(self, rng, context):
        market = context["market"]
        quality = context["quality"]
        bias = context.get("bias", self.bias)
        model = fit_price_model(market, top_k=self.top_k,
                                fallback_mean=self.fallback_mean)
        return max(0.0, bias * model.predict_price(quality))
