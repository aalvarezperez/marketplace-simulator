import numpy as np
from sklearn.linear_model import LinearRegression


class LinearPriceModel:
    """Wraps a fitted price~quality regression."""

    def __init__(self, reg):
        self._reg = reg

    def predict_price(self, quality):
        return float(self._reg.predict(np.array([[float(quality)]]))[0])


def fit_price_model(market, top_k=10):
    """Fit price~quality on the top_k visible listings, when they carry price
    *variation* to learn from (>=2 listings with non-zero price spread).

    Returns None when there is nothing informative — the caller then falls back
    to a prior. Fitting a regression on identical prices would predict that one
    constant for every quality and lock the whole market into it forever.
    """
    listings = sorted(market.listings, key=lambda l: l.quality, reverse=True)[:top_k]
    if len(listings) >= 2:
        y = np.array([l.price for l in listings], dtype=float)
        if y.std() > 0:
            X = np.array([[l.quality] for l in listings], dtype=float)
            return LinearPriceModel(LinearRegression().fit(X, y))
    return None


class EndogenousPrice:
    """A Property value: price emerges from the market.

    With enough varied visible listings, price = bias * regression(price~quality).
    Until then, sellers price from a **quality-anchored prior** with idiosyncratic
    lognormal noise, so the first prices carry quality signal + spread for the
    regression to learn from. A flat constant prior would collapse every price to
    a single value (the regression then only ever sees that constant).

    context = {'market': ..., 'quality': ...}; optional 'bias' scales the result
    (buyers bid below ask with bias < 1).
    """

    def __init__(self, prior_sigma=0.4, top_k=10, bias=1.0):
        self.prior_sigma = prior_sigma
        self.top_k = top_k
        self.bias = bias

    def draw_with_context(self, rng, context):
        market = context["market"]
        quality = float(context["quality"])
        bias = context.get("bias", self.bias)
        model = fit_price_model(market, top_k=self.top_k)
        if model is not None:
            base = model.predict_price(quality)
        else:
            # Quality-anchored prior with lognormal noise (bootstrap variance).
            base = quality * float(np.exp(rng.normal(0.0, self.prior_sigma)))
        return max(0.0, bias * base)
