"""Low volatility long/short: long the calmest S&P 500 stocks, short the jumpiest.

Calm stocks have tended to return about as much as jumpy ones for far less risk. Each stock
is sized to bring the same volatility, so the calm side is held in more size than the jumpy
side and the two sides carry equal risk; the book isn't simply short the market's wildest
names. Swapped at the first close of each month.
"""

import math

import numpy as np
import pandas as pd

from pairs.strategies import momentum, trend

VOL_WINDOW = 252  # a year of daily returns
PERIOD = "2y"

DEFAULTS = {
    "index": "S&P 500",  # which index in universe.csv to rank
    "n_each": 20,  # stocks held long, and the same number short
    "exposure": 1.0,  # stock held, long + short, as a multiple of the book's value
}

tickers = momentum.tickers  # every stock in the index


def targets(prices: pd.DataFrame, params: dict,
            asof=None) -> tuple[dict[str, float], dict[str, str]]:
    """Long the n_each stocks with the lowest volatility over the past year, short the n_each
    with the highest. Weight is proportional to 1 / volatility, scaled so the weights add up
    to `exposure` in absolute terms. Stocks without a year of closes aren't ranked."""
    if asof is not None:
        prices = prices.loc[:asof]
    vols = {}
    for t in prices:
        s = prices[t].dropna()
        if len(s) > VOL_WINDOW and s.index[-1] == prices.index[-1]:
            vol = float(np.log(s).diff().iloc[-VOL_WINDOW:].std() * math.sqrt(252))
            if vol > 0:
                vols[t] = vol
    ranked = sorted(vols, key=vols.get)
    n = min(params["n_each"], len(ranked) // 2)
    sides = ((1, ranked[:n], "calmest", "long"), (-1, ranked[len(ranked) - n:], "jumpiest", "short"))
    scale = params["exposure"] / sum(1 / vols[t] for _, picks, _, _ in sides for t in picks) \
        if n else 0.0
    weights, reasons = {}, {}
    for side, picks, word, label in sides:
        for t in picks:
            weights[t] = side * scale / vols[t]
            reasons[t] = f"{vols[t]:.0%} volatility: among the {n} {word} of {len(ranked)}, {label}"
    return weights, reasons


due = trend.due  # the first close of each month
