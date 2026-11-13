"""Momentum long/short: buy the stocks that rose most over the past year, short those that
fell most, and swap them once a month.

The past year's return leaves out the latest month, which tends to reverse rather than carry
on. The universe is today's S&P 500 from universe.csv, so a backtest of this would miss the
stocks that dropped out of the index; the paper book doesn't have that problem.
"""

import pandas as pd

from pairs.paths import UNIVERSE_PATH
from pairs.strategies import trend

LOOKBACK = 252  # trading days back to the start of the 12 months
SKIP = 21  # the latest month, left out
PERIOD = "2y"

DEFAULTS = {
    "index": "S&P 500",  # which index in universe.csv to rank
    "n_each": 20,  # stocks held long, and the same number short
    "exposure": 1.0,  # stock held, long + short, as a multiple of the book's value
}


def tickers(params: dict) -> list[str]:
    """Every stock in params["index"], from universe.csv."""
    df = pd.read_csv(UNIVERSE_PATH)
    return sorted(set(df.loc[df["index"] == params["index"], "ticker"]))


def targets(prices: pd.DataFrame, params: dict, asof=None,
            held=None) -> tuple[dict[str, float], dict[str, str]]:
    """Long the n_each stocks with the best 12-month return up to a month ago, short the
    n_each worst, each at exposure / 2 / n_each of the book. Stocks without that much
    history aren't ranked. Only the stocks picked get a weight; the book closes the rest."""
    if asof is not None:
        prices = prices.loc[:asof]
    scores = {}
    for t in prices:
        s = prices[t].dropna()
        if len(s) > LOOKBACK and s.index[-1] == prices.index[-1]:
            scores[t] = s.iloc[-1 - SKIP] / s.iloc[-1 - LOOKBACK] - 1
    ranked = sorted(scores, key=scores.get)
    n = min(params["n_each"], len(ranked) // 2)
    size = params["exposure"] / 2 / n if n else 0.0
    weights, reasons = {}, {}
    for side, picks, word, label in ((1, ranked[len(ranked) - n:], "best", "long"),
                                     (-1, ranked[:n], "worst", "short")):
        for t in picks:
            weights[t] = side * size
            reasons[t] = (f"{'up' if scores[t] >= 0 else 'down'} {abs(scores[t]):.1%} over the "
                          f"12 months to a month ago: among the {n} {word} of {len(ranked)}, "
                          f"{label}")
    return weights, reasons


due = trend.due  # the first close of each month
