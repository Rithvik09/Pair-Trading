"""Short-term reversal: buy last week's biggest losers, short its biggest winners.

Big one-week moves in a stock with no news behind them tend to partly reverse the next week.
The edge is small and the turnover high (the book changes most of its holdings each week),
so trading costs take much of it. Swapped at the first close of each week.
"""

import pandas as pd

from pairs.strategies import momentum

LOOKBACK = 5  # trading days in "the past week"
PERIOD = "3mo"

DEFAULTS = {
    "index": "S&P 500",  # which index in universe.csv to rank
    "n_each": 20,  # stocks held long, and the same number short
    "exposure": 1.0,  # stock held, long + short, as a multiple of the book's value
}

tickers = momentum.tickers  # every stock in the index


def targets(prices: pd.DataFrame, params: dict, asof=None,
            held=None) -> tuple[dict[str, float], dict[str, str]]:
    """Long the n_each stocks that did worst over the past week, short the n_each that did
    best, each at exposure / 2 / n_each of the book. Only the stocks picked get a weight."""
    if asof is not None:
        prices = prices.loc[:asof]
    moves = {}
    for t in prices:
        s = prices[t].dropna()
        if len(s) > LOOKBACK and s.index[-1] == prices.index[-1]:
            moves[t] = s.iloc[-1] / s.iloc[-1 - LOOKBACK] - 1
    ranked = sorted(moves, key=moves.get)
    n = min(params["n_each"], len(ranked) // 2)
    size = params["exposure"] / 2 / n if n else 0.0
    weights, reasons = {}, {}
    for side, picks, word, label in ((1, ranked[:n], "worst", "long for a bounce"),
                                     (-1, ranked[len(ranked) - n:], "best", "short for a pullback")):
        for t in picks:
            weights[t] = side * size
            reasons[t] = (f"{'up' if moves[t] >= 0 else 'down'} {abs(moves[t]):.1%} over the past "
                          f"week: among the {n} {word} of {len(ranked)}, {label}")
    return weights, reasons


def due(last_rebalance: str | None, date) -> bool:
    """Rebalance at the first close of each week (by ISO week), or if never rebalanced."""
    if last_rebalance is None:
        return True
    last, now = pd.Timestamp(last_rebalance), pd.Timestamp(str(date))
    return now.isocalendar()[:2] > last.isocalendar()[:2]
