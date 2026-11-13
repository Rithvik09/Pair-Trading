"""Trend following: hold what has been going up, short what has been going down.

Each ETF's signal is the sign of its own return over the past 12 months (time-series
momentum), so this can be long or short everything at once. Positions are sized so each ETF
brings the same expected volatility, and rebalanced once a month.

vol_target is a budget split evenly over the ETFs, not the volatility the book will have: that
would take every ETF moving together. They don't, so the book runs well below it.
"""

import math

import numpy as np
import pandas as pd

LOOKBACK = 252  # trading days in the 12-month return
VOL_WINDOW = 60  # daily returns used to measure an ETF's volatility
PERIOD = "2y"  # history to load: a year for the signal, plus room for market holidays

DEFAULTS = {
    # stocks, bonds, credit, gold, commodities, property, the dollar
    "universe": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "DBC", "VNQ",
                 "UUP"],
    # volatility budget: each ETF is sized for vol_target / N a year. The book's own
    # volatility is lower, because they don't all move together
    "vol_target": 0.10,
    "max_gross": 1.5,  # most stock held, long + short, as a multiple of the book's value
}


def targets(prices: pd.DataFrame, params: dict, asof=None,
            held=None) -> tuple[dict[str, float], dict[str, str]]:
    """Target weight and reason per ETF in params["universe"], from closes up to the last
    row of `prices` (or up to `asof`, to ask what it would have said on an earlier day).

    Weight = ±(vol_target / number of ETFs) / the ETF's own volatility, so a calm bond fund
    is held in more size than a jumpy stock fund, and each brings vol_target / N of
    volatility on its own (the book as a whole less than their sum: see the module note).
    If the weights add up to more than max_gross, every one is scaled down together. An ETF
    without 12 months of closes gets 0.
    """
    if asof is not None:
        prices = prices.loc[:asof]
    universe = params["universe"]
    weights, reasons = {}, {}
    for t in universe:
        s = prices[t].dropna() if t in prices else pd.Series(dtype=float)
        if len(s) < LOOKBACK + 1:
            weights[t], reasons[t] = 0.0, "not enough history"
            continue
        ret = s.iloc[-1] / s.iloc[-1 - LOOKBACK] - 1
        vol = float(np.log(s).diff().iloc[-VOL_WINDOW:].std() * math.sqrt(252))
        if ret == 0 or not vol > 0:
            weights[t], reasons[t] = 0.0, "flat over 12 months: no position"
            continue
        side = 1 if ret > 0 else -1
        weights[t] = side * params["vol_target"] / len(universe) / vol
        reasons[t] = (f"{'up' if side > 0 else 'down'} {abs(ret):.1%} over 12 months: "
                      f"{'long' if side > 0 else 'short'}, sized for {vol:.0%} volatility")
    gross = sum(abs(w) for w in weights.values())
    if gross > params["max_gross"]:
        weights = {t: w * params["max_gross"] / gross for t, w in weights.items()}
    return weights, reasons


def due(last_rebalance: str | None, date) -> bool:
    """Rebalance at the first close of each month: when `date` is in a later month than the
    last rebalance, or the book has never been rebalanced."""
    if last_rebalance is None:
        return True
    last, now = pd.Timestamp(last_rebalance), pd.Timestamp(str(date))
    return (now.year, now.month) > (last.year, last.month)
