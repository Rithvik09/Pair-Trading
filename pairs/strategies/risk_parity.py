"""Risk parity: hold stocks, bonds, gold and commodities, each sized to bring the same risk.

Always long everything. A calm bond fund is held in more size than a jumpy stock fund, so no
one asset class dominates how the book moves. Rebalanced once a month.
"""

import math

import numpy as np
import pandas as pd

from pairs.strategies import trend

VOL_WINDOW = 60  # daily returns used to measure an ETF's volatility
PERIOD = "1y"

DEFAULTS = {
    # US stocks, international stocks, long and medium bonds, gold, commodities
    "universe": ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"],
    "exposure": 1.0,  # stock held as a multiple of the book's value; above 1 borrows to hold more
}


def targets(prices: pd.DataFrame, params: dict, asof=None,
            held=None) -> tuple[dict[str, float], dict[str, str]]:
    """Target weight and reason per ETF in params["universe"], from closes up to the last
    row of `prices` (or up to `asof`).

    Weight is proportional to 1 / the ETF's own volatility, scaled so the weights add up to
    `exposure`. An ETF without enough closes to measure its volatility gets 0 and the others
    share its part.
    """
    if asof is not None:
        prices = prices.loc[:asof]
    inverse, vols, reasons = {}, {}, {}
    for t in params["universe"]:
        s = prices[t].dropna() if t in prices else pd.Series(dtype=float)
        vol = (float(np.log(s).diff().iloc[-VOL_WINDOW:].std() * math.sqrt(252))
               if len(s) > VOL_WINDOW else float("nan"))
        if not vol > 0:
            reasons[t] = "not enough history"
            continue
        inverse[t], vols[t] = 1 / vol, vol
    total = sum(inverse.values())
    weights = {t: params["exposure"] * inverse.get(t, 0.0) / total if total else 0.0
               for t in params["universe"]}
    reasons.update({t: f"{vols[t]:.0%} volatility: {weights[t]:.0%} of the book" for t in vols})
    return weights, reasons


due = trend.due  # the same schedule: the first close of each month
