"""Your own picks: long the stocks you expect to do well, short the ones you expect to lag.

The app doesn't choose anything here. It sizes your picks at equal weights, writes the
orders, and keeps score, so the recap shows whether your judgement adds up to a profit.
Checked every close: a change to the lists in config.toml is traded at the next close.
"""

import pandas as pd

PERIOD = "1mo"

DEFAULTS = {
    "long": [],  # tickers to buy
    "short": [],  # tickers to sell short
    "exposure": 1.0,  # stock held, long + short, as a multiple of the book's value
}


def check(params: dict) -> None:
    """Raise ValueError if the lists can't be traded."""
    if params["enabled"] and not (params["long"] or params["short"]):
        raise ValueError("add tickers to long or short, or set enabled = false")
    if both := sorted(set(params["long"]) & set(params["short"])):
        raise ValueError(f"{', '.join(both)} can't be in both long and short")


def tickers(params: dict) -> list[str]:
    return sorted(set(params["long"]) | set(params["short"]))


def targets(prices: pd.DataFrame, params: dict, asof=None,
            held=None) -> tuple[dict[str, float], dict[str, str]]:
    """Each pick with a price at the last close gets exposure / number of such picks, positive
    for a long and negative for a short."""
    if asof is not None:
        prices = prices.loc[:asof]
    priced = {t for t in prices if pd.notna(prices[t].iloc[-1])} if len(prices) else set()
    sides = [(t, 1) for t in params["long"] if t in priced] + \
        [(t, -1) for t in params["short"] if t in priced]
    size = params["exposure"] / len(sides) if sides else 0.0
    return ({t: side * size for t, side in sides},
            {t: f"your pick: {'long' if side > 0 else 'short'}" for t, side in sides})


def due(last_rebalance, date) -> bool:
    """Every close, so a change to the picks is traded straight away."""
    return True
