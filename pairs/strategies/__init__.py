"""Strategies that run next to pairs trading, each on its own paper book (pairs.portfolio).

A strategy is a module with:
    DEFAULTS                  its config.toml settings ([strategies.<name>]) and their defaults
    PERIOD                    how much price history it needs, e.g. "2y"
    targets(prices, params)   -> (weights, reasons): the share of the book to hold in each
                              ticker (negative = short) as of the last row of `prices`, and
                              one plain-English line per ticker on why
    due(last_rebalance, date) whether to trade the book to those targets at this close
    tickers(params)           optional: what to price on a rebalance day, if not
                              params["universe"] (e.g. every stock in an index)

To add one, write the module and list it here; the daily job, recap and dashboard pick it up.
"""

from pairs.strategies import momentum, risk_parity, trend

STRATEGIES = {"trend": trend, "risk_parity": risk_parity, "momentum": momentum}
