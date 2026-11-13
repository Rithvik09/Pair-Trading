"""Strategies that run next to pairs trading, each on its own paper book (pairs.portfolio).

A strategy is a module with:
    DEFAULTS                  its config.toml settings ([strategies.<name>]) and their defaults
    PERIOD                    how much price history it needs, e.g. "2y"
    targets(prices, params, asof=None, held=None)
                              -> (weights, reasons): the share of the book to hold in each
                              ticker (negative = short) as of the last row of `prices`, and
                              one plain-English line per ticker on why. `held` is the
                              book's shares now, for a strategy that keeps a trade open
                              until its own exit rule says to close it
    due(last_rebalance, date) whether to trade the book to those targets at this close
    tickers(params)           optional: what to price on a rebalance day, if not
                              params["universe"] (e.g. every stock in an index)

To add one, write the module and list it here; the daily job, recap and dashboard pick it up.
"""

from pairs.strategies import low_vol, momentum, reversal, risk_parity, sector_arb, trend

STRATEGIES = {"trend": trend, "risk_parity": risk_parity, "momentum": momentum,
              "low_vol": low_vol, "sector_arb": sector_arb, "reversal": reversal}
