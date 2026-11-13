"""Stock vs. sector ETF stat arb: pairs trading with each stock's own sector as the partner.

When an S&P 500 stock drifts unusually far from its sector ETF (by the same z-score the pairs
trader uses), short whichever ran ahead and buy the other, then close once they're back in
line, or give up at the stop. Hedges on one ETF from several stocks are netted. Checked
every close, like pairs.
"""

import math
from functools import cache

import pandas as pd

from pairs.backtest import BacktestConfig, signal_frame
from pairs.paths import UNIVERSE_PATH
from pairs.signals import decide, leg_weights
from pairs.strategies import momentum

PERIOD = "2y"
SECTOR_ETFS = {  # GICS sector -> its Select Sector SPDR
    "Communication Services": "XLC", "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP", "Energy": "XLE", "Financials": "XLF", "Health Care": "XLV",
    "Industrials": "XLI", "Information Technology": "XLK", "Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU",
}

_SIGNAL = BacktestConfig()
DEFAULTS = {
    "index": "S&P 500",  # which index in universe.csv to trade
    "entry": _SIGNAL.entry, "exit": _SIGNAL.exit, "stop": _SIGNAL.stop,
    "lookback": _SIGNAL.lookback, "z_window": _SIGNAL.z_window,
    "position_size": 0.05,  # each trade's stock + hedge, as a share of the book
    "max_positions": 20,  # trades open at once; the widest gaps are taken first
}


@cache
def sector_etf() -> dict[str, str]:
    """Each stock in universe.csv -> its sector's ETF."""
    df = pd.read_csv(UNIVERSE_PATH)
    return {t: SECTOR_ETFS[s] for t, s in zip(df["ticker"], df["sector"], strict=True)
            if s in SECTOR_ETFS}


def tickers(params: dict) -> list[str]:
    """Every stock in the index, and the sector ETFs."""
    return sorted(set(momentum.tickers(params)) | set(SECTOR_ETFS.values()))


def targets(prices: pd.DataFrame, params: dict, asof=None,
            held: dict[str, int] | None = None) -> tuple[dict[str, float], dict[str, str]]:
    """Target weights from each stock's z-score against its sector ETF at the last close.

    `held` is the book's shares now: a stock held short is a short-spread trade, held long a
    long-spread one, and the pairs rule (signals.decide) says whether to keep, close or stop
    it. Kept trades come first; new ones fill the remaining max_positions by widest gap.
    A trade's position_size splits between the stock and the ETF by the hedge ratio, as a
    pairs trade does. A stock to close gets weight 0 and the reason.
    """
    if asof is not None:
        prices = prices.loc[:asof]
    held = held or {}
    cfg = BacktestConfig(lookback=params["lookback"], z_window=params["z_window"])
    size, etfs = params["position_size"], sector_etf()
    keep, new, weights, reasons = [], [], {}, {}
    for t in prices:
        etf = etfs.get(t)
        if etf is None or etf not in prices:
            continue
        df = prices[[t, etf]].dropna()
        if len(df) <= cfg.lookback + cfg.z_window or df.index[-1] != prices.index[-1]:
            continue
        frame = signal_frame(df[t], df[etf], cfg)
        z, beta = float(frame["z"].iloc[-1]), float(frame["beta"].iloc[-1])
        if math.isnan(z):
            continue
        pos = (held.get(t, 0) > 0) - (held.get(t, 0) < 0)
        action = decide(pos, False, z, params["entry"], params["exit"], params["stop"],
                        can_enter=beta > 0)
        if action == "hold":
            keep.append((t, etf, pos, beta, z,
                         f"z {z:+.2f} vs {etf}: not back yet, holding"))
        elif action in ("exit", "stop"):
            weights[t] = 0.0
            reasons[t] = (f"z {z:+.2f} vs {etf}: back within ±{params['exit']}, converged"
                          if action == "exit" else
                          f"z {z:+.2f} vs {etf}: past the ±{params['stop']} stop, give up")
        elif action in ("short_spread", "long_spread"):
            side = -1 if action == "short_spread" else 1
            ahead, behind = (t, etf) if side == -1 else (etf, t)
            moved = "ran ahead of" if side == -1 else "fell behind"
            why = f"z {z:+.2f} vs {etf}: {t} {moved} its sector, so short {ahead} / buy {behind}"
            new.append((t, etf, side, beta, z, why))
    new.sort(key=lambda trade: -abs(trade[4]))
    for t, etf, side, beta, _z, why in keep + new[:max(params["max_positions"] - len(keep), 0)]:
        w_y, w_x = leg_weights(beta)
        weights[t], reasons[t] = side * size * w_y, why
        weights[etf] = weights.get(etf, 0.0) - side * size * w_x
        reasons[etf] = f"hedge for the {etf} sector trades"
    return weights, reasons


def due(last_rebalance, date) -> bool:
    """Every close, like pairs."""
    return True
