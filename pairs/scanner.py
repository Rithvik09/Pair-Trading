"""Search a universe of tickers for cointegrated (mean-reverting) pairs."""

from itertools import combinations

import pandas as pd

from pairs.backtest import BacktestConfig, backtest
from pairs.stats import pair_stats

COLUMNS = ["group", "y", "x", "pvalue", "beta", "alpha", "half_life", "corr"]


def scan(
    prices: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
    max_pvalue: float = 0.05,
    min_obs: int = 200,
) -> pd.DataFrame:
    """Test every pair within each group; return those with p-value <= max_pvalue.

    Engle-Granger depends on which leg is the dependent variable, so both orderings
    are tested and the stronger one is kept.
    """
    if groups is None:
        groups = {"all": list(prices.columns)}

    rows = []
    for group, tickers in groups.items():
        present = [t for t in tickers if t in prices.columns]
        for a, b in combinations(present, 2):
            df = prices[[a, b]].dropna()
            if len(df) < min_obs:
                continue
            best = None
            for y, x in ((a, b), (b, a)):
                st = pair_stats(df[y], df[x])
                if best is None or st.pvalue < best[2].pvalue:
                    best = (y, x, st)
            y, x, st = best
            if st.pvalue <= max_pvalue:
                rows.append([group, y, x, st.pvalue, st.beta, st.alpha, st.half_life, st.corr])

    return pd.DataFrame(rows, columns=COLUMNS).sort_values("pvalue", ignore_index=True)


def shortlist(
    prices: pd.DataFrame,
    groups: dict[str, list[str]],
    cfg: BacktestConfig,
    borrow_fees: dict[str, float] | None = None,
    max_pvalue: float = 0.05,
    max_pvalue_1y: float = 0.10,
    half_life: tuple[float, float] = (5, 30),
    min_corr: float = 0.6,
    max_corr: float = 0.97,
    min_sharpe: float = 0.5,
    min_trades: int = 4,
    recent: int = 252,
    per_group: int = 3,
    per_ticker: int = 2,
) -> pd.DataFrame:
    """Pairs worth trading: related over the whole history AND the most recent year, reverting
    at a tradeable speed, and profitable in a backtest after costs and borrow, including over
    the most recent year on its own.

    Caveat: choosing pairs by their own past backtest flatters them; the 1-year checks are
    there to reject relationships that have already faded.
    Near-identical funds (return correlation >= max_corr) are skipped: their gaps are too small
    to trade. At most per_group pairs per sector and per_ticker pairs per stock are kept (best
    Sharpe first), so the watchlist isn't one concentrated bet.
    """
    fees = borrow_fees or {}
    rows = []
    for r in scan(prices, groups, max_pvalue=max_pvalue).itertuples():
        if not (half_life[0] <= r.half_life <= half_life[1]
                and min_corr <= r.corr < max_corr and r.beta > 0):
            continue
        df = prices[[r.y, r.x]].dropna()
        if len(df) < 2 * recent:
            continue
        p_1y = pair_stats(df[r.y].iloc[-recent:], df[r.x].iloc[-recent:]).pvalue
        if p_1y > max_pvalue_1y:
            continue
        res = backtest(df[r.y], df[r.x], cfg, borrow_fees=(fees.get(r.y, 0.0), fees.get(r.x, 0.0)))
        s = res.stats
        ret_1y = float(res.equity.iloc[-1] / res.equity.iloc[-recent - 1] - 1)
        if s["sharpe"] < min_sharpe or s["total_return"] <= 0 or s["trades"] < min_trades \
                or ret_1y < 0:
            continue
        rows.append({"group": r.group, "y": r.y, "x": r.x, "pvalue": r.pvalue,
                     "pvalue_1y": p_1y, "half_life": r.half_life, "corr": r.corr,
                     "beta": r.beta, "sharpe": s["sharpe"], "total_return": s["total_return"],
                     "return_1y": ret_1y, "max_drawdown": s["max_drawdown"],
                     "trades": s["trades"]})

    out, by_group, by_ticker = [], {}, {}
    for row in sorted(rows, key=lambda d: d["sharpe"], reverse=True):
        if by_group.get(row["group"], 0) >= per_group:
            continue
        if any(by_ticker.get(t, 0) >= per_ticker for t in (row["y"], row["x"])):
            continue
        out.append(row)
        by_group[row["group"]] = by_group.get(row["group"], 0) + 1
        for t in (row["y"], row["x"]):
            by_ticker[t] = by_ticker.get(t, 0) + 1
    cols = ["group", "y", "x", "pvalue", "pvalue_1y", "half_life", "corr", "beta", "sharpe",
            "total_return", "return_1y", "max_drawdown", "trades"]
    return pd.DataFrame(out, columns=cols)
