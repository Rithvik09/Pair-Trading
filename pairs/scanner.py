"""Search a universe of tickers for cointegrated (mean-reverting) pairs."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd

from pairs.backtest import BacktestConfig, backtest
from pairs.stats import pair_stats

COLUMNS = ["group", "y", "x", "pvalue", "beta", "alpha", "half_life", "corr"]
SHORTLIST_COLUMNS = ["group", "y", "x", "pvalue", "pvalue_1y", "half_life", "corr", "beta",
                     "sharpe", "sharpe_late", "total_return", "return_1y", "max_drawdown",
                     "trades"]


def _per_group(fn, prices: pd.DataFrame, groups: dict[str, list[str]], workers: int,
               *args) -> list:
    """fn(group, that group's prices, *args) for every group, flattened. Groups are
    independent, so with workers > 1 they run in parallel processes."""
    jobs = [(g, prices[[t for t in tickers if t in prices.columns]]) for g, tickers in
            groups.items()]
    if workers <= 1:
        parts = [fn(g, p, *args) for g, p in jobs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(fn, *zip(*jobs, strict=True),
                                  *([a] * len(jobs) for a in args)))
    return [row for part in parts for row in part]


def _scan_group(group: str, prices: pd.DataFrame, max_pvalue: float, min_obs: int) -> list:
    rows = []
    for a, b in combinations(prices.columns, 2):
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
    return rows


def scan(
    prices: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
    max_pvalue: float = 0.05,
    min_obs: int = 200,
    workers: int = 1,
) -> pd.DataFrame:
    """Test every pair within each group; return those with p-value <= max_pvalue.

    Engle-Granger depends on which leg is the dependent variable, so both orderings
    are tested and the stronger one is kept.
    """
    if groups is None:
        groups = {"all": list(prices.columns)}
    rows = _per_group(_scan_group, prices, groups, workers, max_pvalue, min_obs)
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["pvalue", "group", "y", "x"],
                                                           ignore_index=True)


def _qualify_group(group: str, prices: pd.DataFrame, cfg: BacktestConfig, fees: dict,
                   f: dict) -> list[dict]:
    """The shortlist() filters for one group's pairs, before the per-group/ticker caps."""
    rows = []
    for r in _scan_group(group, prices, f["max_pvalue"], 2 * f["recent"]):
        _, y, x, pvalue, beta, _, hl, corr = r
        if not (f["half_life"][0] <= hl <= f["half_life"][1]
                and f["min_corr"] <= corr < f["max_corr"] and beta > 0):
            continue
        df = prices[[y, x]].dropna()
        recent = f["recent"]
        p_1y = pair_stats(df[y].iloc[-recent:], df[x].iloc[-recent:]).pvalue
        if f["max_pvalue_1y"] is not None and p_1y > f["max_pvalue_1y"]:
            continue
        pair_fees = (fees.get(y, 0.0), fees.get(x, 0.0))
        res = backtest(df[y], df[x], cfg, borrow_fees=pair_fees)
        s = res.stats
        ret_1y = float(res.equity.iloc[-1] / res.equity.iloc[-recent - 1] - 1)
        if s["sharpe"] < f["min_sharpe"] or s["total_return"] <= 0 \
                or s["trades"] < f["min_trades"]:
            continue
        if f["min_return_1y"] is not None and ret_1y < f["min_return_1y"]:
            continue
        late = backtest(df[y], df[x], replace(cfg, delay=1), borrow_fees=pair_fees).stats
        if late["sharpe"] < f["min_sharpe_late"]:
            continue
        rows.append({"group": group, "y": y, "x": x, "pvalue": pvalue, "pvalue_1y": p_1y,
                     "half_life": hl, "corr": corr, "beta": beta, "sharpe": s["sharpe"],
                     "sharpe_late": late["sharpe"], "total_return": s["total_return"],
                     "return_1y": ret_1y, "max_drawdown": s["max_drawdown"],
                     "trades": s["trades"]})
    return rows


def shortlist(
    prices: pd.DataFrame,
    groups: dict[str, list[str]],
    cfg: BacktestConfig,
    borrow_fees: dict[str, float] | None = None,
    max_pvalue: float = 0.10,
    half_life: tuple[float, float] = (5, 30),
    min_corr: float = 0.6,
    max_corr: float = 0.97,
    min_sharpe: float = 0.75,
    min_sharpe_late: float = 0.25,
    min_trades: int = 4,
    max_pvalue_1y: float | None = None,
    min_return_1y: float | None = None,
    recent: int = 252,
    per_group: int = 3,
    per_ticker: int = 2,
    workers: int = 1,
) -> pd.DataFrame:
    """Pairs worth trading: cointegrated over the whole history, reverting at a tradeable
    speed, moving together, and profitable in a backtest after costs and borrow, also when
    every signal is filled a day late (sharpe_late: you trade the next morning, not at the
    close that triggered it).

    The defaults did well in walk_forward() over 2022-2026 (S&P 1500), but that is also the
    period they were picked on. Picked on 2022-2025 alone, the best settings lost money in
    2025-26, as did 92% of all settings tried: treat any edge as unproven. Two checks that
    sound sensible made results worse or no better in every year: requiring cointegration
    over the last year on its own (max_pvalue_1y) and a profitable last year
    (min_return_1y); both are off unless set.

    Near-identical funds (return correlation >= max_corr) are skipped: their gaps are too small
    to trade. At most per_group pairs per sub-industry and per_ticker pairs per stock are kept
    (best Sharpe first), so the watchlist isn't one concentrated bet.
    """
    f = {"max_pvalue": max_pvalue, "half_life": half_life, "min_corr": min_corr,
         "max_corr": max_corr, "min_sharpe": min_sharpe, "min_sharpe_late": min_sharpe_late,
         "min_trades": min_trades, "max_pvalue_1y": max_pvalue_1y,
         "min_return_1y": min_return_1y, "recent": recent}
    rows = _per_group(_qualify_group, prices, groups, workers, cfg, borrow_fees or {}, f)

    out, by_group, by_ticker = [], {}, {}
    for row in sorted(rows, key=lambda d: (-d["sharpe"], d["group"], d["y"], d["x"])):
        if by_group.get(row["group"], 0) >= per_group:
            continue
        if any(by_ticker.get(t, 0) >= per_ticker for t in (row["y"], row["x"])):
            continue
        out.append(row)
        by_group[row["group"]] = by_group.get(row["group"], 0) + 1
        for t in (row["y"], row["x"]):
            by_ticker[t] = by_ticker.get(t, 0) + 1
    return pd.DataFrame(out, columns=SHORTLIST_COLUMNS)


def walk_forward(
    prices: pd.DataFrame,
    groups: dict[str, list[str]],
    cfg: BacktestConfig,
    borrow_fees: dict[str, float] | None = None,
    formation: int = 756,
    holdout: int = 252,
    folds: int = 2,
    workers: int = 1,
    **filters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How well does shortlist() pick pairs? An honest answer, unlike the shortlist's own stats.

    For each fold, run shortlist() on the `formation` bars before a cut-off date only, then trade
    the picks over the next `holdout` bars, which the choice never saw. Folds are consecutive
    holdout years ending today, fold 1 the oldest. Returns (one row per pick, one per fold),
    the fold row scoring an equal-weight portfolio of that fold's picks.

    Caveat: the universe is today's index members, so stocks that fell out are missing
    (survivorship bias); real results are likely a bit worse.
    """
    fees = borrow_fees or {}
    picks, folds_out = [], []
    for fold in range(1, folds + 1):
        cut = len(prices) - (folds - fold + 1) * holdout
        if cut - formation < 0:
            raise ValueError(f"need {formation + folds * holdout} bars of prices, "
                             f"have {len(prices)}")
        start = prices.index[cut]
        chosen = shortlist(prices.iloc[cut - formation:cut], groups, cfg, fees,
                           workers=workers, **filters)
        daily = {}
        for r in chosen.itertuples():
            df = prices[[r.y, r.x]].iloc[cut - formation:cut + holdout].dropna()
            res = backtest(df[r.y], df[r.x], cfg,
                           borrow_fees=(fees.get(r.y, 0.0), fees.get(r.x, 0.0)))
            ret = res.daily_pnl[res.daily_pnl.index >= start]
            daily[(r.y, r.x)] = ret
            picks.append({"fold": fold, "holdout_start": start, "group": r.group, "y": r.y,
                          "x": r.x, "sharpe": r.sharpe,
                          "holdout_return": float((1 + ret).prod() - 1),
                          # trades active in the holdout, including one already open at the cut
                          "holdout_trades": int((res.trades["exit_date"] >= start).sum())})
        port = pd.DataFrame(daily).fillna(0.0).mean(axis=1) if daily else pd.Series(dtype=float)
        rets = [p["holdout_return"] for p in picks if p["fold"] == fold]
        sd = port.std() if len(port) > 1 else 0.0
        folds_out.append({
            "fold": fold, "holdout_start": start, "pairs": len(rets),
            "hit_rate": float(np.mean([x > 0 for x in rets])) if rets else float("nan"),
            "mean_return": float(np.mean(rets)) if rets else float("nan"),
            "sharpe": float(port.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
        })
    return pd.DataFrame(picks), pd.DataFrame(folds_out)
