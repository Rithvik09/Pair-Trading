"""Walk-forward backtest of the mean-reversion strategy on one pair.

Everything at bar t is computed from data up to and including t, and the position
decided at t only earns returns from t to t+1, so there is no look-ahead.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairs.signals import leg_weights, positions_from_z, trade_opens
from pairs.stats import zscore


@dataclass(frozen=True)
class BacktestConfig:
    lookback: int = 120  # bars used for the rolling hedge ratio
    z_window: int = 30  # bars used for the spread's rolling mean/std
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    cost_bps: float = 5.0  # per unit of gross notional traded (covers both legs)
    capital: float = 100_000.0
    delay: int = 0  # extra bars between signal and fill; 1 = stress test for filling a day late


@dataclass
class BacktestResult:
    frame: pd.DataFrame  # beta, alpha, spread, z, position per bar
    position: pd.Series
    daily_pnl: pd.Series  # fractional return of capital per bar, after costs and borrow
    borrow_cost: pd.Series  # fraction of capital paid in short-borrow fees per bar
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def signal_frame(y: pd.Series, x: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """Rolling hedge ratio, spread, z-score and target position for each bar."""
    ly, lx = np.log(y), np.log(x)
    roll_x = lx.rolling(cfg.lookback)
    beta = ly.rolling(cfg.lookback).cov(lx) / roll_x.var()
    alpha = ly.rolling(cfg.lookback).mean() - beta * roll_x.mean()
    spr = ly - beta * lx - alpha
    z = zscore(spr, cfg.z_window)
    # only enter while the hedge ratio is positive, i.e. the two stocks move together
    pos = positions_from_z(z, cfg.entry, cfg.exit, cfg.stop, can_enter=beta > 0)
    return pd.DataFrame({"beta": beta, "alpha": alpha, "spread": spr, "z": z, "position": pos})


def backtest(y: pd.Series, x: pd.Series, cfg: BacktestConfig | None = None,
             borrow_fees: tuple[float, float] = (0.0, 0.0)) -> BacktestResult:
    """borrow_fees: annual borrow rate (decimal) for (Y, X), charged on whichever leg is short.

    Today's rates are applied to the whole history, since historical borrow data isn't free.
    """
    cfg = cfg or BacktestConfig()
    y_name, x_name = y.name or "Y", x.name or "X"
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    frame = signal_frame(df["y"], df["x"], cfg)
    if cfg.delay < 0:
        raise ValueError(f"delay must be >= 0 (a negative delay trades on future signals), "
                         f"got {cfg.delay}")
    if cfg.delay:  # fill each signal `delay` bars late, sized on the signal day as real orders are
        frame = frame.assign(position=frame["position"].shift(cfg.delay, fill_value=0),
                             beta=frame["beta"].shift(cfg.delay))
    pos = frame["position"]

    # Hedge ratio is locked at entry for the life of each trade.
    held_beta = frame["beta"].where(trade_opens(pos)).ffill().where(pos.ne(0))
    w_y, w_x = leg_weights(held_beta)

    # Position and weights decided at t-1 earn the move from t-1 to t.
    leg_ret = w_y.shift(1) * df["y"].pct_change() - w_x.shift(1) * df["x"].pct_change()
    gross = (pos.shift(1) * leg_ret).fillna(0.0)
    costs = pos.diff().abs().fillna(pos.abs()) * cfg.cost_bps / 1e4
    # Signed dollar exposure per leg held over the bar; negative = short, pays borrow.
    exp_y, exp_x = (pos * w_y).shift(1), (-pos * w_x).shift(1)
    # actual/360 like the paper book: a Monday bar carries the weekend's borrow
    days = df.index.to_series().diff().dt.days
    borrow = ((-exp_y).clip(lower=0) * borrow_fees[0]
              + (-exp_x).clip(lower=0) * borrow_fees[1]).mul(days / 360).fillna(0.0)
    ret = gross - costs - borrow
    equity = cfg.capital * (1 + ret).cumprod()

    trades = _trades(frame, equity, cfg.capital, str(y_name), str(x_name))
    return BacktestResult(
        frame=frame,
        position=pos,
        daily_pnl=ret,
        borrow_cost=borrow,
        equity=equity,
        trades=trades,
        stats={**_stats(ret, pos, equity, trades, cfg.capital),
               "borrow_paid": float((borrow * equity.shift(1, fill_value=cfg.capital)).sum())},
    )


def _trades(frame: pd.DataFrame, equity: pd.Series, capital: float, y: str, x: str) -> pd.DataFrame:
    pos = frame["position"].to_numpy()
    idx = frame.index
    rows = []
    i, n = 0, len(pos)
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        side = pos[i]
        j = i
        while j + 1 < n and pos[j + 1] == side:
            j += 1
        # positions are earned one bar late, so the trade is realised on the bar after it ends
        exit_i = min(j + 1, n - 1)
        before = equity.iloc[i - 1] if i > 0 else capital
        rows.append(
            {
                "entry_date": idx[i],
                "exit_date": idx[exit_i],
                "side": "short_spread" if side == -1 else "long_spread",
                "short_leg": y if side == -1 else x,
                "long_leg": x if side == -1 else y,
                "entry_z": frame["z"].iloc[i],
                "exit_z": frame["z"].iloc[exit_i],
                "days": exit_i - i,
                "pnl": equity.iloc[exit_i] - before,
                "open": j + 1 >= n,
            }
        )
        i = j + 1
    cols = ["entry_date", "exit_date", "side", "short_leg", "long_leg",
            "entry_z", "exit_z", "days", "pnl", "open"]
    return pd.DataFrame(rows, columns=cols)


def _stats(ret: pd.Series, pos: pd.Series, equity: pd.Series, trades: pd.DataFrame,
           capital: float) -> dict:
    total = equity.iloc[-1] / capital - 1
    years = max(len(ret) / 252, 1e-9)
    sd = ret.std()
    closed = trades[~trades["open"]]
    return {
        "total_return": float(total),
        "cagr": float(max(1 + total, 0.0) ** (1 / years) - 1),
        "sharpe": float(ret.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "trades": len(trades),
        "win_rate": float((closed["pnl"] > 0).mean()) if len(closed) else float("nan"),
        "avg_days": float(trades["days"].mean()) if len(trades) else float("nan"),
        "time_in_market": float(pos.ne(0).mean()),
    }
