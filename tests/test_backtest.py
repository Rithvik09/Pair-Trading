import numpy as np
import pandas as pd

from pairs.backtest import BacktestConfig, backtest


def test_profitable_on_mean_reverting_pair(coint_pair):
    y, x = coint_pair
    res = backtest(y, x, BacktestConfig(cost_bps=0))
    assert res.stats["trades"] >= 5
    assert res.stats["total_return"] > 0
    assert res.equity.index.equals(y.index)


def test_costs_reduce_return(coint_pair):
    y, x = coint_pair
    free = backtest(y, x, BacktestConfig(cost_bps=0)).stats["total_return"]
    costly = backtest(y, x, BacktestConfig(cost_bps=20)).stats["total_return"]
    assert costly < free


def test_no_lookahead(coint_pair):
    """Changing the final day's prices must not change any earlier position."""
    y, x = coint_pair
    base = backtest(y, x, BacktestConfig())
    y2 = y.copy()
    y2.iloc[-1] *= 1.5
    bumped = backtest(y2, x, BacktestConfig())
    pd.testing.assert_series_equal(base.position.iloc[:-1], bumped.position.iloc[:-1])
    # and the final-day shock is earned with the position held *before* it happened
    assert base.daily_pnl.iloc[:-1].equals(bumped.daily_pnl.iloc[:-1])


def test_trades_record_sides(coint_pair):
    y, x = coint_pair
    res = backtest(y, x, BacktestConfig())
    t = res.trades.iloc[0]
    assert t["side"] in ("long_spread", "short_spread")
    if t["side"] == "short_spread":
        assert (t["short_leg"], t["long_leg"]) == ("YYY", "XXX")
    else:
        assert (t["short_leg"], t["long_leg"]) == ("XXX", "YYY")
    assert np.isfinite(res.trades["pnl"]).all()
