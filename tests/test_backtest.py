import numpy as np
import pandas as pd
import pytest

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


def test_delay_trades_a_bar_late(coint_pair):
    y, x = coint_pair
    on_time = backtest(y, x, BacktestConfig())
    late = backtest(y, x, BacktestConfig(delay=1))
    assert late.position.equals(on_time.position.shift(1, fill_value=0).astype(late.position.dtype))
    assert late.stats["total_return"] != on_time.stats["total_return"]
    assert backtest(y, x, BacktestConfig(delay=0)).stats == on_time.stats  # default unchanged


def test_late_fill_keeps_the_signal_days_hedge_ratio(coint_pair):
    y, x = coint_pair
    on_time, late = backtest(y, x, BacktestConfig()), backtest(y, x, BacktestConfig(delay=1))
    # orders are sized on the signal day, so a late fill still uses that day's beta
    assert late.frame["beta"].iloc[1:].equals(on_time.frame["beta"].shift(1).iloc[1:])
    with pytest.raises(ValueError, match="delay"):
        backtest(y, x, BacktestConfig(delay=-1))  # would trade on tomorrow's signal
