"""Regression tests for issues found in review."""

import pandas as pd
import pytest

from pairs.backtest import BacktestConfig
from pairs.cli import replay
from pairs.data import drop_unfinished_bar
from pairs.paper import PaperBook


def _frame(z, beta=1.0):
    idx = pd.bdate_range("2026-03-02", periods=len(z))
    return pd.DataFrame({"z": z, "beta": beta}, index=idx)


def test_replay_catches_stop_on_a_missed_day(tmp_path):
    """z spikes through the stop and comes back while nobody ran the step."""
    book = PaperBook.load(tmp_path / "b.json")
    fr = _frame([2.5, 3.0, 4.5, 2.5, 2.4])
    px = pd.Series(100.0, fr.index)
    cfg = BacktestConfig(cost_bps=0)

    replay(book, ("A", "B"), px.iloc[:1], px.iloc[:1], fr.iloc[:1], cfg, notional=1000)
    assert "A/B" in book.positions  # opened on day 1

    replay(book, ("A", "B"), px, px, fr, cfg, notional=1000)  # run again 4 days later
    assert "A/B" not in book.positions
    assert book.closed[-1]["reason"] == "stop"
    assert book.closed[-1]["exit_date"] == str(fr.index[2].date())
    assert "A/B" in book.blocked


def test_replay_on_new_pair_only_uses_latest_bar(tmp_path):
    book = PaperBook.load(tmp_path / "b.json")
    fr = _frame([2.5, 0.0, 0.0])  # was diverged long ago, normal now
    px = pd.Series(100.0, fr.index)
    replay(book, ("A", "B"), px, px, fr, BacktestConfig(), notional=1000)
    assert book.positions == {} and book.closed == []


def test_rebase_entry_uses_current_adjusted_basis(tmp_path):
    """After a dividend, Yahoo restates history; P&L must compare like with like."""
    book = PaperBook.load(tmp_path / "b.json")
    book.step(("A", "B"), date="2026-03-02", y_px=100.0, x_px=50.0, beta=1.0, z=2.5,
              entry=2, exit=0.5, stop=4, notional=1000)
    # history restated: entry-day closes are now 99 and 49.5, prices unchanged since
    idx = pd.to_datetime(["2026-03-02", "2026-03-03"])
    book.rebase_entry("A/B", pd.Series([99.0, 99.0], idx), pd.Series([49.5, 49.5], idx))
    assert book.mark("A/B", 99.0, 49.5) == pytest.approx(0.0)


def test_paper_costs_charged_on_open_and_close(tmp_path):
    book = PaperBook.load(tmp_path / "b.json", starting_cash=100_000)
    kw = {"beta": 1.0, "entry": 2, "exit": 0.5, "stop": 4, "notional": 10_000, "cost_bps": 10}
    book.step(("A", "B"), date="d1", y_px=10.0, x_px=10.0, z=2.5, **kw)
    book.step(("A", "B"), date="d2", y_px=10.0, x_px=10.0, z=0.0, **kw)
    # 10 bps of $10k gross, twice
    assert book.cash == pytest.approx(100_000 - 20)
    assert book.closed[-1]["pnl"] == pytest.approx(-20)


def test_drop_unfinished_bar():
    idx = pd.to_datetime(["2026-10-15", "2026-10-16"])
    df = pd.DataFrame({"A": [1.0, 2.0]}, idx)
    during = pd.Timestamp("2026-10-16 11:00", tz="America/New_York")
    after = pd.Timestamp("2026-10-16 17:00", tz="America/New_York")
    assert len(drop_unfinished_bar(df, now=during)) == 1
    assert len(drop_unfinished_bar(df, now=after)) == 2
